"""OXTA-CRIT §6 — criticality_instrument.py

The measurement half of the "diferencial": turns the project's own observations
(branch gain on the SPACE axis; replica gradient divergence on the DEPTH axis)
into a live, per-layer instrument the controller (closed loop) can read.

Two measurements:

  1. branch_gains(model)  — OXTA-CRIT Lei 1, ONLINE.
     Per rank-2 weight: gamma = absmean|W|, p0 = ternary zero fraction
     (|W| < 0.5*gamma), g_ternary = gamma^2 * (1-p0) * fan_in.  The critical
     manifold (flat net) sits at g ~ 1; this reports g per layer + the fraction
     inside the critical band [0.5, 2].  Same quantity as the NSOS_CRIT_REG
     controller and scripts/criticality_probe.py, exposed for in-loop use.

  2. gradient_snr(trainer, model, samples) — BACKWARD-LYAPUNOV by replica.
     The gradient of two disjoint micro-batches (same step, same weights) is
     compared per parameter: coherence r_l = cos(g_A, g_l_B); SNR_l = r/(1-r).
     This is the in-situ, per-layer signal-to-noise of the gradient — the
     quantity ESPECTRO_DE_CREDITO predicts should set the per-layer learning
     rate / optimizer averaging horizon.  Uses Trainer.accumulate_gradients
     (gradients only, no optimizer step), so it never mutates the model.

The closed-loop controller that CONSUMES these (per-layer lr/beta + 3-axis
criticality regulation) lives in C++ (trainer.cpp); this script is the probe and
the offline analysis.

Local self-test (no nsos_ext, pure numpy — real local evidence):
    python OXN/nsos/scripts/criticality_instrument.py --selftest

Full run (needs a build + checkpoint, on Colab/T4 or a built tree):
    python OXN/nsos/scripts/criticality_instrument.py \
        --build-dir <dir> --profile mamba_small --model <ckpt.bin>
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── pure math (unit-testable without nsos_ext) ───────────────────────────────
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def snr_from_coherence(r: float) -> float:
    # Signal+noise model: cos of two independent noisy replicas of the same
    # signal estimates r = SNR/(SNR+1)  ->  SNR = r/(1-r).
    r = max(min(r, 1.0 - 1e-9), -1.0 + 1e-9)
    return r / (1.0 - r) if r < 1.0 else float("inf")


def branch_gain(weight: np.ndarray) -> dict:
    """OXTA-CRIT Lei 1 branch gain for one rank-2 weight (numpy)."""
    fan_in = int(weight.shape[-1])
    gamma = float(np.mean(np.abs(weight))) or 1e-12
    q = np.clip(np.round(weight / gamma), -1, 1)
    p0 = float(np.mean(q == 0))
    g = gamma * gamma * (1.0 - p0) * fan_in
    return {"gamma": gamma, "p0": p0, "fan_in": fan_in, "g_ternary": g}


def _layer_key(name: str) -> str:
    # Best-effort: group params by the first integer in the name (layer index).
    cur = ""
    for ch in name:
        if ch.isdigit():
            cur += ch
        elif cur:
            return f"layer{cur}"
    return name.split(".")[0] if name else "?"


# ── instruments over a live model (need nsos_ext) ────────────────────────────
def measure_branch_gains(model) -> dict:
    layers = []
    for p in model.parameters():
        w = np.asarray(p.data.numpy())
        if w.ndim == 2 and min(w.shape) > 1:
            bg = branch_gain(w)
            bg["name"] = p.name
            layers.append(bg)
    if not layers:
        return {"layers": [], "summary": {}}
    gs = np.array([r["g_ternary"] for r in layers])
    summary = {
        "n_linear": len(layers),
        "g_median": float(np.median(gs)),
        "g_min": float(gs.min()),
        "g_max": float(gs.max()),
        "frac_in_critical_band_0.5_2": float(np.mean((gs > 0.5) & (gs < 2.0))),
        "depth_log10_amplification_sum": float(
            np.sum(np.log10(np.maximum(gs, 1e-30)))
        ),
    }
    return {"layers": layers, "summary": summary}


def _snapshot_grads(model) -> dict:
    grads = {}
    for p in model.parameters():
        try:
            arr = np.asarray(p.grad.numpy()).ravel()
        except Exception:
            arr = np.zeros(0, dtype=np.float64)
        grads[p.name] = arr
    return grads


def measure_gradient_snr(trainer, model, samples, num_pairs: int = 16,
                         seed: int = 0) -> dict:
    """Per-parameter gradient coherence/SNR over disjoint micro-batch pairs.

    samples: list of (tokens, targets) int-lists.  Uses accumulate_gradients
    (no optimizer step) so weights/optimizer state are untouched.
    """
    rng = random.Random(seed)
    acc = defaultdict(lambda: [0.0, 0])  # name -> [sum_cos, count]
    n = len(samples)
    if n < 2:
        raise ValueError("need >= 2 samples for replica SNR")
    for _ in range(num_pairs):
        a, b = rng.sample(samples, 2)
        trainer.accumulate_gradients(list(a[0]), list(a[1]))
        gA = _snapshot_grads(model)
        trainer.accumulate_gradients(list(b[0]), list(b[1]))
        gB = _snapshot_grads(model)
        for name, va in gA.items():
            vb = gB.get(name)
            if vb is None or va.size == 0 or vb.size != va.size:
                continue
            acc[name][0] += cosine(va, vb)
            acc[name][1] += 1
    per_param = {}
    per_layer = defaultdict(lambda: [0.0, 0])
    for name, (s, c) in acc.items():
        if c == 0:
            continue
        r = s / c
        per_param[name] = {"coherence": r, "snr": snr_from_coherence(r),
                           "samples": c}
        lk = _layer_key(name)
        per_layer[lk][0] += r
        per_layer[lk][1] += 1
    layers = {lk: {"coherence": s / c, "snr": snr_from_coherence(s / c)}
              for lk, (s, c) in per_layer.items() if c}
    return {"per_param": per_param, "per_layer": layers}


# ── closed loop: push per-layer lr scales into the trainer (DEPTH axis) ──────
def snr_lr_scales(per_param: dict, gamma: float = 1.0,
                  lo: float = 0.25, hi: float = 4.0) -> dict:
    """ESPECTRO_DE_CREDITO §6 law: lr_l ∝ r_l^γ (r = gradient coherence).
    Returns name -> scale, normalized to mean 1 (preserves the global lr) and
    clamped to [lo, hi].  High-SNR (coherent) layers get a larger step; noisy
    layers get a smaller one."""
    names, raw = [], []
    for name, v in per_param.items():
        r = max(float(v["coherence"]), 0.0)
        names.append(name)
        raw.append(r ** gamma)
    if not raw:
        return {}
    mean = sum(raw) / len(raw)
    if mean <= 0:
        return {n: 1.0 for n in names}
    return {n: min(max(x / mean, lo), hi) for n, x in zip(names, raw)}


def apply_snr_lr_control(trainer, per_param: dict, gamma: float = 1.0,
                         lo: float = 0.25, hi: float = 4.0) -> dict:
    """Measure -> control: write the SNR-derived per-layer lr scales into the
    trainer (Trainer.set_lr_scale_by_name).  Call once per N steps after
    measure_gradient_snr.  Composes with the C++ NSOS_CRIT_LR space-axis
    controller if that owns the map; otherwise this owns it."""
    scales = snr_lr_scales(per_param, gamma, lo, hi)
    for name, s in scales.items():
        trainer.set_lr_scale_by_name(name, float(s))
    return scales


# ── self-test: pure numpy, runs locally without a build ──────────────────────
def selftest() -> int:
    assert abs(cosine(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) - 1.0) < 1e-9
    assert abs(cosine(np.array([4.0, -3, 0, 0]), np.array([3.0, 4, 0, 0]))) < 1e-9
    assert abs(cosine(np.array([1.0, 2, 3]), np.array([-1.0, -2, -3])) + 1.0) < 1e-9
    assert abs(snr_from_coherence(0.5) - 1.0) < 1e-9      # r=0.5 -> SNR 1
    assert snr_from_coherence(0.9) > snr_from_coherence(0.5)

    # signal+noise: higher true SNR -> higher measured coherence (monotone).
    rng = np.random.default_rng(0)
    d = 4000
    sig = rng.standard_normal(d)

    def coh(snr_lin):
        amp = math.sqrt(snr_lin)
        ga = sig * amp + rng.standard_normal(d)
        gb = sig * amp + rng.standard_normal(d)
        return cosine(ga, gb)

    lo, hi = coh(0.2), coh(5.0)
    assert hi > lo, (lo, hi)
    # recovered SNR from coherence should track the true SNR ordering
    assert snr_from_coherence(hi) > snr_from_coherence(lo)

    # branch gain: a near-ternary balanced matrix has g ~ fan_in * gamma^2.
    w = np.array([[1.0, -1.0, 0.0, 1.0], [0.0, 1.0, -1.0, 0.0]])
    bg = branch_gain(w)
    assert bg["fan_in"] == 4 and 0.0 <= bg["p0"] <= 1.0 and bg["g_ternary"] > 0
    assert _layer_key("layers.3.mamba.in_proj_robust.weight") == "layer3"

    # closed-loop lr scales: coherent layer gets a bigger step than a noisy one,
    # mean stays ~1 (global lr preserved), within clamp.
    pp = {"layer0.w": {"coherence": 0.9, "snr": 9.0},
          "layer1.w": {"coherence": 0.1, "snr": 0.11}}
    scales = snr_lr_scales(pp, gamma=1.0)
    assert scales["layer0.w"] > scales["layer1.w"]
    assert abs(sum(scales.values()) / len(scales) - 1.0) < 0.5
    assert all(0.25 <= s <= 4.0 for s in scales.values())

    print(f"[selftest] cosine id=1 orth=0 anti=-1 OK | SNR(r=.5)=1 OK")
    print(f"[selftest] coherence monotone in SNR: coh(.2)={lo:.3f} < coh(5)={hi:.3f} OK")
    print(f"[selftest] branch_gain g={bg['g_ternary']:.3f} p0={bg['p0']:.2f} OK")
    print(f"[selftest] snr lr-scales: hi={scales['layer0.w']:.2f} > lo={scales['layer1.w']:.2f} OK")
    print("[selftest] criticality_instrument PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OXTA-CRIT §6 criticality instrument")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-numpy math self-test (no build needed)")
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--profile", default="mamba_small")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    # Full run: load the model + a tiny synthetic batch, report both instruments.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_curriculum import build_model_config, load_nsos, resolve_profile

    nsos = load_nsos(args.build_dir)
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))
    dev = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    cfg = build_model_config(nsos, profile, vocab, dev)
    model = nsos.JambaModel(cfg, dev)
    if args.model:
        model.load(str(args.model), False)
    model.to(dev)

    gains = measure_branch_gains(model)
    print("[branch-gain]", json.dumps(gains["summary"], indent=1))

    # Tiny synthetic next-token samples just to exercise the SNR probe.
    trainer = nsos.Trainer(model, 1e-3)
    rng = random.Random(0)
    samples = []
    for _ in range(8):
        seq = [rng.randint(0, max(vocab - 1, 1)) for _ in range(12)]
        samples.append((seq, []))  # targets=[] -> next-token shift in make_targets
    snr = measure_gradient_snr(trainer, model, samples, num_pairs=8)
    print("[grad-snr per layer]")
    for lk, v in sorted(snr["per_layer"].items()):
        print(f"  {lk:<10} coherence={v['coherence']:+.3f}  SNR={v['snr']:.3f}")

    report = {"branch_gains": gains, "gradient_snr": snr}
    out = args.out or (Path(__file__).resolve().parents[1] / "artifacts"
                       / "crit_instrument.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[crit-instrument] wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
