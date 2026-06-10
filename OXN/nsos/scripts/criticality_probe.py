"""
OXTA-CRIT — criticality_probe.py (v1)

Mede a POSIÇÃO DE FASE do modelo (docs/OXTA_CRIT_THEORY.md):

Lei 1 (espaço/ternário) — por peso linear (rank-2):
  p0   = fração de zeros sob a regra absmean do BitNet (|w| < 0.5·mean|w|)
  γ    = absmean (escala ternária)
  g_t  = γ²·(1−p0)·N   (ganho de ramo ternário, input variância-1, E[φ'²]≈1)
  g_c  = N·Var(w)      (ganho contínuo, referência)
  Manifold crítico (rede plana): g ≈ 1. Em blocos residuais o alvo fino é
  "ganhos uniformes e O(1)" — o probe REPORTA, não julga.

Lei 2 (tempo/Kesten) — por parâmetro A do Mamba (rank-1, nome com 'A'):
  espectro estático de timescales τ = 1/max(A,1e-3) tokens (a dt nominal).
  (v2 medirá λ=E[log a], σ², κ=2|λ|/σ² com tap de dt real.)

Uso:  python scripts/criticality_probe.py [--profile mamba_small] [--model ckpt.bin]
      [--device cpu|gpu] [--build-dir DIR] [--out artifacts/crit_<ts>.json]
v1 é estático (init ou checkpoint); roda em CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from train_curriculum import (
    build_model_config,
    detect_build_dir,
    load_nsos,
    resolve_profile,
)


def tensor_np(t):
    arr = np.asarray(t.numpy(), dtype=np.float64)
    return arr


def analyze_linear(name: str, w: np.ndarray) -> dict:
    fan_in = int(w.shape[-1])
    gamma = float(np.mean(np.abs(w))) or 1e-12
    q = np.clip(np.round(w / gamma), -1, 1)
    p0 = float(np.mean(q == 0))
    p_pos = float(np.mean(q > 0))
    g_tern = gamma * gamma * (1.0 - p0) * fan_in
    g_cont = float(np.var(w)) * fan_in
    return {
        "name": name, "shape": list(w.shape), "fan_in": fan_in,
        "gamma_absmean": round(gamma, 6), "p0": round(p0, 4),
        "p_pos": round(p_pos, 4),
        "g_ternary": round(g_tern, 4), "g_continuous": round(g_cont, 4),
        "log10_g_ternary": round(math.log10(max(g_tern, 1e-30)), 3),
    }


def analyze_decay(name: str, a: np.ndarray) -> dict:
    a_eff = np.maximum(np.abs(a), 1e-3)  # kernel clamps A at 1e-3
    tau = 1.0 / a_eff                     # timescale (tokens) a dt nominal
    return {
        "name": name, "channels": int(a.size),
        "tau_min": round(float(tau.min()), 2),
        "tau_med": round(float(np.median(tau)), 2),
        "tau_max": round(float(tau.max()), 2),
        "tau_log_spread": round(float(np.log10(tau.max() / max(tau.min(), 1e-9))), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="OXTA-CRIT phase-position probe (v1, static)")
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--profile", default="mamba_small")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--model", type=Path, default=None, help="checkpoint .bin opcional")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    nsos = load_nsos(detect_build_dir(args.build_dir))
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))
    dev = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    cfg = build_model_config(nsos, profile, vocab, dev)
    model = nsos.JambaModel(cfg, dev)
    if args.model:
        model.load(str(args.model), False)
        state = f"checkpoint:{args.model.name}"
    else:
        state = "init"

    linears, decays, skipped = [], [], 0
    for p in model.parameters():
        try:
            name = p.name
            w = tensor_np(p.data)
        except Exception:
            skipped += 1
            continue
        if w.ndim == 2 and min(w.shape) > 1:
            linears.append(analyze_linear(name, w))
        elif w.ndim == 1 and ("A" in name.split("_") or name.endswith(".A") or "A_param" in name or name.endswith("A")):
            decays.append(analyze_decay(name, w))

    depth_log_amp = sum(r["log10_g_ternary"] for r in linears)
    gs = np.array([r["g_ternary"] for r in linears]) if linears else np.array([1.0])
    report = {
        "profile": args.profile, "state": state, "device": args.device,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_linear": len(linears), "n_decay": len(decays), "skipped": skipped,
        "summary": {
            "g_ternary_median": round(float(np.median(gs)), 4),
            "g_ternary_min": round(float(gs.min()), 4),
            "g_ternary_max": round(float(gs.max()), 4),
            "frac_layers_in_[0.5,2]": round(float(np.mean((gs > 0.5) & (gs < 2.0))), 3),
            "depth_log10_amplification_sum": round(depth_log_amp, 3),
            "p0_median": round(float(np.median([r["p0"] for r in linears])), 4) if linears else None,
        },
        "linears": linears,
        "decays": decays,
    }

    print(f"[crit] state={state} profile={args.profile} linears={len(linears)} decays={len(decays)}")
    s = report["summary"]
    print(f"[crit] g_ternary: med={s['g_ternary_median']} min={s['g_ternary_min']} "
          f"max={s['g_ternary_max']} | frac em [0.5,2] = {s['frac_layers_in_[0.5,2]']}")
    print(f"[crit] p0 mediana = {s['p0_median']} | sum log10(g) na profundidade = "
          f"{s['depth_log10_amplification_sum']}")
    worst = sorted(linears, key=lambda r: abs(r["log10_g_ternary"]))[-5:]
    for r in reversed(worst):
        print(f"[crit]   fora-da-borda: {r['name']:<40} g_t={r['g_ternary']:<10} p0={r['p0']}")
    for d in decays[:4]:
        print(f"[crit] timescale {d['name']:<40} tau med={d['tau_med']} "
              f"[{d['tau_min']}, {d['tau_max']}] spread(log10)={d['tau_log_spread']}")

    out = args.out or (Path(__file__).resolve().parents[1] / "artifacts"
                       / f"crit_probe_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[crit] artefato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
