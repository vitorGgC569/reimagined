"""
OXTA-CRIT — crit_experiments.py
Validações pequenas e falsificáveis da teoria (docs/OXTA_CRIT_THEORY.md),
dimensionadas para a GTX 1050 Ti (modelo mamba_small, batches pequenos).

  kesten  (E1, CPU, segundos)  — valida a MATEMÁTICA da Lei 2: processo
          h_t = a_t h_{t-1} + u_t com a_t lognormal tem cauda de lei de
          potência com kappa ~= 2|lambda|/sigma^2 (Hill estimator), e a
          retenção flutuante abre DÉCADAS de spread vs decaimento constante.
  recall  (E3, GPU, ~5 min)    — A/B do achado do probe: A=ones (espectro
          degenerado) vs A log-espaçado (NSOS_MAMBA_A_LOGSPACED=1), tarefa
          de memória (carregar um payload por L distratores). Predição: o
          braço log-espaçado aprende mais rápido/melhor.
  qat     (E4, GPU, ~4 min)    — P2 (descritificação): mede p0/ganho-de-ramo
          antes/depois do QAT progressivo engatar; predição: p0 sobe e g
          deriva do manifold sem covariação de alpha.

Uso: python scripts/crit_experiments.py {kesten|recall|qat} [--steps N] ...
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from train_curriculum import (
    build_model_config,
    detect_build_dir,
    load_nsos,
    resolve_profile,
    set_model_training_mode,
)


# ───────────────────────────── E1: Kesten ─────────────────────────────────────
def hill_kappa(samples: np.ndarray, top_frac: float = 0.005) -> float:
    x = np.sort(np.abs(samples))[::-1]
    k = max(int(len(x) * top_frac), 50)
    tail = x[:k]
    return float(1.0 / np.mean(np.log(tail / x[k])))


def exp_kesten(args) -> int:
    rng = np.random.default_rng(7)
    print("[E1] Kesten: kappa previsto = 2|lambda|/sigma^2 vs Hill estimator")
    for lam, sig in [(-0.05, 0.5), (-0.20, 0.4)]:
        kappa_pred = 2.0 * abs(lam) / (sig * sig)
        T = 400_000
        a = np.exp(rng.normal(lam, sig, size=T))
        u = rng.normal(0.0, 1.0, size=T)
        h = np.empty(T)
        acc = 0.0
        for t in range(T):
            acc = a[t] * acc + u[t]
            h[t] = acc
        kappa_meas = hill_kappa(h[1000:])
        print(f"[E1]  lambda={lam:+.2f} sigma={sig:.2f}  kappa_pred={kappa_pred:.2f}  "
              f"kappa_hill={kappa_meas:.2f}")

    # Retenção: impulso unitário, mediana casada, spread de caudas.
    lam_f, sig_f, T2, n = math.log(0.95), 0.6, 100, 20000
    const_ret = 0.95 ** T2
    logret = rng.normal(lam_f, sig_f, size=(n, T2)).sum(axis=1)
    ret = np.exp(logret)
    q50, q99, q999 = np.quantile(ret, [0.5, 0.99, 0.999])
    print(f"[E1] retencao em T={T2}: constante a=0.95 -> {const_ret:.2e} (spread ZERO)")
    print(f"[E1] flutuante (mesma mediana): p50={q50:.2e} p99={q99:.2e} p99.9={q999:.2e} "
          f"-> {math.log10(q999 / max(q50, 1e-300)):.1f} decadas de spread")
    print("[E1] conclusao: seletividade flutuante + linha marginal = retencoes raras e"
          " MUITO longas (memoria livre de escala), impossiveis com decaimento fixo.")
    return 0


# ─────────────────────────── helpers de modelo ────────────────────────────────
def make_model(nsos, profile, vocab, dev, batch):
    cfg = build_model_config(nsos, profile, vocab, dev)
    cfg.default_batch_size = batch
    m = nsos.JambaModel(cfg, dev)
    m.to(dev)
    set_model_training_mode(m, True)
    return m


def a_param_stats(model) -> str:
    taus = []
    for p in model.parameters():
        if p.name.endswith("mamba.A"):
            a = np.abs(np.asarray(p.data.numpy(), dtype=np.float64))
            taus.append(1.0 / np.maximum(a, 1e-3))
    if not taus:
        return "(sem A)"
    t = np.concatenate(taus)
    return (f"tau[min={t.min():.1f} med={np.median(t):.1f} max={t.max():.1f}] "
            f"spread_log10={math.log10(t.max() / max(t.min(), 1e-9)):.2f}")


# ───────────────────────────── E3: recall A/B ─────────────────────────────────
KEY, QUERY = 5, 6


def recall_batch(rng, batch, L, vocab, n_payloads=99):
    prompts, answers = [], []
    for _ in range(batch):
        payload = rng.randint(10, 10 + n_payloads - 1)
        distract = [rng.randint(200, min(3999, vocab - 1)) for _ in range(L)]
        prompts.append([KEY, payload] + distract + [QUERY])
        answers.append([payload, 0])
    return prompts, answers


def eval_recall(model, rng, n, L, vocab, n_payloads=99) -> float:
    set_model_training_mode(model, False)
    hits = 0
    for _ in range(n):
        p, a = recall_batch(rng, 1, L, vocab, n_payloads)
        logits = np.asarray(model.forward_ids(p[0]).numpy())
        pred = int(np.argmax(logits[-1]))
        hits += int(pred == a[0][0])
    set_model_training_mode(model, True)
    return hits / n


def exp_recall(args) -> int:
    nsos = load_nsos(detect_build_dir(args.build_dir))
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))
    dev = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    print(f"[E3] recall A/B: L={args.distractors} batch={args.batch} steps={args.steps} "
          f"profile={args.profile} vocab={vocab}")

    results = {}
    for arm, env_on in [("A=ones(degenerado)", "0"), ("A=logspaced", "1")]:
        os.environ["NSOS_MAMBA_A_LOGSPACED"] = env_on
        rng = random.Random(123)            # mesmo fluxo de dados nos 2 braços
        model = make_model(nsos, profile, vocab, dev, args.batch)
        print(f"[E3] braco {arm}: A inicial -> {a_param_stats(model)}")
        trainer = nsos.Trainer(model, args.lr)
        t0 = time.perf_counter()
        for s in range(args.steps):
            p, a = recall_batch(rng, args.batch, args.distractors, vocab, args.payloads)
            loss = trainer.train_supervised_batch(p, a)
            if (s + 1) % max(args.steps // 6, 1) == 0:
                print(f"[E3]   {arm} step {s+1:>4} loss={float(loss):.4f}")
        acc = eval_recall(model, random.Random(999), args.eval_n, args.distractors, vocab,
                          args.payloads)
        dt = time.perf_counter() - t0
        print(f"[E3] braco {arm}: ACC RECALL = {acc:.3f}  ({dt:.0f}s)  "
              f"A final -> {a_param_stats(model)}")
        results[arm] = acc
        del trainer, model

    os.environ["NSOS_MAMBA_A_LOGSPACED"] = "0"
    arms = list(results.items())
    print(f"\n[E3] VEREDITO: {arms[0][0]}={arms[0][1]:.3f}  vs  {arms[1][0]}={arms[1][1]:.3f}"
          f"  (predicao da Lei 2: logspaced >= ones)")
    return 0


# ───────────────────────────── E4: deriva do QAT ──────────────────────────────
def linear_stats(model):
    p0s, gs = [], []
    for p in model.parameters():
        w = np.asarray(p.data.numpy(), dtype=np.float64)
        if w.ndim != 2 or min(w.shape) <= 1:
            continue
        gamma = float(np.mean(np.abs(w))) or 1e-12
        q = np.clip(np.round(w / gamma), -1, 1)
        p0 = float(np.mean(q == 0))
        p0s.append(p0)
        gs.append(gamma * gamma * (1.0 - p0) * w.shape[-1])
    p0s, gs = np.array(p0s), np.array(gs)
    return (float(np.median(p0s)), float(np.median(gs)),
            float(np.mean((gs > 0.5) & (gs < 2.0))))


def exp_qat(args) -> int:
    nsos = load_nsos(detect_build_dir(args.build_dir))
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))
    dev = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    model = make_model(nsos, profile, vocab, dev, args.batch)
    trainer = nsos.Trainer(model, args.lr)

    sched = nsos.TrainPhaseScheduler()
    sched.progressive_qat_enabled = not args.no_qat
    sched.semantic_warmup_steps = args.steps // 6
    sched.qat_start_step = args.steps // 3
    trainer.total_training_steps = args.steps
    trainer.configure_progressive_qat(sched)
    if args.no_qat:
        print("[E4] CONTROLE: QAT desabilitado (mede drift do treino comum)")

    p0_0, g_0, frac_0 = linear_stats(model)
    print(f"[E4] init : p0_med={p0_0:.4f} g_med={g_0:.4f} frac_g[0.5,2]={frac_0:.3f} "
          f"(qat engata no step {sched.qat_start_step})")

    rng = random.Random(321)
    losses = []
    for s in range(args.steps):
        p, a = recall_batch(rng, args.batch, args.distractors, vocab, args.payloads)
        losses.append(float(trainer.train_supervised_batch(p, a)))
        if (s + 1) % max(args.steps // 6, 1) == 0:
            print(f"[E4]   step {s+1:>4} loss={losses[-1]:.4f}")

    p0_1, g_1, frac_1 = linear_stats(model)
    pre = losses[: sched.qat_start_step]
    post = losses[sched.qat_start_step:]
    spike_pre = max(pre[i] - min(pre[: i + 1]) for i in range(1, len(pre)))
    spike_post = max(post[i] - min(losses[: sched.qat_start_step + i + 1])
                     for i in range(1, len(post)))
    print(f"[E4] final: p0_med={p0_1:.4f} (d={p0_1-p0_0:+.4f}) g_med={g_1:.4f} "
          f"(d={g_1-g_0:+.4f}) frac_g[0.5,2]={frac_1:.3f} (d={frac_1-frac_0:+.3f})")
    print(f"[E4] spike de loss: pre-QAT={spike_pre:.4f}  pos-QAT={spike_post:.4f}")
    print("[E4] P2 sustentada se: p0 subiu E (frac_g caiu OU spike_pos >> spike_pre).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OXTA-CRIT small experiments (1050 Ti)")
    ap.add_argument("exp", choices=["kesten", "recall", "qat"])
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--profile", default="mamba_small")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--distractors", type=int, default=48)
    ap.add_argument("--payloads", type=int, default=99)
    ap.add_argument("--no-qat", action="store_true")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if args.exp == "kesten":
        return exp_kesten(args)
    if args.exp == "recall":
        return exp_recall(args)
    return exp_qat(args)


if __name__ == "__main__":
    raise SystemExit(main())
