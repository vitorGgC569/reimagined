"""
attn_bwd_parity.py — gate de paridade do backward GPU da atencao (v2: GRADIENTES).

>>> EXECUTAR APENAS NO COLAB/T4. NAO rodar no desktop local (o braco host roda
>>> o backward O(B*H*S^2*hd) na CPU).

LICAO DA v1 (pre-registrada e corrigida): comparar TRAJETORIA de loss atraves
do Adam mede caos, nao matematica — no step 1 do Adam o update e ~lr*sign(g),
entao reassociacao FP que troca o sinal de gradientes ~0 gera dloss ~1e-2 com
trajetorias paralelas (foi exatamente o observado: step0 IDENTICO 9.803182,
divergencia so pos-otimizador). O gate correto compara os GRADIENTES em si:
apos 1 train_supervised_batch os param.grad ficam intactos (zero_grad so roda
no inicio do passo seguinte) — snapshot e comparacao direta, sem Adam no meio.

Gate (FP32): rel = ||g_gpu - g_host|| / (||g_host|| + 1e-12) por parametro.
  PASS  : max rel <= 1e-4  (reassociacao pura)
  ZONA  : 1e-4 < rel <= 1e-3 -> investigar antes de promover
  FAIL  : rel > 1e-3        -> bug real; manter NSOS_ATTN_BWD_HOST=1

Uso: python scripts/attn_bwd_parity.py [--build-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np

os.environ["NSOS_MIXED_PRECISION"] = ""   # paridade em FP32 puro
os.environ.setdefault("NSOS_GPU_POOL", "1")

from train_curriculum import (
    build_model_config,
    detect_build_dir,
    load_nsos,
    resolve_profile,
    set_model_training_mode,
)


def make_batch(rng, batch, plen, alen, vocab):
    prompts = [[rng.randint(1, vocab - 1) for _ in range(plen)] for _ in range(batch)]
    answers = [[rng.randint(1, vocab - 1) for _ in range(alen)] + [0] for _ in range(batch)]
    return prompts, answers


def run_arm(nsos, profile, vocab, host_arm: bool):
    """1 step identico (mesma seed/dados); retorna (loss, {nome: grad_np})."""
    os.environ["NSOS_ATTN_BWD_HOST"] = "1" if host_arm else "0"
    nsos.set_seed(1234)
    dev = nsos.Device.GPU
    cfg = build_model_config(nsos, profile, vocab, dev)
    try:
        cfg.default_batch_size = 2
    except Exception:
        pass
    model = nsos.JambaModel(cfg, dev)
    model.to(dev)
    set_model_training_mode(model, True)
    trainer = nsos.Trainer(model, 3e-4)
    rng = random.Random(99)
    p, a = make_batch(rng, 2, 24, 8, vocab)
    loss = float(trainer.train_supervised_batch(p, a))
    grads = {}
    for prm in model.parameters():
        try:
            if prm.grad.size > 0:
                grads[prm.name] = np.array(prm.grad.numpy(), dtype=np.float32, copy=True)
        except Exception:
            continue
    del trainer, model
    return loss, grads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--profile", default="hybrid_v11_colab_t4")
    ap.add_argument("--pass-tol", type=float, default=1e-4)
    ap.add_argument("--fail-tol", type=float, default=1e-3)
    args = ap.parse_args()

    nsos = load_nsos(detect_build_dir(args.build_dir))
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))

    loss_h, g_h = run_arm(nsos, profile, vocab, host_arm=True)
    loss_g, g_g = run_arm(nsos, profile, vocab, host_arm=False)
    os.environ["NSOS_ATTN_BWD_HOST"] = "0"

    print(f"[parity] loss step0: host={loss_h:.6f} gpu={loss_g:.6f} "
          f"(forward identico esperado: dloss={abs(loss_h-loss_g):.2e})")

    common = sorted(set(g_h) & set(g_g))
    if not common:
        print("[parity] ERRO: nenhum gradiente comum capturado")
        return 2
    worst_rel, worst_abs, worst_name = 0.0, 0.0, ""
    rows = []
    for name in common:
        a, b = g_h[name], g_g[name]
        if a.shape != b.shape:
            print(f"[parity] SHAPE DIVERGE em {name}: {a.shape} vs {b.shape}")
            return 2
        diff = b - a
        abs_d = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-12))
        rows.append((rel, abs_d, name))
        if rel > worst_rel:
            worst_rel, worst_abs, worst_name = rel, abs_d, name
    rows.sort(reverse=True)
    print(f"[parity] params comparados: {len(common)}")
    for rel, abs_d, name in rows[:8]:
        print(f"[parity]   rel={rel:.3e} max|d|={abs_d:.3e}  {name}")
    print(f"[parity] PIOR: rel={worst_rel:.3e} ({worst_name})")

    if worst_rel <= args.pass_tol:
        print(f"[parity] PASS (<= {args.pass_tol:.0e}) — reassociacao pura; "
              f"PROMOVER caminho GPU (default ja e GPU)")
        return 0
    if worst_rel <= args.fail_tol:
        print(f"[parity] ZONA CINZA ({args.pass_tol:.0e} < rel <= {args.fail_tol:.0e}) — "
              f"colar a saida para analise antes de promover")
        return 0
    print(f"[parity] FAIL (> {args.fail_tol:.0e}) — bug real; "
          f"usar NSOS_ATTN_BWD_HOST=1 no treino e colar esta saida")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
