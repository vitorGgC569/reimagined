"""
attn_bwd_parity.py — gate de paridade do backward GPU da atenção.

>>> EXECUTAR APENAS NO COLAB/T4 (ou GPU dedicada). NAO rodar no desktop local:
>>> o braco HOST roda o backward O(B*H*S^2*hd) na CPU e pode travar a maquina.

A/B na MESMA arquitetura/dados/seed (perfil hybrid_v11_colab_t4, batch e seq
reduzidos): braco A = NSOS_ATTN_BWD_HOST=1 (referencia host), braco B = GPU.
Os pesos evoluem pelos gradientes, entao igualdade da TRAJETORIA de loss em
3 steps e o teste dos gradientes. Pre-registro (analise estatica, doc
ATTENTION_GPU_TRAIN_STATIC_VERIFICATION.md): max|dloss| <= 1e-3 em FP32.

Uso: python scripts/attn_bwd_parity.py [--build-dir DIR] [--steps 3]
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Paridade em FP32: BF16 reassocia/arredonda demais para um gate de 1e-3.
os.environ["NSOS_MIXED_PRECISION"] = ""
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


def run_arm(nsos, profile, vocab, host_arm: bool, steps: int):
    os.environ["NSOS_ATTN_BWD_HOST"] = "1" if host_arm else "0"
    nsos.set_seed(1234)                      # init deterministica identica
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
    rng = random.Random(99)                  # mesmo fluxo de dados nos 2 bracos
    losses = []
    for _ in range(steps):
        p, a = make_batch(rng, 2, 24, 8, vocab)
        losses.append(float(trainer.train_supervised_batch(p, a)))
    del trainer, model
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--profile", default="hybrid_v11_colab_t4")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    nsos = load_nsos(detect_build_dir(args.build_dir))
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))

    host = run_arm(nsos, profile, vocab, host_arm=True, steps=args.steps)
    gpu = run_arm(nsos, profile, vocab, host_arm=False, steps=args.steps)
    os.environ["NSOS_ATTN_BWD_HOST"] = "0"

    print(f"[parity] host: {[round(x, 6) for x in host]}")
    print(f"[parity] gpu : {[round(x, 6) for x in gpu]}")
    deltas = [abs(h - g) for h, g in zip(host, gpu)]
    worst = max(deltas)
    print(f"[parity] max|dloss| = {worst:.3e}  (tol {args.tol:.0e})")
    if worst <= args.tol:
        print("[parity] PASS — backward GPU da atencao = referencia host")
        return 0
    print("[parity] FAIL — NAO promover; manter NSOS_ATTN_BWD_HOST=1 e investigar")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
