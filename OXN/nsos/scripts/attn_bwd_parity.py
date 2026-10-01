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
    cfg.default_batch_size = 2
    model = nsos.JambaModel(cfg, dev)
    model.to(dev)
    set_model_training_mode(model, True)
    trainer = nsos.Trainer(model, 3e-4)
    rng = random.Random(99)
    p, a = make_batch(rng, 2, 24, 8, vocab)
    loss = float(trainer.train_supervised_batch(p, a))
    grads = {}
    skipped = 0
    first_err = None
    for prm in model.parameters():
        name = getattr(prm, "name", "?")
        try:
            # Tensor nao expoe .size em Python — capture via numpy e cheque la.
            g = np.array(prm.grad.numpy(), dtype=np.float32, copy=True)
        except Exception as exc:  # ex.: grad vazio (param sem gradiente neste passo)
            skipped += 1
            if first_err is None:
                first_err = f"{name}: {type(exc).__name__}: {exc}"
            continue
        if g.size > 0:
            grads[name] = g
        else:
            skipped += 1
    if not grads:
        print(f"[parity] captura de grads falhou em TODOS os params; primeiro erro: {first_err}")
    elif skipped:
        print(f"[parity] aviso: {skipped} params sem grad capturavel "
              f"(primeiro: {first_err})")
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

    # v5: QUATRO bracos — host x2 e gpu x2 — para medir o PISO DE RUIDO do
    # proprio pipeline (atomics existentes: scatter-add do embedding, grad_A do
    # mamba, scatter do MoE) e separa-lo do efeito do kernel novo.
    loss_h, g_h = run_arm(nsos, profile, vocab, host_arm=True)
    loss_h2, g_h2 = run_arm(nsos, profile, vocab, host_arm=True)
    loss_g, g_g = run_arm(nsos, profile, vocab, host_arm=False)
    loss_g2, g_g2 = run_arm(nsos, profile, vocab, host_arm=False)
    os.environ["NSOS_ATTN_BWD_HOST"] = "0"

    print(f"[parity] loss step0: host={loss_h:.6f} host2={loss_h2:.6f} "
          f"gpu={loss_g:.6f} gpu2={loss_g2:.6f}")

    import re as _re

    # Classificador v5: exige ".attn." — na v4, mamba.out_proj (cada layer tem
    # um!) poluiu o grupo "controle", e o backward do mamba-11 roda DEPOIS do
    # backward da atencao-11, vendo a divergencia legitimamente => controle
    # inflado por max() com um tensor que NAO e controle.
    def is_attn(name: str, part: str) -> bool:
        return f"attn.{part}" in name

    def layer_of(name: str):
        m = _re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def compare(ga: dict, gb: dict):
        """Retorna (global_rel, attn_profile{layer: {q,kv,out}}, n)."""
        common = sorted(set(ga) & set(gb))
        sum_d2 = 0.0
        sum_g2 = 0.0
        prof: dict[int, dict[str, float]] = {}
        for name in common:
            a, b = ga[name], gb[name]
            if a.shape != b.shape:
                raise RuntimeError(f"shape diverge em {name}")
            diff = (b.astype(np.float64) - a.astype(np.float64)).ravel()
            a64 = a.astype(np.float64).ravel()
            d2 = float(np.dot(diff, diff))
            g2 = float(np.dot(a64, a64))
            sum_d2 += d2
            sum_g2 += g2
            kind = ("q" if is_attn(name, "q_down_proj")
                    else "kv" if is_attn(name, "kv_down_proj")
                    else "out" if is_attn(name, "out_proj") else None)
            if kind is not None:
                li = layer_of(name)
                if li is not None:
                    rel = float(np.sqrt(d2) / (np.sqrt(g2) + 1e-12))
                    d = prof.setdefault(li, {})
                    d[kind] = max(d.get(kind, 0.0), rel)
        return float(np.sqrt(sum_d2) / (np.sqrt(sum_g2) + 1e-12)), prof, len(common)

    if not (set(g_h) & set(g_g)):
        print("[parity] ERRO: nenhum gradiente comum capturado")
        return 2

    noise_h, prof_hh, _ = compare(g_h, g_h2)
    noise_g, prof_gg, _ = compare(g_g, g_g2)
    effect, prof_hg, n_cmp = compare(g_h, g_g)

    print(f"[parity] params: {n_cmp}")
    print(f"[parity] PISO DE RUIDO  host-vs-host = {noise_h:.3e}   "
          f"gpu-vs-gpu = {noise_g:.3e}")
    print(f"[parity] EFEITO         host-vs-gpu  = {effect:.3e}")

    def top_attn(prof):
        if not prof:
            return None, 0.0, 0.0
        top = max(prof)
        d = prof[top]
        return top, max(d.get("q", 0.0), d.get("kv", 0.0)), d.get("out", 0.0)

    print("[parity] perfil attn REAL por camada (host-vs-gpu; classificador exige '.attn.'):")
    for li in sorted(prof_hg, reverse=True):
        d = prof_hg[li]
        print(f"[parity]   layer {li:>2}: q={d.get('q', 0):.3e} "
              f"kv={d.get('kv', 0):.3e} out={d.get('out', 0):.3e}")

    top, t_target, t_control = top_attn(prof_hg)
    _, nh_target, nh_control = top_attn(prof_hh)
    _, ng_target, ng_control = top_attn(prof_gg)
    print(f"[parity] TOPO attn = layer {top}: alvo={t_target:.3e} controle={t_control:.3e}")
    print(f"[parity]   ruido no topo: host alvo={nh_target:.3e} ctrl={nh_control:.3e} | "
          f"gpu alvo={ng_target:.3e} ctrl={ng_control:.3e}")

    floor = max(noise_h, noise_g)
    floor_ctrl = max(nh_control, ng_control, 1e-9)
    floor_tgt = max(nh_target, ng_target, 1e-9)

    # Matriz de decisao pre-registrada (v5):
    # D1 efeito <= 3x piso global         -> PASS (indistinguivel do nao-determinismo existente)
    # D2 controle topo > 10x piso-ctrl    -> ERRO DE HARNESS (divergencia fora do kernel)
    # D3 alvo topo <= pass_tol            -> PASS (kernel correto; global e caos)
    # D4 alvo topo > fail_tol e > 10x piso-alvo -> FAIL REAL (q vs kv impresso)
    # D5 caso contrario                   -> ZONA CINZA (colar saida)
    if effect <= 3.0 * floor:
        print(f"[parity] PASS (D1) — efeito {effect:.2e} <= 3x piso {floor:.2e}: o kernel "
              f"novo e indistinguivel do nao-determinismo ja existente. PROMOVIDO.")
        return 0
    if t_control > max(10.0 * floor_ctrl, 1e-5):
        print(f"[parity] ERRO DE HARNESS (D2): controle topo {t_control:.2e} > "
              f"10x ruido {floor_ctrl:.2e} — divergencia fora do kernel novo; colar saida")
        return 2
    if t_target <= args.pass_tol:
        print(f"[parity] PASS (D3) — alvo topo {t_target:.2e} <= {args.pass_tol:.0e}; "
              f"global {effect:.1e} e caos de profundidade. PROMOVIDO.")
        return 0
    if t_target > args.fail_tol and t_target > 10.0 * floor_tgt:
        d = prof_hg.get(top, {})
        worst_kind = "q" if d.get("q", 0.0) >= d.get("kv", 0.0) else "kv"
        print(f"[parity] FAIL REAL (D4) — alvo topo {t_target:.2e} > {args.fail_tol:.0e} "
              f"e > 10x ruido {floor_tgt:.2e}: bug no caminho '{worst_kind}'. "
              f"Treinar com NSOS_ATTN_BWD_HOST=1 e colar a saida.")
        return 1
    print(f"[parity] ZONA CINZA (D5) — alvo topo {t_target:.2e}, ruidos "
          f"(alvo {floor_tgt:.2e}, ctrl {floor_ctrl:.2e}); colar a saida")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
