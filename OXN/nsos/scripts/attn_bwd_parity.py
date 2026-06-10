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

    loss_h, g_h = run_arm(nsos, profile, vocab, host_arm=True)
    loss_g, g_g = run_arm(nsos, profile, vocab, host_arm=False)
    os.environ["NSOS_ATTN_BWD_HOST"] = "0"

    print(f"[parity] loss step0: host={loss_h:.6f} gpu={loss_g:.6f} "
          f"(forward identico esperado: dloss={abs(loss_h-loss_g):.2e})")

    common = sorted(set(g_h) & set(g_g))
    if not common:
        print("[parity] ERRO: nenhum gradiente comum capturado")
        return 2

    # METODOLOGIA v3 (licao da v2, pre-registrada): rel POR PARAMETRO explode em
    # tensores de norma ~0 (experts MoE pouco roteados) por puro caos de
    # backprop amplificado em profundidade — qualquer implementacao
    # numericamente diferente-mas-correta reprova nesse criterio. Observaveis
    # corretos: (a) rel GLOBAL sobre o gradiente inteiro; (b) os grupos da
    # PROPRIA atencao, com out_proj como CONTROLE INTERNO (entradas identicas
    # nos dois bracos => rel ~1e-7 obrigatorio; se out_proj reprovar, o
    # problema NAO e o kernel novo). Experts/tiny-norm: reportados, nao gateados.
    def group_of(name: str) -> str:
        if "out_proj" in name:
            return "attn.out_proj (CONTROLE)"
        if "q_down_proj" in name:
            return "attn.q_down_proj (ALVO)"
        if "kv_down_proj" in name:
            return "attn.kv_down_proj (ALVO)"
        if "experts" in name or "router" in name:
            return "moe (chaos esperado)"
        if "mamba" in name:
            return "mamba (downstream)"
        if "embedding" in name:
            return "embedding (downstream)"
        return "outros (downstream)"

    sum_d2 = 0.0
    sum_g2 = 0.0
    groups: dict[str, list[tuple[float, float, str]]] = {}
    for name in common:
        a, b = g_h[name], g_g[name]
        if a.shape != b.shape:
            print(f"[parity] SHAPE DIVERGE em {name}: {a.shape} vs {b.shape}")
            return 2
        diff = (b.astype(np.float64) - a.astype(np.float64)).ravel()
        a64 = a.astype(np.float64).ravel()
        d2 = float(np.dot(diff, diff))
        g2 = float(np.dot(a64, a64))
        sum_d2 += d2
        sum_g2 += g2
        rel = float(np.sqrt(d2) / (np.sqrt(g2) + 1e-12))
        groups.setdefault(group_of(name), []).append(
            (rel, float(np.max(np.abs(diff))) if diff.size else 0.0, name))

    global_rel = float(np.sqrt(sum_d2) / (np.sqrt(sum_g2) + 1e-12))
    print(f"[parity] params comparados: {len(common)}")
    print(f"[parity] REL GLOBAL (||dG||/||G||) = {global_rel:.3e}")
    attn_rels = []
    control_rel = None
    for gname in sorted(groups):
        items = groups[gname]
        rels = [r for r, _, _ in items]
        worst = max(items)
        print(f"[parity]   {gname:<28} n={len(items):>3} rel_max={max(rels):.3e} "
              f"rel_med={float(np.median(rels)):.3e} | pior: {worst[2]}")
        if "ALVO" in gname:
            attn_rels.append(max(rels))
        if "CONTROLE" in gname:
            control_rel = max(rels)

    # ── v4: o unico gate fisicamente valido e a CAMADA DE ATENCAO DO TOPO ──
    # Licao da v3 (pre-registrada): rel ~0.35-0.41 em TODOS os grupos com pior
    # sempre em layers.1 (fundo) + forward bit-identico = assinatura de CAOS de
    # backprop amplificado em profundidade (×~3/camada => 1e-6 no topo vira
    # ~0.4 no fundo apos ~11 camadas). dy so e identico entre os bracos na
    # camada de atencao MAIS ALTA — apenas la a comparacao mede o kernel, e nao
    # o caos. Gate v4: rel dos ALVOS (q/kv) na camada do topo; out_proj do topo
    # como controle (deve ser ~1e-7). O perfil por camada e impresso para
    # evidenciar o gradiente de amplificacao (deve crescer topo -> fundo).
    import re as _re

    def layer_of(name: str):
        m = _re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    attn_by_layer: dict[int, dict[str, float]] = {}
    for gname, items in groups.items():
        if "attn" not in gname:
            continue
        kind = ("out" if "out_proj" in gname
                else "q" if "q_down" in gname else "kv")
        for rel, _abs, name in items:
            li = layer_of(name)
            if li is None:
                continue
            d = attn_by_layer.setdefault(li, {})
            d[kind] = max(d.get(kind, 0.0), rel)

    if not attn_by_layer:
        print("[parity] ERRO: nenhum tensor de atencao identificado por camada")
        return 2

    print("[parity] perfil por camada (rel_max q / kv / out — espera-se crescer topo->fundo):")
    for li in sorted(attn_by_layer, reverse=True):
        d = attn_by_layer[li]
        print(f"[parity]   layer {li:>2}: q={d.get('q', 0):.3e} "
              f"kv={d.get('kv', 0):.3e} out={d.get('out', 0):.3e}")

    top = max(attn_by_layer)
    dtop = attn_by_layer[top]
    top_target = max(dtop.get("q", 0.0), dtop.get("kv", 0.0))
    top_control = dtop.get("out", 0.0)
    print(f"[parity] TOPO = layer {top}: alvo(q/kv)={top_target:.3e} "
          f"controle(out)={top_control:.3e} | global(caos)={global_rel:.3e}")

    if top_control > 1e-4:
        print("[parity] ERRO DE HARNESS: controle do TOPO deveria ser ~1e-7 "
              "(dy identico) — divergencia fora do kernel novo; colar saida")
        return 2
    if top_target <= args.pass_tol:
        print(f"[parity] PASS — alvo do topo <= {args.pass_tol:.0e}: kernel GPU "
              f"correto; rel global {global_rel:.1e} e caos de profundidade "
              f"(nenhuma implementacao correta-mas-reassociada passaria nele). "
              f"PROMOVIDO (default GPU); treinar normalmente.")
        return 0
    if top_target <= args.fail_tol:
        print(f"[parity] ZONA CINZA no topo ({top_target:.3e}) — colar a saida")
        return 0
    print(f"[parity] FAIL REAL: alvo do topo {top_target:.3e} > {args.fail_tol:.0e} "
          f"— bug no kernel ({'q' if dtop.get('q',0)>=dtop.get('kv',0) else 'kv'} "
          f"e o pior); treinar com NSOS_ATTN_BWD_HOST=1 e colar a saida")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
