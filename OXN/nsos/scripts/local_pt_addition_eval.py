"""Local CPU validation: Portuguese addition QA with compositional hold-out.

Mirrors the controlled task in colab/train_gpu_phases_t4_v2.ipynb but as a
standalone script so the corrected stack can be trained + evaluated on a TINY
model locally (no GPU, no Colab) and produce real perplexity + exact-match
numbers — the empirical generalization evidence.

Task: "quanto e A mais B ?" -> word for A+B (A,B in 0..9, sum 0..18).
Compositional hold-out: a disjoint set of (A,B) PAIRS is reserved; every A and
every B still appears (in other pairs) in training, so solving the held-out
pairs requires GENERALIZING addition, not memorizing pairs.

Run (after building nsos_ext):
    python OXN/nsos/scripts/local_pt_addition_eval.py --build-dir <dir-with-nsos_ext.pyd> --epochs 40

Corrected paths are enabled by default (set BEFORE importing nsos_ext):
    NSOS_MAMBA_PROPER_SSM, NSOS_MOE_FP_ROUTER, NSOS_MOE_SWITCH_AUX, NSOS_MAMBA_A_LOGSPACED
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time

# Corrected-stack flags MUST be set before nsos_ext is imported / the model is
# built (they are read at construction / first optimizer step).
for k, v in {
    "NSOS_MAMBA_PROPER_SSM": "1",
    "NSOS_MAMBA_CONV_K": "3",
    "NSOS_MOE_FP_ROUTER": "1",
    "NSOS_MOE_SWITCH_AUX": "1",
    "NSOS_MAMBA_A_LOGSPACED": "1",
}.items():
    os.environ.setdefault(k, v)


def find_pyd(build_dir: str) -> str:
    cands = glob.glob(os.path.join(build_dir, "**", "nsos_ext*.pyd"), recursive=True)
    cands += glob.glob(os.path.join(build_dir, "**", "nsos_ext*.so"), recursive=True)
    if not cands:
        raise SystemExit(f"nsos_ext not found under {build_dir}")
    return os.path.dirname(cands[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, find_pyd(args.build_dir))
    import numpy as np
    import nsos_ext as nsos

    nsos.set_seed(1234)  # reproducible init

    # ── vocabulary (word-level, tiny) ────────────────────────────────────────
    NUM = ["zero", "um", "dois", "tres", "quatro", "cinco", "seis", "sete",
           "oito", "nove", "dez", "onze", "doze", "treze", "catorze", "quinze",
           "dezesseis", "dezessete", "dezoito"]
    STRUCT = ["quanto", "e", "mais", "?"]
    VOCAB = STRUCT + ["<eos>"] + NUM
    STOI = {w: i for i, w in enumerate(VOCAB)}
    EOS = STOI["<eos>"]
    V = len(VOCAB)

    def prompt_ids(a, b):
        return [STOI["quanto"], STOI["e"], STOI[NUM[a]], STOI["mais"],
                STOI[NUM[b]], STOI["?"]]

    def answer_ids(a, b):
        return [STOI[NUM[a + b]], EOS]

    # ── compositional hold-out ───────────────────────────────────────────────
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    held = sorted(all_pairs[:15])
    train = sorted(all_pairs[15:])
    assert {a for a, _ in train} == set(range(10))
    assert {b for _, b in train} == set(range(10))
    base = 1.0 / len(NUM)
    print(f"[data] vocab={V} train_pairs={len(train)} held_pairs={len(held)} "
          f"baseline_exact={base:.3f}")

    # ── tiny corrected-stack model ───────────────────────────────────────────
    dev = nsos.Device.CPU
    cfg = nsos.ModelConfig()
    cfg.num_layers = 4
    cfg.d_model = 128
    cfg.vocab_size = V
    cfg.n_heads = 4
    cfg.n_kv_heads = 2
    cfg.use_moe = True
    cfg.num_experts = 4
    cfg.num_experts_per_token = 2
    cfg.moe_period = 2
    cfg.moe_slot = 1
    cfg.attention_period = 4
    cfg.attention_slot = 3
    cfg.use_ttt = False
    cfg.dropout = 0.0
    cfg.max_context_tokens = 64
    cfg.use_exact_attention_training = True

    model = nsos.JambaModel(cfg, dev)
    model.to(dev)
    model.set_training_mode(True)
    tr = nsos.Trainer(model, 2e-3)
    tr.warmup_steps = 100
    tr.eos_token_id = EOS
    tr.moe_aux_loss_scale = 0.01
    tr.total_training_steps = args.epochs * len(train)
    print(f"[model] layers={cfg.num_layers} d={cfg.d_model} V={V} "
          f"| flags proper_ssm={os.environ['NSOS_MAMBA_PROPER_SSM']} "
          f"switch_aux={os.environ['NSOS_MOE_SWITCH_AUX']} "
          f"fp_router={os.environ['NSOS_MOE_FP_ROUTER']}")

    rng2 = random.Random(123)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        order = train[:]
        rng2.shuffle(order)
        ep_loss = 0.0
        for (a, b) in order:
            ep_loss += tr.train_supervised(prompt_ids(a, b), answer_ids(a, b))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} loss/pair={ep_loss/len(order):.4f} "
                  f"({time.perf_counter()-t0:.0f}s)")

    # ── eval: perplexity + exact-match ───────────────────────────────────────
    model.set_training_mode(False)

    def eval_pairs(pairs):
        nll = 0.0
        ntok = 0
        correct = 0
        for (a, b) in pairs:
            p = prompt_ids(a, b)
            ans = answer_ids(a, b)
            seq = p + ans
            logits = np.asarray(model.forward_ids(seq).numpy()).reshape(len(seq), V)
            pred = int(np.argmax(logits[len(p) - 1]))
            correct += int(pred == ans[0])
            for pos in range(len(p), len(seq)):
                row = logits[pos - 1].astype(np.float64)
                row -= row.max()
                pr = np.exp(row)
                pr /= pr.sum()
                nll += -math.log(max(pr[seq[pos]], 1e-12))
                ntok += 1
        return math.exp(nll / max(ntok, 1)), correct / len(pairs)

    ppl_tr, em_tr = eval_pairs(train)
    ppl_he, em_he = eval_pairs(held)
    print("=" * 60)
    print(f"TRAIN    ppl={ppl_tr:8.3f}  exact-match={em_tr:.3f}")
    print(f"HELD-OUT ppl={ppl_he:8.3f}  exact-match={em_he:.3f}  (baseline {base:.3f})")
    print("=" * 60)
    verdict = ("GENERALIZOU" if em_he > 3 * base else
               ("parcial" if em_he > base else "memorizou/nao aprendeu"))
    print(f"VEREDITO generalizacao: {verdict}")
    print("exemplos held-out:")
    for (a, b) in held[:8]:
        p = prompt_ids(a, b)
        logits = np.asarray(model.forward_ids(p).numpy()).reshape(len(p), V)
        pred = VOCAB[int(np.argmax(logits[-1]))]
        print(f"  {NUM[a]} + {NUM[b]} = {NUM[a+b]:>10} | modelo: {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
