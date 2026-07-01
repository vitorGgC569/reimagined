"""Generator for colab/train_genbench_attn_probe.ipynb.

Focused ATTENTION-recall probe (branch nsos-gpu-phases12).  The genbench v5 run
showed the mixer ranking mamba > hybrid > attn on associative recall, with the
attention control stuck near baseline (n_kv=8: attn 0.19 vs mamba 0.82).  A
line-by-line read of the LIVE CUDA path (batched_gqa_causal_attention_kernel in
kernels.cu:1270 + rope_apply in attention_train_kernels.cu) confirmed the
attention math is CORRECT (gradcheck + gpu-parity pass) -- the recall weakness is
architectural/optimization, not a bug.  Ranked, code-grounded suspects:
  1. GQA n_kv=2 (genbench forces it; prod default is 4): 4 query heads collapse
     to 2 K/V key-spaces -> less content-matching capacity for the induction head.
  2. RoPE x head_dim=32 (d_model/n_heads = 128/4): only ~half the 16 freq pairs
     rotate slowly enough to support position-invariant content match over the
     sequence -> copy (positional) works, recall (content) doesn't.  Lever:
     bigger head_dim via n_heads=2 -> head_dim=64.
  3. LR / underfitting: attention loss falls monotonically but never converges in
     10k steps.

This notebook isolates suspects 1-3 with a CONFIG-ONLY sweep (no code changes):
GQA {n_kv 2 vs 4} x head_dim {32 (n_heads=4) vs 64 (n_heads=2)} x LR {2e-3, 5e-3},
on AR recall (n_kv=8 train; eval n_kv=8 and the harder n_kv=16), with mamba as the
reference ceiling and a PRE-REGISTERED, HONESTLY CALIBRATED verdict (a working
Transformer solves AR n_kv=8 ~1.0 -- that is the real ceiling, not our mamba).

Reuses the proven infra cells (GPU/drive, clone+version-proof, build+gate,
task defs) verbatim from train_genbench_mqar_copy.ipynb so the build cache is
shared (no rebuild if the genbench .so for this SHA is already cached).

Run once (pure JSON authoring, no GPU): python colab/make_attn_probe_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENBENCH = HERE / "train_genbench_mqar_copy.ipynb"
OUT = HERE / "train_genbench_attn_probe.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


genbench = json.loads(GENBENCH.read_text(encoding="utf-8"))
# Infra cells 1..8 = GPU/drive, clone+version-proof, build+gate, task defs.
infra = genbench["cells"][1:9]

cells = []
cells.append(md("""# NSOS — Sondagem da ATENÇÃO (recall): destravar o híbrido?

Investigação focada (branch `nsos-gpu-phases12`), motivada pelo genbench v5:
na sonda de **recall associativo** o ranking foi **mamba > híbrido > atenção**,
com a atenção travada perto do baseline (n_kv=8: attn **0.19** vs mamba **0.82**).

**Auditoria linha-a-linha do caminho VIVO no T4** (`batched_gqa_causal_attention_kernel`
+ `rope_apply`) confirmou: a matemática da atenção está **correta** (gradcheck +
paridade GPU passam) — a fraqueza de recall é **arquitetural/otimização, não bug**.
Suspeitos ranqueados (com evidência de código):
1. **GQA `n_kv=2`** — o genbench força; o default de produção é 4. 4 query heads
   colapsam em 2 espaços de K/V → menos capacidade de casamento-por-conteúdo.
2. **RoPE × `head_dim=32`** (128/4) — só ~metade dos 16 pares de frequência gira
   devagar o bastante p/ casamento invariante-a-posição. Copy (posicional)
   funciona, recall (conteúdo) não. Alavanca: `head_dim=64` via `n_heads=2`.
3. **LR / subajuste** — a loss da atenção cai monotônica mas não converge em 10k.

**Este notebook isola 1-3 com um sweep SÓ-CONFIG (sem mudar código):**
`GQA {n_kv 2 vs 4}` × `head_dim {32 vs 64}` × `LR {2e-3, 5e-3}`, em AR recall
(treina n_kv=8; avalia n_kv=8 e o mais difícil n_kv=16), com Mamba como **teto de
referência** e um veredito **pré-registrado e calibrado honestamente**.

⚠️ **Calibração honesta:** um Transformer que FUNCIONA resolve AR n_kv=8 ~**1.0** —
esse é o teto real, NÃO o nosso Mamba (0.82). O objetivo é ver quão perto a atenção
chega com os fixes de config, e se isso levanta o híbrido em direção ao Mamba.

Requer que o genbench v5 já tenha sido rodado ao menos uma vez (compartilha o
cache do build `.so`). Runtime → GPU (T4)."""))

# Reuse the proven infra cells verbatim.
cells.extend(infra)

cells.append(md("""## 5 — Harness da sondagem (build por-condição + treino AR)

`build_attn_variant(variant, V, n_heads, n_kv)` — mesmo tamanho (d=128, 4
camadas), MoE/KAN/TTT OFF. QAT OFF (isola a arquitetura, float o tempo todo,
igual à comparação de mixer do genbench). `attn`: todas as camadas atenção;
`mamba`: todas SSM; `hybrid`: alternado."""))
cells.append(code("""import os, sys, random, time
import numpy as np
os.environ['NSOS_MAMBA_PROPER_SSM']  = '1'
os.environ['NSOS_MAMBA_CONV_K']      = '3'
os.environ['NSOS_MAMBA_A_LOGSPACED'] = '1'
os.environ['NSOS_GPU_POOL']          = '1'
sys.path.insert(0, str(__import__('pathlib').Path('/content/nsos_ext_genbench')))
import nsos_ext as nsos
dev = nsos.Device.GPU

def build_attn_variant(variant, V, n_heads, n_kv, rope_theta=10000.0):
    c = nsos.ModelConfig()
    c.num_layers = 4; c.d_model = 128; c.vocab_size = V
    c.n_heads = n_heads; c.n_kv_heads = n_kv; c.max_context_tokens = 1024
    c.rope_theta = float(rope_theta)   # config path (portavel; env e instavel no Windows)
    c.use_exact_attention_training = True; c.use_flash_attn = False
    c.use_moe = False; c.use_kan = False; c.use_ttt = False; c.dropout = 0.0
    if variant == 'attn':
        c.attention_period = 1; c.attention_slot = 0      # 100% atencao
    elif variant == 'mamba':
        c.attention_period = 4; c.attention_slot = 4      # 0% atencao (so SSM)
    else:
        c.attention_period = 2; c.attention_slot = 1      # alternado
    m = nsos.JambaModel(c, dev); m.to(dev)
    return m

def train_ar(model, lr, steps, batch, n_kv_train, n_key, n_val, seed, label):
    tr = nsos.Trainer(model, lr); tr.warmup_steps = max(50, steps // 10)
    tr.total_training_steps = steps
    tr.phase_scheduler.progressive_qat_enabled = False   # isola: float o tempo todo
    tr.first_token_loss_scale = 1.0; tr.eos_loss_scale = 1.0
    rng = random.Random(seed * 991 + 7)
    model.set_training_mode(True)
    win = max(1, steps // 5); run = 0.0
    for s in range(steps):
        ps, ans = [], []
        for _ in range(batch):
            p, a = make_ar(rng, n_kv_train, n_key, n_val); ps.append(p); ans.append(a)
        run += tr.train_supervised_batch(ps, ans)
        if (s + 1) % win == 0:
            print(f'    [{label}] {s+1}/{steps} loss~{run/win:.3f}'); run = 0.0
    model.set_training_mode(False)

N_KEY = 16; N_VAL = 16; V = N_KEY + N_VAL + 2; base = 1.0 / N_VAL
KV_TEST = [8, 16]
STEPS = int(os.environ.get('NSOS_PROBE_STEPS', '6000'))
BATCH = int(os.environ.get('NSOS_PROBE_BATCH', '16'))
SEEDS = [int(x) for x in os.environ.get('NSOS_PROBE_SEEDS', '0,1').split(',')]
# (rotulo, variante, n_heads, n_kv, lr)
CONDS = [
    ('attn   hd32 GQA(nkv2) lr2e-3', 'attn',   4, 2, 2e-3),  # baseline (repro genbench)
    ('attn   hd32 MHA(nkv4) lr2e-3', 'attn',   4, 4, 2e-3),  # lever GQA
    ('attn   hd64 MHA(nh2)  lr2e-3', 'attn',   2, 2, 2e-3),  # lever head_dim (RoPE)
    ('attn   hd64 MHA(nh2)  lr5e-3', 'attn',   2, 2, 5e-3),  # head_dim + LR
    ('hybrid hd64 MHA(nh2)  lr2e-3', 'hybrid', 2, 2, 2e-3),  # payoff: levanta o hibrido?
    ('mamba  (teto ref)     lr2e-3', 'mamba',  4, 2, 2e-3),  # referencia
]
print(f'[cfg] d=128 L=4 | AR treina n_kv=8, avalia {KV_TEST} | steps={STEPS} batch={BATCH} seeds={SEEDS}')
print(f'[cfg] {len(CONDS)} condicoes x {len(SEEDS)} seeds = {len(CONDS)*len(SEEDS)} runs | QAT=OFF (float)')
"""))

cells.append(md("""## 6 — Sweep (AR recall) — treina n_kv=8, avalia n_kv=8 e 16"""))
cells.append(code("""res = {}; t0 = time.time()
for (label, variant, nh, nkv, lr) in CONDS:
    accs = {k: [] for k in KV_TEST}
    for seed in SEEDS:
        nsos.set_seed(seed)
        m = build_attn_variant(variant, V, nh, nkv)
        train_ar(m, lr, STEPS, BATCH, 8, N_KEY, N_VAL, seed, f'{label} s{seed}')
        ev = random.Random(seed * 13 + 5)
        for k in KV_TEST:
            d = [make_ar(ev, k, N_KEY, N_VAL) for _ in range(300)]
            accs[k].append(eval_ar(m, d, V))
        print(f'  [{label} s{seed}] ' + ' '.join(f'n_kv={k}:{accs[k][-1]:.2f}' for k in KV_TEST))
    res[label] = {k: (float(np.mean(accs[k])), float(np.std(accs[k]))) for k in KV_TEST}
print('\\n' + '=' * 78)
print(f'ATTN SWEEP — AR recall (media+/-desvio, {len(SEEDS)} seeds; baseline={base:.3f})   ({time.time()-t0:.0f}s)')
print(f'{"condicao":<32}' + ''.join(f'{("n_kv="+str(k)):>18}' for k in KV_TEST))
for (label, *_ ) in CONDS:
    print(f'  {label:<30}' + ''.join(f'{res[label][k][0]:>11.3f}+/-{res[label][k][1]:.2f}' for k in KV_TEST))
print('=' * 78)
"""))

cells.append(md("""## 7 — VEREDITO (calibrado no teto real, não no Mamba)"""))
cells.append(code("""print('=' * 78)
attn_labels = [l for (l, v, *_ ) in CONDS if v == 'attn']
base_lbl = 'attn   hd32 GQA(nkv2) lr2e-3'
best_attn = max(attn_labels, key=lambda l: res[l][8][0])
mamba_lbl = 'mamba  (teto ref)     lr2e-3'
hyb_lbl   = 'hybrid hd64 MHA(nh2)  lr2e-3'
print('SWEEP DA ATENCAO — leitura:')
print(f'  baseline atencao (hd32 GQA): n_kv=8 = {res[base_lbl][8][0]:.3f}')
print(f'  MELHOR atencao ({best_attn.strip()}): n_kv=8 = {res[best_attn][8][0]:.3f}  (n_kv=16 = {res[best_attn][16][0]:.3f})')
print(f'  Mamba (teto interno):        n_kv=8 = {res[mamba_lbl][8][0]:.3f}  (n_kv=16 = {res[mamba_lbl][16][0]:.3f})')
print(f'  Hibrido c/ melhor-attn cfg:  n_kv=8 = {res[hyb_lbl][8][0]:.3f}  (n_kv=16 = {res[hyb_lbl][16][0]:.3f})')
gain = res[best_attn][8][0] - res[base_lbl][8][0]
print(f'\\n  Ganho de config na atencao (melhor - baseline): +{gain:.3f}')
print('\\nCALIBRACAO HONESTA: um Transformer que FUNCIONA resolve AR n_kv=8 ~1.0.')
print('  Esse e o teto REAL -- nao o nosso Mamba. Regua pre-registrada:')
if res[best_attn][8][0] >= 0.70:
    print(f'  -> DESTRAVOU por CONFIG: melhor atencao {res[best_attn][8][0]:.2f} >= 0.70.')
    print('     GQA/head_dim eram o gargalo. Proximo: usar essa cfg no hibrido/producao.')
elif res[best_attn][8][0] >= 0.45:
    print(f'  -> PARCIAL: {res[best_attn][8][0]:.2f} (subiu de {res[base_lbl][8][0]:.2f} mas < 0.70).')
    print('     Config ajuda mas nao basta -> alavancas de CODIGO: RoPE base theta maior')
    print('     e/ou QK-norm (aceleram induction heads). Implementar sob flag + gradcheck.')
else:
    print(f'  -> NAO destravou por config ({res[best_attn][8][0]:.2f}). O teto e mais fundo:')
    print('     provavelmente RoPE-vs-recall estrutural -> testar NoPE-em-alguns-heads / QK-norm')
    print('     (mudancas de codigo), ou aceitar schedule Mamba-pesado no hibrido.')
print('\\nOBS: mesmo destravada, a atencao pode ficar < Mamba nesta escala minuscula')
print('(o estado SSD e memoria associativa direta). "Superar tudo" precisa de escala real,')
print('nao de sonda sintetica -- isto aqui decide "vale escalar", nao "e SOTA".')
print('=' * 78)
"""))

cells.append(md("""## 8 — Sweep de RoPE `theta` (a atenção melhora com mais dims lentos?)

Teste da hipótese RoPE: `theta` maior → mais pares de frequência giram devagar →
mais dimensões por-head **invariantes a posição** para casamento por conteúdo
(recall). **Ressalva honesta:** o par de maior frequência (i=0) gira ~1 rad/token
independente de `theta` — então `theta` grande **não** vira NoPE, só **adiciona
dims lentos**. Como dobrar `head_dim` (que também adiciona dims lentos) rendeu só
+0.03, a expectativa aqui é BAIXA — mas o teste é barato e fecha a questão. Roda
na melhor arquitetura de atenção do sweep anterior (hd64 MHA) + no híbrido."""))
cells.append(code("""THETAS = [float(x) for x in os.environ.get('NSOS_PROBE_THETAS', '10000,1e6,1e8').split(',')]
theta_res = {}; t0 = time.time()
# (rotulo, variante, n_heads, n_kv, lr) -- fixa a melhor arq de atencao (hd64 MHA)
THETA_CONDS = [
    ('attn   hd64 MHA(nh2)', 'attn',   2, 2, 2e-3),
    ('hybrid hd64 MHA(nh2)', 'hybrid', 2, 2, 2e-3),
]
for (label, variant, nh, nkv, lr) in THETA_CONDS:
    for th in THETAS:
        accs = {k: [] for k in KV_TEST}
        for seed in SEEDS:
            nsos.set_seed(seed)
            m = build_attn_variant(variant, V, nh, nkv, rope_theta=th)
            train_ar(m, lr, STEPS, BATCH, 8, N_KEY, N_VAL, seed, f'{label} theta={th:.0e} s{seed}')
            ev = random.Random(seed * 13 + 5)
            for k in KV_TEST:
                d = [make_ar(ev, k, N_KEY, N_VAL) for _ in range(300)]
                accs[k].append(eval_ar(m, d, V))
            print(f'  [{label} theta={th:.0e} s{seed}] ' + ' '.join(f'n_kv={k}:{accs[k][-1]:.2f}' for k in KV_TEST))
        theta_res[(label, th)] = {k: (float(np.mean(accs[k])), float(np.std(accs[k]))) for k in KV_TEST}
print('\\n' + '=' * 78)
print(f'ROPE THETA SWEEP — AR recall (media+/-desvio, {len(SEEDS)} seeds; baseline={base:.3f})   ({time.time()-t0:.0f}s)')
print(f'{"cond / theta":<30}' + ''.join(f'{("n_kv="+str(k)):>18}' for k in KV_TEST))
for (label, variant, nh, nkv, lr) in THETA_CONDS:
    for th in THETAS:
        r = theta_res[(label, th)]
        print(f'  {label+" th="+format(th,".0e"):<28}' + ''.join(f'{r[k][0]:>11.3f}+/-{r[k][1]:.2f}' for k in KV_TEST))
print('=' * 78)
"""))

cells.append(md("""## 9 — VEREDITO theta"""))
cells.append(code("""print('=' * 78)
attn_th = {th: theta_res[('attn   hd64 MHA(nh2)', th)][8][0] for th in THETAS}
best_th = max(THETAS, key=lambda th: attn_th[th])
base_th = attn_th.get(10000.0, attn_th[min(THETAS)])
print('ROPE THETA (atencao hd64 MHA), recall n_kv=8:')
for th in THETAS:
    print(f'  theta={th:.0e}: {attn_th[th]:.3f}')
print(f'\\n  melhor theta = {best_th:.0e} -> {attn_th[best_th]:.3f}  (baseline theta=1e4 -> {base_th:.3f})')
delta = attn_th[best_th] - base_th
print(f'  ganho de theta: +{delta:.3f}')
print('\\nLEITURA:')
if attn_th[best_th] >= 0.70:
    print('  -> RoPE ERA o gargalo: theta grande destravou. Fix principal = rope_theta maior')
    print('     (e/ou NoPE-em-alguns-heads). Adotar no config da atencao.')
elif delta >= 0.10:
    print('  -> RoPE contribui (ganho real mas nao resolve). Combinar theta + QK-norm (codigo).')
else:
    print('  -> RoPE NAO e o gargalo principal (ganho <0.10, como o head_dim ja sugeria).')
    print('     A causa e TRAINABILITY do induction head -> proxima alavanca = QK-norm')
    print('     (normalizar q,k por head antes do dot; acelera formacao de induction heads),')
    print('     ou aceitar que o Mamba e o mixer de recall e usar schedule Mamba-pesado.')
print('=' * 78)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT}")
