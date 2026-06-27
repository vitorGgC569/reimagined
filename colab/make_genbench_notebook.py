"""Generator for colab/train_genbench_mqar_copy.ipynb.

Gold-standard ARCHITECTURE evaluation suite (branch feature/nsos-gpu-phases12),
phase 1: MQAR (multi-query associative recall) + selective-copy length
extrapolation -- the two most decisive synthetic probes the field uses to judge a
sequence-model architecture cheaply, BEFORE scaling.

Why these two:
  * MQAR (Arora et al., "Zoology", HazyResearch): in-context associative recall
    explains ~82% of the quality gap between sub-quadratic mixers and attention on
    real LM. It is the single most predictive synthetic. Multi-query = several
    recalls per forward pass (the version that reflects real language).
  * Selective copy with length extrapolation (Mamba, Gu & Dao): content-selective
    copy that must EXTRAPOLATE to sequences longer than training -- the headline
    strength of a good SSM (linear-time + generalizes in length).

Methodology (gold standard): isolate the SEQUENCE MIXER by comparing three pure
variants at equal size -- attention-only (Transformer baseline), Mamba-only
(proper selective SSM), and the NSOS hybrid -- across multiple seeds, reporting
mean +/- std vs the random baseline, plus capacity (MQAR n_kv) and length
(copy field length) curves.  Pre-registered verdict at the end.

These tasks are pure synthetic token-id sequences: NO tokenizer, NO dataset
download, tiny models -> runs in minutes, so you can iterate without spending the
contabil-scale T4 budget.

Run once (pure JSON authoring, no GPU): python colab/make_genbench_notebook.py
"""
import json
from pathlib import Path

BRANCH = "nsos-gpu-phases12"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md("""# NSOS — Suíte de generalização (padrão-ouro): MQAR + extrapolação de comprimento

Fase 1 da avaliação de **arquitetura** (branch `%BRANCH%`): as duas sondas
sintéticas mais decisivas que a área usa pra julgar um mixer de sequência **barato,
antes de escalar**.

- **MQAR** (associative recall multi-query, *Zoology*/HazyResearch): recall em
  contexto explica ~82% do gap de qualidade entre mixers sub-quadráticos e atenção
  em LM real. É a sonda mais preditiva que existe.
- **Selective copy com extrapolação de comprimento** (Mamba): cópia seletiva que
  precisa **extrapolar** pra sequências mais longas que o treino — a força do SSM.

**Método (padrão-ouro):** isola o MIXER comparando 3 variantes do MESMO tamanho —
**atenção-pura** (baseline Transformer), **Mamba-puro** (SSM seletivo correto) e o
**híbrido NSOS** — com várias seeds, média ± desvio vs baseline aleatório, + curvas
de capacidade (MQAR n_kv) e comprimento (copy). **Veredito pré-registrado** no fim.

Tarefas 100% sintéticas (ids): **sem tokenizer, sem download**, modelos minúsculos
→ roda em **minutos**."""))

cells.append(md("## 1 — GPU + (opcional) Drive p/ cache do build"))
cells.append(code("""import os, subprocess
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_genbench'); DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print('[drive] indisponivel (ok):', e); DRIVE_ROOT = Path('/content/nsos_genbench'); DRIVE_ROOT.mkdir(exist_ok=True)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=False)
"""))

cells.append(md("## 2 — Clone da branch + prova de versão"))
cells.append(code("""import os, subprocess
from pathlib import Path
REPO_URL = 'https://github.com/vitorGgC569/reimagined.git'
REPO_ROOT = Path('/content/reimagined'); BRANCH = '%BRANCH%'
TOKEN = None
try:
    from google.colab import userdata
    for key in ('GITHUB_TOKEN', 'GH_TOKEN'):
        try: TOKEN = userdata.get(key)
        except Exception: TOKEN = None
        if TOKEN: break
except Exception: pass
url = REPO_URL.replace('https://', f'https://oauth2:{TOKEN}@') if TOKEN else REPO_URL
if REPO_ROOT.exists():
    subprocess.run(['git','-C',str(REPO_ROOT),'remote','set-url','origin',url], check=True)
    subprocess.run(['git','-C',str(REPO_ROOT),'fetch','--depth','1','origin',BRANCH], check=True)
    subprocess.run(['git','-C',str(REPO_ROOT),'checkout',BRANCH], check=True)
    subprocess.run(['git','-C',str(REPO_ROOT),'reset','--hard',f'origin/{BRANCH}'], check=True)
else:
    subprocess.run(['git','clone','--depth','1','--branch',BRANCH,url,str(REPO_ROOT)], check=True)
subprocess.run(['git','-C',str(REPO_ROOT),'remote','set-url','origin',REPO_URL], check=False)
os.environ['REPO_ROOT'] = str(REPO_ROOT)
sha = subprocess.run(['git','-C',str(REPO_ROOT),'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
mamba = (REPO_ROOT/'OXN/nsos/src/mamba2.cpp').read_text('utf-8')
assert 'proper_selective_ssm' in mamba and 'forward_proper_step' in mamba, 'fonte stale'
print(f'[ver] HEAD={sha} | Mamba proper + passo on-device: OK')
"""))

cells.append(md("""## 3 — Build `nsos_ext` + gate de gradcheck (CUDA T4 sm_75)"""))
cells.append(code("""import os, sys, subprocess, shutil, glob
from pathlib import Path
if 'nsos_ext' in sys.modules:
    raise RuntimeError('nsos_ext ja carregado — Runtime > Restart.')
PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
sha = subprocess.run(['git','-C',str(REPO_ROOT),'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
BUILD = str(REPO_ROOT / 'OXN/nsos/build-genbench')
EXT_DIR = Path('/content/nsos_ext_genbench'); EXT_DIR.mkdir(exist_ok=True)
CACHE_SO = DRIVE_ROOT / '_bootstrap' / f'nsos_ext_genbench.{PYTAG}.{sha}.so'
subprocess.run([sys.executable,'-m','pip','install','-q','pybind11','numpy'], check=True)
if CACHE_SO.exists():
    shutil.copy2(CACHE_SO, EXT_DIR/'nsos_ext.so'); os.environ['NSOS_BUILD_GENBENCH']=BUILD
    print('[build] .so do cache:', CACHE_SO)
else:
    r = subprocess.run(['cmake','-S',str(REPO_ROOT/'OXN/nsos'),'-B',BUILD,
                        '-DNSOS_ENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=75',
                        '-DNSOS_BUILD_PYTHON=ON','-DNSOS_BUILD_TESTS=ON',
                        '-DNSOS_BUILD_CLI=OFF','-DNSOS_BUILD_API=OFF','-DNSOS_BUILD_OXTAMEM=OFF',
                        '-DCMAKE_BUILD_TYPE=Release',f'-DPython3_EXECUTABLE={sys.executable}'],
                       capture_output=True, text=True)
    print(r.stdout[-1000:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('cmake configure falhou')
    r = subprocess.run(['cmake','--build',BUILD,'-j','2','--target','nsos_ext','test_gradcheck'],
                       capture_output=True, text=True)
    print(r.stdout[-1000:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('build falhou')
    so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True)); assert so
    shutil.copy2(so[0], EXT_DIR/'nsos_ext.so'); CACHE_SO.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(so[0], CACHE_SO); os.environ['NSOS_BUILD_GENBENCH']=BUILD
    print('[build] ok')
# gate de gradcheck (rapido)
B = os.environ['NSOS_BUILD_GENBENCH']
for cand in (f'{B}/test_gradcheck', f'{B}/Release/test_gradcheck', f'{B}/test_gradcheck.exe'):
    if os.path.exists(cand):
        rr = subprocess.run([cand], capture_output=True, text=True)
        print('[gradcheck]', 'PASS' if rr.returncode==0 else 'FAIL'); break
"""))

cells.append(md("""## 4 — Definição das tarefas (geradores sintéticos + avaliação)

**MQAR:** `[BOS k1 v1 ... kN vN SEP]` no prompt; resposta `[q1 v1 q2 v2 ...]` —
o modelo, em cada query `qi`, deve emitir o valor `vi` ligado a ela. A loss
(`train_supervised`) cai na resposta; a acurácia é medida **só nas posições de
valor** (recall). `n_kv` = nº de pares (capacidade).

**Selective copy:** `[BOS <campo de comprimento L com K dados entre BLANK> MARK]`;
resposta `[os K dados em ordem]`. Treina campo curto, testa campo **mais longo**
(extrapolação). Métrica = exact-match dos K tokens."""))
cells.append(code("""import random, numpy as np

def make_mqar(rng, n_kv, n_q, n_sym):
    BOS, SEP = n_sym, n_sym + 1
    keys = rng.sample(range(n_sym), n_kv)
    vals = [rng.randrange(n_sym) for _ in range(n_kv)]
    kv = dict(zip(keys, vals))
    prompt = [BOS]
    for k, v in zip(keys, vals):
        prompt += [k, v]
    prompt += [SEP]
    qk = [rng.choice(keys) for _ in range(n_q)]
    answer = []
    for q in qk:
        answer += [q, kv[q]]
    return prompt, answer, qk, kv

def make_selcopy(rng, field_len, n_data, n_sym):
    BLANK, MARK, BOS = n_sym, n_sym + 1, n_sym + 2
    pos = sorted(rng.sample(range(field_len), n_data))
    data = [rng.randrange(n_sym) for _ in range(n_data)]
    field = [BLANK] * field_len
    for p, d in zip(pos, data):
        field[p] = d
    prompt = [BOS] + field + [MARK]
    return prompt, data, None, None  # (prompt, answer, _, _)

def eval_mqar(model, data, V):
    cor = tot = 0
    for (prompt, answer, qk, kv) in data:
        seq = prompt + answer
        lg = np.asarray(model.forward_ids(seq).cpu().numpy()).reshape(-1, V)
        for i, q in enumerate(qk):
            pred = int(np.argmax(lg[len(prompt) + 2 * i]))  # posicao da query qi
            cor += int(pred == kv[q]); tot += 1
    return cor / max(tot, 1)

def eval_copy(model, data, V):
    cor = tot = 0
    for (prompt, answer, _, _) in data:
        seq = prompt + answer
        lg = np.asarray(model.forward_ids(seq).cpu().numpy()).reshape(-1, V)
        ok = all(int(np.argmax(lg[len(prompt) + j - 1])) == answer[j] for j in range(len(answer)))
        cor += int(ok); tot += 1
    return cor / max(tot, 1)  # exact-match (cópia inteira certa)

print('[tasks] MQAR + selective-copy definidos')
"""))

cells.append(md("""## 5 — Variantes do MIXER (mesmo tamanho) + flags corrigidos

3 variantes isolando o mixer: **atenção-pura**, **Mamba-puro** (proper SSM),
**híbrido NSOS**. MoE/KAN/TTT OFF aqui de propósito — a comparação é do *mixer de
sequência* (é o que o recall mede). Flags corrigidos das vias Mamba ligados."""))
cells.append(code("""import os, sys
os.environ['NSOS_MAMBA_PROPER_SSM']  = '1'
os.environ['NSOS_MAMBA_CONV_K']      = '3'
os.environ['NSOS_MAMBA_A_LOGSPACED'] = '1'
os.environ['NSOS_GPU_POOL']          = '1'
sys.path.insert(0, str(__import__('pathlib').Path('/content/nsos_ext_genbench')))
import nsos_ext as nsos
dev = nsos.Device.GPU
NUM_LAYERS = int(os.environ.get('NSOS_GEN_LAYERS', '4'))
DMODEL = int(os.environ.get('NSOS_GEN_DMODEL', '128'))

def build_variant(variant, V):
    c = nsos.ModelConfig()
    c.num_layers = NUM_LAYERS; c.d_model = DMODEL; c.vocab_size = V
    c.n_heads = 4; c.n_kv_heads = 2; c.max_context_tokens = 1024
    c.use_exact_attention_training = True; c.use_flash_attn = False
    c.use_moe = False; c.use_kan = False; c.use_ttt = False; c.dropout = 0.0
    if variant == 'attn':
        c.attention_period = 1; c.attention_slot = 0      # toda camada = atencao
    elif variant == 'mamba':
        c.attention_period = 4; c.attention_slot = 4      # slot>=period => nenhuma atencao
    else:  # hybrid (NSOS)
        c.attention_period = 2; c.attention_slot = 1      # metade atencao / metade Mamba
    m = nsos.JambaModel(c, dev); m.to(dev)
    return m

def train_on(model, lr, steps, sampler, seed):
    tr = nsos.Trainer(model, lr); tr.warmup_steps = max(50, steps // 10)
    tr.total_training_steps = steps
    rng = random.Random(seed * 991 + 7)
    model.set_training_mode(True)
    for s in range(steps):
        p, a, _, _ = sampler(rng)
        if len(p) >= 2 and len(a) >= 1:
            tr.train_supervised(p, a)
    model.set_training_mode(False)

VARIANTS = ['attn', 'mamba', 'hybrid']
SEEDS = [int(x) for x in os.environ.get('NSOS_GEN_SEEDS', '0,1').split(',')]
STEPS = int(os.environ.get('NSOS_GEN_STEPS', '2000'))
print(f'[cfg] L={NUM_LAYERS} d={DMODEL} | variantes={VARIANTS} | seeds={SEEDS} | steps={STEPS}')
"""))

cells.append(md("""## 6 — MQAR: capacidade de recall (treina n_kv=8, testa n_kv=8 e 16)

Baseline aleatório = 1/n_sym. **A pergunta:** o híbrido NSOS acompanha a atenção?
O Mamba-puro colapsa quando há mais pares (n_kv=16)? (resultado clássico do Zoology)"""))
cells.append(code("""import time, numpy as np, random
N_SYM = int(os.environ.get('NSOS_MQAR_SYM', '32')); N_Q = 4
V = N_SYM + 2
base = 1.0 / N_SYM
mqar = {}
t0 = time.time()
for variant in VARIANTS:
    in8, hard16 = [], []
    for seed in SEEDS:
        nsos.set_seed(seed)
        m = build_variant(variant, V)
        train_on(m, 2e-3, STEPS, lambda r: make_mqar(r, 8, N_Q, N_SYM), seed)
        ev = random.Random(seed * 13 + 5)
        d8  = [make_mqar(ev, 8,  N_Q, N_SYM) for _ in range(200)]
        d16 = [make_mqar(ev, 16, N_Q, N_SYM) for _ in range(200)]
        in8.append(eval_mqar(m, d8, V)); hard16.append(eval_mqar(m, d16, V))
        print(f'  [{variant:6s} seed {seed}] n_kv=8: {in8[-1]:.3f}  n_kv=16: {hard16[-1]:.3f}')
    mqar[variant] = (np.mean(in8), np.std(in8), np.mean(hard16), np.std(hard16))
print('\\n' + '=' * 64)
print(f'MQAR (acuracia de recall; baseline aleatorio={base:.3f})   ({time.time()-t0:.0f}s)')
print(f'{"variante":<10}{"n_kv=8 (treino)":>20}{"n_kv=16 (capacidade)":>24}')
for v in VARIANTS:
    a, sa, b, sb = mqar[v]
    print(f'{v:<10}{a:>12.3f}+/-{sa:.3f}{b:>16.3f}+/-{sb:.3f}')
print('=' * 64)
"""))

cells.append(md("""## 7 — Selective copy: extrapolação de comprimento (treina ≤64, testa 64/128/256)

Métrica = exact-match. **A pergunta:** o Mamba/híbrido **extrapola** pra campos mais
longos que o treino? A atenção costuma cair (codificação posicional). Esse é o
*upside* estrutural do SSM."""))
cells.append(code("""import time, numpy as np, random
N_SYM_C = int(os.environ.get('NSOS_COPY_SYM', '20')); N_DATA = 4
LT = int(os.environ.get('NSOS_COPY_LEN', '64'))
Vc = N_SYM_C + 3
base_c = (1.0 / N_SYM_C) ** N_DATA
copy = {}
t0 = time.time()
TEST_LENS = [LT, 2 * LT, 4 * LT]
for variant in VARIANTS:
    rows = {L: [] for L in TEST_LENS}
    for seed in SEEDS:
        nsos.set_seed(seed)
        m = build_variant(variant, Vc)
        # treino: comprimento de campo uniforme em [16, LT]
        def samp(r):
            return make_selcopy(r, r.randint(16, LT), N_DATA, N_SYM_C)
        train_on(m, 2e-3, STEPS, samp, seed)
        ev = random.Random(seed * 29 + 3)
        for L in TEST_LENS:
            d = [make_selcopy(ev, L, N_DATA, N_SYM_C) for _ in range(200)]
            rows[L].append(eval_copy(m, d, Vc))
        print(f'  [{variant:6s} seed {seed}] ' + '  '.join(f'L={L}:{rows[L][-1]:.3f}' for L in TEST_LENS))
    copy[variant] = {L: (np.mean(rows[L]), np.std(rows[L])) for L in TEST_LENS}
print('\\n' + '=' * 64)
print(f'SELECTIVE COPY exact-match (baseline~{base_c:.1e}; treino campo<=64)   ({time.time()-t0:.0f}s)')
hdr = f'{"variante":<10}' + ''.join(f'{("L="+str(L)+("(treino)" if L==LT else "(extrap)")):>16}' for L in TEST_LENS)
print(hdr)
for v in VARIANTS:
    print(f'{v:<10}' + ''.join(f'{copy[v][L][0]:>16.3f}' for L in TEST_LENS))
print('=' * 64)
"""))

cells.append(md("""## 8 — VEREDITO (régua pré-registrada)

Decidida ANTES de rodar (anti-auto-engano). Lê os resultados acima e classifica."""))
cells.append(code("""def grade(cond): return 'OK' if cond else 'FALHOU'

attn8, _, attn16, _ = mqar['attn']
hyb8,  _, hyb16,  _ = mqar['hybrid']
mam8,  _, mam16,  _ = mqar['mamba']
print('=' * 66)
print('1) TABLE STAKES — todas resolvem o recall facil (n_kv=8 >> baseline)?')
for v in VARIANTS:
    print(f'   {v:<8} n_kv=8={mqar[v][0]:.3f}  -> {grade(mqar[v][0] > 0.5)}')
print('\\n2) DISCRIMINADOR — hibrido NSOS acompanha a atencao na CAPACIDADE (n_kv=16)?')
print(f'   atencao={attn16:.3f}  hibrido={hyb16:.3f}  mamba-puro={mam16:.3f}')
print(f'   hibrido perto da atencao (>= 0.85*attn): {grade(hyb16 >= 0.85*attn16)}')
print(f'   (esperado classico: mamba-puro cai vs atencao -> gap de recall)')
print('\\n3) UPSIDE — Mamba/hibrido EXTRAPOLAM em comprimento (copy L=4x vs treino)?')
for v in VARIANTS:
    tr_acc = copy[v][LT][0]; ex_acc = copy[v][4*LT][0]
    print(f'   {v:<8} treino(L={LT})={tr_acc:.3f}  extrap(L={4*LT})={ex_acc:.3f}  -> {grade(ex_acc >= 0.5)}')
print('=' * 66)
print('LEITURA:')
print(' - Vale escalar se: table stakes OK + hibrido ~ atencao no MQAR n_kv=16 +')
print('   Mamba/hibrido extrapolam em comprimento.')
print(' - Sinal de rework se: hibrido colapsa no MQAR (gap de recall) E nao extrapola')
print('   -> nem tao bom quanto atencao, nem entrega o upside de SSM.')
print(' - As camadas de atencao do hibrido DEVEM fechar o gap de recall do Mamba-puro;')
print('   se fecharem, e a evidencia central de que a arquitetura NSOS se sustenta.')
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

out = Path(__file__).resolve().parent / "train_genbench_mqar_copy.ipynb"
text = json.dumps(nb, indent=1, ensure_ascii=False).replace("%BRANCH%", BRANCH)
out.write_text(text, encoding="utf-8")
print(f"wrote {out}")
