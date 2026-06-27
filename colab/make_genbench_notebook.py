"""Generator for colab/train_genbench_mqar_copy.ipynb.

Gold-standard ARCHITECTURE evaluation suite (branch feature/nsos-gpu-phases12),
phase 1: associative recall (capacity curve) + selective-copy length extrapolation
-- the two most decisive synthetic probes the field uses to judge a sequence-model
architecture cheaply, BEFORE scaling.

v2 of this notebook (the v1 run found ALL variants -- including the attention
positive control -- stuck at the random baseline, i.e. the HARNESS/regime was
broken, not the architecture).  Fixes in v2:
  * POSITIVE CONTROL FIRST: single-query associative recall (query in the prompt,
    answer = just the value) -> loss is pure recall signal (no random-token noise).
    A Transformer must solve this; if it doesn't, we declare the harness/regime
    insufficient and DO NOT draw architecture conclusions.
  * BATCHED training (train_supervised_batch) -> ~batch x more data per step, and
    many more total examples (a from-scratch model needs ~1e4-1e5 examples to
    INDUCE recall, not 2e3).
  * first_token_loss_scale = eos_loss_scale = 1.0 (don't over-weight token 0).
  * prints training loss (visibility) + a pre-registered verdict with the
    attention control as the gate.

Tasks are pure synthetic token-id sequences: NO tokenizer, NO dataset download,
tiny models -> minutes.  Methodology: isolate the SEQUENCE MIXER by comparing
three equal-size pure variants -- attention-only (Transformer baseline),
Mamba-only (proper selective SSM), and the NSOS hybrid -- across seeds.

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

cells.append(md("""# NSOS — Suíte de generalização (padrão-ouro) v2: recall associativo + extrapolação

Fase 1 da avaliação de **arquitetura** (branch `%BRANCH%`). **v2** corrige o harness:
na v1 **todas** as variantes — incluindo o controle de **atenção** — ficaram no
baseline aleatório, o que indica **teste quebrado, não arquitetura ruim** (um
Transformer resolve recall trivialmente). Correções:

- **Controle positivo primeiro:** AR single-query (query no prompt, resposta = só o
  valor) → loss 100% sinal de recall. A atenção *tem* que passar; se não passar,
  o veredito é "harness insuficiente" e **não** se conclui nada de arquitetura.
- **Treino em batch** (`train_supervised_batch`) → ~batch× mais dados/step e MUITO
  mais exemplos totais (um modelo do zero precisa de ~1e4-1e5 exemplos pra
  **induzir** recall, não 2e3).
- `first_token_loss_scale = eos_loss_scale = 1.0`; **imprime a loss**; veredito
  pré-registrado com a atenção como gate.

Tarefas 100% sintéticas (ids): **sem tokenizer, sem download** → minutos. Compara o
MIXER isolado: **atenção-pura** vs **Mamba-puro** (proper SSM) vs **híbrido NSOS**."""))

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
B = os.environ['NSOS_BUILD_GENBENCH']
for cand in (f'{B}/test_gradcheck', f'{B}/Release/test_gradcheck', f'{B}/test_gradcheck.exe'):
    if os.path.exists(cand):
        rr = subprocess.run([cand], capture_output=True, text=True)
        print('[gradcheck]', 'PASS' if rr.returncode==0 else 'FAIL'); break
"""))

cells.append(md("""## 4 — Tarefas (sintéticas) + avaliação

**AR (recall associativo, single-query):** prompt `[BOS k1 v1 ... kN vN SEP q]`,
resposta `[valor de q]`. Loss = só o valor → **100% sinal de recall**. `n_kv` =
capacidade (mais pares = mais difícil). É o **controle positivo**: atenção tem que
acertar ~1.0.

**Selective copy:** prompt `[BOS <campo L com K dados entre BLANK> MARK]`, resposta
`[os K dados em ordem]`. Treina campo curto, testa **mais longo** (extrapolação).
Métrica = exact-match."""))
cells.append(code("""import random, numpy as np

def make_ar(rng, n_kv, n_sym):
    BOS, SEP = n_sym, n_sym + 1
    keys = rng.sample(range(n_sym), n_kv)
    vals = [rng.randrange(n_sym) for _ in range(n_kv)]
    kv = dict(zip(keys, vals))
    q = rng.choice(keys)
    prompt = [BOS]
    for k, v in zip(keys, vals):
        prompt += [k, v]
    prompt += [SEP, q]
    return prompt, [kv[q]]          # (prompt, answer=[value])

def make_selcopy(rng, field_len, n_data, n_sym):
    BLANK, MARK, BOS = n_sym, n_sym + 1, n_sym + 2
    pos = sorted(rng.sample(range(field_len), n_data))
    data = [rng.randrange(n_sym) for _ in range(n_data)]
    field = [BLANK] * field_len
    for p, d in zip(pos, data):
        field[p] = d
    return [BOS] + field + [MARK], data   # (prompt, answer=data)

def eval_ar(model, data, V):
    cor = tot = 0
    for prompt, ans in data:
        lg = np.asarray(model.forward_ids(prompt).cpu().numpy()).reshape(-1, V)
        cor += int(int(np.argmax(lg[-1])) == ans[0]); tot += 1   # último token prevê o valor
    return cor / max(tot, 1)

def eval_copy(model, data, V):
    cor = tot = 0
    for prompt, ans in data:
        seq = prompt + ans
        lg = np.asarray(model.forward_ids(seq).cpu().numpy()).reshape(-1, V)
        ok = all(int(np.argmax(lg[len(prompt) + j - 1])) == ans[j] for j in range(len(ans)))
        cor += int(ok); tot += 1
    return cor / max(tot, 1)         # exact-match (cópia inteira certa)

print('[tasks] AR + selective-copy definidos')
"""))

cells.append(md("""## 5 — Variantes do MIXER (mesmo tamanho) + treino em batch

3 variantes isolando o mixer: **atenção-pura**, **Mamba-puro** (proper SSM),
**híbrido NSOS**. MoE/KAN/TTT OFF (a comparação é do mixer). Treino em batch via
`train_supervised_batch`, com `first_token_loss_scale=eos_loss_scale=1.0` e loss
impressa."""))
cells.append(code("""import os, sys, random
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
        c.attention_period = 1; c.attention_slot = 0
    elif variant == 'mamba':
        c.attention_period = 4; c.attention_slot = 4   # slot>=period => nenhuma atencao
    else:
        c.attention_period = 2; c.attention_slot = 1
    m = nsos.JambaModel(c, dev); m.to(dev)
    return m

def train_batched(model, lr, steps, batch, sampler, seed, label):
    tr = nsos.Trainer(model, lr); tr.warmup_steps = max(50, steps // 10)
    tr.total_training_steps = steps
    tr.first_token_loss_scale = 1.0; tr.eos_loss_scale = 1.0
    rng = random.Random(seed * 991 + 7)
    model.set_training_mode(True)
    win = max(1, steps // 6); run = 0.0
    for s in range(steps):
        ps, ans = [], []
        bs = sampler(rng)              # (field_len fixo p/ o batch) decidido dentro
        for _ in range(batch):
            p, a = bs()
            ps.append(p); ans.append(a)
        run += tr.train_supervised_batch(ps, ans)
        if (s + 1) % win == 0:
            print(f'    [{label}] step {s+1}/{steps}  loss~{run/win:.3f}'); run = 0.0
    model.set_training_mode(False)

VARIANTS = ['attn', 'mamba', 'hybrid']
SEEDS = [int(x) for x in os.environ.get('NSOS_GEN_SEEDS', '0,1').split(',')]
STEPS = int(os.environ.get('NSOS_GEN_STEPS', '1500'))
BATCH = int(os.environ.get('NSOS_GEN_BATCH', '16'))
print(f'[cfg] L={NUM_LAYERS} d={DMODEL} | variantes={VARIANTS} | seeds={SEEDS} | steps={STEPS} batch={BATCH} (= {STEPS*BATCH} exemplos/run)')
"""))

cells.append(md("""## 6 — Recall associativo: capacidade (treina n_kv=8, testa 8/16/32)

Baseline aleatório = 1/n_sym. **Controle positivo:** a atenção tem que acertar
~1.0 em n_kv=8. A pergunta científica: o híbrido NSOS acompanha a atenção quando a
capacidade sobe (n_kv=16, 32)? O Mamba-puro cai? (resultado clássico do Zoology)"""))
cells.append(code("""import time, numpy as np, random
N_SYM = int(os.environ.get('NSOS_AR_SYM', '32')); V = N_SYM + 2
base = 1.0 / N_SYM; KV_TEST = [8, 16, 32]
ar = {}; t0 = time.time()
for variant in VARIANTS:
    accs = {k: [] for k in KV_TEST}
    for seed in SEEDS:
        nsos.set_seed(seed)
        m = build_variant(variant, V)
        train_batched(m, 2e-3, STEPS, BATCH,
                      lambda r: (lambda: make_ar(r, 8, N_SYM)), seed, f'AR {variant} s{seed}')
        ev = random.Random(seed * 13 + 5)
        for k in KV_TEST:
            d = [make_ar(ev, k, N_SYM) for _ in range(300)]
            accs[k].append(eval_ar(m, d, V))
        print(f'  [{variant:6s} seed {seed}] ' + '  '.join(f'n_kv={k}:{accs[k][-1]:.3f}' for k in KV_TEST))
    ar[variant] = {k: (np.mean(accs[k]), np.std(accs[k])) for k in KV_TEST}
print('\\n' + '=' * 70)
print(f'RECALL ASSOCIATIVO (acuracia; baseline aleatorio={base:.3f})   ({time.time()-t0:.0f}s)')
print(f'{"variante":<10}' + ''.join(f'{("n_kv="+str(k)):>16}' for k in KV_TEST))
for v in VARIANTS:
    print(f'{v:<10}' + ''.join(f'{ar[v][k][0]:>10.3f}+/-{ar[v][k][1]:.2f}' for k in KV_TEST))
print('=' * 70)
"""))

cells.append(md("""## 7 — Selective copy: extrapolação de comprimento (treina ≤64, testa 64/128/256)

Métrica = exact-match. O Mamba/híbrido extrapola pra campos mais longos que o
treino? A atenção costuma cair (posicional). Upside estrutural do SSM."""))
cells.append(code("""import time, numpy as np, random
N_SYM_C = int(os.environ.get('NSOS_COPY_SYM', '20')); N_DATA = 4
LT = int(os.environ.get('NSOS_COPY_LEN', '64')); Vc = N_SYM_C + 3
base_c = (1.0 / N_SYM_C) ** N_DATA; TEST_LENS = [LT, 2 * LT, 4 * LT]
copy = {}; t0 = time.time()
for variant in VARIANTS:
    rows = {L: [] for L in TEST_LENS}
    for seed in SEEDS:
        nsos.set_seed(seed)
        m = build_variant(variant, Vc)
        # cada step do batch usa UM comprimento de campo (16..LT) -> batch homogeneo
        def sampler(r):
            L = r.randint(16, LT)
            return lambda: make_selcopy(r, L, N_DATA, N_SYM_C)
        train_batched(m, 2e-3, STEPS, BATCH, sampler, seed, f'COPY {variant} s{seed}')
        ev = random.Random(seed * 29 + 3)
        for L in TEST_LENS:
            d = [make_selcopy(ev, L, N_DATA, N_SYM_C) for _ in range(300)]
            rows[L].append(eval_copy(m, d, Vc))
        print(f'  [{variant:6s} seed {seed}] ' + '  '.join(f'L={L}:{rows[L][-1]:.3f}' for L in TEST_LENS))
    copy[variant] = {L: (np.mean(rows[L]), np.std(rows[L])) for L in TEST_LENS}
print('\\n' + '=' * 70)
print(f'SELECTIVE COPY exact-match (baseline~{base_c:.1e}; treino campo<=64)   ({time.time()-t0:.0f}s)')
print(f'{"variante":<10}' + ''.join(f'{("L="+str(L)+("(tr)" if L==LT else "(ex)")):>16}' for L in TEST_LENS))
for v in VARIANTS:
    print(f'{v:<10}' + ''.join(f'{copy[v][L][0]:>16.3f}' for L in TEST_LENS))
print('=' * 70)
"""))

cells.append(md("""## 8 — VEREDITO (régua pré-registrada, com a ATENÇÃO como gate)

Primeiro o gate de sanidade: **se a atenção não resolve o AR fácil, o teste está
insuficiente** (suba `NSOS_GEN_STEPS`/`NSOS_GEN_BATCH`) e NÃO se conclui nada de
arquitetura. Só com o controle verde a comparação vale."""))
cells.append(code("""def g(c): return 'OK' if c else 'FALHOU'
attn8 = ar['attn'][8][0]
GATE = attn8 >= 0.80
print('=' * 70)
print(f'GATE (controle positivo) — atencao resolve AR n_kv=8?  attn={attn8:.3f}  -> {g(GATE)}')
if not GATE:
    print('  >> HARNESS/REGIME INSUFICIENTE: a atencao (que deveria acertar ~1.0)')
    print('     nao aprendeu. NAO conclua nada de arquitetura. Acoes: suba')
    print('     NSOS_GEN_STEPS (ex. 4000) e/ou NSOS_GEN_BATCH (ex. 32), ou')
    print('     NSOS_GEN_LAYERS=4 NSOS_GEN_DMODEL=128 ja default. Rerode esta celula.')
else:
    print('\\n1) TABLE STAKES — todas resolvem AR n_kv=8?')
    for v in VARIANTS:
        print(f'   {v:<8} {ar[v][8][0]:.3f}  -> {g(ar[v][8][0] > 0.7)}')
    print('\\n2) DISCRIMINADOR — hibrido acompanha a atencao na CAPACIDADE (n_kv=32)?')
    a32, h32, m32 = ar['attn'][32][0], ar['hybrid'][32][0], ar['mamba'][32][0]
    print(f'   atencao={a32:.3f}  hibrido={h32:.3f}  mamba-puro={m32:.3f}')
    print(f'   hibrido >= 0.85*atencao: {g(h32 >= 0.85*a32)}   (classico: mamba-puro cai)')
    print('\\n3) UPSIDE — Mamba/hibrido EXTRAPOLAM (copy L=4x)?')
    for v in VARIANTS:
        print(f'   {v:<8} treino(L={LT})={copy[v][LT][0]:.3f}  extrap(L={4*LT})={copy[v][4*LT][0]:.3f}  -> {g(copy[v][4*LT][0] >= 0.5)}')
    print('\\nLEITURA: vale escalar se table-stakes OK + hibrido~atencao no recall de')
    print('capacidade + Mamba/hibrido extrapolam. As camadas de atencao do hibrido')
    print('DEVEM fechar o gap de recall do Mamba-puro -> evidencia central do NSOS.')
print('=' * 70)
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
