"""Generator for colab/train_genbench_mqar_copy.ipynb.

Gold-standard ARCHITECTURE evaluation suite (branch feature/nsos-gpu-phases12),
phase 1: associative recall (capacity curve) + selective-copy length extrapolation.

v3 of this notebook.  History:
  v1: ALL variants incl. the attention positive control stuck at the random
      baseline -> harness broken, not architecture.
  v2: cleaner single-query AR + batched training -> attention still ~baseline.
  v3 ROOT CAUSE (trainer.cpp:660-669): on GPU the model TRAINS in float (the
     reference path -- there is no packed CPU ternary kernel for device tensors),
     but eval with set_training_mode(False) runs the TERNARY forward.  A recall
     circuit (which needs precise Q.K key-matching) learned in float is wrecked by
     ternary quantization at eval -> accuracy collapses to baseline.  (Addition
     survived this in earlier work because a coarse 24-way map tolerates the
     weight noise; precise recall does not.)

v3 therefore:
  * a DIAGNOSTIC cell that overfits ONE example and prints the prediction in BOTH
    float (training-mode, reference path) and ternary (eval-mode) -> confirms the
    train-float / eval-ternary mismatch directly;
  * evaluates every task in BOTH modes; the architecture comparison (Zoology/Mamba
    style) uses the FLOAT numbers (apples-to-apples mixer comparison), and the
    ternary numbers are reported as the SEPARATE 1.58-bit quantization gap;
  * the verdict GATE is on the float attention control (must solve AR n_kv=8).

Tasks are pure synthetic token-id sequences: NO tokenizer, NO dataset download,
tiny models -> minutes.  Mixer comparison: attention-only / Mamba-only / hybrid.

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

cells.append(md("""# NSOS — Suíte de generalização (padrão-ouro) v3: recall + extrapolação

Fase 1 da avaliação de **arquitetura** (branch `%BRANCH%`).

**Causa-raiz encontrada (v1/v2 falhavam até na atenção):** na GPU o modelo **treina
em FLOAT** (reference path — não há kernel ternário empacotado p/ tensores de
device), mas o `eval` (`set_training_mode(False)`) roda o forward **TERNÁRIO
1.58-bit**. Um circuito de recall (que exige casar chaves com Q·K preciso) aprendido
em float é **destruído pela quantização na avaliação** → cai pro baseline. (A adição
sobrevivia porque um mapa grosseiro tolera o ruído; recall preciso não.)

**v3:** (1) célula de **diagnóstico** que decora 1 exemplo e mostra o pred em
**float** (modo-treino) vs **ternário** (modo-eval) — confirma a causa direto;
(2) avalia **nos dois modos**; a comparação de arquitetura (estilo Zoology/Mamba)
usa o **float** (mixer vs mixer, justo) e o **ternário** vira o *gap de quantização*
1.58-bit, reportado à parte; (3) o **gate** do veredito é a atenção em float."""))

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

cells.append(md("""## 4 — Tarefas + avaliação (em FLOAT e em TERNÁRIO)

**AR:** prompt `[BOS k1 v1 ... kN vN SEP q]`, resposta `[valor de q]` (loss = só o
valor → 100% sinal de recall). `n_kv` = capacidade. **Selective copy:** campo com K
dados entre BLANK + MARK → resposta = os K dados; treina curto, testa longo.
`eval_*` recebe `float_mode`: True = modo-treino (reference path, float);
False = modo-eval (forward ternário 1.58-bit)."""))
cells.append(code("""import random, numpy as np

def make_ar(rng, n_kv, n_sym):
    BOS, SEP = n_sym, n_sym + 1
    keys = rng.sample(range(n_sym), n_kv); vals = [rng.randrange(n_sym) for _ in range(n_kv)]
    kv = dict(zip(keys, vals)); q = rng.choice(keys)
    prompt = [BOS]
    for k, v in zip(keys, vals): prompt += [k, v]
    prompt += [SEP, q]
    return prompt, [kv[q]]

def make_selcopy(rng, field_len, n_data, n_sym):
    BLANK, MARK, BOS = n_sym, n_sym + 1, n_sym + 2
    pos = sorted(rng.sample(range(field_len), n_data)); data = [rng.randrange(n_sym) for _ in range(n_data)]
    field = [BLANK] * field_len
    for p, d in zip(pos, data): field[p] = d
    return [BOS] + field + [MARK], data

def eval_ar(model, data, V, float_mode):
    model.set_training_mode(float_mode)   # True=float(reference)  False=ternario(inferencia)
    cor = tot = 0
    for prompt, ans in data:
        lg = np.asarray(model.forward_ids(prompt).cpu().numpy()).reshape(-1, V)
        cor += int(int(np.argmax(lg[-1])) == ans[0]); tot += 1
    return cor / max(tot, 1)

def eval_copy(model, data, V, float_mode):
    model.set_training_mode(float_mode)
    cor = tot = 0
    for prompt, ans in data:
        seq = prompt + ans
        lg = np.asarray(model.forward_ids(seq).cpu().numpy()).reshape(-1, V)
        ok = all(int(np.argmax(lg[len(prompt) + j - 1])) == ans[j] for j in range(len(ans)))
        cor += int(ok); tot += 1
    return cor / max(tot, 1)

print('[tasks] AR + selective-copy (eval float/ternario) definidos')
"""))

cells.append(md("""## 5 — Variantes do MIXER + treino em batch

Atenção-pura / Mamba-puro (proper SSM) / híbrido NSOS, mesmo tamanho. MoE/KAN/TTT
OFF (compara o mixer). `train_supervised_batch` (batches homogêneos), loss impressa,
`first_token_loss_scale=eos_loss_scale=1.0`. (Na GPU o treino já é float; o ternário
só aparece no forward de inferência.)"""))
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
        c.attention_period = 4; c.attention_slot = 4
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
        bs = sampler(rng); ps, ans = [], []
        for _ in range(batch):
            p, a = bs(); ps.append(p); ans.append(a)
        run += tr.train_supervised_batch(ps, ans)
        if (s + 1) % win == 0:
            print(f'    [{label}] step {s+1}/{steps}  loss~{run/win:.3f}'); run = 0.0
    model.set_training_mode(False)

VARIANTS = ['attn', 'mamba', 'hybrid']
SEEDS = [int(x) for x in os.environ.get('NSOS_GEN_SEEDS', '0,1').split(',')]
STEPS = int(os.environ.get('NSOS_GEN_STEPS', '1500'))
BATCH = int(os.environ.get('NSOS_GEN_BATCH', '16'))
print(f'[cfg] L={NUM_LAYERS} d={DMODEL} | {VARIANTS} | seeds={SEEDS} | steps={STEPS} batch={BATCH} (={STEPS*BATCH} ex/run)')
"""))

cells.append(md("""## 5b — DIAGNÓSTICO: decorar 1 exemplo + float vs ternário

O teste mais decisivo. Treina a atenção pra **decorar UM** exemplo de AR (fácil) e
checa o pred em **float** (modo-treino) e **ternário** (modo-eval).
- loss não cai / float errado → **trainer/gradiente quebrado** (não é dados nem arq).
- float **certo** + ternário **errado** → confirma o **gap de quantização** (a causa).
- ambos certos → quantização não é o problema; era passo/dado."""))
cells.append(code("""import numpy as np, random
_NS = 16; _V = _NS + 2
nsos.set_seed(0)
mdbg = build_variant('attn', _V)
ex_p, ex_a = make_ar(random.Random(1), 4, _NS)   # 1 exemplo fixo, n_kv=4
tr = nsos.Trainer(mdbg, 3e-3); tr.total_training_steps = 400
tr.first_token_loss_scale = 1.0; tr.eos_loss_scale = 1.0
mdbg.set_training_mode(True)
for s in range(400):
    L = tr.train_supervised(ex_p, ex_a)
    if s % 50 == 0 or s == 399: print(f'  overfit step {s:3d}: loss={L:.4f}')

def _pred(float_mode):
    mdbg.set_training_mode(float_mode)
    lg = np.asarray(mdbg.forward_ids(ex_p).cpu().numpy()).reshape(-1, _V)
    return int(np.argmax(lg[-1]))

pf, pt = _pred(True), _pred(False)
print('=' * 60)
print(f'  alvo={ex_a[0]}  | pred FLOAT(treino)={pf}  pred TERNARIO(eval)={pt}')
if pf == ex_a[0] and pt != ex_a[0]:
    print('  >> CONFIRMADO: aprende em FLOAT, quebra no TERNARIO (gap de quantizacao).')
    print('     A comparacao de arquitetura usa o numero FLOAT (mixer vs mixer).')
elif pf != ex_a[0]:
    print('  >> trainer/gradiente/eval quebrado (nem decora 1 exemplo em float).')
else:
    print('  >> quantizacao OK aqui; falha anterior era passo/dado -> suba NSOS_GEN_STEPS.')
print('=' * 60)
"""))

cells.append(md("""## 6 — Recall associativo: capacidade (treina n_kv=8; testa 8/16/32; float & ternário)

Baseline = 1/n_sym. **Gate:** atenção em FLOAT tem que acertar ~1.0 em n_kv=8."""))
cells.append(code("""import time, numpy as np, random
N_SYM = int(os.environ.get('NSOS_AR_SYM', '32')); V = N_SYM + 2
base = 1.0 / N_SYM; KV_TEST = [8, 16, 32]
ar = {}; t0 = time.time()
for variant in VARIANTS:
    fp = {k: [] for k in KV_TEST}; tq = {k: [] for k in KV_TEST}
    for seed in SEEDS:
        nsos.set_seed(seed); m = build_variant(variant, V)
        train_batched(m, 2e-3, STEPS, BATCH, lambda r: (lambda: make_ar(r, 8, N_SYM)), seed, f'AR {variant} s{seed}')
        ev = random.Random(seed * 13 + 5)
        for k in KV_TEST:
            d = [make_ar(ev, k, N_SYM) for _ in range(300)]
            fp[k].append(eval_ar(m, d, V, True)); tq[k].append(eval_ar(m, d, V, False))
        print(f'  [{variant:6s} s{seed}] float ' + ' '.join(f'{k}:{fp[k][-1]:.2f}' for k in KV_TEST) +
              '  | tern ' + ' '.join(f'{k}:{tq[k][-1]:.2f}' for k in KV_TEST))
    ar[variant] = {'float': {k: float(np.mean(fp[k])) for k in KV_TEST},
                   'tern':  {k: float(np.mean(tq[k])) for k in KV_TEST}}
print('\\n' + '=' * 72)
print(f'RECALL ASSOCIATIVO (baseline={base:.3f})   ({time.time()-t0:.0f}s)')
for tag in ('float', 'tern'):
    print(f'-- {tag.upper()} --   ' + ''.join(f'{("n_kv="+str(k)):>12}' for k in KV_TEST))
    for v in VARIANTS:
        print(f'  {v:<8}' + ''.join(f'{ar[v][tag][k]:>12.3f}' for k in KV_TEST))
print('=' * 72)
"""))

cells.append(md("""## 7 — Selective copy: extrapolação (treina ≤64; testa 64/128/256; float & ternário)"""))
cells.append(code("""import time, numpy as np, random
N_SYM_C = int(os.environ.get('NSOS_COPY_SYM', '20')); N_DATA = 4
LT = int(os.environ.get('NSOS_COPY_LEN', '64')); Vc = N_SYM_C + 3
base_c = (1.0 / N_SYM_C) ** N_DATA; TEST_LENS = [LT, 2 * LT, 4 * LT]
copy = {}; t0 = time.time()
for variant in VARIANTS:
    fp = {L: [] for L in TEST_LENS}; tq = {L: [] for L in TEST_LENS}
    for seed in SEEDS:
        nsos.set_seed(seed); m = build_variant(variant, Vc)
        def sampler(r):
            L = r.randint(16, LT)
            return lambda: make_selcopy(r, L, N_DATA, N_SYM_C)
        train_batched(m, 2e-3, STEPS, BATCH, sampler, seed, f'COPY {variant} s{seed}')
        ev = random.Random(seed * 29 + 3)
        for L in TEST_LENS:
            d = [make_selcopy(ev, L, N_DATA, N_SYM_C) for _ in range(300)]
            fp[L].append(eval_copy(m, d, Vc, True)); tq[L].append(eval_copy(m, d, Vc, False))
        print(f'  [{variant:6s} s{seed}] float ' + ' '.join(f'{L}:{fp[L][-1]:.2f}' for L in TEST_LENS) +
              '  | tern ' + ' '.join(f'{L}:{tq[L][-1]:.2f}' for L in TEST_LENS))
    copy[variant] = {'float': {L: float(np.mean(fp[L])) for L in TEST_LENS},
                     'tern':  {L: float(np.mean(tq[L])) for L in TEST_LENS}}
print('\\n' + '=' * 72)
print(f'SELECTIVE COPY exact-match (baseline~{base_c:.0e}; treino<=64)   ({time.time()-t0:.0f}s)')
for tag in ('float', 'tern'):
    print(f'-- {tag.upper()} --   ' + ''.join(f'{("L="+str(L)):>12}' for L in TEST_LENS))
    for v in VARIANTS:
        print(f'  {v:<8}' + ''.join(f'{copy[v][tag][L]:>12.3f}' for L in TEST_LENS))
print('=' * 72)
"""))

cells.append(md("""## 8 — VEREDITO (gate = atenção em FLOAT)"""))
cells.append(code("""def g(c): return 'OK' if c else 'FALHOU'
a8 = ar['attn']['float'][8]
GATE = a8 >= 0.80
print('=' * 72)
print(f'GATE — atencao (FLOAT) resolve AR n_kv=8?  attn={a8:.3f}  -> {g(GATE)}')
if not GATE:
    print('  >> HARNESS INSUFICIENTE: suba NSOS_GEN_STEPS (4000) e/ou NSOS_GEN_BATCH (32).')
else:
    a32, h32, m32 = ar['attn']['float'][32], ar['hybrid']['float'][32], ar['mamba']['float'][32]
    print('\\n1) TABLE STAKES (float, AR n_kv=8):')
    for v in VARIANTS: print(f'   {v:<8} {ar[v]["float"][8]:.3f} -> {g(ar[v]["float"][8] > 0.7)}')
    print(f'\\n2) DISCRIMINADOR — hibrido ~ atencao na capacidade (n_kv=32, float)?')
    print(f'   attn={a32:.3f} hibrido={h32:.3f} mamba={m32:.3f} | hibrido>=0.85*attn: {g(h32>=0.85*a32)}')
    print(f'\\n3) UPSIDE — extrapolacao de comprimento (copy L=4x, float):')
    for v in VARIANTS:
        print(f'   {v:<8} L={LT}:{copy[v]["float"][LT]:.3f} -> L={4*LT}:{copy[v]["float"][4*LT]:.3f}  {g(copy[v]["float"][4*LT]>=0.5)}')
    print(f'\\n4) GAP DE QUANTIZACAO 1.58-bit (float -> ternario, AR n_kv=8):')
    for v in VARIANTS:
        print(f'   {v:<8} float={ar[v]["float"][8]:.3f}  ternario={ar[v]["tern"][8]:.3f}  (queda={ar[v]["float"][8]-ar[v]["tern"][8]:+.3f})')
    print('\\nLEITURA: arquitetura (mixer) se julga no FLOAT; o ternario mede o custo')
    print('da quantizacao 1.58-bit (eixo separado do NSOS). Vale escalar se: float')
    print('table-stakes OK + hibrido~atencao no recall + Mamba/hibrido extrapolam.')
print('=' * 72)
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
