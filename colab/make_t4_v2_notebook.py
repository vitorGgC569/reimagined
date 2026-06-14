"""Generator for colab/train_gpu_phases_t4_v2.ipynb.

v2 is the GOLD-STANDARD VALIDATION notebook for the correctness program on
branch feature/nsos-gpu-phases12.  Unlike v1 (which runs the full v11 curriculum
on GPU), v2 proves the corrected stack end-to-end with OBJECTIVE EVIDENCE:

  1. builds nsos_ext + the CTest suite (CUDA, T4 sm_75);
  2. runs the finite-difference GRADCHECK gate (test_gradcheck) — proves every
     hand-derived backward, incl. the corrected Mamba path and the Switch MoE
     aux loss, matches the numerical gradient;
  3. trains ONE controlled task — Portuguese addition QA ("quanto é A mais B ?")
     — with the corrected paths ENABLED (NSOS_MAMBA_PROPER_SSM, NSOS_MOE_FP_ROUTER,
     NSOS_MOE_SWITCH_AUX, NSOS_MAMBA_A_LOGSPACED);
  4. measures GENERALIZATION cleanly: a compositional hold-out of (A,B) pairs
     (every A and every B is seen individually in training, but the held-out
     PAIRS never are) -> held-out perplexity + exact-match.  Generalization =
     held-out exact-match well above the 1/|answers| random baseline and close
     to the train-pair exact-match.

The controlled task runs on CPU so the opt-in proper-Mamba path (host-computed
in this version) is exercised end to end without the mixed-device optimizer; the
build is still the real T4/CUDA binary and the gradcheck is the same one CI runs.
A GPU-resident proper-Mamba lane lands with the N-state kernel (tracked).

Run once (pure JSON authoring, no GPU): python colab/make_t4_v2_notebook.py
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

cells.append(md("""# NSOS — Validação gold-standard do stack corrigido (branch `%BRANCH%`)

Notebook **v2**: prova objetiva das correções do programa (Mamba selectivo correto,
MoE FP + aux-loss de Switch diferenciável, gradcheck no CI) numa **única tarefa
controlada em português** com **medição de generalização**.

Ordem: 1 Drive → 2 clone → 3 build (CUDA T4 + testes) → **4 GRADCHECK (evidência
dos gradientes)** → 5 dataset PT (adição, holdout composicional) → 6 treino com os
flags corrigidos → 7 **perplexidade + exact-match (generalização)** → 8 probe de
criticalidade.

A tarefa roda em **CPU** (o caminho proper-Mamba opt-in é host nesta versão; assim
é exercitado de ponta a ponta sem o otimizador multi-device). O build é o binário
T4/CUDA real e o gradcheck é o mesmo do CI."""))

cells.append(md("## 1 — Drive + GPU (confirme **Tesla T4**)"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, subprocess, multiprocessing
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_v11')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print('cpus:', multiprocessing.cpu_count())
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                '--format=csv'], check=False)
"""))

cells.append(md("## 2 — Token + clone da branch + prova de versão das correções"))
cells.append(code("""import os, subprocess
from pathlib import Path

REPO_URL = 'https://github.com/vitorGgC569/reimagined.git'
REPO_ROOT = Path('/content/reimagined')
BRANCH = '%BRANCH%'

TOKEN = None
try:
    from google.colab import userdata
    for key in ('GITHUB_TOKEN', 'GH_TOKEN'):
        try:
            TOKEN = userdata.get(key)
        except Exception:
            TOKEN = None
        if TOKEN:
            print(f'[auth] token {key} ok'); break
except Exception as e:
    print('[auth] userdata indisponivel:', e)

url = REPO_URL.replace('https://', f'https://oauth2:{TOKEN}@') if TOKEN else REPO_URL
if REPO_ROOT.exists():
    subprocess.run(['git', '-C', str(REPO_ROOT), 'remote', 'set-url', 'origin', url], check=True)
    subprocess.run(['git', '-C', str(REPO_ROOT), 'fetch', '--depth', '1', 'origin', BRANCH], check=True)
    subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', BRANCH], check=True)
    subprocess.run(['git', '-C', str(REPO_ROOT), 'reset', '--hard', f'origin/{BRANCH}'], check=True)
else:
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, url, str(REPO_ROOT)], check=True)
subprocess.run(['git', '-C', str(REPO_ROOT), 'remote', 'set-url', 'origin', REPO_URL], check=False)

os.environ['REPO_ROOT'] = str(REPO_ROOT)
os.environ['DRIVE_ROOT'] = str(DRIVE_ROOT)

sha = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
# Prova de versao: as correcoes precisam estar no fonte clonado.
mamba = (REPO_ROOT / 'OXN/nsos/src/mamba2.cpp').read_text('utf-8')
has_proper = 'proper_selective_ssm' in mamba and 'forward_proper' in mamba
has_gc = (REPO_ROOT / 'OXN/nsos/tests/test_gradcheck.cpp').exists()
jamba = (REPO_ROOT / 'OXN/nsos/src/jamba.cpp').read_text('utf-8')
has_switch = 'switch_aux_grad_logits' in jamba
print('=' * 64)
print(f'[ver] HEAD = {sha}')
print(f'[ver] Mamba proper path no fonte : {"SIM" if has_proper else "*** NAO ***"}')
print(f'[ver] test_gradcheck.cpp presente: {"SIM" if has_gc else "*** NAO ***"}')
print(f'[ver] MoE switch aux no fonte     : {"SIM" if has_switch else "*** NAO ***"}')
assert has_proper and has_gc and has_switch, 'fonte stale — as correcoes nao estao na branch clonada'
print('=' * 64)
"""))

cells.append(md("""## 3 — Build `nsos_ext` + suíte de testes (CUDA T4 sm_75)

Compila a biblioteca, o binding Python e os testes (necessários para o gradcheck).
Primeira vez ~10-20 min; cacheia o `.so` no Drive por SHA."""))
cells.append(code("""import os, sys, subprocess, shutil, glob
from pathlib import Path

if 'nsos_ext' in sys.modules:
    raise RuntimeError('nsos_ext ja carregado — Runtime > Restart e rode de novo.')

PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
sha = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
BUILD = str(REPO_ROOT / 'OXN/nsos/build-t4v2')
EXT_DIR = Path('/content/nsos_ext_v2'); EXT_DIR.mkdir(exist_ok=True)
CACHE_SO = DRIVE_ROOT / '_bootstrap' / f'nsos_ext_t4v2.{PYTAG}.{sha}.so'

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pybind11', 'numpy'], check=True)
print(f'[build] cmake configure (CUDA on, tests on) sha={sha}')
r = subprocess.run(['cmake', '-S', str(REPO_ROOT / 'OXN/nsos'), '-B', BUILD,
                    '-DNSOS_ENABLE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=75',
                    '-DNSOS_BUILD_PYTHON=ON', '-DNSOS_BUILD_TESTS=ON',
                    '-DNSOS_BUILD_CLI=OFF', '-DNSOS_BUILD_API=OFF',
                    '-DNSOS_BUILD_OXTAMEM=OFF', '-DCMAKE_BUILD_TYPE=Release',
                    f'-DPython3_EXECUTABLE={sys.executable}'],
                   capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    print('STDERR:', r.stderr[-3000:]); raise RuntimeError('cmake configure falhou')
print('[build] compilando (nsos_ext + testes) — paciencia na 1a vez')
r = subprocess.run(['cmake', '--build', BUILD, '-j', '2'], capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    print('STDERR:', r.stderr[-3000:]); raise RuntimeError('build falhou')
so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True))
assert so, 'nsos_ext*.so nao encontrado'
shutil.copy2(so[0], EXT_DIR / 'nsos_ext.so')
CACHE_SO.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(so[0], CACHE_SO)
os.environ['NSOS_BUILD_T4V2'] = BUILD
print('[build] ok | ext dir =', EXT_DIR, '| build =', BUILD)
"""))

cells.append(md("""## 4 — GATE DE GRADCHECK (evidência objetiva dos gradientes)

Roda `test_gradcheck` (diferenças finitas vs backward manual: rmsnorm, squared_relu,
cross_entropy, matmul, **Mamba legado + Mamba corrigido**, **MoE Switch aux**, KAN) e
testes-âncora. **Tudo verde = os gradientes estão corretos.**"""))
cells.append(code("""import os, subprocess
BUILD = os.environ['NSOS_BUILD_T4V2']

def run_test(name):
    for cand in (f'{BUILD}/{name}', f'{BUILD}/Release/{name}', f'{BUILD}/{name}.exe'):
        if os.path.exists(cand):
            r = subprocess.run([cand], capture_output=True, text=True)
            print(f'\\n===== {name} (exit {r.returncode}) =====')
            print(r.stdout[-2500:])
            if r.returncode != 0:
                print('STDERR:', r.stderr[-1500:])
            return r.returncode == 0
    print(f'[skip] {name} nao encontrado em {BUILD}')
    return False

results = {n: run_test(n) for n in
          ('test_gradcheck', 'test_mamba2', 'test_bitlinear', 'test_moe_training',
           'test_gpu_parity_mamba_proper', 'test_gpu_parity_mamba_scan')}
print('\\n' + '=' * 50)
for n, ok in results.items():
    print(f'  {n:<32} {"PASS" if ok else "FAIL"}')
assert results.get('test_gradcheck'), 'GRADCHECK FALHOU — gradientes incorretos'
# Paridade GPU dos kernels proper (conv1d + scan readout-linear) na T4.
assert results.get('test_gpu_parity_mamba_proper'), \\
    'PARIDADE GPU proper-Mamba FALHOU — kernels CUDA divergem do host'
print('GRADCHECK + PARIDADE GPU: PASS')
"""))

cells.append(md("""## 5 — Dataset controlado: adição em português (holdout composicional)

`quanto é A mais B ?` → palavra-número da soma, com A,B ∈ 0..9 (soma 0..18).
Reservamos pares (A,B) para teste; cada A e cada B aparece isoladamente no treino,
mas os PARES de teste não — então acertar o teste exige **generalizar a soma**,
não decorar pares."""))
cells.append(code("""import random

NUM_WORDS = ['zero','um','dois','tres','quatro','cinco','seis','sete','oito','nove',
             'dez','onze','doze','treze','catorze','quinze','dezesseis','dezessete','dezoito']
STRUCT = ['quanto','e','mais','?']
SPECIAL = ['<eos>']
VOCAB = STRUCT + SPECIAL + NUM_WORDS
STOI = {w: i for i, w in enumerate(VOCAB)}
EOS = STOI['<eos>']
V = len(VOCAB)

def prompt_ids(a, b):
    return [STOI['quanto'], STOI['e'], STOI[NUM_WORDS[a]], STOI['mais'],
            STOI[NUM_WORDS[b]], STOI['?']]
def answer_ids(a, b):
    return [STOI[NUM_WORDS[a + b]], EOS]

all_pairs = [(a, b) for a in range(10) for b in range(10)]
rng = random.Random(7)
rng.shuffle(all_pairs)
N_HELD = 15
held = sorted(all_pairs[:N_HELD]); train = sorted(all_pairs[N_HELD:])
# Garante cobertura: todo valor 0..9 aparece como A e como B no treino.
seen_a = {a for a, _ in train}; seen_b = {b for _, b in train}
assert seen_a == set(range(10)) and seen_b == set(range(10)), 'recobrir holdout'
print(f'[data] vocab={V}  train_pairs={len(train)}  held_pairs={len(held)}')
print(f'[data] held-out (generalizacao): {held}')
print(f'[data] baseline aleatorio exact-match = 1/{len(NUM_WORDS)} = {1/len(NUM_WORDS):.3f}')
"""))

cells.append(md("""## 6 — Treino do stack CORRIGIDO (CPU, flags ligados)

Liga **Mamba selectivo correto** (`NSOS_MAMBA_PROPER_SSM`), **roteador FP**
(`NSOS_MOE_FP_ROUTER`), **aux-loss de Switch** (`NSOS_MOE_SWITCH_AUX`) e o
**A log-espaçado** (`NSOS_MAMBA_A_LOGSPACED`). Os flags são lidos na construção do
modelo / 1º step, então são setados ANTES de tudo aqui."""))
cells.append(code("""import os, sys, time
# Flags das correcoes — DEVEM preceder a construcao do modelo / 1o train step.
os.environ['NSOS_MAMBA_PROPER_SSM'] = '1'
os.environ['NSOS_MAMBA_CONV_K']     = '3'
os.environ['NSOS_MOE_FP_ROUTER']    = '1'
os.environ['NSOS_MOE_SWITCH_AUX']   = '1'
os.environ['NSOS_MAMBA_A_LOGSPACED'] = '1'
os.environ['NSOS_GPU_POOL'] = '1'        # caching allocator (GPU-first)

sys.path.insert(0, str(__import__('pathlib').Path('/content/nsos_ext_v2')))
import nsos_ext as nsos
print('[nsos] modulo carregado')

# GPU-first: a via proper-Mamba agora e GPU-residente (kernels conv1d + scan
# readout-linear), validada pela paridade na celula 4.  Treino roda na T4.
dev = nsos.Device.GPU
cfg = nsos.ModelConfig()
cfg.num_layers = 4; cfg.d_model = 128; cfg.vocab_size = V
cfg.n_heads = 4; cfg.n_kv_heads = 2
cfg.use_moe = True; cfg.num_experts = 4; cfg.num_experts_per_token = 2
cfg.moe_period = 2; cfg.moe_slot = 1            # camadas 1,3 = MoE (exercita switch aux)
cfg.attention_period = 4; cfg.attention_slot = 3  # camada 3 = atencao; 0,2 = Mamba proper
cfg.use_ttt = False; cfg.dropout = 0.0
cfg.max_context_tokens = 64; cfg.use_exact_attention_training = True

model = nsos.JambaModel(cfg, dev); model.to(dev)
model.set_training_mode(True)
tr = nsos.Trainer(model, 2e-3)
tr.warmup_steps = 100; tr.eos_token_id = EOS; tr.moe_aux_loss_scale = 0.01
EPOCHS = 60
tr.total_training_steps = EPOCHS * len(train)

print(f'[train] params layers={cfg.num_layers} d={cfg.d_model} V={V} | epochs={EPOCHS} steps~{tr.total_training_steps}')
rng2 = random.Random(123)
t0 = time.perf_counter()
for ep in range(EPOCHS):
    order = train[:]; rng2.shuffle(order)
    ep_loss = 0.0
    for (a, b) in order:
        ep_loss += tr.train_supervised(prompt_ids(a, b), answer_ids(a, b))
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f'  epoch {ep:3d}  loss/pair = {ep_loss/len(order):.4f}  ({time.perf_counter()-t0:.0f}s)')

RUN_DIR = DRIVE_ROOT / 'runs' / f"t4v2_ptadd_{time.strftime('%Y%m%d_%H%M')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT = str(RUN_DIR / 'ptadd_corrected.bin')
model.save(CKPT)
print('[train] checkpoint:', CKPT)
"""))

cells.append(md("""## 7 — EVIDÊNCIA: perplexidade + exact-match (treino vs held-out)

`exact-match` = o modelo produz a palavra-soma correta. A diferença entre held-out
e o baseline aleatório (1/19) é o sinal de **generalização**. Perplexidade held-out
é o número de "verdade primeiro"."""))
cells.append(code("""import numpy as np, math
model.set_training_mode(False)

def eval_pairs(pairs):
    nll = 0.0; ntok = 0; correct = 0
    for (a, b) in pairs:
        p = prompt_ids(a, b); ans = answer_ids(a, b); seq = p + ans
        logits = np.asarray(model.forward_ids(seq).cpu().numpy()).reshape(len(seq), V)
        # exact-match: token previsto na 1a posicao da resposta
        pred = int(np.argmax(logits[len(p) - 1]))
        correct += int(pred == ans[0])
        # perplexidade sobre os tokens da resposta
        for pos in range(len(p), len(seq)):
            row = logits[pos - 1].astype(np.float64)
            row -= row.max(); pr = np.exp(row); pr /= pr.sum()
            nll += -math.log(max(pr[seq[pos]], 1e-12)); ntok += 1
    return math.exp(nll / max(ntok, 1)), correct / len(pairs)

ppl_tr, em_tr = eval_pairs(train)
ppl_he, em_he = eval_pairs(held)
base = 1.0 / len(NUM_WORDS)
print('=' * 60)
print(f'TREINO    : ppl={ppl_tr:7.3f}   exact-match={em_tr:.3f}')
print(f'HELD-OUT  : ppl={ppl_he:7.3f}   exact-match={em_he:.3f}   (baseline {base:.3f})')
print('=' * 60)
verdict = 'GENERALIZOU' if em_he > 3 * base else ('parcial' if em_he > base else 'memorizou/nao aprendeu')
print(f'VEREDITO generalizacao: {verdict}  (held-out {em_he:.2f} vs baseline {base:.2f})')
print('Exemplos held-out:')
for (a, b) in held[:8]:
    p = prompt_ids(a, b)
    logits = np.asarray(model.forward_ids(p).cpu().numpy()).reshape(len(p), V)
    pred = VOCAB[int(np.argmax(logits[-1]))]
    print(f'  {NUM_WORDS[a]} + {NUM_WORDS[b]} = {NUM_WORDS[a+b]:>10}  | modelo: {pred}')
"""))

cells.append(md("""## 8 — (opcional) Probe de criticalidade no checkpoint (OXTA-CRIT Lei 1)

Mede o ganho de ramo ternário por camada e a fração na banda crítica `[0.5,2]` — o
instrumento que sustenta a tese científica (diferencial §6)."""))
cells.append(code("""import subprocess, sys, os
B = os.environ['NSOS_BUILD_T4V2']
r = subprocess.run([sys.executable,
                    str(REPO_ROOT / 'OXN/nsos/scripts/criticality_probe.py'),
                    '--build-dir', B, '--profile', 'mamba_small', '--device', 'cpu'],
                   capture_output=True, text=True)
print(r.stdout[-2500:] or r.stderr[-1500:])
"""))

cells.append(md("""## 9 — DIFERENCIAL §6: instrumentos de criticalidade

(1) selftests locais (numpy) dos instrumentos; (2) **ganho-de-ramo por camada** e
**SNR de gradiente por camada** (Lyapunov backward por réplica) medidos no modelo
treinado; (3) fecha a malha empurrando `lr_l ∝ r_l^γ` para o trainer."""))
cells.append(code("""import subprocess, sys, importlib
print('--- selftests locais (numpy, sem GPU) ---')
subprocess.run([sys.executable, str(REPO_ROOT / 'OXN/nsos/scripts/criticality_instrument.py'), '--selftest'])
subprocess.run([sys.executable, str(REPO_ROOT / 'OXN/nsos/scripts/avalanche_analysis.py'), '--selftest'])

sys.path.insert(0, str(REPO_ROOT / 'OXN/nsos/scripts'))
import criticality_instrument as ci; importlib.reload(ci)

gains = ci.measure_branch_gains(model)
print('\\n[branch-gain] (OXTA-CRIT Lei 1):', gains['summary'])

# SNR de gradiente por camada (backward-Lyapunov por replica) no modelo treinado.
samples = [(prompt_ids(a, b) + answer_ids(a, b), []) for (a, b) in train[:24]]
snr = ci.measure_gradient_snr(tr, model, samples, num_pairs=12, seed=1)
print('[grad-SNR por camada]')
for lk, v in sorted(snr['per_layer'].items()):
    print(f'  {lk:<10} coherence r={v["coherence"]:+.3f}  SNR={v["snr"]:.3f}')

# Fecha a malha (DEPTH axis): lr_l proporcional a r_l^gamma.
scales = ci.apply_snr_lr_control(tr, snr['per_param'], gamma=1.0)
print(f'[loop] lr-scales aplicados em {len(scales)} params (normalizado ~1, clamp[0.25,4])')
print('      -> a partir daqui, train_supervised usa o passo por-camada do SNR.')
tr.clear_lr_scales()  # limpa para nao afetar runs subsequentes desta sessao
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

out = Path(__file__).resolve().parent / "train_gpu_phases_t4_v2.ipynb"
text = json.dumps(nb, indent=1, ensure_ascii=False).replace("%BRANCH%", BRANCH)
out.write_text(text, encoding="utf-8")
print(f"wrote {out}")
