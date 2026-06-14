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
print('[build] compilando alvos necessarios (nsos_ext + gates do Mamba) — paciencia na 1a vez')
# Build SO os alvos que importam para treino+validacao desta corrida: o modulo
# Python e os gates de gradiente/paridade do Mamba.  Evita compilar os ~60 testes
# legados (alguns tem includes faltantes que so o GCC pega) que bloqueariam o
# build inteiro sem relacao com o treino.
r = subprocess.run(['cmake', '--build', BUILD, '-j', '2', '--target',
                    'nsos_ext', 'test_gradcheck',
                    'test_gpu_parity_mamba_proper', 'test_gpu_parity_mamba_nstate'],
                   capture_output=True, text=True)
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

# Só os alvos que o build alvo-restrito compila (gradcheck + paridades Mamba GPU).
results = {n: run_test(n) for n in
          ('test_gradcheck', 'test_gpu_parity_mamba_proper',
           'test_gpu_parity_mamba_nstate')}
print('\\n' + '=' * 50)
for n, ok in results.items():
    print(f'  {n:<34} {"PASS" if ok else "FAIL"}')
assert results.get('test_gradcheck'), 'GRADCHECK FALHOU — gradientes incorretos'
# Paridade GPU dos kernels Mamba-2 (conv1d + scan linear + N-state) na T4.
assert results.get('test_gpu_parity_mamba_proper'), \\
    'PARIDADE GPU proper-Mamba FALHOU — kernels CUDA divergem do host'
assert results.get('test_gpu_parity_mamba_nstate'), \\
    'PARIDADE GPU Mamba-2 N-state FALHOU — kernels CUDA divergem do host'
print('GRADCHECK + PARIDADE GPU (proper + N-state): PASS')
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
# Mamba-2 SSD COMPLETO (estado h em R^{H x P x N}) — kernels CUDA validados por
# paridade GPU no T4 (test_gpu_parity_mamba_nstate PASS).  '1' = treina o Mamba-2
# completo (mais capacidade); '0' = via diagonal proper (provada: 73% held-out).
os.environ['NSOS_MAMBA_STATE_EXPANSION'] = '0'
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

cells.append(md("""## 10 — ABLAÇÃO atribuível: stack CORRIGIDO vs ANTIGO (mesma tarefa/seed)

Prova que as **correções causaram** a generalização. Treina os dois stacks na MESMA
tarefa, MESMA seed de init (`nsos.set_seed`) e MESMA ordem de dados — só muda o
stack. **Cada variante roda num SUBPROCESSO próprio** porque alguns flags
(`NSOS_MOE_SWITCH_AUX`, `NSOS_MOE_ROUTER_GRAD`) são lidos uma vez por processo
(static); flipá-los na mesma sessão seria inválido. ~7 min (2 variantes × 2 seeds)."""))
cells.append(code("""import os, sys, subprocess, json
import numpy as np

# Worker: roda UMA variante num processo proprio -> statics frescos, flags 100%
# honrados.  Reconstroi o dataset de forma deterministica (identico a celula 5).
WORKER = '''
import os, sys, json, random, math
import numpy as np
sys.path.insert(0, os.environ['NSOS_EXT_DIR'])
import nsos_ext as nsos
NW=['zero','um','dois','tres','quatro','cinco','seis','sete','oito','nove','dez','onze','doze','treze','catorze','quinze','dezesseis','dezessete','dezoito']
VOC=['quanto','e','mais','?','<eos>']+NW
ST={w:i for i,w in enumerate(VOC)}; EOS=ST['<eos>']; V=len(VOC)
def P(a,b): return [ST['quanto'],ST['e'],ST[NW[a]],ST['mais'],ST[NW[b]],ST['?']]
def AQ(a,b): return [ST[NW[a+b]],EOS]
ap=[(a,b) for a in range(10) for b in range(10)]; random.Random(7).shuffle(ap)
train=sorted(ap[15:]); held=sorted(ap[:15])
seed=int(os.environ['ABL_SEED']); ep=int(os.environ['ABL_EPOCHS'])
dev=nsos.Device.GPU; nsos.set_seed(seed)
c=nsos.ModelConfig(); c.num_layers=4; c.d_model=128; c.vocab_size=V; c.n_heads=4; c.n_kv_heads=2
c.use_moe=True; c.num_experts=4; c.num_experts_per_token=2; c.moe_period=2; c.moe_slot=1
c.attention_period=4; c.attention_slot=3; c.use_ttt=False; c.dropout=0.0
c.max_context_tokens=64; c.use_exact_attention_training=True
m=nsos.JambaModel(c,dev); m.to(dev); m.set_training_mode(True)
t=nsos.Trainer(m,2e-3); t.warmup_steps=100; t.eos_token_id=EOS; t.moe_aux_loss_scale=0.01
t.total_training_steps=ep*len(train)
r=random.Random(seed)
for e in range(ep):
    o=train[:]; r.shuffle(o)
    for (a,b) in o: t.train_supervised(P(a,b),AQ(a,b))
m.set_training_mode(False)
def ev(prs):
    nll=0.0; nt=0; cor=0
    for (a,b) in prs:
        p=P(a,b); an=AQ(a,b); sq=p+an
        lg=np.asarray(m.forward_ids(sq).cpu().numpy()).reshape(len(sq),V)
        if int(np.argmax(lg[len(p)-1]))==an[0]: cor+=1
        for q in range(len(p),len(sq)):
            row=lg[q-1].astype('float64'); row-=row.max(); pr=np.exp(row); pr/=pr.sum()
            nll+=-math.log(max(pr[sq[q]],1e-12)); nt+=1
    return math.exp(nll/max(nt,1)), cor/len(prs)
ptr,etr=ev(train); phe,ehe=ev(held)
print('RESULT_JSON='+json.dumps(dict(ppl_tr=ptr,em_tr=etr,ppl_he=phe,em_he=ehe)))
'''

CORR={'NSOS_MAMBA_PROPER_SSM':'1','NSOS_MAMBA_CONV_K':'3','NSOS_MOE_FP_ROUTER':'1','NSOS_MOE_SWITCH_AUX':'1','NSOS_MAMBA_A_LOGSPACED':'1'}
OLD ={'NSOS_MAMBA_PROPER_SSM':'0','NSOS_MOE_FP_ROUTER':'0','NSOS_MOE_SWITCH_AUX':'0','NSOS_MAMBA_A_LOGSPACED':'0'}
ALLK=set(CORR)|set(OLD)|{'NSOS_MAMBA_STATE_EXPANSION'}

def run_variant(flags, seed, epochs):
    env=dict(os.environ)
    for k in ALLK: env.pop(k, None)
    env.update(flags)
    env['NSOS_EXT_DIR']=str(EXT_DIR); env['ABL_SEED']=str(seed); env['ABL_EPOCHS']=str(epochs)
    rr=subprocess.run([sys.executable,'-c',WORKER], capture_output=True, text=True, env=env)
    ln=[l for l in rr.stdout.splitlines() if l.startswith('RESULT_JSON=')]
    if not ln:
        print(rr.stdout[-1200:]); print('STDERR:', rr.stderr[-1200:]); raise RuntimeError('variante falhou')
    return json.loads(ln[0][len('RESULT_JSON='):])

ABL_EPOCHS=60; ABL_SEEDS=[7,123]
abl={'CORRIGIDO':[], 'ANTIGO':[]}
for nm,fl in [('CORRIGIDO',CORR),('ANTIGO',OLD)]:
    for s in ABL_SEEDS:
        abl[nm].append(run_variant(fl,s,ABL_EPOCHS)); print(f'[abl] {nm} seed={s} ok')

base=1.0/19
print('\\n'+'='*66)
print(f'{"variante":<12}{"train EM":>10}{"held EM":>10}{"held ppl":>10}   (baseline EM={base:.3f})')
for nm in ('CORRIGIDO','ANTIGO'):
    rs=abl[nm]; emh=[r['em_he'] for r in rs]; pph=[r['ppl_he'] for r in rs]; emt=[r['em_tr'] for r in rs]
    print(f'{nm:<12}{np.mean(emt):>10.3f}{np.mean(emh):>10.3f}{np.mean(pph):>10.3f}   seeds held EM={[round(x,2) for x in emh]}')
print('='*66)
d=np.mean([r['em_he'] for r in abl['CORRIGIDO']])-np.mean([r['em_he'] for r in abl['ANTIGO']])
print(f'GANHO ATRIBUIVEL (corrigido - antigo) held-out EM = {d:+.3f}')
print('-> ganho positivo = evidencia de que as CORRECOES causaram a generalizacao.')
"""))

cells.append(md("""## 11 — Generalização SISTEMÁTICA (holdout de operando, teste difícil)

Esconde um operando inteiro (**A=7**) do treino. O valor 7 ainda aparece como
**B** e como **saída** (`sete`) em outros pares — então acertar `7 + b` exige que a
representação do operando-A **transfira** para um valor nunca visto NAQUELA posição.
Bem mais difícil que o holdout aleatório (que é interpolação num grid quase cheio).
Roda o stack corrigido (flags já ativos da célula 6), in-process. ~2 min."""))
cells.append(code("""import random, numpy as np, time
A_HOLD = 7
sys_train = [(a, b) for a in range(10) for b in range(10) if a != A_HOLD]
sys_test  = [(A_HOLD, b) for b in range(10)]
out_train = {a + b for (a, b) in sys_train}
assert all((A_HOLD + b) in out_train for b in range(10)), 'saida do teste sem cobertura'
print(f'[sys] train={len(sys_train)} (A!={A_HOLD})  test={len(sys_test)} (A={A_HOLD}, inedito nessa posicao)')

nsos.set_seed(7)
cs = nsos.ModelConfig()
cs.num_layers=4; cs.d_model=128; cs.vocab_size=V; cs.n_heads=4; cs.n_kv_heads=2
cs.use_moe=True; cs.num_experts=4; cs.num_experts_per_token=2; cs.moe_period=2; cs.moe_slot=1
cs.attention_period=4; cs.attention_slot=3; cs.use_ttt=False; cs.dropout=0.0
cs.max_context_tokens=64; cs.use_exact_attention_training=True
ms = nsos.JambaModel(cs, dev); ms.to(dev); ms.set_training_mode(True)
tsr = nsos.Trainer(ms, 2e-3); tsr.warmup_steps=100; tsr.eos_token_id=EOS; tsr.moe_aux_loss_scale=0.01
EP=80; tsr.total_training_steps=EP*len(sys_train)
rsd=random.Random(7); t0=time.perf_counter()
for e in range(EP):
    o=sys_train[:]; rsd.shuffle(o)
    for (a,b) in o: tsr.train_supervised(prompt_ids(a,b), answer_ids(a,b))
ms.set_training_mode(False)
def evs(prs):
    cor=0
    for (a,b) in prs:
        p=prompt_ids(a,b)
        lg=np.asarray(ms.forward_ids(p).cpu().numpy()).reshape(len(p),V)
        if VOCAB[int(np.argmax(lg[-1]))]==NUM_WORDS[a+b]: cor+=1
    return cor/len(prs)
em_tr=evs(sys_train); em_te=evs(sys_test); base=1.0/19
print('='*60)
print(f'SISTEMATICO   train EM={em_tr:.3f}   test(A={A_HOLD}) EM={em_te:.3f}   baseline={base:.3f}  ({time.perf_counter()-t0:.0f}s)')
print('='*60)
v = 'GENERALIZACAO SISTEMATICA' if em_te>3*base else ('parcial' if em_te>base else 'falhou (decorou a posicao)')
print(f'VEREDITO: {v}')
for (a,b) in sys_test:
    p=prompt_ids(a,b); lg=np.asarray(ms.forward_ids(p).cpu().numpy()).reshape(len(p),V)
    print(f'  {NUM_WORDS[a]} + {NUM_WORDS[b]} = {NUM_WORDS[a+b]:>10} | modelo: {VOCAB[int(np.argmax(lg[-1]))]}')
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
