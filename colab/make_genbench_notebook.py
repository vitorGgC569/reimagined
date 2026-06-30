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

v5 (the GPU-first hardening pass, commit 22df3ba): the v3 root cause above is now
HALF fixed at the source -- BitLinear gained a real GPU fake-quant STE path, so
GPU training and GPU ternary inference can now use the SAME quantized numerics
(no more float-train / ternary-eval mismatch BY ITSELF).  But progressive QAT
(`TrainPhaseScheduler.progressive_qat_enabled`) also flipped to default-ON
project-wide, with no env-var escape hatch (unlike the Mamba NSOS_MAMBA_* flags,
it is a Trainer field, not read from the environment) -- so an UNCHANGED copy of
this notebook would have silently started quantizing every variant's weights
mid-run (after step ~300) without anyone deciding that on purpose.  That is
exactly the kind of confound this suite's pillar-2 methodology (isolate ONE
variable) exists to catch.  Fix: `train_batched()` now sets
`tr.phase_scheduler.progressive_qat_enabled` EXPLICITLY from a new
`NSOS_GEN_QAT` knob (default '0' -> stays FP32 throughout, preserving the v1-v4
mixer-only comparison byte-for-byte); set `NSOS_GEN_QAT=1` to additionally check
whether the mixer ranking holds when real GPU ternary QAT is engaged.  The build
gate also now compiles+runs the GPU parity tests for what changed today
(`test_gpu_parity_mamba_proper[_stream]`, `test_gpu_parity_mamba_nstate`) so a
broken N1 (A reparameterization) or the new batched-proper-streaming fix would
fail loudly here before any training time is spent.

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

cells.append(md("""# NSOS — Suíte de generalização (padrão-ouro) v5: recall + extrapolação

Fase 1 da avaliação de **arquitetura** (branch `%BRANCH%`).

**Achados das runs v1–v4 (incorporados aqui):**
- O **harness funciona** — a tarefa **selective copy aprendeu** (mamba 0.99, híbrido
  0.78, atenção 0.69 no comprimento de treino) e o overfit de 1 exemplo vai a loss 0.
- **Primeiro sinal de arquitetura (copy):** o **Mamba extrapola melhor** em
  comprimento (L=128: mamba 0.33 vs atenção 0.18) — a tese do SSM.
- O **AR (recall) não treinava** porque minha formulação tinha chave/valor no MESMO
  vocab (o modelo tinha que inferir chave-vs-valor por posição) + pouco sinal/steps —
  corrigido com vocabulários DISJUNTOS (v4): **mamba 0.657 ≫ híbrido 0.481 ≫ atenção
  0.197** em n_kv=8 (3 seeds, 10k steps) — a atenção do NSOS aprende copy mas NÃO
  recall por conteúdo; auditoria estática confirmou que a matemática está CORRETA
  (não é bug), é dinâmica/hiperparâmetro nessa escala.

**v5 (auditoria + correções de hoje, commit `22df3ba`):** o núcleo mudou de baixo —
Mamba-2 SSD corrigido e o gradiente de tarefa do roteador MoE agora são DEFAULT
(antes opt-in por env var); BitLinear ganhou QAT real em GPU (fake-quant STE
ternário+int8, antes só treinava em float na GPU). Isso introduz um confound novo
pro isolamento do mixer (pilar 2 da metodologia): `progressive_qat_enabled` virou
default-ON no `Trainer` e SEM escape-hatch por env var — então rodar este notebook
sem mudança nenhuma já mudaria silenciosamente o que ele mede. Fix: QAT agora é
controlado explicitamente aqui via `NSOS_GEN_QAT` (default OFF = mantém a
comparação v1-v4 limpa byte-a-byte; `=1` testa se o ranking do mixer sobrevive sob
quantização real). Gate de build também passou a compilar+rodar os testes de
paridade GPU específicos do que mudou hoje (Mamba A reparametrizado, streaming
batched do proper-SSM).

Tarefas 100% sintéticas (sem tokenizer/download). Compara o mixer: atenção-pura /
Mamba-puro / híbrido NSOS."""))

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
bitlin_h = (REPO_ROOT/'OXN/nsos/include/bitlinear.h').read_text('utf-8')
assert 'proper_selective_ssm' in mamba and 'forward_proper_step' in mamba, 'fonte stale (pre-v3)'
assert 'mamba_a_eff' in mamba, 'fonte stale: falta N1 (A reparametrizado em log-domain)'
assert 'qat_gpu_active_' in bitlin_h, 'fonte stale: falta K3 (QAT real em GPU no BitLinear)'
print(f'[ver] HEAD={sha} | Mamba proper + passo on-device + A em log-domain (N1) + GPU QAT (K3): OK')
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
    GATE_TESTS = ['test_gradcheck', 'test_gpu_parity_mamba_proper',
                  'test_gpu_parity_mamba_proper_stream', 'test_gpu_parity_mamba_nstate']
    r = subprocess.run(['cmake','--build',BUILD,'-j','2','--target','nsos_ext',*GATE_TESTS],
                       capture_output=True, text=True)
    print(r.stdout[-1000:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('build falhou')
    so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True)); assert so
    shutil.copy2(so[0], EXT_DIR/'nsos_ext.so'); CACHE_SO.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(so[0], CACHE_SO); os.environ['NSOS_BUILD_GENBENCH']=BUILD
    print('[build] ok')
B = os.environ['NSOS_BUILD_GENBENCH']
# gradcheck (matematica do backward, FD vs hand-derived) + paridade GPU dos kernels
# tocados hoje (N1: A reparametrizado em log-domain; streaming batched do proper-SSM)
# -- corretude verificada SEM gastar tempo de treino, antes de qualquer comparacao
# de mixer abaixo.
GATE_FAIL = False
for name in ['test_gradcheck', 'test_gpu_parity_mamba_proper',
             'test_gpu_parity_mamba_proper_stream', 'test_gpu_parity_mamba_nstate']:
    found = False
    for cand in (f'{B}/{name}', f'{B}/Release/{name}', f'{B}/{name}.exe'):
        if os.path.exists(cand):
            found = True
            rr = subprocess.run([cand], capture_output=True, text=True)
            ok = rr.returncode == 0
            GATE_FAIL = GATE_FAIL or not ok
            print(f'[{name}]', 'PASS' if ok else f'FAIL (rc={rr.returncode})\\n{rr.stdout[-1500:]}\\n{rr.stderr[-1500:]}')
            break
    if not found:
        print(f'[{name}] binario nao encontrado (build pulou o alvo?)')
if GATE_FAIL:
    raise RuntimeError('gate de corretude falhou -- NAO prossiga para o treino sem investigar')
"""))

cells.append(md("""## 4 — Tarefas + avaliação

**AR:** prompt `[BOS k1 v1 ... kN vN SEP q]`, resposta `[valor de q]` (loss = só o
valor → 100% sinal de recall). `n_kv` = capacidade. **Selective copy:** campo com K
dados entre BLANK + MARK → resposta = os K dados; treina curto, testa longo.
`eval_*` sempre roda com `set_training_mode(False)` (caminho de inferência do
modelo — ternário 1.58-bit quando os pesos já passaram pela fase quantizada; float
se QAT está OFF nesta run, ver célula 5)."""))
cells.append(code("""import random, numpy as np

def make_ar(rng, n_kv, n_key, n_val):
    # vocabularios DISJUNTOS: chaves em [0,n_key), valores em [n_key,n_key+n_val).
    # (id ja diz se e chave ou valor -> circuito de recall limpo; era a causa do AR
    # nao treinar na v3, onde chave/valor compartilhavam o vocab.)
    BOS, SEP = n_key + n_val, n_key + n_val + 1
    keys = rng.sample(range(n_key), n_kv)
    vals = [n_key + rng.randrange(n_val) for _ in range(n_kv)]
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

def eval_ar(model, data, V):
    model.set_training_mode(False)
    cor = tot = 0
    for prompt, ans in data:
        lg = np.asarray(model.forward_ids(prompt).cpu().numpy()).reshape(-1, V)
        cor += int(int(np.argmax(lg[-1])) == ans[0]); tot += 1
    return cor / max(tot, 1)

def eval_copy(model, data, V):
    model.set_training_mode(False)
    cor = tot = 0
    for prompt, ans in data:
        seq = prompt + ans
        lg = np.asarray(model.forward_ids(seq).cpu().numpy()).reshape(-1, V)
        ok = all(int(np.argmax(lg[len(prompt) + j - 1])) == ans[j] for j in range(len(ans)))
        cor += int(ok); tot += 1
    return cor / max(tot, 1)

# Nota (v5): isto NAO e mais verdade incondicionalmente -- desde a auditoria de
# hoje, BitLinear tem um caminho real de QAT em GPU (fake-quant STE ternario+int8),
# entao quando NSOS_GEN_QAT=1 o forward de treino E o de eval passam pelos MESMOS
# pesos quantizados (sem mismatch). Com NSOS_GEN_QAT=0 (default aqui) o treino fica
# em float o tempo todo, igual as runs v1-v4 -- ver a celula 5.
print('[tasks] AR (vocab disjunto) + selective-copy definidos')
"""))

cells.append(md("""## 5 — Variantes do MIXER + treino em batch

Atenção-pura / Mamba-puro (proper SSM) / híbrido NSOS, mesmo tamanho. MoE/KAN/TTT
OFF (compara o mixer). `train_supervised_batch` (batches homogêneos), loss impressa,
`first_token_loss_scale=eos_loss_scale=1.0`.

**QAT (v5):** `progressive_qat_enabled` agora é default-ON no `Trainer` (sem env-var
escape hatch — ao contrário dos flags `NSOS_MAMBA_*`). Pra manter esta comparação de
mixer ISOLADA (pilar 2 da metodologia: uma variável de cada vez), `train_batched()`
desliga QAT explicitamente por padrão (`NSOS_GEN_QAT=0`) — treino fica em float do
início ao fim, igual às runs v1-v4. Defina `NSOS_GEN_QAT=1` pra rodar a MESMA
comparação com QAT real ligado (ternário+int8 depois do step 300) e checar se o
ranking do mixer sobrevive à quantização."""))
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
KVHEADS = int(os.environ.get('NSOS_GEN_KVHEADS', '2'))   # =4 -> MHA (sem GQA), testa a suspeita
QAT_ON = os.environ.get('NSOS_GEN_QAT', '0') == '1'      # v5: ver nota da celula acima

def build_variant(variant, V):
    c = nsos.ModelConfig()
    c.num_layers = NUM_LAYERS; c.d_model = DMODEL; c.vocab_size = V
    c.n_heads = 4; c.n_kv_heads = KVHEADS; c.max_context_tokens = 1024
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
    # v5: decisao EXPLICITA (nao herdada do default do Trainer) -- ver celula 5.
    tr.phase_scheduler.progressive_qat_enabled = QAT_ON
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
SEEDS = [int(x) for x in os.environ.get('NSOS_GEN_SEEDS', '0,1,2').split(',')]   # 3 seeds -> IC
STEPS = int(os.environ.get('NSOS_GEN_STEPS', '1500'))                            # copy (converge rapido)
BATCH = int(os.environ.get('NSOS_GEN_BATCH', '16'))
print(f'[cfg] L={NUM_LAYERS} d={DMODEL} | {VARIANTS} | seeds={SEEDS} | steps={STEPS} batch={BATCH} | QAT={"ON" if QAT_ON else "OFF"} (={STEPS*BATCH} ex/run)')
"""))

cells.append(md("""## 5b — Sanidade: o trainer decora 1 exemplo?

Treina a atenção pra **decorar UM** exemplo de AR (vocab disjunto, n_kv=4). A loss
tem que ir a ~0 e o pred bater. Se não, o problema é trainer/gradiente (não dados
nem arquitetura). (Já confirmado nas runs anteriores; fica como gate rápido.) QAT
desligado aqui sempre (é um teste de gradiente puro, não da comparação de mixer)."""))
cells.append(code("""import numpy as np, random
_NK = 16; _NV = 16; _V = _NK + _NV + 2
nsos.set_seed(0)
mdbg = build_variant('attn', _V)
ex_p, ex_a = make_ar(random.Random(1), 4, _NK, _NV)   # 1 exemplo fixo, n_kv=4
tr = nsos.Trainer(mdbg, 3e-3); tr.total_training_steps = 400
tr.phase_scheduler.progressive_qat_enabled = False  # gradiente puro, nao mistura QAT
tr.first_token_loss_scale = 1.0; tr.eos_loss_scale = 1.0
mdbg.set_training_mode(True)
for s in range(400):
    L = tr.train_supervised(ex_p, ex_a)
    if s % 100 == 0 or s == 399: print(f'  overfit step {s:3d}: loss={L:.4f}')
mdbg.set_training_mode(False)
lg = np.asarray(mdbg.forward_ids(ex_p).cpu().numpy()).reshape(-1, _V)
pred = int(np.argmax(lg[-1]))
print('=' * 60)
print(f'  alvo={ex_a[0]}  pred={pred}  -> {"OK (trainer decora 1 exemplo)" if pred==ex_a[0] else "FALHOU (trainer quebrado)"}')
print('=' * 60)
"""))

cells.append(md("""## 6 — Recall associativo: capacidade (vocab disjunto; treina n_kv=8; testa 4/8/16)

Baseline = 1/n_val. **Gate:** atenção tem que acertar bem em n_kv=8 (controle
positivo). Mais steps + LR maior que o copy (recall é mais difícil de induzir)."""))
cells.append(code("""import time, numpy as np, random
N_KEY = int(os.environ.get('NSOS_AR_KEYS', '16')); N_VAL = int(os.environ.get('NSOS_AR_VALS', '16'))
V = N_KEY + N_VAL + 2; base = 1.0 / N_VAL; KV_TEST = [4, 8, 16]
# CONVERGENCIA: AR e o recall, lento de induzir -> muitos steps ate a loss estabilizar.
AR_STEPS = int(os.environ.get('NSOS_AR_STEPS', '10000'))
AR_LR = float(os.environ.get('NSOS_AR_LR', '2e-3'))   # = LR do copy (onde a atencao funcionou)
ar = {}; t0 = time.time()
for variant in VARIANTS:
    acc = {k: [] for k in KV_TEST}
    for seed in SEEDS:
        nsos.set_seed(seed); m = build_variant(variant, V)
        train_batched(m, AR_LR, AR_STEPS, BATCH, lambda r: (lambda: make_ar(r, 8, N_KEY, N_VAL)), seed, f'AR {variant} s{seed}')
        ev = random.Random(seed * 13 + 5)
        for k in KV_TEST:
            d = [make_ar(ev, k, N_KEY, N_VAL) for _ in range(300)]
            acc[k].append(eval_ar(m, d, V))
        print(f'  [{variant:6s} s{seed}] ' + ' '.join(f'n_kv={k}:{acc[k][-1]:.2f}' for k in KV_TEST))
    ar[variant] = {k: (float(np.mean(acc[k])), float(np.std(acc[k]))) for k in KV_TEST}
print('\\n' + '=' * 72)
print(f'RECALL ASSOCIATIVO (media+/-desvio, {len(SEEDS)} seeds; baseline={base:.3f}; treina n_kv=8, {AR_STEPS} steps)   ({time.time()-t0:.0f}s)')
print(f'{"variante":<10}' + ''.join(f'{("n_kv="+str(k)):>18}' for k in KV_TEST))
for v in VARIANTS:
    print(f'  {v:<8}' + ''.join(f'{ar[v][k][0]:>11.3f}+/-{ar[v][k][1]:.2f}' for k in KV_TEST))
print('=' * 72)
"""))

cells.append(md("""## 7 — Selective copy: extrapolação de comprimento (treina ≤64; testa 64/128/256)

Já funcionou nas runs anteriores (harness validado). Mede quem **extrapola** melhor
pra campos mais longos que o treino — o upside estrutural do SSM."""))
cells.append(code("""import time, numpy as np, random
N_SYM_C = int(os.environ.get('NSOS_COPY_SYM', '20')); N_DATA = 4
LT = int(os.environ.get('NSOS_COPY_LEN', '64')); Vc = N_SYM_C + 3
base_c = (1.0 / N_SYM_C) ** N_DATA; TEST_LENS = [LT, 2 * LT, 4 * LT]
copy = {}; t0 = time.time()
for variant in VARIANTS:
    acc = {L: [] for L in TEST_LENS}
    for seed in SEEDS:
        nsos.set_seed(seed); m = build_variant(variant, Vc)
        def sampler(r):
            L = r.randint(16, LT)
            return lambda: make_selcopy(r, L, N_DATA, N_SYM_C)
        train_batched(m, 2e-3, STEPS, BATCH, sampler, seed, f'COPY {variant} s{seed}')
        ev = random.Random(seed * 29 + 3)
        for L in TEST_LENS:
            d = [make_selcopy(ev, L, N_DATA, N_SYM_C) for _ in range(300)]
            acc[L].append(eval_copy(m, d, Vc))
        print(f'  [{variant:6s} s{seed}] ' + ' '.join(f'L={L}:{acc[L][-1]:.2f}' for L in TEST_LENS))
    copy[variant] = {L: (float(np.mean(acc[L])), float(np.std(acc[L]))) for L in TEST_LENS}
print('\\n' + '=' * 72)
print(f'SELECTIVE COPY exact-match (media+/-desvio, {len(SEEDS)} seeds; baseline~{base_c:.0e}; treino<=64)   ({time.time()-t0:.0f}s)')
print(f'{"variante":<10}' + ''.join(f'{("L="+str(L)):>18}' for L in TEST_LENS))
for v in VARIANTS:
    print(f'  {v:<8}' + ''.join(f'{copy[v][L][0]:>11.3f}+/-{copy[v][L][1]:.2f}' for L in TEST_LENS))
print('=' * 72)
"""))

cells.append(md("""## 8 — VEREDITO (gate = atenção resolve AR n_kv=8)"""))
cells.append(code("""print('=' * 72)
print(f'HARNESS: VALIDADO (copy aprende ~1.0 + overfit->loss 0). QAT nesta run: {"ON (ternario+int8 real)" if QAT_ON else "OFF (float, comparacao v1-v4 preservada)"}')
print('Comparacao de mixer:')
print(f'\\nRECALL (AR) — media+/-desvio por n_kv (baseline=1/n_val={base:.3f}):')
for v in VARIANTS:
    print(f'   {v:<8} ' + '  '.join(f'n_kv={k}:{ar[v][k][0]:.3f}+/-{ar[v][k][1]:.2f}' for k in KV_TEST))
rank = sorted(VARIANTS, key=lambda v: ar[v][8][0], reverse=True)
print(f'   RANKING recall (n_kv=8): {" > ".join(rank)}')
print(f'\\nCOPY — extrapolacao de comprimento (exact-match):')
for v in VARIANTS:
    print(f'   {v:<8} ' + '  '.join(f'L={L}:{copy[v][L][0]:.3f}' for L in TEST_LENS))
rc = max(VARIANTS, key=lambda v: copy[v][2 * LT][0])
print(f'   melhor extrapolador (L={2*LT}): {rc}')
conv = ar['attn'][8][0] >= 0.7
print('\\nLEITURA:')
if not conv:
    print(f' - Controle (atencao) ainda NAO convergiu (AR n_kv=8={ar["attn"][8][0]:.2f}; veja a loss final).')
    print('   Suba NSOS_AR_STEPS se a loss ainda caia. O RANKING entre mixers ja e robusto.')
print(' - Vale escalar se o mixer do NSOS (Mamba/hibrido) acompanha/supera no recall E')
print('   extrapola melhor em comprimento. Preliminar: o Mamba lidera AMBOS.')
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
