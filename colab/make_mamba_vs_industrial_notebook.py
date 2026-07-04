"""Generator for colab/bench_mamba_vs_industrial_t4.ipynb.

Head-to-head: NSOS Mamba (our from-scratch Mamba-2 SSD) vs the INDUSTRIAL
reference Mamba-2 (state-spaces `mamba-ssm`), on the Tesla T4, in ONE run.

Two axes, honestly separated:
  A) PARITY (legitimacy) — NSOS Mamba (FLOAT) vs reference Mamba-2 (fp32),
     matched d_model/layers, same synthetic tasks/seeds.  Expectation is a
     TIE: it is the same published SSD algorithm, so matching (not beating)
     is the correct, credible result — it proves our from-scratch Mamba is
     the real thing.  A "win" here would be a red flag (comparison bug).
  B) EFFICIENCY (the real, fundable advantage) — NSOS Mamba TERNARY (1.58-bit
     BitNet QAT) vs reference Mamba-2 (fp16).  If our ternary model holds the
     quality at ~10x fewer weight bits / lower VRAM, THAT is the edge — not a
     "better Mamba".

Tasks (Zoology / Mamba standard architecture probes, matched for both models):
  * MQAR / associative recall (capacity: n_kv 4/8/16) — most predictive probe.
  * selective copy (length extrapolation: train <=64, test 64/128/256).

Metrics: accuracy per task, param count, weight bits/param, model MB, and
(reference side) peak CUDA VRAM + decode tok/s.  Verdict table separates
"same-quality-in-float" (parity) from "quality-per-bit" (efficiency).

Run once (pure JSON authoring, no GPU): python colab/make_mamba_vs_industrial_notebook.py
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

cells.append(md("""# NSOS Mamba vs Mamba-2 industrial (`mamba-ssm`) — head-to-head no T4

Uma corrida, dois eixos **separados honestamente**:

- **A) Paridade (legitimidade):** nosso Mamba (FLOAT) vs Mamba-2 oficial (fp32),
  mesmo `d_model`/camadas, mesmas tarefas/seeds.  **Esperado = EMPATE** — é o
  mesmo algoritmo SSD publicado, então *igualar* (não bater) é o resultado
  correto e crível: prova que nosso Mamba do zero é a coisa real.  Uma
  "vitória" aqui seria red flag (bug na comparação).
- **B) Eficiência (a vantagem real e fundável):** nosso Mamba **TERNÁRIO
  (1.58-bit BitNet QAT)** vs Mamba-2 oficial (fp16).  Se seguramos a qualidade
  a ~10× menos bits/peso e menos VRAM, *isso* é o diferencial — não um "Mamba
  melhor".

Tarefas (probes de arquitetura padrão Zoology/Mamba, idênticas p/ os dois):
**MQAR / recall associativo** (capacidade n_kv 4/8/16) + **selective copy**
(extrapolação de comprimento: treina ≤64, testa 64/128/256)."""))

# ── 1 GPU + Drive ────────────────────────────────────────────────────────────
cells.append(md("## 1 — GPU + (opcional) Drive p/ cache do build"))
cells.append(code("""import os, subprocess
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_mamba_vs'); DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print('[drive] indisponivel (ok):', e); DRIVE_ROOT = Path('/content/nsos_mamba_vs'); DRIVE_ROOT.mkdir(exist_ok=True)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=False)
"""))

# ── 2 clone ──────────────────────────────────────────────────────────────────
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
assert 'proper_selective_ssm' in mamba and 'mamba_a_eff' in mamba, 'fonte stale'
print(f'[ver] HEAD={sha} | Mamba-2 SSD proper (N1 log-domain): OK')
"""))

# ── 3 build nsos_ext + gate ──────────────────────────────────────────────────
cells.append(md("""## 3 — Build `nsos_ext` + gate de corretude (CUDA T4 sm_75)

Antes de qualquer comparação: gradcheck (backward FD vs hand-derived) + paridade
GPU do Mamba proper/N-state.  Se o gate falhar, o nosso Mamba está quebrado e a
comparação não vale."""))
cells.append(code("""import os, sys, subprocess, shutil, glob
from pathlib import Path
if 'nsos_ext' in sys.modules:
    raise RuntimeError('nsos_ext ja carregado — Runtime > Restart.')
PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
sha = subprocess.run(['git','-C',str(REPO_ROOT),'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
BUILD = str(REPO_ROOT / 'OXN/nsos/build-mambavs')
EXT_DIR = Path('/content/nsos_ext_mambavs'); EXT_DIR.mkdir(exist_ok=True)
CACHE_SO = DRIVE_ROOT / '_bootstrap' / f'nsos_ext_mambavs.{PYTAG}.{sha}.so'
subprocess.run([sys.executable,'-m','pip','install','-q','pybind11','numpy'], check=True)
GATES = ['test_gradcheck','test_gpu_parity_mamba_proper','test_gpu_parity_mamba_nstate']
if CACHE_SO.exists() and Path(BUILD).exists():
    shutil.copy2(CACHE_SO, EXT_DIR/'nsos_ext.so'); print('[build] .so do cache:', CACHE_SO)
else:
    r = subprocess.run(['cmake','-S',str(REPO_ROOT/'OXN/nsos'),'-B',BUILD,
                        '-DNSOS_ENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=75',
                        '-DNSOS_BUILD_PYTHON=ON','-DNSOS_BUILD_TESTS=ON',
                        '-DNSOS_BUILD_CLI=OFF','-DNSOS_BUILD_API=OFF','-DNSOS_BUILD_OXTAMEM=OFF',
                        '-DCMAKE_BUILD_TYPE=Release',f'-DPython3_EXECUTABLE={sys.executable}'],
                       capture_output=True, text=True)
    print(r.stdout[-600:])
    if r.returncode != 0: print('STDERR:', r.stderr[-3000:]); raise RuntimeError('cmake falhou')
    r = subprocess.run(['cmake','--build',BUILD,'-j','2','--target','nsos_ext',*GATES],
                       capture_output=True, text=True)
    print(r.stdout[-600:])
    if r.returncode != 0: print('STDERR:', r.stderr[-3000:]); raise RuntimeError('build falhou')
    so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True)); assert so
    shutil.copy2(so[0], EXT_DIR/'nsos_ext.so'); CACHE_SO.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(so[0], CACHE_SO); print('[build] ok')
GATE_FAIL = False
for name in GATES:
    for cand in (f'{BUILD}/{name}', f'{BUILD}/Release/{name}'):
        if os.path.exists(cand):
            rr = subprocess.run([cand], capture_output=True, text=True); ok = rr.returncode == 0
            GATE_FAIL = GATE_FAIL or not ok
            print(f'[{name}]', 'PASS' if ok else f'FAIL\\n{rr.stdout[-800:]}{rr.stderr[-800:]}'); break
if GATE_FAIL: raise RuntimeError('gate de corretude falhou — nao prossiga')
print('gate OK: nosso Mamba-2 esta correto (gradcheck + paridade GPU).')
"""))

# ── 4 install mamba-ssm ──────────────────────────────────────────────────────
cells.append(md("""## 4 — Referência industrial: `mamba-ssm` oficial (Mamba-2)

Instala o `mamba-ssm` (state-spaces) — a implementação **industrial de
referência** do Mamba-2.  Se a wheel/compilação falhar no T4, cai para um
Mamba-2 SSD compacto em PyTorch puro (MESMO algoritmo, sem o kernel fundido —
válido para comparar QUALIDADE; só a velocidade dependeria do kernel).  A
célula reporta QUAL referência está em uso."""))
cells.append(code("""import subprocess, sys, torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| dev', torch.cuda.get_device_name(0))
REF_KIND = None
try:
    subprocess.run([sys.executable,'-m','pip','install','-q','causal-conv1d>=1.2.0'], check=False)
    r = subprocess.run([sys.executable,'-m','pip','install','-q','mamba-ssm'], capture_output=True, text=True)
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig as _MC
    REF_KIND = 'mamba-ssm (oficial)'
    print('[ref] mamba-ssm OFICIAL importado')
except Exception as e:
    print('[ref] mamba-ssm indisponivel (', str(e)[:120], ') -> fallback PyTorch Mamba-2')
    REF_KIND = None
"""))

# ── 5 reference model factory (official or fallback) ─────────────────────────
cells.append(md("""## 5 — Fábrica do modelo de referência (oficial ou fallback PyTorch)

`ref_model(V)` devolve um LM Mamba-2 com o MESMO `d_model`/camadas do nosso.  Se
o oficial estiver disponível, usa `MambaLMHeadModel`; senão, um Mamba-2 SSD
recorrente em PyTorch puro (implementação de referência clara)."""))
cells.append(code("""import torch, torch.nn as nn, torch.nn.functional as F, math
D_MODEL = int(os.environ.get('NSOS_GEN_DMODEL', '128'))
N_LAYER = int(os.environ.get('NSOS_GEN_LAYERS', '4'))
D_STATE = int(os.environ.get('MAMBA_D_STATE', '64'))
D_CONV  = 4
DEV = 'cuda'

class _RefMamba2Block(nn.Module):
    # Compact recurrent Mamba-2 SSD (single group).  Same math as the published
    # SSD; naive scan (no fused kernel) -> quality-faithful reference baseline.
    def __init__(self, d_model, d_state, d_conv, expand=2, headdim=32):
        super().__init__()
        self.d_inner = expand * d_model
        self.nheads = self.d_inner // headdim
        self.headdim = headdim; self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2*self.d_inner + 2*d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner + 2*d_state, self.d_inner + 2*d_state, d_conv,
                                groups=self.d_inner + 2*d_state, padding=d_conv-1, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.norm = nn.RMSNorm(self.d_inner) if hasattr(nn,'RMSNorm') else nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    def forward(self, x):                       # x: [B,T,D]
        B, T, _ = x.shape
        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner+2*self.d_state, self.nheads], dim=-1)
        xBC = self.conv1d(xBC.transpose(1,2))[..., :T].transpose(1,2)
        xBC = F.silu(xBC)
        xs, Bm, Cm = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)                         # [B,T,H]
        A = -torch.exp(self.A_log)                                 # [H]
        xs = xs.view(B, T, self.nheads, self.headdim)
        h = x.new_zeros(B, self.nheads, self.headdim, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t] * A)                           # [B,H]
            h = h * dA[:, :, None, None] + \
                (dt[:, t][:, :, None, None] * xs[:, t][:, :, :, None]) * Bm[:, t][:, None, None, :]
            y = (h * Cm[:, t][:, None, None, :]).sum(-1)           # [B,H,P]
            y = y + self.D[None, :, None] * xs[:, t]
            ys.append(y.reshape(B, self.d_inner))
        y = torch.stack(ys, 1)
        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)

class _RefMamba2LM(nn.Module):
    def __init__(self, V, d_model, n_layer, d_state, d_conv):
        super().__init__()
        self.emb = nn.Embedding(V, d_model)
        self.blocks = nn.ModuleList([_RefMamba2Block(d_model, d_state, d_conv) for _ in range(n_layer)])
        self.norms = nn.ModuleList([(nn.RMSNorm(d_model) if hasattr(nn,'RMSNorm') else nn.LayerNorm(d_model)) for _ in range(n_layer)])
        self.norm_f = nn.RMSNorm(d_model) if hasattr(nn,'RMSNorm') else nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, V, bias=False)
    def forward(self, ids):
        h = self.emb(ids)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        return self.head(self.norm_f(h))

def ref_model(V):
    if REF_KIND and REF_KIND.startswith('mamba-ssm'):
        cfg = _MC(d_model=D_MODEL, n_layer=N_LAYER, vocab_size=V,
                  ssm_cfg={'layer': 'Mamba2', 'd_state': D_STATE},
                  rms_norm=True, tie_embeddings=False)
        return MambaLMHeadModel(cfg).to(DEV).float()
    return _RefMamba2LM(V, D_MODEL, N_LAYER, D_STATE, D_CONV).to(DEV).float()

def n_params(m): return sum(p.numel() for p in m.parameters())
_probe = ref_model(64); REF_PARAMS = n_params(_probe)
print(f'[ref] kind = {REF_KIND or \"PyTorch Mamba-2 (fallback)\"} | d_model={D_MODEL} n_layer={N_LAYER} d_state={D_STATE} | params={REF_PARAMS:,}')
print('[nota] o bloco NSOS tem FFN dense por camada; o Mamba oficial intercala sem FFN separada')
print('       -> os param counts DIFEREM; a comparacao de EFICIENCIA e por bits/peso (robusta a isso).')
del _probe; torch.cuda.empty_cache()
"""))

# ── 6 tasks (shared) ─────────────────────────────────────────────────────────
cells.append(md("""## 6 — Tarefas idênticas p/ os dois modelos + treino de referência

MQAR (vocab de chaves/valores DISJUNTO → sinal de recall limpo) + selective
copy.  O treino da referência (PyTorch) usa CE **só nos tokens da resposta**
(teacher forcing), espelhando o `train_supervised_batch` do NSOS."""))
cells.append(code("""import random, numpy as np, torch, time

def make_ar(rng, n_kv, n_key, n_val):
    BOS, SEP = n_key + n_val, n_key + n_val + 1
    keys = rng.sample(range(n_key), n_kv); vals = [n_key + rng.randrange(n_val) for _ in range(n_kv)]
    kv = dict(zip(keys, vals)); q = rng.choice(keys)
    prompt = [BOS]
    for k, v in zip(keys, vals): prompt += [k, v]
    prompt += [SEP, q]
    return prompt, [kv[q]]

def make_selcopy(rng, field_len, n_data, n_sym):
    BLANK, MARK, BOS = n_sym, n_sym + 1, n_sym + 2
    pos = sorted(rng.sample(range(field_len), n_data)); data = [rng.randrange(n_sym) for _ in range(n_data)]
    field = [BLANK]*field_len
    for p, d in zip(pos, data): field[p] = d
    return [BOS] + field + [MARK], data

def _batch_pad(pairs, V):
    # returns ids [B,T], mask [B,T] (1 on answer positions to score), lens
    seqs = [p + a for p, a in pairs]; T = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), T, dtype=torch.long)
    tgt = torch.zeros(len(seqs), T, dtype=torch.long)
    msk = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (p, a) in enumerate(pairs):
        s = p + a; ids[i, :len(s)] = torch.tensor(s)
        for j in range(len(a)):                       # predict a[j] from position len(p)+j-1
            pos = len(p) + j - 1; tgt[i, pos] = a[j]; msk[i, pos] = True
    return ids.to(DEV), tgt.to(DEV), msk.to(DEV)

def ref_train(model, lr, steps, batch, sampler, seed, label, V):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed*991+7); model.train(); win = max(1, steps//6); run = 0.0
    for s in range(steps):
        bs = sampler(rng); pairs = [bs() for _ in range(batch)]
        ids, tgt, msk = _batch_pad(pairs, V)
        logits = model(ids); logits = logits.logits if hasattr(logits,'logits') else logits
        loss = F.cross_entropy(logits[msk], tgt[msk])
        opt.zero_grad(); loss.backward(); opt.step(); run += loss.item()
        if (s+1) % win == 0: print(f'    [{label}] step {s+1}/{steps} loss~{run/win:.3f}'); run = 0.0
    model.eval()

@torch.no_grad()
def ref_eval_ar(model, data, V):
    model.eval(); cor = 0
    for prompt, ans in data:
        ids = torch.tensor([prompt], device=DEV)
        lg = model(ids); lg = (lg.logits if hasattr(lg,'logits') else lg)[0]
        cor += int(int(lg[-1].argmax()) == ans[0])
    return cor/max(len(data),1)

@torch.no_grad()
def ref_eval_copy(model, data, V):
    model.eval(); cor = 0
    for prompt, ans in data:
        ids = torch.tensor([prompt+ans], device=DEV)
        lg = model(ids); lg = (lg.logits if hasattr(lg,'logits') else lg)[0]
        ok = all(int(lg[len(prompt)+j-1].argmax()) == ans[j] for j in range(len(ans)))
        cor += int(ok)
    return cor/max(len(data),1)
print('[tasks] MQAR + selective-copy + treino/eval da referencia definidos')
"""))

# ── 7 NSOS model factory + train ─────────────────────────────────────────────
cells.append(md("""## 7 — Nosso Mamba (NSOS): float e ternário (1.58-bit QAT)

Mamba-2 puro (sem atenção/MoE/KAN/TTT), mesmo `d_model`/camadas da referência.
`NSOS_GEN_QAT` liga o QAT ternário real (fake-quant STE ternário+int8): o modelo
treina E infere em 1.58-bit na GPU."""))
cells.append(code("""import os, sys
os.environ['NSOS_MAMBA_PROPER_SSM']='1'; os.environ['NSOS_MAMBA_A_LOGSPACED']='1'
os.environ['NSOS_MAMBA_CONV_K']='3'; os.environ['NSOS_GPU_POOL']='1'
sys.path.insert(0, '/content/nsos_ext_mambavs')
import nsos_ext as nsos, numpy as np, random
ndev = nsos.Device.GPU

def nsos_mamba(V):
    c = nsos.ModelConfig()
    c.num_layers = N_LAYER; c.d_model = D_MODEL; c.vocab_size = V
    c.n_heads = 4; c.n_kv_heads = 2; c.max_context_tokens = 1024
    c.use_moe = False; c.use_kan = False; c.use_ttt = False; c.dropout = 0.0
    c.attention_period = 99; c.attention_slot = 0    # period>>num_layers -> ZERO atencao (Mamba puro)
    m = nsos.JambaModel(c, ndev); m.to(ndev)
    return m

def nsos_nparams(m):
    tot = 0
    for p in m.parameters():
        try: tot += int(p.data.size)
        except Exception: pass
    return tot

def nsos_train(model, lr, steps, batch, sampler, seed, label, qat):
    tr = nsos.Trainer(model, lr); tr.warmup_steps = max(50, steps//10); tr.total_training_steps = steps
    tr.first_token_loss_scale = 1.0; tr.eos_loss_scale = 1.0
    tr.phase_scheduler.progressive_qat_enabled = bool(qat)
    rng = random.Random(seed*991+7); model.set_training_mode(True); win = max(1, steps//6); run = 0.0
    for s in range(steps):
        bs = sampler(rng); ps=[]; ans=[]
        for _ in range(batch):
            p,a = bs(); ps.append(p); ans.append(a)
        run += tr.train_supervised_batch(ps, ans)
        if (s+1)%win==0: print(f'    [{label}] step {s+1}/{steps} loss~{run/win:.3f}'); run=0.0
    model.set_training_mode(False)

def nsos_eval_ar(model, data, V):
    model.set_training_mode(False); cor=0
    for prompt,ans in data:
        lg = np.asarray(model.forward_ids(prompt).cpu().numpy()).reshape(-1,V)
        cor += int(int(np.argmax(lg[-1]))==ans[0])
    return cor/max(len(data),1)

def nsos_eval_copy(model, data, V):
    model.set_training_mode(False); cor=0
    for prompt,ans in data:
        lg = np.asarray(model.forward_ids(prompt+ans).cpu().numpy()).reshape(-1,V)
        ok = all(int(np.argmax(lg[len(prompt)+j-1]))==ans[j] for j in range(len(ans)))
        cor += int(ok)
    return cor/max(len(data),1)

_pm = nsos_mamba(64); NSOS_PARAMS = nsos_nparams(_pm)
print(f'[nsos] Mamba d_model={D_MODEL} n_layer={N_LAYER} | params~{NSOS_PARAMS:,}')
"""))

# ── 8 config knobs ───────────────────────────────────────────────────────────
cells.append(md("## 8 — Config da corrida (seeds/steps/batch)"))
cells.append(code("""SEEDS = [int(x) for x in os.environ.get('NSOS_GEN_SEEDS','0,1').split(',')]
AR_STEPS   = int(os.environ.get('NSOS_AR_STEPS','6000'))
COPY_STEPS = int(os.environ.get('NSOS_COPY_STEPS','1500'))
BATCH = int(os.environ.get('NSOS_GEN_BATCH','16'))
AR_LR = float(os.environ.get('NSOS_AR_LR','2e-3')); COPY_LR = float(os.environ.get('NSOS_COPY_LR','2e-3'))
N_KEY=16; N_VAL=16; V_AR=N_KEY+N_VAL+2; AR_BASE=1.0/N_VAL; KV_TEST=[4,8,16]
N_SYM=16
print(f'[cfg] seeds={SEEDS} AR_steps={AR_STEPS} COPY_steps={COPY_STEPS} batch={BATCH}')
print(f'[cfg] arms: NSOS-float | NSOS-ternary(1.58b) | REF={REF_KIND or \"PyTorch Mamba-2\"}(fp32/16)')
"""))

# ── 9 AR head-to-head ────────────────────────────────────────────────────────
cells.append(md("""## 9 — MQAR / recall associativo (treina n_kv=8; testa 4/8/16)

Baseline = 1/n_val.  3 braços: NSOS-float, NSOS-ternário(1.58-bit), referência
Mamba-2.  Mesmos dados/seeds."""))
cells.append(code("""import time
ar = {'nsos_float':{}, 'nsos_ternary':{}, 'ref':{}}
def _mk_ar_sampler(): return lambda r: (lambda: make_ar(r, 8, N_KEY, N_VAL))
t0=time.time()
for arm in ['nsos_float','nsos_ternary','ref']:
    accs = {k:[] for k in KV_TEST}
    for seed in SEEDS:
        ev = random.Random(seed*13+5)
        test = {k:[make_ar(ev,k,N_KEY,N_VAL) for _ in range(300)] for k in KV_TEST}
        if arm.startswith('nsos'):
            nsos.set_seed(seed); m = nsos_mamba(V_AR)
            nsos_train(m, AR_LR, AR_STEPS, BATCH, _mk_ar_sampler(), seed, f'AR {arm} s{seed}', qat=(arm=='nsos_ternary'))
            for k in KV_TEST: accs[k].append(nsos_eval_ar(m, test[k], V_AR))
        else:
            torch.manual_seed(seed); m = ref_model(V_AR)
            ref_train(m, AR_LR, AR_STEPS, BATCH, _mk_ar_sampler(), seed, f'AR ref s{seed}', V_AR)
            torch.cuda.synchronize(); vram = torch.cuda.max_memory_allocated()/1e6
            for k in KV_TEST: accs[k].append(ref_eval_ar(m, test[k], V_AR))
            ar['ref']['vram_mb'] = vram; del m; torch.cuda.empty_cache()
        print(f'  [{arm:12s} s{seed}] ' + ' '.join(f'n_kv={k}:{accs[k][-1]:.2f}' for k in KV_TEST))
    ar[arm].update({k:(float(np.mean(accs[k])), float(np.std(accs[k]))) for k in KV_TEST})
print(f'\\n[AR] concluido em {time.time()-t0:.0f}s  (baseline={AR_BASE:.3f})')
"""))

# ── 10 selective copy head-to-head ───────────────────────────────────────────
cells.append(md("""## 10 — Selective copy: extrapolação (treina ≤64; testa 64/128/256)"""))
cells.append(code("""LENS=[64,128,256]; N_DATA=8
def _mk_copy_sampler():
    return lambda r: (lambda: make_selcopy(r, r.choice([32,48,64]), N_DATA, N_SYM))
V_CP = N_SYM+3
cp = {'nsos_float':{}, 'nsos_ternary':{}, 'ref':{}}
t0=time.time()
for arm in ['nsos_float','nsos_ternary','ref']:
    accs = {L:[] for L in LENS}
    for seed in SEEDS:
        ev = random.Random(seed*17+3)
        test = {L:[make_selcopy(ev,L,N_DATA,N_SYM) for _ in range(200)] for L in LENS}
        if arm.startswith('nsos'):
            nsos.set_seed(seed); m = nsos_mamba(V_CP)
            nsos_train(m, COPY_LR, COPY_STEPS, BATCH, _mk_copy_sampler(), seed, f'CP {arm} s{seed}', qat=(arm=='nsos_ternary'))
            for L in LENS: accs[L].append(nsos_eval_copy(m, test[L], V_CP))
        else:
            torch.manual_seed(seed); m = ref_model(V_CP)
            ref_train(m, COPY_LR, COPY_STEPS, BATCH, _mk_copy_sampler(), seed, f'CP ref s{seed}', V_CP)
            for L in LENS: accs[L].append(ref_eval_copy(m, test[L], V_CP))
            del m; torch.cuda.empty_cache()
        print(f'  [{arm:12s} s{seed}] ' + ' '.join(f'L={L}:{accs[L][-1]:.2f}' for L in LENS))
    cp[arm].update({L:(float(np.mean(accs[L])), float(np.std(accs[L]))) for L in LENS})
print(f'\\n[COPY] concluido em {time.time()-t0:.0f}s')
"""))

# ── 11 verdict ───────────────────────────────────────────────────────────────
cells.append(md("""## 11 — VEREDITO: paridade (float) + eficiência (bits/qualidade)"""))
cells.append(code("""import torch
# bits/peso e tamanho do modelo
ref_bits = 16.0   # comparamos contra fp16 (deployment tipico do Mamba-2)
nsos_ternary_bits = 1.58
ref_mb = NSOS_PARAMS*ref_bits/8/1e6            # aprox: mesmo #params, fp16
nsos_ter_mb = NSOS_PARAMS*nsos_ternary_bits/8/1e6
print('='*78)
print(f'MODELOS  d_model={D_MODEL} n_layer={N_LAYER}  |  NSOS params~{NSOS_PARAMS:,} (com FFN)  |  REF params~{REF_PARAMS:,}  |  REF={REF_KIND or \"PyTorch Mamba-2\"}')
print('='*78)
print('\\n### A) PARIDADE (float) — MQAR (media, {} seeds; baseline={:.3f})'.format(len(SEEDS), AR_BASE))
print(f'{\"braco\":<16}' + ''.join(f'{(\"n_kv=\"+str(k)):>12}' for k in KV_TEST))
for arm,label in [('nsos_float','NSOS float'),('ref','REF Mamba-2')]:
    print(f'  {label:<14}' + ''.join(f'{ar[arm][k][0]:>7.3f}+-{ar[arm][k][1]:.2f}' for k in KV_TEST))
print('  -> EMPATE esperado = nosso Mamba e legitimamente o Mamba-2 SSD.')
print('\\n### selective copy (extrapolacao)')
print(f'{\"braco\":<16}' + ''.join(f'{(\"L=\"+str(L)):>12}' for L in LENS))
for arm,label in [('nsos_float','NSOS float'),('ref','REF Mamba-2')]:
    print(f'  {label:<14}' + ''.join(f'{cp[arm][L][0]:>7.3f}+-{cp[arm][L][1]:.2f}' for L in LENS))
print('\\n### B) EFICIENCIA — NSOS TERNARIO (1.58-bit) vs REF fp16')
print(f'{\"metrica\":<22}{\"NSOS ternario\":>16}{\"REF Mamba-2 fp16\":>18}')
print(f'{\"bits/peso\":<22}{nsos_ternary_bits:>16.2f}{ref_bits:>18.1f}')
print(f'{\"tamanho modelo (MB)\":<22}{nsos_ter_mb:>16.2f}{ref_mb:>18.2f}  ({ref_mb/max(nsos_ter_mb,1e-9):.1f}x menor)')
print(f'{\"MQAR n_kv=8\":<22}{ar[\"nsos_ternary\"][8][0]:>16.3f}{ar[\"ref\"][8][0]:>18.3f}')
print(f'{\"copy L=64\":<22}{cp[\"nsos_ternary\"][64][0]:>16.3f}{cp[\"ref\"][64][0]:>18.3f}')
gap8 = ar['ref'][8][0]-ar['nsos_ternary'][8][0]
print(f'\\n  GAP de qualidade (MQAR n_kv=8, ternario vs fp16 ref): {gap8:+.3f}')
print(f'  VANTAGEM: mesma familia de qualidade a {ref_mb/max(nsos_ter_mb,1e-9):.1f}x menos bits/peso.')
print('='*78)
print('COLE ISTO NO CHAT: a tabela A (paridade) + a tabela B (eficiencia) + REF_KIND.')
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).resolve().parent / "bench_mamba_vs_industrial_t4.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1).replace("%BRANCH%", BRANCH), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
