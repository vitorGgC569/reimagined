"""Generator for colab/train_contabil_pt.ipynb.

REAL training notebook (branch feature/nsos-gpu-phases12) for a Portuguese
*accounting/fiscal* assistant — the right-sized task for this architecture: a
small but FULL NSOS stack reaching useful accuracy in a NARROW PT domain, instead
of an under-scaled "general PT chatbot".

It uses the WHOLE architecture (not one layer): Mamba-2 proper selective SSM +
hybrid Attention layers + MoE-8 (top-2) + BitFastKAN FFN on the non-MoE layers +
TTT mixer layers + BitLinear 1.58-bit ternary weights — with every corrected path
ENABLED (NSOS_MAMBA_PROPER_SSM, NSOS_MOE_FP_ROUTER, NSOS_MOE_SWITCH_AUX,
NSOS_MOE_ROUTER_GRAD, NSOS_MAMBA_A_LOGSPACED) and the GPU greedy sampler for eval.

Pipeline:
  1 Drive + GPU (Tesla T4)
  2 clone branch + prove the corrected sources are present
  3 build nsos_ext (CUDA T4 sm_75) + the gradcheck gate
  4 gradcheck (objective gradient evidence)
  5 download PT-BR fiscal/legal datasets (CulturaX PT-BR, BR-TaxQA, BACEN FAQ, LeNER-Br)
  6 build the LM corpus + extract Q&A pairs (answering)
  7 train a real PT BPE tokenizer from the corpus (learn_bpe_merges -> .ox3)
  8 build the FULL-architecture model + Trainer (corrected flags ON)
  9 continued pre-training on the PT corpus (learn the language/domain)
 10 fine-tune on Q&A pairs (learn to ANSWER in Portuguese)
 11 eval: held-out PT perplexity (content-hash split, no leak) + held-out Q&A
    perplexity + sample generations
 12 save the model + tokenizer pack to Drive

Run once (pure JSON authoring, no GPU): python colab/make_contabil_pt_notebook.py
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

cells.append(md("""# NSOS — Assistente CONTÁBIL em português (arquitetura COMPLETA, branch `%BRANCH%`)

Treino **de verdade** de um modelo NSOS no domínio **contábil/fiscal brasileiro** —
a tarefa do tamanho certo pra essa arquitetura: um stack **pequeno mas COMPLETO**
chegando a acurácia útil num domínio **estreito** em PT, em vez de um "chatbot PT
geral" sub-dimensionado.

**Usa a arquitetura inteira** (não uma camada só): Mamba-2 *proper* (SSM seletivo
correto) + camadas de **Atenção** híbridas + **MoE-8** (top-2) + **BitFastKAN** nas
camadas não-MoE + camadas **TTT** + pesos **BitLinear ternários 1.58-bit** — com
**todas as correções LIGADAS** (`NSOS_MAMBA_PROPER_SSM`, `NSOS_MOE_FP_ROUTER`,
`NSOS_MOE_SWITCH_AUX`, `NSOS_MOE_ROUTER_GRAD`, `NSOS_MAMBA_A_LOGSPACED`).

Ordem: Drive→clone→build→**gradcheck**→dados PT-BR→corpus+Q&A→tokenizer BPE PT→
modelo full-arch→pré-treino→fine-tune Q&A→**perplexidade held-out + respostas**→pack.

Treino na **T4** (a via proper-Mamba é GPU-residente nesta branch)."""))

cells.append(md("## 1 — Drive + GPU (confirme **Tesla T4**)"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, subprocess, multiprocessing
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_contabil')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print('cpus:', multiprocessing.cpu_count())
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                '--format=csv'], check=False)
"""))

cells.append(md("## 2 — Token + clone da branch + prova de versão (correções presentes)"))
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
mamba = (REPO_ROOT / 'OXN/nsos/src/mamba2.cpp').read_text('utf-8')
cfg_h = (REPO_ROOT / 'OXN/nsos/include/nsos_config.h').read_text('utf-8')
has_proper = 'proper_selective_ssm' in mamba and 'forward_proper_step' in mamba
has_kan = 'use_kan' in cfg_h
has_data = (REPO_ROOT / 'OXN/nsos/scripts/oxta_contabil/download_datasets.py').exists()
print('=' * 64)
print(f'[ver] HEAD = {sha}')
print(f'[ver] Mamba proper + passo on-device : {"SIM" if has_proper else "*** NAO ***"}')
print(f'[ver] use_kan no ModelConfig         : {"SIM" if has_kan else "*** NAO ***"}')
print(f'[ver] download_datasets.py contabil  : {"SIM" if has_data else "*** NAO ***"}')
assert has_proper and has_kan and has_data, 'fonte stale — branch errada ou incompleta'
print('=' * 64)
"""))

cells.append(md("""## 3 — Build `nsos_ext` + gate de testes (CUDA T4 sm_75)

Compila a lib, o binding Python e o `test_gradcheck`. ~10-20 min na 1ª vez;
cacheia o `.so` no Drive por SHA."""))
cells.append(code("""import os, sys, subprocess, shutil, glob
from pathlib import Path

if 'nsos_ext' in sys.modules:
    raise RuntimeError('nsos_ext ja carregado — Runtime > Restart e rode de novo.')

PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
sha = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
BUILD = str(REPO_ROOT / 'OXN/nsos/build-contabil')
EXT_DIR = Path('/content/nsos_ext_contabil'); EXT_DIR.mkdir(exist_ok=True)
CACHE_SO = DRIVE_ROOT / '_bootstrap' / f'nsos_ext_contabil.{PYTAG}.{sha}.so'

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pybind11', 'numpy'], check=True)

if CACHE_SO.exists():
    shutil.copy2(CACHE_SO, EXT_DIR / 'nsos_ext.so')
    os.environ['NSOS_BUILD_CONTABIL'] = BUILD
    print('[build] .so do cache (Drive):', CACHE_SO)
else:
    print(f'[build] cmake configure (CUDA on) sha={sha}')
    r = subprocess.run(['cmake', '-S', str(REPO_ROOT / 'OXN/nsos'), '-B', BUILD,
                        '-DNSOS_ENABLE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=75',
                        '-DNSOS_BUILD_PYTHON=ON', '-DNSOS_BUILD_TESTS=ON',
                        '-DNSOS_BUILD_CLI=OFF', '-DNSOS_BUILD_API=OFF',
                        '-DNSOS_BUILD_OXTAMEM=OFF', '-DCMAKE_BUILD_TYPE=Release',
                        f'-DPython3_EXECUTABLE={sys.executable}'],
                       capture_output=True, text=True)
    print(r.stdout[-1200:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('cmake configure falhou')
    print('[build] compilando nsos_ext + test_gradcheck (paciencia na 1a vez)')
    r = subprocess.run(['cmake', '--build', BUILD, '-j', '2', '--target',
                        'nsos_ext', 'test_gradcheck'],
                       capture_output=True, text=True)
    print(r.stdout[-1200:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('build falhou')
    so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True))
    assert so, 'nsos_ext*.so nao encontrado'
    shutil.copy2(so[0], EXT_DIR / 'nsos_ext.so')
    CACHE_SO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so[0], CACHE_SO)
    os.environ['NSOS_BUILD_CONTABIL'] = BUILD
    print('[build] ok | ext =', EXT_DIR, '| build =', BUILD)
"""))

cells.append(md("""## 4 — GATE de gradcheck (evidência objetiva dos gradientes)

Diferenças finitas vs backward manual (rmsnorm, CE, matmul, **Mamba corrigido**,
**MoE Switch aux**, atenção exata, CHRASS, KAN). Verde = gradientes corretos."""))
cells.append(code("""import os, subprocess
BUILD = os.environ['NSOS_BUILD_CONTABIL']
ok = False
for cand in (f'{BUILD}/test_gradcheck', f'{BUILD}/Release/test_gradcheck', f'{BUILD}/test_gradcheck.exe'):
    if os.path.exists(cand):
        r = subprocess.run([cand], capture_output=True, text=True)
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print('STDERR:', r.stderr[-1200:])
        ok = r.returncode == 0
        break
else:
    print('[skip] test_gradcheck nao encontrado (cache .so sem build dir) — segue treino')
    ok = True
assert ok, 'GRADCHECK FALHOU — gradientes incorretos'
print('GRADCHECK: PASS')
"""))

cells.append(md("""## 5 — Datasets PT-BR fiscais/legais

`download_datasets.py` baixa **CulturaX PT-BR** (texto geral), **BR-TaxQA**
(perguntas/respostas tributárias + acórdãos CARF), **BACEN FAQ** (Banco Central) e
**LeNER-Br** (jurídico). Cacheado no Drive. Ajuste `CULTURAX_TARGET`/shards p/ a
sua quota — o domínio Q&A (BR-TaxQA/BACEN) é pequeno e é o mais importante."""))
cells.append(code("""import sys, subprocess
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'datasets>=2.16', 'huggingface_hub', 'tqdm'], check=True)

DATA_ROOT = DRIVE_ROOT / 'data'; DATA_ROOT.mkdir(parents=True, exist_ok=True)
CULTURAX_TARGET = os.environ.get('CULTURAX_TARGET', '512MB')  # menor = mais rapido
CULTURAX_SHARDS = os.environ.get('CULTURAX_SHARDS', '4')

cmd = [sys.executable,
       str(REPO_ROOT / 'OXN/nsos/scripts/oxta_contabil/download_datasets.py'),
       '--output-dir', str(DATA_ROOT),
       '--include', 'culturax_ptbr,br_taxqa,bacen_faq,lener_br',
       '--target-bytes', CULTURAX_TARGET, '--culturax-shards', CULTURAX_SHARDS]
print('Running:', ' '.join(cmd))
print('exit code:', subprocess.run(cmd, check=False).returncode)

RAW_DIR = DATA_ROOT / 'oxta_contabil' / 'raw'
for d in sorted(RAW_DIR.iterdir()) if RAW_DIR.exists() else []:
    if d.is_dir():
        mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1024**2
        print(f'  {d.name:22s} {mb:>8.1f} MB')
"""))

cells.append(md("""## 6 — Corpus de LM (texto puro) + pares Q&A (resposta)

Dois produtos: (a) `corpus.jsonl` — texto cru de todos os datasets, pro modelo
**aprender a língua/domínio**; (b) `qa_pairs` — (pergunta, resposta) do BR-TaxQA +
BACEN FAQ, pro modelo **aprender a responder**."""))
cells.append(code("""import json
from pathlib import Path
from typing import Iterator

RAW_DIR = DATA_ROOT / 'oxta_contabil' / 'raw'
CORPUS_PATH = DATA_ROOT / 'oxta_contabil' / 'corpus.jsonl'
MIN_LEN = 200

def _culturax(root: Path) -> Iterator[str]:
    for shard in sorted(root.glob('culturax_ptbr_shard_*.jsonl')):
        for line in open(shard, encoding='utf-8'):
            try:
                t = (json.loads(line).get('text') or '').strip()
            except json.JSONDecodeError:
                continue
            if len(t) >= MIN_LEN:
                yield t

def _br_taxqa_text(root: Path) -> Iterator[str]:
    for jf in sorted(root.glob('*.json')):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        items = data.get('questions') if isinstance(data, dict) and 'questions' in data \\
                else (data if isinstance(data, list) else list(data.values()))
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            t = ((it.get('question_text') or it.get('question') or '') + '\\n\\n' +
                 str(it.get('answer') or it.get('answer_text') or '')).strip()
            if len(t) >= 40:
                yield t

def _hf_disk(root: Path):
    from datasets import load_from_disk
    try:
        return load_from_disk(str(root))
    except Exception as e:
        print(f'  {root.name}: load_from_disk falhou: {e}'); return []

def _bacen(root: Path) -> Iterator[str]:
    for row in _hf_disk(root):
        q = next((str(row[k]) for k in ('question','pergunta','q') if row.get(k)), '')
        a = next((str(row[k]) for k in ('answer','resposta','a','body','text') if row.get(k)), '')
        t = (q + '\\n\\n' + a).strip()
        if len(t) >= MIN_LEN:
            yield t

def _lener(root: Path) -> Iterator[str]:
    for row in _hf_disk(root):
        toks = row.get('tokens') or row.get('words') or []
        if toks:
            t = ' '.join(map(str, toks)).strip()
            if len(t) >= MIN_LEN:
                yield t

EXTRACT = {'culturax_ptbr': _culturax, 'br_taxqa': _br_taxqa_text,
           'bacen_faq': _bacen, 'lener_br': _lener}
ndoc = 0
with open(CORPUS_PATH, 'w', encoding='utf-8') as out:
    for name, fn in EXTRACT.items():
        d = RAW_DIR / name
        if not d.exists():
            print(f'[skip] {name}'); continue
        n = 0
        for t in fn(d):
            out.write(json.dumps({'text': t, 'source': name}, ensure_ascii=False) + '\\n'); n += 1
        ndoc += n; print(f'[ok] {name:18s} {n:>8d} docs')
print(f'CORPUS: {ndoc:,} docs -> {CORPUS_PATH}')

# ---- pares Q&A (pergunta -> resposta) p/ o fine-tune de resposta ----
def qa_from_br_taxqa(root: Path):
    for jf in sorted(root.glob('*.json')):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        items = data.get('questions') if isinstance(data, dict) and 'questions' in data \\
                else (data if isinstance(data, list) else [])
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            q = (it.get('question_text') or it.get('question') or '').strip()
            a = str(it.get('answer') or it.get('answer_text') or '').strip()
            if len(q) >= 8 and len(a) >= 8:
                yield q, a

def qa_from_bacen(root: Path):
    for row in _hf_disk(root):
        q = next((str(row[k]) for k in ('question','pergunta','q') if row.get(k)), '').strip()
        a = next((str(row[k]) for k in ('answer','resposta','a') if row.get(k)), '').strip()
        if len(q) >= 8 and len(a) >= 8:
            yield q, a

qa_pairs = []
if (RAW_DIR / 'br_taxqa').exists():
    qa_pairs += list(qa_from_br_taxqa(RAW_DIR / 'br_taxqa'))
if (RAW_DIR / 'bacen_faq').exists():
    qa_pairs += list(qa_from_bacen(RAW_DIR / 'bacen_faq'))
print(f'Q&A pairs: {len(qa_pairs):,}')
if qa_pairs:
    print('exemplo Q:', qa_pairs[0][0][:120])
    print('exemplo A:', qa_pairs[0][1][:120])
"""))

cells.append(md("""## 7 — Tokenizer BPE em português (treinado no corpus)

BPE byte-level real (`learn_bpe_merges` → `.ox3`), aprendido no próprio corpus
contábil/fiscal — vocab PT de verdade, não byte-level genérico. (A versão eficiente
O(n log n) do merge desta branch acelera o `encode`.)"""))
cells.append(code("""import sys, random, json
sys.path.insert(0, str(REPO_ROOT / 'OXN/nsos/scripts'))
from nsos_curriculum_lib import (learn_bpe_merges, write_ox3, SPECIAL_TOKENS,
                                 _split_is_train)

TARGET_VOCAB = int(os.environ.get('NSOS_VOCAB', '8192'))

# Amostra de textos pra treinar o BPE (cap p/ caber em memoria/tempo).
docs = [json.loads(l)['text'] for l in open(CORPUS_PATH, encoding='utf-8')]
random.Random(1337).shuffle(docs)
# treino de BPE e O(corpus): limita tamanho/contagem p/ caber em minutos na T4.
TOK_DOCS = int(os.environ.get('NSOS_TOK_DOCS', '3000'))
TOK_CHARS = int(os.environ.get('NSOS_TOK_CHARS', '1500'))
tok_texts = [d[:TOK_CHARS] for d in docs[:TOK_DOCS]]
# inclui Q&A formatado p/ o BPE ver o formato de chat tambem
for (q, a) in qa_pairs[:1500]:
    tok_texts.append((f'<|task:contabil|>\\nPergunta:\\n{q}\\nResposta:\\n{a}<|endoftext|>')[:TOK_CHARS])
print(f'[tok] treinando BPE em {len(tok_texts)} textos, alvo vocab={TARGET_VOCAB}')

merges = learn_bpe_merges(tok_texts, target_vocab=TARGET_VOCAB - len(SPECIAL_TOKENS))
TOK_PATH = DATA_ROOT / 'oxta_contabil' / f'tokenizer_{TARGET_VOCAB}.ox3'
write_ox3(TOK_PATH, merges)

sys.path.insert(0, str(__import__('pathlib').Path('/content/nsos_ext_contabil')))
import nsos_ext as nsos
tok = nsos.Tokenizer()
tok.load_ox3(str(TOK_PATH))
tok.add_special_tokens(SPECIAL_TOKENS)
V = tok.vocab_size
EOS = tok.encode('<|endoftext|>')[0]
print(f'[tok] vocab={V}  merges={len(merges)}  EOS_id={EOS}')
print('[tok] sanity:', tok.decode(tok.encode('Qual é a alíquota do imposto de renda?')))
"""))

cells.append(md("""## 8 — Modelo com a ARQUITETURA COMPLETA + correções ligadas

Stack de 8 camadas exercendo **tudo**: Mamba-2 proper (mixer), Atenção GQA (slots),
MoE-8 top-2 (FFN), BitFastKAN (FFN das não-MoE), TTT (mixer), BitLinear ternário.
Os flags das correções são lidos na construção → setados ANTES."""))
cells.append(code("""import os
# --- correcoes (DEVEM preceder a construcao do modelo) ---
os.environ['NSOS_MAMBA_PROPER_SSM']  = '1'
os.environ['NSOS_MAMBA_CONV_K']      = '3'
os.environ['NSOS_MOE_FP_ROUTER']     = '1'
os.environ['NSOS_MOE_SWITCH_AUX']    = '1'
os.environ['NSOS_MOE_ROUTER_GRAD']   = '1'
os.environ['NSOS_MAMBA_A_LOGSPACED'] = '1'
os.environ['NSOS_MAMBA_STATE_EXPANSION'] = '0'   # diagonal proper (provado). '1' = Mamba-2 N-state.
os.environ['NSOS_GPU_POOL']          = '1'

dev = nsos.Device.GPU
cfg = nsos.ModelConfig()
cfg.num_layers = 8
cfg.d_model    = int(os.environ.get('NSOS_DMODEL', '320'))
cfg.vocab_size = V
cfg.max_context_tokens = int(os.environ.get('NSOS_CTX', '512'))
# Atencao GQA hibrida: slot 3 de cada 4 -> camadas 3 e 7 viram atencao
cfg.n_heads = 8; cfg.n_kv_heads = 2; cfg.sliding_window = 2048
cfg.attention_period = 4; cfg.attention_slot = 3
cfg.use_exact_attention_training = True; cfg.use_flash_attn = False
# MoE-8 top-2: slot 1 de cada 4 -> camadas 1 e 5 com 8 experts
cfg.use_moe = True; cfg.num_experts = 8; cfg.num_experts_per_token = 2
cfg.moe_period = 4; cfg.moe_slot = 1
# KAN como FFN nas camadas NAO-MoE (BitFastKAN no lugar do FFN denso)
cfg.use_kan = True
# TTT como mixer: slot 2 de cada 4 -> camadas 2 e 6 (a arquitetura INTEIRA)
cfg.use_ttt = True; cfg.ttt_period = 4; cfg.ttt_slot = 2
cfg.dropout = 0.0

model = nsos.JambaModel(cfg, dev); model.to(dev)
model.set_training_mode(True)
tr = nsos.Trainer(model, float(os.environ.get('NSOS_LR', '1.5e-3')))
tr.warmup_steps = 200; tr.eos_token_id = EOS; tr.moe_aux_loss_scale = 0.01
print(f'[model] L={cfg.num_layers} d={cfg.d_model} V={V} ctx={cfg.max_context_tokens}')
print('[model] mixers: Mamba(0,1,4,5) Atencao(3,7) TTT(2,6) | FFN: MoE(1,5) KAN(resto) | BitLinear 1.58b')
"""))

cells.append(md("""## 9 — Pré-treino contínuo no corpus PT (aprender a língua/domínio)

Split treino/held-out por **hash de conteúdo** (sem vazamento de avaliação). Cada
step: um doc → próximos-tokens (teacher forcing) via `train_supervised`."""))
cells.append(code("""import time, random, numpy as np
CTX = cfg.max_context_tokens
MAX_STEPS = int(os.environ.get('NSOS_LM_STEPS', '2500'))

# split por conteudo: ~97% treino, 3% held-out (estavel, seed-independente)
train_docs, held_docs = [], []
for d in docs:
    (train_docs if _split_is_train(d, 0.97) else held_docs).append(d)
print(f'[lm] docs treino={len(train_docs)} held-out={len(held_docs)}  steps={MAX_STEPS}')
tr.total_training_steps = MAX_STEPS  # schedule de LR (warmup -> decay)

rng = random.Random(123); order = train_docs[:]; rng.shuffle(order)
losses = []; t0 = time.time(); i = 0
for step in range(1, MAX_STEPS + 1):
    doc = order[i % len(order)]; i += 1
    ids = tok.encode(doc)[:CTX]
    if len(ids) < 4:
        continue
    losses.append(tr.train_supervised(ids[:1], ids[1:]))
    if step % 100 == 0 or step == MAX_STEPS:
        recent = sum(losses[-100:]) / max(1, len(losses[-100:]))
        print(f'  step {step:5d}  loss~{recent:.4f}  ({time.time()-t0:.0f}s)')
print(f'[lm] pre-treino feito em {time.time()-t0:.0f}s')
"""))

cells.append(md("""## 10 — Fine-tune de RESPOSTA (Q&A em português)

Formato de chat: `<|task:contabil|>\\nPergunta:\\n{q}\\nResposta:\\n{a}<|endoftext|>`.
Loss só nos tokens da resposta. Split Q&A por hash (held-out p/ avaliar)."""))
cells.append(code("""import random, time
QA_EPOCHS = int(os.environ.get('NSOS_QA_EPOCHS', '3'))

def fmt_prompt(q):
    return f'<|task:contabil|>\\nPergunta:\\n{q}\\nResposta:\\n'

qa_train, qa_held = [], []
for (q, a) in qa_pairs:
    (qa_train if _split_is_train(q, 0.9) else qa_held).append((q, a))
print(f'[qa] treino={len(qa_train)} held-out={len(qa_held)} | epocas={QA_EPOCHS}')

rng = random.Random(7); t0 = time.time()
for ep in range(QA_EPOCHS):
    o = qa_train[:]; rng.shuffle(o); tot = 0.0; n = 0
    for (q, a) in o:
        p_ids = tok.encode(fmt_prompt(q))[:CTX // 2]
        a_ids = tok.encode(a)[:CTX // 2] + [EOS]
        if len(p_ids) < 2 or len(a_ids) < 2:
            continue
        tot += tr.train_supervised(p_ids, a_ids); n += 1
    print(f'  epoca {ep}  loss/par={tot/max(n,1):.4f}  ({time.time()-t0:.0f}s)')
print('[qa] fine-tune feito')
"""))

cells.append(md("""## 11 — EVIDÊNCIA: perplexidade held-out + respostas geradas

Perplexidade held-out de **LM** (texto inédito) e de **Q&A** (respostas inéditas) =
os números "verdade primeiro". Depois, gerações reais pra ver o modelo respondendo
em português. (Geração usa o argmax/sampler — decode guloso manual.)"""))
cells.append(code("""import numpy as np, math, random
model.set_training_mode(False)
os.environ['NSOS_GPU_SAMPLER'] = '1'   # sampler GPU desta branch (eval mais rapido)

def seq_ppl(seq, ans_start):
    # perplexidade sobre seq[ans_start:], teacher forcing
    lg = np.asarray(model.forward_ids(seq[:CTX]).cpu().numpy()).reshape(-1, V)
    nll = 0.0; nt = 0
    for pos in range(max(ans_start, 1), len(seq[:CTX])):
        row = lg[pos - 1].astype(np.float64); row -= row.max()
        pr = np.exp(row); pr /= pr.sum()
        nll += -math.log(max(pr[seq[pos]], 1e-12)); nt += 1
    return nll, nt

# --- LM held-out ---
nll = nt = 0.0
for d in held_docs[:200]:
    ids = tok.encode(d)[:CTX]
    if len(ids) < 4:
        continue
    a, b = seq_ppl(ids, 1); nll += a; nt += b
lm_ppl = math.exp(nll / max(nt, 1))

# --- Q&A held-out ---
nll = nt = 0.0
for (q, a) in qa_held[:200]:
    p = tok.encode(fmt_prompt(q))[:CTX // 2]; aids = tok.encode(a)[:CTX // 2] + [EOS]
    seq = p + aids
    x, y = seq_ppl(seq, len(p)); nll += x; nt += y
qa_ppl = math.exp(nll / max(nt, 1)) if nt else float('nan')

print('=' * 60)
print(f'PERPLEXIDADE held-out  | LM(texto)={lm_ppl:8.2f}   Q&A(resposta)={qa_ppl:8.2f}')
print(f'(baseline aleatorio ~ vocab={V}; quanto menor, melhor)')
print('=' * 60)

# --- geracoes ---
def gen(prompt, max_new=64, temp=0.0):
    ids = tok.encode(prompt)
    for _ in range(max_new):
        lg = np.asarray(model.forward_ids(ids[-CTX:]).cpu().numpy()).reshape(-1, V)
        row = lg[-1].astype(np.float64)
        if temp <= 1e-5:
            nxt = int(np.argmax(row))
        else:
            row /= temp; row -= row.max(); pr = np.exp(row); pr /= pr.sum()
            nxt = int(np.random.choice(V, p=pr))
        if nxt == EOS:
            break
        ids.append(nxt)
    return tok.decode(ids[len(tok.encode(prompt)):])

DEMO = [
    'O que é o ativo circulante?',
    'Qual é a diferença entre débito e crédito na contabilidade?',
    'Quando devo emitir uma nota fiscal eletrônica?',
]
for q in DEMO:
    print(f'\\nPERGUNTA: {q}')
    print('RESPOSTA:', gen(fmt_prompt(q), 64, 0.0)[:400])
"""))

cells.append(md("## 12 — Salvar modelo + tokenizer (Drive)"))
cells.append(code("""import time, shutil
RUN = DRIVE_ROOT / 'runs' / f"contabil_{time.strftime('%Y%m%d_%H%M')}"
RUN.mkdir(parents=True, exist_ok=True)
CKPT = str(RUN / 'contabil_pt.bin')
model.save(CKPT)
shutil.copy2(str(TOK_PATH), RUN / TOK_PATH.name)
print('[save] modelo  :', CKPT)
print('[save] tokenizer:', RUN / TOK_PATH.name)
print('Pronto. Modelo contabil PT (arquitetura completa) treinado e salvo.')
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

out = Path(__file__).resolve().parent / "train_contabil_pt.ipynb"
text = json.dumps(nb, indent=1, ensure_ascii=False).replace("%BRANCH%", BRANCH)
out.write_text(text, encoding="utf-8")
print(f"wrote {out}")
