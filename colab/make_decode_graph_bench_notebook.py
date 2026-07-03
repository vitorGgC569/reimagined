"""Generator for colab/bench_decode_graph_t4.ipynb.

CUDA-graph decode validation + D2H measurement on the Tesla T4 (branch
feature/nsos-gpu-phases12 / alias nsos-gpu-phases12).

What this notebook proves/measures (all GPU-runtime properties that no local
build can establish — the local GPU is sm_61, incompatible with the CUDA 13
toolchain):

  1. The FULL existing gate suite (gradcheck + GPU parity) stays green under
     the NSOS_CUDA_PTDS build (per-thread default stream — a global stream-
     semantics change, so the whole suite must pass under it, not just the new
     test).
  2. test_gpu_parity_decode_graph: graph decode token sequence == eager token
     sequence (the frozen-host-argument failure class).
  3. tok/s: eager decode vs CUDA-graph decode on the same tiny hybrid model
     (launch-overhead amortization is the whole point — hundreds of per-token
     kernel launches collapse into one cudaGraphLaunch).
  4. nsos.bench_d2h_copy: measured per-copy latency of the per-token D2H
     through PAGEABLE vs PINNED staging (the number behind the pinned staging
     in GpuGreedySampler, commit d8cad0d).

Run once (pure JSON authoring, no GPU): python colab/make_decode_graph_bench_notebook.py
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

cells.append(md("""# NSOS — CUDA-graph decode + medição D2H (T4)

Valida no T4 o decode via **CUDA graph** (`NSOS_CUDA_GRAPH_DECODE=1`, build
`NSOS_CUDA_PTDS=ON`) e mede o **D2H por-token** (pinned vs pageable).

Ordem: (1) GPU/Drive → (2) clone+prova de versão → (3) build PTDS + TODOS os
gates (gradcheck + paridades + o novo `test_gpu_parity_decode_graph`) →
(4) bench D2H → (5) tok/s eager vs graph no mesmo modelo.

**Interpretação honesta:** o build PTDS muda a semântica de stream globalmente;
por isso o gate roda a suíte INTEIRA sob PTDS, não só o teste novo.  Se
qualquer gate falhar, os números de tok/s abaixo NÃO valem."""))

cells.append(md("## 1 — GPU + Drive (cache do build)"))
cells.append(code("""import os, subprocess
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_decode_graph'); DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print('[drive] indisponivel (ok):', e); DRIVE_ROOT = Path('/content/nsos_decode_graph'); DRIVE_ROOT.mkdir(exist_ok=True)
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
jamba_h = (REPO_ROOT/'OXN/nsos/include/jamba.h').read_text('utf-8')
cmakel  = (REPO_ROOT/'OXN/nsos/CMakeLists.txt').read_text('utf-8')
kern_h  = (REPO_ROOT/'OXN/nsos/include/cuda/kernels.cuh').read_text('utf-8')
assert 'forward_ids_decode_graph' in jamba_h, 'fonte stale: falta o decode via CUDA graph'
assert 'NSOS_CUDA_PTDS' in cmakel, 'fonte stale: falta a opcao de build PTDS'
assert 'nsos_bench_d2h_copy' in kern_h, 'fonte stale: falta o bench D2H'
print(f'[ver] HEAD={sha} | decode-graph + PTDS + bench D2H: OK')
"""))

cells.append(md("""## 3 — Build **PTDS** sm_75 + gates (suíte inteira sob PTDS)

`-DNSOS_CUDA_PTDS=ON` muda a stream default p/ per-thread (pré-requisito da
captura).  Por ser mudança global de semântica, TODOS os gates rodam sob ela."""))
cells.append(code("""import os, sys, subprocess, shutil, glob
from pathlib import Path
if 'nsos_ext' in sys.modules:
    raise RuntimeError('nsos_ext ja carregado — Runtime > Restart.')
PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
sha = subprocess.run(['git','-C',str(REPO_ROOT),'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
BUILD = str(REPO_ROOT / 'OXN/nsos/build-decode-graph')
EXT_DIR = Path('/content/nsos_ext_dgraph'); EXT_DIR.mkdir(exist_ok=True)
CACHE_SO = DRIVE_ROOT / '_bootstrap' / f'nsos_ext_dgraph.{PYTAG}.{sha}.so'
subprocess.run([sys.executable,'-m','pip','install','-q','pybind11','numpy'], check=True)
GATE_TESTS = ['test_gradcheck',
              'test_gpu_parity_jamba',
              'test_gpu_parity_mamba_proper',
              'test_gpu_parity_mamba_proper_stream',
              'test_gpu_parity_mamba_nstate_stream',
              'test_gpu_parity_moe_router',
              'test_gpu_parity_decode_graph']
need_build = not CACHE_SO.exists()
if not need_build:
    shutil.copy2(CACHE_SO, EXT_DIR/'nsos_ext.so')
    # gates precisam dos binarios: se o build dir nao existe, rebuild mesmo com cache
    need_build = not Path(BUILD).exists()
    if not need_build: print('[build] .so do cache:', CACHE_SO)
if need_build:
    r = subprocess.run(['cmake','-S',str(REPO_ROOT/'OXN/nsos'),'-B',BUILD,
                        '-DNSOS_ENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=75',
                        '-DNSOS_CUDA_PTDS=ON',
                        '-DNSOS_BUILD_PYTHON=ON','-DNSOS_BUILD_TESTS=ON',
                        '-DNSOS_BUILD_CLI=OFF','-DNSOS_BUILD_API=OFF','-DNSOS_BUILD_OXTAMEM=OFF',
                        '-DCMAKE_BUILD_TYPE=Release',f'-DPython3_EXECUTABLE={sys.executable}'],
                       capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('cmake configure falhou')
    r = subprocess.run(['cmake','--build',BUILD,'-j','2','--target','nsos_ext',*GATE_TESTS],
                       capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-3000:]); raise RuntimeError('build falhou')
    so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True)); assert so
    shutil.copy2(so[0], EXT_DIR/'nsos_ext.so'); CACHE_SO.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(so[0], CACHE_SO)
    print('[build] ok (PTDS ON)')
os.environ['NSOS_BUILD_DGRAPH'] = BUILD
GATE_FAIL = False
for name in GATE_TESTS:
    found = False
    for cand in (f'{BUILD}/{name}', f'{BUILD}/Release/{name}', f'{BUILD}/{name}.exe'):
        if os.path.exists(cand):
            found = True
            rr = subprocess.run([cand], capture_output=True, text=True)
            ok = rr.returncode == 0
            GATE_FAIL = GATE_FAIL or not ok
            tail = (rr.stdout[-1200:] + rr.stderr[-1200:]) if not ok else rr.stdout.strip().splitlines()[-1]
            print(f'[{name}]', 'PASS' if ok else f'FAIL (rc={rr.returncode})\\n{tail}')
            break
    if not found:
        GATE_FAIL = True
        print(f'[{name}] binario nao encontrado')
if GATE_FAIL:
    raise RuntimeError('GATES FALHARAM sob PTDS — nao medir tok/s; reporte a saida acima')
print('\\nTODOS OS GATES PASS sob PTDS — medicoes abaixo valem.')
"""))

cells.append(md("""## 4 — Medição D2H por-token: pageable vs pinned

O decode move 1 int/token device→host.  `bench_d2h_copy` mede a latência média
das duas vias de staging — o número que justifica (ou não) o staging pinned do
`GpuGreedySampler` (commit d8cad0d)."""))
cells.append(code("""import sys
sys.path.insert(0, '/content/nsos_ext_dgraph')
import nsos_ext as nsos
print('cuda_ptds_build =', nsos.cuda_ptds_build())
assert nsos.fast_gpu_supported(), 'GPU indisponivel'
res = nsos.bench_d2h_copy(4000)
print(f"D2H 4 bytes x {res['iters']}: pageable={res['pageable_us']:.2f} us/copia | "
      f"pinned={res['pinned_us']:.2f} us/copia | "
      f"ganho={res['pageable_us']/max(res['pinned_us'],1e-9):.2f}x")
"""))

cells.append(md("""## 5 — tok/s: decode eager vs CUDA graph (ARQUITETURA COMPLETA)

Modelo híbrido GPU-first: Mamba-2 **N-state** (passo device fundido) + atenção
GQA (camadas 2/4) + **MoE-4 top-2** (caminho denso single-row device — pesos
top-k mascarados lidos na GPU, zero D2H por token).  A seleção do token é o
MESMO argmax host nos dois braços — o que muda é só o forward por token
(cascata de launches vs 1 `cudaGraphLaunch`).  Paridade de tokens é verificada
in-loco além do gate C++."""))
cells.append(code("""import os, time
import numpy as np
# gates de env ANTES de qualquer uso do modelo (cacheiam na 1a leitura)
os.environ['NSOS_MAMBA_GPU_STEP'] = '1'
os.environ['NSOS_CUDA_GRAPH_DECODE'] = '1'
import nsos_ext as nsos

nsos.set_seed(1234)
c = nsos.ModelConfig()
c.num_layers = 4; c.d_model = 128; c.vocab_size = 512
c.n_heads = 4; c.n_kv_heads = 2
c.attention_period = 2; c.attention_slot = 1
# ARQUITETURA COMPLETA sob o graph (GPU-first): N-state Mamba (default do
# config) via passo device fundido + MoE via caminho denso single-row
# device-resident (pesos top-k mascarados lidos NA GPU).
c.use_moe = True; c.num_experts = 4; c.num_experts_per_token = 2
c.moe_period = 2; c.moe_slot = 0
c.use_ttt = False
c.max_context_tokens = 1024; c.use_cuda = True
model = nsos.JambaModel(c, nsos.Device.GPU); model.to(nsos.Device.GPU)
model.set_training_mode(False)

PROMPT = [7, 21, 93, 402, 11, 88, 300, 5]
DECODE = 256
RESERVE = len(PROMPT) + DECODE + 8

def greedy(logits):
    arr = logits.cpu().numpy()
    row = arr.reshape(-1, arr.shape[-1])[-1]
    return int(row.argmax())

def run(use_graph):
    model.reset_session()
    model.set_streaming_inference(True)
    model.reserve_kv_cache(RESERVE, nsos.Device.GPU, 1)
    logits = model.forward_ids(PROMPT, None)
    toks = []
    tok = greedy(logits)
    t_steady = None
    for s in range(DECODE):
        toks.append(tok)
        if use_graph:
            g = model.forward_ids_decode_graph(tok)
            logits = g if g.size > 0 else model.forward_ids([tok], None)
        else:
            logits = model.forward_ids([tok], None)
        if s == 7:  # descarta warm-up + captura (graph) / aquecimento (eager)
            t_steady = time.perf_counter()
        tok = greedy(logits)
    elapsed = time.perf_counter() - t_steady
    status = model.decode_graph_status()
    model.set_streaming_inference(False)
    return toks, (DECODE - 8) / elapsed, status

toks_eager, tps_eager, _ = run(False)
toks_graph, tps_graph, status = run(True)
assert toks_eager == toks_graph, f'PARIDADE QUEBROU: eager={toks_eager[:12]} graph={toks_graph[:12]}'
print(f'status do graph : {status}')
print(f'tokens          : {len(toks_graph)} identicos nos 2 bracos')
print(f'eager           : {tps_eager:8.1f} tok/s')
print(f'graph           : {tps_graph:8.1f} tok/s')
print(f'SPEEDUP         : {tps_graph/tps_eager:8.2f}x')
if status != 'active':
    print('ATENCAO: graph nao ativou — speedup acima nao mede o graph; status explica o porque')
"""))

cells.append(md("""## 6 — O que reportar de volta

Cole no chat: (a) a linha de cada gate da célula 3; (b) as latências D2H da
célula 4; (c) `status`, `tok/s` e `SPEEDUP` da célula 5.  Se algum gate falhar
sob PTDS, isso é O achado (semântica de stream) — não seguir para os benches."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "bench_decode_graph_t4.ipynb"
text = json.dumps(nb, ensure_ascii=False, indent=1).replace("%BRANCH%", BRANCH)
out.write_text(text, encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
