"""Generator for colab/train_gpu_phases_t4.ipynb.

Emits a clean, valid Colab notebook that runs REAL training on a T4 from the
feature/nsos-gpu-phases12 branch (where the GPU caching pool ~3.9x, BF16-runtime,
fused grad-norm and the opt-in parallel scan live).  Reuses the proven
colab_bootstrap + train_curriculum flow from train_v11.ipynb, with two changes:
  * authenticate with the user's GITHUB_TOKEN secret (GH_TOKEN as fallback);
  * clone the feature branch (it is not on main yet).
Run once: `python colab/make_t4_notebook.py` (pure JSON authoring, no GPU).
"""
import json
from pathlib import Path

# Slash-free branch name: Colab's /github/<owner>/<repo>/blob/<branch>/<path>
# URL resolver mis-parses branch names that contain "/", so we publish + open
# from a slash-free alias of feature/nsos-gpu-phases12.
BRANCH = "nsos-gpu-phases12"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = []

cells.append(md(f"""# NSOS — Treino GPU-first na T4 (branch `{BRANCH}`)

Roda **treino real** na T4 com as otimizações GPU-first desta branch:
- **GPU caching allocator** (pool, ~3.9× tok/s medido) — ligado por padrão.
- **BF16 mixed-precision** (Tensor Cores da T4, sm_75) — `NSOS_MIXED_PRECISION=bf16`.
- fused grad-norm, copy stream, GIL release, optimizer na GPU.
- Mamba2 **parallel-prefix scan** opt-in (`NSOS_MAMBA_PARALLEL_SCAN`) — **default OFF**;
  validar paridade 1e-4 (Célula 7) antes de ligar.

Ordem: Drive → token → clone da branch → build (cacheado por SHA) → dataset bundle
→ treino com checkpoint no Drive. Em timeout, reabra e re-rode a célula de treino:
ela retoma do último checkpoint."""))

cells.append(md("## 1 — Montar Drive + checar GPU (confirme **Tesla T4**)"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, subprocess
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_v11')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

print('---  GPU  ---')
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                '--format=csv'], check=False)
"""))

cells.append(md("""## 2 — Token + clone da branch

Usa o secret **`GITHUB_TOKEN`** (ícone de chave na barra lateral → "Notebook access" ON).
Cai para `GH_TOKEN` se existir. Clona exatamente a branch `%s`.""" % BRANCH))
cells.append(code(f"""# Auto-suficiente: roda mesmo se a celula 1 nao tiver rodado neste runtime.
import os, subprocess
from pathlib import Path

REPO_URL = 'https://github.com/vitorGgC569/reimagined.git'
REPO_ROOT = Path('/content/reimagined')
BRANCH = '{BRANCH}'

TOKEN = None
try:
    from google.colab import userdata
    for key in ('GITHUB_TOKEN', 'GH_TOKEN'):
        try:
            TOKEN = userdata.get(key)
        except Exception:
            TOKEN = None
        if TOKEN:
            print(f'[auth] token de {{key}} carregado ({{TOKEN[:6]}}…, len {{len(TOKEN)}})')
            break
    if not TOKEN:
        print('[auth] AVISO: nenhum secret GITHUB_TOKEN/GH_TOKEN acessível a este notebook')
except Exception as e:
    print(f'[auth] AVISO: userdata indisponível: {{e}}')

def gh_url():
    if TOKEN and 'github.com' in REPO_URL:
        return REPO_URL.replace('https://', f'https://oauth2:{{TOKEN}}@')
    return REPO_URL

if REPO_ROOT.exists():
    print(f'[git] repo existe — fetch+checkout {{BRANCH}}')
    subprocess.run(['git', '-C', str(REPO_ROOT), 'remote', 'set-url', 'origin', gh_url()], check=True)
    subprocess.run(['git', '-C', str(REPO_ROOT), 'fetch', '--depth', '1', 'origin', BRANCH], check=True)
    subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', BRANCH], check=True)
    subprocess.run(['git', '-C', str(REPO_ROOT), 'reset', '--hard', f'origin/{{BRANCH}}'], check=True)
else:
    print(f'[git] clone --branch {{BRANCH}} (depth 1)')
    r = subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, gh_url(), str(REPO_ROOT)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('[git] STDERR:', r.stderr)
        raise RuntimeError('git clone falhou — confira o GITHUB_TOKEN (Contents: Read) e o "Notebook access".')
    print('[git] clone ok')

os.environ['REPO_ROOT'] = str(REPO_ROOT)
os.environ['DRIVE_ROOT'] = str(DRIVE_ROOT)

# ── PROVA DE VERSÃO (anti binario/fonte stale) ─────────────────────────────
sha = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
msg = subprocess.run(['git', '-C', str(REPO_ROOT), 'log', '-1', '--format=%s'],
                     capture_output=True, text=True).stdout.strip()
print('=' * 70)
print(f'[git] >>> HEAD em uso: {{sha}}  ({{msg[:60]}})')
marker = (REPO_ROOT / 'OXN/nsos/src/mamba2.cpp').read_text(encoding='utf-8').count('NSOS_MAMBA_A_LOGSPACED')
print(f'[git] >>> fix A-logspaced no fonte clonado: {{"SIM" if marker > 0 else "*** NAO — FONTE ANTIGO ***"}} ({{marker}} refs)')
print('[git] >>> a celula 3 deve mostrar "cache MISS/HIT for sha=' + sha + '..." — se o sha')
print('[git] >>> divergir, o .so e de outro commit. E o boot do treino imprime')
print('[git] >>> "[boot] mamba A spectrum ... fix ATIVO/INATIVO" como prova final de runtime.')
print('=' * 70)
"""))

cells.append(md("""## 3 — Build do `nsos_ext` (CUDA, arch da T4 = sm_75)

Força rebuild (as mudanças C++ desta branch precisam recompilar) e usa o
`colab_bootstrap` para detectar a GPU, escolher o perfil e cachear o `.so` no Drive."""))
cells.append(code("""# Força recompilar com os commits C++ desta branch (pool, BF16, scan, loss-path).
# CRÍTICO: limpa o cache no Drive E o build-colab local. Sem o rm do build-colab,
# um .so de um pull anterior na MESMA sessão fica STALE e o setup_environment o
# reusa -> os fixes novos não entram no binário (o step-time não muda apesar dos
# commits). rm -rf garante recompilação a partir do código que a célula 2 puxou.
!rm -rf /content/reimagined/OXN/nsos/build-colab
!rm -f /content/drive/MyDrive/nsos_v11/_bootstrap/*.so.* 2>/dev/null || true

import sys, os
sys.path.insert(0, str(REPO_ROOT / 'colab'))
os.environ['NSOS_PARAM_SCALE'] = '40m'      # Chinchilla-ótimo p/ o budget; ~2x mais rápido
os.environ['NSOS_TRAIN_CHUNK_SIZE'] = '0'   # full batch (rápido na T4; chunk=2 é só p/ 1050 Ti)

from colab_bootstrap import setup_environment
info = setup_environment(verbose=True)
print('[build] profile =', info['profile'], '| build_dir =', info['build_dir'])
"""))

cells.append(md("""## 4 — Dataset bundle v11 (cacheado no Drive)

Primeira sessão baixa + monta (~1-3h, depende do HuggingFace); sessões seguintes
extraem do cache em <30s."""))
cells.append(code("""import os, sys, shutil, subprocess, json
from pathlib import Path

V11_BUNDLE = REPO_ROOT / 'OXN/nsos/scripts/distillation_bundle_v11'
RAW_DRIVE = DRIVE_ROOT / '_datasets_raw'
PROCESSED_DRIVE = DRIVE_ROOT / '_datasets_processed'
RAW_REPO_LINK = REPO_ROOT / 'OXN/nsos/artifacts/real_datasets'
PROCESSED_DRIVE.mkdir(parents=True, exist_ok=True); RAW_DRIVE.mkdir(parents=True, exist_ok=True)

cached = PROCESSED_DRIVE / 'distillation_bundle_v11.zip'
if cached.exists():
    print(f'[data] cache HIT: {cached}')
    if V11_BUNDLE.exists(): shutil.rmtree(V11_BUNDLE)
    V11_BUNDLE.mkdir(parents=True)
    subprocess.run(['unzip', '-q', str(cached), '-d', str(V11_BUNDLE)], check=True)
else:
    print('[data] sem cache — fetch + build (lento na 1a vez)')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'duckdb', 'huggingface_hub', 'requests'], check=True)
    RAW_REPO_LINK.parent.mkdir(parents=True, exist_ok=True)
    if RAW_REPO_LINK.is_symlink(): RAW_REPO_LINK.unlink()
    elif RAW_REPO_LINK.exists(): shutil.rmtree(RAW_REPO_LINK)
    os.symlink(str(RAW_DRIVE), str(RAW_REPO_LINK))
    r = subprocess.run([sys.executable, str(REPO_ROOT / 'OXN/nsos/scripts/fetch_real_datasets.py'),
                        '--out-dir', str(RAW_DRIVE), '--scale-factor', '10'], capture_output=True, text=True)
    print((r.stdout or '')[-2000:])
    if r.returncode != 0:
        print('STDERR:', (r.stderr or '')[-2000:]); raise RuntimeError('fetch_real_datasets falhou')
    r = subprocess.run([sys.executable, str(REPO_ROOT / 'OXN/nsos/scripts/build_curriculum.py'),
                        '--preset', 'v11'], capture_output=True, text=True)
    print((r.stdout or '')[-2000:])
    if r.returncode != 0:
        print('STDERR:', (r.stderr or '')[-2000:]); raise RuntimeError('build_curriculum falhou')
    shutil.make_archive(str(cached.with_suffix('')), 'zip', V11_BUNDLE)
    print(f'[data] cacheado -> {cached}')

manifest = json.loads((V11_BUNDLE / 'curriculum_manifest.json').read_text('utf-8'))
print('[data] train total:', sum(p['train_samples'] for p in manifest['phases']))
"""))

cells.append(md("""## 5 — TREINO (BF16 + pool, checkpoint no Drive a cada 100 steps)

Liga **BF16** (Tensor Cores T4) e o **pool** (default ON). O parallel-scan fica
**OFF** (precisa passar a paridade da Célula 7 antes). Em timeout, re-rode: retoma
do último checkpoint."""))
cells.append(code("""import time, os, sys, subprocess, shutil as _sh
from pathlib import Path

# "alog" = A log-espaçado (OXTA-CRIT Lei 2). Nome de run NOVO de propósito:
# retomar checkpoint de um run antigo carregaria o A=ones salvo e desfaria o fix.
RUN_NAME = os.environ.get('NSOS_RUN_NAME', f"t4_alog_{time.strftime('%Y%m%d')}")
RUN_DIR = DRIVE_ROOT / 'runs' / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Resume por MTIME sobre TODOS os .bin do run (inclui os checkpoints
# intra-fase {fase}_stepN.bin salvos a cada 100 steps — a versao anterior so
# achava fases COMPLETAS, entao um disconnect no meio da fase resetava do zero).
# Semantica honesta: --resume-model e WARM-START (carrega pesos; o curriculo
# re-roda as fases com pesos quentes — granularidade de fase, nao de step).
def latest_ckpt(rd):
    cands = [f for f in rd.glob('*.bin') if f.name != 'audit_reload_probe.bin']
    return max(cands, key=lambda f: f.stat().st_mtime) if cands else None
resume = latest_ckpt(RUN_DIR)

env = os.environ.copy()
env['PYTHONPATH'] = str(Path(info['ext_so']).parent)
env['NSOS_BUILD_DIR'] = info['build_dir']
env['NSOS_TRAIN_CHUNK_SIZE'] = '0'
env['PYTHONUNBUFFERED'] = '1'
env['NSOS_MIXED_PRECISION'] = 'bf16'   # T4 Tensor Cores (NÃO ligar ao retomar de checkpoint FP32)
env['NSOS_GPU_POOL'] = '1'             # caching allocator ~3.9x (default já é ON)
env['NSOS_MAMBA_PARALLEL_SCAN'] = '0'  # OFF até passar a paridade 1e-4 (Célula 7)
# OXTA-CRIT Lei 2 (validado na 1050 Ti: recall chance->100%, E3 do
# docs/OXTA_CRIT_THEORY.md): espectro de timescales log-espacado na init do
# Mamba em vez do A=ones degenerado. Predicao: melhora held-out de longo alcance.
env['NSOS_MAMBA_A_LOGSPACED'] = '1'
# NSOS_CRIT_REG (P4) fica OFF neste run de proposito: 1 variavel por vez
# (este run A/B-a apenas o espectro de A contra o run anterior).

cmd = [sys.executable, '-u', str(REPO_ROOT / 'OXN/nsos/scripts/train_curriculum.py'),
       '--profile', info['profile'], '--bundle-dir', str(V11_BUNDLE),
       '--build-dir', info['build_dir'], '--run-dir', str(RUN_DIR),
       '--device', 'gpu', '--checkpoint-every-steps', '100']
if resume: cmd += ['--resume-model', str(resume)]
if _sh.which('stdbuf'): cmd = ['stdbuf','-oL','-eL'] + cmd

print('[train] profile=', info['profile'], '| resume=', resume or '(fresh)')
print('[train] BF16=on  POOL=on  PARALLEL_SCAN=off\\n')
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
try:
    for line in proc.stdout: print(line, end='', flush=True)
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired: proc.kill()
    raise
print(f'\\n[train] exit code {proc.returncode}')
"""))

cells.append(md("""## 6 — (opcional) Throughput: pool ON vs OFF + BF16 vs FP32 na T4

Comprova o ganho em escala (batches que cabem nos 16 GB da T4)."""))
cells.append(code("""B = info['build_dir']
# pool ON vs OFF
!python {REPO_ROOT}/OXN/nsos/scripts/validate_pool_scale.py --build-dir {B} --profile mamba_small --seq-len 256 --steps 10
# BF16 vs FP32 (processos separados; o modo é lido na 1a alocação)
import os
for prec in ('fp32','bf16'):
    print(f'\\n===== {prec} =====')
    !NSOS_GPU_POOL=1 NSOS_MIXED_PRECISION={prec} python {REPO_ROOT}/OXN/nsos/scripts/profile_bottlenecks.py --device gpu --profile mamba_small --batch-sizes 32,64 --seq-len 256 --steps 10 --skip-forward --configs baseline,full
"""))

cells.append(md("""## 7 — (opcional) Gate de paridade do parallel-prefix scan (1e-4)

Roda os gates da Fase 1 e o teste de paridade do scan. Se passar 1e-4, pode
ligar `NSOS_MAMBA_PARALLEL_SCAN=1` na Célula 5 para sequências longas."""))
cells.append(code("""B = info['build_dir']
# Fase 1 gates
!{B}/test_gpu_parity || {B}/Release/test_gpu_parity
!python {REPO_ROOT}/OXN/scripts/verify_determinism.py --build-dir {B}
# Paridade do scan: ver docs/COLAB_GPU_VALIDATION.md (célula 5) — compara o kernel
# sequencial (NSOS_MAMBA_PARALLEL_SCAN=0) vs o parallel-prefix (=1), tol 1e-4.
print('Para o gate do scan, siga docs/COLAB_GPU_VALIDATION.md')
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

out = Path(__file__).resolve().parent / "train_gpu_phases_t4.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}")
