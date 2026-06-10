"""Gera colab/nsos_lab_t4.ipynb — o LABORATORIO de testes do NSOS na T4.

Separado do notebook de treino: aqui rodam gates, paridades e perfis.
Protocolo do projeto: NADA disso roda no desktop local (analise estatica +
build sao a verificacao local; execucao real e' deste notebook).
Regenerar: python colab/make_lab_notebook.py
"""
import json
from pathlib import Path

BRANCH = "nsos-gpu-phases12"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md("""# NSOS LAB — gates, paridades e perfis (T4)
Branch `%BRANCH%`. Ordem: 1 GPU -> 2 clone+prova de versao -> 3 build limpo ->
4 gates -> 5 paridade do backward da atencao (NOVO) -> 6 timing real do step ->
7 probe de criticalidade. Cada celula e' autossuficiente."""))

cells.append(code("""# [1] GPU do runtime (precisa ser T4/L4/A100)
import subprocess
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=False)
"""))

cells.append(code("""# [2] Clone + PROVA DE VERSAO (anti fonte/binario stale)
import os, subprocess
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
            print(f'[auth] token {key} ok')
            break
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
os.environ['REPO_ROOT'] = str(REPO_ROOT)

sha = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
msg = subprocess.run(['git', '-C', str(REPO_ROOT), 'log', '-1', '--format=%s'],
                     capture_output=True, text=True).stdout.strip()
attn = (REPO_ROOT / 'OXN/nsos/src/cuda/attention_train_kernels.cu').exists()
alog = (REPO_ROOT / 'OXN/nsos/src/mamba2.cpp').read_text(encoding='utf-8').count('NSOS_MAMBA_A_LOGSPACED')
print('=' * 70)
print(f'[ver] HEAD = {sha} ({msg[:58]})')
print(f'[ver] attention_train_kernels.cu presente: {attn}')
print(f'[ver] fix A-logspaced no fonte: {alog > 0}')
print('=' * 70)
"""))

cells.append(code("""# [3] Build LIMPO (cache por sha; rm garante recompilar o codigo da celula 2)
import sys, os
from pathlib import Path
REPO_ROOT = Path('/content/reimagined')
os.system('rm -rf /content/reimagined/OXN/nsos/build-colab')
os.system('rm -f /content/drive/MyDrive/nsos_v11/_bootstrap/*.so.* 2>/dev/null')
sys.path.insert(0, str(REPO_ROOT / 'colab'))
os.environ['NSOS_PARAM_SCALE'] = '40m'
from colab_bootstrap import setup_environment
info = setup_environment(verbose=True)
BUILD = info['build_dir']
print('[build] ok:', BUILD)
"""))

cells.append(code("""# [4] GATES: parity CPU<->GPU 6/6 + determinismo + ctest do parallel-scan
import subprocess, os
from pathlib import Path
REPO_ROOT = Path('/content/reimagined')
for exe in (Path(BUILD) / 'test_gpu_parity', Path(BUILD) / 'Release' / 'test_gpu_parity'):
    if exe.exists():
        subprocess.run([str(exe)], check=False)
        break
subprocess.run(['python', str(REPO_ROOT / 'OXN/scripts/verify_determinism.py'),
                '--build-dir', BUILD], check=False)
subprocess.run(['ctest', '--test-dir', BUILD, '-R', 'parallel_scan',
                '--output-on-failure'], check=False)
"""))

cells.append(code("""# [5] PARIDADE do backward GPU da atencao (gate 1e-3, FP32, 3 steps A/B)
import subprocess, os
env = dict(os.environ)
env['PYTHONPATH'] = os.path.dirname(info['ext_so'])
r = subprocess.run(['python', '/content/reimagined/OXN/nsos/scripts/attn_bwd_parity.py',
                    '--build-dir', BUILD], env=env,
                   capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-1500:] if r.returncode != 0 else '')
print('PASS' if r.returncode == 0 else 'FAIL -> manter NSOS_ATTN_BWD_HOST=1 no treino')
"""))

cells.append(code("""# [6] TIMING real do step (v11, answer-len realista) — fwd/loss/bwd/opt
# Braco GPU (novo backward) 4 steps; braco HOST 1 step so para o contraste.
import os, sys, time, random
sys.path.insert(0, '/content/reimagined/OXN/nsos/scripts')
os.environ['NSOS_TRAIN_TIMING'] = '1'
os.environ['NSOS_GPU_POOL'] = '1'
os.environ['NSOS_MIXED_PRECISION'] = 'bf16'
from train_curriculum import build_model_config, load_nsos, resolve_profile, set_model_training_mode
from pathlib import Path
nsos = load_nsos(Path(BUILD))
_, profile = resolve_profile('hybrid_v11_colab_t4')
vocab = int(profile.get('target_vocab', profile.get('vocab_size', 4096)))

def arm(name, host, steps):
    os.environ['NSOS_ATTN_BWD_HOST'] = '1' if host else '0'
    nsos.set_seed(7)
    cfg = build_model_config(nsos, profile, vocab, nsos.Device.GPU)
    m = nsos.JambaModel(cfg, nsos.Device.GPU); m.to(nsos.Device.GPU)
    set_model_training_mode(m, True)
    tr = nsos.Trainer(m, 3e-4)
    rng = random.Random(0)
    B, P, A = 32, 32, 128   # forma real de treino (bs32); host-arm roda 1 step so p/ contraste
    pb = [[rng.randint(1, vocab-1) for _ in range(P)] for _ in range(B)]
    ab = [[rng.randint(1, vocab-1) for _ in range(A)] + [0] for _ in range(B)]
    for s in range(steps):
        t0 = time.perf_counter()
        loss = tr.train_supervised_batch(pb, ab)
        dt = time.perf_counter() - t0
        print(f'[{name}] step {s} loss={float(loss):.4f} wall={dt:.2f}s '
              f'({B*(P+A)/dt:.0f} tok/s)', flush=True)
    del tr, m

arm('GPU-attn', host=False, steps=4)
arm('HOST-attn (referencia lenta)', host=True, steps=1)
os.environ['NSOS_ATTN_BWD_HOST'] = '0'
"""))

cells.append(code("""# [7] Probe de criticalidade (posicao de fase do perfil v11)
import subprocess, os
env = dict(os.environ)
env['PYTHONPATH'] = os.path.dirname(info['ext_so'])
subprocess.run(['python', '/content/reimagined/OXN/nsos/scripts/criticality_probe.py',
                '--profile', 'hybrid_v11_colab_t4', '--device', 'cpu',
                '--build-dir', BUILD], env=env, check=False)
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

out = Path(__file__).resolve().parent / "nsos_lab_t4.ipynb"
text = json.dumps(nb, indent=1, ensure_ascii=False).replace("%BRANCH%", BRANCH)
out.write_text(text, encoding="utf-8")
print(f"wrote {out}")
