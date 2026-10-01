"""Gera colab/nsos_cpu_eval.ipynb — avaliacao qualitativa do checkpoint em CPU.

Para quando a quota de GPU do Colab acabou: runtime CPU nao tem quota, o
modelo (40M, FP32 ~160MB) carrega na RAM e o caminho CPU e' exatamente a tese
edge do projeto.  Build CPU-only (sem CUDA) com cache do .so no Drive por sha.
Regenerar: python colab/make_cpu_eval_notebook.py
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

cells.append(md("""# NSOS — eval do checkpoint em CPU (sem quota de GPU)
Branch `%BRANCH%`. Runtime: **CPU** (Runtime > Change runtime type > CPU).
Ordem: 1 Drive -> 2 clone -> 3 build CPU (cache no Drive; 1a vez ~10-20 min,
depois segundos) -> 4 bundle (cache zip do Drive) -> 5 perguntas e respostas.
Expectativa honesta de velocidade: CPU do Colab tem 2 cores — algo como
1-5 tok/s no decode; 5 amostras x 32 tokens = poucos minutos."""))

cells.append(code("""# [1] Drive + info da maquina (runtime CPU esperado)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, multiprocessing
DRIVE_ROOT = Path('/content/drive/MyDrive/nsos_v11')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print('cpus:', multiprocessing.cpu_count())
print('gpu? (vazio/erro = ok, queremos CPU):')
os.system('nvidia-smi -L 2>/dev/null || echo "  sem GPU - perfeito para este notebook"')
"""))

cells.append(code("""# [2] Clone + prova de versao
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

subprocess.run(['git', '-C', str(REPO_ROOT), 'remote', 'set-url', 'origin', REPO_URL], check=False)  # nao persistir token no .git/config

sha = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
print('=' * 60)
print(f'[ver] HEAD = {sha}')
print('=' * 60)
"""))

cells.append(code("""# [3] Build CPU-only (sem CUDA) com cache do .so no Drive
import os, sys, subprocess, shutil, glob
from pathlib import Path

if 'nsos_ext' in sys.modules:
    raise RuntimeError('nsos_ext ja carregado neste runtime — Runtime > Restart e rode de novo.')

PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
sha = subprocess.run(['git', '-C', '/content/reimagined', 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
CACHE = DRIVE_ROOT / '_bootstrap' / f'nsos_ext_cpu.{PYTAG}.{sha}.so'
EXT_DIR = Path('/content/nsos_cpu_ext')
EXT_DIR.mkdir(exist_ok=True)

if CACHE.exists():
    dst = EXT_DIR / CACHE.name.replace(f'nsos_ext_cpu.{PYTAG}.{sha}', 'nsos_ext')
    shutil.copy2(CACHE, EXT_DIR / 'nsos_ext.so')
    print(f'[build] cache HIT: {CACHE.name}')
else:
    print(f'[build] cache MISS para sha={sha} — compilando CPU-only (~10-20 min na 1a vez)')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pybind11'], check=True)
    BUILD = '/content/reimagined/OXN/nsos/build-cpu'
    r = subprocess.run(['cmake', '-S', '/content/reimagined/OXN/nsos', '-B', BUILD,
                        '-DNSOS_ENABLE_CUDA=OFF', '-DNSOS_BUILD_PYTHON=ON',
                        '-DNSOS_BUILD_TESTS=OFF', '-DNSOS_BUILD_CLI=OFF',
                        '-DNSOS_BUILD_API=OFF', '-DCMAKE_BUILD_TYPE=Release',
                        f'-DPython3_EXECUTABLE={sys.executable}'],
                       capture_output=True, text=True)
    print(r.stdout[-1200:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-2000:]); raise RuntimeError('cmake configure falhou')
    r = subprocess.run(['cmake', '--build', BUILD, '--target', 'nsos_ext', '-j', '2'],
                       capture_output=True, text=True)
    print(r.stdout[-1200:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-2000:]); raise RuntimeError('build falhou')
    so = sorted(glob.glob(f'{BUILD}/**/nsos_ext*.so', recursive=True))
    assert so, 'nsos_ext*.so nao encontrado pos-build'
    shutil.copy2(so[0], EXT_DIR / 'nsos_ext.so')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so[0], CACHE)
    print(f'[build] ok + cacheado -> {CACHE.name}')

BUILD_DIR = str(EXT_DIR)
print('[build] ext dir:', BUILD_DIR)
"""))

cells.append(code("""# [4] Bundle v11 (cache zip do Drive — rapido; so baixa datasets se nunca treinou)
import os, sys, shutil, subprocess, json
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
    print((r.stdout or '')[-1500:])
    if r.returncode != 0:
        print('STDERR:', (r.stderr or '')[-1500:]); raise RuntimeError('fetch_real_datasets falhou')
    r = subprocess.run([sys.executable, str(REPO_ROOT / 'OXN/nsos/scripts/build_curriculum.py'),
                        '--preset', 'v11'], capture_output=True, text=True)
    print((r.stdout or '')[-1500:])
    if r.returncode != 0:
        print('STDERR:', (r.stderr or '')[-1500:]); raise RuntimeError('build_curriculum falhou')
    shutil.make_archive(str(cached.with_suffix('')), 'zip', V11_BUNDLE)

print('[data] bundle pronto')
"""))

cells.append(code("""# [4.5] PADRAO OURO do tokenizer + raio-x do corpus (segundos, CPU)
# G1 roundtrip byte-exato em todos os eval rows | G2 specials atomicos +
# teste de injecao | G3 determinismo | G4 prefix-stability prompt|answer |
# G5 cobertura | G6 composicao do corpus por fonte (o raio-x que explica
# "por que o modelo responde C++": phase3 inclui o fonte do NSOS por design).
import subprocess, sys
r = subprocess.run([sys.executable,
                    '/content/reimagined/OXN/nsos/scripts/tokenizer_gold_gate.py',
                    '--bundle-dir', '/content/reimagined/OXN/nsos/scripts/distillation_bundle_v11',
                    '--build-dir', BUILD_DIR], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr[-1500:])
print('TOKENIZER GATE:', 'PASS' if r.returncode == 0 else 'FAIL')
"""))

cells.append(code("""# [5] Perguntas e RESPOSTAS do checkpoint — tudo em CPU/RAM
import os, sys, time, random
from pathlib import Path
sys.path.insert(0, str(REPO_ROOT / 'OXN/nsos/scripts'))
from train_curriculum import (build_model_config, load_nsos, resolve_profile,
                              set_model_training_mode, greedy_generate, SPECIAL_TOKENS)
from nsos_curriculum_lib import curriculum_texts_for_phase

PHASE = 'phase1_algorithms'   # mude para phase3_curated_text etc. se quiser
N_SAMPLES = 5
MAX_NEW = 32                  # CPU 2-core: ~1-5 tok/s; suba se tiver paciencia

BUNDLE = REPO_ROOT / 'OXN/nsos/scripts/distillation_bundle_v11'
runs_root = DRIVE_ROOT / 'runs'
cands = [f for f in runs_root.glob('*/*.bin') if f.name != 'audit_reload_probe.bin']
assert cands, f'nenhum checkpoint .bin em {runs_root}'
ckpt = max(cands, key=lambda f: f.stat().st_mtime)
print(f'[ckpt] {ckpt}  ({ckpt.stat().st_size/1e6:.1f} MB)')

nsos = load_nsos(Path(BUILD_DIR))
_, profile = resolve_profile('hybrid_v11_colab_t4')
tok = nsos.Tokenizer()
tok.load(str(BUNDLE / 'tokenizer_8192.ox3'))
tok.add_special_tokens(SPECIAL_TOKENS)
eos_id = tok.encode('<|endoftext|>')[0]

dev = nsos.Device.CPU
cfg = build_model_config(nsos, profile, tok.vocab_size, dev)
model = nsos.JambaModel(cfg, dev)
t0 = time.perf_counter()
try:
    model.load(str(ckpt), True)
    print(f'[ckpt] load estrito OK em {time.perf_counter()-t0:.1f}s (RAM, sem GPU)')
except Exception as exc:
    print(f'[ckpt] load estrito falhou ({exc}); tentando parcial...')
    model.load(str(ckpt), False)
set_model_training_mode(model, False)

rows = curriculum_texts_for_phase(BUNDLE, PHASE, 'eval')
print(f'[eval] {PHASE}: {len(rows)} amostras; gerando {N_SAMPLES} (max_new={MAX_NEW})\\n')
rng = random.Random(42)
for i, row in enumerate(rng.sample(rows, min(N_SAMPLES, len(rows)))):
    prompt = f"<|task:{row['kind']}|>\\nPrompt:\\n{row['prompt']}\\nAnswer:\\n"
    t0 = time.perf_counter()
    out = greedy_generate(nsos, model, tok, prompt, max_new_tokens=MAX_NEW,
                          eos_token_id=eos_id)
    dt = time.perf_counter() - t0
    ntok = len(tok.encode(out)) if out else 0
    print('=' * 72)
    print(f'[{i+1}] tarefa: {row["kind"]}  ({dt:.1f}s, ~{ntok/max(dt,1e-9):.1f} tok/s)')
    print(f'PERGUNTA : {row["prompt"][:300]}')
    print(f'ESPERADO : {str(row["answer"])[:200]}')
    print(f'MODELO   : {out[:200]}')
print('=' * 72)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent / "nsos_cpu_eval.ipynb"
text = json.dumps(nb, indent=1, ensure_ascii=False).replace("%BRANCH%", BRANCH)
out.write_text(text, encoding="utf-8")
print(f"wrote {out}")
