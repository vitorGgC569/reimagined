#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  Oxta Contábil 200M — RTX 2080 Ti One-Shot Trainer (Linux / WSL2)
# ─────────────────────────────────────────────────────────────────────────
#
# Versão: 1.0  (target: hybrid_rtx2080ti_200m_chinchilla25b profile)
#
# Mesmo fluxo do .bat para Windows mas pra Linux nativo ou WSL2.
#
# Uso:
#   ./train_oxta_contabil_200m.sh              # datasets curtos (~5GB)
#   ./train_oxta_contabil_200m.sh --full       # datasets completos (~30GB)
#   ./train_oxta_contabil_200m.sh --resume     # retoma do último checkpoint
#
# Pré-requisitos:
#   - Ubuntu 22.04+ ou Debian recente (ou WSL2 com mesma)
#   - NVIDIA driver >= 535
#   - CUDA Toolkit 12.x
#   - GCC >= 11 ou Clang >= 14
#   - CMake 3.18+
#   - Python 3.11+
#   - Git, build-essential
#   - 80GB+ livre

set -euo pipefail

# ── Argumentos ───────────────────────────────────────────────────────────
DATA_MODE="fast"
RESUME_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --full)   DATA_MODE="full" ;;
        --resume) RESUME_FLAG="--resume-model latest" ;;
    esac
done

echo
echo "============================================================"
echo " Oxta Contábil 200M — RTX 2080 Ti One-Shot Trainer (Linux)"
echo "============================================================"
echo " data mode:       $DATA_MODE"
[[ -n "$RESUME_FLAG" ]] && echo " resume mode:     ON"
echo " profile:         hybrid_rtx2080ti_200m_chinchilla25b"
echo " estimated time:  ~6 days 24/7 (after setup)"
echo "============================================================"
echo

# ── 1) Verifica pré-requisitos ───────────────────────────────────────────
echo "[1/6] Verificando pré-requisitos..."
command -v python3   >/dev/null || { echo "ERROR: python3 não encontrado"; exit 1; }
command -v cmake     >/dev/null || { echo "ERROR: cmake não encontrado"; exit 1; }
command -v g++       >/dev/null || { echo "ERROR: g++ não encontrado"; exit 1; }
command -v nvidia-smi>/dev/null || { echo "ERROR: nvidia-smi não encontrado"; exit 1; }

PY_VER=$(python3 --version | awk '{print $2}')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  Python:        $PY_VER"
echo "  GPU:           $GPU_NAME"

# Compatibilidade GPU
if ! echo "$GPU_NAME" | grep -qiE "2080|2070|2060|T4|Titan RTX|Quadro RTX"; then
    echo "WARNING: GPU '$GPU_NAME' não claramente Turing sm_75."
    echo "  Profile é calibrado pra sm_75 com 11GB.  Continuando."
fi

# ── 2) Setup Python ──────────────────────────────────────────────────────
echo
echo "[2/6] Instalando dependências Python..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "pybind11>=2.10" numpy "datasets>=2.16" \
    huggingface_hub tqdm matplotlib

# ── 3) Build NSOS ────────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NSOS_SRC="$SCRIPT_DIR/../.."
BUILD_DIR="$NSOS_SRC/build-rtx2080ti"
CUDA_ARCH=75

echo
echo "[3/6] Building NSOS (15-25 min primeira vez)..."

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "  Configurando $BUILD_DIR ..."
    cmake -S "$NSOS_SRC" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DNSOS_ENABLE_CUDA=ON \
        -DNSOS_BUILD_PYTHON=ON \
        -DNSOS_BUILD_TESTS=OFF \
        -DNSOS_BUILD_CLI=OFF \
        -DNSOS_BUILD_API=OFF \
        -DNSOS_BUILD_OXTAMEM=OFF \
        -DNSOS_CUDA_ARCHITECTURES=$CUDA_ARCH
fi

echo "  Compilando..."
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
echo "  Build OK."

# ── 4) Download datasets ─────────────────────────────────────────────────
DATA_ROOT="$HOME/oxta_data"
mkdir -p "$DATA_ROOT"

if [[ "$DATA_MODE" == "full" ]]; then
    CULTURAX_BYTES="30GB"
    CULTURAX_SHARDS=100
else
    CULTURAX_BYTES="5GB"
    CULTURAX_SHARDS=20
fi

echo
echo "[4/6] Baixando datasets (mode=$DATA_MODE, CulturaX ≈ $CULTURAX_BYTES)..."
python3 "$SCRIPT_DIR/../oxta_contabil/download_datasets.py" \
    --output-dir "$DATA_ROOT" \
    --include culturax_ptbr,br_taxqa,bacen_faq,lener_br \
    --target-bytes "$CULTURAX_BYTES" \
    --culturax-shards "$CULTURAX_SHARDS" || \
    echo "WARNING: alguns datasets falharam (continuando com o que baixou)"

# ── 5) Bundle ────────────────────────────────────────────────────────────
BUNDLE_DIR="$SCRIPT_DIR/../distillation_bundle_v11"
if [[ ! -f "$BUNDLE_DIR/tokenizer_8192.ox3" ]]; then
    echo "ERROR: bundle não encontrado em $BUNDLE_DIR"
    echo "  Precisa do tokenizer + curriculum do v11."
    echo "  Pede pro Vitor enviar distillation_bundle_v11/"
    exit 1
fi
echo
echo "[5/6] Bundle OK: $BUNDLE_DIR"

# ── 6) Treino ────────────────────────────────────────────────────────────
RUN_DIR="$HOME/oxta_packs/rtx2080ti_200m_$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN_DIR"

# Power limit opcional (220W reduz temp, perde ~5% perf)
nvidia-smi -pl 220 2>/dev/null || true

export NSOS_BUILD_DIR="$BUILD_DIR"
export NSOS_MIXED_PRECISION="bf16"
export NSOS_USE_LUT_SIMD=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BUILD_DIR:${PYTHONPATH:-}"

echo
echo "[6/6] Iniciando treino 200M (~140h / ~6 dias 24/7)..."
echo "  Run dir:       $RUN_DIR"
echo "  Checkpoint:    cada 100 steps (~25 min)"
echo

python3 -u "$SCRIPT_DIR/../train_curriculum.py" \
    --profile hybrid_rtx2080ti_200m_chinchilla25b \
    --bundle-dir "$BUNDLE_DIR" \
    --build-dir "$BUILD_DIR" \
    --run-dir "$RUN_DIR" \
    --device gpu \
    --checkpoint-every-steps 100 \
    $RESUME_FLAG

echo
echo "============================================================"
echo " Treino COMPLETO"
echo "============================================================"
echo " Pack final:    $RUN_DIR/final_edge_linear.nsos"
[[ -f "$RUN_DIR/final_edge_linear.nsos" ]] && \
    echo " Tamanho:       $(du -h "$RUN_DIR/final_edge_linear.nsos" | cut -f1)"
echo
echo " Compartilha o arquivo final_edge_linear.nsos com o Vitor."
echo " Esse é o 'cérebro' do Oxta Contábil 200M treinado."
echo "============================================================"
