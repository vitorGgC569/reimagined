#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  build_standalone.sh — bundle builder for Linux/WSL2
# ─────────────────────────────────────────────────────────────────────────
#
# Mirror of build_standalone.bat but for Linux.  Produces a tarball that
# bundles an .AppImage-like binary + data/.  Note: PyInstaller on Linux
# produces ELF binaries that won't run on Windows — only useful if your
# friend's PC is Linux too.  If he's on Windows, run build_standalone.bat
# on a Windows machine instead.
#
# Usage:
#   ./build_standalone.sh                       # CulturaX 15GB
#   ./build_standalone.sh --full                # CulturaX 30GB
#   ./build_standalone.sh --compress            # zstd-compress CulturaX
#   ./build_standalone.sh --full --compress     # both (recommended)
#
# Pre-reqs:
#   - Ubuntu 22.04+ / Debian recent
#   - CUDA Toolkit 12.x (CUDA_HOME env var set)
#   - GCC 11+ / Clang 14+
#   - CMake 3.18+
#   - Python 3.11+
#   - p7zip-full
#   - ~80GB disk free

set -euo pipefail

CULTURAX_GB=15
COMPRESS_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --full)     CULTURAX_GB=30 ;;
        --compress) COMPRESS_FLAG="--compress" ;;
    esac
done

echo ""
echo "============================================================"
echo " OxtaTrainer Standalone Bundle Builder (Linux)"
echo "============================================================"
echo " CulturaX size:   ${CULTURAX_GB} GB"
[[ -n "$COMPRESS_FLAG" ]] && echo " Compression:     ON"
echo "============================================================"
echo ""

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NSOS_ROOT="$HERE/../.."
BUILD_ROOT="$HERE/_build"
BUILD_DIR="$BUILD_ROOT/nsos_compile"
PYINSTALLER_WORK="$BUILD_ROOT/pyinstaller_work"
STAGE_DIR="$BUILD_ROOT/stage"
FINAL_DIR="$HERE/OxtaTrainer_v1_linux"
FINAL_ARCHIVE="$HERE/OxtaTrainer_v1_linux.7z"

# ── 1) Pre-flight ────────────────────────────────────────────────────────
echo "[1/7] Checking pre-requisites..."
command -v cmake   >/dev/null || { echo "ERROR: cmake not found"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }
command -v g++     >/dev/null || { echo "ERROR: g++ not found"; exit 1; }
command -v 7z      >/dev/null || { echo "WARNING: 7z not found; will skip compression step"; SKIP_7Z=1; }

if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
    echo "ERROR: CUDA_HOME or CUDA_PATH not set"
    exit 1
fi
: "${CUDA_HOME:=${CUDA_PATH}}"
echo "  CUDA:    $CUDA_HOME"
python3 --version

# ── 2) Build nsos_ext .so ───────────────────────────────────────────────
echo ""
echo "[2/7] Building nsos_ext for sm_75 ..."
mkdir -p "$BUILD_DIR"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$NSOS_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DNSOS_ENABLE_CUDA=ON \
        -DNSOS_BUILD_PYTHON=ON \
        -DNSOS_BUILD_TESTS=OFF \
        -DNSOS_BUILD_CLI=OFF \
        -DNSOS_BUILD_API=OFF \
        -DNSOS_BUILD_OXTAMEM=OFF \
        -DNSOS_CUDA_ARCHITECTURES=75
fi
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)" --target nsos_ext

NSOS_SO=$(find "$BUILD_DIR" -name "nsos_ext*.so" | head -n1)
if [[ -z "$NSOS_SO" ]]; then
    echo "ERROR: nsos_ext .so not found after build"
    exit 1
fi
echo "  so OK: $NSOS_SO"

# ── 3) Stage ────────────────────────────────────────────────────────────
echo ""
echo "[3/7] Staging..."
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp "$HERE/trainer_main.py"       "$STAGE_DIR/"
cp "$HERE/pyinstaller_spec.spec" "$STAGE_DIR/"

# ── 4) Prepare data ─────────────────────────────────────────────────────
echo ""
echo "[4/7] prepare_data.py (CulturaX=${CULTURAX_GB}GB)..."
python3 "$HERE/prepare_data.py" \
    --output "$BUILD_ROOT" \
    --culturax-gb "$CULTURAX_GB" \
    $COMPRESS_FLAG

# ── 5) PyInstaller ──────────────────────────────────────────────────────
echo ""
echo "[5/7] Installing PyInstaller + zstandard..."
python3 -m pip install --quiet --upgrade pyinstaller zstandard

echo ""
echo "[5/7 cont] Running PyInstaller..."
export NSOS_EXT_PYD="$NSOS_SO"
export CUDA_DIR="$CUDA_HOME"

pushd "$STAGE_DIR" >/dev/null
python3 -m PyInstaller pyinstaller_spec.spec \
    --workpath "$PYINSTALLER_WORK" \
    --distpath "$PYINSTALLER_WORK/dist" \
    --noconfirm
popd >/dev/null

DIST_DIR="$PYINSTALLER_WORK/dist/OxtaTrainer"
if [[ ! -f "$DIST_DIR/OxtaTrainer" ]]; then
    echo "ERROR: OxtaTrainer binary not generated"
    exit 1
fi

# ── 6) Assemble final folder ─────────────────────────────────────────────
echo ""
echo "[6/7] Assembling $FINAL_DIR..."
rm -rf "$FINAL_DIR"
mkdir -p "$FINAL_DIR"
cp -r "$DIST_DIR/"*           "$FINAL_DIR/"
cp -r "$BUILD_ROOT/data"      "$FINAL_DIR/"
cp    "$HERE/README_FRIEND.txt" "$FINAL_DIR/README.txt"
echo "  final size: $(du -sh "$FINAL_DIR" | cut -f1)"

# ── 7) Compress ──────────────────────────────────────────────────────────
echo ""
echo "[7/7] Compressing to $FINAL_ARCHIVE..."
if [[ -n "${SKIP_7Z:-}" ]]; then
    echo "  7z not installed; skipping"
else
    rm -f "$FINAL_ARCHIVE"
    7z a -t7z -mx=7 -m0=lzma2 "$FINAL_ARCHIVE" "$FINAL_DIR"/*
    echo "  archive: $(du -h "$FINAL_ARCHIVE" | cut -f1)"
fi

echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
echo " Folder:   $FINAL_DIR"
[[ -z "${SKIP_7Z:-}" ]] && echo " Archive:  $FINAL_ARCHIVE"
echo "============================================================"
