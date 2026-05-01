#!/bin/bash
set -e

echo "=========================================="
echo "   NSOS (Neuro-Symbiotic Omni-Scale)      "
echo "        Linux Build Script                "
echo "=========================================="

# 1. Directory Setup
echo "[1/4] Setting up build directory..."
mkdir -p build
cd build

# 2. CMake Configuration
echo "[2/4] Configuring project with CMake..."
# Assumes standard build tools (gcc/g++, cmake) are installed
PYBIND_DIR=$(python3 -m pybind11 --cmakedir)
PY_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PY_CONF=$(python3-config --includes)
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$PYBIND_DIR -DCMAKE_CXX_FLAGS="-fPIC -I$PY_INC $PY_CONF"

# 3. Compilation
echo "[3/4] Compiling source code..."
make -j$(nproc)

# 4. Verification
echo "[4/4] Running Unit Tests..."
ctest --output-on-failure

echo "=========================================="
echo "Build Successful!"
echo "Run the chat application: ./build/chat_app"
echo "=========================================="
