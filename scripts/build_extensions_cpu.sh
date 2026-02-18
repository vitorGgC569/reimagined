#!/bin/bash
set -e

# Setup
mkdir -p build
PYTHON_INC=$(python3 -m pybind11 --includes)
EXT_SUFFIX=$(python3-config --extension-suffix)

echo "[Build] Compiling AION Core (CPU)..."

# 1. Compile AVX2 Kernel (Scoped)
echo "  -> Compiling AVX2 Kernel..."
g++ -O3 -Wall -std=c++17 -fPIC -mavx2 -mfma -DENABLE_AVX2 $PYTHON_INC \
    -I OXN/nsos/include -I OXN/nsos/src -I KernelOpen/src/common \
    -c OXN/nsos/src/simd/bitlinear_avx2.cpp -o build/bitlinear_avx2.o

# 2. Compile Main Extension (Without AVX2 flags globally)
echo "  -> Compiling Main Extension..."

SRC_NSOS="OXN/nsos/src/bindings.cpp \
OXN/nsos/src/bitlinear.cpp \
OXN/nsos/src/chrass_layer.cpp \
OXN/nsos/src/components.cpp \
OXN/nsos/src/embedding.cpp \
OXN/nsos/src/jamba.cpp \
OXN/nsos/src/mamba2.cpp \
OXN/nsos/src/memory_system.cpp \
OXN/nsos/src/tensor.cpp \
OXN/nsos/src/ttt_layer.cpp \
OXN/nsos/src/kan.cpp \
OXN/nsos/src/fabric.cpp \
OXN/nsos/src/holographic.cpp \
OXN/nsos/src/nsos_sdk.cpp \
OXN/nsos/src/tokenizer.cpp \
OXN/nsos/src/dataloader.cpp \
OXN/nsos/src/sprecher_kan.cpp \
OXN/nsos/src/lean_integration.cpp \
OXN/nsos/src/simd/simd_dispatch.cpp \
OXN/nsos/src/simd/bitlinear_scalar.cpp \
OXN/nsos/src/simd/hadamard.cpp \
OXN/nsos/src/trainer.cpp \
OXN/nsos/src/globals.cpp"

g++ -O3 -Wall -shared -std=c++17 -fPIC $PYTHON_INC \
    -I OXN/nsos/include \
    -I OXN/nsos/src \
    -I KernelOpen/src/common \
    $SRC_NSOS build/bitlinear_avx2.o \
    -o build/nsos_ext$EXT_SUFFIX \
    -fopenmp -Wno-reorder -Wno-unused-variable -Wno-sign-compare

echo "[Build] CPU Build Success."
ls -l build/
