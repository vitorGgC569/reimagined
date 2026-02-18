#!/bin/bash
set -e

# Setup
mkdir -p build
PYTHON_INC=$(python3 -m pybind11 --includes)
EXT_SUFFIX=$(python3-config --extension-suffix)

echo "[Build] Compiling AION Core (GPU)..."
# Check if nvcc exists
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Cannot build for GPU. (Use build_extensions_cpu.sh for CPU-only)"
    exit 1
fi

# 1. Compile CUDA Kernels
# Set default CUDA_ARCH to sm_86 (Ampere/RTX 3000), but allow override via env var
CUDA_ARCH=${CUDA_ARCH:-sm_86}
echo "  -> Compiling CUDA Kernels (Arch: $CUDA_ARCH)..."
# Single kernel file now contains both BitNet and Math logic
nvcc -c OXN/nsos/src/cuda/kernels.cu -o build/kernels.o -O3 -arch=$CUDA_ARCH -Xcompiler -fPIC -I OXN/nsos/include

# 2. Compile AVX2 Kernel (Scoped, for CPU fallback support in hybrid mode)
echo "  -> Compiling AVX2 Kernel (CPU Fallback)..."
g++ -O3 -Wall -std=c++17 -fPIC -mavx2 -mfma -DENABLE_AVX2 $PYTHON_INC \
    -I OXN/nsos/include -I OXN/nsos/src -I KernelOpen/src/common \
    -c OXN/nsos/src/simd/bitlinear_avx2.cpp -o build/bitlinear_avx2.o

# 3. Compile Main Extension
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
OXN/nsos/src/jamba_device.cpp \
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
OXN/nsos/src/globals.cpp \
src/trainer.cpp"

# Adjust SRC_NSOS to match actual file existence (trainer.cpp might not be in OXN/nsos/src if it was root src)
# Based on ls, src/trainer.cpp is in OXN/nsos/src/trainer.cpp from CMakeLists?
# CMakeLists says: src/trainer.cpp.
# Previous CPU script didn't include it. I will follow CPU script pattern but keep it if crucial.
# Let's align with the CPU script I wrote earlier which worked.

SRC_NSOS="OXN/nsos/src/bindings.cpp \
OXN/nsos/src/bitlinear.cpp \
OXN/nsos/src/chrass_layer.cpp \
OXN/nsos/src/components.cpp \
OXN/nsos/src/embedding.cpp \
OXN/nsos/src/jamba.cpp \
OXN/nsos/src/mamba2.cpp \
OXN/nsos/src/memory_system.cpp \
OXN/nsos/src/tensor.cpp \
OXN/nsos/src/jamba_device.cpp \
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
    -I /usr/local/cuda/include \
    $SRC_NSOS build/kernels.o build/bitlinear_avx2.o \
    -o build/nsos_ext$EXT_SUFFIX \
    -fopenmp -Wno-reorder -Wno-unused-variable -Wno-sign-compare \
    -DUSE_CUDA \
    -L/usr/local/cuda/lib64 -lcudart -lcublas

echo "[Build] GPU Build Success."
ls -l build/
