#!/bin/bash
set -e

echo "[Build] Compiling Pure C++ Test..."

# Include paths
INC="-I OXN/nsos/include -I OXN/nsos/src -I KernelOpen/include -I KernelOpen/src/common"

# Sources (No bindings.cpp)
SRC="tests/test_pure_mamba.cpp \
OXN/nsos/src/mamba2.cpp \
OXN/nsos/src/bitlinear.cpp \
OXN/nsos/src/tensor.cpp \
OXN/nsos/src/simd/simd_dispatch.cpp \
OXN/nsos/src/simd/bitlinear_scalar.cpp \
OXN/nsos/src/simd/bitlinear_avx2.cpp \
OXN/nsos/src/simd/hadamard.cpp"

# Output
OUT="tests/test_pure_mamba"

g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp $INC $SRC -o $OUT

echo "[Build] Success. Running test..."
./$OUT
