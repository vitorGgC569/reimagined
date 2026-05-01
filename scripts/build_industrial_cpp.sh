#!/bin/bash
set -e

echo "[Build] Compiling Industrial C++ Trainer..."

# Include paths
INC="-I OXN/nsos/include -I OXN/nsos/src -I KernelOpen/include -I KernelOpen/src/common"

# Sources
SRC="benchmarks/run_industrial_train.cpp \
OXN/nsos/src/mamba2.cpp \
OXN/nsos/src/jamba.cpp \
OXN/nsos/src/ttt_layer.cpp \
OXN/nsos/src/bitlinear.cpp \
OXN/nsos/src/tensor.cpp \
OXN/nsos/src/embedding.cpp \
OXN/nsos/src/tokenizer.cpp \
OXN/nsos/src/kan.cpp \
OXN/nsos/src/memory_system.cpp \
OXN/nsos/src/holographic.cpp \
OXN/nsos/src/components.cpp \
OXN/nsos/src/chrass_layer.cpp \
OXN/nsos/src/simd/simd_dispatch.cpp \
OXN/nsos/src/simd/bitlinear_scalar.cpp \
OXN/nsos/src/simd/bitlinear_avx2.cpp \
OXN/nsos/src/simd/hadamard.cpp"

OUT="benchmarks/run_industrial_train"

g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp $INC $SRC -o $OUT

echo "[Run] Executing..."
./$OUT
