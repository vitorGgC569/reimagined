#!/bin/bash
set -e

# Universal Heterogeneous Kernel - Linux Build Script
# Usage: ./scripts/build_linux.sh

echo "========================================"
echo "   UHK Build System (Linux/Bash)"
echo "========================================"

# 1. Check for Compiler
if ! command -v g++ &> /dev/null; then
    echo "[Error] g++ could not be found. Please install g++ (build-essential)."
    exit 1
fi

# 2. Prepare Build Directory
echo "[*] Creating build directory..."
mkdir -p build

# 3. Compile Unit Tests
echo "[*] Compiling Unit Tests..."
g++ -O3 -std=c++17 -I./src tests/unit_tests.cpp -o build/unit_tests -lpthread
if [ $? -eq 0 ]; then
    echo "    -> build/unit_tests created."
else
    echo "    -> Compilation failed."
    exit 1
fi

# 4. Compile Benchmarks
echo "[*] Compiling Benchmarks..."
g++ -O3 -std=c++17 -I./src tests/benchmark_uhk.cpp -o build/benchmark_uhk -lpthread
if [ $? -eq 0 ]; then
    echo "    -> build/benchmark_uhk created."
else
    echo "    -> Compilation failed."
    exit 1
fi

echo "========================================"
echo "   Build Complete."
echo "========================================"
echo ""
echo "To run tests:"
echo "  ./build/unit_tests"
echo ""
echo "To run benchmarks:"
echo "  ./build/benchmark_uhk"
echo "========================================"

# Auto-run if requested
if [ "$1" == "--run" ]; then
    echo "[*] Running Suite..."
    ./build/unit_tests
    ./build/benchmark_uhk
fi
