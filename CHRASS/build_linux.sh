#!/bin/bash

echo "=== Compiling Project Kimera (Linux) ==="

# Check if g++ exists
if ! command -v g++ &> /dev/null
then
    echo "❌ Error: g++ could not be found. Please install GCC."
    exit 1
fi

echo ">> Compiling Kimera Core (kimera_native)..."
g++ -O3 -march=native -funroll-loops -std=c++17 -fopenmp kimera.cpp -o kimera_native
if [ $? -eq 0 ]; then
    echo "✅ Success: ./kimera_native"
else
    echo "❌ Compilation failed."
    exit 1
fi

echo ">> Compiling Kimera Stress Test (kimera_stress)..."
g++ -O3 -march=native -funroll-loops -std=c++17 -fopenmp kimera_stress.cpp -o kimera_stress
if [ $? -eq 0 ]; then
    echo "✅ Success: ./kimera_stress"
else
    echo "❌ Compilation failed."
    exit 1
fi

echo "=== Done ==="
