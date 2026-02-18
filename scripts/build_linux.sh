#!/bin/bash

# Marco Zero - Build Script for Linux/MacOS
# Automates dependency installation and C++ compilation

set -e # Exit on error

echo "=================================================="
echo "🚀 Marco Zero: Initializing Build Environment"
echo "=================================================="

# 1. Check for Python
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 could not be found. Please install python3."
    exit 1
fi
echo "✅ Python3 found."

# 2. Check for CMake
if ! command -v cmake &> /dev/null
then
    echo "❌ CMake could not be found. Please install cmake (sudo apt install cmake)."
    exit 1
fi
echo "✅ CMake found."

# 3. Create Virtual Environment
echo "📦 Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   -> Created 'venv'"
else
    echo "   -> 'venv' already exists."
fi

# Activate venv
source venv/bin/activate

# 4. Install Python Dependencies
echo "📦 Installing Dependencies..."
pip install --upgrade pip
pip install torch numpy pybind11 transformers

# Create models directory
mkdir -p models

# Get Pybind11 path
PYBIND_PATH=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# 5. Compile KernelOpen (Hardware Abstraction Layer)
echo "🔧 Compiling KernelOpen..."
cd KernelOpen
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake ..
make -j$(nproc)
echo "   -> KernelOpen Compiled."
cd ../../

# 6. Compile Pantheon (Root)
echo "🔧 Compiling Pantheon (Root)..."
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build

# Pass KernelOpen include path if needed, though CMakeLists.txt handles it relative
cmake -Dpybind11_DIR="$PYBIND_PATH" ..
make -j$(nproc)

echo "   -> Installing Pantheon to models/"
cp pantheon*.so ../models/

cd ..

# 7. Compile OXN Engine (NSOS)
echo "🔧 Compiling OXN Engine (nsos_ext)..."
cd OXN/nsos

if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build

cmake -Dpybind11_DIR="$PYBIND_PATH" ..
make -j$(nproc)

echo "   -> Installing NSOS to models/"
cp nsos_ext*.so ../../../models/

# Return to root
cd ../../../

echo "=================================================="
echo "🎉 Build Success! modules are in 'models/'."
echo "   To start, run: source venv/bin/activate"
echo "   Then try: python3 benchmarks/run_math_sovereign_v2.py"
echo "=================================================="
