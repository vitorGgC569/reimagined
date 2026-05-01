@echo off
setlocal

echo ==================================================
echo 🚀 Marco Zero: Initializing Build Environment (Windows)
echo ==================================================

REM 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python could not be found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)
echo ✅ Python found.

REM 2. Check for CMake
cmake --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ CMake could not be found. Please install CMake and add to PATH.
    pause
    exit /b 1
)
echo ✅ CMake found.

REM 3. Create Virtual Environment
echo 📦 Setting up Python Virtual Environment...
if not exist "venv" (
    python -m venv venv
    echo    -> Created 'venv'
) else (
    echo    -> 'venv' already exists.
)

REM Activate venv
call venv\Scripts\activate

REM 4. Install Python Dependencies
echo 📦 Installing Dependencies...
pip install --upgrade pip
pip install torch numpy pybind11 transformers

REM Create models directory
if not exist "models" mkdir models

REM Get Pybind11 path
for /f "delims=" %%i in ('python -c "import pybind11; print(pybind11.get_cmake_dir())"') do set PYBIND_PATH=%%i
echo    -> Pybind11 found at: %PYBIND_PATH%

REM 5. Compile KernelOpen (Hardware Abstraction Layer)
echo 🔧 Compiling KernelOpen...
cd KernelOpen
if exist "build" (
    rmdir /s /q build
)
mkdir build
cd build
cmake ..
cmake --build . --config Release
echo    -> KernelOpen Compiled.
cd ..\..

REM 6. Compile Pantheon (Root)
echo 🔧 Compiling Pantheon (Root)...
if exist "build" (
    rmdir /s /q build
)
mkdir build
cd build

cmake -Dpybind11_DIR="%PYBIND_PATH%" ..
cmake --build . --config Release

echo    -> Installing Pantheon to models/
copy Release\pantheon*.pyd ..\models\ >nul 2>&1
copy pantheon*.pyd ..\models\ >nul 2>&1

cd ..

REM 7. Compile OXN Engine (NSOS)
echo 🔧 Compiling OXN Engine (nsos_ext)...
cd OXN\nsos

if exist "build" (
    rmdir /s /q build
)
mkdir build
cd build

cmake -Dpybind11_DIR="%PYBIND_PATH%" ..
cmake --build . --config Release

echo    -> Installing NSOS to models/
copy Release\nsos_ext*.pyd ..\..\..\models\ >nul 2>&1
copy nsos_ext*.pyd ..\..\..\models\ >nul 2>&1

REM Return to root
cd ..\..\..

echo ==================================================
echo 🎉 Build Success!
echo    Modules 'pantheon' and 'nsos_ext' are in 'models/'.
echo.
echo    To start:
echo    1. venv\Scripts\activate
echo    2. python benchmarks/run_math_sovereign_v2.py
echo ==================================================
pause
