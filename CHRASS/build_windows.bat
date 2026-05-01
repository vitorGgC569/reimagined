@echo off
echo === Compiling Project Kimera (Windows) ===

where g++ >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: g++ not found in PATH. Please install MinGW-w64.
    exit /b 1
)

echo >> Compiling Kimera Core (kimera_native.exe)...
g++ -O3 -march=native -funroll-loops -std=c++17 -fopenmp kimera.cpp -o kimera_native.exe
if %errorlevel% neq 0 (
    echo ❌ Compilation failed.
    exit /b 1
) else (
    echo ✅ Success: kimera_native.exe
)

echo >> Compiling Kimera Stress Test (kimera_stress.exe)...
g++ -O3 -march=native -funroll-loops -std=c++17 -fopenmp kimera_stress.cpp -o kimera_stress.exe
if %errorlevel% neq 0 (
    echo ❌ Compilation failed.
    exit /b 1
) else (
    echo ✅ Success: kimera_stress.exe
)

echo === Done ===
pause
