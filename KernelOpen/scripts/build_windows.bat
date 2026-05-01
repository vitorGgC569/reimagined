@echo off
REM Universal Heterogeneous Kernel - Windows Build Script
REM Usage: scripts\build_windows.bat

echo ========================================
echo    UHK Build System (Windows)
echo ========================================

REM Check for G++ (MinGW)
where g++ >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] g++ not found. Please install MinGW-w64 and add to PATH.
    exit /b 1
)

REM Prepare Build Directory
if not exist build mkdir build

REM Compile Unit Tests
echo [*] Compiling Unit Tests...
g++ -O3 -std=c++17 -I./src tests/unit_tests.cpp -o build/unit_tests.exe -lpthread
if %errorlevel% neq 0 (
    echo    -> Compilation failed.
    exit /b 1
) else (
    echo    -> build/unit_tests.exe created.
)

REM Compile Benchmarks
echo [*] Compiling Benchmarks...
g++ -O3 -std=c++17 -I./src tests/benchmark_uhk.cpp -o build/benchmark_uhk.exe -lpthread
if %errorlevel% neq 0 (
    echo    -> Compilation failed.
    exit /b 1
) else (
    echo    -> build/benchmark_uhk.exe created.
)

echo ========================================
echo    Build Complete.
echo ========================================
echo.
echo To run tests:
echo   build\unit_tests.exe
echo.
echo To run benchmarks:
echo   build\benchmark_uhk.exe
echo ========================================

REM Auto-run check (simple check for --run arg)
if "%1"=="--run" (
    echo [*] Running Suite...
    build\unit_tests.exe
    build\benchmark_uhk.exe
)
