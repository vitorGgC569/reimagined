@echo off
setlocal

set BUILD_DIR=build_ultra_v2
if exist %BUILD_DIR% rd /s /q %BUILD_DIR%
mkdir %BUILD_DIR%
cd %BUILD_DIR%

echo [1/3] Configuring CMake (Force VS 2022)...
cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo [2/3] Building test_ultra_integration...
cmake --build . --config Release --target test_ultra_integration
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo [3/3] Running test_ultra_integration...
.\Release\test_ultra_integration.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test execution failed.
    exit /b 1
)

echo [SUCCESS] Ultra Integration Test Passed!
