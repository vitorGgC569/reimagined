@echo off
setlocal enabledelayedexpansion

echo [Build] Starting NSOS Swarm Build...

set BUILD_DIR=build_swarm
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

cd %BUILD_DIR%

echo [Build] Configuring CMake...
cmake .. -DNSOS_BUILD_TESTS=ON -DNSOS_ENABLE_CUDA=OFF
if %ERRORLEVEL% neq 0 (
    echo [Error] CMake configuration failed.
    exit /b %ERRORLEVEL%
)

echo [Build] Building test_swarm_orchestration...
cmake --build . --config Release --target test_swarm_orchestration
if %ERRORLEVEL% neq 0 (
    echo [Error] Build failed.
    exit /b %ERRORLEVEL%
)

echo [Test] Running Swarm Orchestration Test...
cd ..
%BUILD_DIR%\Release\test_swarm_orchestration.exe
if %ERRORLEVEL% neq 0 (
    echo [Error] Test failed.
    exit /b %ERRORLEVEL%
)

echo [Success] Swarm integration verified!
cd ..
