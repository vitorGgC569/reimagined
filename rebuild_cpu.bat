@echo off
echo ==========================================
echo   NSOS GPU BUILDER - CUDA v12.2
echo ==========================================

cd OXN
if exist build rmdir /s /q build
mkdir build
cd build

echo.
echo [1/3] Configuring CMake with CUDA...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo [ERROR] CMake Configuration Failed!
    exit /b %errorlevel%
)

echo.
echo [2/3] Building Release...
cmake --build . --config Release --parallel 8

if %errorlevel% neq 0 (
    echo [ERROR] Build Failed!
    exit /b %errorlevel%
)

echo.
echo [3/3] Build Complete!
cd ..\..\
