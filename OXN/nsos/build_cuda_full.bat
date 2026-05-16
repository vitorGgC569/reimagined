@echo off
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo FAIL: vcvars64.bat
    exit /b 1
)

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH_V12_9=%CUDA_PATH%"
set "PATH=C:\Users\Oxta\AppData\Local\Programs\Python\Python311\Scripts;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

set "ROOT=C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos"
set "BUILD=%ROOT%\build-cuda-validation"

cd /d "%ROOT%"
if exist "%BUILD%" rmdir /s /q "%BUILD%"

REM ── Detect host GPU architecture (was hardcoded to 61 = Pascal/1050ti) ──
REM Uses scripts/gpu_arch_detect.py which queries nvidia-smi.
REM Honors $env:NSOS_CUDA_ARCHITECTURES override if set (e.g. "75" for T4).
REM Falls back to "61;75;80;86" if detection fails.
for /f "delims=" %%A in ('python scripts\gpu_arch_detect.py') do set "DETECTED_ARCH=%%A"
if "%DETECTED_ARCH%"=="" set "DETECTED_ARCH=61;75;80;86"
echo [arch] using CUDA architectures: %DETECTED_ARCH%

echo === Configure ===
cmake -G Ninja ^
    -S "%ROOT%" -B "%BUILD%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe" ^
    -DCMAKE_CUDA_HOST_COMPILER=cl.exe ^
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -Wno-deprecated-gpu-targets" ^
    -DNSOS_ENABLE_CUDA=ON ^
    -DNSOS_BUILD_PYTHON=ON ^
    -DNSOS_BUILD_TESTS=ON ^
    -DNSOS_BUILD_CLI=OFF ^
    -DNSOS_BUILD_API=OFF ^
    -DNSOS_BUILD_OXTAMEM=OFF ^
    -DNSOS_CUDA_ARCHITECTURES="%DETECTED_ARCH%" ^
    -DNSOS_CUDA_ALLOW_UNSUPPORTED_COMPILER=ON
if errorlevel 1 (
    echo === Configure FAIL ===
    exit /b 2
)

echo === Build ===
cmake --build "%BUILD%" --config Release --parallel
if errorlevel 1 (
    echo === Build FAIL ===
    exit /b 3
)

echo === SUCCESS ===
exit /b 0
