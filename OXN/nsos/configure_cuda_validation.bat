@echo off
setlocal enabledelayedexpansion
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "PATH=C:\Users\Oxta\AppData\Local\Programs\Python\Python311\Scripts;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
cd /d "C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos"

REM ── Detect host GPU architecture (was hardcoded to 61 = Pascal/1050ti) ──
REM Uses scripts/gpu_arch_detect.py which queries nvidia-smi.
REM Honors $env:NSOS_CUDA_ARCHITECTURES override if set (e.g. "75" for T4).
REM Falls back to "61;75;80;86" if detection fails.
for /f "delims=" %%A in ('python scripts\gpu_arch_detect.py') do set "DETECTED_ARCH=%%A"
if "!DETECTED_ARCH!"=="" set "DETECTED_ARCH=61;75;80;86"
echo [arch] using CUDA architectures: !DETECTED_ARCH!

rmdir /s /q build-cuda-validation 2>nul
cmake -G Ninja -S . -B build-cuda-validation ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DNSOS_ENABLE_CUDA=ON ^
  -DNSOS_BUILD_PYTHON=OFF ^
  -DNSOS_BUILD_TESTS=ON ^
  -DNSOS_BUILD_CLI=OFF ^
  -DNSOS_BUILD_API=OFF ^
  -DNSOS_BUILD_OXTAMEM=OFF ^
  -DNSOS_CUDA_ARCHITECTURES="!DETECTED_ARCH!" ^
  -DNSOS_CUDA_ALLOW_UNSUPPORTED_COMPILER=ON ^
  "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe"
exit /b %ERRORLEVEL%
