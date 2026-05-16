@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "OUT=C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos\build-cuda-validation\cpp-syntax"
set "INC=C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos\include"
set "SRC=C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos\src"
mkdir "%OUT%" 2>nul
cd /d "%OUT%"

echo ===== mamba2.cpp =====
cl.exe /nologo /c /EHsc /std:c++20 /O2 /MD /DUSE_CUDA /DNSOS_CUDA_MIN_ARCH=61 /D_WIN32 ^
  /I"%INC%" /I"%INC%\nsos" /I"%CUDA_PATH%\include" ^
  /Fo"mamba2.obj" "%SRC%\mamba2.cpp"
if errorlevel 1 ( echo FAIL mamba2.cpp & exit /b 1 )

echo ===== jamba.cpp =====
cl.exe /nologo /c /EHsc /std:c++20 /O2 /MD /DUSE_CUDA /DNSOS_CUDA_MIN_ARCH=61 /D_WIN32 ^
  /I"%INC%" /I"%INC%\nsos" /I"%CUDA_PATH%\include" ^
  /Fo"jamba.obj" "%SRC%\jamba.cpp"
if errorlevel 1 ( echo FAIL jamba.cpp & exit /b 1 )

echo ===== bitnet_gpu_dispatch.cpp =====
cl.exe /nologo /c /EHsc /std:c++20 /O2 /MD /DUSE_CUDA /DNSOS_CUDA_MIN_ARCH=61 /D_WIN32 ^
  /I"%INC%" /I"%INC%\nsos" /I"%CUDA_PATH%\include" ^
  /Fo"bitnet_gpu_dispatch.obj" "%SRC%\bitnet_gpu_dispatch.cpp"
if errorlevel 1 ( echo FAIL bitnet_gpu_dispatch.cpp & exit /b 1 )

echo.
echo ===== ALL CUDA-DEPENDENT C++ COMPILATION UNITS PASS =====
exit /b 0
