@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /MD /openmp /std:c++20 /EHsc /DUSE_CUDA /DNSOS_CUDA_MIN_ARCH=61 /I..\include /I..\include\nsos /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" toy_tasks.cpp ..\build-cuda-validation\nsos_core.lib /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" cudart.lib cublas.lib cusparse.lib curand.lib /Fe:toy_tasks.exe
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
