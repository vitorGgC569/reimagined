@echo off
REM Rebuild ONLY nsos_ext.pyd (the engine) for sm_75, from the current worktree
REM source (includes the QAT + RMSNorm-backward fixes + this session's changes).
REM No data bundling, no PyInstaller, no 11GB archive -- just the ~30MB .pyd.
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
python -m pip install --quiet pybind11 2>nul
set "SRC=%~dp0..\.."
set "OUT=%~dp0_build\pyd_rebuild"
echo [1/2] cmake configure (CUDA sm_75 + Python) ...
cmake -S "%SRC%" -B "%OUT%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DNSOS_ENABLE_CUDA=ON ^
    -DNSOS_BUILD_PYTHON=ON ^
    -DNSOS_BUILD_TESTS=OFF ^
    -DNSOS_BUILD_CLI=OFF ^
    -DNSOS_BUILD_API=OFF ^
    -DNSOS_BUILD_OXTAMEM=OFF ^
    -DNSOS_CUDA_ARCHITECTURES=75
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )
echo [2/2] building nsos_ext ...
cmake --build "%OUT%" --target nsos_ext
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo === produced .pyd ===
dir /b /s "%OUT%\*nsos_ext*.pyd"
echo PYD_BUILD_OK
