@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  build_package_only.bat — rebuild ONLY the runnable package
REM  (OxtaTrainer.exe + _internal\) with the fixed trainer_main.py.
REM
REM  Does NOT rebuild the .pyd (reuses the already-built sm_75 engine) and
REM  does NOT touch / bundle the 9.8GB data\ (the friend already has it).
REM  Output: _build\pyinstaller_work\dist\OxtaTrainer\  (~1.4GB).
REM ─────────────────────────────────────────────────────────────────────────
setlocal
set "HERE=%~dp0"
set "BUILD_ROOT=%HERE%_build"
set "STAGE_DIR=%BUILD_ROOT%\stage"
set "PYINSTALLER_WORK=%BUILD_ROOT%\pyinstaller_work"
set "NSOS_PYD=%BUILD_ROOT%\pyd_rebuild\nsos_ext.cp311-win_amd64.pyd"

if not exist "%NSOS_PYD%" (
    echo ERRO: pyd nao encontrado em %NSOS_PYD%
    exit /b 1
)

if defined CUDA_PATH_V12_9 (
    set "CUDA_DIR=%CUDA_PATH_V12_9%"
) else (
    set "CUDA_DIR=%CUDA_PATH%"
)
echo   pyd:      %NSOS_PYD%
echo   CUDA_DIR: %CUDA_DIR%

echo [1/2] Staging trainer_main.py + spec + rthook ...
if exist "%STAGE_DIR%" rmdir /s /q "%STAGE_DIR%"
mkdir "%STAGE_DIR%"
copy /Y "%HERE%trainer_main.py"       "%STAGE_DIR%\" >nul
copy /Y "%HERE%pyinstaller_spec.spec" "%STAGE_DIR%\" >nul
copy /Y "%HERE%rthook_cuda_dlls.py"   "%STAGE_DIR%\" >nul

echo [2/2] PyInstaller (onedir, no data) ...
set "NSOS_EXT_PYD=%NSOS_PYD%"
python -m pip install --quiet pyinstaller zstandard
pushd "%STAGE_DIR%"
python -m PyInstaller pyinstaller_spec.spec --workpath "%PYINSTALLER_WORK%" --distpath "%PYINSTALLER_WORK%\dist" --noconfirm
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
    echo PYINSTALLER_FAILED rc=%RC%
    exit /b 1
)

set "DIST_DIR=%PYINSTALLER_WORK%\dist\OxtaTrainer"
if not exist "%DIST_DIR%\OxtaTrainer.exe" (
    echo ERRO: OxtaTrainer.exe nao gerado em %DIST_DIR%
    exit /b 1
)
echo === pacote gerado em %DIST_DIR% ===
dir "%DIST_DIR%"
echo PACKAGE_OK
endlocal
