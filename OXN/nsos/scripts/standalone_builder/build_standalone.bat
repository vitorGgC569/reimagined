@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  build_standalone.bat — end-to-end bundle builder (for the maintainer)
REM ─────────────────────────────────────────────────────────────────────────
REM
REM Runs on YOUR Windows dev machine (the maintainer's, not the friend's).
REM Produces a self-contained OxtaTrainer/ folder ready to zip and ship.
REM
REM Steps:
REM   1. Verify pre-requisites (CUDA Toolkit, CMake, Python, MSVC, PyInstaller)
REM   2. cmake build of nsos_ext.pyd targeting sm_75 (Turing)
REM   3. Stage build dir + copy .pyd
REM   4. Run prepare_data.py to download + bake the data/ folder
REM   5. PyInstaller via pyinstaller_spec.spec
REM   6. Move dist/OxtaTrainer/ + data/ together into final OxtaTrainer_v1/
REM   7. 7z compress to OxtaTrainer_v1.7z (single file to upload)
REM
REM Usage:
REM   build_standalone.bat               REM 15GB CulturaX, uncompressed JSONL
REM   build_standalone.bat --full        REM 30GB CulturaX
REM   build_standalone.bat --compress    REM zstd-compress CulturaX shards
REM   build_standalone.bat --full --compress    REM both
REM
REM Pre-requisitos (CHECAR ANTES):
REM   - Windows 10/11 64-bit
REM   - CUDA Toolkit 12.x instalado (CUDA_PATH no env)
REM   - Visual Studio 2022 (Build Tools mínimo) com C++ workload
REM   - CMake 3.18+ no PATH
REM   - Python 3.11+ no PATH
REM   - 7-Zip instalado (https://www.7-zip.org/)
REM   - ~80GB livre em disco
REM
REM Tempo total: ~3-4h (download eh o passo dominante)
REM ─────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

set "CULTURAX_GB=15"
set "COMPRESS_FLAG="
for %%i in (%*) do (
    if "%%i"=="--full" set "CULTURAX_GB=30"
    if "%%i"=="--compress" set "COMPRESS_FLAG=--compress"
)

echo.
echo ============================================================
echo  OxtaTrainer Standalone Bundle Builder
echo ============================================================
echo  CulturaX size:    %CULTURAX_GB% GB
if not "%COMPRESS_FLAG%"=="" echo  Compression:      ON
echo ============================================================
echo.

REM ── Paths ───────────────────────────────────────────────────────────────
set "HERE=%~dp0"
set "NSOS_ROOT=%HERE%..\..\"
set "BUILD_ROOT=%HERE%_build"
set "BUILD_DIR=%BUILD_ROOT%\nsos_compile"
set "PYINSTALLER_WORK=%BUILD_ROOT%\pyinstaller_work"
set "STAGE_DIR=%BUILD_ROOT%\stage"
set "FINAL_DIR=%HERE%OxtaTrainer_v1"
set "FINAL_ARCHIVE=%HERE%OxtaTrainer_v1.7z"

REM ── 1) Pre-flight ───────────────────────────────────────────────────────
echo [1/7] Checking pre-requisites...

where cmake >nul 2>&1 || (echo ERROR: cmake nao encontrado & exit /b 1)
where python >nul 2>&1 || (echo ERROR: python nao encontrado & exit /b 1)
where cl >nul 2>&1 || (
    echo WARNING: cl.exe nao encontrado no PATH atual.
    echo   Voce precisa rodar este script de um 'x64 Native Tools Command Prompt for VS 2022'.
    echo   Ou abrir Developer PowerShell e setar o env primeiro.
    exit /b 1
)
where 7z >nul 2>&1 || (
    echo WARNING: 7z nao encontrado.  Etapa 7 (compactacao) sera pulada.
    set "SKIP_7Z=1"
)

if not defined CUDA_PATH (
    echo ERROR: CUDA_PATH nao setado. Instale o CUDA Toolkit 12.x.
    exit /b 1
)
echo   CUDA:    %CUDA_PATH%

python --version
cmake --version | findstr /R "version"
echo   ok

REM ── 2) Build nsos_ext.pyd ───────────────────────────────────────────────
echo.
echo [2/7] Building nsos_ext.pyd for sm_75 (Turing — RTX 2080 Ti) ...

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    cmake -S "%NSOS_ROOT%" -B "%BUILD_DIR%" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DNSOS_ENABLE_CUDA=ON ^
        -DNSOS_BUILD_PYTHON=ON ^
        -DNSOS_BUILD_TESTS=OFF ^
        -DNSOS_BUILD_CLI=OFF ^
        -DNSOS_BUILD_API=OFF ^
        -DNSOS_BUILD_OXTAMEM=OFF ^
        -DNSOS_CUDA_ARCHITECTURES=75
    if errorlevel 1 (
        echo ERROR: cmake configure failed.
        exit /b 1
    )
)

cmake --build "%BUILD_DIR%" --config Release -j --target nsos_ext
if errorlevel 1 (
    echo ERROR: nsos_ext build failed.
    exit /b 1
)

REM Find the produced .pyd
set "NSOS_PYD="
for /f "delims=" %%f in ('dir /b /s "%BUILD_DIR%\Release\nsos_ext*.pyd" 2^>nul') do (
    set "NSOS_PYD=%%f"
    goto :pyd_found
)
:pyd_found
if not defined NSOS_PYD (
    echo ERROR: nsos_ext.pyd nao encontrado apos build em %BUILD_DIR%\Release\
    exit /b 1
)
echo   pyd OK: %NSOS_PYD%

REM ── 3) Stage ────────────────────────────────────────────────────────────
echo.
echo [3/7] Setting up staging directory...
if exist "%STAGE_DIR%" rmdir /s /q "%STAGE_DIR%"
mkdir "%STAGE_DIR%"
xcopy /Y "%HERE%trainer_main.py"          "%STAGE_DIR%\"  >nul
xcopy /Y "%HERE%pyinstaller_spec.spec"    "%STAGE_DIR%\"  >nul
echo   stage: %STAGE_DIR%

REM ── 4) Prepare data (download + bake) ──────────────────────────────────
echo.
echo [4/7] Running prepare_data.py (CulturaX=%CULTURAX_GB%GB)...
python "%HERE%prepare_data.py" --output "%BUILD_ROOT%" --culturax-gb %CULTURAX_GB% %COMPRESS_FLAG%
if errorlevel 1 (
    echo ERROR: prepare_data falhou.
    exit /b 1
)

REM ── 5) PyInstaller ──────────────────────────────────────────────────────
echo.
echo [5/7] Installing PyInstaller + zstandard...
python -m pip install --quiet --upgrade pyinstaller zstandard

echo.
echo [5/7 cont] Running PyInstaller...
set "NSOS_EXT_PYD=%NSOS_PYD%"
set "CUDA_DIR=%CUDA_PATH%"

pushd "%STAGE_DIR%"
python -m PyInstaller pyinstaller_spec.spec ^
    --workpath "%PYINSTALLER_WORK%" ^
    --distpath "%PYINSTALLER_WORK%\dist" ^
    --noconfirm
if errorlevel 1 (
    popd
    echo ERROR: PyInstaller falhou.
    exit /b 1
)
popd

set "DIST_DIR=%PYINSTALLER_WORK%\dist\OxtaTrainer"
if not exist "%DIST_DIR%\OxtaTrainer.exe" (
    echo ERROR: OxtaTrainer.exe nao gerado em %DIST_DIR%
    exit /b 1
)
echo   exe: %DIST_DIR%\OxtaTrainer.exe

REM ── 6) Assemble final folder ────────────────────────────────────────────
echo.
echo [6/7] Assembling final folder %FINAL_DIR%...
if exist "%FINAL_DIR%" rmdir /s /q "%FINAL_DIR%"
mkdir "%FINAL_DIR%"

xcopy /E /I /Y "%DIST_DIR%\*"          "%FINAL_DIR%\"          >nul
xcopy /E /I /Y "%BUILD_ROOT%\data"     "%FINAL_DIR%\data\"     >nul
xcopy /Y       "%HERE%README_FRIEND.txt" "%FINAL_DIR%\README.txt" >nul

REM Compute total size
set "TOTAL=0"
for /f "tokens=3" %%a in ('dir /s "%FINAL_DIR%" ^| findstr /R "arquivo(s)" ^| findstr /R "[0-9] bytes"') do (
    set "TOTAL=%%a"
)
echo   final folder ready
echo.
dir "%FINAL_DIR%" | findstr /R "Dir(s)\|arquivo(s)"

REM ── 7) Compress to single .7z ────────────────────────────────────────────
echo.
echo [7/7] Compressing to %FINAL_ARCHIVE%...
if defined SKIP_7Z (
    echo   7z not installed, skipping.  You can compress manually.
) else (
    if exist "%FINAL_ARCHIVE%" del "%FINAL_ARCHIVE%"
    7z a -t7z -mx=7 -m0=lzma2 "%FINAL_ARCHIVE%" "%FINAL_DIR%\*"
    if errorlevel 1 (
        echo WARNING: 7z compression failed.
    ) else (
        for %%A in ("%FINAL_ARCHIVE%") do echo   archive size: %%~zA bytes
    )
)

echo.
echo ============================================================
echo  DONE
echo ============================================================
echo  Folder:   %FINAL_DIR%
if not defined SKIP_7Z echo  Archive:  %FINAL_ARCHIVE%
echo.
echo  Para enviar ao amigo: faz upload do .7z para WeTransfer Pro
echo  ou Google Drive 2TB. Ele baixa, descompacta, roda OxtaTrainer.exe.
echo ============================================================

endlocal
