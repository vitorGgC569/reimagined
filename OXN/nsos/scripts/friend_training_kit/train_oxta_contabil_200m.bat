@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  Oxta Contábil 200M — RTX 2080 Ti One-Shot Trainer
REM ─────────────────────────────────────────────────────────────────────────
REM
REM Versão: 1.0  (target: hybrid_rtx2080ti_200m_chinchilla25b profile)
REM
REM Este script executa o pipeline completo de treino do modelo Oxta
REM Contábil 200M:
REM   1. Verifica pré-requisitos (Python, CUDA, NVIDIA driver)
REM   2. Confirma GPU é compatível (sm_75 Turing — RTX 2080/2080 Ti/Titan/T4)
REM   3. Build do nsos_ext (~15-25 min na primeira vez)
REM   4. Download dos datasets brasileiros (~30GB se passar --full)
REM   5. Treino do modelo (~140 horas / ~6 dias 24/7)
REM   6. Salva pack final em .\oxta_packs\
REM
REM Uso:
REM   train_oxta_contabil_200m.bat              REM datasets curtos (~5GB)
REM   train_oxta_contabil_200m.bat --full       REM datasets completos (~30GB)
REM   train_oxta_contabil_200m.bat --resume     REM retoma do último checkpoint
REM
REM Pré-requisitos no PC:
REM   - Windows 10/11 64-bit
REM   - NVIDIA driver >= 535
REM   - CUDA Toolkit 12.x (12.5+ recomendado)
REM   - Visual Studio 2022 + workload "Desktop development with C++"
REM   - CMake 3.18+
REM   - Python 3.11+
REM   - Git
REM   - Ao menos 80GB de espaço livre (build + datasets + checkpoints)
REM
REM Tempo estimado:
REM   Setup primeira vez:   30-60 min  (build NSOS + download datasets)
REM   Treino completo:      ~140h      (6 dias 24/7, ou 9 dias com PC usado de dia)
REM   Total wall clock:     ~7-10 dias
REM ─────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

REM ── Argumentos ──────────────────────────────────────────────────────────
set DATA_MODE=fast
set RESUME_FLAG=
for %%i in (%*) do (
    if "%%i"=="--full" set DATA_MODE=full
    if "%%i"=="--resume" set RESUME_FLAG=--resume
)

echo.
echo ============================================================
echo  Oxta Contabil 200M — RTX 2080 Ti One-Shot Trainer
echo ============================================================
echo  data mode:       %DATA_MODE%
if not "%RESUME_FLAG%"=="" echo  resume mode:     ON
echo  profile:         hybrid_rtx2080ti_200m_chinchilla25b
echo  estimated time:  ~6 days 24/7 (after setup)
echo ============================================================
echo.

REM ── 1) Verifica pre-requisitos ──────────────────────────────────────────
echo [1/6] Verificando pre-requisitos...

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python nao encontrado. Instala Python 3.11+ e rode de novo.
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo   Python:        %PY_VER%

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: CMake nao encontrado. Instala CMake 3.18+ e adiciona ao PATH.
    exit /b 1
)
for /f "tokens=3" %%v in ('cmake --version ^| findstr /R "version"') do set CMAKE_VER=%%v
echo   CMake:         %CMAKE_VER%

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvidia-smi nao encontrado. Verifica driver NVIDIA esta instalado.
    exit /b 1
)
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader') do set GPU_NAME=%%g
echo   GPU:           %GPU_NAME%

REM Confirmacao explicita de GPU compativel
echo %GPU_NAME% | findstr /I "2080 2070 2060 T4 Titan RTX Quadro RTX" >nul
if errorlevel 1 (
    echo WARNING: GPU detectada nao e claramente Turing sm_75.
    echo   O profile hybrid_rtx2080ti_200m foi calibrado para sm_75 com 11GB.
    echo   Se a GPU tem mais VRAM, tudo bem.  Se tem menos, vai dar OOM.
    set /p PROCEED="Continuar mesmo assim? [y/N] "
    if /I not "!PROCEED!"=="y" exit /b 1
)

REM ── 2) Setup do ambiente Python ─────────────────────────────────────────
echo.
echo [2/6] Instalando dependencias Python...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet "pybind11>=2.10" numpy "datasets>=2.16" huggingface_hub tqdm matplotlib

REM ── 3) Build do NSOS ────────────────────────────────────────────────────
echo.
echo [3/6] Building NSOS C++ runtime (15-25 min primeira vez)...

set BUILD_DIR=%~dp0..\..\build-rtx2080ti
set NSOS_SRC=%~dp0..\..\

REM Detecta CUDA arch — 2080 Ti = 75
set CUDA_ARCH=75

REM Se build dir ja existe, faz incremental.  Senao, configura.
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo   Configurando build em %BUILD_DIR% ...
    cmake -S "%NSOS_SRC%" -B "%BUILD_DIR%" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DNSOS_ENABLE_CUDA=ON ^
        -DNSOS_BUILD_PYTHON=ON ^
        -DNSOS_BUILD_TESTS=OFF ^
        -DNSOS_BUILD_CLI=OFF ^
        -DNSOS_BUILD_API=OFF ^
        -DNSOS_BUILD_OXTAMEM=OFF ^
        -DNSOS_CUDA_ARCHITECTURES=%CUDA_ARCH%
    if errorlevel 1 (
        echo ERROR: cmake configure failed.
        exit /b 1
    )
)

echo   Compilando (Release)...
cmake --build "%BUILD_DIR%" --config Release -j
if errorlevel 1 (
    echo ERROR: build failed.
    exit /b 1
)
echo   Build OK.

REM ── 4) Download dos datasets ────────────────────────────────────────────
echo.
echo [4/6] Baixando datasets brasileiros (mode=%DATA_MODE%)...

set DATA_ROOT=%~dp0..\..\..\..\..\oxta_data
if not exist "%DATA_ROOT%" mkdir "%DATA_ROOT%"

if "%DATA_MODE%"=="full" (
    set CULTURAX_BYTES=30GB
    set CULTURAX_SHARDS=100
) else (
    set CULTURAX_BYTES=5GB
    set CULTURAX_SHARDS=20
)

python "%~dp0..\oxta_contabil\download_datasets.py" ^
    --output-dir "%DATA_ROOT%" ^
    --include culturax_ptbr,br_taxqa,bacen_faq,lener_br ^
    --target-bytes %CULTURAX_BYTES% ^
    --culturax-shards %CULTURAX_SHARDS%
if errorlevel 1 (
    echo WARNING: alguns datasets falharam download.  Continuando com o que baixou.
)

REM ── 5) Bundle ───────────────────────────────────────────────────────────
echo.
echo [5/6] Verificando distillation bundle...

set BUNDLE_DIR=%~dp0..\distillation_bundle_v11
if not exist "%BUNDLE_DIR%\tokenizer_8192.ox3" (
    echo ERROR: bundle nao encontrado em %BUNDLE_DIR%.
    echo   Precisa do tokenizer + curriculum data do v11.
    echo   Pede pro Vitor te enviar a pasta distillation_bundle_v11/.
    exit /b 1
)
echo   Bundle OK: %BUNDLE_DIR%

REM ── 6) Treino ───────────────────────────────────────────────────────────
echo.
echo [6/6] Iniciando treino 200M ^(~140h / ~6 dias 24/7^)...

set RUN_DIR=%~dp0..\..\..\..\..\oxta_packs\rtx2080ti_200m_%date:~6,4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%
set RUN_DIR=%RUN_DIR: =0%
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

REM Power-limit a 220W reduz temp 5-8C, perde so ~5% perf (opcional)
nvidia-smi -pl 220 >nul 2>&1

set NSOS_BUILD_DIR=%BUILD_DIR%
set NSOS_MIXED_PRECISION=bf16
set NSOS_USE_LUT_SIMD=1
set PYTHONUNBUFFERED=1

python -u "%~dp0..\train_curriculum.py" ^
    --profile hybrid_rtx2080ti_200m_chinchilla25b ^
    --bundle-dir "%BUNDLE_DIR%" ^
    --build-dir "%BUILD_DIR%" ^
    --run-dir "%RUN_DIR%" ^
    --device gpu ^
    --checkpoint-every-steps 100 %RESUME_FLAG%

if errorlevel 1 (
    echo.
    echo Treino terminou com erro.  Checkpoint salvo em:
    echo   %RUN_DIR%
    echo Pra retomar: train_oxta_contabil_200m.bat --resume
    exit /b 1
)

echo.
echo ============================================================
echo  Treino COMPLETO
echo ============================================================
echo  Pack final em:   %RUN_DIR%\final_edge_linear.nsos
echo  Tamanho:
for %%A in ("%RUN_DIR%\final_edge_linear.nsos") do echo                   %%~zA bytes
echo.
echo  Compartilha o arquivo final_edge_linear.nsos com o Vitor.
echo  Esse e o "cerebro" do Oxta Contabil 200M treinado.
echo ============================================================

endlocal
