@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  finalize_bundle.bat — etapas 6 e 7 do build_standalone.bat isoladas
REM ─────────────────────────────────────────────────────────────────────────
REM
REM Use quando o .pyd ja foi compilado, o prepare_data.py ja rodou (gerou
REM _build/data/), e o PyInstaller ja produziu _build/pyinstaller_work/dist/
REM OxtaTrainer/.  So executa:
REM   - etapa 6: copia dist/OxtaTrainer/* + _build/data/ + README_FRIEND.txt
REM              para OxtaTrainer_v1/
REM   - etapa 7: comprime OxtaTrainer_v1/ -> OxtaTrainer_v1.7z
REM
REM Sem rebuild de nada.  Roda em ~30-40 min (7z e o gargalo).
REM
REM Usage:
REM   finalize_bundle.bat
REM ─────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

set "HERE=%~dp0"
set "BUILD_ROOT=%HERE%_build"
set "PYINSTALLER_WORK=%BUILD_ROOT%\pyinstaller_work"
set "DIST_DIR=%PYINSTALLER_WORK%\dist\OxtaTrainer"
set "DATA_DIR=%BUILD_ROOT%\data"
set "FINAL_DIR=%HERE%OxtaTrainer_v1"
set "FINAL_ARCHIVE=%HERE%OxtaTrainer_v1.7z"

echo.
echo ============================================================
echo  finalize_bundle.bat
echo ============================================================
echo  Stage:    %BUILD_ROOT%
echo  Source:   %DIST_DIR%
echo  Data:     %DATA_DIR%
echo  Output:   %FINAL_DIR%
echo  Archive:  %FINAL_ARCHIVE%
echo ============================================================

REM ── Pre-flight ─────────────────────────────────────────────────────────
if not exist "%DIST_DIR%\OxtaTrainer.exe" (
    echo ERRO: %DIST_DIR%\OxtaTrainer.exe nao encontrado.
    echo   Rode build_standalone.bat primeiro para gerar o bundle do PyInstaller.
    exit /b 1
)
if not exist "%DATA_DIR%\manifest.json" (
    echo ERRO: %DATA_DIR%\manifest.json nao encontrado.
    echo   Rode prepare_data.py primeiro para gerar data/.
    exit /b 1
)
if not exist "%DATA_DIR%\bundle\tokenizer_8192.ox3" (
    echo ERRO: tokenizer nao encontrado em %DATA_DIR%\bundle\
    echo   Rode prepare_data.py de novo apos copiar distillation_bundle_v* para OXN/nsos/scripts/
    exit /b 1
)

REM ── Etapa 6: assemble OxtaTrainer_v1/ ─────────────────────────────────
echo.
echo [6/7] Montando %FINAL_DIR% ...
if exist "%FINAL_DIR%" rmdir /s /q "%FINAL_DIR%"
mkdir "%FINAL_DIR%"

xcopy /E /I /Y /Q "%DIST_DIR%\*"             "%FINAL_DIR%\"
if errorlevel 1 (echo ERRO: copia do dist falhou & exit /b 2)

xcopy /E /I /Y /Q "%DATA_DIR%"               "%FINAL_DIR%\data\"
if errorlevel 1 (echo ERRO: copia do data falhou & exit /b 2)

if exist "%HERE%README_FRIEND.txt" (
    xcopy /Y /Q "%HERE%README_FRIEND.txt"     "%FINAL_DIR%\README.txt"  >nul
) else (
    echo AVISO: README_FRIEND.txt nao encontrado em %HERE%
)
echo   ok

REM Show what we assembled
echo.
echo === Conteudo de %FINAL_DIR% ===
dir "%FINAL_DIR%" | findstr /R "Dir(s)\|arquivo(s)"

REM ── Etapa 7: comprime para .7z ──────────────────────────────────────
echo.
echo [7/7] Comprimindo para %FINAL_ARCHIVE% ...
where 7z >nul 2>&1
if errorlevel 1 (
    echo AVISO: 7z nao encontrado no PATH.
    echo   Instale 7-Zip e adicione ao PATH, ou compacte manualmente.
    echo   Pasta a comprimir: %FINAL_DIR%
    goto :done
)

if exist "%FINAL_ARCHIVE%" (
    echo   Removendo archive antigo: %FINAL_ARCHIVE%
    del "%FINAL_ARCHIVE%"
)

7z a -t7z -mx=7 -m0=lzma2 -mmt=on "%FINAL_ARCHIVE%" "%FINAL_DIR%\*"
if errorlevel 1 (
    echo ERRO: 7z falhou.
    exit /b 3
)

for %%A in ("%FINAL_ARCHIVE%") do echo   archive: %%~zA bytes (%%~zA / 1073741824 = ~GB)

:done
echo.
echo ============================================================
echo  DONE
echo ============================================================
echo  Folder:   %FINAL_DIR%
if exist "%FINAL_ARCHIVE%" echo  Archive:  %FINAL_ARCHIVE%
echo.
echo  Para enviar ao amigo: upload do .7z (~17GB) para WeTransfer Pro
echo  ou Google Drive 2TB.  Ele baixa, descompacta com 7-Zip, da
echo  duplo-clique em OxtaTrainer.exe.
echo ============================================================

endlocal
