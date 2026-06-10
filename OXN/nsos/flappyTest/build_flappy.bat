@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /MD /openmp /std:c++17 /EHsc /I..\include /I..\include\nsos flappy_test.cpp ..\build-validate\nsos_core.lib /Fe:flappy_test.exe
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
