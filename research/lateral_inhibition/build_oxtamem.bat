@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 oxtamem_demo.cpp /Fe:oxtamem_demo.exe
if errorlevel 1 exit /b 1
.\oxtamem_demo.exe
