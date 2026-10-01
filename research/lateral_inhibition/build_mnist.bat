@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 mnist_showdown.cpp /Fe:mnist_showdown.exe
if errorlevel 1 exit /b 1
.\mnist_showdown.exe
