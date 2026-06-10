@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 rescue_demo.cpp /Fe:rescue_demo.exe
