@echo off
setlocal enabledelayedexpansion

echo [NSOS] Building SRE Engineering Suite (V11-V40)

:: Check for compiler
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] MSVC Compiler (cl.exe) not found.
    echo Please run this from a Visual Studio Developer Command Prompt.
    exit /b 1
)

:: Include Paths
set INC_DIR=..\include
set SRC_DIR=..\src

:: Compile verify_sre_engineering.cpp
:: Flags: /EHsc (Exceptions), /std:c++20 (Modern C++), /O2 (Optimize)
echo [Compiling...]
cl /EHsc /std:c++20 /I%INC_DIR% /Fe:verify_sre.exe verify_sre_engineering.cpp ^
   ..\src\nsos_mpi.cpp ..\src\tensor.cpp ..\src\globals.cpp ..\src\memory_system.cpp ^
   /link /SUBSYSTEM:CONSOLE

if %errorlevel% neq 0 (
    echo [Fail] Compilation failed.
    exit /b %errorlevel%
)

echo [Running SRE Suite...]
verify_sre.exe

endlocal
