@echo off
setlocal

:: 1. Setup Build Directory
if not exist build mkdir build
cd build

:: 2. Check for Visual Studio (Optional, CMake usually finds it)
:: If you need to force a generator:
:: cmake -G "Visual Studio 17 2022" -A x64 ..

:: 3. Configure CMake
:: Auto-detect Python is handled by CMake find_package(Python3)
:: We explicitly enable CUDA and MPI if available (Smart Mode)
echo [Configuring...]
cmake .. -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_CUDA=AUTO ^
    -DUSE_MPI=AUTO

if %errorlevel% neq 0 (
    echo [Error] CMake Configuration Failed.
    exit /b %errorlevel%
)

:: 4. Build
echo [Building...]
cmake --build . --config Release --parallel 4
if %errorlevel% neq 0 (
    echo [Error] Build Failed.
    exit /b %errorlevel%
)

:: 5. Install Python Extension (Local)
echo [Installing Python Extension...]
:: Move the .pyd file to the root or a python lib folder
:: Usually located in Release/nsos_ext.cp3x-win_amd64.pyd
copy Release\nsos_ext*.pyd ..\
if %errorlevel% neq 0 (
    echo [Warning] Could not copy .pyd extension. Check build/Release folder.
) else (
    echo [Success] Python extension installed to project root.
    echo Try: python -c "import nsos_ext; print(help(nsos_ext))"
)

echo [Build Success!]
echo Run tests with: ctest -C Release
endlocal
