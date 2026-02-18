cmake_minimum_required(VERSION 3.18)
project(NSOS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# Include Directories
include_directories(include)
include_directories(include/nsos)

# Sources - STRICTLY STABLE CORE ONLY
set(NSOS_LIB_SOURCES
    src/bindings.cpp
    src/tensor.cpp
    src/nsos_math.cpp
    src/jamba.cpp
    src/mamba2.cpp
    src/bitlinear.cpp
    src/embedding.cpp
    src/kan.cpp
    src/ttt_layer.cpp
    src/optimizers.cpp
    src/trainer.cpp
    src/tokenizer.cpp
    src/nsos_sdk.cpp
    src/globals.cpp
)

# Compiler Flags for MSVC (Stability)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /openmp /wd4849)
else()
    add_compile_options(-O3 -march=native -fopenmp)
endif()

find_package(OpenMP REQUIRED)

# Optional CUDA
option(NSOS_ENABLE_CUDA "Enable CUDA" ON)
if(NSOS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        list(APPEND NSOS_LIB_SOURCES
            src/cuda/kernels.cu
            src/cuda/persistent_kernel.cu
            src/cuda/mamba_kernels.cu
        )
        add_definitions(-DUSE_CUDA)
    endif()
endif()

pybind11_add_module(nsos_ext ${NSOS_LIB_SOURCES})
target_link_libraries(nsos_ext PRIVATE OpenMP::OpenMP_CXX)

if(NSOS_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(nsos_ext PRIVATE CUDA::cudart)
endif()
