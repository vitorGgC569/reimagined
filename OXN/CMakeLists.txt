cmake_minimum_required(VERSION 3.10)
project(NSOS_CPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- AUTO-DETECÇÃO DE GPU (SMART MODE) ---
include(CMakeDependentOption)
include(CheckLanguage)

# 1. Tenta verificar se o compilador CUDA (nvcc) está disponível
# Removed hardcoded path to allow CMake/User to select version

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    set(CUDA_DETECTED ON)
    message(STATUS "=============================================")
    message(STATUS "🚀 HARDWARE: NVIDIA GPU DETECTADA")
    message(STATUS "   Compilador: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "   Modo: HYBRID (CPU + GPU)")
    message(STATUS "=============================================")
else()
    set(CUDA_DETECTED OFF)
    message(STATUS "=============================================")
    message(STATUS "🐢 HARDWARE: GPU NÃO ENCONTRADA (ou nvcc ausente)")
    message(STATUS "   Modo: CPU ONLY (BitNet AVX)")
    message(STATUS "=============================================")
endif()

# 2. Define a flag baseada na detecção
option(USE_CUDA "Enable CUDA support" ${CUDA_DETECTED})
option(USE_MPI "Enable MPI support for Distributed Training" OFF)
option(SANITIZE "Enable Address Sanitizer (ASAN)" OFF)

if(SANITIZE)
    message(STATUS "🛡️ MODO PARANOICO: Address & Undefined Sanitizer Ativado")
    if(MSVC)
        add_compile_options(/fsanitize=address)
    else()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Optimized Compiler Flags
if(MSVC)
    # Set global flags for C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /fp:fast")
    # For CUDA, we'll set them on the target specifically
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# 3. Lógica de Ativação CUDA
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(USE_CUDA)

    file(GLOB_RECURSE CUDA_SOURCES "nsos/src/cuda/*.cu")
else()
    set(CUDA_SOURCES "")
endif()

# 4. Lógica de Ativação MPI
if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    add_compile_definitions(USE_MPI)
    # Note: Target linking is done later
endif()

include_directories(nsos/include)

# Source files
file(GLOB_RECURSE SOURCES "nsos/src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*server.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*chat.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*benchmarks.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*bench_superiority.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*compiler_stub.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*bindings.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*fabric.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*dataloader.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*chrass_layer_backward.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*chrass_layer.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*chrass_layer_backward_v2.cpp$")

file(GLOB_RECURSE SIMD_SOURCES "nsos/src/simd/*.cpp")

# Create library
add_library(nsos_lib ${SOURCES} ${CUDA_SOURCES} ${SIMD_SOURCES})

target_include_directories(nsos_lib PUBLIC nsos/include)

if(USE_CUDA)
    target_include_directories(nsos_lib PRIVATE nsos/include/cuda ${CUDAToolkit_INCLUDE_DIRS})
    
    if(MSVC)
        target_compile_options(nsos_lib PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2 -Xcompiler=/arch:AVX2 -Xcompiler=/fp:fast -O3 --use_fast_math>
        )
    else()
        target_compile_options(nsos_lib PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        )
    endif()
    
    # HPC: Explicit SM 6.1 for GTX 1050 Ti (Pascal)
    # Using explicit arch instead of "native" to ensure correct targeting
    set_target_properties(nsos_lib PROPERTIES CUDA_ARCHITECTURES "61")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nsos_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET nsos_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

if(USE_MPI)
    target_link_libraries(nsos_lib PUBLIC ${MPI_CXX_LIBRARIES})
endif()

if(USE_CUDA)
    target_link_libraries(nsos_lib PRIVATE CUDA::cudart)
endif()

# Server
add_executable(nsos_server nsos/src/server.cpp)
target_link_libraries(nsos_server nsos_lib)
if(WIN32)
    target_link_libraries(nsos_server ws2_32)
endif()

# Benchmarks & Comprehensive Tests
add_executable(test_suite nsos/tests/test_suite.cpp)
target_link_libraries(test_suite nsos_lib)

add_executable(nsos_bench nsos/src/benchmarks.cpp)
target_link_libraries(nsos_bench nsos_lib)

add_executable(nsos_superiority nsos/src/bench_superiority.cpp)
target_link_libraries(nsos_superiority nsos_lib)

add_executable(nsos_circuit_trainer nsos/src/nsos_circuit_trainer.cpp)
target_link_libraries(nsos_circuit_trainer nsos_lib)

# Tests
enable_testing()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_bitlinear.cpp")
    add_executable(test_bitlinear nsos/tests/test_bitlinear.cpp)
    target_link_libraries(test_bitlinear nsos_lib)
    add_test(NAME BitLinearTest COMMAND test_bitlinear)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_mamba2.cpp")
    add_executable(test_mamba2 nsos/tests/test_mamba2.cpp)
    target_link_libraries(test_mamba2 nsos_lib)
    add_test(NAME Mamba2Test COMMAND test_mamba2)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_jamba.cpp")
    add_executable(test_jamba nsos/tests/test_jamba.cpp)
    target_link_libraries(test_jamba nsos_lib)
    add_test(NAME JambaTest COMMAND test_jamba)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_components.cpp")
    add_executable(test_components nsos/tests/test_components.cpp)
    target_link_libraries(test_components nsos_lib)
    add_test(NAME ComponentsTest COMMAND test_components)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_stability.cpp")
    add_executable(test_stability nsos/tests/test_stability.cpp)
    target_link_libraries(test_stability nsos_lib)
    add_test(NAME StabilityTest COMMAND test_stability)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_holographic_full.cpp")
    add_executable(test_holographic_full nsos/tests/test_holographic_full.cpp)
    target_link_libraries(test_holographic_full nsos_lib)
    add_test(NAME HolographicTest COMMAND test_holographic_full)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_v2_features.cpp")
    add_executable(test_v2_features nsos/tests/test_v2_features.cpp)
    target_link_libraries(test_v2_features nsos_lib)
    add_test(NAME V2FeaturesTest COMMAND test_v2_features)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_vulnerability_fixes.cpp")
    add_executable(test_vulnerability_fixes nsos/tests/test_vulnerability_fixes.cpp)
    target_link_libraries(test_vulnerability_fixes nsos_lib)
    add_test(NAME VulnerabilityFixesTest COMMAND test_vulnerability_fixes)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_moe_training.cpp")
    add_executable(test_moe_training nsos/tests/test_moe_training.cpp)
    target_link_libraries(test_moe_training nsos_lib)
    add_test(NAME MoETrainingTest COMMAND test_moe_training)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_kan_backward.cpp")
    add_executable(test_kan_backward nsos/tests/test_kan_backward.cpp)
    target_link_libraries(test_kan_backward nsos_lib)
    add_test(NAME KANBackwardTest COMMAND test_kan_backward)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nsos/tests/test_ttt_layer_kernel.cpp")
    add_executable(test_ttt nsos/tests/test_ttt_layer_kernel.cpp)
    target_link_libraries(test_ttt PRIVATE nsos_lib)

add_executable(test_mcts_v3 nsos/tests/test_mcts_reasoning_v3.cpp)
target_link_libraries(test_mcts_v3 PRIVATE nsos_lib)
    add_test(NAME TTTKernelTest COMMAND test_ttt)
endif()

add_executable(chat_app nsos/src/chat.cpp nsos/src/tokenizer.cpp)
target_link_libraries(chat_app nsos_lib)

# Python Bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Robust Pybind11 Fetch
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL      https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(nsos_ext nsos/src/bindings.cpp)
target_link_libraries(nsos_ext PRIVATE nsos_lib)
target_include_directories(nsos_ext PRIVATE nsos/include)

if(WIN32)
    if(MINGW)
        target_link_options(nsos_lib PRIVATE "-static-libgcc" "-static-libstdc++")
        target_link_options(nsos_ext PRIVATE "-static-libgcc" "-static-libstdc++")
    endif()
endif()
