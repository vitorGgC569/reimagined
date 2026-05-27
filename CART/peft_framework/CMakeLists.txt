cmake_minimum_required(VERSION 3.10)
project(CARTPeft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

include_directories(include)

# All implementation files except bindings (pybind11-only) and main (own exe).
set(CART_SOURCES
    src/BOFT.cpp
    src/DoRA.cpp
    src/FullFinetuning.cpp
    src/IA3.cpp
    src/KAN.cpp
    src/LinAlg.cpp
    src/LoRA.cpp
    src/LoRA_GA.cpp
    src/MiSS.cpp
    src/SSM.cpp
    src/Tensor.cpp
    src/TensorOps.cpp
    src/TransMamba.cpp
    src/TurboFusion.cpp
)

# Existing demo binary (src/main.cpp).
add_executable(cart_demo ${CART_SOURCES} src/main.cpp)

# Existing test runner (run_tests.cpp at peft_framework root).
add_executable(cart_run_tests ${CART_SOURCES} run_tests.cpp)

# New extreme validation battery (tests/validation_battery.cpp).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/validation_battery.cpp")
    add_executable(cart_validation ${CART_SOURCES} tests/validation_battery.cpp)
endif()
