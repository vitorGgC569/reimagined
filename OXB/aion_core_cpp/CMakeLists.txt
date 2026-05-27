cmake_minimum_required(VERSION 3.10)
project(AionCore)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -fPIC)
endif()

# Detect Linux
if(UNIX AND NOT APPLE)
    add_definitions(-DAION_PLATFORM_LINUX)
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")
# SmartLoader uses Linux mmap/unistd.h; exclude on Windows.
if(WIN32)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/SmartLoader.cpp")
endif()

add_executable(aion_test ${SOURCES} "tests/main_test.cpp")
add_executable(aion_bench ${SOURCES} "benchmarks/bench_main.cpp")
add_executable(aion_validation ${SOURCES} "tests/validation_battery.cpp")
if(NOT WIN32)
    add_executable(aion_loader_bench ${SOURCES} "benchmarks/bench_loader.cpp")
endif()

# Pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    execute_process(COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())" OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 REQUIRED)
endif()

pybind11_add_module(aion_core "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp" ${SOURCES})
