cmake_minimum_required(VERSION 3.18)
project(PantheonStabillish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PANTHEON_BUILD_PYTHON "Build the pantheon Python extension" ON)

add_subdirectory(KernelOpen)

if (PANTHEON_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 REQUIRED)

    set(PANTHEON_SOURCES
        bindings/python_bindings.cpp
        src/pantheon_engine.cpp
        src/pantheon/cognition/chain_of_thought.cpp
        src/pantheon/physics/gradient_matching.cpp
        src/pantheon/response/logit_distillation.cpp
        src/pantheon/structure/contrastive.cpp
        src/pantheon/structure/msdcrd.cpp
        src/pantheon/structure/relational.cpp
    )

    pybind11_add_module(pantheon ${PANTHEON_SOURCES})
    target_include_directories(pantheon PRIVATE include KernelOpen/src KernelOpen/include)
    target_link_libraries(pantheon PRIVATE kernelopen)
endif()
