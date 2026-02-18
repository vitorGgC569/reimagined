cmake_minimum_required(VERSION 3.10)
project(KernelOpen)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(src)
include_directories(src/common)
include_directories(include/aion) # Add AION includes

# Define sources explicitly
set(KERNEL_SOURCES
    src/kernel_entry.cpp
    # AION Sources (Ingestion Engine)
    src/aion/BitPacking.cpp
    src/aion/Hilbert.cpp
    src/aion/LinearModel.cpp
    src/aion/RingBuffer.cpp
    src/aion/SmartLoader.cpp
    # Note: bindings.cpp is for Pybind11, usually separate, but if we link statically
    # we might just ignore it unless we expose AION directly via KernelOpen lib.
    # For now, we compile the core logic.
)

add_library(kernelopen STATIC ${KERNEL_SOURCES})

# Expose include directories for consumers
target_include_directories(kernelopen PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
