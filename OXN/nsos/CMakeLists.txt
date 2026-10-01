cmake_minimum_required(VERSION 3.18)
project(NSOS LANGUAGES CXX)

# CMake 4.4 can double-transcode MSVC's UTF-8 Portuguese /showIncludes
# prefix while detecting the compiler.  Ninja then misses every header
# dependency, so an incremental build can link objects compiled against
# different class layouts.  Correct only that known-bad detected value;
# other compiler languages and generators keep CMake's native detection.
if (MSVC
    AND CMAKE_GENERATOR MATCHES "Ninja"
    AND CMAKE_CXX_CL_SHOWINCLUDES_PREFIX
        STREQUAL "Observa├º├úo: incluindo arquivo: ")
    set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX
        "Observação: incluindo arquivo: ")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX
        "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NSOS_HIP_SCALAR_DP4A "Use the scalar signed INT8 dot oracle on HIP" OFF)
if(NSOS_HIP_SCALAR_DP4A)
    add_compile_definitions(NSOS_HIP_SCALAR_DP4A=1)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    # NSOS reads process environment controls through std::getenv and copies
    # values immediately under the runtime-policy contract; no returned pointer
    # is retained across an environment mutation. The UCRT deprecation is a
    # Microsoft portability warning, not a safety defect in this usage. Keep it
    # disabled so real diagnostics (notably ignored HIP errors) remain visible.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(NSOS_GPU_BACKEND "AUTO" CACHE STRING
    "GPU backend: AUTO, NONE, CUDA, or HIP")
set_property(CACHE NSOS_GPU_BACKEND PROPERTY STRINGS AUTO NONE CUDA HIP)
option(NSOS_ENABLE_CUDA
       "Allow AUTO to select CUDA (kept for backward compatibility)" ON)
option(NSOS_ENABLE_HIP "Allow AUTO to select AMD HIP/ROCm" ON)
option(NSOS_BUILD_PYTHON "Build the Python extension" ON)
option(NSOS_BUILD_TESTS "Build curated native smoke tests" ON)
set(NSOS_MAMBA3_UPSTREAM_ARTIFACT_DIR "" CACHE PATH
    "Optional pinned Mamba3 CPU reference/dependencies for upstream parity tests")
option(NSOS_BUILD_CIRCUIT_TOOLS "Build the circuit trainer executable" ON)
option(NSOS_BUILD_CLI "Build the official NSOS CLI" ON)
option(NSOS_BUILD_API "Build the official NSOS HTTP API server" ON)
option(NSOS_BUILD_OXTAMEM "Build the copied OxtaMem engine with Cargo when available" ON)
# OPT-IN profiler: builds nsos_profiler.lib + nsos_profile_tool exe.
# Default OFF so production / Colab training builds DON'T link the
# profiler code.  Enable explicitly with -DNSOS_BUILD_PROFILER=ON
# when running architectural analysis sessions.
option(NSOS_BUILD_PROFILER "Build the standalone inference profiler tool" OFF)
option(NSOS_ENABLE_TURBOQUANT "Enable external TurboQuant integration when configured" ON)
option(NSOS_CUDA_ALLOW_UNSUPPORTED_COMPILER "Allow nvcc to compile with an unsupported host MSVC version" OFF)
# Per-thread default stream: required for CUDA-graph decode (the graph capture
# needs kernels on a capturable stream; the legacy default stream cannot be
# captured).  Changes stream semantics globally — kernels/API calls from
# different threads no longer serialize through the legacy stream — so it is
# OFF by default; the T4 notebook builds with it ON and runs the full parity/
# gradcheck gate suite under it before measuring graph decode.
option(NSOS_CUDA_PTDS "Compile CUDA with per-thread default stream (enables NSOS_CUDA_GRAPH_DECODE)" OFF)
option(NSOS_ENABLE_NATIVE_OPTIMIZATIONS "Enable host-specific CPU tuning flags such as -march=native" OFF)
option(NSOS_ENABLE_X86_AVX2_KERNELS "Build runtime-dispatched AVX2 CPU kernels on x86_64" ON)
set(NSOS_SANITIZERS "" CACHE STRING "Semicolon-separated sanitizers to enable for CPU builds (address;undefined;thread)")

set(NSOS_CUDA_TOOLKIT_ROOT "" CACHE PATH "Preferred CUDA toolkit root directory")
set(NSOS_CUDA_ARCHITECTURES "61;75;80;86" CACHE STRING "CUDA architectures for NSOS (e.g. 61;75;86)")
set(NSOS_HIP_ROOT "" CACHE PATH "Preferred AMD ROCm/HIP SDK root directory")
set(NSOS_HIP_ARCHITECTURES "gfx1102" CACHE STRING
    "AMD GPU architectures for NSOS (the RX 7600 is gfx1102)")
set(NSOS_HIP_DEVICE_LIB_PATH "" CACHE PATH
    "Directory containing ROCm device bitcode libraries such as ocml.bc")
set(NSOS_ZIG_EXECUTABLE "" CACHE FILEPATH "Path to zig executable used to build TurboQuant")
set(NSOS_TURBOQUANT_DIR "" CACHE PATH "Path to TurboQuant source directory")
set(NSOS_TURBOQUANT_LIB "" CACHE FILEPATH "Path to a prebuilt TurboQuant static library")
set(NSOS_OXTAMEM_DIR "${CMAKE_CURRENT_LIST_DIR}/../../modules/oxtamem/oxta_engine" CACHE PATH "Path to the copied OxtaMem Rust engine")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(NSOS_CORE_SOURCES
    src/tensor.cpp
    src/gpu_backend.cpp
    src/gpu_execution.cpp
    src/gpu_projection_group.cpp
    src/gpu_kv_cache.cpp
    src/gpu_gemm_provider.cpp
    src/nsos_math.cpp
    src/jamba.cpp
    src/jamba_utils.cpp
    src/mamba2.cpp
    src/mamba3_reference.cpp
    src/mamba3_preprocessing.cpp
    src/gpu_mamba3_siso.cpp
    src/gpu_mamba3_projected.cpp
    src/gpu_attention_training.cpp
    src/mamba3_layer.cpp
    src/bitlinear.cpp
    src/bitlinear_cce.cpp
    src/tiled_cross_entropy.cpp
    src/bitnet_adapter.cpp
    src/lut_tmac.cpp
    src/lut_cas.cpp
    src/embedding.cpp
    src/kan.cpp
    src/ttt_layer.cpp
    src/ttt_recurrence.cpp
    src/gpu_moe_training.cpp
    src/gpu_sparse_adam.cpp
    src/trainer.cpp
    src/runtime_execution_identity.cpp
    src/optimizer_4bit.cpp
    src/sha256.cpp
    src/layer_audit.cpp
    src/tokenizer.cpp
    src/nsos_sdk.cpp
    src/nsos_mpi.cpp
    src/globals.cpp
    src/holographic.cpp
    src/chrass_layer_v2.cpp
    src/mcts_reasoning.cpp
    src/causal_memory_store.cpp
    src/oxtamem_ffi.cpp
    src/http_api_server.cpp
    src/determinism.cpp
    src/components.cpp
    src/memory_system.cpp
    src/dataloader_v2.cpp
    src/fabric_v2.cpp
    src/smart_loader.cpp
    src/sprecher_kan.cpp
    src/simd/hadamard.cpp
    src/coordinator.cpp
    src/self_healer.cpp
    src/numerical_guard.cpp
    src/turboquant_fallback.cpp
    src/bitnet_gpu_dispatch.cpp
    src/sparse_attention.cpp
 )

if (MSVC)
    # Let CMake's selected configuration own /Od vs /O2. Forcing /O2 into a
    # Debug build conflicts with MSVC's /RTC1 and made the documented test
    # configuration impossible to compile.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
else()
    # Release configurations already provide optimization flags; OpenMP flags
    # come from the imported OpenMP::OpenMP_CXX target below.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x64)$")
        # The core SIMD layer (nsos_simd.h SimdPacket<float>) uses AVX/FMA
        # intrinsics, so the default core requires a Haswell-class CPU.  Build
        # with -DNSOS_BASELINE_X86=ON for a portable binary that runs on pre-AVX
        # CPUs: nsos_simd.h then falls back to the scalar packet path and the
        # AVX2 BitLinear kernels stay isolated + runtime-dispatched.
        if (NOT NSOS_BASELINE_X86)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
        endif()
    endif()
    if (NSOS_ENABLE_NATIVE_OPTIMIZATIONS)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
    if (NSOS_SANITIZERS AND NOT NSOS_ENABLE_CUDA)
        string(REPLACE ";" "," NSOS_SANITIZER_FLAGS "${NSOS_SANITIZERS}")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${NSOS_SANITIZER_FLAGS}>)
        add_link_options(-fsanitize=${NSOS_SANITIZER_FLAGS})
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    endif()
endif()

find_package(OpenMP REQUIRED)
add_library(nsos_core STATIC ${NSOS_CORE_SOURCES})
target_link_libraries(nsos_core PUBLIC OpenMP::OpenMP_CXX)
if (NSOS_BUILD_TESTS)
    # Explicit, disabled-by-default fault points used only by the CTest
    # recovery matrix. Production configurations that omit tests expose no
    # injection API or branch in checkpoint loading.
    target_compile_definitions(nsos_core PUBLIC NSOS_ENABLE_TEST_HOOKS=1)
endif()
target_include_directories(nsos_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/nsos>
)

if (NSOS_ENABLE_X86_AVX2_KERNELS AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x64)$")
    target_sources(nsos_core PRIVATE
        src/bitnet_adapter_avx2.cpp
        # SIMD Gap #1 + #4 (2026-05-17): row-level activation quantize
        # kernels for BitLinear::quantize_activations_bitnet and the
        # weight row-sum reduction for BitLinear pack.  Live in their
        # own TU so we can apply -mavx2 / /arch:AVX2 to just this file
        # without forcing the whole target onto AVX2 (which would
        # break non-AVX2 distribution builds).
        src/bitlinear_quantize_avx2.cpp
    )
    target_compile_definitions(nsos_core PUBLIC NSOS_ENABLE_AVX2_KERNELS=1)
    if (MSVC)
        set_source_files_properties(src/bitnet_adapter_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(src/bitlinear_quantize_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/bitnet_adapter_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2")
        set_source_files_properties(src/bitlinear_quantize_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2")
    endif()
endif()

if (NSOS_ENABLE_TURBOQUANT)
    set(NSOS_TURBOQUANT_LIB_PATH "")

    if (NSOS_TURBOQUANT_LIB)
        set(NSOS_TURBOQUANT_LIB_PATH "${NSOS_TURBOQUANT_LIB}")
    elseif (NSOS_TURBOQUANT_DIR AND NSOS_ZIG_EXECUTABLE)
        if (MSVC)
            set(NSOS_TURBOQUANT_LIB_PATH "${NSOS_TURBOQUANT_DIR}/turboquant_c.lib")
        else()
            set(NSOS_TURBOQUANT_LIB_PATH "${NSOS_TURBOQUANT_DIR}/libturboquant_c.a")
        endif()

        add_custom_target(turboquant_build
            COMMAND ${NSOS_ZIG_EXECUTABLE} build-lib src/c_api.zig -O ReleaseFast -femit-bin=${NSOS_TURBOQUANT_LIB_PATH} -lc
            WORKING_DIRECTORY ${NSOS_TURBOQUANT_DIR}
            COMMENT "Building TurboQuant external library"
        )
    endif()

    if (NSOS_TURBOQUANT_LIB_PATH)
        add_library(turboquant_c STATIC IMPORTED)
        set_target_properties(turboquant_c PROPERTIES IMPORTED_LOCATION ${NSOS_TURBOQUANT_LIB_PATH})
        if (TARGET turboquant_build)
            add_dependencies(turboquant_c turboquant_build)
        endif()
        target_compile_definitions(nsos_core PUBLIC NSOS_USE_EXTERNAL_TURBOQUANT=1)
        target_link_libraries(nsos_core PUBLIC turboquant_c)
        message(STATUS "Using external TurboQuant library: ${NSOS_TURBOQUANT_LIB_PATH}")
    else()
        message(STATUS "TurboQuant external library not configured; using built-in fallback implementation.")
    endif()
endif()

string(TOUPPER "${NSOS_GPU_BACKEND}" NSOS_GPU_BACKEND_REQUESTED)
if (NOT NSOS_GPU_BACKEND_REQUESTED MATCHES "^(AUTO|NONE|CUDA|HIP)$")
    message(FATAL_ERROR
        "NSOS_GPU_BACKEND must be AUTO, NONE, CUDA, or HIP "
        "(got '${NSOS_GPU_BACKEND}')")
endif()

set(NSOS_GPU_BACKEND_RESOLVED "NONE")
set(NSOS_GPU_ENABLED OFF)
set(NSOS_GPU_SOURCES
    src/cuda/kernels.cu
    src/cuda/mamba_kernels.cu
    src/cuda/mamba3_siso_kernels.cu
    src/cuda/mamba3_preprocess_kernels.cu
    src/cuda/mamba3_layer_kernels.cu
    src/cuda/moe_kernels.cu
    src/cuda/moe_training_kernels.cu
    src/cuda/moe_training_wmma.cu
    src/cuda/bitnet_kernels.cu
    src/cuda/kan_kernels.cu
    src/cuda/sparse_attention_kernels.cu
    src/cuda/attention_train_kernels.cu
    src/cuda/attention_rdna_training.cu
    src/cuda/tiled_cross_entropy.cu
    src/cuda/fused_optimizer_kernels.cu
    src/cuda/sparse_optimizer_activity.cu
)

# Prefer an explicitly discoverable HIP SDK in AUTO mode. This makes an AMD
# workstation deterministic even when a dormant CUDA toolkit is also present.
set(NSOS_HIP_HINTED OFF)
if (NSOS_HIP_ROOT OR DEFINED ENV{HIP_PATH} OR DEFINED ENV{ROCM_PATH})
    set(NSOS_HIP_HINTED ON)
endif()

if (NSOS_GPU_BACKEND_REQUESTED STREQUAL "HIP" OR
    (NSOS_GPU_BACKEND_REQUESTED STREQUAL "AUTO" AND
     NSOS_ENABLE_HIP AND NSOS_HIP_HINTED))
    if (NOT NSOS_ENABLE_HIP)
        message(FATAL_ERROR
            "NSOS_GPU_BACKEND=HIP conflicts with NSOS_ENABLE_HIP=OFF")
    endif()

    if (NSOS_HIP_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${NSOS_HIP_ROOT}")
        set(_NSOS_HIP_RESOLVED_ROOT "${NSOS_HIP_ROOT}")
    elseif (DEFINED ENV{HIP_PATH})
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}")
        set(_NSOS_HIP_RESOLVED_ROOT "$ENV{HIP_PATH}")
    elseif (DEFINED ENV{ROCM_PATH})
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
        set(_NSOS_HIP_RESOLVED_ROOT "$ENV{ROCM_PATH}")
    endif()

    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "The HIP backend requires AMD clang++. Reconfigure from a clean "
            "build directory with "
            "-DCMAKE_CXX_COMPILER=<HIP_PATH>/lib/llvm/bin/clang++.exe on Windows or "
            "/opt/rocm/bin/amdclang++ on Linux.")
    endif()

    # ROCm's supported CXX integration is used deliberately. It works on
    # Windows installations where CMake's native HIP language is not exposed.
    foreach(_hip_arch IN LISTS NSOS_HIP_ARCHITECTURES)
        if (NOT _hip_arch MATCHES
            "^gfx[0-9A-Fa-f]+(:[A-Za-z0-9_+\\-]+)*$")
            message(FATAL_ERROR
                "Invalid NSOS_HIP_ARCHITECTURES entry '${_hip_arch}'. "
                "Use an explicit gfx target such as gfx1102.")
        endif()
    endforeach()
    set(GPU_TARGETS "${NSOS_HIP_ARCHITECTURES}" CACHE STRING
        "ROCm GPU targets" FORCE)
    find_package(hip CONFIG REQUIRED)
    find_package(hipblas CONFIG REQUIRED)

    # A Windows process resolves DLLs before main() runs.  Merely linking the
    # import libraries is therefore insufficient when a developer launches a
    # freshly built executable directly (outside CTest or a HIP-enabled shell).
    # Stage the minimal transitive hipBLAS runtime closure next to every binary
    # so tests, benchmarks, the CLI and Python module do not depend on PATH.
    if (WIN32)
        set(_NSOS_HIP_RUNTIME_BIN "${_NSOS_HIP_RESOLVED_ROOT}/bin")
        set(_NSOS_HIP_RUNTIME_REQUIRED_NAMES
            amd_comgr.dll
            hipblas.dll
            rocsolver.dll
            rocblas.dll
            libhipblaslt.dll
            rocm_kpack.dll)
        set(_NSOS_HIP_RUNTIME_DLLS)
        foreach(_runtime_name IN LISTS _NSOS_HIP_RUNTIME_REQUIRED_NAMES)
            set(_runtime_path "${_NSOS_HIP_RUNTIME_BIN}/${_runtime_name}")
            if (NOT EXISTS "${_runtime_path}")
                message(FATAL_ERROR
                    "The HIP runtime is incomplete: ${_runtime_path} is missing")
            endif()
            list(APPEND _NSOS_HIP_RUNTIME_DLLS "${_runtime_path}")
        endforeach()
        file(GLOB _NSOS_AMDHIP_RUNTIME_DLLS
             LIST_DIRECTORIES FALSE
             "${_NSOS_HIP_RUNTIME_BIN}/amdhip64*.dll")
        if (NOT _NSOS_AMDHIP_RUNTIME_DLLS)
            message(FATAL_ERROR
                "The HIP runtime is incomplete: no amdhip64*.dll was found in "
                "${_NSOS_HIP_RUNTIME_BIN}")
        endif()
        list(APPEND _NSOS_HIP_RUNTIME_DLLS ${_NSOS_AMDHIP_RUNTIME_DLLS})
        list(REMOVE_DUPLICATES _NSOS_HIP_RUNTIME_DLLS)

        # rocBLAS loads its precompiled Tensile kernels dynamically from a
        # directory relative to rocblas.dll.  Copy only the configured GPU
        # architectures; omitting these files makes GEMM abort at runtime even
        # though every DLL was found successfully.
        set(_NSOS_ROCBLAS_LIBRARY_SOURCE
            "${_NSOS_HIP_RUNTIME_BIN}/rocblas/library")
        set(_NSOS_ROCBLAS_RUNTIME_FILES)
        foreach(_hip_arch IN LISTS NSOS_HIP_ARCHITECTURES)
            string(REGEX REPLACE ":.*$" "" _hip_arch_base "${_hip_arch}")
            file(GLOB _hip_arch_rocblas_files
                 LIST_DIRECTORIES FALSE
                 "${_NSOS_ROCBLAS_LIBRARY_SOURCE}/*${_hip_arch_base}*")
            if (NOT _hip_arch_rocblas_files)
                message(FATAL_ERROR
                    "No rocBLAS Tensile kernels for ${_hip_arch_base} were "
                    "found in ${_NSOS_ROCBLAS_LIBRARY_SOURCE}")
            endif()
            list(APPEND _NSOS_ROCBLAS_RUNTIME_FILES
                 ${_hip_arch_rocblas_files})
        endforeach()
        list(REMOVE_DUPLICATES _NSOS_ROCBLAS_RUNTIME_FILES)

        set(_NSOS_HIP_RUNTIME_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
        if (CMAKE_CONFIGURATION_TYPES)
            set(_NSOS_HIP_RUNTIME_DESTINATION
                "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        endif()
        add_custom_target(nsos_hip_runtime ALL
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${_NSOS_HIP_RUNTIME_DESTINATION}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    ${_NSOS_HIP_RUNTIME_DLLS}
                    "${_NSOS_HIP_RUNTIME_DESTINATION}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${_NSOS_HIP_RUNTIME_DESTINATION}/rocblas/library"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    ${_NSOS_ROCBLAS_RUNTIME_FILES}
                    "${_NSOS_HIP_RUNTIME_DESTINATION}/rocblas/library"
            COMMAND_EXPAND_LISTS
            COMMENT "Staging the AMD HIP runtime beside NSOS binaries"
            VERBATIM)
        add_dependencies(nsos_core nsos_hip_runtime)
        message(STATUS
            "HIP runtime DLLs will be staged in "
            "${_NSOS_HIP_RUNTIME_DESTINATION}")
    endif()

    set_source_files_properties(${NSOS_GPU_SOURCES} PROPERTIES LANGUAGE CXX)
    if (NOT NSOS_HIP_DEVICE_LIB_PATH)
        foreach(_device_lib_candidate
                "${_NSOS_HIP_RESOLVED_ROOT}/lib/llvm/amdgcn/bitcode"
                "${_NSOS_HIP_RESOLVED_ROOT}/amdgcn/bitcode"
                "${_NSOS_HIP_RESOLVED_ROOT}/lib/bitcode")
            if (EXISTS "${_device_lib_candidate}/ocml.bc")
                set(NSOS_HIP_DEVICE_LIB_PATH "${_device_lib_candidate}")
                break()
            endif()
        endforeach()
    endif()
    if (NOT EXISTS "${NSOS_HIP_DEVICE_LIB_PATH}/ocml.bc")
        message(FATAL_ERROR
            "Could not find ROCm device libraries (ocml.bc). Set "
            "NSOS_HIP_DEVICE_LIB_PATH to the SDK bitcode directory.")
    endif()

    # hip::device intentionally adds `-x hip` to every CXX source that links
    # it. Keep that usage isolated to the kernel object library so regular
    # NSOS host sources and downstream tests remain ordinary C++.
    add_library(nsos_hip_kernels OBJECT ${NSOS_GPU_SOURCES})
    # This provider depends on finite checks and ordinary expf. Keep its
    # arithmetic contract even when a caller requests fast math globally.
    set_property(SOURCE src/cuda/attention_rdna_training.cu APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math)
    set_property(SOURCE src/cuda/mamba3_siso_kernels.cu APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math)
    set_property(SOURCE src/cuda/mamba3_preprocess_kernels.cu APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math)
    set_property(SOURCE src/cuda/mamba3_layer_kernels.cu APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math)
    set_property(SOURCE src/cuda/tiled_cross_entropy.cu APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math)
    set_property(SOURCE src/cuda/sparse_optimizer_activity.cu APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math)
    target_include_directories(nsos_hip_kernels PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/nsos")
    target_compile_definitions(nsos_hip_kernels PRIVATE
        USE_CUDA=1
        NSOS_USE_GPU=1
        NSOS_GPU_BACKEND_HIP=1)
    target_compile_options(nsos_hip_kernels PRIVATE
        "--rocm-device-lib-path=${NSOS_HIP_DEVICE_LIB_PATH}")
    target_link_libraries(nsos_hip_kernels PRIVATE hip::device)
    # Optional RDNA3 WMMA provider. Requests fail closed when SDK/architecture
    # support is absent; installing headers never changes the default policy.
    find_path(NSOS_ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp
        HINTS "${_NSOS_HIP_RESOLVED_ROOT}/include")
    set(NSOS_ROCWMMA_ENABLED OFF)
    foreach(_hip_arch IN LISTS NSOS_HIP_ARCHITECTURES)
        if (NSOS_ROCWMMA_INCLUDE_DIR AND _hip_arch MATCHES "^gfx110[012]$")
            set(NSOS_ROCWMMA_ENABLED ON)
        endif()
    endforeach()
    # rocWMMA headers assert on unsupported device compiler targets. Keep the
    # optional provider disabled for heterogeneous fat binaries until those
    # targets have a separately compiled provider, without breaking HIP core.
    foreach(_hip_arch IN LISTS NSOS_HIP_ARCHITECTURES)
        if (NOT _hip_arch MATCHES "^gfx110[012]$")
            set(NSOS_ROCWMMA_ENABLED OFF)
        endif()
    endforeach()
    if (NSOS_ROCWMMA_ENABLED)
        target_include_directories(nsos_hip_kernels PRIVATE "${NSOS_ROCWMMA_INCLUDE_DIR}")
        target_compile_definitions(nsos_hip_kernels PRIVATE NSOS_HAS_ROCWMMA=1)
        target_compile_definitions(nsos_core PUBLIC NSOS_HAS_ROCWMMA=1)
    endif()
    target_sources(nsos_core PRIVATE $<TARGET_OBJECTS:nsos_hip_kernels>)
    string(REPLACE ";" "," _NSOS_HIP_ARCHITECTURES_CSV
           "${NSOS_HIP_ARCHITECTURES}")
    target_compile_definitions(nsos_core PUBLIC
        USE_CUDA=1
        NSOS_USE_GPU=1
        NSOS_GPU_BACKEND_HIP=1
        NSOS_HIP_ARCHITECTURES_CSV="${_NSOS_HIP_ARCHITECTURES_CSV}")
    target_link_libraries(nsos_core PUBLIC hip::host roc::hipblas)
    target_link_options(nsos_core INTERFACE
        --hip-link
        "--rocm-device-lib-path=${NSOS_HIP_DEVICE_LIB_PATH}")
    foreach(_hip_arch IN LISTS NSOS_HIP_ARCHITECTURES)
        target_link_options(nsos_core INTERFACE
            "--offload-arch=${_hip_arch}")
    endforeach()
    set(NSOS_GPU_BACKEND_RESOLVED "HIP")
    set(NSOS_GPU_ENABLED ON)
    message(STATUS
        "Using AMD HIP backend; architectures: ${NSOS_HIP_ARCHITECTURES}")
elseif (NSOS_GPU_BACKEND_REQUESTED STREQUAL "CUDA" OR
        (NSOS_GPU_BACKEND_REQUESTED STREQUAL "AUTO" AND NSOS_ENABLE_CUDA))
    include(CheckLanguage)
    if (NSOS_CUDA_TOOLKIT_ROOT)
        set(CUDAToolkit_ROOT "${NSOS_CUDA_TOOLKIT_ROOT}" CACHE PATH "Selected CUDA toolkit root" FORCE)
        if (EXISTS "${NSOS_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe")
            set(CMAKE_CUDA_COMPILER "${NSOS_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "Selected CUDA compiler" FORCE)
        endif()
    endif()
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if (NSOS_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "${NSOS_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
        endif()
        foreach(_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if (NOT _cuda_arch MATCHES
                "^[0-9]+(-(real|virtual))?$")
                message(FATAL_ERROR
                    "Invalid NSOS_CUDA_ARCHITECTURES entry '${_cuda_arch}'. "
                    "Use an explicit numeric target such as 75, 75-real, "
                    "or 75-virtual.")
            endif()
        endforeach()
        set(CMAKE_CUDA_STANDARD 17)
        if (NSOS_CUDA_ALLOW_UNSUPPORTED_COMPILER)
            string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set_property(TARGET nsos_core PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        target_compile_definitions(nsos_core PUBLIC
            USE_CUDA=1
            NSOS_USE_GPU=1
            NSOS_GPU_BACKEND_CUDA=1)
        set(NSOS_CUDA_MIN_ARCH 0)
        foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
            string(REGEX MATCH "^[0-9]+" _arch_numeric "${_arch}")
            if (NSOS_CUDA_MIN_ARCH EQUAL 0 OR
                _arch_numeric LESS NSOS_CUDA_MIN_ARCH)
                set(NSOS_CUDA_MIN_ARCH ${_arch_numeric})
            endif()
        endforeach()
        if (NOT NSOS_CUDA_MIN_ARCH EQUAL 0)
            target_compile_definitions(nsos_core PUBLIC NSOS_CUDA_MIN_ARCH=${NSOS_CUDA_MIN_ARCH})
        endif()
        string(REPLACE ";" "," _NSOS_CUDA_ARCHITECTURES_CSV
               "${CMAKE_CUDA_ARCHITECTURES}")
        target_compile_definitions(
            nsos_core PUBLIC
            NSOS_CUDA_ARCHITECTURES_CSV="${_NSOS_CUDA_ARCHITECTURES_CSV}")
        target_sources(nsos_core PRIVATE ${NSOS_GPU_SOURCES})
        # PUBLIC because tensor.h includes <cuda_runtime.h> when USE_CUDA
        # is defined; tests/CLI/API targets that link nsos_core need to
        # inherit the CUDA include directory to compile that header.
        target_link_libraries(nsos_core PUBLIC CUDA::cudart CUDA::cublas)
        if (NSOS_CUDA_PTDS)
            # 1) nvcc: __global__ launches without an explicit stream go to
            #    cudaStreamPerThread instead of the legacy stream.
            # 2) Host TUs: CUDA_API_PER_THREAD_DEFAULT_STREAM makes runtime
            #    API calls (memcpy/memset/sync) use the per-thread stream too.
            #    PUBLIC so every consumer TU sees the SAME stream view (mixed
            #    views across TUs is a documented CUDA footgun).
            # 3) NSOS_CUDA_PTDS gates the code that needs the capturable
            #    stream (cublasSetStream, decode-graph capture).
            target_compile_options(nsos_core PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
            target_compile_definitions(nsos_core PUBLIC
                CUDA_API_PER_THREAD_DEFAULT_STREAM=1
                NSOS_CUDA_PTDS=1)
            message(STATUS "NSOS_CUDA_PTDS=ON: per-thread default stream (CUDA-graph decode enabled)")
        endif()
        message(STATUS "Using CUDA toolkit: ${CUDAToolkit_BIN_DIR}")
        message(STATUS "Using CUDA toolkit version: ${CUDAToolkit_VERSION}")
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        set(NSOS_GPU_BACKEND_RESOLVED "CUDA")
        set(NSOS_GPU_ENABLED ON)
    elseif (NSOS_GPU_BACKEND_REQUESTED STREQUAL "CUDA")
        message(FATAL_ERROR
            "NSOS_GPU_BACKEND=CUDA was requested but no CUDA compiler was found")
    endif()
endif()

message(STATUS "NSOS GPU backend: ${NSOS_GPU_BACKEND_RESOLVED}")

if (NSOS_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 REQUIRED)

    pybind11_add_module(nsos_ext src/bindings.cpp)
    target_link_libraries(nsos_ext PRIVATE nsos_core)
endif()

if (NSOS_BUILD_CIRCUIT_TOOLS)
    add_executable(nsos_circuit_trainer src/nsos_circuit_trainer.cpp)
    target_link_libraries(nsos_circuit_trainer PRIVATE nsos_core)
endif()

if (NSOS_BUILD_CLI)
    add_executable(nsos_cli src/nsos_cli.cpp)
    target_link_libraries(nsos_cli PRIVATE nsos_core)
endif()

if (NSOS_BUILD_API)
    add_executable(nsos_api_server src/api_server.cpp)
    target_link_libraries(nsos_api_server PRIVATE nsos_core)
    if (WIN32)
        target_link_libraries(nsos_api_server PRIVATE ws2_32)
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────
# NSOS Profiler — opt-in module for architectural analysis sessions.
# Builds a static library `nsos_profiler` with the cycle counters,
# event ring buffer, cache probe, and the hook callbacks that the
# main `nsos_core` calls via function pointers (default null, set at
# runtime by nsos_profiler_install_hooks()).  Production training
# builds leave NSOS_BUILD_PROFILER=OFF and never link this code.
# ──────────────────────────────────────────────────────────────────────
if (NSOS_BUILD_PROFILER)
    add_library(nsos_profiler STATIC
        src/profiler/inference_profiler.cpp
        src/profiler/cache_probe.cpp
        src/profiler/jamba_profiler_hooks.cpp
    )
    target_include_directories(nsos_profiler PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    # The profiler exposes a thin pybind module so the Python driver
    # can attach/detach/drain without re-importing nsos_ext.  We build
    # it as a separate Python module so production users never see it.
    if (NSOS_BUILD_PYTHON)
        pybind11_add_module(nsos_profiler_ext src/profiler/profiler_bindings.cpp)
        target_link_libraries(nsos_profiler_ext PRIVATE
            nsos_profiler
            nsos_core
        )
    endif()
    message(STATUS "NSOS Profiler enabled — static lib + nsos_profiler_ext built")
endif()

if (NSOS_BUILD_OXTAMEM)
    find_program(CARGO_EXECUTABLE cargo)
    if (CARGO_EXECUTABLE AND EXISTS "${NSOS_OXTAMEM_DIR}/Cargo.toml")
        if (WIN32)
            set(NSOS_OXTAMEM_LIBRARY_FILE "${NSOS_OXTAMEM_DIR}/target/release/oxta_mem.dll")
        elseif(APPLE)
            set(NSOS_OXTAMEM_LIBRARY_FILE "${NSOS_OXTAMEM_DIR}/target/release/liboxta_mem.dylib")
        else()
            set(NSOS_OXTAMEM_LIBRARY_FILE "${NSOS_OXTAMEM_DIR}/target/release/liboxta_mem.so")
        endif()
        add_custom_target(oxtamem_engine
            COMMAND ${CARGO_EXECUTABLE} build --release --locked --no-default-features
            WORKING_DIRECTORY ${NSOS_OXTAMEM_DIR}
            COMMENT "Building copied OxtaMem C-ABI engine (Python-free runtime)"
        )
        target_compile_definitions(nsos_core PUBLIC NSOS_OXTAMEM_DEFAULT_LIBRARY="${NSOS_OXTAMEM_LIBRARY_FILE}")
        if (TARGET nsos_cli)
            add_dependencies(nsos_cli oxtamem_engine)
        endif()
        if (TARGET nsos_api_server)
            add_dependencies(nsos_api_server oxtamem_engine)
        endif()
    else()
        message(STATUS "Cargo or copied OxtaMem source not found; skipping OxtaMem build target.")
    endif()
endif()

if (NSOS_BUILD_TESTS)
    enable_testing()
    find_package(Threads REQUIRED)
    # CMake's Release flags define NDEBUG, which would otherwise compile the
    # historical assert-based checks out of many test binaries and let them
    # report success without evaluating their invariants. Keep assertions
    # active in every test/diagnostic target created below, independently of
    # the selected build configuration.
    if (MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/UNDEBUG>)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
    endif()
    # Every test_*.cpp must either become a CMake target or be explicitly
    # quarantined here. The configure-time audit below fails on a new silent
    # orphan. These four are retained only as historical source:
    # - test_suite: duplicated by dedicated dataloader/memory/component gates;
    # - test_v2_features: stale dynamic_chrass include and duplicated MCTS/API;
    # - test_vulnerability_fixes: superseded by focused fail-closed gates;
    # - unit/test_tensor: incomplete GoogleTest prototype with unavailable
    #   nsos/test_context.h and a commented-out NaN assertion.
    set(NSOS_QUARANTINED_TEST_SOURCES
        tests/test_suite.cpp
        tests/test_v2_features.cpp
        tests/test_vulnerability_fixes.cpp
        tests/unit/test_tensor.cpp)
    list(SORT NSOS_QUARANTINED_TEST_SOURCES)

    # Sources under tests/gpu plus the two legacy root-level GPU gates are
    # conditionally registered: a CPU-only configuration must account for
    # them without pretending they are runnable, while a GPU configuration
    # must still prove that every source is attached to a real target.  The
    # configure-time audit below enforces the latter and catches any newly
    # added GPU source that lacks an add_executable registration.
    file(GLOB NSOS_GPU_DIRECTORY_TEST_SOURCES
         CONFIGURE_DEPENDS
         RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu/test_*.cpp")
    set(NSOS_GPU_ONLY_TEST_SOURCES
        tests/test_gpu_parity.cpp
        tests/test_mamba_parallel_scan_parity.cpp
        ${NSOS_GPU_DIRECTORY_TEST_SOURCES})
    list(REMOVE_DUPLICATES NSOS_GPU_ONLY_TEST_SOURCES)
    list(SORT NSOS_GPU_ONLY_TEST_SOURCES)
    set(NSOS_OXTAMEM_ONLY_TEST_SOURCES
        tests/test_oxtamem_ffi.cpp)

    add_executable(test_sanity tests/test_sanity.cpp)
    target_link_libraries(test_sanity PRIVATE nsos_core)
    add_test(NAME test_sanity COMMAND test_sanity)

    # Supporting runtime components not exercised through the main trainer:
    # Muon, MCTS, SprecherKAN, single-process Fabric, arena rewind and the
    # packed host GEMM primitive.
    add_executable(test_components tests/test_components.cpp)
    target_link_libraries(test_components PRIVATE nsos_core)
    add_test(NAME test_components COMMAND test_components)

    add_executable(test_core_v2 tests/test_core_v2.cpp)
    target_link_libraries(test_core_v2 PRIVATE nsos_core)
    add_test(NAME test_core_v2 COMMAND test_core_v2)

    # Allocation churn, deep-copy slicing and repeated model backward health.
    add_executable(test_stability tests/test_stability.cpp)
    target_link_libraries(test_stability PRIVATE nsos_core)
    add_test(NAME test_stability COMMAND test_stability)
    set_tests_properties(test_stability PROPERTIES TIMEOUT 300 LABELS "stress")

    add_executable(test_bitlinear tests/test_bitlinear.cpp)
    target_link_libraries(test_bitlinear PRIVATE nsos_core)
    add_test(NAME test_bitlinear COMMAND test_bitlinear)

    # SSA - Subquadratic Sparse Attention (subq.ai research validation).
    # Parity with dense attention, content-selection retrieval vs fixed window,
    # and subquadratic attended-pair / FLOP accounting.
    add_executable(test_sparse_attention tests/test_sparse_attention.cpp)
    target_link_libraries(test_sparse_attention PRIVATE nsos_core)
    add_test(NAME test_sparse_attention COMMAND test_sparse_attention)

    # Validates the BLIS-style MathOps::gemm against the naive matmul before
    # wiring it into the production path (#7).
    add_executable(test_gemm_parity tests/test_gemm_parity.cpp)
    target_link_libraries(test_gemm_parity PRIVATE nsos_core)
    add_test(NAME test_gemm_parity COMMAND test_gemm_parity)

    # Basic Tensor::matmul contract. This legacy test previously returned zero
    # even when its assertions printed FAIL; it is now a real fail-closed gate.
    add_executable(test_matmul tests/test_matmul.cpp)
    target_link_libraries(test_matmul PRIVATE nsos_core)
    add_test(NAME test_matmul COMMAND test_matmul)

    # Router indices, normalized top-k weights, product-block integration and a
    # deterministic non-collapse fixture.
    add_executable(test_moe_router tests/test_moe_router.cpp)
    target_link_libraries(test_moe_router PRIVATE nsos_core)
    add_test(NAME test_moe_router COMMAND test_moe_router)

    # MoE gradient flow (rewritten against the current expert_gate_up/expert_down
    # API; the old version referenced a stale KAN-expert members member).
    add_executable(test_moe_training tests/test_moe_training.cpp)
    target_link_libraries(test_moe_training PRIVATE nsos_core)
    add_test(NAME test_moe_training COMMAND test_moe_training)

    # Subsystem wiring: KAN FFN (use_kan) forward/backward/params + use_kan=false
    # regression, and Self-Healer reachable from the live model.
    add_executable(test_systems_wiring tests/test_systems_wiring.cpp)
    target_link_libraries(test_systems_wiring PRIVATE nsos_core)
    add_test(NAME test_systems_wiring COMMAND test_systems_wiring)

    # Cherry-pick #2 (Slender-Mamba head-to-toe quantization).
    # Tests the slender forward path on Embedding: cache lifecycle,
    # round-trip bounds, STE backward routing, and edge cases.
    # See docs/SLENDER_INTEGRATION.md.
    add_executable(test_slender_embedding tests/test_slender_embedding.cpp)
    target_link_libraries(test_slender_embedding PRIVATE nsos_core)
    add_test(NAME test_slender_embedding COMMAND test_slender_embedding)

    # Cherry-pick #1 (bitmamba UNPACK_LUT + AVX2 SIMD).
    # Tests gemm_158bit_lut SIMD path: small deterministic, env routing,
    # scalar tail handling, high zero-density sparse weights.
    # See docs/BITMAMBA_LUT_INTEGRATION.md.
    add_executable(test_bitmamba_lut tests/test_bitmamba_lut.cpp)
    target_link_libraries(test_bitmamba_lut PRIVATE nsos_core)
    add_test(NAME test_bitmamba_lut COMMAND test_bitmamba_lut)

    # CHRASS validation battery (2026-05-25).
    # Tests ChrassLayer v2 exhaustively: forward correctness vs dense
    # reference, backward shapes + identity check + sparse support,
    # gradcheck (numerical vs analytical), edge cases (empty/complete/
    # self-loops/negatives/1x1), saturation clamps, NaN/Inf robustness,
    # determinism, AdamW step, topology preservation, 3-layer stack
    # integration, 256x256 perf budget.
    add_executable(test_chrass_validation tests/test_chrass_validation.cpp)
    target_link_libraries(test_chrass_validation PRIVATE nsos_core)
    add_test(NAME test_chrass_validation COMMAND test_chrass_validation)

    # TTT validation battery (2026-05-25).
    # Deepens test_ttt_layer_kernel.cpp.  Validates: training_mode toggle,
    # reset/snapshot/restore, Hamiltonian on/off, friction/temperature
    # hyperparams, long-sequence stability, BitLinear integration,
    # determinism across instances.
    add_executable(test_ttt_validation tests/test_ttt_validation.cpp)
    target_link_libraries(test_ttt_validation PRIVATE nsos_core)
    add_test(NAME test_ttt_validation COMMAND test_ttt_validation)

    # JambaModel + CHRASS end-to-end smoke (2026-05-25).
    # Confirms use_chrass=true wires CHRASS into every block, forward and
    # backward complete without NaN/Inf, parameters() exposes chrass weights,
    # and CHRASS params receive gradient during training step.
    add_executable(test_jamba_with_chrass tests/test_jamba_with_chrass.cpp)
    target_link_libraries(test_jamba_with_chrass PRIVATE nsos_core)
    add_test(NAME test_jamba_with_chrass COMMAND test_jamba_with_chrass)

    # CHRASS overhead benchmark — mede custo de params + latência ON vs OFF.
    add_executable(bench_chrass_overhead tests/bench_chrass_overhead.cpp)
    target_link_libraries(bench_chrass_overhead PRIVATE nsos_core)

    # Pantheon VIB integration smoke (2026-05-25).
    # Verifies pantheon_vib_beta is wired in Trainer and produces a
    # measurable, gradient-propagating effect without breaking training.
    add_executable(test_pantheon_vib_integration tests/test_pantheon_vib_integration.cpp)
    target_link_libraries(test_pantheon_vib_integration PRIVATE nsos_core)
    add_test(NAME test_pantheon_vib_integration COMMAND test_pantheon_vib_integration)

    add_executable(test_lut_tmac tests/test_lut_tmac.cpp)
    target_link_libraries(test_lut_tmac PRIVATE nsos_core)
    add_test(NAME test_lut_tmac COMMAND test_lut_tmac)

    # Microbench — not wired into ctest because it's a perf measurement,
    # not a correctness gate.  Built so `cmake --build` produces the
    # executable; run manually to compare LUT vs scalar reference.
    add_executable(bench_lut_tmac tests/bench_lut_tmac.cpp)
    target_link_libraries(bench_lut_tmac PRIVATE nsos_core)

    # Content-addressed ternary storage — round-trip + multi-matrix dedup
    add_executable(test_lut_cas tests/test_lut_cas.cpp)
    target_link_libraries(test_lut_cas PRIVATE nsos_core)
    add_test(NAME test_lut_cas COMMAND test_lut_cas)

    add_executable(test_turboquant tests/test_turboquant.cpp)
    target_link_libraries(test_turboquant PRIVATE nsos_core)
    add_test(NAME test_turboquant COMMAND test_turboquant)

    add_executable(test_bitnet_integrated tests/test_bitnet_integrated.cpp)
    target_link_libraries(test_bitnet_integrated PRIVATE nsos_core)
    add_test(NAME test_bitnet_integrated COMMAND test_bitnet_integrated)

    add_executable(test_mamba2 tests/test_mamba2.cpp)
    target_link_libraries(test_mamba2 PRIVATE nsos_core)
    add_test(NAME test_mamba2 COMMAND test_mamba2)

    add_executable(test_mamba2_reference tests/test_mamba2_reference.cpp)
    target_link_libraries(test_mamba2_reference PRIVATE nsos_core)
    add_test(NAME test_mamba2_reference COMMAND test_mamba2_reference)

    add_executable(test_mamba3_reference tests/test_mamba3_reference.cpp)
    target_link_libraries(test_mamba3_reference PRIVATE nsos_core)
    add_test(NAME test_mamba3_reference COMMAND test_mamba3_reference)
    add_executable(test_mamba3_integral_reference tests/test_mamba3_integral_reference.cpp)
    target_link_libraries(test_mamba3_integral_reference PRIVATE nsos_core)
    add_test(NAME test_mamba3_integral_reference COMMAND test_mamba3_integral_reference)
    add_executable(test_mamba3_layer tests/test_mamba3_layer.cpp)
    target_link_libraries(test_mamba3_layer PRIVATE nsos_core)
    add_test(NAME test_mamba3_layer COMMAND test_mamba3_layer)
    if (NSOS_GPU_ENABLED)
        add_test(NAME test_gpu_mamba3_layer COMMAND test_mamba3_layer --gpu)
        set_tests_properties(test_gpu_mamba3_layer PROPERTIES RUN_SERIAL TRUE)
    endif()

    add_executable(test_tiled_head_loss tests/test_tiled_head_loss.cpp)
    target_link_libraries(test_tiled_head_loss PRIVATE nsos_core)
    add_test(NAME test_tiled_head_loss COMMAND test_tiled_head_loss)
    if (NSOS_GPU_ENABLED)
        add_test(NAME test_gpu_tiled_head_loss COMMAND test_tiled_head_loss --gpu)
    endif()

    add_executable(test_kan tests/test_kan.cpp)
    target_link_libraries(test_kan PRIVATE nsos_core)
    add_test(NAME test_kan COMMAND test_kan)

    # Finite-difference gradient gate: proves every hand-derived backward (the
    # shared tensor bricks + Mamba2 SSD + KAN) matches the numerical gradient.
    # No generic autograd exists, so this is the safety net against silent
    # chain-rule bugs.  Product release gate.
    add_executable(test_gradcheck tests/test_gradcheck.cpp)
    target_link_libraries(test_gradcheck PRIVATE nsos_core)
    add_test(NAME test_gradcheck COMMAND test_gradcheck)

    add_executable(test_training_invariants tests/test_training_invariants.cpp)
    target_link_libraries(test_training_invariants PRIVATE nsos_core)
    add_test(NAME test_training_invariants COMMAND test_training_invariants)

    add_executable(test_training_snapshot_frozen tests/test_training_snapshot_frozen.cpp)
    target_link_libraries(test_training_snapshot_frozen PRIVATE nsos_core)
    add_test(NAME test_training_snapshot_frozen COMMAND test_training_snapshot_frozen)

    add_executable(test_runtime_execution_identity tests/test_runtime_execution_identity.cpp)
    target_link_libraries(test_runtime_execution_identity PRIVATE nsos_core)
    add_test(NAME test_runtime_execution_identity COMMAND test_runtime_execution_identity)

    # 4-bit optimizer states (Li et al. 2023): roundtrip error, zero-point
    # robustness of v, end-to-end Adam parity vs FP32, ~8x memory saving.
    add_executable(test_optimizer_4bit tests/test_optimizer_4bit.cpp)
    target_link_libraries(test_optimizer_4bit PRIVATE nsos_core)
    add_test(NAME test_optimizer_4bit COMMAND test_optimizer_4bit)

    add_executable(test_ttt_layer_kernel tests/test_ttt_layer_kernel.cpp)
    target_link_libraries(test_ttt_layer_kernel PRIVATE nsos_core)
    add_test(NAME test_ttt_layer_kernel COMMAND test_ttt_layer_kernel)
    add_executable(test_ttt_full_bptt tests/test_ttt_full_bptt.cpp)
    target_link_libraries(test_ttt_full_bptt PRIVATE nsos_core)
    add_test(NAME test_ttt_full_bptt COMMAND test_ttt_full_bptt)

    add_executable(test_inference_engine tests/test_inference_engine.cpp)
    target_link_libraries(test_inference_engine PRIVATE nsos_core)
    add_test(NAME test_inference_engine COMMAND test_inference_engine)

    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE nsos_core)
    add_test(NAME test_tokenizer COMMAND test_tokenizer)

    add_executable(test_dataloader tests/test_dataloader.cpp)
    target_link_libraries(test_dataloader PRIVATE nsos_core)
    add_test(NAME test_dataloader COMMAND test_dataloader)

    if (NOT Python3_Interpreter_FOUND)
        find_package(Python3 COMPONENTS Interpreter QUIET)
    endif()
    if (Python3_Interpreter_FOUND)
        add_test(
            NAME test_core_runtime_contracts
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core_runtime_contracts.py
        )
        add_test(NAME test_p0_evaluation_contracts
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_p0_evaluation_contracts.py)
        add_test(
            NAME test_gpu_backend_configuration
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_backend_configuration.py
        )
        add_test(
            NAME test_product_benchmark_policy
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_product_benchmark_policy.py
        )
        add_test(
            NAME test_speculative_reference
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_speculative_reference.py
        )
        add_test(
            NAME test_decode_measurement_contract
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_decode_measurement_contract.py
        )
        add_test(
            NAME test_curriculum_integrity
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_curriculum_integrity.py
        )
        add_test(
            NAME test_ptbr_conversational_script
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ptbr_conversational_script.py
        )
        add_test(
            NAME test_ptbr_edu_synth
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ptbr_edu_synth.py
        )
        add_test(
            NAME test_ptbr_verified_pilot
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ptbr_verified_pilot.py
        )
        add_test(
            NAME test_training_engineering_probe
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_training_engineering_probe.py
        )
    endif()

    if (NSOS_BUILD_PYTHON)
        add_test(
            NAME test_python_binding_smoke
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_binding_smoke.py
        )
        set_tests_properties(
            test_python_binding_smoke
            PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:nsos_ext>")
        if (NSOS_MAMBA3_UPSTREAM_ARTIFACT_DIR)
            if (NOT EXISTS "${NSOS_MAMBA3_UPSTREAM_ARTIFACT_DIR}/upstream-mamba3/test_mamba3_mimo.py")
                message(FATAL_ERROR "Pinned upstream Mamba3 reference is missing")
            endif()
            add_test(NAME test_mamba3_pinned_upstream
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mamba3_pinned_upstream.py
                    --module-dir $<TARGET_FILE_DIR:nsos_ext>
                    --artifact-dir ${NSOS_MAMBA3_UPSTREAM_ARTIFACT_DIR})
            if (NSOS_GPU_ENABLED)
                add_test(NAME test_gpu_mamba3_pinned_upstream
                    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mamba3_pinned_upstream.py
                        --module-dir $<TARGET_FILE_DIR:nsos_ext>
                        --artifact-dir ${NSOS_MAMBA3_UPSTREAM_ARTIFACT_DIR} --gpu)
                set_tests_properties(test_gpu_mamba3_pinned_upstream PROPERTIES RUN_SERIAL TRUE)
            endif()
        endif()
    endif()

    add_executable(test_mcts_reasoning_v3 tests/test_mcts_reasoning_v3.cpp)
    target_link_libraries(test_mcts_reasoning_v3 PRIVATE nsos_core)
    add_test(NAME test_mcts_reasoning_v3 COMMAND test_mcts_reasoning_v3)
    add_executable(test_verified_reasoning tests/test_verified_reasoning.cpp)
    target_link_libraries(test_verified_reasoning PRIVATE nsos_core)
    add_test(NAME test_verified_reasoning COMMAND test_verified_reasoning)
    if (NSOS_GPU_ENABLED)
        add_test(NAME test_verified_reasoning_gpu COMMAND test_verified_reasoning --gpu)
        set_tests_properties(test_verified_reasoning_gpu PROPERTIES ENVIRONMENT "NSOS_GPU_DEBUG_SYNC=1")
    endif()

    add_executable(test_memory_causal_store tests/test_memory_causal_store.cpp)
    target_link_libraries(test_memory_causal_store PRIVATE nsos_core)
    add_test(NAME test_memory_causal_store COMMAND test_memory_causal_store)

    if (TARGET oxtamem_engine)
        add_executable(test_oxtamem_ffi tests/test_oxtamem_ffi.cpp)
        target_link_libraries(test_oxtamem_ffi PRIVATE nsos_core)
        add_dependencies(test_oxtamem_ffi oxtamem_engine)
        add_test(NAME test_oxtamem_ffi COMMAND test_oxtamem_ffi)
    endif()

    add_executable(test_jamba tests/test_jamba.cpp)
    target_link_libraries(test_jamba PRIVATE nsos_core)
    add_test(NAME test_jamba COMMAND test_jamba)

    add_executable(test_model_pack tests/test_model_pack.cpp)
    target_link_libraries(test_model_pack PRIVATE nsos_core)
    add_test(NAME test_model_pack COMMAND test_model_pack)

    add_executable(test_layer_audit tests/test_layer_audit.cpp)
    target_link_libraries(test_layer_audit PRIVATE nsos_core)
    add_test(NAME test_layer_audit COMMAND test_layer_audit)

    # GPU validation is registered for either CUDA/NVIDIA or HIP/AMD builds.
    # Every binary is fail-closed when invoked without a selectable device.
    if (NSOS_GPU_ENABLED)
    add_executable(test_gpu_parity tests/test_gpu_parity.cpp)
    target_link_libraries(test_gpu_parity PRIVATE nsos_core)
    add_test(NAME test_gpu_parity COMMAND test_gpu_parity)

    # Fase 2 gate: opt-in Mamba2 parallel-prefix scan vs sequential (1e-4).
    add_executable(test_mamba_parallel_scan_parity tests/test_mamba_parallel_scan_parity.cpp)
    target_link_libraries(test_mamba_parallel_scan_parity PRIVATE nsos_core)
    add_test(NAME test_mamba_parallel_scan_parity COMMAND test_mamba_parallel_scan_parity)

    # Decomposed GPU parity sub-tests.  Each binary covers one kernel /
    # surface so that a CUDA segfault in one (historically the streaming
    # mamba scan on CUDA 12.9) does not mask passing kernels in the rest
    # of the parity gate.  Header-only helpers in tests/gpu/.
    add_executable(test_gpu_parity_basic tests/gpu/test_gpu_parity_basic.cpp)
    target_link_libraries(test_gpu_parity_basic PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_basic PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_basic COMMAND test_gpu_parity_basic)

    add_executable(test_gpu_parity_bitlinear tests/gpu/test_gpu_parity_bitlinear.cpp)
    target_link_libraries(test_gpu_parity_bitlinear PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_bitlinear PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_bitlinear COMMAND test_gpu_parity_bitlinear)

    add_executable(test_gpu_parity_mamba_stream tests/gpu/test_gpu_parity_mamba_stream.cpp)
    target_link_libraries(test_gpu_parity_mamba_stream PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_stream PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_stream COMMAND test_gpu_parity_mamba_stream)

    add_executable(test_gpu_parity_mamba_scan tests/gpu/test_gpu_parity_mamba_scan.cpp)
    target_link_libraries(test_gpu_parity_mamba_scan PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_scan PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_scan COMMAND test_gpu_parity_mamba_scan)

    # PROPER selective SSM (NSOS_MAMBA_PROPER_SSM): conv1d_causal + linear-readout
    # scan CUDA kernels vs the host reference (forward output + input gradient).
    add_executable(test_gpu_parity_mamba_proper tests/gpu/test_gpu_parity_mamba_proper.cpp)
    target_link_libraries(test_gpu_parity_mamba_proper PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_proper PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_proper COMMAND test_gpu_parity_mamba_proper)

    # Proper-path on-device single-token decode (NSOS_MAMBA_GPU_STEP):
    # mamba_proper_step_kernel vs the GPU full-sequence scan.
    add_executable(test_gpu_parity_mamba_proper_stream tests/gpu/test_gpu_parity_mamba_proper_stream.cpp)
    target_link_libraries(test_gpu_parity_mamba_proper_stream PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_proper_stream PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_proper_stream COMMAND test_gpu_parity_mamba_proper_stream)

    # Fused N-STATE decode step (full Mamba-2) vs the full N-state scan —
    # sibling of the diagonal proper_stream gate above.
    add_executable(test_gpu_parity_mamba_nstate_stream tests/gpu/test_gpu_parity_mamba_nstate_stream.cpp)
    target_link_libraries(test_gpu_parity_mamba_nstate_stream PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_nstate_stream PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_nstate_stream COMMAND test_gpu_parity_mamba_nstate_stream)

    # Real hybrid Mamba + attention incremental decode.  Checks every CUDA
    # token step against CPU streaming and a fresh full-prefix CPU forward.
    add_executable(test_gpu_parity_decode_incremental tests/gpu/test_gpu_parity_decode_incremental.cpp)
    target_link_libraries(test_gpu_parity_decode_incremental PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_decode_incremental PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_decode_incremental COMMAND test_gpu_parity_decode_incremental)

    # Full Mamba-2 SSD with N-state expansion: mamba_nstate_* CUDA kernels
    # (per-head dt/A, N-dim B/C, h in R^{H*P*N}) vs the host reference.
    add_executable(test_gpu_parity_mamba_nstate tests/gpu/test_gpu_parity_mamba_nstate.cpp)
    target_link_libraries(test_gpu_parity_mamba_nstate PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_nstate PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_nstate COMMAND test_gpu_parity_mamba_nstate)

    add_executable(test_gpu_parity_mamba_faithful tests/gpu/test_gpu_parity_mamba_faithful.cpp)
    target_link_libraries(test_gpu_parity_mamba_faithful PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_mamba_faithful PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_mamba_faithful COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_rollback
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_shared_carry_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_decay_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_decay_rollback
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_state_parallel_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_history_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_history_rollback
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_chunked_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_chunked_rollback
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_chunk_lds_state_major_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_chunk_lds_state_major_rollback
        COMMAND test_gpu_parity_mamba_faithful)
    # Compile-time state-width specialization of the forward scan.
    add_test(NAME test_gpu_parity_mamba_faithful_fixed_state_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_fixed_state_rollback
        COMMAND test_gpu_parity_mamba_faithful)
    # Time-parallel forward scan. The opt-in entry forces the minimum chunk
    # width so the short parity fixtures actually exercise the multi-chunk
    # carry path instead of collapsing to a single chunk.
    add_test(NAME test_gpu_parity_mamba_faithful_forward_chunked_optin
        COMMAND test_gpu_parity_mamba_faithful)
    add_test(NAME test_gpu_parity_mamba_faithful_forward_chunked_rollback
        COMMAND test_gpu_parity_mamba_faithful)

    add_executable(test_gpu_parity_jamba tests/gpu/test_gpu_parity_jamba.cpp)
    target_link_libraries(test_gpu_parity_jamba PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_jamba PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_jamba COMMAND test_gpu_parity_jamba)
    add_test(NAME test_gpu_parity_jamba_redesign COMMAND test_gpu_parity_jamba)

    add_executable(test_gpu_training_redesign tests/gpu/test_gpu_training_redesign.cpp)
    target_link_libraries(test_gpu_training_redesign PRIVATE nsos_core)
    target_include_directories(test_gpu_training_redesign PRIVATE tests/gpu)
    add_test(NAME test_gpu_training_redesign COMMAND test_gpu_training_redesign)
    add_executable(test_gpu_device_sparse_optimizer tests/gpu/test_gpu_device_sparse_optimizer.cpp)
    target_link_libraries(test_gpu_device_sparse_optimizer PRIVATE nsos_core)
    target_include_directories(test_gpu_device_sparse_optimizer PRIVATE tests/gpu)
    add_test(NAME test_gpu_device_sparse_optimizer COMMAND test_gpu_device_sparse_optimizer)
    add_executable(test_gpu_sparse_adam_transaction tests/gpu/test_gpu_sparse_adam_transaction.cpp)
    target_link_libraries(test_gpu_sparse_adam_transaction PRIVATE nsos_core)
    target_include_directories(test_gpu_sparse_adam_transaction PRIVATE tests/gpu)
    add_test(NAME test_gpu_sparse_adam_transaction COMMAND test_gpu_sparse_adam_transaction)
    add_executable(test_gpu_sparse_adam_trainer tests/gpu/test_gpu_sparse_adam_trainer.cpp)
    target_link_libraries(test_gpu_sparse_adam_trainer PRIVATE nsos_core)
    target_include_directories(test_gpu_sparse_adam_trainer PRIVATE tests/gpu)
    add_test(NAME test_gpu_sparse_adam_trainer COMMAND test_gpu_sparse_adam_trainer)
    if (NSOS_GPU_BACKEND_RESOLVED STREQUAL "HIP" AND
        ("gfx1100" IN_LIST NSOS_HIP_ARCHITECTURES OR "gfx1101" IN_LIST NSOS_HIP_ARCHITECTURES OR "gfx1102" IN_LIST NSOS_HIP_ARCHITECTURES))
        add_executable(test_gpu_mamba3_siso tests/gpu/test_gpu_mamba3_siso.cpp)
        target_link_libraries(test_gpu_mamba3_siso PRIVATE nsos_core)
        target_include_directories(test_gpu_mamba3_siso PRIVATE tests/gpu)
        add_test(NAME test_gpu_mamba3_siso COMMAND test_gpu_mamba3_siso)
        add_executable(test_gpu_mamba3_projected tests/gpu/test_gpu_mamba3_projected.cpp)
        target_link_libraries(test_gpu_mamba3_projected PRIVATE nsos_core)
        target_include_directories(test_gpu_mamba3_projected PRIVATE tests/gpu)
        add_test(NAME test_gpu_mamba3_projected COMMAND test_gpu_mamba3_projected)
    endif()
    add_executable(test_gpu_moe_grouped_training tests/gpu/test_gpu_moe_grouped_training.cpp)
    target_link_libraries(test_gpu_moe_grouped_training PRIVATE nsos_core)
    target_include_directories(test_gpu_moe_grouped_training PRIVATE tests/gpu)
    add_test(NAME test_gpu_moe_grouped_training COMMAND test_gpu_moe_grouped_training)
    if (NSOS_ROCWMMA_ENABLED)
        add_test(NAME test_gpu_moe_wmma_training COMMAND test_gpu_moe_grouped_training)
        add_executable(test_gpu_attention_training_provider tests/gpu/test_gpu_attention_training_provider.cpp)
        target_link_libraries(test_gpu_attention_training_provider PRIVATE nsos_core)
        target_include_directories(test_gpu_attention_training_provider PRIVATE tests/gpu)
        add_test(NAME test_gpu_attention_training_provider COMMAND test_gpu_attention_training_provider)
        add_executable(test_gpu_attention_training_consumer tests/gpu/test_gpu_attention_training_consumer.cpp)
        target_link_libraries(test_gpu_attention_training_consumer PRIVATE nsos_core)
        target_include_directories(test_gpu_attention_training_consumer PRIVATE tests/gpu)
        add_test(NAME test_gpu_attention_training_consumer COMMAND test_gpu_attention_training_consumer)
        set_tests_properties(test_gpu_attention_training_provider test_gpu_attention_training_consumer PROPERTIES RUN_SERIAL TRUE)
        add_executable(test_gpu_hybrid_integral tests/gpu/test_gpu_hybrid_integral.cpp)
        target_link_libraries(test_gpu_hybrid_integral PRIVATE nsos_core)
        target_include_directories(test_gpu_hybrid_integral PRIVATE tests/gpu)
        add_test(NAME test_gpu_hybrid_integral COMMAND test_gpu_hybrid_integral)
        set_tests_properties(test_gpu_hybrid_integral PROPERTIES RUN_SERIAL TRUE)
        add_executable(test_gpu_attention_rdna_training tests/gpu/test_gpu_attention_rdna_training.cpp)
        target_link_libraries(test_gpu_attention_rdna_training PRIVATE nsos_core)
        target_include_directories(test_gpu_attention_rdna_training PRIVATE tests/gpu)
        add_test(NAME test_gpu_attention_rdna_training COMMAND test_gpu_attention_rdna_training)
    endif()

    # MoE router top-k mask + load accumulation parity (Phase 4 GPU plan).
    add_executable(test_gpu_parity_moe_router tests/gpu/test_gpu_parity_moe_router.cpp)
    target_link_libraries(test_gpu_parity_moe_router PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_moe_router PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_moe_router COMMAND test_gpu_parity_moe_router)

    # MoE batched forward (Phase 4-extended): full JambaModel with MoE
    # blocks, parity through forward_ids_batch.
    add_executable(test_gpu_parity_moe_batched tests/gpu/test_gpu_parity_moe_batched.cpp)
    target_link_libraries(test_gpu_parity_moe_batched PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_moe_batched PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_moe_batched COMMAND test_gpu_parity_moe_batched)

    # BitNet 1.58-bit __dp4a GPU dispatch vs CPU int8 GEMM (Phase 5a).
    add_executable(test_gpu_parity_bitnet_dispatch tests/gpu/test_gpu_parity_bitnet_dispatch.cpp)
    target_link_libraries(test_gpu_parity_bitnet_dispatch PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_bitnet_dispatch PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_bitnet_dispatch COMMAND test_gpu_parity_bitnet_dispatch)

    add_executable(test_gpu_decode_runtime tests/gpu/test_gpu_decode_runtime.cpp)
    target_link_libraries(test_gpu_decode_runtime PRIVATE nsos_core)
    target_include_directories(test_gpu_decode_runtime PRIVATE tests/gpu)
    add_test(NAME test_gpu_decode_runtime COMMAND test_gpu_decode_runtime)

    # Manual, separately measured experiment; never a timing assertion in CTest.
    add_executable(bench_decode_gemv benchmarks/bench_decode_gemv.cpp)
    target_link_libraries(bench_decode_gemv PRIVATE nsos_core)

    # KAN RBF-basis CUDA kernels (forward + backward derivative) parity.
    add_executable(test_gpu_parity_kan tests/gpu/test_gpu_parity_kan.cpp)
    target_link_libraries(test_gpu_parity_kan PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_kan PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_kan COMMAND test_gpu_parity_kan)

    add_executable(test_gpu_parity_ttt tests/gpu/test_gpu_parity_ttt.cpp)
    target_link_libraries(test_gpu_parity_ttt PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_ttt PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_ttt COMMAND test_gpu_parity_ttt)
    add_test(NAME test_gpu_parity_ttt_device COMMAND test_gpu_parity_ttt)
    add_test(NAME test_gpu_parity_ttt_full COMMAND test_gpu_parity_ttt)

    # SSA block-sparse attention CUDA kernels parity (raw + learned Wsel router).
    add_executable(test_gpu_parity_sparse_attention tests/gpu/test_gpu_parity_sparse_attention.cpp)
    target_link_libraries(test_gpu_parity_sparse_attention PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_sparse_attention PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_sparse_attention COMMAND test_gpu_parity_sparse_attention)

    # End-to-end: Jamba forward with SSA enabled must match CPU on GPU and must
    # actually differ from dense (gate for the WS-6b sparse_forward wiring).
    add_executable(test_gpu_parity_jamba_sparse tests/gpu/test_gpu_parity_jamba_sparse.cpp)
    target_link_libraries(test_gpu_parity_jamba_sparse PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_jamba_sparse PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_jamba_sparse COMMAND test_gpu_parity_jamba_sparse)

    # BitLinear::forward GPU __dp4a inference fast path (Phase 5b).
    add_executable(test_gpu_parity_bitlinear_dp4a tests/gpu/test_gpu_parity_bitlinear_dp4a.cpp)
    target_link_libraries(test_gpu_parity_bitlinear_dp4a PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_bitlinear_dp4a PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_bitlinear_dp4a COMMAND test_gpu_parity_bitlinear_dp4a)

    # QAT pack load + serving clone must automatically preserve and execute
    # packed DP4A inference.
    add_executable(test_gpu_model_pack_dp4a_lifecycle tests/gpu/test_gpu_model_pack_dp4a_lifecycle.cpp)
    target_link_libraries(test_gpu_model_pack_dp4a_lifecycle PRIVATE nsos_core)
    target_include_directories(test_gpu_model_pack_dp4a_lifecycle PRIVATE tests/gpu)
    add_test(NAME test_gpu_model_pack_dp4a_lifecycle COMMAND test_gpu_model_pack_dp4a_lifecycle)

    # cudaMalloc storage can be passed to CUDA APIs only; host inspection must
    # cross an explicit GPU->CPU copy boundary.
    add_executable(test_gpu_device_memory_contract tests/gpu/test_gpu_device_memory_contract.cpp)
    target_link_libraries(test_gpu_device_memory_contract PRIVATE nsos_core)
    target_include_directories(test_gpu_device_memory_contract PRIVATE tests/gpu)
    add_test(NAME test_gpu_device_memory_contract COMMAND test_gpu_device_memory_contract)

    add_executable(test_gpu_device_selection tests/gpu/test_gpu_device_selection.cpp)
    target_link_libraries(
        test_gpu_device_selection PRIVATE nsos_core Threads::Threads)
    target_include_directories(test_gpu_device_selection PRIVATE tests/gpu)
    add_test(NAME test_gpu_device_selection COMMAND test_gpu_device_selection)

    add_executable(test_gpu_checkpoint_continuation tests/gpu/test_gpu_checkpoint_continuation.cpp)
    target_link_libraries(test_gpu_checkpoint_continuation PRIVATE nsos_core)
    target_include_directories(test_gpu_checkpoint_continuation PRIVATE tests/gpu)
    add_test(NAME test_gpu_checkpoint_continuation COMMAND test_gpu_checkpoint_continuation)
    add_test(NAME test_gpu_checkpoint_continuation_redesign COMMAND test_gpu_checkpoint_continuation)
    add_test(NAME test_gpu_checkpoint_continuation_ttt_full COMMAND test_gpu_checkpoint_continuation)
    add_test(NAME test_gpu_checkpoint_continuation_moe_grouped COMMAND test_gpu_checkpoint_continuation)
    add_test(NAME test_gpu_checkpoint_continuation_kan_recompute COMMAND test_gpu_checkpoint_continuation)
    if (NSOS_ROCWMMA_ENABLED)
        add_test(NAME test_gpu_checkpoint_continuation_moe_wmma COMMAND test_gpu_checkpoint_continuation)
        add_test(NAME test_gpu_checkpoint_continuation_kan_wmma COMMAND test_gpu_checkpoint_continuation)
    endif()
    add_test(NAME test_gpu_checkpoint_continuation_embedding_rollback
        COMMAND test_gpu_checkpoint_continuation)
    add_test(NAME test_gpu_checkpoint_continuation_combined_finite_gate
        COMMAND test_gpu_checkpoint_continuation)
    add_test(NAME test_gpu_checkpoint_continuation_finite_scan_rollback
        COMMAND test_gpu_checkpoint_continuation)

    add_executable(test_gpu_mixed_precision_contract tests/gpu/test_gpu_mixed_precision_contract.cpp)
    target_link_libraries(test_gpu_mixed_precision_contract PRIVATE nsos_core)
    target_include_directories(test_gpu_mixed_precision_contract PRIVATE tests/gpu)
    add_test(NAME test_gpu_mixed_precision_contract COMMAND test_gpu_mixed_precision_contract)

    add_executable(test_gpu_lowp_weight_cache tests/gpu/test_gpu_lowp_weight_cache.cpp)
    target_link_libraries(test_gpu_lowp_weight_cache PRIVATE nsos_core)
    target_include_directories(test_gpu_lowp_weight_cache PRIVATE tests/gpu)
    add_test(NAME test_gpu_lowp_weight_cache COMMAND test_gpu_lowp_weight_cache)
    add_test(NAME test_gpu_lowp_weight_cache_zero_budget COMMAND test_gpu_lowp_weight_cache)

    add_executable(test_gpu_deterministic_adamw_1000 tests/gpu/test_gpu_deterministic_adamw_1000.cpp)
    target_link_libraries(test_gpu_deterministic_adamw_1000 PRIVATE nsos_core)
    target_include_directories(test_gpu_deterministic_adamw_1000 PRIVATE tests/gpu)
    add_test(NAME test_gpu_deterministic_adamw_1000 COMMAND test_gpu_deterministic_adamw_1000)
    add_test(NAME test_gpu_deterministic_adamw_1000_device_clip COMMAND test_gpu_deterministic_adamw_1000)
    set_tests_properties(test_gpu_deterministic_adamw_1000_device_clip PROPERTIES
        ENVIRONMENT "NSOS_REQUIRE_GPU_TESTS=1;NSOS_GPU_MEMORY=device;NSOS_NO_MEMADVISE=1;NSOS_DEVICE_GRAD_CLIP=1")
    add_test(NAME test_gpu_deterministic_adamw_1000_combined_finite_gate
             COMMAND test_gpu_deterministic_adamw_1000)
    set_tests_properties(test_gpu_deterministic_adamw_1000 PROPERTIES
        ENVIRONMENT "NSOS_REQUIRE_GPU_TESTS=1;NSOS_GPU_MEMORY=device;NSOS_NO_MEMADVISE=1")
    set_tests_properties(test_gpu_deterministic_adamw_1000_combined_finite_gate PROPERTIES
        ENVIRONMENT "NSOS_REQUIRE_GPU_TESTS=1;NSOS_GPU_MEMORY=device;NSOS_NO_MEMADVISE=1;NSOS_DETERMINISTIC_FINITE_GATE_DEFERRED=1")
    endif()

    # SmartLoader async semantics (real future-based wait, drain barrier,
    # concurrent producers, missing-file error path).  CPU-only, runs on
    # every build.
    add_executable(test_smart_loader_async tests/test_smart_loader_async.cpp)
    target_link_libraries(test_smart_loader_async PRIVATE nsos_core)
    add_test(NAME test_smart_loader_async COMMAND test_smart_loader_async)

    # Shared-model inference plus MemorySystem, NumericalGuard and Tensor
    # concurrency stress. Required by the product robustness gate.
    add_executable(test_thread_safety tests/test_thread_safety.cpp)
    target_link_libraries(
        test_thread_safety PRIVATE nsos_core Threads::Threads)
    add_test(NAME test_thread_safety COMMAND test_thread_safety)

    add_executable(test_http_api tests/test_http_api.cpp)
    target_link_libraries(test_http_api PRIVATE nsos_core)
    if (WIN32)
        target_link_libraries(test_http_api PRIVATE ws2_32)
    endif()
    add_test(NAME test_http_api COMMAND test_http_api)

    add_executable(test_circuit_api_pipeline tests/test_circuit_api_pipeline.cpp)
    target_link_libraries(test_circuit_api_pipeline PRIVATE nsos_core)
    if (WIN32)
        target_link_libraries(test_circuit_api_pipeline PRIVATE ws2_32)
    endif()
    add_test(NAME test_circuit_api_pipeline COMMAND test_circuit_api_pipeline)

    add_executable(test_holographic_full tests/test_holographic_full.cpp)
    target_link_libraries(test_holographic_full PRIVATE nsos_core)
    add_test(NAME test_holographic_full COMMAND test_holographic_full)

    add_executable(test_ultra_integration tests/test_ultra_integration.cpp)
    target_link_libraries(test_ultra_integration PRIVATE nsos_core)
    add_test(NAME test_ultra_integration COMMAND test_ultra_integration)

    add_executable(test_swarm_orchestration tests/test_swarm_orchestration.cpp)
    target_link_libraries(test_swarm_orchestration PRIVATE nsos_core)
    add_test(NAME test_swarm_orchestration COMMAND test_swarm_orchestration)

    add_executable(train_e2e tests/train_e2e.cpp)
    target_link_libraries(train_e2e PRIVATE nsos_core)
    target_include_directories(train_e2e PRIVATE include include/nsos)
    add_test(NAME train_e2e COMMAND train_e2e)

    # GPU vs CPU microbenchmark (Phase 3/4/5 surfaces).  Build target,
    # not registered as a CTest entry — runs ad-hoc with --report.
    add_executable(bench_gpu_vs_cpu tests/bench_gpu_vs_cpu.cpp)
    target_link_libraries(bench_gpu_vs_cpu PRIVATE nsos_core)
    target_include_directories(bench_gpu_vs_cpu PRIVATE include include/nsos)

    if (NSOS_GPU_ENABLED)
        # Build the list of tests to annotate dynamically — some targets
        # (test_oxtamem_ffi) are only created when their dependencies
        # are satisfied.  Listing them unconditionally would break
        # configuration when those dependencies are disabled.
        set(_NSOS_GPU_SYNC_TESTS
            test_sanity
            test_bitlinear
            test_bitnet_integrated
            test_mamba2
            test_kan
            test_ttt_layer_kernel
            test_inference_engine
            test_tokenizer
            test_mcts_reasoning_v3
            test_memory_causal_store
            test_jamba
            test_model_pack
            test_layer_audit
            test_gpu_parity
            test_gpu_parity_basic
            test_gpu_parity_bitlinear
            test_gpu_parity_mamba_stream
            test_gpu_parity_mamba_proper_stream
            test_gpu_parity_mamba_scan
            test_gpu_parity_mamba_proper
            test_gpu_parity_mamba_nstate
            test_gpu_parity_mamba_nstate_stream
            test_gpu_parity_mamba_faithful
            test_gpu_parity_mamba_faithful_rollback
            test_gpu_parity_mamba_faithful_shared_carry_optin
            test_gpu_parity_mamba_faithful_decay_optin
            test_gpu_parity_mamba_faithful_decay_rollback
            test_gpu_parity_mamba_faithful_state_parallel_optin
            test_gpu_parity_mamba_faithful_history_optin
            test_gpu_parity_mamba_faithful_history_rollback
            test_gpu_parity_mamba_faithful_chunked_optin
            test_gpu_parity_mamba_faithful_chunked_rollback
            test_gpu_parity_mamba_faithful_chunk_lds_state_major_optin
            test_gpu_parity_mamba_faithful_chunk_lds_state_major_rollback
            test_gpu_parity_mamba_faithful_fixed_state_optin
            test_gpu_parity_mamba_faithful_fixed_state_rollback
            test_gpu_parity_mamba_faithful_forward_chunked_optin
            test_gpu_parity_mamba_faithful_forward_chunked_rollback
            test_gpu_parity_decode_incremental
            test_gpu_parity_jamba
            test_gpu_parity_jamba_redesign
            test_gpu_training_redesign
            test_gpu_device_sparse_optimizer
            test_gpu_sparse_adam_transaction
            test_gpu_sparse_adam_trainer
            test_gpu_moe_grouped_training
            test_gpu_parity_moe_router
            test_gpu_parity_moe_batched
            test_gpu_parity_bitnet_dispatch
            test_gpu_decode_runtime
            test_gpu_parity_bitlinear_dp4a
            test_gpu_parity_kan
            test_gpu_parity_ttt
            test_gpu_parity_ttt_device
            test_gpu_parity_ttt_full
            test_gpu_parity_sparse_attention
            test_gpu_parity_jamba_sparse
            test_gpu_model_pack_dp4a_lifecycle
            test_gpu_device_memory_contract
            test_gpu_device_selection
            test_gpu_checkpoint_continuation
            test_gpu_checkpoint_continuation_kan_recompute
            test_gpu_checkpoint_continuation_redesign
            test_gpu_checkpoint_continuation_ttt_full
            test_gpu_checkpoint_continuation_moe_grouped
            test_gpu_checkpoint_continuation_embedding_rollback
            test_gpu_checkpoint_continuation_combined_finite_gate
            test_gpu_checkpoint_continuation_finite_scan_rollback
            test_gpu_mixed_precision_contract
            test_gpu_lowp_weight_cache
            test_gpu_lowp_weight_cache_zero_budget
            test_mamba_parallel_scan_parity
            test_smart_loader_async
            test_http_api
            test_circuit_api_pipeline
            test_holographic_full
            test_ultra_integration
            test_swarm_orchestration
            train_e2e
        )
        if (TARGET test_oxtamem_ffi)
            list(APPEND _NSOS_GPU_SYNC_TESTS test_oxtamem_ffi)
        endif()
        if (TARGET test_gpu_mamba3_siso)
            list(APPEND _NSOS_GPU_SYNC_TESTS test_gpu_mamba3_siso)
            list(APPEND _NSOS_GPU_SYNC_TESTS test_gpu_mamba3_projected)
        endif()
        if (NSOS_ROCWMMA_ENABLED)
            list(APPEND _NSOS_GPU_SYNC_TESTS test_gpu_moe_wmma_training test_gpu_checkpoint_continuation_moe_wmma test_gpu_checkpoint_continuation_kan_wmma test_gpu_attention_rdna_training)
        endif()
        set_tests_properties(
            ${_NSOS_GPU_SYNC_TESTS}
            PROPERTIES ENVIRONMENT
                "NSOS_CUDA_SYNC=1;NSOS_REQUIRE_GPU_TESTS=1;NSOS_GPU_MEMORY=device;NSOS_NO_MEMADVISE=1")
        set_property(TEST test_gpu_parity_jamba_redesign APPEND PROPERTY ENVIRONMENT
            "NSOS_ATTN_TILED_TRAINING=1;NSOS_MOE_ORDERED_DEVICE=1")
        set_property(TEST test_gpu_parity_ttt_device APPEND PROPERTY ENVIRONMENT
            "NSOS_TTT_DEVICE_RECURRENCE=1")
        set_property(TEST test_gpu_parity_ttt_full APPEND PROPERTY ENVIRONMENT
            "NSOS_TTT_FULL_BPTT=1")
        set_property(TEST test_gpu_checkpoint_continuation_redesign APPEND PROPERTY ENVIRONMENT
            "NSOS_MAMBA_BOUNDARY_HISTORY=1;NSOS_MAMBA_BACKWARD_CHUNK_SIZE=32;NSOS_DEVICE_GRAD_CLIP=1;NSOS_ATTN_TILED_TRAINING=1;NSOS_MOE_ORDERED_DEVICE=1;NSOS_TTT_DEVICE_RECURRENCE=1")
        set_property(TEST test_gpu_checkpoint_continuation_ttt_full APPEND PROPERTY ENVIRONMENT
            "NSOS_TTT_FULL_BPTT=1")
        set_property(TEST test_gpu_checkpoint_continuation_moe_grouped APPEND PROPERTY ENVIRONMENT
            "NSOS_MOE_GROUPED_TRAINING=1;NSOS_MOE_ORDERED_DEVICE=1")
        set_property(TEST test_gpu_checkpoint_continuation_kan_recompute APPEND PROPERTY ENVIRONMENT
            "NSOS_KAN_RECOMPUTE_TRAINING=1")
        if (NSOS_ROCWMMA_ENABLED)
            set_property(TEST test_gpu_checkpoint_continuation_kan_wmma APPEND PROPERTY ENVIRONMENT
                "NSOS_KAN_RECOMPUTE_TRAINING=1;NSOS_KAN_WMMA_TRAINING=1;NSOS_MIXED_PRECISION=bf16")
            set_property(TEST test_gpu_moe_wmma_training APPEND PROPERTY ENVIRONMENT "NSOS_MOE_WMMA_TRAINING=1")
            set_property(TEST test_gpu_checkpoint_continuation_moe_wmma APPEND PROPERTY ENVIRONMENT
                "NSOS_MOE_WMMA_TRAINING=1;NSOS_MOE_GROUPED_TRAINING=1;NSOS_MOE_ORDERED_DEVICE=1;NSOS_MIXED_PRECISION=bf16")
        endif()
        set_property(
            TEST test_gpu_lowp_weight_cache_zero_budget
            APPEND PROPERTY ENVIRONMENT "NSOS_LOWP_WEIGHT_CACHE_MIB=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY=1;NSOS_MAMBA_CONV_K4=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_shared_carry_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_DETERMINISTIC_SHARED_CARRY=1")
        set_property(
            TEST test_gpu_parity_mamba_faithful_decay_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_PRECOMPUTE_DECAY=1")
        set_property(
            TEST test_gpu_parity_mamba_faithful_decay_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_PRECOMPUTE_DECAY=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_state_parallel_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_STATE_PARALLEL_BACKWARD=1")
        set_property(
            TEST test_gpu_parity_mamba_faithful_history_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_STATE_MAJOR_HISTORY=1")
        set_property(
            TEST test_gpu_parity_mamba_faithful_history_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_STATE_MAJOR_HISTORY=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_chunked_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_CHUNKED_BACKWARD=1;NSOS_MAMBA_BACKWARD_CHUNK_SIZE=32")
        set_property(
            TEST test_gpu_parity_mamba_faithful_chunked_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_CHUNKED_BACKWARD=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_chunk_lds_state_major_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_CHUNK_LDS_STATE_MAJOR=1")
        set_property(
            TEST test_gpu_parity_mamba_faithful_chunk_lds_state_major_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_CHUNK_LDS_STATE_MAJOR=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_fixed_state_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_FAITHFUL_FIXED_STATE=1")
        set_property(
            TEST test_gpu_parity_mamba_faithful_fixed_state_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_FAITHFUL_FIXED_STATE=0")
        set_property(
            TEST test_gpu_parity_mamba_faithful_forward_chunked_optin
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD=1;NSOS_MAMBA_FORWARD_CHUNK_SIZE=8")
        set_property(
            TEST test_gpu_parity_mamba_faithful_forward_chunked_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD=0")
        set_property(
            TEST test_gpu_checkpoint_continuation_embedding_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_EMBEDDING_DENSE_DETERMINISTIC=1")
        set_property(
            TEST test_gpu_checkpoint_continuation_combined_finite_gate
            APPEND PROPERTY ENVIRONMENT
                "NSOS_DETERMINISTIC_FINITE_GATE_DEFERRED=1")
        set_property(
            TEST test_gpu_checkpoint_continuation_finite_scan_rollback
            APPEND PROPERTY ENVIRONMENT
                "NSOS_OPTIMIZER_FINITE_CHUNKED=0")

        # Windows does not search the HIP SDK runtime directory from an
        # arbitrary test working directory. Make every HIP-linked CTest
        # self-contained instead of relying on the terminal's PATH; otherwise
        # the loader opens modal "hipblas.dll not found" dialogs and stalls the
        # entire suite. ENVIRONMENT_MODIFICATION preserves the caller's PATH
        # while prepending the configured SDK runtime directory.
        if (NSOS_GPU_BACKEND_RESOLVED STREQUAL "HIP"
            AND _NSOS_HIP_RESOLVED_ROOT)
            # Every native test links nsos_core, and nsos_core links hipBLAS in
            # a HIP build. Even tests that execute only CPU assertions therefore
            # need the HIP runtime DLLs at process load time.
            get_property(
                _NSOS_HIP_RUNTIME_TESTS
                DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                PROPERTY TESTS)
            if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.22")
                set_tests_properties(
                    ${_NSOS_HIP_RUNTIME_TESTS}
                    PROPERTIES ENVIRONMENT_MODIFICATION
                        "PATH=path_list_prepend:${_NSOS_HIP_RESOLVED_ROOT}/bin")
            else()
                message(WARNING
                    "CMake <3.22 cannot inject the HIP runtime PATH into CTest; "
                    "launch tests with ${_NSOS_HIP_RESOLVED_ROOT}/bin on PATH")
            endif()
        endif()
    endif()

    # Configuration-derived inventory: documentation and validation artifacts
    # consume this file instead of maintaining fragile hand-written test
    # counts. Optional Python/OxtaMem/GPU gates are represented exactly as they
    # were registered in this build directory.
    get_property(
        _NSOS_REGISTERED_TESTS
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PROPERTY TESTS)
    list(SORT _NSOS_REGISTERED_TESTS)
    list(LENGTH _NSOS_REGISTERED_TESTS _NSOS_REGISTERED_TEST_COUNT)
    string(JOIN "\n" _NSOS_REGISTERED_TEST_LINES
           ${_NSOS_REGISTERED_TESTS})
    list(LENGTH NSOS_QUARANTINED_TEST_SOURCES
         _NSOS_QUARANTINED_TEST_COUNT)
    string(JOIN "\n" _NSOS_QUARANTINED_TEST_LINES
           ${NSOS_QUARANTINED_TEST_SOURCES})
    list(LENGTH NSOS_GPU_ONLY_TEST_SOURCES
         _NSOS_GPU_ONLY_TEST_COUNT)
    string(JOIN "\n" _NSOS_GPU_ONLY_TEST_LINES
           ${NSOS_GPU_ONLY_TEST_SOURCES})
    list(LENGTH NSOS_OXTAMEM_ONLY_TEST_SOURCES
         _NSOS_OXTAMEM_ONLY_TEST_COUNT)
    string(JOIN "\n" _NSOS_OXTAMEM_ONLY_TEST_LINES
           ${NSOS_OXTAMEM_ONLY_TEST_SOURCES})
    set(_NSOS_OXTAMEM_TEST_TARGET OFF)
    if (TARGET oxtamem_engine)
        set(_NSOS_OXTAMEM_TEST_TARGET ON)
    endif()
    file(WRITE "${CMAKE_BINARY_DIR}/nsos_test_inventory.txt"
        "format=nsos-ctest-inventory-v1\n"
        "source=${CMAKE_CURRENT_SOURCE_DIR}\n"
        "gpu_backend=${NSOS_GPU_BACKEND_RESOLVED}\n"
        "python_extension=${NSOS_BUILD_PYTHON}\n"
        "oxtamem_target=${_NSOS_OXTAMEM_TEST_TARGET}\n"
        "test_count=${_NSOS_REGISTERED_TEST_COUNT}\n"
        "tests_begin\n"
        "${_NSOS_REGISTERED_TEST_LINES}\n"
        "tests_end\n"
        "quarantined_count=${_NSOS_QUARANTINED_TEST_COUNT}\n"
        "quarantined_begin\n"
        "${_NSOS_QUARANTINED_TEST_LINES}\n"
        "quarantined_end\n"
        "gpu_only_source_count=${_NSOS_GPU_ONLY_TEST_COUNT}\n"
        "gpu_only_sources_begin\n"
        "${_NSOS_GPU_ONLY_TEST_LINES}\n"
        "gpu_only_sources_end\n"
        "oxtamem_only_source_count=${_NSOS_OXTAMEM_ONLY_TEST_COUNT}\n"
        "oxtamem_only_sources_begin\n"
        "${_NSOS_OXTAMEM_ONLY_TEST_LINES}\n"
        "oxtamem_only_sources_end\n")

endif()

# Standalone repro for the TTT-in-model construction crash (debug diagnosis).
# Keep diagnostic binaries behind the test option so production/Python-only
# configurations do not depend on files from the tests tree.
if(NSOS_BUILD_TESTS)
    add_executable(repro_ttt tests/repro_ttt.cpp)
    target_link_libraries(repro_ttt PRIVATE nsos_core)

    # Fail configuration when a new C++ test source is silently left outside
    # the executable graph, the explicit quarantine or a GPU-only source set.
    # GPU-only sources are accepted without targets solely in a CPU build; a
    # GPU build must classify every one through the executable graph.
    get_property(
        _NSOS_BUILD_TARGETS
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PROPERTY BUILDSYSTEM_TARGETS)
    set(_NSOS_TARGET_SOURCE_PATHS)
    foreach(_NSOS_TARGET IN LISTS _NSOS_BUILD_TARGETS)
        get_target_property(_NSOS_TARGET_SOURCES
                            ${_NSOS_TARGET} SOURCES)
        if (NOT _NSOS_TARGET_SOURCES OR
            _NSOS_TARGET_SOURCES STREQUAL
                "_NSOS_TARGET_SOURCES-NOTFOUND")
            continue()
        endif()
        foreach(_NSOS_SOURCE IN LISTS _NSOS_TARGET_SOURCES)
            if (_NSOS_SOURCE MATCHES "^\\$<")
                continue()
            endif()
            get_filename_component(
                _NSOS_SOURCE_ABSOLUTE "${_NSOS_SOURCE}" ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
            file(TO_CMAKE_PATH "${_NSOS_SOURCE_ABSOLUTE}"
                 _NSOS_SOURCE_ABSOLUTE)
            list(APPEND _NSOS_TARGET_SOURCE_PATHS
                 "${_NSOS_SOURCE_ABSOLUTE}")
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _NSOS_TARGET_SOURCE_PATHS)

    file(GLOB_RECURSE _NSOS_DISCOVERED_CPP_TESTS
         CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(_NSOS_TEST_SOURCE IN LISTS _NSOS_DISCOVERED_CPP_TESTS)
        file(TO_CMAKE_PATH "${_NSOS_TEST_SOURCE}"
             _NSOS_TEST_SOURCE_NORMALIZED)
        list(FIND _NSOS_TARGET_SOURCE_PATHS
             "${_NSOS_TEST_SOURCE_NORMALIZED}" _NSOS_TARGET_INDEX)
        file(RELATIVE_PATH _NSOS_TEST_SOURCE_RELATIVE
             "${CMAKE_CURRENT_SOURCE_DIR}"
             "${_NSOS_TEST_SOURCE_NORMALIZED}")
        file(TO_CMAKE_PATH "${_NSOS_TEST_SOURCE_RELATIVE}"
             _NSOS_TEST_SOURCE_RELATIVE)
        list(FIND NSOS_QUARANTINED_TEST_SOURCES
             "${_NSOS_TEST_SOURCE_RELATIVE}" _NSOS_QUARANTINE_INDEX)
        list(FIND NSOS_GPU_ONLY_TEST_SOURCES
             "${_NSOS_TEST_SOURCE_RELATIVE}" _NSOS_GPU_ONLY_INDEX)
        list(FIND NSOS_OXTAMEM_ONLY_TEST_SOURCES
             "${_NSOS_TEST_SOURCE_RELATIVE}" _NSOS_OXTAMEM_ONLY_INDEX)
        set(_NSOS_CONDITIONAL_GPU_INDEX -1)
        if (NOT NSOS_GPU_ENABLED AND
            NOT _NSOS_GPU_ONLY_INDEX EQUAL -1)
            set(_NSOS_CONDITIONAL_GPU_INDEX
                ${_NSOS_GPU_ONLY_INDEX})
        endif()
        set(_NSOS_CONDITIONAL_OXTAMEM_INDEX -1)
        if (NOT TARGET oxtamem_engine AND
            NOT _NSOS_OXTAMEM_ONLY_INDEX EQUAL -1)
            set(_NSOS_CONDITIONAL_OXTAMEM_INDEX
                ${_NSOS_OXTAMEM_ONLY_INDEX})
        endif()
        if (_NSOS_TARGET_INDEX EQUAL -1 AND
            _NSOS_QUARANTINE_INDEX EQUAL -1 AND
            _NSOS_CONDITIONAL_GPU_INDEX EQUAL -1 AND
            _NSOS_CONDITIONAL_OXTAMEM_INDEX EQUAL -1)
            message(FATAL_ERROR
                "Unclassified NSOS test source: "
                "${_NSOS_TEST_SOURCE_RELATIVE}. Register a target or add an "
                "explicitly justified quarantine entry.")
        endif()
        if (NOT _NSOS_TARGET_INDEX EQUAL -1 AND
            NOT _NSOS_QUARANTINE_INDEX EQUAL -1)
            message(FATAL_ERROR
                "NSOS test source is both registered and quarantined: "
                "${_NSOS_TEST_SOURCE_RELATIVE}")
        endif()
    endforeach()

    # Python gates are commands rather than build targets. Require every
    # test_*.py path to occur in the CMake registration graph.
    file(READ "${CMAKE_CURRENT_LIST_FILE}" _NSOS_CMAKE_SOURCE)
    file(GLOB_RECURSE _NSOS_DISCOVERED_PYTHON_TESTS
         CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py")
    foreach(_NSOS_PYTHON_TEST IN LISTS _NSOS_DISCOVERED_PYTHON_TESTS)
        file(RELATIVE_PATH _NSOS_PYTHON_TEST_RELATIVE
             "${CMAKE_CURRENT_SOURCE_DIR}" "${_NSOS_PYTHON_TEST}")
        file(TO_CMAKE_PATH "${_NSOS_PYTHON_TEST_RELATIVE}"
             _NSOS_PYTHON_TEST_RELATIVE)
        string(FIND "${_NSOS_CMAKE_SOURCE}"
                    "${_NSOS_PYTHON_TEST_RELATIVE}"
                    _NSOS_PYTHON_TEST_INDEX)
        if (_NSOS_PYTHON_TEST_INDEX EQUAL -1)
            message(FATAL_ERROR
                "Unclassified NSOS Python test source: "
                "${_NSOS_PYTHON_TEST_RELATIVE}")
        endif()
    endforeach()
endif()
