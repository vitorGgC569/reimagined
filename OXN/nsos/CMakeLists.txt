cmake_minimum_required(VERSION 3.18)
project(NSOS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NSOS_ENABLE_CUDA "Enable CUDA support when a compiler is available" ON)
option(NSOS_BUILD_PYTHON "Build the Python extension" ON)
option(NSOS_BUILD_TESTS "Build curated native smoke tests" ON)
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
option(NSOS_ENABLE_NATIVE_OPTIMIZATIONS "Enable host-specific CPU tuning flags such as -march=native" OFF)
option(NSOS_ENABLE_X86_AVX2_KERNELS "Build runtime-dispatched AVX2 CPU kernels on x86_64" ON)
set(NSOS_SANITIZERS "" CACHE STRING "Semicolon-separated sanitizers to enable for CPU builds (address;undefined;thread)")

set(NSOS_CUDA_TOOLKIT_ROOT "" CACHE PATH "Preferred CUDA toolkit root directory")
set(NSOS_CUDA_ARCHITECTURES "61;75;80;86" CACHE STRING "CUDA architectures for NSOS (e.g. 61;75;86)")
set(NSOS_ZIG_EXECUTABLE "" CACHE FILEPATH "Path to zig executable used to build TurboQuant")
set(NSOS_TURBOQUANT_DIR "" CACHE PATH "Path to TurboQuant source directory")
set(NSOS_TURBOQUANT_LIB "" CACHE FILEPATH "Path to a prebuilt TurboQuant static library")
set(NSOS_OXTAMEM_DIR "${CMAKE_CURRENT_LIST_DIR}/../../modules/oxtamem/oxta_engine" CACHE PATH "Path to the copied OxtaMem Rust engine")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(NSOS_CORE_SOURCES
    src/tensor.cpp
    src/nsos_math.cpp
    src/jamba.cpp
    src/jamba_utils.cpp
    src/mamba2.cpp
    src/bitlinear.cpp
    src/bitnet_adapter.cpp
    src/lut_tmac.cpp
    src/lut_cas.cpp
    src/embedding.cpp
    src/kan.cpp
    src/ttt_layer.cpp
    src/trainer.cpp
    src/layer_audit.cpp
    src/tokenizer.cpp
    src/nsos_sdk.cpp
    src/nsos_mpi.cpp
    src/globals.cpp
    src/holographic.cpp
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
 )

if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/O2> $<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
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

if (NSOS_ENABLE_CUDA)
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
        set(CMAKE_CUDA_STANDARD 17)
        if (NSOS_CUDA_ALLOW_UNSUPPORTED_COMPILER)
            string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set_property(TARGET nsos_core PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        target_compile_definitions(nsos_core PUBLIC USE_CUDA)
        set(NSOS_CUDA_MIN_ARCH 0)
        foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
            if (_arch MATCHES "^[0-9]+$")
                if (NSOS_CUDA_MIN_ARCH EQUAL 0 OR _arch LESS NSOS_CUDA_MIN_ARCH)
                    set(NSOS_CUDA_MIN_ARCH ${_arch})
                endif()
            endif()
        endforeach()
        if (NOT NSOS_CUDA_MIN_ARCH EQUAL 0)
            target_compile_definitions(nsos_core PUBLIC NSOS_CUDA_MIN_ARCH=${NSOS_CUDA_MIN_ARCH})
        endif()
        target_sources(nsos_core PRIVATE
            src/cuda/kernels.cu
            src/cuda/persistent_kernel.cu
            src/cuda/mamba_kernels.cu
            src/cuda/moe_kernels.cu
            src/cuda/bitnet_kernels.cu
            src/cuda/flash_attention.cu
        )
        # PUBLIC because tensor.h includes <cuda_runtime.h> when USE_CUDA
        # is defined; tests/CLI/API targets that link nsos_core need to
        # inherit the CUDA include directory to compile that header.
        target_link_libraries(nsos_core PUBLIC CUDA::cudart CUDA::cublas)
        message(STATUS "Using CUDA toolkit: ${CUDAToolkit_BIN_DIR}")
        message(STATUS "Using CUDA toolkit version: ${CUDAToolkit_VERSION}")
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

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
            COMMAND ${CARGO_EXECUTABLE} build --release
            WORKING_DIRECTORY ${NSOS_OXTAMEM_DIR}
            COMMENT "Building copied OxtaMem engine"
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

    add_executable(test_sanity tests/test_sanity.cpp)
    target_link_libraries(test_sanity PRIVATE nsos_core)
    add_test(NAME test_sanity COMMAND test_sanity)

    add_executable(test_bitlinear tests/test_bitlinear.cpp)
    target_link_libraries(test_bitlinear PRIVATE nsos_core)
    add_test(NAME test_bitlinear COMMAND test_bitlinear)

    # Cherry-pick #2 (Slender-Mamba head-to-toe quantization).
    # Tests the slender forward path on Embedding: cache lifecycle,
    # round-trip bounds, STE backward routing, and edge cases.
    # See docs/SLENDER_INTEGRATION.md.
    add_executable(test_slender_embedding tests/test_slender_embedding.cpp)
    target_link_libraries(test_slender_embedding PRIVATE nsos_core)
    add_test(NAME test_slender_embedding COMMAND test_slender_embedding)

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

    add_executable(test_bitnet_integrated tests/test_bitnet_integrated.cpp)
    target_link_libraries(test_bitnet_integrated PRIVATE nsos_core)
    add_test(NAME test_bitnet_integrated COMMAND test_bitnet_integrated)

    add_executable(test_mamba2 tests/test_mamba2.cpp)
    target_link_libraries(test_mamba2 PRIVATE nsos_core)
    add_test(NAME test_mamba2 COMMAND test_mamba2)

    add_executable(test_kan tests/test_kan.cpp)
    target_link_libraries(test_kan PRIVATE nsos_core)
    add_test(NAME test_kan COMMAND test_kan)

    add_executable(test_ttt_layer_kernel tests/test_ttt_layer_kernel.cpp)
    target_link_libraries(test_ttt_layer_kernel PRIVATE nsos_core)
    add_test(NAME test_ttt_layer_kernel COMMAND test_ttt_layer_kernel)

    add_executable(test_inference_engine tests/test_inference_engine.cpp)
    target_link_libraries(test_inference_engine PRIVATE nsos_core)
    add_test(NAME test_inference_engine COMMAND test_inference_engine)

    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE nsos_core)
    add_test(NAME test_tokenizer COMMAND test_tokenizer)

    if (NSOS_BUILD_PYTHON)
        add_test(
            NAME test_python_binding_smoke
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_binding_smoke.py
        )
        set_tests_properties(
            test_python_binding_smoke
            PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:nsos_ext>")
    endif()

    add_executable(test_mcts_reasoning_v3 tests/test_mcts_reasoning_v3.cpp)
    target_link_libraries(test_mcts_reasoning_v3 PRIVATE nsos_core)
    add_test(NAME test_mcts_reasoning_v3 COMMAND test_mcts_reasoning_v3)

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

    add_executable(test_gpu_parity tests/test_gpu_parity.cpp)
    target_link_libraries(test_gpu_parity PRIVATE nsos_core)
    add_test(NAME test_gpu_parity COMMAND test_gpu_parity)

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

    add_executable(test_gpu_parity_jamba tests/gpu/test_gpu_parity_jamba.cpp)
    target_link_libraries(test_gpu_parity_jamba PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_jamba PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_jamba COMMAND test_gpu_parity_jamba)

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

    # BitLinear::forward GPU __dp4a inference fast path (Phase 5b).
    add_executable(test_gpu_parity_bitlinear_dp4a tests/gpu/test_gpu_parity_bitlinear_dp4a.cpp)
    target_link_libraries(test_gpu_parity_bitlinear_dp4a PRIVATE nsos_core)
    target_include_directories(test_gpu_parity_bitlinear_dp4a PRIVATE tests/gpu)
    add_test(NAME test_gpu_parity_bitlinear_dp4a COMMAND test_gpu_parity_bitlinear_dp4a)

    # SmartLoader async semantics (real future-based wait, drain barrier,
    # concurrent producers, missing-file error path).  CPU-only, runs on
    # every build.
    add_executable(test_smart_loader_async tests/test_smart_loader_async.cpp)
    target_link_libraries(test_smart_loader_async PRIVATE nsos_core)
    add_test(NAME test_smart_loader_async COMMAND test_smart_loader_async)

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

    if (CMAKE_CUDA_COMPILER)
        # Build the list of tests to annotate dynamically — some targets
        # (test_oxtamem_ffi) are only created when their dependencies
        # are satisfied.  Listing them unconditionally would break
        # configuration when those dependencies are disabled.
        set(_NSOS_CUDA_SYNC_TESTS
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
            test_gpu_parity_mamba_scan
            test_gpu_parity_jamba
            test_gpu_parity_moe_router
            test_gpu_parity_moe_batched
            test_gpu_parity_bitnet_dispatch
            test_gpu_parity_bitlinear_dp4a
            test_smart_loader_async
            test_http_api
            test_circuit_api_pipeline
            test_holographic_full
            test_ultra_integration
            test_swarm_orchestration
            train_e2e
        )
        if (TARGET test_oxtamem_ffi)
            list(APPEND _NSOS_CUDA_SYNC_TESTS test_oxtamem_ffi)
        endif()
        set_tests_properties(
            ${_NSOS_CUDA_SYNC_TESTS}
            PROPERTIES ENVIRONMENT "NSOS_CUDA_SYNC=1")
    endif()
endif()
