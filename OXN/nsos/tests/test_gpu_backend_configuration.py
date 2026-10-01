#!/usr/bin/env python3
"""Static invariants shared by the CUDA/NVIDIA and HIP/AMD build paths.

This suite deliberately needs neither a GPU nor a vendor toolkit.  It is not
a replacement for compiling and running on each vendor; it prevents the build
graph and compatibility header from silently becoming vendor-specific between
hardware validation runs.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
GPU_HEADER = (ROOT / "include" / "gpu_backend.h").read_text(encoding="utf-8")
CORE_KERNELS = (ROOT / "src" / "cuda" / "kernels.cu").read_text(
    encoding="utf-8"
)
BITNET_KERNELS = (ROOT / "src" / "cuda" / "bitnet_kernels.cu").read_text(
    encoding="utf-8"
)
ALL_GPU_LAUNCH_SOURCES = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "src" / "cuda").glob("*.cu"))
)

EXPECTED_GPU_SOURCES = {
    "src/cuda/attention_train_kernels.cu",
    "src/cuda/attention_rdna_training.cu",
    "src/cuda/tiled_cross_entropy.cu",
    "src/cuda/sparse_optimizer_activity.cu",
    "src/cuda/bitnet_kernels.cu",
    "src/cuda/fused_optimizer_kernels.cu",
    "src/cuda/kan_kernels.cu",
    "src/cuda/kernels.cu",
    "src/cuda/mamba_kernels.cu",
    "src/cuda/mamba3_siso_kernels.cu",
    "src/cuda/mamba3_preprocess_kernels.cu",
    "src/cuda/mamba3_layer_kernels.cu",
    "src/cuda/moe_kernels.cu",
    "src/cuda/moe_training_kernels.cu",
    "src/cuda/moe_training_wmma.cu",
    "src/cuda/sparse_attention_kernels.cu",
}


class GpuBackendConfigurationTests(unittest.TestCase):
    def test_shared_kernel_manifest_is_complete(self) -> None:
        match = re.search(
            r"set\s*\(\s*NSOS_GPU_SOURCES(?P<body>.*?)\)",
            CMAKE,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "NSOS_GPU_SOURCES is missing")
        configured = set(
            re.findall(r"src/cuda/[A-Za-z0-9_./-]+\.cu", match.group("body"))
        )
        on_disk = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "src" / "cuda").glob("*.cu")
        }
        self.assertEqual(configured, EXPECTED_GPU_SOURCES)
        self.assertEqual(on_disk, EXPECTED_GPU_SOURCES)

    def test_same_kernel_manifest_feeds_both_compilers(self) -> None:
        self.assertIn(
            "add_library(nsos_hip_kernels OBJECT ${NSOS_GPU_SOURCES})", CMAKE
        )
        self.assertIn(
            "target_sources(nsos_core PRIVATE ${NSOS_GPU_SOURCES})", CMAKE
        )

    def test_cuda_build_contract_is_preserved(self) -> None:
        cuda_start = CMAKE.index(
            'elseif (NSOS_GPU_BACKEND_REQUESTED STREQUAL "CUDA"'
        )
        cuda_end = CMAKE.index(
            'message(STATUS "NSOS GPU backend:', cuda_start
        )
        cuda_branch = CMAKE[cuda_start:cuda_end]
        for required in (
            "check_language(CUDA)",
            "enable_language(CUDA)",
            "find_package(CUDAToolkit REQUIRED)",
            "NSOS_GPU_BACKEND_CUDA=1",
            "NSOS_CUDA_ARCHITECTURES_CSV",
            "Invalid NSOS_CUDA_ARCHITECTURES entry",
            "CUDA::cudart",
            "CUDA::cublas",
            'set(NSOS_GPU_BACKEND_RESOLVED "CUDA")',
            "set(NSOS_GPU_ENABLED ON)",
            "NSOS_GPU_BACKEND=CUDA was requested but no CUDA compiler was found",
        ):
            self.assertIn(required, cuda_branch)

    def test_hip_build_contract_is_isolated_from_host_sources(self) -> None:
        hip_start = CMAKE.index(
            'if (NSOS_GPU_BACKEND_REQUESTED STREQUAL "HIP"'
        )
        hip_end = CMAKE.index(
            'elseif (NSOS_GPU_BACKEND_REQUESTED STREQUAL "CUDA"', hip_start
        )
        hip_branch = CMAKE[hip_start:hip_end]
        for required in (
            "find_package(hip CONFIG REQUIRED)",
            "find_package(hipblas CONFIG REQUIRED)",
            "add_library(nsos_hip_kernels OBJECT ${NSOS_GPU_SOURCES})",
            "NSOS_GPU_BACKEND_HIP=1",
            "NSOS_HIP_ARCHITECTURES_CSV",
            "Invalid NSOS_HIP_ARCHITECTURES entry",
            "hip::device",
            "hip::host",
            "roc::hipblas",
            'set(NSOS_GPU_BACKEND_RESOLVED "HIP")',
            "set(NSOS_GPU_ENABLED ON)",
        ):
            self.assertIn(required, hip_branch)

    def test_vendor_runtime_headers_are_centralized(self) -> None:
        vendor_runtime_include = re.compile(
            r'#\s*include\s*[<"]'
            r"(?:cuda_runtime|cuda_bf16|cuda_fp16|cublas_v2|hip/|hipblas/)"
        )
        offenders: list[str] = []
        for extension in ("*.h", "*.hpp", "*.cuh", "*.cpp", "*.cu"):
            for path in ROOT.rglob(extension):
                if any(part.startswith("build") for part in path.parts) or "artifacts" in path.relative_to(ROOT).parts:
                    continue
                if path == ROOT / "include" / "gpu_backend.h":
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if vendor_runtime_include.search(text):
                    offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_compatibility_header_keeps_vendor_branches_distinct(self) -> None:
        cuda_marker = "#if defined(NSOS_GPU_BACKEND_CUDA)\n"
        hip_marker = "#elif defined(NSOS_GPU_BACKEND_HIP)\n"
        cuda_start = GPU_HEADER.index(cuda_marker) + len(cuda_marker)
        hip_start = GPU_HEADER.index(hip_marker, cuda_start)
        none_start = GPU_HEADER.index("\n#else\n", hip_start)
        cuda_section = GPU_HEADER[cuda_start:hip_start]
        hip_section = GPU_HEADER[hip_start:none_start]

        for header in (
            "<cuda_bf16.h>",
            "<cuda_fp16.h>",
            "<cuda_runtime.h>",
            "<cublas_v2.h>",
        ):
            self.assertIn(header, cuda_section)
        self.assertNotIn("<hip/", cuda_section)
        self.assertNotIn("<hipblas/", cuda_section)

        for header in (
            "<hip/hip_bfloat16.h>",
            "<hip/hip_fp16.h>",
            "<hip/hip_runtime.h>",
            "<hipblas/hipblas.h>",
        ):
            self.assertIn(header, hip_section)
        self.assertNotIn("<cuda_runtime.h>", hip_section)
        self.assertIn("#define cudaMalloc hipMalloc", hip_section)
        self.assertIn(
            "#define cudaRuntimeGetVersion hipRuntimeGetVersion", hip_section
        )
        self.assertIn(
            "#define cudaDriverGetVersion hipDriverGetVersion", hip_section
        )
        self.assertIn("#define cudaStreamWaitEvent hipStreamWaitEvent", hip_section)
        self.assertIn("#define cublasSgemm hipblasSgemm", hip_section)
        self.assertIn("#define cublasGetVersion hipblasGetVersion", hip_section)

    def test_reduction_primitives_follow_native_wave_width(self) -> None:
        self.assertIn("constexpr int kReductionThreads = 256", CORE_KERNELS)
        self.assertIn("warpSize / 2", CORE_KERNELS)
        self.assertIn("threadIdx.x % warpSize", CORE_KERNELS)
        self.assertIn("threadIdx.x / warpSize", CORE_KERNELS)
        self.assertNotIn("#define WARP_SIZE 32", CORE_KERNELS)
        for launcher in (
            "rmsnorm_kernel<<<n_rows, kReductionThreads, 0, nsos::gpu::current_stream()>>>",
            "layernorm_kernel<<<n_rows, kReductionThreads, 0, nsos::gpu::current_stream()>>>",
            "softmax_kernel<<<outer, kReductionThreads, 0, nsos::gpu::current_stream()>>>",
            "rmsnorm_backward_kernel<<<outer, kReductionThreads, 0, nsos::gpu::current_stream()>>>",
        ):
            self.assertIn(launcher, CORE_KERNELS)
        self.assertIn("constexpr int kQuantThreads = 256", BITNET_KERNELS)
        self.assertIn("warpSize / 2", BITNET_KERNELS)
        self.assertIn("threadIdx.x % warpSize", BITNET_KERNELS)
        self.assertNotIn("BITNET_WARP_SIZE", BITNET_KERNELS)

    def test_grid_ceiling_arithmetic_is_overflow_safe(self) -> None:
        self.assertIn("constexpr std::common_type_t", GPU_HEADER)
        self.assertIn("ceil_div_positive(", GPU_HEADER)
        self.assertIn("n / d + static_cast<Common>((n % d) != 0)", GPU_HEADER)
        unsafe_grid_ceiling = re.compile(
            r"\([^\n;]*\+[^\n;]*(?:threads|kThreads|block_size|block|BLOCK)"
            r"[^\n;]*-[^\n;]*1[^\n;]*\)\s*/\s*"
            r"(?:threads|kThreads|block_size|block|BLOCK)"
        )
        self.assertIsNone(unsafe_grid_ceiling.search(ALL_GPU_LAUNCH_SOURCES))

    def test_gpu_tests_are_backend_neutral_and_fail_closed(self) -> None:
        tests_start = CMAKE.index(
            "if (NSOS_BUILD_TESTS)\n"
            "    enable_testing()"
        )
        tests_block = CMAKE[tests_start:]
        self.assertNotIn("if (CMAKE_CUDA_COMPILER)", tests_block)
        self.assertGreaterEqual(tests_block.count("if (NSOS_GPU_ENABLED)"), 2)
        for required in (
            "NSOS_CUDA_SYNC=1",
            "NSOS_REQUIRE_GPU_TESTS=1",
            "NSOS_GPU_MEMORY=device",
            "NSOS_NO_MEMADVISE=1",
            "test_gpu_device_selection",
        ):
            self.assertIn(required, tests_block)


if __name__ == "__main__":
    unittest.main()
