#ifndef NSOS_TESTS_GPU_PARITY_COMMON_H
#define NSOS_TESTS_GPU_PARITY_COMMON_H

// =====================================================================
// Common helpers for the decomposed GPU parity test suite.
//
// The previous monolithic `test_gpu_parity.cpp` ran every CUDA kernel
// (tensor add, matmul, rmsnorm, bitlinear, mamba streaming, jamba batch
// forward) inside a single binary.  When any one of those segfaulted —
// notably the streaming mamba scan on CUDA 12.9 — the entire suite
// failed and the others were never observed.  Splitting one kernel per
// binary lets CTest report which surface broke and keeps the rest of
// the parity gate honest.
//
// All helpers here are header-only and side-effect free.  They are not
// part of the product surface; including them outside the `tests/gpu/`
// tree is unsupported.
// =====================================================================

#include "tensor.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include "gpu_backend.h"
#endif

namespace nsos {
namespace gpu_parity_test {

// GPU binaries are fail-closed by design: invoking one without a GPU build
// and a selectable device is a test failure, never a successful "skip".
inline void require_cuda_device(const char* test_name) {
#ifndef USE_CUDA
  throw std::runtime_error(
      std::string("[GPUParity:") + test_name +
      "] a GPU backend is required but was not enabled at build time");
#else
  int selected_device = -1;
  std::string selection_error;
  if (!gpu::select_preferred_device(&selected_device, &selection_error)) {
    throw std::runtime_error(
        std::string("[GPUParity:") + test_name +
        "] GPU device is required but unavailable: " + selection_error);
  }
  cudaDeviceProp props{};
  const cudaError_t props_status =
      cudaGetDeviceProperties(&props, selected_device);
  if (props_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("[GPUParity:") + test_name +
        "] GPU device property query failed: " +
        cudaGetErrorString(props_status));
  }
  std::cout << "[GPUParity:" << test_name << "] device=" << props.name
            << " backend=" << gpu::backend_name();
#if defined(NSOS_GPU_BACKEND_HIP)
  std::cout << " arch=" << props.gcnArchName;
#else
  std::cout << " arch=sm_" << props.major << props.minor;
#endif
  std::cout << " runtime=" << CUDART_VERSION << std::endl;
#endif
}

// Synchronizes the GPU and throws a std::runtime_error if the prior
// launch failed.  Test bodies must call this between phases so that
// asynchronous kernel errors surface here instead of crashing the
// process at an arbitrary later point.
inline void cuda_sync_or_throw(const char* phase) {
#ifdef USE_CUDA
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("GPU sync failed at ") + phase +
                             ": " + cudaGetErrorString(err));
  }
#else
  (void)phase;
#endif
}

// Asserts elementwise closeness between two tensors.  Materializes both
// on CPU first so it works for either device.  Throws on mismatch with
// the offending index, the two values and the absolute delta — those
// are the data points debugging requires; throwing a bare assertion
// silently strips them.
inline void assert_close(const Tensor& lhs, const Tensor& rhs, float atol,
                         const char* label, float rtol = 0.0f) {
  Tensor lhs_cpu = lhs.cpu();
  Tensor rhs_cpu = rhs.cpu();
  if (lhs_cpu.size != rhs_cpu.size) {
    throw std::runtime_error(std::string(label) +
                             ": size mismatch lhs.size=" +
                             std::to_string(lhs_cpu.size) +
                             " rhs.size=" + std::to_string(rhs_cpu.size));
  }
  for (int i = 0; i < lhs_cpu.size; ++i) {
    const float a = lhs_cpu.data()[i];
    const float b = rhs_cpu.data()[i];
    const float diff = std::abs(a - b);
    const float threshold =
        atol + rtol * std::max(std::abs(a), std::abs(b));
    if (!(diff <= threshold)) {
      char buf[320];
      std::snprintf(
          buf, sizeof(buf),
          "%s: mismatch at index %d "
          "(lhs=%.6f rhs=%.6f |delta|=%.6f atol=%.6f rtol=%.8f "
          "threshold=%.6f)",
          label, i, a, b, diff, atol, rtol, threshold);
      throw std::runtime_error(buf);
    }
  }
}

// Wraps a test body, catches std::exception, prints a single-line
// PASS/FAIL banner, and returns a process exit code.  Test mains call
// this exactly once.
template <typename Fn>
int run_parity(const char* test_name, Fn&& body) {
  try {
    require_cuda_device(test_name);
    std::cout << "[GPUParity:" << test_name << "] begin" << std::endl;
    body();
    cuda_sync_or_throw(test_name);
    std::cout << "[GPUParity:" << test_name << "] PASS" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[GPUParity:" << test_name << "] FAIL: " << ex.what()
              << std::endl;
    return 1;
  }
}

} // namespace gpu_parity_test
} // namespace nsos

#endif // NSOS_TESTS_GPU_PARITY_COMMON_H
