#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

#ifdef USE_CUDA
inline bool gpu_custom_kernels_supported() {
#ifndef NSOS_CUDA_MIN_ARCH
#define NSOS_CUDA_MIN_ARCH 0
#endif
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }

  const cudaError_t context_status = cudaFree(nullptr);
  if (context_status != cudaSuccess) {
    cached = 0;
    return false;
  }

  int device_id = 0;
  cudaError_t status = cudaGetDevice(&device_id);
  if (status != cudaSuccess) {
    cached = 0;
    return false;
  }

  cudaDeviceProp props{};
  status = cudaGetDeviceProperties(&props, device_id);
  if (status != cudaSuccess) {
    cached = 0;
    return false;
  }

  const int device_arch = props.major * 10 + props.minor;
  cached =
      (NSOS_CUDA_MIN_ARCH <= 0 || device_arch >= NSOS_CUDA_MIN_ARCH) ? 1 : 0;
  return cached == 1;
}
#else
inline bool gpu_custom_kernels_supported() { return false; }
#endif

} // namespace nsos
