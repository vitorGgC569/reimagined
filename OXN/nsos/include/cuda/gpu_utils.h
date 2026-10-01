#pragma once

#ifdef USE_CUDA
#include "../gpu_backend.h"
#include <mutex>
#endif

namespace nsos {

#ifdef USE_CUDA
inline bool gpu_custom_kernels_supported() {
#ifndef NSOS_CUDA_MIN_ARCH
#define NSOS_CUDA_MIN_ARCH 0
#endif
  int device_id = -1;
  if (!gpu::select_preferred_device(&device_id, nullptr)) {
    return false;
  }

  static std::once_flag capability_once;
  static int capability_device = -1;
  static bool supported = false;
  std::call_once(capability_once, [device_id] {
    cudaDeviceProp props{};
    const cudaError_t status =
        cudaGetDeviceProperties(&props, device_id);
    if (status != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }

#if defined(NSOS_GPU_BACKEND_CUDA)
    const int device_arch = props.major * 10 + props.minor;
    supported =
        NSOS_CUDA_MIN_ARCH <= 0 ||
        device_arch >= NSOS_CUDA_MIN_ARCH;
#else
    // select_preferred_device already rejected HIP architectures absent from
    // this binary.
    supported = true;
#endif
    capability_device = device_id;
  });
  return capability_device == device_id && supported;
}
#else
inline bool gpu_custom_kernels_supported() { return false; }
#endif

} // namespace nsos
