#ifndef CUDA_MEMORY_CUH
#define CUDA_MEMORY_CUH

#include <iostream>
#include <stdexcept>
#include <string>

// CUDA Runtime Stub/Header if nvcc not present
#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
// Minimal mocks for compilation on CPU-only envs to verify structure
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
typedef int cudaError_t;
#define cudaSuccess 0
extern "C" {
inline cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                     unsigned int flags = 0) {
  *devPtr = nullptr;
  return cudaSuccess;
}
inline cudaError_t cudaFree(void *devPtr) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline const char *cudaGetErrorString(cudaError_t err) {
  return "CUDA disabled";
}
inline cudaError_t cudaGetDeviceCount(int *count) {
  *count = 0;
  return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device) { return cudaSuccess; }
}
#endif

class CudaAllocator {
private:
  static bool s_initialized;

  static bool initialize() {
    if (s_initialized)
      return true;

#ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
      std::cerr << "[CUDA] cudaGetDeviceCount failed: "
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

    if (deviceCount == 0) {
      std::cerr << "[CUDA] No CUDA-capable devices found" << std::endl;
      return false;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      std::cerr << "[CUDA] cudaSetDevice(0) failed: " << cudaGetErrorString(err)
                << std::endl;
      return false;
    }

    // Warm up the context by doing a small allocation
    void *warmup_ptr = nullptr;
    err = cudaMallocManaged(&warmup_ptr, 4);
    if (err != cudaSuccess) {
      std::cerr << "[CUDA] Context warmup allocation failed: "
                << cudaGetErrorString(err) << std::endl;
      return false;
    }
    cudaFree(warmup_ptr);

    std::cerr << "[CUDA] Initialized successfully with " << deviceCount
              << " device(s)" << std::endl;
    s_initialized = true;
    return true;
#else
    return false;
#endif
  }

public:
  static void *allocate(size_t size) {
    if (!initialize()) {
      std::cerr << "[CUDA] Allocation failed: CUDA not initialized"
                << std::endl;
      return nullptr;
    }

    void *ptr = nullptr;
#ifdef USE_CUDA
    // Allocate Unified Memory (accessible by both CPU and GPU)
    // This is crucial for Zero-Copy requirements.
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
      std::cerr << "[CUDA] cudaMallocManaged(" << size
                << " bytes) failed: " << cudaGetErrorString(err) << std::endl;
      return nullptr;
    }

    // Prefetch to GPU for better performance
    cudaMemPrefetchAsync(ptr, size, 0, 0);
#endif
    return ptr;
  }

  static void deallocate(void *ptr) {
    if (ptr) {
#ifdef USE_CUDA
      cudaFree(ptr);
#endif
    }
  }
};

// Static member initialization
inline bool CudaAllocator::s_initialized = false;

#endif
