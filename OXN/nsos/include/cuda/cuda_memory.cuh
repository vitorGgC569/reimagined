#ifndef CUDA_MEMORY_CUH
#define CUDA_MEMORY_CUH

#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

// The real HIP/CUDA runtime API is available only when the GPU backend is
// compiled. CPU-only builds retain the allocator type, whose initialize()
// contract fails closed without declaring runtime functions that could mimic
// successful GPU work.
#ifdef USE_CUDA
#include "../gpu_backend.h"
#endif

class CudaAllocator {
private:
  static bool s_initialized;
  static int s_device_id;
  static std::once_flag s_initialize_once;

  static bool initialize() {
#ifdef USE_CUDA
    std::call_once(s_initialize_once, [] {
      std::string selection_error;
      if (!nsos::gpu::select_preferred_device(
              &s_device_id, &selection_error)) {
        std::cerr << "[GPU] device selection failed: "
                  << selection_error << std::endl;
        return;
      }
      void *warmup_ptr = nullptr;
      const cudaError_t allocation_status = cudaMalloc(&warmup_ptr, 4);
      if (allocation_status != cudaSuccess) {
        std::cerr << "[GPU] context warmup allocation failed: "
                  << cudaGetErrorString(allocation_status) << std::endl;
        return;
      }
      const cudaError_t free_status = cudaFree(warmup_ptr);
      if (free_status != cudaSuccess) {
        std::cerr << "[GPU] context warmup free failed: "
                  << cudaGetErrorString(free_status) << std::endl;
        return;
      }
      s_initialized = true;
    });
    if (s_initialized) {
      int selected_device = -1;
      if (!nsos::gpu::select_preferred_device(
              &selected_device, nullptr) ||
          selected_device != s_device_id) {
        return false;
      }
    }
    return s_initialized;
#else
    return false;
#endif
  }

public:
  static void *allocate(size_t size) {
    if (size == 0) {
      return nullptr;
    }
    if (!initialize()) {
      std::cerr << "[GPU] Allocation failed: GPU runtime not initialized"
                << std::endl;
      return nullptr;
    }

    void *ptr = nullptr;
#ifdef USE_CUDA
    // This allocator follows the production device-only memory policy. Host
    // access requires an explicit checked transfer through the owning API.
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      std::cerr << "[GPU] device allocation (" << size
                << " bytes) failed: " << cudaGetErrorString(err) << std::endl;
      return nullptr;
    }
#endif
    return ptr;
  }

  static void deallocate(void *ptr) {
    if (ptr) {
#ifdef USE_CUDA
      int selected_device = -1;
      std::string selection_error;
      if (!nsos::gpu::select_preferred_device(
              &selected_device, &selection_error)) {
        std::cerr << "[GPU] device deallocation selection failed: "
                  << selection_error << std::endl;
        return;
      }
      if (selected_device != s_device_id) {
        std::cerr << "[GPU] device deallocation refused: allocator device "
                  << s_device_id << ", active device " << selected_device
                  << std::endl;
        return;
      }
      const cudaError_t status = cudaFree(ptr);
      if (status != cudaSuccess) {
        std::cerr << "[GPU] device deallocation failed: "
                  << cudaGetErrorString(status) << std::endl;
        (void)cudaGetLastError();
      }
#endif
    }
  }
};

// Static member initialization
inline bool CudaAllocator::s_initialized = false;
inline int CudaAllocator::s_device_id = -1;
inline std::once_flag CudaAllocator::s_initialize_once;

#endif
