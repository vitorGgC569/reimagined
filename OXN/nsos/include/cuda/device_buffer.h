#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace nsos::cuda_detail {

// Thread-local and process-local CUDA scratch must still have deterministic
// ownership. Relying on CUDA context teardown hides real leaks from
// Compute Sanitizer and makes repeated runtime initialization unsafe.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() noexcept { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(other.pointer_), capacity_(other.capacity_) {
    other.pointer_ = nullptr;
    other.capacity_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      pointer_ = other.pointer_;
      capacity_ = other.capacity_;
      other.pointer_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  // Grow geometrically and preserve the old allocation if the replacement
  // cannot be allocated. Returns nullptr on failure without leaking either
  // allocation.
  T* ensure(std::size_t requested) noexcept {
    if (requested == 0) {
      return pointer_;
    }
    if (pointer_ != nullptr && requested <= capacity_) {
      return pointer_;
    }
    if (requested > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return nullptr;
    }

    std::size_t next_capacity = requested;
    if (capacity_ != 0 &&
        capacity_ <= std::numeric_limits<std::size_t>::max() / 2) {
      next_capacity = (std::max)(requested, capacity_ * 2);
    }
    if (next_capacity >
        std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      next_capacity = requested;
    }

    T* replacement = nullptr;
    const cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&replacement), next_capacity * sizeof(T));
    if (status != cudaSuccess) {
      (void)cudaGetLastError();
      return nullptr;
    }

    T* previous = pointer_;
    pointer_ = replacement;
    capacity_ = next_capacity;
    if (previous != nullptr) {
      const cudaError_t free_status = cudaFree(previous);
      if (free_status != cudaSuccess) {
        (void)cudaGetLastError();
      }
    }
    return pointer_;
  }

  void release() noexcept {
    if (pointer_ != nullptr) {
      const cudaError_t status = cudaFree(pointer_);
      if (status != cudaSuccess) {
        (void)cudaGetLastError();
      }
      pointer_ = nullptr;
      capacity_ = 0;
    }
  }

  T* get() noexcept { return pointer_; }
  const T* get() const noexcept { return pointer_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  T* pointer_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace nsos::cuda_detail

#endif  // USE_CUDA
