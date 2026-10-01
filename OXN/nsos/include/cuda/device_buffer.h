#pragma once

#ifdef USE_CUDA

#include "../gpu_backend.h"

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
      : pointer_(other.pointer_),
        deferred_release_(other.deferred_release_),
        capacity_(other.capacity_),
        device_id_(other.device_id_) {
    other.pointer_ = nullptr;
    other.deferred_release_ = nullptr;
    other.capacity_ = 0;
    other.device_id_ = -1;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      // Swap ownership so a release failure of this object's previous
      // allocation cannot be hidden by overwriting its bookkeeping. The
      // moved-from object remains valid and will release that allocation.
      std::swap(pointer_, other.pointer_);
      std::swap(deferred_release_, other.deferred_release_);
      std::swap(capacity_, other.capacity_);
      std::swap(device_id_, other.device_id_);
    }
    return *this;
  }

  // Grow geometrically and preserve the old allocation if the replacement
  // cannot be allocated. Returns nullptr on failure without leaking either
  // allocation.
  T* ensure(std::size_t requested) noexcept {
    int selected_device = -1;
    try {
      if (!gpu::select_preferred_device(
              &selected_device, nullptr)) {
        return nullptr;
      }
    } catch (...) {
      return nullptr;
    }
    if (deferred_release_ != nullptr ||
        (pointer_ != nullptr && device_id_ != selected_device)) {
      if (!release() || pointer_ != nullptr ||
          deferred_release_ != nullptr) {
        return nullptr;
      }
      try {
        if (!gpu::select_preferred_device(
                &selected_device, nullptr)) {
          return nullptr;
        }
      } catch (...) {
        return nullptr;
      }
    }
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

    if (pointer_ != nullptr) {
      // Commit only after the old allocation is known to be released. If a
      // pending launch/runtime error prevents that release, preserve the old
      // buffer and discard the uncommitted replacement.
      const cudaError_t prior_status = cudaGetLastError();
      const cudaError_t free_status =
          prior_status == cudaSuccess
              ? cudaFree(pointer_)
              : prior_status;
      if (free_status != cudaSuccess) {
        (void)cudaGetLastError();
        const cudaError_t replacement_status =
            cudaFree(replacement);
        if (replacement_status != cudaSuccess) {
          (void)cudaGetLastError();
          deferred_release_ = replacement;
        }
        return nullptr;
      }
    }
    pointer_ = replacement;
    capacity_ = next_capacity;
    device_id_ = selected_device;
    return pointer_;
  }

  bool release() noexcept {
    if (pointer_ == nullptr && deferred_release_ == nullptr) {
      device_id_ = -1;
      capacity_ = 0;
      return true;
    }

    int previous_device = -1;
    bool restore_device = false;
    const cudaError_t query_status =
        cudaGetDevice(&previous_device);
    if (query_status != cudaSuccess) {
      (void)cudaGetLastError();
    }
    if (device_id_ >= 0 && previous_device != device_id_) {
      if (cudaSetDevice(device_id_) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
      }
      restore_device = previous_device >= 0;
    }

    bool released = true;
    const cudaError_t prior_status = cudaGetLastError();
    if (prior_status != cudaSuccess) {
      released = false;
    } else if (pointer_ != nullptr) {
      const cudaError_t free_status = cudaFree(pointer_);
      if (free_status != cudaSuccess) {
        (void)cudaGetLastError();
        released = false;
      } else {
        pointer_ = nullptr;
        capacity_ = 0;
      }
    }
    if (released && deferred_release_ != nullptr) {
      const cudaError_t deferred_status =
          cudaFree(deferred_release_);
      if (deferred_status != cudaSuccess) {
        (void)cudaGetLastError();
        released = false;
      } else {
        deferred_release_ = nullptr;
      }
    }
    if (pointer_ == nullptr && deferred_release_ == nullptr) {
      device_id_ = -1;
    }

    if (restore_device) {
      const cudaError_t restore_status =
          cudaSetDevice(previous_device);
      if (restore_status != cudaSuccess) {
        (void)cudaGetLastError();
        released = false;
      }
    }
    return released &&
           pointer_ == nullptr && deferred_release_ == nullptr;
  }

  // Used only when an asynchronous completion boundary itself failed. The
  // device may still reference the allocation, so relinquish local ownership
  // without asking the runtime to free an ambiguously in-flight pointer.
  void abandon() noexcept {
    pointer_ = nullptr;
    deferred_release_ = nullptr;
    capacity_ = 0;
    device_id_ = -1;
  }

  T* get() noexcept { return pointer_; }
  const T* get() const noexcept { return pointer_; }
  std::size_t capacity() const noexcept { return capacity_; }
  int device_id() const noexcept { return device_id_; }

 private:
  T* pointer_ = nullptr;
  T* deferred_release_ = nullptr;
  std::size_t capacity_ = 0;
  int device_id_ = -1;
};

}  // namespace nsos::cuda_detail

#endif  // USE_CUDA
