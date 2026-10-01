#pragma once

#ifdef USE_CUDA

#include "../gpu_backend.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace nsos::cuda_detail {

// Process/thread workspaces use pinned host memory for asynchronous copies.
// This owner preserves the old allocation when growth or release fails and
// never overwrites ownership bookkeeping with an ambiguous driver result.
template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  ~PinnedHostBuffer() noexcept { release(); }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
      : pointer_(other.pointer_),
        deferred_release_(other.deferred_release_),
        capacity_(other.capacity_) {
    other.pointer_ = nullptr;
    other.deferred_release_ = nullptr;
    other.capacity_ = 0;
  }

  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
      std::swap(pointer_, other.pointer_);
      std::swap(deferred_release_, other.deferred_release_);
      std::swap(capacity_, other.capacity_);
    }
    return *this;
  }

  T* ensure(std::size_t requested) noexcept {
    if (deferred_release_ != nullptr) {
      const cudaError_t deferred_status =
          cudaFreeHost(deferred_release_);
      if (deferred_status != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
      }
      deferred_release_ = nullptr;
    }
    if (requested == 0) {
      return pointer_;
    }
    if (pointer_ != nullptr && requested <= capacity_) {
      return pointer_;
    }
    if (requested >
        std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return nullptr;
    }

    std::size_t next_capacity = requested;
    if (capacity_ != 0 &&
        capacity_ <=
            std::numeric_limits<std::size_t>::max() / 2) {
      next_capacity =
          (std::max)(requested, capacity_ * 2);
    }
    if (next_capacity >
        std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      next_capacity = requested;
    }

    T* replacement = nullptr;
    const cudaError_t allocation_status = cudaMallocHost(
        reinterpret_cast<void**>(&replacement),
        next_capacity * sizeof(T));
    if (allocation_status != cudaSuccess) {
      (void)cudaGetLastError();
      return nullptr;
    }

    if (pointer_ != nullptr) {
      const cudaError_t release_status =
          cudaFreeHost(pointer_);
      if (release_status != cudaSuccess) {
        (void)cudaGetLastError();
        const cudaError_t replacement_status =
            cudaFreeHost(replacement);
        if (replacement_status != cudaSuccess) {
          (void)cudaGetLastError();
          deferred_release_ = replacement;
        }
        return nullptr;
      }
    }
    pointer_ = replacement;
    capacity_ = next_capacity;
    return pointer_;
  }

  void release() noexcept {
    if (pointer_ != nullptr) {
      const cudaError_t status = cudaFreeHost(pointer_);
      if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return;
      }
      pointer_ = nullptr;
      capacity_ = 0;
    }
    if (deferred_release_ != nullptr) {
      const cudaError_t status =
          cudaFreeHost(deferred_release_);
      if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return;
      }
      deferred_release_ = nullptr;
    }
  }

  // Used only after an asynchronous completion boundary itself failed. It is
  // safer to retain the allocation for process teardown than to let a member
  // destructor free memory the device may still reference.
  void abandon() noexcept {
    pointer_ = nullptr;
    deferred_release_ = nullptr;
    capacity_ = 0;
  }

  T* get() noexcept { return pointer_; }
  const T* get() const noexcept { return pointer_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  T* pointer_ = nullptr;
  T* deferred_release_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace nsos::cuda_detail

#endif  // USE_CUDA
