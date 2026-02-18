#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#ifdef _WIN32
#include <malloc.h> // For _aligned_malloc, _aligned_free
#endif

template <class T,
          std::size_t Alignment = 64> // 64 bytes for cache-line/AVX-512
struct AlignedAllocator {
  using value_type = T;

  // Default constructor
  AlignedAllocator() noexcept {}

  // Copy constructor
  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

  // Allocation safe for POSIX (Linux) and Windows
  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(n * sizeof(T), Alignment);
    if (!ptr)
      throw std::bad_alloc();
#else
    // posix_memalign guarantees the address is a multiple of Alignment
    if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
#endif
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
  }

  bool operator==(const AlignedAllocator &) const noexcept { return true; }
  bool operator!=(const AlignedAllocator &) const noexcept { return false; }
};

#endif
