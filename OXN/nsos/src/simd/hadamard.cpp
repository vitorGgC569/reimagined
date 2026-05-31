#include "simd_dispatch.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// =========================================================================
// HPC-Optimized Fast Walsh-Hadamard Transform (FWHT)
//
// Complexity: O(N log N) additions/subtractions. No multiplications.
// Works in-place on a float array of size N (must be power of 2).
//
// Optimization: When the butterfly stride h >= 8, we use AVX2 SIMD
// to process 8 butterfly pairs simultaneously. For h < 8, we fall
// back to scalar code (the number of operations is tiny anyway).
// =========================================================================

void fwht_cpu(float *a, int n) {
  if (n == 1)
    return;

  for (int h = 1; h < n; h <<= 1) {
    int step = h << 1;

#ifdef __AVX2__
    if (h >= 8) {
      // AVX2 vectorized butterfly: process 8 floats at once
      // For each pair block (i, i+step), process j = i..i+h-1 in 8-wide chunks
      for (int i = 0; i < n; i += step) {
        int j = 0;
        for (; j <= h - 8; j += 8) {
          __m256 vx = _mm256_loadu_ps(a + i + j);
          __m256 vy = _mm256_loadu_ps(a + i + j + h);
          __m256 vplus = _mm256_add_ps(vx, vy);
          __m256 vminus = _mm256_sub_ps(vx, vy);
          _mm256_storeu_ps(a + i + j, vplus);
          _mm256_storeu_ps(a + i + j + h, vminus);
        }
        // Scalar remainder (if h is not a multiple of 8)
        for (; j < h; ++j) {
          float x = a[i + j];
          float y = a[i + j + h];
          a[i + j] = x + y;
          a[i + j + h] = x - y;
        }
      }
    } else
#endif
    {
      // Scalar butterfly for small strides (h < 8) or no AVX2
      for (int i = 0; i < n; i += step) {
        for (int j = i; j < i + h; ++j) {
          float x = a[j];
          float y = a[j + h];
          a[j] = x + y;
          a[j + h] = x - y;
        }
      }
    }
  }

  // Normalization handled by caller (BitLinear scales by 1/sqrt(N)).
}

// In-place Hadamard transform wrapper
void hadamard_transform(float *data, int batch, int dim) {
  // The Fast Walsh-Hadamard Transform is only defined for power-of-two dims.
  // Silently returning the input unchanged (the previous behavior) corrupted
  // the math while looking like success; fail loudly instead.
  if (dim <= 0 || (dim & (dim - 1)) != 0) {
    throw std::invalid_argument(
        "hadamard_transform requires dim to be a positive power of 2");
  }

  // Parallelize across batch dimension with OpenMP
#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    fwht_cpu(data + b * dim, dim);
  }
}
