#include "simd_dispatch.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// Check for AVX2 support
#if defined(__AVX2__)
    #include <immintrin.h>
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif

// KernelOpen Math Primitives
#include "../../include/cuda/bitnet_math.cuh"

void matmul_158bit_scalar(float* res, const float* x, const float* w, int M, int K, int N) {
#if HAS_AVX2
    // =========================================================================
    // AVX2 OPTIMIZED KERNEL (Real Hardware Acceleration)
    // =========================================================================
    // Constraints: This kernel assumes best performance when N is multiple of 8.
    // Handles remainders via scalar fallback.

    for (int i = 0; i < M; ++i) {
        int j = 0;
        // Process 8 output elements at a time (AVX2 register is 256-bit = 8 floats)
        for (; j <= N - 8; j += 8) {
            __m256 sum_vec = _mm256_setzero_ps();

            for (int k = 0; k < K; ++k) {
                // Broadcast X[i, k] to all 8 positions
                __m256 x_val = _mm256_set1_ps(x[i * K + k]);

                // Load 8 weights W[k, j...j+7]
                __m256 w_vals = _mm256_loadu_ps(&w[k * N + j]);

                // Fused Multiply-Add: sum += x * w
                // In standard float32, this is fast.
                // For 1.58-bit strictly, W is {-1, 0, 1}.
                // We could optimize further by avoiding mul and using blend/add/sub,
                // but FMADD on modern CPUs is often as fast as logic ops and handles the 0 case implicitly.
                // Given the input is float-quantized-to-ternary, FMADD is the robust path.
                sum_vec = _mm256_fmadd_ps(x_val, w_vals, sum_vec);
            }

            // Accumulate into Result (Load, Add, Store)
            __m256 current_res = _mm256_loadu_ps(&res[i * N + j]);
            current_res = _mm256_add_ps(current_res, sum_vec);
            _mm256_storeu_ps(&res[i * N + j], current_res);
        }

        // Scalar Cleanup for remaining columns
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += x[i * K + k] * w[k * N + j];
            }
            res[i * N + j] += sum;
        }
    }

#else
    // =========================================================================
    // SCALAR FALLBACK (Legacy / Non-AVX)
    // =========================================================================
    // Using branchless logic for 1.58-bit where possible

    const int TILE = 64;
    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int k0 = 0; k0 < K; k0 += TILE) {
            for (int j0 = 0; j0 < N; j0 += TILE) {
                int i_max = std::min(i0 + TILE, M);
                int k_max = std::min(k0 + TILE, K);
                int j_max = std::min(j0 + TILE, N);

                for (int i = i0; i < i_max; ++i) {
                    for (int k = k0; k < k_max; ++k) {
                        float x_val = x[i * K + k];
                        for (int j = j0; j < j_max; ++j) {
                            float w_val = w[k * N + j];
                            // Branchy opt for sparse ternary
                            if (w_val > 0.5f) res[i * N + j] += x_val;
                            else if (w_val < -0.5f) res[i * N + j] -= x_val;
                        }
                    }
                }
            }
        }
    }
#endif
}
