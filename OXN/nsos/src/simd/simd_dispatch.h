#ifndef SIMD_DISPATCH_H
#define SIMD_DISPATCH_H

#include <vector>

// Forward declare Tensor for pointer access if needed, or just pass raw pointers
// Passing raw pointers for maximum speed and decoupling
// Output: res [M, N]
// Input: x [M, K]
// Weight: w [K, N] (Quantized -1, 0, 1)
// M, K, N dimensions
void matmul_158bit_dispatch(float* res, const float* x, const float* w, int M, int K, int N);

// Individual kernels (exposed for testing if needed)
void matmul_158bit_scalar(float* res, const float* x, const float* w, int M, int K, int N);
void matmul_158bit_avx2(float* res, const float* x, const float* w, int M, int K, int N);

#endif
