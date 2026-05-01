#pragma once
#include <vector>
#include <cstddef>

namespace nsos {

class MathOps {
public:
    static void gemm(
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc
    );
    
    // SOTA: Pre-Packed GEMM API
    // Pack B once, reuse many times.
    // buffer size needed: get_packed_B_size(N, K)
    static size_t get_packed_B_size(int N, int K);
    static void pack_B_matrix(int N, int K, const float* B, int ldb, float* buffer);
    
    static void gemm_prepacked(
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B_packed, // Pre-packed buffer
        float beta,
        float* C, int ldc
    );

    static void vec_add(int n, const float* a, const float* b, float* y);
    static void vec_mul(int n, const float* a, const float* b, float* y);
};

} // namespace nsos
