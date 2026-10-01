#include "../include/nsos_math.h"
#include "../include/nsos_simd.h"
#include "../include/nsos_arena.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace nsos {

#ifdef NSOS_ARCH_X86
    const int MR = 6;
    const int NR = 16;
#else
    const int MR = 8;
    const int NR = 8;
#endif

const int KC = 256; 
const int NC = 1024; 

// Pack A
static void pack_A(int M, int K, const float* A, int lda, float* buffer) {
    for (int i = 0; i < M; i += MR) {
        int m_eff = std::min(MR, M - i);
        const float* a_ptr = A + i * lda;
        for (int k = 0; k < K; ++k) {
            for (int r = 0; r < m_eff; ++r) *buffer++ = a_ptr[r * lda + k];
            for (int r = m_eff; r < MR; ++r) *buffer++ = 0.0f;
        }
    }
}

// Pack B (Standard)
static void pack_B(int K, int N, const float* B, int ldb, float* buffer) {
    for (int j = 0; j < N; j += NR) {
        int n_eff = std::min(NR, N - j);
        const float* b_ptr = B + j;
        for (int k = 0; k < K; ++k) {
            for (int c = 0; c < n_eff; ++c) *buffer++ = b_ptr[k * ldb + c];
            for (int c = n_eff; c < NR; ++c) *buffer++ = 0.0f;
        }
    }
}

// Micro-Kernel
static void micro_kernel(int K, const float* A, const float* B, float* C, int ldc, float alpha) {
    #ifdef NSOS_ARCH_X86
    __m256 c[6][2];
    for(int i=0; i<6; ++i) { c[i][0] = _mm256_setzero_ps(); c[i][1] = _mm256_setzero_ps(); }
    
    for(int k=0; k<K; ++k) {
        __m256 b0 = _mm256_loadu_ps(B + k*16);
        __m256 b1 = _mm256_loadu_ps(B + k*16 + 8);
        for(int i=0; i<6; ++i) {
            __m256 a = _mm256_set1_ps(A[k*6 + i]);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }
    for(int i=0; i<6; ++i) {
        float* ptr = C + i * ldc;
        __m256 v_alpha = _mm256_set1_ps(alpha);
        _mm256_storeu_ps(ptr, _mm256_add_ps(_mm256_loadu_ps(ptr), _mm256_mul_ps(c[i][0], v_alpha)));
        _mm256_storeu_ps(ptr+8, _mm256_add_ps(_mm256_loadu_ps(ptr+8), _mm256_mul_ps(c[i][1], v_alpha)));
    }
    #else
    for(int k=0; k<K; ++k) {
        for(int i=0; i<MR; ++i) {
            float a = A[k*MR + i];
            for(int j=0; j<NR; ++j) C[i*ldc + j] += alpha * a * B[k*NR + j];
        }
    }
    #endif
}

// ------------------------------------------------------------------------
// Pre-Packed API
// ------------------------------------------------------------------------

size_t MathOps::get_packed_B_size(int N, int K) {
    // Pad N to NR
    int N_padded = (N + NR - 1) / NR * NR;
    return (size_t)N_padded * K * sizeof(float);
}

void MathOps::pack_B_matrix(int N, int K, const float* B, int ldb, float* buffer) {
    // Pack entire B matrix: Layout [N/NR][K][NR] (Block-major)
    // BLIS layout: Pack strips of width NR and height K
    #pragma omp parallel for
    for (int j = 0; j < N; j += NR) {
        int n_eff = std::min(NR, N - j);
        float* dst = buffer + (size_t)j * K;
        const float* src = B + j;
        
        for (int k = 0; k < K; ++k) {
            for (int c = 0; c < n_eff; ++c) *dst++ = src[k * ldb + c];
            for (int c = n_eff; c < NR; ++c) *dst++ = 0.0f;
        }
    }
}

void MathOps::gemm_prepacked(int M, int N, int K, float alpha, const float* A, int lda, const float* B_packed, float beta, float* C, int ldc) {
    // MC aligned to MR so every m-block tiles cleanly into MR micro-rows and a
    // single GLOBAL pack of A is addressable by row (packA + (i+ir)*K).
    const int MC = (256 / MR) * MR;

    // Scale C.  BLAS contract: when beta == 0, C is NOT read (it may hold
    // uninitialized garbage / NaN that 0*x would propagate) -- SET it to 0
    // instead of multiplying.  beta == 1 leaves C untouched (pure accumulate).
    if (beta == 0.0f) {
        #pragma omp parallel for
        for(int i=0; i<M; ++i) { for(int j=0; j<N; ++j) C[(size_t)i*ldc + j] = 0.0f; }
    } else if (beta != 1.0f) {
        #pragma omp parallel for
        for(int i=0; i<M; ++i) { for(int j=0; j<N; ++j) C[(size_t)i*ldc + j] *= beta; }
    }

    // Pack ALL of A ONCE.  The previous loop re-packed each m-block once PER
    // n-panel (n_panels times redundant) -- on a 1024^3 gemm that is ~60x more
    // packing traffic than needed.  pack_A zero-pads each MR-strip, so the
    // buffer is roundup(M,MR)*K and the MR over-read of the final strip lands in
    // that padding (in-bounds).  thread_local STORAGE is reused across calls and
    // owned by the calling thread; the worker threads read it via the shared raw
    // pointer `packA` (a plain local, NOT the thread_local variable -- accessing
    // the thread_local symbol inside the parallel region would resolve to each
    // worker's own empty copy).
    thread_local std::vector<float> packA_storage;
    const int M_padded = ((M + MR - 1) / MR) * MR;
    packA_storage.resize(static_cast<size_t>(M_padded) * K);
    float* packA = packA_storage.data();
    pack_A(M, K, A, lda, packA);

    const int n_panels = (N + NR - 1) / NR;
    const int m_blocks = (M + MC - 1) / MC;
    #pragma omp parallel for
    for(int tile=0; tile<n_panels*m_blocks; ++tile) {
        const int panel = tile / m_blocks;
        const int block = tile - panel * m_blocks;
        const int j = panel * NR; // Panel of B (NR columns)
        const int i = block * MC;
        const int mc_eff = std::min(MC, M-i);
        const int n_eff = std::min(NR, N-j);
        const float* b_panel = B_packed + (size_t)j * K;

        for(int ir=0; ir<mc_eff; ir+=MR) {
            const int mr_eff = std::min(MR, mc_eff-ir);
            const float* a_micro = packA + (size_t)(i+ir) * K;  // global MR-strip
            float* c_ptr = C + (size_t)(i+ir)*ldc + j;

            if (mr_eff == MR && n_eff == NR) {
                micro_kernel(K, a_micro, b_panel, c_ptr, ldc, alpha);
            } else {
                float tile_values[MR*NR];
                std::memset(tile_values, 0, sizeof(tile_values));
                micro_kernel(K, a_micro, b_panel, tile_values, NR, alpha);
                for(int r=0; r<mr_eff; ++r)
                    for(int c=0; c<n_eff; ++c)
                        c_ptr[(size_t)r*ldc+c] += tile_values[r*NR+c];
            }
        }
    }
}

// Fallback GEMM calls prepacked logic internally if not packed
void MathOps::gemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    // Pack B once into a thread_local buffer, reused across calls and owned by
    // the calling thread (pack_B_matrix's omp workers and gemm_prepacked read it
    // via the shared raw pointer).  Replaces the never-rewound arena alloc, which
    // grew unboundedly across gemm calls.
    thread_local std::vector<float> b_buf;
    b_buf.resize(get_packed_B_size(N, K) / sizeof(float));
    pack_B_matrix(N, K, B, ldb, b_buf.data());
    gemm_prepacked(M, N, K, alpha, A, lda, b_buf.data(), beta, C, ldc);
}

// Vector Ops remain same
void MathOps::vec_add(int n, const float* a, const float* b, float* y) {
    // Step by the actual packet width (8 AVX / 4 NEON / 1 generic); the fixed
    // i+=8 left elements uncomputed on non-AVX builds (garbage in the output).
    constexpr int W = SimdPacket<float>::width;
    int i = 0;
    for (; i <= n - W; i += W) {
        SimdPacket<float> va = SimdPacket<float>::load(a + i);
        SimdPacket<float> vb = SimdPacket<float>::load(b + i);
        (va + vb).store(y + i);
    }
    for (; i < n; ++i) y[i] = a[i] + b[i];
}
void MathOps::vec_mul(int n, const float* a, const float* b, float* y) {
    constexpr int W = SimdPacket<float>::width;
    int i = 0;
    for (; i <= n - W; i += W) {
        SimdPacket<float> va = SimdPacket<float>::load(a + i);
        SimdPacket<float> vb = SimdPacket<float>::load(b + i);
        (va * vb).store(y + i);
    }
    for (; i < n; ++i) y[i] = a[i] * b[i];
}

} // namespace nsos
