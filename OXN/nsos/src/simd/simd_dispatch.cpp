#include "simd_dispatch.h"
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// Forward declare fallback
void matmul_158bit_scalar(float *res, const float *x, const float *w, int M,
                          int K, int N);

#ifdef ENABLE_AVX2
void matmul_158bit_avx2(float *res, const float *x, const float *w, int M,
                        int K, int N);
#endif

bool check_avx2_support() {
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 for AVX2
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

void matmul_158bit_dispatch(float *res, const float *x, const float *w, int M,
                            int K, int N) {
#ifdef ENABLE_AVX2
    static bool has_avx2 = check_avx2_support();
    if (has_avx2) {
        matmul_158bit_avx2(res, x, w, M, K, N);
        return;
    }
#endif
    // Fallback
    matmul_158bit_scalar(res, x, w, M, K, N);
}
