// SIMD Gap #1 fix (2026-05-17): AVX2 kernel for
// BitLinear::quantize_activations_bitnet inner loop.
//
// The original implementation (in bitlinear.cpp) does two scalar
// loops per input row M:
//   1. max_val = max(max_val, |x[i*K + j]|) over j in [0, K)
//   2. q[i*K + j] = round(x[i*K + j] * scale)  over j in [0, K)
// This is called from BitLinear::forward — once per BitLinear per
// layer per token.  For a 12-layer model with attention QKV
// (3 BitLinears) + Mamba projections (~2 BitLinears) + FFN
// (2 BitLinears) + 8 MoE experts (16 BitLinears) per MoE block ~=
// 20-40 BitLinear forwards per token.  Each is K=512 or 2048
// elements scalar — significant CPU time.
//
// This file ships an AVX2 implementation:
//   * Pass 1 (max abs) uses _mm256_max_ps with the IEEE sign mask
//     trick to compute abs without a branch, plus a tree reduce at
//     the end to extract the scalar max across 8 lanes.
//   * Pass 2 (scale + round) uses _mm256_round_ps (round-to-nearest
//     even, matching std::round in IEEE 754 mode), then writes
//     the float result back as f32 (matching the original output
//     dtype — the quantization step itself stays float-valued
//     because downstream gemm_158bit_i8 does the int8 cast at its
//     own call site).
//
// Compiled with /arch:AVX2 (MSVC) or -mavx2 (GCC/Clang) via the
// CMake property set for this file.
//
// Wired into bitlinear.cpp via a runtime check on
// NSOS_ENABLE_AVX2_KERNELS — if the build defined that macro AND
// we're on x86_64, we call this; otherwise the scalar fallback in
// bitlinear.cpp runs unchanged.

#include "../include/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace nsos {

namespace {

// Reduce 8 float lanes to a single scalar maximum.  Three steps:
//   v = [a b c d e f g h]
//   1. tmp = max(low_half, high_half)   = [max(a,e) max(b,f) max(c,g) max(d,h)]
//   2. tmp = max(tmp, shuffle(tmp,2))   = [max(.,.) max(.,.) ? ?]   (front 2 valid)
//   3. tmp = max(tmp, shuffle(tmp,1))   = [overall_max ? ? ?]
// Returns the single max as a scalar float.
inline float reduce_max_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));        // max of top half into low half
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55)); // max of remaining two scalars
    return _mm_cvtss_f32(m);
}

}  // namespace

// Compute max |x[col]| for col in [0, K) using AVX2.
// IEEE sign-bit clear via AND with NOT(sign_mask) — branch-free abs.
float bitlinear_row_max_abs_avx2(const float* row_ptr, int K) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);   // 0x80000000
    __m256 vmax = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= K; j += 8) {
        const __m256 v = _mm256_loadu_ps(row_ptr + j);
        // abs(x) = x & ~sign_mask; _mm256_andnot_ps(a, b) = (~a) & b
        const __m256 v_abs = _mm256_andnot_ps(sign_mask, v);
        vmax = _mm256_max_ps(vmax, v_abs);
    }
    float max_val = reduce_max_ps(vmax);
    // Scalar tail for K not divisible by 8.
    for (; j < K; ++j) {
        const float v = std::fabs(row_ptr[j]);
        if (v > max_val) max_val = v;
    }
    return max_val;
}

// Scale by `scale` and round-to-nearest-even (IEEE 754 default),
// writing the float result.  Output stays float to preserve the
// existing contract: downstream gemm_158bit_i8 reads x_q.data() as
// float and does its own int8 cast.  This kernel just replaces
// `q[j] = round(x[j] * scale)` with vectorized ops.
void bitlinear_row_scale_round_avx2(float* dst, const float* src,
                                     int K, float scale) {
    const __m256 vscale = _mm256_set1_ps(scale);
    int j = 0;
    for (; j + 8 <= K; j += 8) {
        const __m256 v = _mm256_loadu_ps(src + j);
        __m256 scaled = _mm256_mul_ps(v, vscale);
        scaled = _mm256_round_ps(scaled,
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_ps(dst + j, scaled);
    }
    for (; j < K; ++j) {
        dst[j] = std::round(src[j] * scale);
    }
}

// SIMD Gap #4 fix (2026-05-17): vectorize compute_weight_row_sums.
// Called only at repack/load time (not in the per-token forward),
// so the wall-time impact is small.  Worth doing because the
// implementation is trivial AND the same SIMD path appears in
// pack-acceleration tests we'll want to run consistently.
//
// Sums `cols` int8 values into a single int32.  AVX2 has no direct
// "horizontal sum of 32 int8s" intrinsic, so we widen incrementally:
//   1. Load 32 int8 values into __m256i.
//   2. Use _mm256_sad_epu8 with zero to sum 8 absolute differences
//      per 64-bit lane — but we have SIGNED int8, so we use
//      _mm256_maddubs_epi16 with a vector of 1s to sign-extend +
//      pairwise-add into int16 lanes.
//   3. Reduce int16 lanes to int32 via _mm256_madd_epi16 with 1s.
//   4. Horizontal reduce int32 across the 8 lanes.
//
// Conservatively we use _mm256_cvtepi8_epi16 + accumulator pattern
// which is more readable and equally fast.
int32_t bitlinear_row_sum_i8_avx2(const int8_t* row_ptr, int cols) {
    __m256i acc = _mm256_setzero_si256();
    int col = 0;
    for (; col + 16 <= cols; col += 16) {
        // Load 16 int8 values from the row.
        const __m128i v8 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(row_ptr + col));
        // Sign-extend to 16 int16 values in __m256i.
        const __m256i v16 = _mm256_cvtepi8_epi16(v8);
        // _mm256_madd_epi16 multiplies pairs of int16 and adds them
        // into int32 lanes.  Multiplying by a vector of 1s gives the
        // pairwise sum: lanes 0..7 of the int32 result hold (a0+a1,
        // a2+a3, ..., a14+a15).  We accumulate into acc.
        const __m256i ones = _mm256_set1_epi16(1);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(v16, ones));
    }
    // Horizontal reduce the 8 int32 lanes to a single scalar.  Reuse
    // the same pattern used by horizontal_sum_epi32 in
    // bitnet_adapter_avx2.cpp.
    const __m128i lo = _mm256_castsi256_si128(acc);
    const __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t total = _mm_cvtsi128_si32(s);
    // Scalar tail for cols not divisible by 16.
    for (; col < cols; ++col) {
        total += static_cast<int32_t>(row_ptr[col]);
    }
    return total;
}

}  // namespace nsos
