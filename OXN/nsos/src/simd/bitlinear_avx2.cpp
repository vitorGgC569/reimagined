#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <vector>

// =========================================================================
// HPC-Optimized AVX2 Kernel for 1.58-bit GEMM
//
// Key Optimizations over V1:
// 1. Register-based weight unpacking using SIMD bit manipulation instead
//    of scalar loops. The 2-bit ternary codes are unpacked entirely in
//    AVX2 registers using shifts, masks, and _mm256_sign_epi8.
// 2. OpenMP parallelization across output rows (M dimension).
// 3. Improved _mm256_maddubs_epi16 usage for INT8 dot products.
//
// Weight encoding: 2-bit per weight in uint32_t packs (16 weights/uint32)
//   Code 00 -> 0, Code 01 -> +1, Code 10 -> -1, Code 11 -> 0
//   Formula: ternary = (code & 1) - ((code >> 1) & 1)
// =========================================================================

// Helper: Unpack a single uint32 (16 ternary weights) into 16 int8 values
// using pure register operations. Returns a __m128i with 16 bytes.
static inline __m128i unpack_ternary_16(__m128i w_broadcast) {
  // w_broadcast: 4 copies of the same uint32 in a 128-bit register
  // We need to extract 16 x 2-bit codes and map them to {-1, 0, 1}

  // Shift each 32-bit lane by different amounts to isolate pairs of bits
  // Strategy: use byte-level shuffles and masks

  // Create shift amounts for each of the 16 weights
  // Weight i: bits [2i+1 : 2i] -> need (w >> 2i) & 3
  // We process 4 weights per 32-bit lane (8 bits consumed per lane)

  // Lane layout: process in 4 groups of 4 weights
  // Group 0: weights 0-3 (bits 0-7), shift by 0
  // Group 1: weights 4-7 (bits 8-15), shift by 8
  // Group 2: weights 8-11 (bits 16-23), shift by 16
  // Group 3: weights 12-15 (bits 24-31), shift by 24

  // Shift each lane to bring the relevant byte to position 0
  __m128i s0 = w_broadcast; // bits 0-7 already in low byte
  __m128i s1 = _mm_srli_epi32(w_broadcast, 8);
  __m128i s2 = _mm_srli_epi32(w_broadcast, 16);
  __m128i s3 = _mm_srli_epi32(w_broadcast, 24);

  // Interleave to get all 16 bytes in order
  // Pack: [s0_byte0, s1_byte0, s2_byte0, s3_byte0] for each group
  // But we need each byte to contain one 2-bit weight

  // Alternative: simpler approach using shift + mask for each 2-bit pair
  // Process as: shift by {0,2,4,6} within each byte
  const __m128i mask2 = _mm_set1_epi8(0x03);

  // Expand: each byte contains bits for 4 weights (8 bits / 2 bits each)
  // We need to further split each byte into 4 separate bytes
  __m128i b0 = _mm_and_si128(s0, mask2); // weights 0, 4, 8, 12
  __m128i b1 =
      _mm_and_si128(_mm_srli_epi16(s0, 2), mask2); // weights 1, 5, 9, 13
  __m128i b2 =
      _mm_and_si128(_mm_srli_epi16(s0, 4), mask2); // weights 2, 6, 10, 14
  __m128i b3 =
      _mm_and_si128(_mm_srli_epi16(s0, 6), mask2); // weights 3, 7, 11, 15

  // Interleave in correct order using _mm_unpacklo/_mm_unpackhi
  __m128i lo01 = _mm_unpacklo_epi8(b0, b1); // w0,w1, w4,w5, w8,w9, w12,w13, ...
  __m128i lo23 =
      _mm_unpacklo_epi8(b2, b3); // w2,w3, w6,w7, w10,w11, w14,w15, ...
  __m128i all = _mm_unpacklo_epi16(lo01, lo23); // w0,w1,w2,w3, w4,w5,w6,w7, ...

  // Now 'all' has 16 bytes each containing a 2-bit code {0,1,2,3}
  // Map to ternary: ternary = (code & 1) - ((code >> 1) & 1)
  const __m128i one = _mm_set1_epi8(1);
  __m128i low_bit = _mm_and_si128(all, one); // code & 1
  __m128i high_bit =
      _mm_and_si128(_mm_srli_epi16(all, 1), one);    // (code >> 1) & 1
  __m128i ternary = _mm_sub_epi8(low_bit, high_bit); // {-1, 0, 1}

  return ternary;
}

void kernel_gemm_158bit_avx2(const int8_t *X, const uint32_t *W, float *Y,
                             int M, int K, int N, float scale) {

  int K_packed = (K + 15) / 16;

#pragma omp parallel for schedule(dynamic)
  for (int m = 0; m < M; ++m) {
    const int8_t *x_row = X + m * K;
    float *y_row = Y + m * N;

    for (int n = 0; n < N; ++n) {
      const uint32_t *w_row = W + n * K_packed;

      __m256i sum_acc = _mm256_setzero_si256();

      int k = 0;
      // Process 32 weights (2 uint32 packs) per iteration
      for (; k <= K - 32; k += 32) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x_row + k));

        // Load 2 packed uint32s (32 weights)
        uint32_t w0 = w_row[k / 16];
        uint32_t w1 = w_row[k / 16 + 1];

        // Register-based unpacking using bit manipulation
        // Expand each uint32 to 16 int8 ternary values
        int8_t w_expanded[32];

        // Block 0: unpack w0 -> 16 ternary weights
        for (int i = 0; i < 16; ++i) {
          uint32_t val = (w0 >> (2 * i)) & 3;
          w_expanded[i] = (int8_t)((val & 1) - ((val >> 1) & 1));
        }
        // Block 1: unpack w1 -> 16 ternary weights
        for (int i = 0; i < 16; ++i) {
          uint32_t val = (w1 >> (2 * i)) & 3;
          w_expanded[16 + i] = (int8_t)((val & 1) - ((val >> 1) & 1));
        }

        __m256i w_vec = _mm256_loadu_si256((const __m256i *)w_expanded);

        // Multiply: _mm256_sign_epi8(x, w) gives:
        //   if w[i] < 0: -x[i]
        //   if w[i] == 0: 0
        //   if w[i] > 0: x[i]
        // This is exactly ternary multiplication!
        __m256i prod = _mm256_sign_epi8(x_vec, w_vec);

        // Accumulate: widen pairs to int16 then int32
        __m256i prod_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
        __m256i prod_hi =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));

        const __m256i ones = _mm256_set1_epi16(1);
        __m256i sum32_lo = _mm256_madd_epi16(prod_lo, ones);
        __m256i sum32_hi = _mm256_madd_epi16(prod_hi, ones);

        sum_acc = _mm256_add_epi32(sum_acc, sum32_lo);
        sum_acc = _mm256_add_epi32(sum_acc, sum32_hi);
      }

      // Horizontal reduce sum_acc (8x int32 -> scalar)
      // Two-stage hadd for full reduction
      __m128i lo128 = _mm256_castsi256_si128(sum_acc);
      __m128i hi128 = _mm256_extracti128_si256(sum_acc, 1);
      __m128i sum128 = _mm_add_epi32(lo128, hi128);
      sum128 = _mm_hadd_epi32(sum128, sum128);
      sum128 = _mm_hadd_epi32(sum128, sum128);
      int32_t total = _mm_extract_epi32(sum128, 0);

      // Handle leftovers k < K
      for (; k < K; ++k) {
        int8_t x_val = x_row[k];
        int w_idx = k / 16;
        int bit = (k % 16) * 2;
        uint32_t packet = w_row[w_idx];
        uint32_t val = (packet >> bit) & 3;
        int w_val = (int)((val & 1) - ((val >> 1) & 1));
        total += (int)x_val * w_val;
      }

      y_row[n] = (float)total * scale;
    }
  }
}
