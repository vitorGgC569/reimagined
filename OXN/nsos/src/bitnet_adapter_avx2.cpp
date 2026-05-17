#include "../include/bitnet_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace nsos {

namespace {

// SIMD Gap #2 fix (2026-05-17): in-register horizontal reduction
// of __m256i.  The old implementation did _mm256_store_si256 + a
// scalar loop, which forces 32 bytes through L1 each call.  This
// function is called once per (row, out_col) in the bitnet GEMM
// inner loop, so for out_cols=512 that's 512 unnecessary stores
// per input row.
//
// The reduction here is the canonical "Agner Fog" pattern:
//   1. Split the 256-bit register into two 128-bit halves.
//   2. Add them (4 lanes).
//   3. Horizontal-add pairs into another 128-bit register (2 lanes).
//   4. Horizontal-add pairs again (1 lane = the final sum).
//   5. Extract the bottom 32-bit lane as an int.
// Total: 5 SSE instructions, all in registers, zero memory traffic.
inline int horizontal_sum_epi32(__m256i value) {
    const __m128i lo = _mm256_castsi256_si128(value);           // lanes 0..3
    const __m128i hi = _mm256_extracti128_si256(value, 1);      // lanes 4..7
    __m128i sum128 = _mm_add_epi32(lo, hi);                     // 4 lanes
    sum128 = _mm_hadd_epi32(sum128, sum128);                    // 2 lanes
    sum128 = _mm_hadd_epi32(sum128, sum128);                    // 1 lane
    return _mm_cvtsi128_si32(sum128);                           // extract bottom
}

inline float apply_output_affine(float value,
                                 int out_col,
                                 const float* magnitude,
                                 const float* bias,
                                 bool use_bias) {
    if (magnitude) {
        value *= magnitude[out_col];
    }
    if (use_bias && bias) {
        value += bias[out_col];
    }
    return value;
}

} // namespace

void gemm_158bit_i8_avx2_kernel(const Tensor& input,
                                const std::vector<int8_t>& unpacked_weights,
                                const std::vector<float>& act_scales,
                                float weight_scale,
                                Tensor& output,
                                const float* magnitude,
                                const float* bias,
                                bool use_bias) {
    const int rows = input.shape[0];
    const int cols = input.shape[1];
    const int out_cols = output.shape[1];
    const float* x_ptr = input.data();
    float* y_ptr = output.data();

    std::vector<int8_t> row_quant(static_cast<size_t>(cols), 0);
    for (int row = 0; row < rows; ++row) {
        const float act_scale =
            row < static_cast<int>(act_scales.size()) ? act_scales[row] : 1.0f;
        const float combined_scale = act_scale * weight_scale;
        const float* row_ptr = x_ptr + row * cols;

        // SIMD Gap #3 fix (2026-05-17): vectorize the row quantization.
        // The scalar version ran round + clamp + cast for each column,
        // about cols * (3 op + 1 cast) cycles per row, called once per
        // input row.  With AVX2 we process 8 floats per iteration:
        //   1. Round-to-nearest-even via _mm256_round_ps (one instr)
        //   2. Clamp to [-127, 127] via max+min on the float vector
        //   3. Convert to 32-bit ints, pack down to 16-bit, then 8-bit
        //   4. Store 8 packed int8 values
        // We also use FMA-style ops where they fit.  The clamp range is
        // [-127, 127] (not [-128, 127]) to keep the result symmetric;
        // this matches the original scalar code and the BitNet paper
        // convention for activation quantization (signed 8-bit
        // symmetric).
        const __m256 v_neg127 = _mm256_set1_ps(-127.0f);
        const __m256 v_pos127 = _mm256_set1_ps(127.0f);
        // _mm256_packs_epi32 saturates to int16 range; we need int8.
        // After two packs (32->16, 16->8) the lanes from each 256-bit
        // input are NOT in natural order: the result of packus across
        // two halves uses an interleave pattern.  For a single 8-wide
        // input we cast through int32, pack to int16, then to int8 in
        // a temporary register and extract the low 8 bytes.
        int col = 0;
        for (; col + 8 <= cols; col += 8) {
            __m256 vals = _mm256_loadu_ps(row_ptr + col);
            vals = _mm256_round_ps(vals,
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            vals = _mm256_max_ps(vals, v_neg127);
            vals = _mm256_min_ps(vals, v_pos127);
            // Convert 8 floats -> 8 int32.  Already rounded above so
            // _mm256_cvttps_epi32 (truncation) is exact.
            const __m256i i32 = _mm256_cvttps_epi32(vals);
            // Pack int32 -> int16.  _mm256_packs_epi32 produces an
            // interleaved result across the two 128-bit halves; we
            // unscramble below.  For a single 8-lane input we want
            // lanes 0..7 of the result to be the 8 int16 values.
            // permute4x64 reorders the 64-bit chunks so the four
            // valid int16s from each half end up contiguous.
            __m256i i16 = _mm256_packs_epi32(i32, i32);  // upper half = dup
            i16 = _mm256_permute4x64_epi64(i16, 0xD8);   // 0b11011000
            // Now the low 128 bits hold our 8 int16 values in order.
            const __m128i i16_lo = _mm256_castsi256_si128(i16);
            // Pack int16 -> int8 with signed saturation.  Same trick:
            // packs across two halves; we only need the low 8 bytes.
            const __m128i i8 = _mm_packs_epi16(i16_lo, i16_lo);
            // Store the low 8 bytes (one int64 worth) to row_quant.
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(row_quant.data() + col), i8);
        }
        // Scalar tail for the last <8 elements.
        for (; col < cols; ++col) {
            const float rounded = std::round(row_ptr[col]);
            const float clamped = std::max(-127.0f, std::min(127.0f, rounded));
            row_quant[static_cast<size_t>(col)] = static_cast<int8_t>(clamped);
        }

        for (int out_col = 0; out_col < out_cols; ++out_col) {
            const int8_t* weight_row =
                unpacked_weights.data() + static_cast<size_t>(out_col) * cols;
            __m256i acc_vec = _mm256_setzero_si256();
            int acc_i32 = 0;
            int col = 0;
            for (; col + 16 <= cols; col += 16) {
                _mm_prefetch(reinterpret_cast<const char*>(weight_row + col + 32), _MM_HINT_T0);
                const __m128i x8 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(row_quant.data() + col));
                const __m128i w8 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(weight_row + col));
                const __m256i x16 = _mm256_cvtepi8_epi16(x8);
                const __m256i w16 = _mm256_cvtepi8_epi16(w8);
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(x16, w16));
            }
            acc_i32 += horizontal_sum_epi32(acc_vec);
            for (; col < cols; ++col) {
                acc_i32 += static_cast<int>(row_quant[static_cast<size_t>(col)]) *
                           static_cast<int>(weight_row[col]);
            }

            y_ptr[row * out_cols + out_col] =
                apply_output_affine(static_cast<float>(acc_i32) * combined_scale,
                                    out_col,
                                    magnitude,
                                    bias,
                                    use_bias);
        }
    }
}

} // namespace nsos
