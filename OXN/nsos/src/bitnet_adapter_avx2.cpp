#include "../include/bitnet_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace nsos {

namespace {

inline int horizontal_sum_epi32(__m256i value) {
    alignas(32) int lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), value);
    int total = 0;
    for (int lane : lanes) {
        total += lane;
    }
    return total;
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
        for (int col = 0; col < cols; ++col) {
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
