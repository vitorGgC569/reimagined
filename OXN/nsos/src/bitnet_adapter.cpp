#include "../include/bitnet_adapter.h"
#include "../include/nsos_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && defined(NSOS_ARCH_X86)
#include <intrin.h>
#endif

namespace nsos {

#if defined(NSOS_ENABLE_AVX2_KERNELS)
void gemm_158bit_i8_avx2_kernel(const Tensor& input,
                                const std::vector<int8_t>& unpacked_weights,
                                const std::vector<float>& act_scales,
                                float weight_scale,
                                Tensor& output,
                                const float* magnitude,
                                const float* bias,
                                bool use_bias);
#endif

namespace {

constexpr float kDecodeLut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
constexpr int8_t kDecodeI8Lut[4] = {-1, 0, 1, 0};

enum class BitNetCpuKernel {
    Scalar,
    AVX2,
    AVXVNNI,
    AMXInt8
};

#if defined(NSOS_ARCH_X86) && defined(__AVX2__)
inline int horizontal_sum_epi32(__m256i value) {
    alignas(32) int lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), value);
    int total = 0;
    for (int lane : lanes) {
        total += lane;
    }
    return total;
}
#endif

bool cpu_supports_avx2() {
#if defined(NSOS_ENABLE_AVX2_KERNELS) && defined(_MSC_VER) && defined(NSOS_ARCH_X86)
    int cpu_info[4] = {};
    __cpuid(cpu_info, 1);
    const bool osxsave = (cpu_info[2] & (1 << 27)) != 0;
    const bool avx = (cpu_info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return false;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 5)) != 0;
#elif defined(NSOS_ENABLE_AVX2_KERNELS) && (defined(__GNUC__) || defined(__clang__)) && defined(NSOS_ARCH_X86)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

BitNetCpuKernel select_best_kernel() {
#if defined(NSOS_ARCH_X86) && defined(__AMX_INT8__)
    return BitNetCpuKernel::AMXInt8;
#elif defined(NSOS_ARCH_X86) && (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
    return BitNetCpuKernel::AVXVNNI;
#elif defined(NSOS_ENABLE_AVX2_KERNELS)
    if (cpu_supports_avx2()) {
        return BitNetCpuKernel::AVX2;
    }
    return BitNetCpuKernel::Scalar;
#elif defined(NSOS_ARCH_X86) && defined(__AVX2__)
    return BitNetCpuKernel::AVX2;
#else
    return BitNetCpuKernel::Scalar;
#endif
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

void gemm_158bit_i8_scalar_impl(const Tensor& input,
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
            int acc_i32 = 0;
            for (int col = 0; col < cols; ++col) {
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

#if defined(NSOS_ARCH_X86) && defined(__AVX2__)
void gemm_158bit_i8_avx2_impl(const Tensor& input,
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
                apply_output_affine(static_cast<float>(acc_i32) * act_scale * weight_scale,
                                    out_col,
                                    magnitude,
                                    bias,
                                    use_bias);
        }
    }
}
#endif

#if defined(NSOS_ARCH_X86) && (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
void gemm_158bit_i8_vnni_impl(const Tensor& input,
                              const std::vector<int8_t>& unpacked_weights,
                              const std::vector<int32_t>& weight_row_sums,
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

    std::vector<uint8_t> row_quant_u8(static_cast<size_t>(cols), 0);
    for (int row = 0; row < rows; ++row) {
        const float act_scale =
            row < static_cast<int>(act_scales.size()) ? act_scales[row] : 1.0f;
        const float* row_ptr = x_ptr + row * cols;
        for (int col = 0; col < cols; ++col) {
            const float rounded = std::round(row_ptr[col]);
            const float clamped = std::max(-127.0f, std::min(127.0f, rounded));
            row_quant_u8[static_cast<size_t>(col)] =
                static_cast<uint8_t>(static_cast<int>(clamped) + 127);
        }

        for (int out_col = 0; out_col < out_cols; ++out_col) {
            const int8_t* weight_row =
                unpacked_weights.data() + static_cast<size_t>(out_col) * cols;
            __m256i acc_vec = _mm256_setzero_si256();
            int acc_i32 = 0;
            int col = 0;
            for (; col + 32 <= cols; col += 32) {
                _mm_prefetch(reinterpret_cast<const char*>(weight_row + col + 64), _MM_HINT_T0);
                const __m256i x8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row_quant_u8.data() + col));
                const __m256i w8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weight_row + col));
                acc_vec = _mm256_dpbusd_epi32(acc_vec, x8, w8);
            }
            acc_i32 += horizontal_sum_epi32(acc_vec);
            for (; col < cols; ++col) {
                acc_i32 += static_cast<int>(row_quant_u8[static_cast<size_t>(col)]) *
                           static_cast<int>(weight_row[col]);
            }
            acc_i32 -= 127 * weight_row_sums[static_cast<size_t>(out_col)];

            y_ptr[row * out_cols + out_col] =
                apply_output_affine(static_cast<float>(acc_i32) * act_scale * weight_scale,
                                    out_col,
                                    magnitude,
                                    bias,
                                    use_bias);
        }
    }
}
#endif

} // namespace

void BitNetAdapter::gemm_158bit_lut(const Tensor& input,
                                    const std::vector<uint32_t>& packed_weights,
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
    const uint8_t* packed_ptr =
        reinterpret_cast<const uint8_t*>(packed_weights.data());

    for (int row = 0; row < rows; ++row) {
        const float act_scale =
            row < static_cast<int>(act_scales.size()) ? act_scales[row] : 1.0f;
        const float* row_ptr = x_ptr + row * cols;
        for (int out_col = 0; out_col < out_cols; ++out_col) {
            float acc = 0.0f;
            const int row_weight_base = out_col * cols;
            for (int col = 0; col < cols; ++col) {
                const int flat_index = row_weight_base + col;
                const int packed_index = flat_index >> 2;
                const int packed_shift = (flat_index & 0x3) << 1;
                const uint8_t encoded = (packed_ptr[packed_index] >> packed_shift) & 0x3;
                acc += row_ptr[col] * kDecodeLut[encoded];
            }
            y_ptr[row * out_cols + out_col] =
                apply_output_affine(acc * act_scale * weight_scale,
                                    out_col,
                                    magnitude,
                                    bias,
                                    use_bias);
        }
    }
}

void BitNetAdapter::gemm_158bit_i8(const Tensor& input,
                                   const std::vector<int8_t>& unpacked_weights,
                                   const std::vector<int32_t>& weight_row_sums,
                                   const std::vector<float>& act_scales,
                                   float weight_scale,
                                   Tensor& output,
                                   const float* magnitude,
                                   const float* bias,
                                   bool use_bias) {
    switch (select_best_kernel()) {
        case BitNetCpuKernel::AMXInt8:
#if defined(NSOS_ARCH_X86) && (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
            gemm_158bit_i8_vnni_impl(input, unpacked_weights, weight_row_sums, act_scales,
                                     weight_scale, output, magnitude, bias, use_bias);
            return;
#elif defined(NSOS_ARCH_X86) && defined(__AVX2__)
            gemm_158bit_i8_avx2_impl(input, unpacked_weights, act_scales, weight_scale,
                                     output, magnitude, bias, use_bias);
            return;
#else
            break;
#endif
        case BitNetCpuKernel::AVXVNNI:
#if defined(NSOS_ARCH_X86) && (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
            gemm_158bit_i8_vnni_impl(input, unpacked_weights, weight_row_sums, act_scales,
                                     weight_scale, output, magnitude, bias, use_bias);
            return;
#else
            break;
#endif
        case BitNetCpuKernel::AVX2:
#if defined(NSOS_ENABLE_AVX2_KERNELS)
            gemm_158bit_i8_avx2_kernel(input, unpacked_weights, act_scales, weight_scale,
                                       output, magnitude, bias, use_bias);
            return;
#elif defined(NSOS_ARCH_X86) && defined(__AVX2__)
            gemm_158bit_i8_avx2_impl(input, unpacked_weights, act_scales, weight_scale,
                                     output, magnitude, bias, use_bias);
            return;
#else
            break;
#endif
        case BitNetCpuKernel::Scalar:
        default:
            gemm_158bit_i8_scalar_impl(input, unpacked_weights, act_scales, weight_scale,
                                       output, magnitude, bias, use_bias);
            return;
    }
}

void BitNetAdapter::pack_weights_microsoft_style(const float* src,
                                                 uint8_t* dst,
                                                 int rows,
                                                 int cols) {
    const int total = rows * cols;
    const int packed_size = (total + 3) / 4;
    std::fill_n(dst, packed_size, 0u);

    for (int i = 0; i < total; i += 4) {
        uint8_t packed = 0;
        for (int j = 0; j < 4 && i + j < total; ++j) {
            const float value = src[i + j];
            const uint8_t ternary =
                value < -0.25f ? 0u : (value > 0.25f ? 2u : 1u);
            packed |= static_cast<uint8_t>(ternary << (j * 2));
        }
        dst[i / 4] = packed;
    }
}

void BitNetAdapter::unpack_weights_microsoft_style_to_i8(
    const std::vector<uint32_t>& packed_weights,
    int rows,
    int cols,
    std::vector<int8_t>& dst) {
    const int total = rows * cols;
    dst.assign(static_cast<size_t>(total), 0);
    const uint8_t* packed_ptr =
        reinterpret_cast<const uint8_t*>(packed_weights.data());

    for (int index = 0; index < total; ++index) {
        const int packed_index = index >> 2;
        const int packed_shift = (index & 0x3) << 1;
        const uint8_t encoded = (packed_ptr[packed_index] >> packed_shift) & 0x3;
        dst[static_cast<size_t>(index)] = kDecodeI8Lut[encoded];
    }
}

} // namespace nsos
