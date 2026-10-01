#include "../include/bitnet_adapter.h"
#include "../include/nsos_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && defined(NSOS_ARCH_X86)
#include <intrin.h>
#endif

#if defined(NSOS_ARCH_X86) && defined(__AVX2__)
#include <immintrin.h>
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

// ── Cherry-pick #1 (bitmamba.cpp UNPACK_LUT + AVX2 path) ───────────────────
// SIMD implementation of the LUT GEMM, ported from external_refs/repos/
// bitmamba.cpp/src/kernels.cpp:220-240.  Three combined optimizations:
//
//   1. Use UNPACK_LUT_4WAY (1 KB constexpr table in bitnet_adapter.h) to
//      unpack 4 ternary weights from a single packed byte via a single
//      uint32_t load — no shifts, no masks, no branches per element.
//
//   2. Process 32 weights per inner iteration via __m256i SIMD register.
//      Eight UNPACK_LUT_4WAY[byte] lookups fill an aligned int8[32] buffer
//      that loads directly into a YMM register.
//
//   3. Use _mm256_sign_epi8 for the ternary multiplication: this is a
//      single-cycle instruction that does exactly what we want — when
//      the weight is +1 the activation passes through, when -1 the
//      activation is negated, when 0 the lane is zeroed.  Followed by
//      _mm256_madd_epi16 + _mm256_add_epi32 for horizontal accumulation.
//
// IMPORTANT: input must already be quantized to int8 in the caller's
// scaling space.  This function works in int8 throughout the inner loop
// and rescales to float at the end via act_scale * weight_scale.
//
// Currently the input Tensor is float (matching the existing scalar
// path's contract).  For maximum SIMD benefit we'd want a parallel
// path that takes int8 input directly.  For Phase 1 we mirror the
// existing scalar signature and quantize-on-the-fly per row.
//
// Falls back to the scalar implementation when AVX2 is not available
// or cols is too small to benefit from vectorization.
#if defined(NSOS_ARCH_X86) && defined(__AVX2__)
static void gemm_158bit_lut_simd_impl(const Tensor& input,
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

    // 4 weights packed per byte → cols/4 bytes per weight row.
    const int packed_stride = (cols + 3) / 4;

    const __m256i ones_16 = _mm256_set1_epi16(1);

    // Per-row x_quant scratch buffer.  +32 tail pad to allow safe SIMD
    // read past cols in the loop tail.
    std::vector<int8_t> x_quant(static_cast<size_t>(cols + 32), 0);

    for (int row = 0; row < rows; ++row) {
        const float act_scale =
            row < static_cast<int>(act_scales.size()) ? act_scales[row] : 1.0f;
        const float* row_ptr = x_ptr + row * cols;

        // Quantize this row to int8 once, reuse across all out_cols.
        // The existing scalar path computes x * kDecodeLut[e] in float
        // (no input quant) — but we need int8 to use _mm256_sign_epi8.
        // To preserve numerical equivalence, we mimic an "implicit" quant
        // where the activation scale is rolled into the final dequant.
        //
        // Strategy: find max-abs of the row, scale to fill int8 range,
        // then divide by the same factor when accumulating.
        float row_max_abs = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float abs_v = std::fabs(row_ptr[c]);
            if (abs_v > row_max_abs) row_max_abs = abs_v;
        }
        const float row_quant_scale = (row_max_abs > 1e-9f)
                                      ? 127.0f / row_max_abs
                                      : 0.0f;
        const float row_dequant_scale = (row_max_abs > 1e-9f)
                                        ? row_max_abs / 127.0f
                                        : 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float scaled = row_ptr[c] * row_quant_scale;
            const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
            x_quant[static_cast<size_t>(c)] =
                static_cast<int8_t>(std::lround(clamped));
        }
        // Zero the tail pad so SIMD reads past cols don't pollute the sum.
        for (int c = cols; c < cols + 32; ++c) {
            x_quant[static_cast<size_t>(c)] = 0;
        }

        for (int out_col = 0; out_col < out_cols; ++out_col) {
            const int row_offset = out_col * packed_stride;
            __m256i acc_vec = _mm256_setzero_si256();

            int c = 0;
            for (; c <= cols - 32; c += 32) {
                // Step 1: 8 UNPACK_LUT loads fill a 32-byte aligned buffer
                // with the unpacked ternary weights for this 32-wide chunk.
                const uint8_t* p = packed_ptr + row_offset + (c >> 2);
                alignas(32) int8_t w_temp[32];
                uint32_t* w_ptr32 = reinterpret_cast<uint32_t*>(w_temp);
                w_ptr32[0] = UNPACK_LUT_4WAY[p[0]];
                w_ptr32[1] = UNPACK_LUT_4WAY[p[1]];
                w_ptr32[2] = UNPACK_LUT_4WAY[p[2]];
                w_ptr32[3] = UNPACK_LUT_4WAY[p[3]];
                w_ptr32[4] = UNPACK_LUT_4WAY[p[4]];
                w_ptr32[5] = UNPACK_LUT_4WAY[p[5]];
                w_ptr32[6] = UNPACK_LUT_4WAY[p[6]];
                w_ptr32[7] = UNPACK_LUT_4WAY[p[7]];

                // Step 2: SIMD load of weights + activations.
                __m256i w_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(w_temp));
                __m256i x_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&x_quant[c]));

                // Step 3: ternary multiply via _mm256_sign_epi8.
                __m256i prod = _mm256_sign_epi8(x_vec, w_vec);

                // Step 4: widen to int16 then accumulate horizontally into int32.
                __m256i prod_lo = _mm256_cvtepi8_epi16(
                    _mm256_castsi256_si128(prod));
                __m256i prod_hi = _mm256_cvtepi8_epi16(
                    _mm256_extracti128_si256(prod, 1));
                acc_vec = _mm256_add_epi32(acc_vec,
                                           _mm256_madd_epi16(prod_lo, ones_16));
                acc_vec = _mm256_add_epi32(acc_vec,
                                           _mm256_madd_epi16(prod_hi, ones_16));
            }

            // Reduce the 8 int32 lanes to a scalar.
            int32_t lanes[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), acc_vec);
            int32_t total = 0;
            for (int k = 0; k < 8; ++k) total += lanes[k];

            // Scalar tail for cols not a multiple of 32.
            for (; c < cols; ++c) {
                const int flat = out_col * cols + c;
                const int byte_idx = flat >> 2;
                const int bit_shift = (flat & 0x3) << 1;
                const uint8_t enc = (packed_ptr[byte_idx] >> bit_shift) & 0x3;
                const int8_t w_val = kDecodeI8Lut[enc];
                if (w_val != 0) {
                    total += static_cast<int32_t>(x_quant[c]) * w_val;
                }
            }

            // Dequantize: divide by row_quant_scale to undo the int8 scaling,
            // then apply the existing weight_scale + act_scale + affine pipeline.
            const float deq = static_cast<float>(total) * row_dequant_scale;
            y_ptr[row * out_cols + out_col] =
                apply_output_affine(deq * act_scale * weight_scale,
                                    out_col,
                                    magnitude,
                                    bias,
                                    use_bias);
        }
    }
}
#endif  // NSOS_ARCH_X86 && __AVX2__

// Runtime selection of LUT path based on env var.  Default = scalar
// (zero risk of regression on existing pipeline).  Set
// NSOS_USE_LUT_SIMD=1 to opt into the new bitmamba-style AVX2 path.
//
// Cached per-process: the env lookup happens once on first call, then
// reused so we don't hit getenv() on every GEMM.
static bool lut_simd_enabled() {
#if defined(NSOS_ARCH_X86) && defined(__AVX2__)
    static const bool cached = []() {
        const char* env = std::getenv("NSOS_USE_LUT_SIMD");
        if (env == nullptr) return false;
        const std::string val(env);
        return val == "1" || val == "true" || val == "TRUE" || val == "yes";
    }();
    return cached;
#else
    return false;
#endif
}

} // namespace

// Single source of truth for the AVX2 runtime check. NSOS_ENABLE_AVX2_KERNELS
// is a *build* switch (default ON for x86_64) and says nothing about the CPU
// that will actually execute the binary, so every AVX2 entry point outside
// this translation unit must consult this before dispatching.
bool avx2_runtime_supported() {
    static const bool supported = cpu_supports_avx2();
    return supported;
}

void BitNetAdapter::gemm_158bit_lut(const Tensor& input,
                                    const std::vector<uint32_t>& packed_weights,
                                    const std::vector<float>& act_scales,
                                    float weight_scale,
                                    Tensor& output,
                                    const float* magnitude,
                                    const float* bias,
                                    bool use_bias) {
#if defined(NSOS_ARCH_X86) && defined(__AVX2__)
    // Cherry-pick #1: route to bitmamba SIMD path when opted in.
    // See OXN/nsos/docs/BITMAMBA_LUT_INTEGRATION.md.
    if (lut_simd_enabled()) {
        gemm_158bit_lut_simd_impl(input, packed_weights, act_scales,
                                  weight_scale, output, magnitude, bias,
                                  use_bias);
        return;
    }
#endif

    // Default: existing scalar implementation, byte-for-byte preserved
    // to avoid any regression on the current production pipeline.
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
