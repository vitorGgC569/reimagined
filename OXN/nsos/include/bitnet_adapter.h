#pragma once

#include "tensor.h"

#include <array>
#include <cstdint>
#include <vector>

namespace nsos {

// ── UNPACK_LUT for Cherry-pick #1 (bitmamba.cpp SIMD+LUT path) ──────────────
// 256-entry table mapping each possible packed byte to its 4 unpacked
// ternary weights as a single uint32_t (4 × int8 in little-endian byte
// order: byte0 = lowest 2 bits of input).
//
// Encoding matches both the existing `kDecodeLut` (bitnet_adapter.cpp:29)
// and bitmamba.cpp's encoding for the valid range {0, 1, 2}:
//   - packed 2-bit value 0 → int8 -1
//   - packed 2-bit value 1 → int8  0
//   - packed 2-bit value 2 → int8 +1
//   - packed 2-bit value 3 → int8  0 (treated as 0; our packer never writes 3)
//
// Computed at compile time (consteval) so the table lives in .rodata with
// zero runtime cost.  Total size: 256 × 4 bytes = 1 KB, fits comfortably
// in L1 data cache on any modern CPU.
//
// See OXN/nsos/docs/BITMAMBA_LUT_INTEGRATION.md for the full design.
consteval std::array<uint32_t, 256> make_unpack_lut_4way() {
    // Decode map for a single 2-bit slot. Matches kDecodeLut on the FP side.
    constexpr int8_t decode[4] = {-1, 0, 1, 0};
    std::array<uint32_t, 256> lut{};
    for (int byte = 0; byte < 256; ++byte) {
        const int8_t s0 = decode[(byte >> 0) & 0x3];
        const int8_t s1 = decode[(byte >> 2) & 0x3];
        const int8_t s2 = decode[(byte >> 4) & 0x3];
        const int8_t s3 = decode[(byte >> 6) & 0x3];
        // Little-endian pack: slot0 in lowest byte, slot3 in highest.
        // Matches how _mm256_load_si256 sees a 32-byte aligned array
        // of int8_t when we cast a uint32_t* to int8_t*.
        const uint32_t u0 = static_cast<uint32_t>(static_cast<uint8_t>(s0));
        const uint32_t u1 = static_cast<uint32_t>(static_cast<uint8_t>(s1));
        const uint32_t u2 = static_cast<uint32_t>(static_cast<uint8_t>(s2));
        const uint32_t u3 = static_cast<uint32_t>(static_cast<uint8_t>(s3));
        lut[byte] = u0 | (u1 << 8) | (u2 << 16) | (u3 << 24);
    }
    return lut;
}

inline constexpr std::array<uint32_t, 256> UNPACK_LUT_4WAY = make_unpack_lut_4way();

class BitNetAdapter {
public:
  static void gemm_158bit_lut(const Tensor &input,
                              const std::vector<uint32_t> &packed_weights,
                              const std::vector<float> &act_scales,
                              float weight_scale,
                              Tensor &output,
                              const float *magnitude = nullptr,
                              const float *bias = nullptr,
                              bool use_bias = false);

  static void gemm_158bit_i8(const Tensor &input,
                             const std::vector<int8_t> &unpacked_weights,
                             const std::vector<int32_t> &weight_row_sums,
                             const std::vector<float> &act_scales,
                             float weight_scale,
                             Tensor &output,
                             const float *magnitude = nullptr,
                             const float *bias = nullptr,
                             bool use_bias = false);

  static void pack_weights_microsoft_style(const float *src,
                                           uint8_t *dst,
                                           int rows,
                                           int cols);

  static void unpack_weights_microsoft_style_to_i8(
      const std::vector<uint32_t> &packed_weights,
      int rows,
      int cols,
      std::vector<int8_t> &dst);
};

// True only when the executing CPU really implements AVX2 (CPUID leaf 1 +
// XCR0 state check + leaf 7 AVX2 bit), cached after the first query.
//
// NSOS_ENABLE_AVX2_KERNELS only records that AVX2 kernels were *compiled*; it
// defaults to ON for every x86_64 build. A binary produced on a modern host
// and shipped to an older x86_64 CPU would execute an illegal instruction if
// any call site dispatched on the build switch alone, so every AVX2 entry
// point must gate on this predicate.
bool avx2_runtime_supported();

} // namespace nsos
