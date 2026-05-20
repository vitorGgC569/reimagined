// T-MAC-style LUT GEMM — CPU implementation.
// See include/lut_tmac.h + docs/LUT_TMAC_DESIGN.md.

#include "../include/lut_tmac.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nsos {
namespace lut_tmac {

// Decode table: maps 2-bit packed values back to ternary {-1, 0, +1}.
// Convention follows BitNetAdapter (encoded values 0,1,2 → -1,0,+1; 3 reserved/zero).
// We keep our own copy here so the header dependency stays minimal.
static constexpr int8_t kDecode[4] = {-1, 0, 1, 0};

// ────────────────────────────────────────────────────────────────────
//   Packing helpers
// ────────────────────────────────────────────────────────────────────

// Unpack 4 ternary weights at `flat_index` from packed_weights.
// flat_index is in units of WEIGHTS (not bytes).  Returns array of 4 int8
// values each in {-1, 0, +1}.
static inline std::array<int8_t, 4> unpack4(
    const uint8_t* packed, int flat_index) noexcept {
    std::array<int8_t, 4> out{};
    for (int i = 0; i < 4; ++i) {
        const int fi = flat_index + i;
        const int packed_index = fi >> 2;
        const int packed_shift = (fi & 0x3) << 1;
        const uint8_t code = (packed[packed_index] >> packed_shift) & 0x3;
        out[static_cast<size_t>(i)] = kDecode[code];
    }
    return out;
}

// Encode 4 ternary values into a single LUT index in [0, 80] (3^4 = 81).
// w[i] in {-1, 0, +1}.  Maps to base-3 digit: -1→0, 0→1, +1→2.
// Then index = d0 + 3*d1 + 9*d2 + 27*d3.
static inline int code_from_weights(const std::array<int8_t, 4>& w) noexcept {
    int idx = 0;
    int mul = 1;
    for (int i = 0; i < 4; ++i) {
        const int digit = w[static_cast<size_t>(i)] + 1;  // -1→0, 0→1, +1→2
        idx += digit * mul;
        mul *= 3;
    }
    return idx;
}

// ────────────────────────────────────────────────────────────────────
//   Heat map computation (one-time at weight load)
// ────────────────────────────────────────────────────────────────────

bool format_supported(int K_features) noexcept {
    // Need K_features % 4 == 0 so groups align.  Future: relax via tail
    // handling in gemm_158bit_lut_tmac.
    return (K_features > 0) && (K_features % kGroupSize == 0);
}

std::vector<uint8_t> compute_heat_map(
    const std::vector<uint32_t>& packed_weights,
    int N, int K) {
    if (!format_supported(K)) {
        return {};  // caller treats empty as "no heat map, run dense"
    }
    const int n_groups = K / kGroupSize;          // weight-groups per row
    const int n_out_tiles = (N + kTileRows - 1) / kTileRows;
    // Bits per (out_tile × group): 1.  Layout: [out_tile][group_byte_offset].
    // We pack each group's per-tile bit linearly, 8 bits per byte.
    const int bits_total = n_out_tiles * n_groups;
    const int bytes = (bits_total + 7) / 8;

    std::vector<uint8_t> mask(static_cast<size_t>(bytes), 0u);
    const uint8_t* packed = reinterpret_cast<const uint8_t*>(packed_weights.data());

    // For each output row tile and weight group, OR-reduce the kTileRows
    // weight rows.  If ANY weight in the (tile × group) block is non-zero,
    // set the bit.
    int bit_pos = 0;
    for (int t = 0; t < n_out_tiles; ++t) {
        const int row_lo = t * kTileRows;
        const int row_hi = std::min(row_lo + kTileRows, N);
        for (int g = 0; g < n_groups; ++g) {
            const int col_base = g * kGroupSize;
            bool any_nonzero = false;
            for (int row = row_lo; row < row_hi && !any_nonzero; ++row) {
                const int flat = row * K + col_base;
                const auto w = unpack4(packed, flat);
                for (int i = 0; i < kGroupSize; ++i) {
                    if (w[static_cast<size_t>(i)] != 0) {
                        any_nonzero = true;
                        break;
                    }
                }
            }
            if (any_nonzero) {
                mask[static_cast<size_t>(bit_pos >> 3)] |=
                    static_cast<uint8_t>(1u << (bit_pos & 7));
            }
            ++bit_pos;
        }
    }
    return mask;
}

HeatMapStats heat_map_stats(const std::vector<uint8_t>& heat_map,
                             int N, int K) noexcept {
    HeatMapStats out{0, 0, 0.0};
    if (!format_supported(K)) return out;
    const int n_groups = K / kGroupSize;
    const int n_out_tiles = (N + kTileRows - 1) / kTileRows;
    const int bits_total = n_out_tiles * n_groups;
    out.total_tiles = bits_total;
    for (int b = 0; b < bits_total; ++b) {
        const bool set =
            (heat_map[static_cast<size_t>(b >> 3)] >> (b & 7)) & 1u;
        if (!set) out.zero_tiles += 1;
    }
    out.sparsity = (out.total_tiles > 0)
        ? static_cast<double>(out.zero_tiles) / out.total_tiles
        : 0.0;
    return out;
}

// ────────────────────────────────────────────────────────────────────
//   LUT construction (per-row, hot path)
// ────────────────────────────────────────────────────────────────────

// Build the LUT for one batch row and all weight groups.
// LUT layout: [n_groups][81] floats.
// LUT[g][c] = Σ_{k in group g} a[g*4 + k] · digit_to_signed(c, k)
// where digit_to_signed extracts the k-th base-3 digit of c and maps
// 0→-1, 1→0, 2→+1.
//
// Total adds per LUT slot: ~4 (one per weight in the group).
// Total LUT building: n_groups × 81 × 4 adds ≈ K × 81 adds.
// vs gemm-skipping cost: K × N multiplies per row.
// Break-even at N ≈ 81.  Most NSOS layers have N ≥ 256 → 3-5× amortization.
static void build_row_lut(
    const float* row_in,                // [K]
    int K,
    std::vector<float>& lut_out         // resized to n_groups × 81
) {
    const int n_groups = K / kGroupSize;
    lut_out.resize(static_cast<size_t>(n_groups) * kLutEntries);
    // For each weight group within this row...
    for (int g = 0; g < n_groups; ++g) {
        const float a0 = row_in[g * 4 + 0];
        const float a1 = row_in[g * 4 + 1];
        const float a2 = row_in[g * 4 + 2];
        const float a3 = row_in[g * 4 + 3];
        float* lut_g = lut_out.data() + static_cast<size_t>(g) * kLutEntries;
        // 3^4 = 81 entries.  Use 4 nested loops with -1/0/+1 explicitly
        // so the compiler can constant-fold the sign tables.
        int c = 0;
        for (int d3 = -1; d3 <= 1; ++d3) {
            const float p3 = d3 * a3;
            for (int d2 = -1; d2 <= 1; ++d2) {
                const float p23 = p3 + d2 * a2;
                for (int d1 = -1; d1 <= 1; ++d1) {
                    const float p123 = p23 + d1 * a1;
                    for (int d0 = -1; d0 <= 1; ++d0) {
                        lut_g[c++] = p123 + d0 * a0;
                    }
                }
            }
        }
        assert(c == kLutEntries);
    }
}

// ────────────────────────────────────────────────────────────────────
//   Output-affine (same convention as bitnet_adapter.cpp)
// ────────────────────────────────────────────────────────────────────

static inline float apply_output_affine(float v, int out_col,
                                         const float* magnitude,
                                         const float* bias,
                                         bool use_bias) noexcept {
    if (magnitude != nullptr) v *= magnitude[out_col];
    if (use_bias && bias != nullptr) v += bias[out_col];
    return v;
}

// ────────────────────────────────────────────────────────────────────
//   Main GEMM
// ────────────────────────────────────────────────────────────────────

void gemm_158bit_lut_tmac(
    const Tensor& input,
    const std::vector<uint32_t>& packed_weights,
    const std::vector<uint8_t>& heat_map,
    const std::vector<float>& act_scales,
    float weight_scale,
    Tensor& output,
    const float* magnitude,
    const float* bias,
    bool use_bias) {
    const int B = input.shape[0];
    const int K = input.shape[1];
    const int N = output.shape[1];

    if (!format_supported(K)) {
        throw std::runtime_error(
            "lut_tmac::gemm requires K %% kGroupSize == 0; got K=" +
            std::to_string(K));
    }
    const int n_groups = K / kGroupSize;
    const int n_out_tiles = (N + kTileRows - 1) / kTileRows;
    const bool have_heat_map =
        !heat_map.empty()
        && static_cast<int>(heat_map.size()) >=
               (n_out_tiles * n_groups + 7) / 8;

    const float* x_ptr = input.data();
    float* y_ptr = output.data();
    const uint8_t* packed_ptr =
        reinterpret_cast<const uint8_t*>(packed_weights.data());

    std::vector<float> lut;
    lut.reserve(static_cast<size_t>(n_groups) * kLutEntries);

    for (int b = 0; b < B; ++b) {
        const float act_scale = (b < static_cast<int>(act_scales.size()))
            ? act_scales[static_cast<size_t>(b)] : 1.0f;
        const float* row_in = x_ptr + b * K;
        build_row_lut(row_in, K, lut);

        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            const int weight_row_base = n * K;
            const int out_tile = n / kTileRows;
            const int tile_bit_base = out_tile * n_groups;

            for (int g = 0; g < n_groups; ++g) {
                // Block-sparse skip
                if (have_heat_map) {
                    const int bit = tile_bit_base + g;
                    const bool any_nonzero =
                        (heat_map[static_cast<size_t>(bit >> 3)] >>
                         (bit & 7)) & 1u;
                    if (!any_nonzero) continue;
                }
                const int flat = weight_row_base + g * kGroupSize;
                const auto w = unpack4(packed_ptr, flat);
                const int code = code_from_weights(w);
                acc += lut[static_cast<size_t>(g) * kLutEntries + code];
            }
            y_ptr[b * N + n] = apply_output_affine(
                acc * act_scale * weight_scale, n, magnitude, bias, use_bias);
        }
    }
}

// ────────────────────────────────────────────────────────────────────
//   Env opt-in
// ────────────────────────────────────────────────────────────────────

bool env_opt_in() noexcept {
    const char* v = std::getenv("NSOS_TMAC_LUT_GEMM");
    if (v == nullptr) return false;
    const std::string s(v);
    return (s == "1" || s == "true" || s == "TRUE" || s == "yes");
}

}  // namespace lut_tmac
}  // namespace nsos
