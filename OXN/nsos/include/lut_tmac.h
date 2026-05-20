// T-MAC-style LUT GEMM for BitNet 1.58.
//
// See docs/LUT_TMAC_DESIGN.md for the technique.
//
// Public API:
//   compute_heat_map(packed_weights, ...) -> std::vector<uint8_t>
//     One-time, at weight-load.  Returns block-sparse mask for skip.
//
//   gemm_158bit_lut_tmac(input, packed_weights, heat_map, ...)
//     Per-call.  Builds the per-row activation LUT then iterates outputs.
//
// Both functions are CPU-only in this MVP.  GPU port is the natural
// follow-on (see design doc).

#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace nsos {

namespace lut_tmac {

// Compile-time constants tuned for K=4 (see design doc, "Why K=4")
constexpr int kGroupSize = 4;
constexpr int kLutEntries = 81;          // 3^4
constexpr int kTileRows = 16;            // block-sparse mask granularity
constexpr int kTileCols = 4;             // weight-group granularity inside tile

// Returns true if the current ternary pack format is supported.  We use
// the same 2-bits-per-weight pack the existing kernel uses (4 weights
// per byte).  Future packs (e.g. base-3 5/byte) can route here when the
// unpacker is generalized.
bool format_supported(int K_features) noexcept;

// Pre-compute the block-sparse heat map.
//
// For each (output_row_tile=n/kTileRows, weight_group_g=k/kGroupSize)
// pair, store 1 bit indicating whether ANY of the kTileRows × kGroupSize
// weights in that tile is non-zero.
//
// Layout: row-major over (out_tiles, weight_groups), packed 8 bits per
// byte.  Total size = ceil(N / kTileRows) * ceil(K / kGroupSize) / 8 bytes.
//
// packed_weights expected in the existing 2-bits-per-weight format
// (kDecodeLut from bitnet_adapter.cpp).
std::vector<uint8_t> compute_heat_map(
    const std::vector<uint32_t>& packed_weights,
    int N,                                // output features (out_cols)
    int K);                               // input features (cols)

// Density / sparsity stats on the heat map.  Used by callers to decide
// whether the LUT path is worth running vs falling back to dense kernel.
struct HeatMapStats {
    int total_tiles;
    int zero_tiles;            // entirely sparse, skipped at GEMM time
    double sparsity;           // zero_tiles / total_tiles
};
HeatMapStats heat_map_stats(const std::vector<uint8_t>& heat_map,
                             int N, int K) noexcept;

// Main entry: T-MAC-style LUT GEMM.
//
// Equivalence: produces the same output (within fp roundoff) as
// BitNetAdapter::gemm_158bit_lut for the same packed weights + inputs.
// Verified by tests/test_lut_tmac.cpp.
//
// Args:
//   input          - dense FP32 activations [B, K]
//   packed_weights - 2-bits-per-weight packed, layout [N, K] flat
//   heat_map       - from compute_heat_map(); if empty, no block-sparse skip
//   act_scales     - per-row activation scale [B]; size 0 == 1.0 each
//   weight_scale   - global weight scale
//   output         - dense FP32 output [B, N] (overwritten)
//   magnitude      - optional per-output multiplier (nullptr = ignored)
//   bias           - optional bias (nullptr = ignored)
//   use_bias       - whether to add bias
void gemm_158bit_lut_tmac(
    const Tensor& input,
    const std::vector<uint32_t>& packed_weights,
    const std::vector<uint8_t>& heat_map,
    const std::vector<float>& act_scales,
    float weight_scale,
    Tensor& output,
    const float* magnitude = nullptr,
    const float* bias = nullptr,
    bool use_bias = false);

// Convenience: returns true when env var `NSOS_TMAC_LUT_GEMM=1` is set.
// Callers (BitLinear::gemm_158bit_ultra) use this to decide dispatch.
bool env_opt_in() noexcept;

}  // namespace lut_tmac
}  // namespace nsos
