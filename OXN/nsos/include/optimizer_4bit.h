#pragma once
// 4-bit optimizer states — Li, Chen & Zhu, "Memory Efficient Optimizers with
// 4-bit States", NeurIPS 2023 (arXiv:2309.01507).  Stores Adam's first and
// second moments in 4 bits to cut optimizer-state memory ~8x versus FP32
// (FP32 keeps two full-size float buffers, m and v, i.e. 8 bytes/param).
//
// Two ideas from the paper are implemented faithfully:
//
//   * First moment m (signed): block-wise abs-max normalisation with block
//     size B=128 (the paper drops the usual 2048 to 128 because moment outlier
//     patterns are local), then a dynamic-exponent-style *signed* code map that
//     is denser near zero — that is where momentum mass concentrates, so a
//     linear map would waste resolution.
//
//   * Second moment v (non-negative): for a 2-D state, rank-1 normalisation
//
//         N(x_ij) = x_ij / min{ r_i, c_j },   r_i = max_j x_ij,  c_j = max_i x_ij
//
//     then a LINEAR code map  T(k) = (k+1)/2^b  (k = 0..2^b-1).  For b=4 the
//     smallest level is 1/16 = 0.0625 and **no level is zero**.  This is the
//     fix for the "zero-point problem": v enters the update as 1/sqrt(v_hat),
//     so a v entry rounded to 0 would make the step explode.  Non-2-D states
//     fall back to block-wise abs-max with the same no-zero linear map (still
//     zero-safe, just without the rank-1 row/column structure).
//
//   * Tensors with <= kQuant4MinElems elements stay in FP32 (per paper §4.2:
//     small tensors are cheap to keep exact and sensitive to quantisation).
//
// Usage model (matches the trainer's Adam step): the state is dequantised to
// FP32 scratch, the ordinary Adam update runs, then it is re-quantised.  This
// costs a little compute per step for the memory saving; the optimiser math is
// unchanged.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nsos {

constexpr int kQuant4BlockSize = 128;  // paper's reduced block size
constexpr int kQuant4Levels = 16;      // 4 bits
constexpr int kQuant4MinElems = 4096;  // below this, keep FP32

// Per-parameter quantised Adam state.  Holds either the packed 4-bit state or
// (small tensors) an FP32 fallback.  `quantized == false` selects the fallback.
struct Quant4OptState {
    int n = 0;
    bool quantized = false;

    // First moment (m): block-wise abs-max, dynamic-exponent-style signed map.
    std::vector<uint8_t> m_codes;   // two 4-bit codes per byte, ceil(n/2) bytes
    std::vector<float> m_absmax;    // one scale per block of kQuant4BlockSize

    // Second moment (v): rank-1 (2-D) or block-wise (otherwise), linear map.
    bool v_rank1 = false;
    int v_rows = 0;
    int v_cols = 0;
    std::vector<uint8_t> v_codes;   // two 4-bit codes per byte
    std::vector<float> v_row;       // rank-1: r_i (size v_rows)
    std::vector<float> v_col;       // rank-1: c_j (size v_cols)
    std::vector<float> v_absmax;    // block-wise fallback: one scale per block

    // FP32 fallback (tensors with n <= kQuant4MinElems).
    std::vector<float> m_fp32;
    std::vector<float> v_fp32;

    // Resident bytes of this state (for the memory-saving assertion in tests).
    std::size_t bytes() const;
};

// First moment: quantise `m[0..n)` into `st`, or dequantise `st` back to `m_out`.
void quant4_store_m(const float* m, int n, Quant4OptState& st);
void quant4_load_m(const Quant4OptState& st, float* m_out, int n);

// Second moment: quantise `v[0..n)` into `st`.  When rows > 0 and rows*cols==n
// the rank-1 path is used (recommended for 2-D weight states); pass rows <= 0
// to force the block-wise fallback.
void quant4_store_v(const float* v, int n, int rows, int cols, Quant4OptState& st);
void quant4_load_v(const Quant4OptState& st, float* v_out, int n);

}  // namespace nsos
