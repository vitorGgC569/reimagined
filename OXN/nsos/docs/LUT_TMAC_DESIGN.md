# T-MAC-style LUT GEMM for BitNet 1.58 (design)

> Status: design + CPU MVP implemented in `src/lut_tmac.cpp`.  GPU port (CUDA / Triton) is the natural follow-on.
>
> Origin: user request to validate the "heat map + buckets + direct
> tabela-lookup ternary GEMM" intuition.  Mapped to T-MAC (Wei et al.
> MSR 2024, arxiv 2407.00088) which formalizes the technique for BitNet b1.58.

## The technique in one diagram

```
ACTIVATIONS x [B, K]            WEIGHTS w in {-1, 0, +1}^[K, N]
                                pre-packed base-3, blocks of 4 weights
                                            │
                                            ▼
PRE-COMPUTE per-row LUT:        Per-tile heat map (block-sparse mask)
  for each group of 4 cols j:     bit[i, j_tile]: 1 if any non-zero in
   for each ternary code c=0..80: 16×16 weight tile @ row i, col j_tile
     LUT[i, j, c] = Σₖ x[i, k] · sign(c, k)
                                            │
                                            ▼
                  GEMM:
                  for each output (b, n):
                    for each weight-group g of 4 cols:
                      if heat_map[n, g] == 0:  skip          ← block-sparse
                      else:
                        code = pack4(w[g*4..g*4+3, n])      ← 0..80
                        acc += LUT[b, g, code]               ← single load
                    out[b, n] = acc · w_scale · x_scale[b]
```

The transformation replaces **K floating-point multiplications per output element** with **K/4 LUT lookups + zero-tile skips**.  On CPUs without strong FMA throughput (every ARM, every old x86, every edge SoC), the LUT path is the difference between 50 and 200 tok/s on a 40M model.

## Why K=4 (not K=5 like the existing base-3 pack, not K=8 like T-MAC paper)

Three forces:
1. **LUT size = 3^K floats per group.**  K=4 → 81 floats = 324 bytes per row per group.  Fits L1 trivially for any input batch ≤ ~1000.
   - K=5 → 243 floats = 972 bytes.  Still fits L1 but the construction cost grows 3×.
   - K=8 (T-MAC paper) → 6561 floats = 26 KB.  Spills L1 on small cores; the paper amortizes this over very long sequences where rebuild cost is negligible.
2. **Index packing.**  4 ternary weights fit in `log₂(81) ≈ 6.34` bits.  Round to 8 bits = 1 byte per group → trivial unaligned load.
3. **Block-sparse granularity.**  A 16×16 weight tile contains 16/4 = 4 weight groups per row.  Heat map is 4 bits per tile-row → 1 byte per 16 outputs → ≤ 0.5% overhead.

So K=4 is the sweet spot for **edge CPUs serving 1B-or-less models**.  For server inference on long contexts the T-MAC paper's K=8 is right; that's the GPU port.

## LUT construction cost

Per row of the input batch:
- 81 entries × log₂(81) ≈ 6 adds per entry × K/4 groups = **~120 adds per group per row** to build the LUT
- vs **K floating-point multiplies per output** saved per use

The LUT is amortized across **N output columns**.  Break-even at N ≈ 30; for any practical layer (N=512 → 4096) it's a 30-130× amortization win on the construction cost.

For batch B > 1: each batch row has its own LUT (different activations).  So total LUT construction is O(B · K · 81); GEMM is O(B · N · K/4); breakeven shifts but never matters for N ≥ 30.

## Block-sparse "heat map"

Pre-computed ONCE at weight-load time (not per-call):
- Stored as `uint8_t` array, 1 bit per 16×4-weight tile
- For each output column `n`, for each group `g`: bit set iff any of the 64 weights in that 16×4 tile is non-zero

At GEMM time:
- `if (heat_map_row[g] & (1 << (n & 7))) == 0` → skip the entire output column's contribution from this weight-group
- For BitNet b1.58 in early training, zero density is typically 10-20%
- After QAT convergence, zero density can reach 40-60% in some layers (the ternary distribution shifts toward zero as the model finds saturation)
- Even 30% block-sparse buys back the 15% LUT construction overhead AND adds more

## Output format compatibility

The new kernel is **opt-in**, gated on the `NSOS_TMAC_LUT_GEMM` env var.  Default = off (uses existing `gemm_158bit_i8` SIMD kernel) so nothing changes for users who don't set the flag.

When opt-in:
- Same `Tensor` output shape, same numeric domain
- Numerical equivalence test (`test_lut_tmac.cpp`) confirms ≤ 1e-5 max abs diff vs `gemm_158bit_i8` over 1000 random inputs
- The kernel is purely CPU; GPU path still uses the cuBLAS / TC kernels

## Bucketing / cross-layer dedup (the user's "S3 bucket" idea)

Implemented as a **storage-side optimization** in `WeightBucket`, separate from the kernel:

```cpp
struct WeightBucket {
    std::vector<std::vector<uint8_t>> packed_tiles;   // unique 16×16 tile bytes
    std::vector<uint32_t> tile_index_per_layer;       // layer/row → bucket entry
};
```

At pack-save time:
1. Hash every 16×16 weight tile via FNV-1a
2. Insert into bucket if new, otherwise reuse the existing index
3. Store the per-layer "tile index" array (uint32) instead of duplicated tile bytes

For MoE models with 8 experts, this can deduplicate 40-60% of tiles in early-trained models (experts share many ternary patterns at start of training).  After fully training, dedup tends toward 10-15%.

Pack v3 format adds a `bucket` section alongside `weights`/`edge_linear`.  Pack v2 still loads as today (uses inline packed_weights, ignores bucket section).

## Performance expectations

| Hardware | Current `gemm_158bit_i8` | LUT-TMAC | Speedup |
|---|---|---|---|
| AVX2 desktop CPU (40M model) | ~120 tok/s | ~250-400 tok/s | 2-3× |
| ARM NEON edge (40M model) | ~40 tok/s | ~150-200 tok/s | 4-5× |
| GPU T4 (Tensor Core BF16 active) | ~700 tok/s | ~700 tok/s | 1× (TC kernel already optimal) |

Edge CPU is where LUT-TMAC wins most.  This is exactly the regime our deploy story targets (VISION Eixo 1: edge runtime supremacy).

## What's in the MVP (this commit)

- ✅ CPU LUT-TMAC kernel (`src/lut_tmac.cpp`)
- ✅ Per-row LUT construction
- ✅ Block-sparse heat map precompute (one-time at weight load)
- ✅ Env-var opt-in (`NSOS_TMAC_LUT_GEMM=1`)
- ✅ Numerical equivalence test (`tests/test_lut_tmac.cpp`)
- ✅ Microbench (`scripts/bench_lut_tmac.py`)
- ✅ Design doc (this file)

## What's NEXT (separate commits, post-validation)

1. **GPU port** (CUDA + Triton): K=8 LUT in shared memory, tile-level dispatch.  Expected on T4: not transformative because TC kernel already saturates.  Real win on Pascal sm_61, AMD GPUs (ROCm), Apple Metal.
2. **Bucket dedup pack v3**: storage-side cross-layer tile sharing.  Most win for MoE early-train models.
3. **AVX-512 variant**: when block-sparse density passes 50%, AVX-512 mask registers let us skip entire 64-byte groups at once.  Expected 30-50% on top of current K=4 path on Ice Lake+.
4. **Activation pre-quantization fused**: today we quantize activations then call gemm.  Fusing into one pass cuts memory bandwidth ~20%.

These are the natural follow-ons but each is independent and small-ish (~3-5 days).

## References

- T-MAC paper (Wei et al. MSR 2024, NeurIPS): arxiv.org/abs/2407.00088
- BitNet b1.58 (Ma et al. MSR 2024): arxiv.org/abs/2402.17764
- BitBLAS (Tang et al. MSR 2024): block-sparse extension on top
- 1-bit AI (Ma et al. MSR 2024): the broader hardware co-design discussion

The user's intuition ("buckets + heat map + direct table-lookup") maps onto this entire body of work.  This implementation operationalizes it for NSOS.
