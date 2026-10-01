# Bitmamba LUT Integration — Cherry-pick #1

> Porting the `UNPACK_LUT` + AVX2 SIMD kernel pattern from `bitmamba.cpp`
> (Zhayr1) into our `BitNetAdapter::gemm_158bit_lut`.
>
> **Source repo:** `external_refs/repos/bitmamba.cpp/src/kernels.cpp`
> (lines 100-130 for the inner AVX2 loop with UNPACK_LUT).
>
> **License:** MIT — compatible with Oxta's product license.

---

## Critical finding from code archaeology

Our codebase already has TWO kernels for 1.58-bit ternary GEMM:

| Function | File | Format | Implementation |
|----------|------|--------|----------------|
| `gemm_158bit_i8` | `bitnet_adapter.cpp:310` | Unpacked `int8_t[]` | **AVX2 SIMD** (state-of-the-art) |
| `gemm_158bit_lut` | `bitnet_adapter.cpp:269` | Packed `uint32_t[]` (4 weights/byte) | **Pure scalar** (1 mult per element) |

The existing `gemm_158bit_lut` uses the SAME packed format that bitmamba uses,
but processes scalar one element at a time inside the inner loop:

```cpp
// Our current gemm_158bit_lut (bitnet_adapter.cpp:293-298)
for (int col = 0; col < cols; ++col) {
    const int flat_index = row_weight_base + col;
    const int packed_index = flat_index >> 2;
    const int packed_shift = (flat_index & 0x3) << 1;
    const uint8_t encoded = (packed_ptr[packed_index] >> packed_shift) & 0x3;
    acc += row_ptr[col] * kDecodeLut[encoded];   // ← scalar mult!
}
```

This explains the historical benchmark result documented in
`docs/LUT_TMAC_DESIGN.md`:

> "LUT-TMAC vs SIMD AVX2 prod path: 0.05× (LUT-TMAC LOSES 20×)"

Bitmamba.cpp's contribution is showing how to keep the packed memory layout
(memory bandwidth wins) **while ALSO using AVX2 SIMD** on the unpacked
weights (compute wins).  The combination is what gives the speedup.

---

## bitmamba.cpp's pattern (the gold)

From `external_refs/repos/bitmamba.cpp/src/kernels.cpp:220-240`:

```cpp
const uint8_t* p = packed_ptr + row_offset + c/4;

// 1. Unpack 8 bytes (32 weights) via LUT — single uint32 load per byte
alignas(32) int8_t w_temp[32];
uint32_t* w_ptr32 = (uint32_t*)w_temp;
w_ptr32[0] = UNPACK_LUT[p[0]];   // 4 weights → uint32 with 4 int8 values
w_ptr32[1] = UNPACK_LUT[p[1]];
// ... 8 unrolled loads cover 32 weights

// 2. Load into AVX2 register
__m256i w_vec = _mm256_load_si256((__m256i*)w_temp);
__m256i x_vec = _mm256_loadu_si256((__m256i*)&x_quant[c]);

// 3. Multiply via sign_epi8: when w=+1 keeps x, w=-1 flips x, w=0 zeros it
__m256i prod = _mm256_sign_epi8(x_vec, w_vec);

// 4. Widen + horizontal add via madd_epi16
__m256i prod_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
__m256i prod_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_lo, ones_16));
acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_hi, ones_16));
```

**Three innovations together:**
1. **Memory:** 4 weights per byte (4× less DRAM bandwidth vs unpacked int8)
2. **Compute unpack:** `UNPACK_LUT[byte]` → uint32 with 4 int8 values, single load
3. **Compute math:** `_mm256_sign_epi8` is a single-cycle instruction that does
   the ternary multiplication implicitly — no scalar comparison needed

---

## Encoding compatibility analysis

**bitmamba.cpp encoding** (line 119):
```cpp
int8_t w_val = ((packed_ptr[byte_idx] >> bit_shift) & 0x03) - 1;
//                                                          ^^^^^ shift to {-1, 0, +1, +2}
```
- `0b00` (0) → −1
- `0b01` (1) →  0
- `0b10` (2) → +1
- `0b11` (3) → +2 *(assumed to never appear — invalid)*

**Our encoding** (`bitnet_adapter.cpp:29`):
```cpp
constexpr float kDecodeLut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
```
- `0b00` (0) → −1.0
- `0b01` (1) →  0.0
- `0b10` (2) → +1.0
- `0b11` (3) →  0.0 *(treated as 0)*

**Result: COMPATIBLE for valid inputs.** For values 0, 1, 2 the encodings
match exactly.  Difference only matters if the packer writes a 3 — which our
packer NEVER does (we always use round-then-clip-to-{-1,0,+1} during
quantization).  Confirmed by reviewing `pack_weights()` in
`src/bitlinear.cpp:97-106`.

This means we can use bitmamba's `UNPACK_LUT` directly with our `packed_weights`
buffer.  No re-encoding pass needed.

---

## The UNPACK_LUT generator

Given the encoding above, `UNPACK_LUT[byte]` returns a `uint32_t` where
each of the 4 bytes is the ternary value of the corresponding 2-bit slot:

```python
def make_unpack_lut():
    decode = [-1, 0, 1, 0]   # Slot value → int8 ternary (our encoding)
    lut = []
    for byte in range(256):
        slot0 = decode[(byte >> 0) & 0x3]
        slot1 = decode[(byte >> 2) & 0x3]
        slot2 = decode[(byte >> 4) & 0x3]
        slot3 = decode[(byte >> 6) & 0x3]
        # Pack as little-endian uint32: slot0 in lowest byte, slot3 in highest
        packed = (slot0 & 0xFF) | ((slot1 & 0xFF) << 8) | \
                 ((slot2 & 0xFF) << 16) | ((slot3 & 0xFF) << 24)
        lut.append(packed)
    return lut  # 256 entries × uint32 = 1 KB total
```

This will live as a `constexpr std::array<uint32_t, 256>` in
`include/bitnet_adapter.h`, computed at compile time so it has zero
runtime cost and fits comfortably in L1 cache (1 KB << 32-64 KB L1).

---

## Implementation plan

### Files to modify

| File | Change |
|------|--------|
| `include/bitnet_adapter.h` | Add `constexpr UNPACK_LUT_4WAY` array (generated by a `consteval` lambda) |
| `src/bitnet_adapter.cpp` | Add `gemm_158bit_lut_simd_impl` function (AVX2 + LUT) parallel to existing scalar |
| `src/bitnet_adapter.cpp` | Update `BitNetAdapter::gemm_158bit_lut` to dispatch: env var `NSOS_USE_LUT_SIMD=1` selects new path; default = existing scalar (zero regression) |
| `OXN/nsos/scripts/bench_lut_tmac.py` | Add 5th path `bitmamba_lut_simd` for A/B benchmark vs existing 4 paths |
| `OXN/nsos/tests/test_bitmamba_lut.cpp` (NEW) | Correctness: 100 random shapes, scalar vs SIMD outputs bit-exact within FP tolerance |

### Scope decisions

- **No changes to upstream packer** — `packed_weights` format stays identical
- **No changes to call sites** — `BitNetAdapter::gemm_158bit_lut` signature unchanged
- **Backward-compat default** — env var off = existing scalar behavior
- **Build-time UNPACK_LUT** — `constexpr` initialization, zero runtime cost

### Why dual-path (not replace)

1. **De-risk regression:** scalar path is validated by existing tests; we don't
   tear it out until SIMD path has proven equivalence over time
2. **CPU fallback:** scalar works on every CPU; SIMD path requires AVX2 detected
3. **A/B benchmark:** dual-path makes it trivial to compare on real workloads
4. **Easy rollback:** if SIMD shows numerical issue in some edge case, flip
   the env var off in production without redeploy

---

## Expected impact (extrapolated from bitmamba README)

bitmamba.cpp reports ~50 tokens/sec on Intel i3 CPUs for their 1B model.
Our current `gemm_158bit_i8` AVX2 path gets us solid throughput on the
unpacked format.  By moving to packed + LUT + SIMD, we expect:

- **Memory bandwidth:** 4× less DRAM traffic for weight loads → directly
  helps batch=1 inference (the dominant case for single-user contábil)
- **Per-element compute:** `_mm256_sign_epi8` + `_mm256_madd_epi16` are
  faster than the explicit int8 multiply + accumulate
- **Combined effect:** estimated **+20-40% throughput** vs current
  `gemm_158bit_i8`, measurable via the existing `bench_lut_tmac.py` once
  extended

These are estimates — the real number comes from the A/B benchmark.

---

## Status

- ✅ bitmamba kernel pattern fully read and understood
- ✅ Encoding compatibility verified (our format = bitmamba's format for valid inputs)
- ✅ Our existing scalar `gemm_158bit_lut` analyzed; established why it
      historically lost 20× to `gemm_158bit_i8`
- ✅ **UNPACK_LUT_4WAY constexpr table** in `include/bitnet_adapter.h`.
      256 × uint32_t = 1 KB, computed at compile time via `consteval`.
      Validated against Python reference: 5/5 test cases bit-exact.
- ✅ **gemm_158bit_lut_simd_impl** in `src/bitnet_adapter.cpp`. AVX2 path
      using 8 UNPACK_LUT loads → __m256i load → `_mm256_sign_epi8`
      ternary multiply → widen via `_mm256_cvtepi8_epi16` → accumulate
      via `_mm256_madd_epi16`. Tail loop for cols % 32 ≠ 0.
- ✅ **Env var dispatch** via `NSOS_USE_LUT_SIMD=1`. Lazy-cached on first
      call to avoid getenv() overhead on every GEMM. Default unset =
      preserves existing scalar behavior byte-for-byte (zero regression).
- ✅ **tests/test_bitmamba_lut.cpp**: 4 tests covering small deterministic,
      env routing self-consistency, scalar tail (cols not multiple of 32),
      and high zero density (90% sparse). Wired to CTest as
      `test_bitmamba_lut`.
- ✅ **scripts/bench_bitmamba_lut.py**: subprocess-per-measurement bench
      comparing scalar LUT vs SIMD LUT vs i8 baseline across 5 shapes
      representative of Mamba/Jamba layers. Writes JSON report. Each
      subprocess sets its own env var so the lazy cache picks the right
      path. Requires built nsos_ext to run; CLI validated standalone.

**ALL CHERRY-PICK #1 ARTIFACTS COMPLETE.**

Pending only: real-world A/B benchmark in Colab (requires built .so).
Expected outcome: 20-40% throughput improvement for batch=1 inference
(the dominant case for single-user Oxta Contábil) due to 4× less DRAM
bandwidth for weight loads + faster ternary multiplication via
`_mm256_sign_epi8`.
