# Release Notes — Cherry-picks #1 + #2 + #4

> Branch: `feature/cherry-picks-3`
> Safety tag (rollback target): `oxta-pre-cherrypicks-2026-05-24`
> Final commit: see `git log --oneline` on the branch
> Status: **READY FOR COLAB ABLATION + REAL TRAINING**

---

## Summary

Three independent improvements integrated into NSOS, each opt-in and
zero-regression by default.  Total: **8 commits, ~1500 lines added**,
0 lines of existing functional code removed.

| Cherry-pick | Source | What it does | How to enable |
|------------|--------|--------------|---------------|
| **#1 bitmamba LUT** | `external_refs/repos/bitmamba.cpp` | UNPACK_LUT + AVX2 SIMD for `gemm_158bit_lut`.  Expected +20-40% throughput on batch=1 ternary GEMM | `NSOS_USE_LUT_SIMD=1` env var |
| **#2 Slender-Mamba** | Yu et al. 2025 COLING + `Slender-Mamba` repo | Head-to-toe 1.58-bit quantization on embedding layer (paper Sec 3.3, eqs 7-13).  Expected ~40% disk size reduction | `engine.model.embedding.set_slender_quantization(true)` |
| **#4 Nemotron K·m** | NVIDIA Nemotron 3 Super technical report | Configurable MoE expert FFN intermediate dim (`m`), enabling K·m invariant ablation | `ModelConfig.moe_expert_hidden_dim = N` (default 0 = `d_model*4`) |

---

## Files changed

### Code (additive only)
- `OXN/nsos/include/embedding.h` (+47 lines) — Slender flag + cache fields + helper declarations
- `OXN/nsos/src/embedding.cpp` (+185 lines) — `ensure_slender_cache_()`, `slender_forward_cpu_()`, dispatch, STE backward docs
- `OXN/nsos/include/jamba.h` (+8 lines) — `JambaBlock` ctor accepts `configured_expert_hidden_dim`
- `OXN/nsos/src/jamba.cpp` (+15 lines) — split `default_ffn_hidden` vs `moe_expert_hidden` (scope fix from review)
- `OXN/nsos/include/nsos_config.h` (+10 lines) — `ModelConfig::moe_expert_hidden_dim` field
- `OXN/nsos/include/bitnet_adapter.h` (+42 lines) — `consteval` UNPACK_LUT_4WAY table
- `OXN/nsos/src/bitnet_adapter.cpp` (+180 lines) — `gemm_158bit_lut_simd_impl`, env dispatch
- `OXN/nsos/src/bindings.cpp` (+4 lines) — expose `moe_expert_hidden_dim` to Python

### New tests (2 binaries, 8 test functions)
- `OXN/nsos/tests/test_slender_embedding.cpp` — cache lifecycle, round-trip, STE routing, edge cases
- `OXN/nsos/tests/test_bitmamba_lut.cpp` — small deterministic, env routing, scalar tail, sparse weights

### New tooling
- `OXN/nsos/scripts/bench_bitmamba_lut.py` — subprocess-per-measurement A/B benchmark

### Profiles (`OXN/nsos/scripts/train_curriculum.py`)
- `hybrid_v11_colab_t4_80m_km_A` — K=4, m=1280, N=8 (K·m=5120)
- `hybrid_v11_colab_t4_80m_km_B` — K=2, m=2560, N=16 (K·m=5120)
- `hybrid_v11_colab_t4_80m_km_C` — K=4, m=1280, N=16 (K·m=5120)

### Documentation
- `OXN/nsos/docs/SLENDER_INTEGRATION.md` (242 lines) — Phase plan, fórmulas, status
- `OXN/nsos/docs/NEMOTRON_KM_INTEGRATION.md` (253 lines) — 5 principles, current vs proposed config
- `OXN/nsos/docs/BITMAMBA_LUT_INTEGRATION.md` (180 lines) — kernel design, encoding compatibility
- `OXN/nsos/docs/RELEASE_NOTES_CHERRYPICKS.md` (this file)

---

## Test status

**CTest suite on Windows MSVC Release: 32/34 passing (94%).**

The 2 failures (`test_layer_audit`, `test_circuit_api_pipeline`) are
**pre-existing** — verified by checking out the safety tag
`oxta-pre-cherrypicks-2026-05-24` and re-running: both fail there too.
These tests have known issues unrelated to this branch.

**Tests added by this branch (both passing):**
- `test_slender_embedding` — 4/4 tests pass in 0.08s
- `test_bitmamba_lut` — 4/4 tests pass in 0.07s

**Compilation:** all touched files compile cleanly with `cl.exe /std:c++20
/EHsc /arch:AVX2` (zero errors, zero warnings).

---

## What to do in Colab

### Step 1 — Pull this branch
```bash
cd /content
git clone https://github.com/<YOUR_REPO>.git nsos
cd nsos
git checkout feature/cherry-picks-3
```

### Step 2 — Build
```bash
cmake -S OXN/nsos -B OXN/nsos/build-mvp \
    -DCMAKE_BUILD_TYPE=Release \
    -DNSOS_ENABLE_CUDA=ON \
    -DNSOS_BUILD_PYTHON=ON \
    -DNSOS_BUILD_TESTS=ON \
    -DNSOS_BUILD_CLI=OFF \
    -DNSOS_BUILD_API=OFF \
    -DNSOS_BUILD_OXTAMEM=OFF
cmake --build OXN/nsos/build-mvp --config Release -j
```

### Step 3 — Validate new tests pass
```bash
ctest --test-dir OXN/nsos/build-mvp -C Release \
    -R 'test_slender_embedding|test_bitmamba_lut' --output-on-failure
```

### Step 4 — Run K·m ablation (cherry-pick #4)
Edit `train_curriculum.py` invocation to use one of:
- `--profile hybrid_v11_colab_t4_80m`     (baseline)
- `--profile hybrid_v11_colab_t4_80m_km_A` (K=4, m=1280)
- `--profile hybrid_v11_colab_t4_80m_km_B` (K=2, m=2560, N=16)
- `--profile hybrid_v11_colab_t4_80m_km_C` (K=4, m=1280, N=16)

Run all 4 with `--override-phase-steps 15` for ~10-15min each.  Pick
the winner by lowest final-phase eval loss.

### Step 5 — Bench bitmamba LUT (cherry-pick #1)
```bash
python OXN/nsos/scripts/bench_bitmamba_lut.py \
    --build-dir OXN/nsos/build-mvp/Release \
    --iters 50 \
    --report-path /content/drive/MyDrive/bench_bitmamba_lut.json
```

### Step 6 — Enable Slender (cherry-pick #2) in training
Modify training script to call `model.embedding.set_slender_quantization(true)`
after model construction, then run a short smoke (200 steps) to verify
loss still descends.  If loss diverges or stays flat, debug; otherwise
adopt for the real long run.

---

## Rollback plan

If any cherry-pick causes problems in production:

```bash
# Disable all three cherry-picks at runtime (no rebuild needed):
unset NSOS_USE_LUT_SIMD                         # disable cherry-pick #1
# Don't call set_slender_quantization(true)     # disable cherry-pick #2
# Don't set ModelConfig.moe_expert_hidden_dim   # disable cherry-pick #4
```

If you need a full code revert:

```bash
git checkout oxta-pre-cherrypicks-2026-05-24
```

All three cherry-picks were designed to be **default-off** so this is a
zero-risk deployment.

---

## Acknowledgments

Cherry-pick #1 ports the SIMD+LUT pattern from
[bitmamba.cpp](https://github.com/Zhayr1/bitmamba.cpp) (MIT license,
Zhayr1).  Cherry-pick #2 implements the algorithm from Yu et al. 2025
"Slender-Mamba: Fully Quantized Mamba in 1.58 Bits From Head to Toe"
(COLING 2025, Apache 2.0 reference impl).  Cherry-pick #4 applies the
K·m invariant from NVIDIA's Nemotron 3 Super technical report
(April 2026).

Built with care for Oxta — first Brazilian LLM from zero, edge-first,
sovereign by construction.
