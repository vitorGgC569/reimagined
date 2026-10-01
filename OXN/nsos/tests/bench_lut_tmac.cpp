// Microbench: LUT-TMAC kernel vs scalar reference.
//
// Times both paths on the same shapes / zero-density grid used in
// test_lut_tmac.cpp.  Reports wall time per iter + speedup.
//
// Honest scope: comparing LUT-TMAC against the SCALAR REFERENCE
// (`reference_gemm` — straightforward triple loop, no SIMD).  This
// tells us if the LUT + heat-map trick actually helps over a naive
// impl.  It does NOT tell us if LUT-TMAC beats the existing SIMD
// production path `gemm_158bit_ultra` — that comparison requires
// linking BitNetAdapter. That is intentionally outside this scalar
// microbenchmark; production promotion relies on the end-to-end model gate,
// where dispatch, packing and the complete runtime are measured together.
//
// Build via tests CMake target `bench_lut_tmac`.  Run with no args.

#include "../include/lut_tmac.h"
#include "../include/tensor.h"
#include "../include/bitnet_adapter.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

// Scalar reference — same as in tests/test_lut_tmac.cpp.
static void reference_gemm(const float* input, const uint8_t* packed,
                            const std::vector<float>& act_scales,
                            float weight_scale, float* out,
                            int B, int K, int N) {
    for (int b = 0; b < B; ++b) {
        const float* x = input + b * K;
        float* y = out + b * N;
        const float as = act_scales[static_cast<size_t>(b)];
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int idx = n * K + k;
                const int byte_idx = idx >> 2;
                const int shift = (idx & 0x3) << 1;
                const uint8_t code = (packed[byte_idx] >> shift) & 0x3u;
                if (code == 0u) {
                    acc -= x[k];
                } else if (code == 2u) {
                    acc += x[k];
                }
                // code == 1u → zero, no-op
            }
            y[n] = acc * as * weight_scale;
        }
    }
}

static void random_packed(std::vector<uint32_t>& packed, int N, int K,
                            int seed, float zero_density) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    const int total = N * K;
    const int word_count = (total + 15) / 16;
    packed.assign(static_cast<size_t>(word_count), 0u);
    uint8_t* p = reinterpret_cast<uint8_t*>(packed.data());
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < total; ++i) {
        uint8_t code;
        if (u(rng) < zero_density) {
            code = 1;
        } else {
            code = (u(rng) < 0.5f) ? 0 : 2;
        }
        const int byte_idx = i >> 2;
        const int shift = (i & 0x3) << 1;
        p[byte_idx] = static_cast<uint8_t>(p[byte_idx] | (code << shift));
    }
}

struct BenchResult {
    int B, K, N;
    float zero_density;
    double scalar_ms;        // naive triple loop
    double old_lut_ms;       // BitNetAdapter::gemm_158bit_lut (existing scalar LUT)
    double simd_i8_ms;       // BitNetAdapter::gemm_158bit_i8 (AVX2/VNNI SIMD prod path)
    double lut_tmac_ms;      // lut_tmac::gemm_158bit_lut_tmac (our new path)
    double speedup_vs_scalar;
    double speedup_vs_old_lut;
    double speedup_vs_simd;
    double heat_map_sparsity;
};

// Compute weight row sums needed by gemm_158bit_i8 (sum of int8 weights
// in each output row).  This is what BitLinear caches at weight-load time.
static void compute_row_sums(const std::vector<int8_t>& unpacked,
                              int N, int K,
                              std::vector<int32_t>& out_row_sums) {
    out_row_sums.assign(static_cast<size_t>(N), 0);
    for (int n = 0; n < N; ++n) {
        int32_t s = 0;
        const int8_t* row = unpacked.data() + static_cast<size_t>(n) * K;
        for (int k = 0; k < K; ++k) s += row[k];
        out_row_sums[static_cast<size_t>(n)] = s;
    }
}

static BenchResult bench_one(int B, int K, int N, float zero_density,
                              int iters, int warmup) {
    using namespace nsos;
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);

    // IMPORTANT: BitNetAdapter::gemm_158bit_i8 pre-quantizes its input
    // via std::round + clamp to [-127, 127] (bitnet_adapter.cpp:125-127).
    // For a fair bench, scalar/oldLUT/TMAC must receive the SAME quantized
    // values — otherwise we're comparing y_simd = round(x)*w against
    // y_ref = x*w which produces different outputs and meaningless timing.
    // Production callers (BitLinear::forward) always pre-quantize before
    // dispatch, so this matches the real workload.
    Tensor input({B, K}, Device::CPU);
    std::uniform_int_distribution<int> u_int(-100, 100);
    for (int i = 0; i < B * K; ++i) input.data()[i] = static_cast<float>(u_int(rng));

    std::vector<uint32_t> packed;
    random_packed(packed, N, K, 7 * B + 13 * K + N, zero_density);
    const uint8_t* packed_ptr = reinterpret_cast<const uint8_t*>(packed.data());

    // Unpacked int8 weights for the SIMD path
    std::vector<int8_t> unpacked;
    BitNetAdapter::unpack_weights_microsoft_style_to_i8(packed, N, K, unpacked);
    std::vector<int32_t> row_sums;
    compute_row_sums(unpacked, N, K, row_sums);

    std::vector<float> act_scales(static_cast<size_t>(B));
    for (int i = 0; i < B; ++i) act_scales[i] = 0.5f + u(rng) * 0.1f;
    const float weight_scale = 0.7f;

    std::vector<float> y_ref(static_cast<size_t>(B * N));
    Tensor y_old_lut({B, N}, Device::CPU);
    Tensor y_simd({B, N}, Device::CPU);
    Tensor y_tmac({B, N}, Device::CPU);

    // Precompute heat-map ONCE (would be cached at layer level in real model).
    const auto hm = lut_tmac::compute_heat_map(packed, N, K);
    const auto hm_stats = lut_tmac::heat_map_stats(hm, N, K);

    // ── Warmup all four paths ────────────────────────────────────
    for (int w = 0; w < warmup; ++w) {
        reference_gemm(input.data(), packed_ptr, act_scales, weight_scale,
                        y_ref.data(), B, K, N);
        BitNetAdapter::gemm_158bit_lut(input, packed, act_scales,
                                        weight_scale, y_old_lut);
        BitNetAdapter::gemm_158bit_i8(input, unpacked, row_sums, act_scales,
                                       weight_scale, y_simd);
        lut_tmac::gemm_158bit_lut_tmac(input, packed, hm, act_scales,
                                         weight_scale, y_tmac);
    }

    // Helper: time N iters of a callable
    auto time_ms = [iters](auto fn) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    const double scalar_ms = time_ms([&]() {
        reference_gemm(input.data(), packed_ptr, act_scales, weight_scale,
                        y_ref.data(), B, K, N);
    });
    const double old_lut_ms = time_ms([&]() {
        BitNetAdapter::gemm_158bit_lut(input, packed, act_scales,
                                        weight_scale, y_old_lut);
    });
    const double simd_ms = time_ms([&]() {
        BitNetAdapter::gemm_158bit_i8(input, unpacked, row_sums, act_scales,
                                       weight_scale, y_simd);
    });
    const double tmac_ms = time_ms([&]() {
        lut_tmac::gemm_158bit_lut_tmac(input, packed, hm, act_scales,
                                         weight_scale, y_tmac);
    });

    // Sanity: confirm outputs agree (within fp tolerance) so we're
    // benching the same computation.  Otherwise the speedup number
    // is meaningless.
    double max_diff_simd_vs_ref = 0.0;
    double max_diff_tmac_vs_ref = 0.0;
    for (int i = 0; i < B * N; ++i) {
        const float r = y_ref[static_cast<size_t>(i)];
        max_diff_simd_vs_ref = std::max(max_diff_simd_vs_ref,
                                         static_cast<double>(std::fabs(y_simd.data()[i] - r)));
        max_diff_tmac_vs_ref = std::max(max_diff_tmac_vs_ref,
                                         static_cast<double>(std::fabs(y_tmac.data()[i] - r)));
    }
    if (max_diff_simd_vs_ref > 1e-2 || max_diff_tmac_vs_ref > 1e-2) {
        std::printf("  WARNING: large output drift at (%d,%d,%d) zd=%.2f: "
                    "SIMD=%.4g, TMAC=%.4g\n",
                    B, K, N, zero_density,
                    max_diff_simd_vs_ref, max_diff_tmac_vs_ref);
    }

    BenchResult r;
    r.B = B; r.K = K; r.N = N;
    r.zero_density = zero_density;
    r.scalar_ms = scalar_ms;
    r.old_lut_ms = old_lut_ms;
    r.simd_i8_ms = simd_ms;
    r.lut_tmac_ms = tmac_ms;
    r.speedup_vs_scalar = scalar_ms / std::max(tmac_ms, 1e-9);
    r.speedup_vs_old_lut = old_lut_ms / std::max(tmac_ms, 1e-9);
    r.speedup_vs_simd    = simd_ms / std::max(tmac_ms, 1e-9);
    r.heat_map_sparsity = hm_stats.sparsity;
    return r;
}

}  // namespace

int main() {
    std::printf("=== Ternary GEMM 4-way bench (scalar / oldLUT / SIMD i8 / LUT-TMAC) ===\n");
    std::printf("  %-18s  %-9s  %-9s  %-9s  %-9s   %-7s %-7s %-7s\n",
                "shape (B,K,N) zd",
                "scalar", "oldLUT", "SIMD i8", "LUT-TMAC",
                "vs scal", "vs oLUT", "vs SIMD");
    std::printf("  %s\n", std::string(98, '-').c_str());

    struct Shape { int B, K, N; float zd; int iters; int warmup; };
    const std::vector<Shape> shapes = {
        {1, 128,  512,  0.30f, 200, 5},
        {1, 256,  1024, 0.30f, 100, 5},
        {1, 512,  2048, 0.30f, 50,  3},
        {1, 1024, 4096, 0.30f, 20,  2},
        {8, 256,  1024, 0.30f, 50,  3},
        {16, 512, 2048, 0.30f, 20,  2},
        {32, 256, 1024, 0.30f, 20,  2},
        {1, 1024, 4096, 0.00f, 20,  2},
        {1, 1024, 4096, 0.50f, 20,  2},
        {1, 1024, 4096, 0.80f, 20,  2},
        {1, 1024, 4096, 0.95f, 20,  2},
    };

    int wins_vs_simd = 0, losses_vs_simd = 0;
    double geomean_log_simd = 0.0;
    double geomean_log_old = 0.0;
    double geomean_log_scalar = 0.0;

    for (const auto& s : shapes) {
        auto r = bench_one(s.B, s.K, s.N, s.zd, s.iters, s.warmup);
        std::printf("  (%2d,%4d,%4d) zd=%.2f  %8.4f  %8.4f  %8.4f  %8.4f   %6.2fx %6.2fx %6.2fx\n",
                    r.B, r.K, r.N, r.zero_density,
                    r.scalar_ms, r.old_lut_ms, r.simd_i8_ms, r.lut_tmac_ms,
                    r.speedup_vs_scalar, r.speedup_vs_old_lut, r.speedup_vs_simd);
        if (r.speedup_vs_simd >= 1.0) ++wins_vs_simd;
        else                           ++losses_vs_simd;
        geomean_log_simd   += std::log(r.speedup_vs_simd);
        geomean_log_old    += std::log(r.speedup_vs_old_lut);
        geomean_log_scalar += std::log(r.speedup_vs_scalar);
    }
    const double n = static_cast<double>(shapes.size());

    std::printf("\n");
    std::printf("  Summary across %zu shapes (LUT-TMAC vs each baseline):\n", shapes.size());
    std::printf("    geo-mean speedup vs scalar       : %.2fx\n", std::exp(geomean_log_scalar / n));
    std::printf("    geo-mean speedup vs old LUT      : %.2fx\n", std::exp(geomean_log_old    / n));
    std::printf("    geo-mean speedup vs SIMD i8 prod : %.2fx  (THE GATE — production path)\n",
                std::exp(geomean_log_simd / n));
    std::printf("    wins / losses vs SIMD i8         : %d / %d\n",
                wins_vs_simd, losses_vs_simd);
    std::printf("\n");
    std::printf("  Interpretation:\n");
    std::printf("    vs SIMD < 1.0x  -> LUT-TMAC slower than production; do NOT deploy as default.\n");
    std::printf("    vs SIMD 1.0-1.3x -> marginal; deploy as opt-in only on hosts w/o AVX2.\n");
    std::printf("    vs SIMD > 1.3x  -> clear win; promote to default with env-var rollback.\n");
    return 0;
}
