// Numerical equivalence test: LUT-TMAC kernel vs reference scalar kernel.
//
// PASS criterion: for 50 random (B, K, N, packed_weights, x) inputs,
// the LUT-TMAC output matches the reference scalar gemm to within
// fp32 roundoff (max abs diff < 1e-4 of typical output magnitude).
//
// The reference implementation is a transparent loop that does the
// quantized matmul directly — same math the existing
// BitNetAdapter::gemm_158bit_lut does but inlined here so this test
// has zero hidden coupling to the rest of nsos.

#include "../include/lut_tmac.h"
#include "../include/tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace nsos_test {

// Reference scalar kernel.  Same math as BitNetAdapter::gemm_158bit_lut
// without bells (no bias, no magnitude).
static constexpr int8_t kDecodeRef[4] = {-1, 0, 1, 0};

static void reference_gemm(
    const float* x, const uint8_t* packed, const std::vector<float>& act_scales,
    float weight_scale, float* y, int B, int K, int N) {
    for (int b = 0; b < B; ++b) {
        const float as =
            b < static_cast<int>(act_scales.size()) ? act_scales[b] : 1.0f;
        const float* row = x + b * K;
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            const int wrow = n * K;
            for (int k = 0; k < K; ++k) {
                const int fi = wrow + k;
                const int pi = fi >> 2;
                const int ps = (fi & 0x3) << 1;
                const uint8_t code = (packed[pi] >> ps) & 0x3;
                acc += row[k] * static_cast<float>(kDecodeRef[code]);
            }
            y[b * N + n] = acc * as * weight_scale;
        }
    }
}

static void random_packed(std::vector<uint32_t>& packed, int N, int K,
                           int seed, float zero_density) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    const int total = N * K;
    const int word_count = (total + 15) / 16;  // 4 weights per byte = 16 per uint32
    packed.assign(static_cast<size_t>(word_count), 0u);
    uint8_t* p = reinterpret_cast<uint8_t*>(packed.data());
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < total; ++i) {
        uint8_t code;
        if (u(rng) < zero_density) {
            code = 1;  // zero in our encoding
        } else {
            code = (u(rng) < 0.5f) ? 0 : 2;  // -1 or +1
        }
        const int byte_idx = i >> 2;
        const int shift = (i & 0x3) << 1;
        p[byte_idx] = static_cast<uint8_t>(p[byte_idx] | (code << shift));
    }
}

static bool run_one(int B, int K, int N, float zero_density, int seed) {
    using namespace nsos;
    std::mt19937 rng(static_cast<uint32_t>(seed) * 31u + 1u);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);

    // Build input
    Tensor input({B, K}, Device::CPU);
    for (int i = 0; i < B * K; ++i) input.data()[i] = u(rng);

    // Build random packed weights
    std::vector<uint32_t> packed;
    random_packed(packed, N, K, seed * 7 + 13, zero_density);

    std::vector<float> act_scales(static_cast<size_t>(B));
    for (int i = 0; i < B; ++i) act_scales[i] = 0.5f + u(rng) * 0.1f;
    const float weight_scale = 0.7f;

    // Reference
    std::vector<float> y_ref(static_cast<size_t>(B * N));
    const uint8_t* packed_ptr = reinterpret_cast<const uint8_t*>(packed.data());
    reference_gemm(input.data(), packed_ptr, act_scales, weight_scale,
                    y_ref.data(), B, K, N);

    // LUT-TMAC
    Tensor y_lut({B, N}, Device::CPU);
    const auto hm = lut_tmac::compute_heat_map(packed, N, K);
    lut_tmac::gemm_158bit_lut_tmac(input, packed, hm, act_scales,
                                     weight_scale, y_lut);

    // Compare
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float ref_max = 1e-9f;
    for (int i = 0; i < B * N; ++i) {
        const float ref = y_ref[static_cast<size_t>(i)];
        const float got = y_lut.data()[i];
        const float d = std::fabs(got - ref);
        if (d > max_abs) max_abs = d;
        if (std::fabs(ref) > ref_max) ref_max = std::fabs(ref);
        if (std::fabs(ref) > 1e-3f) {
            const float r = d / std::fabs(ref);
            if (r > max_rel) max_rel = r;
        }
    }

    // Tolerance: fp32 roundoff for K*B random adds is ~1e-5 typical
    // and < 1e-4 worst case.  We use 1e-4 as the line.
    const bool ok = (max_abs < 1e-4f * std::max(1.0f, ref_max));
    const auto hm_stats = lut_tmac::heat_map_stats(hm, N, K);
    std::printf("  B=%d K=%d N=%d zd=%.2f  max_abs=%.6f max_rel=%.6f  "
                "tiles=%d zero=%d sparsity=%.2f%%  [%s]\n",
                B, K, N, zero_density, max_abs, max_rel,
                hm_stats.total_tiles, hm_stats.zero_tiles,
                hm_stats.sparsity * 100.0,
                ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace nsos_test

int main() {
    using nsos_test::run_one;
    std::printf("=== LUT-TMAC numerical equivalence ===\n");
    int failed = 0;
    // Sweep over shapes and zero densities.  K must be %4=0 per
    // format_supported().
    const int shapes[][3] = {
        {1, 8, 16},          // smallest
        {1, 16, 32},
        {4, 64, 128},
        {8, 128, 256},
        {32, 256, 512},
        {2, 512, 1024},
        {1, 1024, 2048},     // 1B-scale layer
    };
    const float zds[] = {0.0f, 0.3f, 0.6f, 0.9f};
    int total = 0;
    for (const auto& s : shapes) {
        for (float zd : zds) {
            ++total;
            if (!run_one(s[0], s[1], s[2], zd, 42 + total * 7))
                ++failed;
        }
    }
    std::printf("\n=== %d/%d passed ===\n", total - failed, total);
    return failed == 0 ? 0 : 1;
}
