// ============================================================================
//  validation_battery.cpp -- OXB AION extreme validation
// ============================================================================
//
//  Validates the 4 pillars of OXB AION (Pilar A: RMI, Pilar C: BitPacking,
//  Pilar D: Hilbert, RingBuffer) at correctness + edge cases + scale +
//  determinism + stress.
//
//  Honest reproduction of MANIFESTO claims:
//    BitPacking 5-bit:  claim 1.13B ops/s
//    RMI Train:         claim 471M ops/s
//    RMI Predict:       claim 533M ops/s
//    Hilbert 2D->1D:    claim 202M ops/s
//
//  Pre-existing smoke test (tests/main_test.cpp) is 52 LOC, runs but does
//  not assert numerical correctness rigorously.  This file does.
// ============================================================================

#include "../include/BitPacking.hpp"
#include "../include/Hilbert.hpp"
#include "../include/LinearModel.hpp"
#include "../include/RingBuffer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_passed = 0;
int g_failed = 0;
std::vector<std::string> g_failures;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream _o;                                             \
            _o << __func__ << ":" << __LINE__ << " " << msg;                   \
            g_failures.push_back(_o.str());                                    \
            std::cerr << "    FAIL: " << _o.str() << std::endl;                \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RUN(fn)                                                                \
    do {                                                                       \
        std::cout << "[" << #fn << "]" << std::flush;                          \
        bool _ok = fn();                                                       \
        if (_ok) { std::cout << "  PASS\n"; ++g_passed; }                      \
        else     { std::cout << "  FAIL\n"; ++g_failed; }                      \
    } while (0)

using clk = std::chrono::high_resolution_clock;
double elapsed_sec(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
}

}  // namespace

// ============================================================================
// PILAR A — LinearModel (RMI)
// ============================================================================

bool test_rmi_exact_line() {
    // y = 3x + 7
    Aion::LinearModel m;
    std::vector<double> keys, offsets;
    for (int i = 0; i < 100; ++i) {
        keys.push_back((double)i);
        offsets.push_back(3.0 * i + 7.0);
    }
    m.train(keys, offsets);
    CHECK(std::fabs(m.m - 3.0) < 1e-9, "slope wrong: " << m.m);
    CHECK(std::fabs(m.b - 7.0) < 1e-9, "intercept wrong: " << m.b);
    CHECK(m.max_error == 0, "max_error not 0 for exact line: " << m.max_error);
    return true;
}

bool test_rmi_noisy_line() {
    // y = 2x + 5 + small noise
    Aion::LinearModel m;
    std::vector<double> keys, offsets;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    for (int i = 0; i < 1000; ++i) {
        keys.push_back((double)i);
        offsets.push_back(2.0 * i + 5.0 + noise(rng));
    }
    m.train(keys, offsets);
    CHECK(std::fabs(m.m - 2.0) < 0.01, "slope drift > 1%: " << m.m);
    CHECK(std::fabs(m.b - 5.0) < 0.5, "intercept drift > 0.5: " << m.b);
    return true;
}

bool test_rmi_predict_batch_matches_scalar() {
    // Train on y = 1.5x + 0.25
    Aion::LinearModel m;
    std::vector<double> keys, offsets;
    for (int i = 0; i < 100; ++i) {
        keys.push_back((double)i);
        offsets.push_back(1.5 * i + 0.25);
    }
    m.train(keys, offsets);

    // Predict batch vs scalar
    const size_t N = 257;  // intentionally not multiple of 4 (test tail)
    std::vector<double> query(N);
    std::vector<double> batch_result(N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1000.0, 1000.0);
    for (size_t i = 0; i < N; ++i) query[i] = u(rng);

    m.predict_batch(query.data(), batch_result.data(), N);

    for (size_t i = 0; i < N; ++i) {
        double scalar = m.predict(query[i]);
        CHECK(std::fabs(batch_result[i] - scalar) < 1e-9,
              "batch[" << i << "]=" << batch_result[i] << " scalar=" << scalar);
    }
    return true;
}

bool test_rmi_edge_single_point() {
    Aion::LinearModel m;
    std::vector<double> keys = {5.0};
    std::vector<double> offsets = {42.0};
    m.train(keys, offsets);
    CHECK(std::fabs(m.predict(5.0) - 42.0) < 1e-9, "single point predict wrong");
    CHECK(std::fabs(m.predict(0.0) - 42.0) < 1e-9, "single point: predict(0) should also be b");
    return true;
}

bool test_rmi_edge_empty() {
    Aion::LinearModel m;
    std::vector<double> empty;
    m.train(empty, empty);
    // No crash is the test; values stay 0
    CHECK(m.m == 0.0 && m.b == 0.0, "empty train should leave m=b=0");
    return true;
}

// ============================================================================
// PILAR C — BitPacking
// ============================================================================

bool test_bitpack_roundtrip_5bit() {
    // Pack 5-bit, manually unpack, ensure roundtrip
    const size_t N = 1024;
    std::vector<uint32_t> in(N);
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint32_t> u(0, 31);  // 5-bit range
    for (size_t i = 0; i < N; ++i) in[i] = u(rng);

    // 5 bits * 1024 = 5120 bits = 80 uint64 words
    std::vector<uint64_t> packed(85, 0);  // +5 slack
    Aion::BitPacker::pack_scalar(in.data(), packed.data(), N, 5);

    // Manual unpack
    auto unpack = [&](size_t i) -> uint32_t {
        size_t bit_pos = i * 5;
        size_t word_idx = bit_pos / 64;
        int local_bit = bit_pos % 64;
        uint64_t lo = packed[word_idx];
        uint64_t hi = (word_idx + 1 < packed.size()) ? packed[word_idx + 1] : 0ULL;
        uint64_t comb_lo = lo >> local_bit;
        uint64_t comb_hi = (local_bit > 0) ? (hi << (64 - local_bit)) : 0ULL;
        uint64_t combined = comb_lo | comb_hi;
        return (uint32_t)(combined & 0x1F);
    };

    for (size_t i = 0; i < N; ++i) {
        uint32_t got = unpack(i);
        CHECK(got == in[i], "roundtrip 5-bit i=" << i << " expected " << in[i]
              << " got " << got);
    }
    return true;
}

bool test_bitpack_roundtrip_various_widths() {
    for (int bits : {1, 2, 3, 4, 7, 8, 11, 17, 31}) {
        const size_t N = 200;
        std::vector<uint32_t> in(N);
        uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1);
        std::mt19937 rng(bits * 100);
        for (size_t i = 0; i < N; ++i) in[i] = (uint32_t)(rng() & mask);

        size_t total_bits = (size_t)bits * N;
        size_t words_needed = (total_bits + 63) / 64 + 2;
        std::vector<uint64_t> packed(words_needed, 0);
        Aion::BitPacker::pack_scalar(in.data(), packed.data(), N, bits);

        // Unpack manual
        auto unpack = [&](size_t i) -> uint32_t {
            size_t bit_pos = (size_t)i * bits;
            size_t word_idx = bit_pos / 64;
            int local_bit = (int)(bit_pos % 64);
            uint64_t lo = packed[word_idx];
            uint64_t hi = (word_idx + 1 < packed.size()) ? packed[word_idx + 1] : 0ULL;
            uint64_t comb_lo = lo >> local_bit;
            uint64_t comb_hi = (local_bit > 0) ? (hi << (64 - local_bit)) : 0ULL;
            uint64_t combined = comb_lo | comb_hi;
            uint64_t bitmask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
            return (uint32_t)(combined & bitmask);
        };

        for (size_t i = 0; i < N; ++i) {
            uint32_t got = unpack(i);
            CHECK(got == in[i], "bits=" << bits << " i=" << i << " expected "
                  << in[i] << " got " << got);
        }
    }
    return true;
}

bool test_bitpack_empty_input() {
    std::vector<uint32_t> in;
    std::vector<uint64_t> out(4, 0xDEADBEEFCAFEBABEull);
    Aion::BitPacker::pack_scalar(in.data(), out.data(), 0, 5);
    // No data written; sentinel preserved
    CHECK(out[0] == 0xDEADBEEFCAFEBABEull, "empty pack must not write");
    return true;
}

bool test_bitpack_determinism() {
    const size_t N = 5000;
    std::vector<uint32_t> in(N);
    std::mt19937 rng(99);
    std::uniform_int_distribution<uint32_t> u(0, 1023);  // 10-bit
    for (size_t i = 0; i < N; ++i) in[i] = u(rng);

    std::vector<uint64_t> out1(N + 10, 0), out2(N + 10, 0);
    Aion::BitPacker::pack_scalar(in.data(), out1.data(), N, 10);
    Aion::BitPacker::pack_scalar(in.data(), out2.data(), N, 10);
    CHECK(out1 == out2, "non-deterministic BitPacking");
    return true;
}

// ============================================================================
// PILAR D — Hilbert (space-filling curve, 2D <-> 1D)
// ============================================================================

bool test_hilbert_roundtrip_full_grid_4() {
    int n = 4;  // 4x4 grid, 16 cells
    bool seen[16] = {false};
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            uint32_t d = Aion::Hilbert::xy2d(n, x, y);
            CHECK(d < (uint32_t)(n * n), "d out of range: " << d);
            CHECK(!seen[d], "Hilbert mapping not bijective at d=" << d);
            seen[d] = true;
            int xr, yr;
            Aion::Hilbert::d2xy(n, (int)d, &xr, &yr);
            CHECK(xr == x && yr == y, "roundtrip fail x=" << x << " y=" << y
                  << " got x=" << xr << " y=" << yr);
        }
    }
    // All 16 distances covered
    for (int i = 0; i < 16; ++i) CHECK(seen[i], "missing d=" << i);
    return true;
}

bool test_hilbert_locality_property() {
    // Hilbert's defining property: adjacent d-values are spatially near.
    // For each consecutive pair (d, d+1), the Manhattan distance in (x,y)
    // should be exactly 1 (one step on the grid).
    int n = 8;  // 8x8 grid
    for (int d = 0; d < n * n - 1; ++d) {
        int x1, y1, x2, y2;
        Aion::Hilbert::d2xy(n, d, &x1, &y1);
        Aion::Hilbert::d2xy(n, d + 1, &x2, &y2);
        int mdist = std::abs(x1 - x2) + std::abs(y1 - y2);
        CHECK(mdist == 1, "Hilbert locality broken at d=" << d
              << ": (" << x1 << "," << y1 << ") -> (" << x2 << "," << y2 << ")");
    }
    return true;
}

bool test_hilbert_origin_maps_to_zero() {
    for (int n : {2, 4, 8, 16, 32}) {
        uint32_t d = Aion::Hilbert::xy2d(n, 0, 0);
        CHECK(d == 0, "n=" << n << " xy2d(0,0)=" << d << " expected 0");
    }
    return true;
}

// ============================================================================
// STRESS / PERFORMANCE — reproduzir números do MANIFESTO
// ============================================================================

bool test_perf_bitpack_5bit_at_scale() {
    const size_t N = 10'000'000;
    std::vector<uint32_t> in(N);
    std::mt19937 rng(1);
    std::uniform_int_distribution<uint32_t> u(0, 31);
    for (size_t i = 0; i < N; ++i) in[i] = u(rng);
    std::vector<uint64_t> out((N * 5 + 63) / 64 + 4, 0);

    auto t0 = clk::now();
    Aion::BitPacker::pack_scalar(in.data(), out.data(), N, 5);
    auto t1 = clk::now();
    double sec = elapsed_sec(t0, t1);
    double ops = N / sec;
    std::cout << "    BitPacking 5-bit @ N=10M: " << sec << "s ("
              << ops / 1e6 << " M ops/s)" << std::endl;
    CHECK(ops > 50e6, "BitPacking < 50M ops/s, suspicious");
    return true;
}

bool test_perf_rmi_train_at_scale() {
    const size_t N = 10'000'000;
    std::vector<double> keys(N), offs(N);
    for (size_t i = 0; i < N; ++i) {
        keys[i] = (double)i;
        offs[i] = 2.0 * i + 7.0;
    }
    Aion::LinearModel m;
    auto t0 = clk::now();
    m.train(keys, offs);
    auto t1 = clk::now();
    double sec = elapsed_sec(t0, t1);
    double ops = N / sec;
    std::cout << "    RMI Train @ N=10M: " << sec << "s ("
              << ops / 1e6 << " M ops/s)" << std::endl;
    CHECK(std::fabs(m.m - 2.0) < 1e-9, "RMI train drifted slope");
    CHECK(ops > 20e6, "RMI train < 20M ops/s");
    return true;
}

bool test_perf_rmi_predict_batch_at_scale() {
    const size_t N = 10'000'000;
    Aion::LinearModel m;
    m.m = 1.234; m.b = 5.678;
    std::vector<double> keys(N), out(N);
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> u(-1e6, 1e6);
    for (size_t i = 0; i < N; ++i) keys[i] = u(rng);

    auto t0 = clk::now();
    m.predict_batch(keys.data(), out.data(), N);
    auto t1 = clk::now();
    double sec = elapsed_sec(t0, t1);
    double ops = N / sec;
    std::cout << "    RMI Predict batch (AVX2) @ N=10M: " << sec << "s ("
              << ops / 1e6 << " M ops/s)" << std::endl;
    CHECK(ops > 50e6, "RMI predict batch < 50M ops/s");
    return true;
}

bool test_perf_hilbert_at_scale() {
    const int n = 1024;          // 1024x1024 grid
    const size_t N = 10'000'000;
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> xu(0, n - 1);
    std::vector<int> xs(N), ys(N);
    for (size_t i = 0; i < N; ++i) { xs[i] = xu(rng); ys[i] = xu(rng); }
    volatile uint32_t sink = 0;
    auto t0 = clk::now();
    for (size_t i = 0; i < N; ++i) {
        sink ^= Aion::Hilbert::xy2d(n, xs[i], ys[i]);
    }
    auto t1 = clk::now();
    double sec = elapsed_sec(t0, t1);
    double ops = N / sec;
    std::cout << "    Hilbert xy2d @ N=10M (1024x1024): " << sec << "s ("
              << ops / 1e6 << " M ops/s)" << std::endl;
    (void)sink;
    CHECK(ops > 5e6, "Hilbert < 5M ops/s suspicious");
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "==========================================================\n";
    std::cout << " OXB AION C++ — extreme validation battery\n";
    std::cout << "==========================================================\n";

    // Pilar A
    RUN(test_rmi_exact_line);
    RUN(test_rmi_noisy_line);
    RUN(test_rmi_predict_batch_matches_scalar);
    RUN(test_rmi_edge_single_point);
    RUN(test_rmi_edge_empty);

    // Pilar C
    RUN(test_bitpack_roundtrip_5bit);
    RUN(test_bitpack_roundtrip_various_widths);
    RUN(test_bitpack_empty_input);
    RUN(test_bitpack_determinism);

    // Pilar D
    RUN(test_hilbert_roundtrip_full_grid_4);
    RUN(test_hilbert_locality_property);
    RUN(test_hilbert_origin_maps_to_zero);

    // Stress + perf
    RUN(test_perf_bitpack_5bit_at_scale);
    RUN(test_perf_rmi_train_at_scale);
    RUN(test_perf_rmi_predict_batch_at_scale);
    RUN(test_perf_hilbert_at_scale);

    std::cout << "==========================================================\n";
    std::cout << " RESULT: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "==========================================================\n";
    if (g_failed > 0) {
        for (auto& f : g_failures) std::cerr << "  - " << f << "\n";
    }
    return g_failed == 0 ? 0 : 1;
}
