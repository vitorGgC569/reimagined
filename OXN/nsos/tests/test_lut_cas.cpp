// test_lut_cas.cpp — round-trip correctness for content-addressed
// ternary weight storage.
//
// For each (rows, cols, k) shape:
//   1. Generate random packed bytes
//   2. encode → CasPack
//   3. decode → bytes
//   4. assert byte-for-byte equal to original
//
// Plus multi-matrix encode/decode round-trip + file I/O round-trip.

#include "../include/lut_cas.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

static void random_packed(std::vector<uint8_t>& packed, int rows, int cols,
                            uint32_t seed, float zero_density) {
    std::mt19937 rng(seed);
    const int total = rows * cols;
    const int byte_count = (total + 3) / 4;
    packed.assign(static_cast<size_t>(byte_count), 0u);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < total; ++i) {
        uint8_t code;
        if (u(rng) < zero_density) {
            code = 1u;
        } else {
            code = (u(rng) < 0.5f) ? 0u : 2u;
        }
        const int byte_idx = i >> 2;
        const int shift = (i & 0x3) << 1;
        packed[byte_idx] = static_cast<uint8_t>(packed[byte_idx] | (code << shift));
    }
}

static bool roundtrip_single(int rows, int cols, nsos::lut_cas::TileK k,
                              float zd, uint32_t seed) {
    std::vector<uint8_t> packed;
    random_packed(packed, rows, cols, seed, zd);

    auto cas = nsos::lut_cas::encode(packed, rows, cols, k);
    auto back = nsos::lut_cas::decode(cas);

    if (back.size() != packed.size()) {
        std::printf("  FAIL: roundtrip size mismatch %zu vs %zu\n",
                    back.size(), packed.size());
        return false;
    }
    for (size_t i = 0; i < packed.size(); ++i) {
        if (back[i] != packed[i]) {
            std::printf("  FAIL: byte %zu differs (%u vs %u)\n",
                        i, back[i], packed[i]);
            return false;
        }
    }
    const auto s = nsos::lut_cas::stats(cas);
    std::printf("  PASS  rows=%4d cols=%5d  k=%d zd=%.2f  "
                "unique=%6zu / %6zu tiles  ratio=%.3f  avg_occ=%.1fx\n",
                rows, cols, static_cast<int>(k), zd,
                s.unique_tiles, s.total_tiles, s.compression_ratio,
                s.avg_tile_occurrence);
    return true;
}

}  // namespace

int main() {
    std::printf("=== lut_cas single-matrix round-trip ===\n");
    int ok = 0, total = 0;
    const std::vector<std::pair<int, int>> shapes = {
        {128, 256}, {256, 512}, {512, 1024}, {1024, 2048},
        {512, 4096}, {1024, 8192},
    };
    for (auto [rows, cols] : shapes) {
        for (auto k : {nsos::lut_cas::TileK::K4, nsos::lut_cas::TileK::K8}) {
            for (float zd : {0.0f, 0.3f, 0.6f, 0.9f}) {
                ++total;
                if (roundtrip_single(rows, cols, k, zd, 42u + total)) ++ok;
            }
        }
    }
    std::printf("  single-matrix: %d/%d passed\n\n", ok, total);

    std::printf("=== lut_cas multi-matrix dedup (simulating MoE-8 experts) ===\n");
    // Simulate 8 experts each with 4 linear layers of shape (rows, cols).
    // Each "expert" shares a similar weight distribution → high cross-dedup.
    std::vector<std::vector<uint8_t>> matrices;
    std::vector<std::pair<int, int>> rows_cols;
    const int n_experts = 8;
    const int layers_per_expert = 4;
    const int rows = 512;
    const int cols = 1024;
    for (int e = 0; e < n_experts; ++e) {
        for (int l = 0; l < layers_per_expert; ++l) {
            std::vector<uint8_t> packed;
            // All experts share seed structure so cross-dedup is high
            random_packed(packed, rows, cols, 100u + e * 7 + l, 0.4f);
            matrices.push_back(std::move(packed));
            rows_cols.emplace_back(rows, cols);
        }
    }
    auto multi = nsos::lut_cas::encode_multi(matrices, rows_cols,
                                                nsos::lut_cas::TileK::K8);
    auto stats_multi = nsos::lut_cas::stats(multi);
    std::printf("  matrices=%zu  total_tiles=%zu  unique_tiles=%zu\n",
                multi.matrices.size(), stats_multi.total_tiles,
                stats_multi.unique_tiles);
    std::printf("  raw_bytes=%zu  cas_bytes=%zu  compression=%.3f  "
                "avg_occ=%.1fx  top10_coverage=%.1f%%\n",
                stats_multi.raw_bytes, stats_multi.cas_bytes,
                stats_multi.compression_ratio,
                stats_multi.avg_tile_occurrence,
                stats_multi.dict_top10_coverage * 100.0);

    // Round-trip
    auto decoded = nsos::lut_cas::decode_multi(multi);
    bool multi_ok = (decoded.size() == matrices.size());
    if (multi_ok) {
        for (size_t i = 0; i < matrices.size(); ++i) {
            if (decoded[i] != matrices[i]) { multi_ok = false; break; }
        }
    }
    std::printf("  multi round-trip: %s\n\n", multi_ok ? "PASS" : "FAIL");

    std::printf("=== lut_cas file I/O round-trip ===\n");
    const std::string tmp_path = "test_lut_cas_tmp.cas";
    bool wrote = nsos::lut_cas::write_to_file(multi.matrices[0], tmp_path);
    nsos::lut_cas::CasPack reloaded;
    bool read_ok = nsos::lut_cas::read_from_file(tmp_path, reloaded);
    bool roundtrip = (reloaded.total_tiles == multi.matrices[0].total_tiles)
                  && (reloaded.tile_indices == multi.matrices[0].tile_indices);
    std::remove(tmp_path.c_str());
    std::printf("  write=%s read=%s roundtrip=%s\n\n",
                wrote ? "ok" : "FAIL",
                read_ok ? "ok" : "FAIL",
                roundtrip ? "PASS" : "FAIL");

    const std::string multi_path = "test_lut_cas_multi_tmp.cas";
    nsos::lut_cas::write_multi_to_file(multi, multi_path);
    nsos::lut_cas::CasMultiPack reloaded_multi;
    nsos::lut_cas::read_multi_from_file(multi_path, reloaded_multi);
    bool multi_io_ok = (reloaded_multi.matrices.size() == multi.matrices.size())
                    && (reloaded_multi.shared_dictionary.size() ==
                         multi.shared_dictionary.size());
    std::remove(multi_path.c_str());
    std::printf("  multi-file I/O: %s (matrices=%zu, dict=%zu)\n",
                multi_io_ok ? "PASS" : "FAIL",
                reloaded_multi.matrices.size(),
                reloaded_multi.shared_dictionary.size());

    const bool all_ok = (ok == total) && multi_ok && roundtrip && multi_io_ok;
    std::printf("\n=== overall: %s ===\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
