// lut_cas.cpp — implementation of content-addressed storage for
// ternary weight tiles.  See include/lut_cas.h for the API contract.

#include "../include/lut_cas.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace nsos::lut_cas {

namespace {

// FNV-1a 64-bit checksum used at end of file format.
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime  = 1099511628211ull;

static uint64_t fnv1a_update(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= kFnvPrime;
    }
    return h;
}

// Read N bytes from packed[byte_off:byte_off+N] and pack k ternary
// weights into a single integer in base-3 (so the integer IS the
// dictionary key — unique per pattern).  We only handle K4 and K8.
//
// Layout: byte b at offset (byte_off) holds 4 codes:
//   code[0] = (b >> 0) & 0x3
//   code[1] = (b >> 2) & 0x3
//   code[2] = (b >> 4) & 0x3
//   code[3] = (b >> 6) & 0x3
//
// Encoding maps code 3 → 0 (same as kDecodeI8Lut).  Then we project
// {0, 1, 2} to {-1, 0, +1} for ternary semantics, then back to a
// base-3 digit {0, 1, 2} → represents -1/0/+1 as base-3 (we use
// 0=-1, 1=0, 2=+1 for the base-3 digits so the integer is canonical).
static uint32_t tile_key_k4(const uint8_t* packed_bytes, int flat_tile_idx) {
    const uint8_t b = packed_bytes[flat_tile_idx];  // 1 byte = 1 K4 tile
    uint32_t key = 0;
    uint32_t mult = 1;
    for (int j = 0; j < 4; ++j) {
        uint8_t code = static_cast<uint8_t>((b >> (j * 2)) & 0x3u);
        if (code == 3u) code = 1u;   // unused → 0
        // code 0 → -1 → digit 0;  code 1 → 0 → digit 1;  code 2 → +1 → digit 2
        const uint32_t digit = (code == 0u) ? 0u : ((code == 1u) ? 1u : 2u);
        key += digit * mult;
        mult *= 3u;
    }
    return key;   // 0..80
}

static uint32_t tile_key_k8(const uint8_t* packed_bytes, int flat_tile_idx) {
    // 2 bytes per K8 tile (8 weights × 2 bits = 16 bits)
    const uint8_t lo = packed_bytes[flat_tile_idx * 2];
    const uint8_t hi = packed_bytes[flat_tile_idx * 2 + 1];
    uint32_t key = 0;
    uint32_t mult = 1;
    for (int j = 0; j < 4; ++j) {
        uint8_t code = static_cast<uint8_t>((lo >> (j * 2)) & 0x3u);
        if (code == 3u) code = 1u;
        const uint32_t digit = (code == 0u) ? 0u : ((code == 1u) ? 1u : 2u);
        key += digit * mult;
        mult *= 3u;
    }
    for (int j = 0; j < 4; ++j) {
        uint8_t code = static_cast<uint8_t>((hi >> (j * 2)) & 0x3u);
        if (code == 3u) code = 1u;
        const uint32_t digit = (code == 0u) ? 0u : ((code == 1u) ? 1u : 2u);
        key += digit * mult;
        mult *= 3u;
    }
    return key;   // 0..6560
}

// Reverse: turn a base-3 key back into a byte representation.
static void unpack_key_k4(uint32_t key, uint8_t* dst_byte) {
    uint8_t b = 0;
    for (int j = 0; j < 4; ++j) {
        const uint32_t digit = key % 3u;
        key /= 3u;
        // digit 0 → code 0 (-1); 1 → 1 (0); 2 → 2 (+1)
        const uint8_t code = static_cast<uint8_t>(digit);
        b |= static_cast<uint8_t>(code << (j * 2));
    }
    *dst_byte = b;
}

static void unpack_key_k8(uint32_t key, uint8_t* dst_two_bytes) {
    uint8_t lo = 0, hi = 0;
    for (int j = 0; j < 4; ++j) {
        const uint32_t digit = key % 3u;
        key /= 3u;
        lo |= static_cast<uint8_t>(static_cast<uint8_t>(digit) << (j * 2));
    }
    for (int j = 0; j < 4; ++j) {
        const uint32_t digit = key % 3u;
        key /= 3u;
        hi |= static_cast<uint8_t>(static_cast<uint8_t>(digit) << (j * 2));
    }
    dst_two_bytes[0] = lo;
    dst_two_bytes[1] = hi;
}

}  // namespace

// ── Single-matrix encode ────────────────────────────────────────────

CasPack encode(const std::vector<uint8_t>& packed, int rows, int cols, TileK k) {
    CasPack out;
    out.k = k;
    out.rows = rows;
    out.cols = cols;

    const int total_weights = rows * cols;
    const int k_int = static_cast<int>(k);
    if (cols % k_int != 0) {
        throw std::runtime_error("lut_cas::encode: cols must be divisible by k");
    }
    const int tiles_per_row = cols / k_int;
    out.total_tiles = rows * tiles_per_row;
    out.tile_indices.reserve(static_cast<size_t>(out.total_tiles));

    std::unordered_map<uint32_t, uint32_t> pattern_to_dict_idx;
    pattern_to_dict_idx.reserve(static_cast<size_t>(out.total_tiles / 4));

    for (int t = 0; t < out.total_tiles; ++t) {
        uint32_t key;
        if (k == TileK::K4) {
            key = tile_key_k4(packed.data(), t);
        } else {
            key = tile_key_k8(packed.data(), t);
        }
        auto it = pattern_to_dict_idx.find(key);
        uint32_t dict_idx;
        if (it == pattern_to_dict_idx.end()) {
            dict_idx = static_cast<uint32_t>(out.dictionary.size());
            out.dictionary.push_back({key, 1});
            pattern_to_dict_idx[key] = dict_idx;
        } else {
            dict_idx = it->second;
            out.dictionary[dict_idx].count += 1;
        }
        out.tile_indices.push_back(static_cast<uint16_t>(dict_idx));
    }

    out.unique_tiles = static_cast<int>(out.dictionary.size());

    // Compression ratio: raw vs CAS-serialized
    const size_t raw_bytes = packed.size();
    const size_t cas_bytes =
        sizeof(uint32_t) + sizeof(uint8_t) + 4 * sizeof(uint32_t)  // header
        + out.dictionary.size() * sizeof(DictEntry)
        + out.tile_indices.size() * sizeof(uint16_t)
        + sizeof(uint64_t);                                        // checksum
    out.compression_ratio = static_cast<double>(cas_bytes) /
                             static_cast<double>(std::max(raw_bytes, size_t{1}));

    // Dict coverage: fraction of dictionary entries used 10+ times
    size_t heavy = 0;
    for (const auto& e : out.dictionary) if (e.count >= 10) ++heavy;
    out.dict_coverage = static_cast<double>(heavy) /
                        static_cast<double>(std::max(out.dictionary.size(), size_t{1}));

    return out;
}

// ── Single-matrix decode ────────────────────────────────────────────

std::vector<uint8_t> decode(const CasPack& pack) {
    const int k_int = static_cast<int>(pack.k);
    const int bytes_per_tile = k_int / 4;     // K4 = 1 byte; K8 = 2 bytes
    const int total_bytes = pack.total_tiles * bytes_per_tile;

    std::vector<uint8_t> out(static_cast<size_t>(total_bytes), 0);
    for (int t = 0; t < pack.total_tiles; ++t) {
        const uint16_t dict_idx = pack.tile_indices[static_cast<size_t>(t)];
        const uint32_t key = pack.dictionary[dict_idx].pattern;
        if (pack.k == TileK::K4) {
            unpack_key_k4(key, &out[static_cast<size_t>(t)]);
        } else {
            unpack_key_k8(key, &out[static_cast<size_t>(t) * 2]);
        }
    }
    return out;
}

// ── Multi-matrix encode ─────────────────────────────────────────────

CasMultiPack encode_multi(
    const std::vector<std::vector<uint8_t>>& matrices,
    const std::vector<std::pair<int, int>>& rows_cols,
    TileK k) {
    if (matrices.size() != rows_cols.size()) {
        throw std::runtime_error("encode_multi: matrices and rows_cols mismatch");
    }
    CasMultiPack out;
    out.k = k;
    out.total_matrices = matrices.size();

    std::unordered_map<uint32_t, uint32_t> global_pattern_to_dict;

    const int k_int = static_cast<int>(k);
    for (size_t m = 0; m < matrices.size(); ++m) {
        const int rows = rows_cols[m].first;
        const int cols = rows_cols[m].second;
        if (cols % k_int != 0) {
            throw std::runtime_error("encode_multi: matrix cols must be divisible by k");
        }
        const int tiles_per_row = cols / k_int;
        const int total_tiles = rows * tiles_per_row;

        CasPack pack;
        pack.k = k;
        pack.rows = rows;
        pack.cols = cols;
        pack.total_tiles = total_tiles;
        pack.tile_indices.reserve(static_cast<size_t>(total_tiles));

        const uint8_t* packed_ptr = matrices[m].data();
        for (int t = 0; t < total_tiles; ++t) {
            uint32_t key = (k == TileK::K4)
                ? tile_key_k4(packed_ptr, t)
                : tile_key_k8(packed_ptr, t);
            auto it = global_pattern_to_dict.find(key);
            uint32_t dict_idx;
            if (it == global_pattern_to_dict.end()) {
                dict_idx = static_cast<uint32_t>(out.shared_dictionary.size());
                out.shared_dictionary.push_back({key, 1});
                global_pattern_to_dict[key] = dict_idx;
            } else {
                dict_idx = it->second;
                out.shared_dictionary[dict_idx].count += 1;
            }
            pack.tile_indices.push_back(static_cast<uint16_t>(dict_idx));
        }
        pack.unique_tiles = static_cast<int>(out.shared_dictionary.size());
        out.matrices.push_back(std::move(pack));
        out.total_tiles_all += static_cast<size_t>(total_tiles);
    }

    // Stamp each per-matrix pack with a reference to the shared dictionary
    // for convenience (so single-matrix API works on them too).
    for (auto& pack : out.matrices) {
        pack.dictionary = out.shared_dictionary;
        pack.unique_tiles = static_cast<int>(out.shared_dictionary.size());
    }

    // Aggregate compression ratio
    size_t raw_total = 0;
    for (const auto& m : matrices) raw_total += m.size();
    size_t cas_total = 16  // header
                       + out.shared_dictionary.size() * sizeof(DictEntry);
    for (const auto& pack : out.matrices) {
        cas_total += 8;  // per-matrix header (rows, cols)
        cas_total += pack.tile_indices.size() * sizeof(uint16_t);
    }
    cas_total += sizeof(uint64_t);  // checksum
    out.aggregate_compression = static_cast<double>(cas_total) /
                                 static_cast<double>(std::max(raw_total, size_t{1}));

    return out;
}

std::vector<std::vector<uint8_t>> decode_multi(const CasMultiPack& multi) {
    std::vector<std::vector<uint8_t>> out;
    out.reserve(multi.matrices.size());
    for (const auto& pack : multi.matrices) {
        out.push_back(decode(pack));
    }
    return out;
}

// ── File I/O (single) ───────────────────────────────────────────────

bool write_to_file(const CasPack& pack, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t checksum = kFnvOffset;
    auto write = [&](const void* data, size_t n) {
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        checksum = fnv1a_update(checksum, data, n);
    };

    const char magic[4] = {'C', 'A', 'S', '1'};
    write(magic, 4);
    const uint32_t version = 1;
    write(&version, sizeof(version));
    const uint8_t k = static_cast<uint8_t>(pack.k);
    write(&k, sizeof(k));
    const uint32_t rows = static_cast<uint32_t>(pack.rows);
    const uint32_t cols = static_cast<uint32_t>(pack.cols);
    const uint32_t total = static_cast<uint32_t>(pack.total_tiles);
    const uint32_t uniq  = static_cast<uint32_t>(pack.unique_tiles);
    write(&rows, sizeof(rows));
    write(&cols, sizeof(cols));
    write(&total, sizeof(total));
    write(&uniq, sizeof(uniq));
    for (const auto& e : pack.dictionary) {
        write(&e.pattern, sizeof(e.pattern));
        write(&e.count, sizeof(e.count));
    }
    write(pack.tile_indices.data(),
          pack.tile_indices.size() * sizeof(uint16_t));
    // Checksum is NOT itself checksummed.
    f.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    return static_cast<bool>(f);
}

bool read_from_file(const std::string& path, CasPack& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t checksum_running = kFnvOffset;
    auto read = [&](void* data, size_t n) -> bool {
        f.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
        if (!f) return false;
        checksum_running = fnv1a_update(checksum_running, data, n);
        return true;
    };

    char magic[4];
    if (!read(magic, 4)) return false;
    if (magic[0] != 'C' || magic[1] != 'A' || magic[2] != 'S' || magic[3] != '1') {
        return false;
    }
    uint32_t version; if (!read(&version, sizeof(version))) return false;
    if (version != 1) return false;
    uint8_t k; if (!read(&k, sizeof(k))) return false;
    if (k != 4 && k != 8) return false;
    out.k = static_cast<TileK>(k);
    uint32_t rows, cols, total, uniq;
    if (!read(&rows, sizeof(rows))) return false;
    if (!read(&cols, sizeof(cols))) return false;
    if (!read(&total, sizeof(total))) return false;
    if (!read(&uniq, sizeof(uniq))) return false;
    out.rows = static_cast<int>(rows);
    out.cols = static_cast<int>(cols);
    out.total_tiles = static_cast<int>(total);
    out.unique_tiles = static_cast<int>(uniq);

    out.dictionary.assign(static_cast<size_t>(uniq), DictEntry{});
    for (uint32_t i = 0; i < uniq; ++i) {
        if (!read(&out.dictionary[i].pattern, sizeof(uint32_t))) return false;
        if (!read(&out.dictionary[i].count, sizeof(uint32_t))) return false;
    }
    out.tile_indices.assign(static_cast<size_t>(total), 0);
    if (!read(out.tile_indices.data(), total * sizeof(uint16_t))) return false;

    uint64_t expected_checksum;
    f.read(reinterpret_cast<char*>(&expected_checksum), sizeof(expected_checksum));
    if (!f) return false;
    return expected_checksum == checksum_running;
}

// ── File I/O (multi) ────────────────────────────────────────────────

bool write_multi_to_file(const CasMultiPack& multi, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t checksum = kFnvOffset;
    auto write = [&](const void* data, size_t n) {
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        checksum = fnv1a_update(checksum, data, n);
    };

    const char magic[4] = {'C', 'A', 'S', 'M'};
    write(magic, 4);
    const uint32_t version = 1;
    write(&version, sizeof(version));
    const uint8_t k = static_cast<uint8_t>(multi.k);
    write(&k, sizeof(k));
    const uint32_t n_matrices = static_cast<uint32_t>(multi.matrices.size());
    const uint32_t dict_size  = static_cast<uint32_t>(multi.shared_dictionary.size());
    write(&n_matrices, sizeof(n_matrices));
    write(&dict_size, sizeof(dict_size));
    for (const auto& e : multi.shared_dictionary) {
        write(&e.pattern, sizeof(e.pattern));
        write(&e.count, sizeof(e.count));
    }
    for (const auto& pack : multi.matrices) {
        const uint32_t rows = static_cast<uint32_t>(pack.rows);
        const uint32_t cols = static_cast<uint32_t>(pack.cols);
        const uint32_t total = static_cast<uint32_t>(pack.total_tiles);
        write(&rows, sizeof(rows));
        write(&cols, sizeof(cols));
        write(&total, sizeof(total));
        write(pack.tile_indices.data(),
              pack.tile_indices.size() * sizeof(uint16_t));
    }
    f.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    return static_cast<bool>(f);
}

bool read_multi_from_file(const std::string& path, CasMultiPack& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t checksum_running = kFnvOffset;
    auto read = [&](void* data, size_t n) -> bool {
        f.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
        if (!f) return false;
        checksum_running = fnv1a_update(checksum_running, data, n);
        return true;
    };

    char magic[4];
    if (!read(magic, 4)) return false;
    if (magic[0] != 'C' || magic[1] != 'A' || magic[2] != 'S' || magic[3] != 'M') return false;
    uint32_t version; if (!read(&version, sizeof(version))) return false;
    if (version != 1) return false;
    uint8_t k; if (!read(&k, sizeof(k))) return false;
    out.k = static_cast<TileK>(k);
    uint32_t n_matrices, dict_size;
    if (!read(&n_matrices, sizeof(n_matrices))) return false;
    if (!read(&dict_size, sizeof(dict_size))) return false;
    out.total_matrices = n_matrices;

    out.shared_dictionary.assign(dict_size, DictEntry{});
    for (uint32_t i = 0; i < dict_size; ++i) {
        if (!read(&out.shared_dictionary[i].pattern, sizeof(uint32_t))) return false;
        if (!read(&out.shared_dictionary[i].count, sizeof(uint32_t))) return false;
    }
    out.matrices.clear();
    out.matrices.reserve(n_matrices);
    out.total_tiles_all = 0;
    for (uint32_t m = 0; m < n_matrices; ++m) {
        CasPack pack;
        pack.k = out.k;
        uint32_t rows, cols, total;
        if (!read(&rows, sizeof(rows))) return false;
        if (!read(&cols, sizeof(cols))) return false;
        if (!read(&total, sizeof(total))) return false;
        pack.rows = static_cast<int>(rows);
        pack.cols = static_cast<int>(cols);
        pack.total_tiles = static_cast<int>(total);
        pack.unique_tiles = static_cast<int>(dict_size);
        pack.dictionary = out.shared_dictionary;
        pack.tile_indices.assign(total, 0);
        if (!read(pack.tile_indices.data(), total * sizeof(uint16_t))) return false;
        out.matrices.push_back(std::move(pack));
        out.total_tiles_all += total;
    }

    uint64_t expected;
    f.read(reinterpret_cast<char*>(&expected), sizeof(expected));
    if (!f) return false;
    return expected == checksum_running;
}

// ── Stats ───────────────────────────────────────────────────────────

CompressionStats stats(const CasMultiPack& multi) {
    CompressionStats s{};
    s.total_tiles = multi.total_tiles_all;
    s.unique_tiles = multi.shared_dictionary.size();
    s.cas_bytes = 16 + multi.shared_dictionary.size() * sizeof(DictEntry) + sizeof(uint64_t);
    s.raw_bytes = 0;
    for (const auto& pack : multi.matrices) {
        const int k_int = static_cast<int>(pack.k);
        const size_t bytes_per_tile = static_cast<size_t>(k_int / 4);
        s.raw_bytes += static_cast<size_t>(pack.total_tiles) * bytes_per_tile;
        s.cas_bytes += 12 + pack.tile_indices.size() * sizeof(uint16_t);
    }
    s.compression_ratio = (s.raw_bytes > 0)
        ? static_cast<double>(s.cas_bytes) / static_cast<double>(s.raw_bytes)
        : 0.0;
    s.avg_tile_occurrence = (s.unique_tiles > 0)
        ? static_cast<double>(s.total_tiles) / static_cast<double>(s.unique_tiles)
        : 0.0;
    // Top-10 coverage
    auto sorted = multi.shared_dictionary;
    std::partial_sort(sorted.begin(),
                      sorted.begin() + std::min(size_t{10}, sorted.size()),
                      sorted.end(),
                      [](const DictEntry& a, const DictEntry& b) {
                          return a.count > b.count;
                      });
    size_t top10_count = 0;
    for (size_t i = 0; i < std::min(size_t{10}, sorted.size()); ++i) {
        top10_count += sorted[i].count;
    }
    s.dict_top10_coverage = (s.total_tiles > 0)
        ? static_cast<double>(top10_count) / static_cast<double>(s.total_tiles)
        : 0.0;
    return s;
}

CompressionStats stats(const CasPack& pack) {
    CompressionStats s{};
    s.total_tiles = static_cast<size_t>(pack.total_tiles);
    s.unique_tiles = pack.dictionary.size();
    const int k_int = static_cast<int>(pack.k);
    const size_t bytes_per_tile = static_cast<size_t>(k_int / 4);
    s.raw_bytes = s.total_tiles * bytes_per_tile;
    s.cas_bytes = 21 + pack.dictionary.size() * sizeof(DictEntry)
                  + pack.tile_indices.size() * sizeof(uint16_t)
                  + sizeof(uint64_t);
    s.compression_ratio = (s.raw_bytes > 0)
        ? static_cast<double>(s.cas_bytes) / static_cast<double>(s.raw_bytes) : 0.0;
    s.avg_tile_occurrence = (s.unique_tiles > 0)
        ? static_cast<double>(s.total_tiles) / static_cast<double>(s.unique_tiles)
        : 0.0;
    return s;
}

}  // namespace nsos::lut_cas
