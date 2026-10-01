// lut_cas.cpp — implementation of content-addressed storage for
// ternary weight tiles.  See include/lut_cas.h for the API contract.

#include "../include/lut_cas.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nsos::lut_cas {

namespace {

// FNV-1a 64-bit checksum used at end of file format.
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime  = 1099511628211ull;
constexpr uint64_t kMaxCasFileBytes = 512ull * 1024ull * 1024ull;
constexpr uint32_t kMaxMatrices = 100'000u;
constexpr uint64_t kSingleHeaderBytes = 25u;
constexpr uint64_t kMultiHeaderBytes = 17u;
constexpr uint64_t kChecksumBytes = sizeof(uint64_t);

bool valid_tile_k(TileK k) {
    return k == TileK::K4 || k == TileK::K8;
}

uint32_t max_patterns(TileK k) {
    return k == TileK::K4 ? 81u : (k == TileK::K8 ? 6561u : 0u);
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t& out) {
    if (lhs > (std::numeric_limits<uint64_t>::max)() - rhs) return false;
    out = lhs + rhs;
    return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t& out) {
    if (lhs != 0 && rhs > (std::numeric_limits<uint64_t>::max)() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool expected_layout(int rows, int cols, TileK k, uint64_t& tiles,
                     uint64_t& packed_bytes) {
    if (rows <= 0 || cols <= 0 || !valid_tile_k(k)) return false;
    const uint64_t k_value = static_cast<uint64_t>(static_cast<int>(k));
    if ((static_cast<uint64_t>(cols) % k_value) != 0) return false;
    if (!checked_mul(static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(cols) / k_value, tiles)) {
        return false;
    }
    if (!checked_mul(tiles, k_value / 4u, packed_bytes)) return false;
    return tiles <= static_cast<uint64_t>((std::numeric_limits<int>::max)()) &&
           packed_bytes <= kMaxCasFileBytes;
}

bool validate_dictionary(const std::vector<DictEntry>& dictionary, TileK k,
                         uint64_t expected_occurrences) {
    const uint32_t pattern_cap = max_patterns(k);
    if (pattern_cap == 0 || dictionary.empty() ||
        dictionary.size() > pattern_cap ||
        dictionary.size() >
            static_cast<size_t>((std::numeric_limits<uint16_t>::max)())) {
        return false;
    }
    uint64_t occurrences = 0;
    std::unordered_set<uint32_t> seen;
    seen.reserve(dictionary.size());
    for (const auto& entry : dictionary) {
        if (entry.pattern >= pattern_cap || entry.count == 0 ||
            !seen.insert(entry.pattern).second ||
            !checked_add(occurrences, entry.count, occurrences)) {
            return false;
        }
    }
    return occurrences == expected_occurrences;
}

bool validate_pack(const CasPack& pack,
                   const std::vector<DictEntry>& dictionary) {
    uint64_t expected_tiles = 0;
    uint64_t packed_bytes = 0;
    if (!expected_layout(pack.rows, pack.cols, pack.k, expected_tiles,
                         packed_bytes) ||
        static_cast<uint64_t>(pack.total_tiles) != expected_tiles ||
        pack.tile_indices.size() != expected_tiles ||
        pack.unique_tiles != static_cast<int>(dictionary.size()) ||
        !validate_dictionary(dictionary, pack.k, expected_tiles)) {
        return false;
    }
    for (uint16_t index : pack.tile_indices) {
        if (static_cast<size_t>(index) >= dictionary.size()) return false;
    }
    return true;
}

bool validate_multi_pack(const CasMultiPack& multi, uint64_t* serialized_bytes) {
    if (!valid_tile_k(multi.k) || multi.matrices.empty() ||
        multi.matrices.size() > kMaxMatrices ||
        multi.total_matrices != multi.matrices.size()) {
        return false;
    }
    uint64_t aggregate_tiles = 0;
    uint64_t file_bytes = kMultiHeaderBytes;
    uint64_t dictionary_bytes = 0;
    if (!checked_mul(multi.shared_dictionary.size(), sizeof(DictEntry),
                     dictionary_bytes) ||
        !checked_add(file_bytes, dictionary_bytes, file_bytes)) {
        return false;
    }
    for (const auto& pack : multi.matrices) {
        uint64_t expected_tiles = 0;
        uint64_t expected_bytes = 0;
        uint64_t index_bytes = 0;
        if (pack.k != multi.k ||
            !expected_layout(pack.rows, pack.cols, pack.k, expected_tiles,
                             expected_bytes) ||
            static_cast<uint64_t>(pack.total_tiles) != expected_tiles ||
            pack.tile_indices.size() != expected_tiles ||
            pack.unique_tiles !=
                static_cast<int>(multi.shared_dictionary.size()) ||
            !checked_add(aggregate_tiles, expected_tiles, aggregate_tiles) ||
            !checked_mul(expected_tiles, sizeof(uint16_t), index_bytes) ||
            !checked_add(file_bytes, 12u, file_bytes) ||
            !checked_add(file_bytes, index_bytes, file_bytes)) {
            return false;
        }
        for (uint16_t index : pack.tile_indices) {
            if (static_cast<size_t>(index) >= multi.shared_dictionary.size()) {
                return false;
            }
        }
    }
    if (aggregate_tiles != multi.total_tiles_all ||
        !validate_dictionary(multi.shared_dictionary, multi.k,
                             aggregate_tiles) ||
        !checked_add(file_bytes, kChecksumBytes, file_bytes) ||
        file_bytes > kMaxCasFileBytes) {
        return false;
    }
    if (serialized_bytes) *serialized_bytes = file_bytes;
    return true;
}

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
    uint64_t total_tiles_u64 = 0;
    uint64_t expected_bytes = 0;
    if (!expected_layout(rows, cols, k, total_tiles_u64, expected_bytes)) {
        throw std::runtime_error("lut_cas::encode: invalid or oversized dimensions");
    }
    if (packed.size() != expected_bytes) {
        throw std::runtime_error("lut_cas::encode: packed payload size mismatch");
    }
    CasPack out;
    out.k = k;
    out.rows = rows;
    out.cols = cols;

    out.total_tiles = static_cast<int>(total_tiles_u64);
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
    if (!validate_pack(pack, pack.dictionary)) {
        throw std::runtime_error("lut_cas::decode: invalid pack");
    }
    const int k_int = static_cast<int>(pack.k);
    const int bytes_per_tile = k_int / 4;     // K4 = 1 byte; K8 = 2 bytes
    const size_t total_bytes =
        static_cast<size_t>(pack.total_tiles) *
        static_cast<size_t>(bytes_per_tile);

    std::vector<uint8_t> out(total_bytes, 0);
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
    if (matrices.size() > kMaxMatrices || !valid_tile_k(k)) {
        throw std::runtime_error("encode_multi: invalid tile width or too many matrices");
    }
    uint64_t aggregate_input_bytes = 0;
    for (const auto& matrix : matrices) {
        if (!checked_add(aggregate_input_bytes, matrix.size(),
                         aggregate_input_bytes) ||
            aggregate_input_bytes > kMaxCasFileBytes) {
            throw std::runtime_error("encode_multi: aggregate payload is too large");
        }
    }

    CasMultiPack out;
    out.k = k;
    out.total_matrices = matrices.size();

    std::unordered_map<uint32_t, uint32_t> global_pattern_to_dict;

    for (size_t m = 0; m < matrices.size(); ++m) {
        const int rows = rows_cols[m].first;
        const int cols = rows_cols[m].second;
        uint64_t total_tiles_u64 = 0;
        uint64_t expected_bytes = 0;
        if (!expected_layout(rows, cols, k, total_tiles_u64, expected_bytes) ||
            matrices[m].size() != expected_bytes) {
            throw std::runtime_error(
                "encode_multi: invalid dimensions or packed payload size");
        }
        const int total_tiles = static_cast<int>(total_tiles_u64);

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

    // Matrix packs intentionally do not duplicate the shared dictionary.
    // decode_multi supplies it by reference, keeping memory proportional to
    // one dictionary instead of matrices * dictionary.
    for (auto& pack : out.matrices) {
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
    if (!valid_tile_k(multi.k) || multi.matrices.size() > kMaxMatrices) {
        throw std::runtime_error("decode_multi: invalid multi-pack");
    }
    uint64_t aggregate_tiles = 0;
    for (const auto& pack : multi.matrices) {
        if (pack.k != multi.k) {
            throw std::runtime_error("decode_multi: mixed tile widths");
        }
        uint64_t expected_tiles = 0;
        uint64_t expected_bytes = 0;
        if (!expected_layout(pack.rows, pack.cols, pack.k, expected_tiles,
                             expected_bytes) ||
            static_cast<uint64_t>(pack.total_tiles) != expected_tiles ||
            pack.tile_indices.size() != expected_tiles ||
            pack.unique_tiles !=
                static_cast<int>(multi.shared_dictionary.size()) ||
            !checked_add(aggregate_tiles, expected_tiles, aggregate_tiles)) {
            throw std::runtime_error("decode_multi: invalid matrix metadata");
        }
        for (uint16_t index : pack.tile_indices) {
            if (static_cast<size_t>(index) >= multi.shared_dictionary.size()) {
                throw std::runtime_error("decode_multi: dictionary index out of range");
            }
        }
    }
    if (!validate_dictionary(multi.shared_dictionary, multi.k,
                             aggregate_tiles)) {
        throw std::runtime_error("decode_multi: invalid shared dictionary");
    }
    std::vector<std::vector<uint8_t>> out;
    out.reserve(multi.matrices.size());
    for (const auto& pack : multi.matrices) {
        const int bytes_per_tile = static_cast<int>(pack.k) / 4;
        std::vector<uint8_t> decoded(
            static_cast<size_t>(pack.total_tiles) *
                static_cast<size_t>(bytes_per_tile),
            0);
        for (int tile = 0; tile < pack.total_tiles; ++tile) {
            const uint16_t dictionary_index =
                pack.tile_indices[static_cast<size_t>(tile)];
            const uint32_t key =
                multi.shared_dictionary[static_cast<size_t>(dictionary_index)]
                    .pattern;
            if (pack.k == TileK::K4) {
                unpack_key_k4(key, &decoded[static_cast<size_t>(tile)]);
            } else {
                unpack_key_k8(key, &decoded[static_cast<size_t>(tile) * 2]);
            }
        }
        out.push_back(std::move(decoded));
    }
    return out;
}

// ── File I/O (single) ───────────────────────────────────────────────

bool write_to_file(const CasPack& pack, const std::string& path) {
    uint64_t dictionary_bytes = 0;
    uint64_t index_bytes = 0;
    uint64_t serialized_bytes = kSingleHeaderBytes;
    if (!validate_pack(pack, pack.dictionary) ||
        !checked_mul(pack.dictionary.size(), sizeof(DictEntry),
                     dictionary_bytes) ||
        !checked_mul(pack.tile_indices.size(), sizeof(uint16_t), index_bytes) ||
        !checked_add(serialized_bytes, dictionary_bytes, serialized_bytes) ||
        !checked_add(serialized_bytes, index_bytes, serialized_bytes) ||
        !checked_add(serialized_bytes, kChecksumBytes, serialized_bytes) ||
        serialized_bytes > kMaxCasFileBytes) {
        return false;
    }
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
    std::error_code file_error;
    const uint64_t file_size = std::filesystem::file_size(path, file_error);
    if (file_error || file_size < kSingleHeaderBytes + kChecksumBytes ||
        file_size > kMaxCasFileBytes) {
        return false;
    }
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
    CasPack parsed;
    parsed.k = static_cast<TileK>(k);
    uint32_t rows, cols, total, uniq;
    if (!read(&rows, sizeof(rows))) return false;
    if (!read(&cols, sizeof(cols))) return false;
    if (!read(&total, sizeof(total))) return false;
    if (!read(&uniq, sizeof(uniq))) return false;
    if (rows > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        cols > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        total > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
        uniq == 0 || uniq > max_patterns(parsed.k)) {
        return false;
    }
    parsed.rows = static_cast<int>(rows);
    parsed.cols = static_cast<int>(cols);
    parsed.total_tiles = static_cast<int>(total);
    parsed.unique_tiles = static_cast<int>(uniq);

    uint64_t expected_size = kSingleHeaderBytes;
    uint64_t dictionary_bytes = 0;
    uint64_t index_bytes = 0;
    if (!checked_mul(uniq, sizeof(DictEntry), dictionary_bytes) ||
        !checked_mul(total, sizeof(uint16_t), index_bytes) ||
        !checked_add(expected_size, dictionary_bytes, expected_size) ||
        !checked_add(expected_size, index_bytes, expected_size) ||
        !checked_add(expected_size, kChecksumBytes, expected_size) ||
        expected_size != file_size) {
        return false;
    }

    parsed.dictionary.assign(static_cast<size_t>(uniq), DictEntry{});
    for (uint32_t i = 0; i < uniq; ++i) {
        if (!read(&parsed.dictionary[i].pattern, sizeof(uint32_t))) return false;
        if (!read(&parsed.dictionary[i].count, sizeof(uint32_t))) return false;
    }
    parsed.tile_indices.assign(static_cast<size_t>(total), 0);
    if (!read(parsed.tile_indices.data(),
              static_cast<size_t>(total) * sizeof(uint16_t))) return false;

    uint64_t expected_checksum;
    f.read(reinterpret_cast<char*>(&expected_checksum), sizeof(expected_checksum));
    if (!f || expected_checksum != checksum_running ||
        !validate_pack(parsed, parsed.dictionary)) {
        return false;
    }
    out = std::move(parsed);
    return true;
}

// ── File I/O (multi) ────────────────────────────────────────────────

bool write_multi_to_file(const CasMultiPack& multi, const std::string& path) {
    if (!validate_multi_pack(multi, nullptr)) {
        return false;
    }
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
    std::error_code file_error;
    const uint64_t file_size = std::filesystem::file_size(path, file_error);
    if (file_error || file_size < kMultiHeaderBytes + kChecksumBytes ||
        file_size > kMaxCasFileBytes) {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t checksum_running = kFnvOffset;
    uint64_t consumed = 0;
    auto read = [&](void* data, size_t n) -> bool {
        if (static_cast<uint64_t>(n) > file_size - consumed) return false;
        f.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
        if (!f) return false;
        checksum_running = fnv1a_update(checksum_running, data, n);
        consumed += static_cast<uint64_t>(n);
        return true;
    };

    char magic[4];
    if (!read(magic, 4)) return false;
    if (magic[0] != 'C' || magic[1] != 'A' || magic[2] != 'S' || magic[3] != 'M') return false;
    uint32_t version; if (!read(&version, sizeof(version))) return false;
    if (version != 1) return false;
    uint8_t k; if (!read(&k, sizeof(k))) return false;
    if (k != 4 && k != 8) return false;
    CasMultiPack parsed;
    parsed.k = static_cast<TileK>(k);
    uint32_t n_matrices, dict_size;
    if (!read(&n_matrices, sizeof(n_matrices))) return false;
    if (!read(&dict_size, sizeof(dict_size))) return false;
    if (n_matrices == 0 || n_matrices > kMaxMatrices ||
        dict_size == 0 || dict_size > max_patterns(parsed.k)) {
        return false;
    }
    uint64_t minimum_size = kMultiHeaderBytes;
    uint64_t dictionary_bytes = 0;
    uint64_t matrix_headers = 0;
    if (!checked_mul(dict_size, sizeof(DictEntry), dictionary_bytes) ||
        !checked_mul(n_matrices, 12u, matrix_headers) ||
        !checked_add(minimum_size, dictionary_bytes, minimum_size) ||
        !checked_add(minimum_size, matrix_headers, minimum_size) ||
        !checked_add(minimum_size, kChecksumBytes, minimum_size) ||
        minimum_size > file_size) {
        return false;
    }
    parsed.total_matrices = n_matrices;

    parsed.shared_dictionary.assign(dict_size, DictEntry{});
    for (uint32_t i = 0; i < dict_size; ++i) {
        if (!read(&parsed.shared_dictionary[i].pattern, sizeof(uint32_t))) return false;
        if (!read(&parsed.shared_dictionary[i].count, sizeof(uint32_t))) return false;
    }
    parsed.matrices.reserve(n_matrices);
    parsed.total_tiles_all = 0;
    for (uint32_t m = 0; m < n_matrices; ++m) {
        CasPack pack;
        pack.k = parsed.k;
        uint32_t rows, cols, total;
        if (!read(&rows, sizeof(rows))) return false;
        if (!read(&cols, sizeof(cols))) return false;
        if (!read(&total, sizeof(total))) return false;
        if (rows > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
            cols > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
            total > static_cast<uint32_t>((std::numeric_limits<int>::max)())) {
            return false;
        }
        pack.rows = static_cast<int>(rows);
        pack.cols = static_cast<int>(cols);
        pack.total_tiles = static_cast<int>(total);
        pack.unique_tiles = static_cast<int>(dict_size);
        uint64_t expected_tiles = 0;
        uint64_t expected_bytes = 0;
        uint64_t index_bytes = 0;
        if (!expected_layout(pack.rows, pack.cols, pack.k, expected_tiles,
                             expected_bytes) ||
            expected_tiles != total ||
            !checked_mul(total, sizeof(uint16_t), index_bytes)) {
            return false;
        }
        const uint64_t remaining_headers =
            static_cast<uint64_t>(n_matrices - m - 1) * 12u;
        uint64_t required_tail = 0;
        if (!checked_add(index_bytes, remaining_headers, required_tail) ||
            !checked_add(required_tail, kChecksumBytes, required_tail) ||
            required_tail > file_size - consumed) {
            return false;
        }
        pack.tile_indices.assign(total, 0);
        if (!read(pack.tile_indices.data(),
                  static_cast<size_t>(total) * sizeof(uint16_t))) return false;
        parsed.matrices.push_back(std::move(pack));
        if (parsed.total_tiles_all >
            (std::numeric_limits<size_t>::max)() - total) {
            return false;
        }
        parsed.total_tiles_all += total;
    }

    uint64_t expected;
    f.read(reinterpret_cast<char*>(&expected), sizeof(expected));
    if (!f || consumed + kChecksumBytes != file_size ||
        expected != checksum_running ||
        !validate_multi_pack(parsed, nullptr)) {
        return false;
    }
    out = std::move(parsed);
    return true;
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
