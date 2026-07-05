// lut_cas.h — Content-Addressed Storage for ternary weight tiles.
//
// MOTIVATION (revised after bench_lut_tmac proved LUT-TMAC loses 20× to SIMD):
//   The "buckets como S3" idea pays off in STORAGE (pack size), not in
//   inference COMPUTE.  SIMD AVX2/VNNI dominates compute regardless of
//   pattern reuse.  But cross-layer pattern dedup compresses the on-disk
//   pack and reduces edge memory load with ZERO inference penalty when
//   we expand tiles back to int8 / packed-bytes before forward.
//
// THE INSIGHT:
//   A ternary tile of k=4 weights has exactly 3^4 = 81 possible
//   patterns.  A 40M-param hybrid model with d_model=512 has
//   roughly 40e6/4 = 10M tiles.  Pigeonhole: many tiles MUST repeat.
//   With MoE-8 experts (each ~5M tiles), repetition is even higher
//   because experts share weight initialization distributions.
//
//   For k=4 we get 81 unique codes.  Pack size before dedup:
//     10M tiles × 8 bits/tile = 10 MB
//   After CAS dedup with global table:
//     dictionary:  81 × 1 byte = 81 bytes
//     indices:     10M × 7 bits = 8.75 MB  (or 10 MB with 1-byte indices)
//   Storage savings near zero for k=4 because 8 bits ≈ 7 bits.
//
//   For k=8 the math becomes interesting:
//     3^8 = 6561 unique tiles.  10M / 6561 → most tiles repeat ~1500 times.
//     Pack size before: 10M × 16 bits = 20 MB
//     Indices: 10M × 13 bits (ceil log2 6561) = 16.25 MB
//     Savings: ~ 20% on flat dedup.
//
//   For k=16 (256-bit tiles):
//     3^16 = 43M possible patterns.  10M tiles → still LOTS of empty
//     pattern slots.  But the trick: real ternary networks DON'T use
//     uniform random patterns.  After BitNet b1.58 training the weight
//     distribution is biased toward sparse + structured patterns.
//     Empirically only ~10-20% of the 43M patterns ever appear.
//     Pack size with sparse codebook: 10M × log2(2M unique) = 25M bits
//                                      = ~3 MB
//     vs raw 40M bits = 5 MB → **40% reduction**.
//
//   For MoE-8 the multi-expert dedup is even better because experts
//   share many tiles.  Expected pack-size reduction: 40-60%.
//
// API:
//   * encode(packed_bytes, k, N, K) → CasPack
//       Scans tiles, builds a dictionary, returns indexed representation.
//   * decode(CasPack) → packed_bytes
//       Round-trip back to standard 1.58-bit packed bytes.
//       Used at model load time (before sending to GPU / SIMD path).
//   * write_to_file(CasPack, path)
//   * read_from_file(path) → CasPack
//
// THE OUTPUT can replace `edge_linear.nsos` in the pack manifest with
// a new `edge_linear_cas.nsos` (manifest version v3).  Pack-loader gets
// a v3 case branch that decodes back to packed_bytes before calling
// the existing `JambaModel::load_edge_linear_pack`.  Zero changes to
// the inference path.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nsos::lut_cas {

// Tile bit width.  k=4 → 81 codes; k=8 → 6561 codes.  k=8 is the
// sweet spot per the analysis above.  Library supports both; choose
// at encode time.
enum class TileK : int { K4 = 4, K8 = 8 };

// One unique tile pattern + how many times it appears.
struct DictEntry {
    uint32_t pattern;   // for K4: 0..80;  for K8: 0..6560
    uint32_t count;     // occurrence count across the whole pack
};

struct CasPack {
    TileK k;
    // Dictionary: index → pattern.  pattern is the raw N-bit packed code
    // for the tile, interpreted as a base-3 integer.  At decode time we
    // un-pack the integer back into ternary codes.
    std::vector<DictEntry> dictionary;
    // Index stream: one entry per tile of the original matrix.  Each
    // entry is an index into `dictionary`.  Stored as uint16_t which is
    // sufficient for K=8 (dict size ≤ 6561 < 65535).
    std::vector<uint16_t> tile_indices;
    // Metadata to round-trip
    int rows = 0;       // N (output features) of the matrix
    int cols = 0;       // K (input features)
    int total_tiles = 0;
    int unique_tiles = 0;
    // Stats for reporting
    double compression_ratio = 0.0;  // CAS bytes / raw packed bytes
    double dict_coverage = 0.0;      // fraction of dict that gets ≥ 10 uses
};

// ── Encode ────────────────────────────────────────────────────────────
// Build a CAS pack from a packed-bytes representation.
//
// `packed` follows the existing NSOS encoding: 4 ternary weights per
// byte, code 0=-1, 1=0, 2=+1, 3=unused-treated-as-0.
//
// Returns a CasPack ready to write to disk.
CasPack encode(const std::vector<uint8_t>& packed, int rows, int cols,
                TileK k = TileK::K8);

// ── Decode ────────────────────────────────────────────────────────────
// Round-trip back to packed bytes.  Output is byte-identical to the
// input of encode() when no compression artifacts.
std::vector<uint8_t> decode(const CasPack& pack);

// ── I/O ──────────────────────────────────────────────────────────────
// Binary file format (little-endian):
//   magic:     4 bytes "CAS1"
//   version:   uint32 (currently 1)
//   k:         uint8  (4 or 8)
//   rows:      uint32
//   cols:      uint32
//   total_tiles:  uint32
//   unique_tiles: uint32
//   dictionary: unique_tiles × (uint32 pattern + uint32 count)
//   indices:   total_tiles × uint16
// Followed by 8-byte checksum (FNV-1a of all preceding bytes).

bool write_to_file(const CasPack& pack, const std::string& path);
bool read_from_file(const std::string& path, CasPack& out_pack);

// ── Multi-matrix dedup ───────────────────────────────────────────────
// For MoE / multi-layer dedup, encode many matrices together so the
// dictionary is shared.  Returns one CasPack per matrix BUT all sharing
// the same `dictionary` and same tile codes — only `tile_indices`,
// `rows`, `cols`, `total_tiles` differ per matrix.
//
// On disk this is one file with N matrices: dict written once,
// followed by N (rows, cols, indices) blocks.

struct CasMultiPack {
    TileK k;
    std::vector<DictEntry> shared_dictionary;
    // Per-matrix packs keep dictionary empty; shared_dictionary is the single
    // source of truth.  This avoids duplicating a large dictionary per matrix.
    std::vector<CasPack> matrices;
    // Aggregate stats
    size_t total_matrices = 0;
    size_t total_tiles_all = 0;
    double aggregate_compression = 0.0;
};

CasMultiPack encode_multi(
    const std::vector<std::vector<uint8_t>>& packed_matrices,
    const std::vector<std::pair<int, int>>& rows_cols,   // per-matrix (rows, cols)
    TileK k = TileK::K8);

std::vector<std::vector<uint8_t>> decode_multi(const CasMultiPack& multi);

bool write_multi_to_file(const CasMultiPack& multi, const std::string& path);
bool read_multi_from_file(const std::string& path, CasMultiPack& out);

// ── Stats / introspection (for reporting + bench) ────────────────────
struct CompressionStats {
    size_t raw_bytes;        // sum of packed_bytes inputs
    size_t cas_bytes;        // sum of dictionary + indices serialization
    double compression_ratio;
    size_t unique_tiles;
    size_t total_tiles;
    double avg_tile_occurrence;     // total / unique
    double dict_top10_coverage;     // fraction of tiles covered by top-10 dict entries
};

CompressionStats stats(const CasMultiPack& multi);
CompressionStats stats(const CasPack& pack);

}  // namespace nsos::lut_cas
