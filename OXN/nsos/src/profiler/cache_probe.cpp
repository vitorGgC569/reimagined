// cache_probe.cpp — empirical memory hierarchy latency probe.
//
// Method (classic pointer-chase, Hennessy & Patterson):
//   For each candidate working set size:
//     1. Allocate a buffer of N * sizeof(size_t).
//     2. Initialize as a random permutation cycle: buf[i] = perm[i],
//        such that following the chain buf[0] -> buf[buf[0]] -> ...
//        visits every entry exactly once before returning to 0.
//        Random order defeats the hardware prefetcher.
//     3. Time M chained reads.
//     4. Median / p90 cycles per access.
//   Plot the curve and look for inflection points — those are the
//   cache boundaries on the actual host (not what CPUID claims).
//
// Why this matters for our analysis:
//   The NSOS BitNet weights at 1.58 bits are ~8 MB total for the 40M
//   model.  If L2 is, say, 1 MB, the weights spill into L3 / DRAM
//   on the second read of each layer's row.  Knowing where the
//   inflections actually are tells us the upper bound on what
//   weight-in-cache strategies can achieve.

#include "../../include/profiler/cache_probe.h"
#include "../../include/profiler/rdtsc_cycle_counter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

namespace nsos {
namespace profiler {

namespace {

// Standard pointer-chase: each slot holds the index of the next slot.
// Returns a vector of `n` size_t in random permutation cycle order.
// Seed is fixed so probes are reproducible across runs.
std::vector<size_t> build_chase_buffer(size_t n, uint64_t seed) {
    std::vector<size_t> buf(n);
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    std::mt19937_64 rng(seed);
    // Shuffle into a single cycle: pick a random successor for each
    // position, ensuring the chain visits every index once.
    for (size_t i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i - 1);
        const size_t j = dist(rng);
        std::swap(perm[i], perm[j]);
    }
    // perm[i] is the position visited at step i.  Build the
    // pointer chain such that buf[perm[i]] = perm[(i+1) % n].
    for (size_t i = 0; i < n; ++i) {
        buf[perm[i]] = perm[(i + 1) % n];
    }
    return buf;
}

CacheLatencySample measure_one(size_t working_set_bytes, size_t n_accesses,
                                uint64_t seed) {
    const size_t n_slots = working_set_bytes / sizeof(size_t);
    if (n_slots < 16) {
        return {working_set_bytes, 0.0, 0.0, 0};
    }
    auto buf = build_chase_buffer(n_slots, seed);

    // Warm: walk the buffer once to bring it into cache (or DRAM,
    // depending on size), so the timed loop measures steady state.
    {
        volatile size_t p = 0;
        for (size_t i = 0; i < n_slots; ++i) {
            p = buf[p];
        }
        (void)p;
    }

    // Record per-block latency over several windows of `window_size`
    // chained reads, take median + p90 across windows.  Window size
    // chosen so each window observes ~1K cycles minimum.
    constexpr size_t window_size = 1024;
    const size_t n_windows = std::max<size_t>(8, n_accesses / window_size);

    std::vector<uint64_t> per_window_cycles;
    per_window_cycles.reserve(n_windows);

    volatile size_t p = 0;
    for (size_t w = 0; w < n_windows; ++w) {
        const uint64_t t0 = read_cycles_serialized();
        for (size_t i = 0; i < window_size; ++i) {
            p = buf[p];
        }
        const uint64_t t1 = read_cycles_serialized();
        per_window_cycles.push_back(t1 - t0);
    }
    (void)p;

    std::sort(per_window_cycles.begin(), per_window_cycles.end());
    const uint64_t median_total = per_window_cycles[n_windows / 2];
    const uint64_t p90_total =
        per_window_cycles[std::min(n_windows - 1, (n_windows * 9) / 10)];

    CacheLatencySample s;
    s.working_set_bytes = working_set_bytes;
    s.median_cycles_per_access =
        static_cast<double>(median_total) / static_cast<double>(window_size);
    s.p90_cycles_per_access =
        static_cast<double>(p90_total) / static_cast<double>(window_size);
    s.accesses_measured = static_cast<uint64_t>(n_windows) * window_size;
    return s;
}

// Infer cache boundaries from the curve.  Look for "elbows" where
// the median latency jumps by >= 1.6x relative to the prior sample.
// First elbow = L1->L2; second = L2->L3; third = L3->DRAM.  If fewer
// elbows are detected, only the corresponding fields are filled.
void infer_hierarchy(CacheLatencyReport& report) {
    if (report.samples.size() < 3) return;
    std::vector<size_t> elbow_indices;
    for (size_t i = 1; i < report.samples.size(); ++i) {
        const double prev = report.samples[i - 1].median_cycles_per_access;
        const double cur  = report.samples[i].median_cycles_per_access;
        if (prev <= 0.0) continue;
        if (cur / prev >= 1.6) {
            elbow_indices.push_back(i);
        }
    }
    if (elbow_indices.size() >= 1) {
        report.inferred_l1_bytes =
            report.samples[elbow_indices[0] - 1].working_set_bytes;
    }
    if (elbow_indices.size() >= 2) {
        report.inferred_l2_bytes =
            report.samples[elbow_indices[1] - 1].working_set_bytes;
    }
    if (elbow_indices.size() >= 3) {
        report.inferred_l3_bytes =
            report.samples[elbow_indices[2] - 1].working_set_bytes;
    }
    // DRAM latency = the largest sample (assumed to be in DRAM
    // regime).  If we don't have a clear plateau, this is just the
    // worst-case observation.
    report.dram_latency_cycles =
        report.samples.back().median_cycles_per_access;
}

}  // namespace

CacheLatencyReport probe_cache_latencies() {
    CacheLatencyReport report;
    // Working set sizes: 4KB (L1), 32KB, 256KB (typical L2),
    // 2MB, 16MB (typical L3 boundary), 128MB (definitely DRAM),
    // 1GB (DRAM far).  Skip the 1GB on hosts where malloc fails.
    const std::vector<size_t> sizes = {
        4 * 1024,
        32 * 1024,
        256 * 1024,
        2 * 1024 * 1024,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
        // 1 * 1024 * 1024 * 1024,  // optional; commented out for safety
    };
    const size_t n_accesses_target = 1 << 20;  // ~1M reads per size
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    for (const size_t sz : sizes) {
        try {
            report.samples.push_back(
                measure_one(sz, n_accesses_target, seed));
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        } catch (const std::bad_alloc&) {
            break;
        }
    }
    infer_hierarchy(report);
    return report;
}

}  // namespace profiler
}  // namespace nsos
