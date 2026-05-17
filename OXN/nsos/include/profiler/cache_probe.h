// cache_probe.h — empirical cache-level latency measurement.
//
// Purpose:
//   Measure the L1 / L2 / L3 / DRAM latency on the current host so
//   the analysis report can interpret observed cycle counts.  E.g.:
//   if a BitLinear forward takes 50k cycles and the model is 8MB
//   total in 1.58-bit weights — which SHOULD fit in L2 — but the
//   measured cycles are consistent with DRAM access, that's a
//   diagnosis: weights are spilling, not staying hot.
//
// Method:
//   Classic pointer-chase benchmark (Hennessy & Patterson).  Allocate
//   a working set of varying size, build a random permutation so
//   the CPU prefetcher can't help, time N reads.  Smallest size
//   gives L1 latency, growth steps map to L2 / L3 / DRAM.
//
// Output:
//   `CacheLatencyReport` with median latency in cycles for working
//   sets of 4KB, 32KB, 256KB, 2MB, 16MB, 128MB, 1GB.  The Python
//   driver charts this as a "memory hierarchy step plot" and uses
//   the inflection points to identify L1 / L2 / L3 / DRAM
//   boundaries empirically (rather than relying on CPUID which
//   often lies on virtualized hosts).

#pragma once

#include <cstdint>
#include <vector>

namespace nsos {
namespace profiler {

struct CacheLatencySample {
    size_t   working_set_bytes;
    double   median_cycles_per_access;
    double   p90_cycles_per_access;
    uint64_t accesses_measured;
};

struct CacheLatencyReport {
    std::vector<CacheLatencySample> samples;
    // Inferred memory hierarchy boundaries.  E.g., l1_bytes is the
    // largest working set that still fits in L1 (latency below the
    // first step).  Zero if undetected.
    size_t inferred_l1_bytes = 0;
    size_t inferred_l2_bytes = 0;
    size_t inferred_l3_bytes = 0;
    double dram_latency_cycles = 0.0;
};

// Probe the host cache hierarchy.  Takes ~1-2 seconds total.
CacheLatencyReport probe_cache_latencies();

}  // namespace profiler
}  // namespace nsos
