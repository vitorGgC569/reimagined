// rdtsc_cycle_counter.h — high-precision cycle counter wrappers.
//
// Purpose:
//   Per-architecture inline reads of the CPU/GPU cycle counter,
//   intended for the InferenceProfiler ONLY.  Never linked into the
//   production training path.
//
// Cycle counters by platform:
//   * x86_64 host: __rdtsc() — invariant TSC on modern CPUs, ~1ns
//                  precision, no kernel transition.  Reads count of
//                  reference cycles since boot.  Recommended: pair
//                  with __rdtscp() at section end to also serialize
//                  out-of-order execution (otherwise the surrounding
//                  instructions may overlap the rdtsc result).
//   * aarch64 host: cntvct_el0 system register, ~1ns precision.
//   * GPU side: CUDA cudaEventRecord + cudaEventElapsedTime, ~1us.
//
// Design constraints:
//   * Inline header-only.  Zero function call overhead.
//   * NO heap allocations.  NO syscalls.  NO memory barriers other
//     than the rdtsc-internal serialization.
//   * Compile-time architecture detection via predefined macros.
//   * Returns uint64_t — the unit is "reference cycles" (which on
//     modern Intel/AMD is constant-frequency, NOT the variable CPU
//     frequency — see Intel SDM Vol 3 17.17).

#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define NSOS_PROFILER_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define NSOS_PROFILER_ARCH_AARCH64 1
#else
#define NSOS_PROFILER_ARCH_OTHER 1
#endif

namespace nsos {
namespace profiler {

// Read the raw cycle counter.  Cheapest call site (just rdtsc).
//
// NOTE: rdtsc on x86 does NOT serialize out-of-order execution.  The
// CPU may reorder rdtsc relative to surrounding instructions, so a
// naive bracket of `auto t0 = read_cycles(); work(); auto t1 = ...`
// can measure t0 happening AFTER work started or t1 happening BEFORE
// work finished.  For most uses the noise this introduces is in the
// few-cycles range — acceptable for measuring kernels of thousands
// of cycles, but for fine-grained kernel-by-kernel breakdowns we
// recommend `read_cycles_serialized` which uses rdtscp + cpuid.
inline uint64_t read_cycles() noexcept {
#if defined(NSOS_PROFILER_ARCH_X86)
#if defined(_MSC_VER)
    return static_cast<uint64_t>(__rdtsc());
#else
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#endif
#elif defined(NSOS_PROFILER_ARCH_AARCH64)
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    // Fallback: monotonic clock in ns, returned as "cycles" (unitless).
    // Loses ns precision; useful only for non-x86, non-ARM debug.
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

// Serialized cycle read for fine-grained measurement.  Uses rdtscp
// (which fences against earlier loads/stores but not later ones)
// followed by an lfence to fence against later loads.  Combined cost
// is ~30 cycles vs 25 for plain rdtsc — the precision is worth it
// for kernel-level profiling.
inline uint64_t read_cycles_serialized() noexcept {
#if defined(NSOS_PROFILER_ARCH_X86)
#if defined(_MSC_VER)
    unsigned int aux;
    const uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#else
    uint32_t lo, hi, aux;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    __asm__ __volatile__("lfence" : : : "memory");
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#endif
#elif defined(NSOS_PROFILER_ARCH_AARCH64)
    // ARM64: isb to fence, then read counter.
    __asm__ __volatile__("isb" : : : "memory");
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

// Calibrate cycles -> nanoseconds.  Called once at profiler init.
// Sleeps for ~10ms (steady_clock) and reads cycle delta, then
// computes the cycles/ns ratio.  On Intel TSC this is the reference
// frequency (constant), not the running CPU frequency.  Returns
// (cycles_per_ns, calibration_error_pct).
double calibrate_cycles_per_ns(int sample_ms = 50) noexcept;

}  // namespace profiler
}  // namespace nsos
