// inference_profiler.h — NSOS InferenceProfiler module.
//
// Purpose:
//   A self-contained, OPT-IN profiling module that captures
//   per-layer and per-op cycle counts during an inference forward
//   pass.  Used to inform architectural research — NEVER linked
//   into production training or production inference.
//
// Why a separate module:
//   * Training and production inference must have ZERO overhead from
//     profiling.  This file defines the *interface*; the
//     implementation lives in src/profiler/ and is compiled into a
//     separate static library `nsos_profiler` that the main
//     `nsos_core` does not link.
//   * The model talks to the profiler via a single null-checked
//     callback pointer.  When no profiler is attached, the hot path
//     is one load + one branch — typically removed by the branch
//     predictor on first iteration.
//   * Attachment is explicit: caller invokes
//     `model->attach_profiler(profiler.get())` only from the
//     profiling tool, never from production paths.
//
// Data model:
//   * Events are categorized hierarchically:
//       PHASE     -- top-level inference phase (prefill / decode_step)
//       LAYER     -- one JambaBlock forward
//       OP        -- a sub-operation inside a layer (attn, mamba,
//                    ffn_gate_up, ffn_down, moe_router, moe_expert_X,
//                    rmsnorm, squared_relu, ...)
//       KERNEL    -- a single GPU kernel launch or CPU kernel call
//   * Each event carries: (kind, name, layer_idx, start_cycle,
//     end_cycle, optional bytes_in/out and FLOPs estimate).
//   * Events are stored in a lock-free ring buffer of fixed size
//     (default 64K entries).  Wraparound discards oldest.
//
// Output:
//   * `drain_to_json(path)` writes all collected events to a JSON
//     file.  The Python driver (`heatmap_profiler.py`) consumes that.
//   * `drain_summary()` returns a compact in-memory summary
//     (per-layer aggregate cycles, total cycles, event count).

#pragma once

#include "rdtsc_cycle_counter.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nsos {
namespace profiler {

// Event kind.  Hierarchy is implied by ordering: every OP belongs to
// the most recently opened LAYER, every KERNEL to the most recently
// opened OP, etc.  Drain logic reconstructs the tree.
enum class EventKind : uint8_t {
    PHASE  = 0,
    LAYER  = 1,
    OP     = 2,
    KERNEL = 3,
};

// One profiling event.  Fits in 64 bytes (one cache line) so the
// ring buffer is dense and stays in L1 during a forward pass.
//
// The `name` is a pointer to a static string literal — not a copy.
// Callers MUST pass string literals or pointers with static
// lifetime.  This trades safety for the perf win of no string copy
// in the hot path.
struct alignas(64) Event {
    uint64_t   start_cycle;     // result of read_cycles_serialized() at entry
    uint64_t   end_cycle;       // result at exit
    const char* name;           // static string pointer (NEVER freed)
    int32_t    layer_idx;       // -1 if not layer-scoped
    int32_t    sub_idx;         // expert id, head id, etc; -1 if none
    uint32_t   bytes_in;        // optional, 0 if unknown
    uint32_t   bytes_out;       // optional, 0 if unknown
    uint32_t   flops;           // optional, 0 if unknown.  In megaflops to fit u32.
    EventKind  kind;
    uint8_t    pad[7];          // to 64 bytes
};
static_assert(sizeof(Event) == 64, "Event must be one cache line");

// Aggregated per-layer summary used by the Python report.
struct LayerSummary {
    int32_t  layer_idx;
    uint64_t total_cycles;      // sum across all forward passes
    uint64_t call_count;        // how many forwards observed
    uint64_t min_cycles;        // min across passes (for jitter)
    uint64_t max_cycles;        // max across passes
    uint64_t total_bytes_in;    // optional bandwidth estimate
    uint64_t total_bytes_out;
    uint64_t total_flops;       // in megaflops
};

// Aggregated per-op summary (across all layers, grouped by op name).
struct OpSummary {
    std::string name;
    uint64_t    total_cycles;
    uint64_t    call_count;
    uint64_t    min_cycles;
    uint64_t    max_cycles;
};

// Top-level summary returned by drain_summary().
struct ProfilerSummary {
    uint64_t                  total_events;
    uint64_t                  total_cycles_observed;
    double                    cycles_per_ns;
    double                    wall_seconds;
    std::vector<LayerSummary> layers;
    std::vector<OpSummary>    ops;
};

// The InferenceProfiler instance.  Constructed and held by the
// profiling tool; attached to a JambaModel via
// `model->attach_profiler(profiler.get())`.  The model's forward
// path checks `if (profiler) profiler->begin_op(...)` — one null
// check when disabled.
class InferenceProfiler {
public:
    // Construct with a ring-buffer capacity (events).  Default 64K
    // events x 64 bytes = 4 MB — fits comfortably in any L2.
    explicit InferenceProfiler(size_t ring_capacity = 65536);
    ~InferenceProfiler();

    // Public reset hook for between runs.  Clears the ring buffer
    // and resets sequence counters.  Safe to call from any thread
    // (synchronizes via the same atomic write_idx as the producer).
    void reset() noexcept;

    // Begin/end of a hierarchical scope.  Cheap (rdtsc + one ring
    // buffer write) but NOT free — only call from the model hot
    // path when the profiler is genuinely attached.
    void begin_event(EventKind kind, const char* name,
                     int32_t layer_idx = -1, int32_t sub_idx = -1) noexcept;
    void end_event(uint32_t bytes_in = 0, uint32_t bytes_out = 0,
                   uint32_t flops_megaflops = 0) noexcept;

    // Convenience: scoped RAII guard.
    class Scope {
    public:
        Scope(InferenceProfiler* p, EventKind k, const char* name,
              int32_t layer_idx = -1, int32_t sub_idx = -1) noexcept
            : profiler_(p) {
            if (profiler_) {
                profiler_->begin_event(k, name, layer_idx, sub_idx);
            }
        }
        ~Scope() noexcept {
            if (profiler_) {
                profiler_->end_event(bytes_in_, bytes_out_, flops_);
            }
        }
        void annotate(uint32_t bytes_in, uint32_t bytes_out,
                      uint32_t flops_megaflops) noexcept {
            bytes_in_ = bytes_in;
            bytes_out_ = bytes_out;
            flops_ = flops_megaflops;
        }
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        InferenceProfiler* profiler_;
        uint32_t bytes_in_  = 0;
        uint32_t bytes_out_ = 0;
        uint32_t flops_     = 0;
    };

    // Drain to JSON file.  Atomically truncates and rewrites.
    // Includes raw events AND aggregated summaries.
    bool drain_to_json(const std::string& path) const;

    // Drain to in-memory summary.  Useful for unit tests.
    ProfilerSummary drain_summary() const;

    // Cycle-to-ns calibration.  Called automatically at construction.
    // Can be re-called if the user pinned the CPU to a fixed
    // frequency after construction.
    void recalibrate(int sample_ms = 50) noexcept;

    // Stats: total observed events and overflows since reset.
    uint64_t event_count_observed() const noexcept;
    uint64_t event_count_dropped() const noexcept;

private:
    // Lock-free SPSC ring buffer.  We assume a single producer (the
    // model's forward thread) and a single consumer (the drain
    // function called after forward completes).
    size_t                              capacity_;
    std::unique_ptr<Event[]>            ring_;
    std::atomic<uint64_t>               write_idx_;   // monotonically increasing
    std::atomic<uint64_t>               dropped_;     // count of overflows
    double                              cycles_per_ns_ = 1.0;
    std::chrono::steady_clock::time_point started_at_;

    // Tracks the index of the most recent begin_event so end_event
    // can write back the end_cycle and annotation.  Per-depth so
    // nested scopes don't corrupt each other.  Bounded depth: 16 is
    // far more than we ever need.
    static constexpr size_t MAX_DEPTH = 16;
    uint64_t   open_event_indices_[MAX_DEPTH] = {0};
    uint8_t    depth_                          = 0;
};

}  // namespace profiler
}  // namespace nsos
