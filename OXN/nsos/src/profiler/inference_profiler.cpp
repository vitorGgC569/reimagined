// inference_profiler.cpp — Implementation of the OPT-IN profiling
// module declared in include/profiler/inference_profiler.h.
//
// Threading model:
//   Single-producer / single-consumer (SPSC).  The model's forward
//   thread is the only producer of events; the consumer is whoever
//   calls drain_to_json/drain_summary after the forward returns.
//   We use a monotonically-increasing 64-bit write_idx with modulo
//   indexing into the ring buffer.  No locks, no fences beyond what
//   rdtsc already inserts.
//
// Calibration:
//   On construction we sleep for `sample_ms` and measure cycles
//   delta; cycles_per_ns is computed from that.  Modern Intel/AMD
//   have invariant TSC (constant frequency regardless of CPU
//   frequency scaling) so this calibration is stable for the
//   lifetime of the process.  ARM cntvct is similarly stable.

#include "../../include/profiler/inference_profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>
#include <utility>

namespace nsos {
namespace profiler {

// ────────────────────────────────────────────────────────────────────
// Cycle counter calibration
// ────────────────────────────────────────────────────────────────────

double calibrate_cycles_per_ns(int sample_ms) noexcept {
    if (sample_ms <= 0) sample_ms = 50;
    const auto wall_start = std::chrono::steady_clock::now();
    const uint64_t cyc_start = read_cycles_serialized();
    std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
    const uint64_t cyc_end = read_cycles_serialized();
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns =
        std::chrono::duration<double, std::nano>(wall_end - wall_start).count();
    if (wall_ns <= 0.0) return 1.0;
    const double cyc_delta = static_cast<double>(cyc_end - cyc_start);
    return cyc_delta / wall_ns;
}

// ────────────────────────────────────────────────────────────────────
// InferenceProfiler
// ────────────────────────────────────────────────────────────────────

InferenceProfiler::InferenceProfiler(size_t ring_capacity)
    : capacity_(ring_capacity == 0 ? 1 : ring_capacity),
      ring_(new Event[ring_capacity == 0 ? 1 : ring_capacity]),
      write_idx_(0),
      dropped_(0),
      started_at_(std::chrono::steady_clock::now()) {
    // Zero-init.  Important because end_event reads the prior
    // entry's start_cycle to compute deltas if the user only
    // captured end markers.
    std::memset(ring_.get(), 0, capacity_ * sizeof(Event));
    cycles_per_ns_ = calibrate_cycles_per_ns(50);
}

InferenceProfiler::~InferenceProfiler() = default;

void InferenceProfiler::recalibrate(int sample_ms) noexcept {
    cycles_per_ns_ = calibrate_cycles_per_ns(sample_ms);
}

void InferenceProfiler::reset() noexcept {
    // Reset producer index.  We don't bother zeroing the ring
    // contents — new events will overwrite as they're written.
    write_idx_.store(0, std::memory_order_release);
    dropped_.store(0, std::memory_order_release);
    depth_ = 0;
    for (size_t i = 0; i < MAX_DEPTH; ++i) {
        open_event_indices_[i] = 0;
    }
    started_at_ = std::chrono::steady_clock::now();
}

void InferenceProfiler::begin_event(EventKind kind, const char* name,
                                     int32_t layer_idx, int32_t sub_idx) noexcept {
    // Bounds check on depth — silently drop if we exceed (callers
    // are misusing the API, but we don't want to crash production
    // through any subtle bug here).
    if (depth_ >= MAX_DEPTH) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const uint64_t idx = write_idx_.fetch_add(1, std::memory_order_acq_rel);
    const size_t slot = static_cast<size_t>(idx % capacity_);
    Event& e = ring_[slot];
    e.kind        = kind;
    e.name        = name;
    e.layer_idx   = layer_idx;
    e.sub_idx     = sub_idx;
    e.bytes_in    = 0;
    e.bytes_out   = 0;
    e.flops       = 0;
    e.end_cycle   = 0;
    // Read cycle counter LAST so the measurement isn't biased by
    // the slot-write overhead above.
    e.start_cycle = read_cycles_serialized();
    open_event_indices_[depth_] = idx;
    ++depth_;
}

void InferenceProfiler::end_event(uint32_t bytes_in, uint32_t bytes_out,
                                   uint32_t flops_megaflops) noexcept {
    // Read cycle counter FIRST so end_cycle isn't biased by the
    // bookkeeping that follows.
    const uint64_t end = read_cycles_serialized();
    if (depth_ == 0) {
        // begin/end mismatch — drop silently and increment counter.
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    --depth_;
    const uint64_t idx = open_event_indices_[depth_];
    const size_t slot = static_cast<size_t>(idx % capacity_);
    Event& e = ring_[slot];
    e.end_cycle  = end;
    e.bytes_in   = bytes_in;
    e.bytes_out  = bytes_out;
    e.flops      = flops_megaflops;
}

uint64_t InferenceProfiler::event_count_observed() const noexcept {
    return write_idx_.load(std::memory_order_acquire);
}

uint64_t InferenceProfiler::event_count_dropped() const noexcept {
    return dropped_.load(std::memory_order_acquire);
}

// ────────────────────────────────────────────────────────────────────
// Drain helpers
// ────────────────────────────────────────────────────────────────────

static const char* event_kind_name(EventKind k) {
    switch (k) {
        case EventKind::PHASE:  return "phase";
        case EventKind::LAYER:  return "layer";
        case EventKind::OP:     return "op";
        case EventKind::KERNEL: return "kernel";
    }
    return "unknown";
}

ProfilerSummary InferenceProfiler::drain_summary() const {
    ProfilerSummary out;
    out.cycles_per_ns = cycles_per_ns_;
    const auto wall_end = std::chrono::steady_clock::now();
    out.wall_seconds =
        std::chrono::duration<double>(wall_end - started_at_).count();

    const uint64_t total_events = write_idx_.load(std::memory_order_acquire);
    out.total_events = total_events;

    // Walk events in arrival order (oldest first).  If we wrapped,
    // the oldest still-valid entries are at (write_idx - capacity)
    // and we walk forward `capacity` events.  If we didn't wrap,
    // we walk from 0 to write_idx.
    const uint64_t first = (total_events > capacity_)
                               ? (total_events - capacity_)
                               : 0;
    const uint64_t last = total_events;

    // Aggregations.
    std::map<int32_t, LayerSummary> layers_map;
    std::map<std::string, OpSummary> ops_map;

    for (uint64_t i = first; i < last; ++i) {
        const Event& e = ring_[static_cast<size_t>(i % capacity_)];
        if (e.end_cycle <= e.start_cycle) continue;  // unclosed scope
        const uint64_t dt = e.end_cycle - e.start_cycle;
        out.total_cycles_observed += dt;

        if (e.kind == EventKind::LAYER) {
            LayerSummary& ls = layers_map[e.layer_idx];
            ls.layer_idx = e.layer_idx;
            ls.total_cycles += dt;
            ls.call_count += 1;
            ls.min_cycles = (ls.call_count == 1) ? dt : std::min(ls.min_cycles, dt);
            ls.max_cycles = std::max(ls.max_cycles, dt);
            ls.total_bytes_in  += e.bytes_in;
            ls.total_bytes_out += e.bytes_out;
            ls.total_flops     += e.flops;
        } else if (e.kind == EventKind::OP) {
            const std::string key = e.name ? e.name : "(null)";
            OpSummary& os = ops_map[key];
            if (os.call_count == 0) os.name = key;
            os.total_cycles += dt;
            os.call_count += 1;
            os.min_cycles = (os.call_count == 1) ? dt : std::min(os.min_cycles, dt);
            os.max_cycles = std::max(os.max_cycles, dt);
        }
    }

    out.layers.reserve(layers_map.size());
    for (auto& kv : layers_map) {
        out.layers.push_back(kv.second);
    }
    out.ops.reserve(ops_map.size());
    for (auto& kv : ops_map) {
        out.ops.push_back(std::move(kv.second));
    }
    std::sort(out.layers.begin(), out.layers.end(),
              [](const LayerSummary& a, const LayerSummary& b) {
                  return a.layer_idx < b.layer_idx;
              });
    std::sort(out.ops.begin(), out.ops.end(),
              [](const OpSummary& a, const OpSummary& b) {
                  return a.total_cycles > b.total_cycles;
              });
    return out;
}

bool InferenceProfiler::drain_to_json(const std::string& path) const {
    // Atomic write: write to tmp then rename.  Avoids torn reads if
    // the Python driver polls while we drain.
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::out | std::ios::trunc);
    if (!out) return false;

    out << "{\n";
    out << "  \"cycles_per_ns\": " << cycles_per_ns_ << ",\n";
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_s =
        std::chrono::duration<double>(wall_end - started_at_).count();
    out << "  \"wall_seconds\": " << wall_s << ",\n";
    out << "  \"event_count_observed\": "
        << event_count_observed() << ",\n";
    out << "  \"event_count_dropped\": "
        << event_count_dropped() << ",\n";

    // Events array (raw, in arrival order)
    out << "  \"events\": [\n";
    const uint64_t total_events = write_idx_.load(std::memory_order_acquire);
    const uint64_t first = (total_events > capacity_)
                               ? (total_events - capacity_)
                               : 0;
    bool first_emit = true;
    for (uint64_t i = first; i < total_events; ++i) {
        const Event& e = ring_[static_cast<size_t>(i % capacity_)];
        if (e.end_cycle <= e.start_cycle) continue;
        if (!first_emit) out << ",\n";
        first_emit = false;
        out << "    {"
            << "\"kind\":\""  << event_kind_name(e.kind) << "\","
            << "\"name\":\""  << (e.name ? e.name : "") << "\","
            << "\"layer\":"   << e.layer_idx << ","
            << "\"sub\":"     << e.sub_idx << ","
            << "\"start\":"   << e.start_cycle << ","
            << "\"end\":"     << e.end_cycle << ","
            << "\"cycles\":"  << (e.end_cycle - e.start_cycle) << ","
            << "\"bytes_in\":"  << e.bytes_in << ","
            << "\"bytes_out\":" << e.bytes_out << ","
            << "\"mflops\":"  << e.flops
            << "}";
    }
    out << "\n  ],\n";

    // Per-layer summary
    const ProfilerSummary summary = drain_summary();
    out << "  \"layer_summary\": [\n";
    for (size_t i = 0; i < summary.layers.size(); ++i) {
        const auto& ls = summary.layers[i];
        if (i > 0) out << ",\n";
        out << "    {"
            << "\"layer\":"        << ls.layer_idx << ","
            << "\"total_cycles\":" << ls.total_cycles << ","
            << "\"calls\":"        << ls.call_count << ","
            << "\"min_cycles\":"   << ls.min_cycles << ","
            << "\"max_cycles\":"   << ls.max_cycles << ","
            << "\"bytes_in\":"     << ls.total_bytes_in << ","
            << "\"bytes_out\":"    << ls.total_bytes_out << ","
            << "\"mflops\":"       << ls.total_flops
            << "}";
    }
    out << "\n  ],\n";

    // Per-op summary
    out << "  \"op_summary\": [\n";
    for (size_t i = 0; i < summary.ops.size(); ++i) {
        const auto& os = summary.ops[i];
        if (i > 0) out << ",\n";
        out << "    {"
            << "\"name\":\""       << os.name << "\","
            << "\"total_cycles\":" << os.total_cycles << ","
            << "\"calls\":"        << os.call_count << ","
            << "\"min_cycles\":"   << os.min_cycles << ","
            << "\"max_cycles\":"   << os.max_cycles
            << "}";
    }
    out << "\n  ]\n";
    out << "}\n";
    out.close();

    // Atomic rename.  On POSIX this is one syscall; on Windows
    // std::rename returns 0 on success.
    std::remove(path.c_str());
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

}  // namespace profiler
}  // namespace nsos
