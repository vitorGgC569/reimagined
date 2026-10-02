#pragma once
#include "autograd.h"
#include "optimizer_runtime_policy.h"
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

namespace nsos {
enum class GpuSparseAlgorithm : unsigned char { AdamW = 0, MuonNs5Fp32 = 1 };
struct GpuSparseAdamSlot {
    Parameter* parameter = nullptr;
    float learning_rate = 0; // absolute LR, caller applies schedule/criticality
    bool weight_decay = false;
    bool dense_contributed = false; // ignored for explicitly device-bound slots
    GpuSparseAlgorithm algorithm = GpuSparseAlgorithm::AdamW;
};
struct GpuSparseAdamState {
    std::string name;
    uint64_t version = 0;
    bool initialized = false;
    Tensor m, v; // absent iff !initialized, FP32 otherwise
    GpuSparseAlgorithm algorithm = GpuSparseAlgorithm::AdamW;
};
struct GpuSparseAdamOptions {
    float beta1 = 0.9f, beta2 = 0.999f, bc1 = 0, bc2 = 0, eps = 1e-8f;
    float weight_decay = 0, max_norm = 1;
    int accumulation_steps = 1;
    bool deterministic = true;
    // Post-VJP transactional update+gradient-clear epilogue. Activity metadata
    // belongs to the producer and is reset by the enclosing group, never here.
    bool fused_epilogue = false;
    bool clear_gradients = false;
};
struct GpuSparseAdamResult { bool committed = false; double norm = 0; };
// Process-local counts; only successfully finite-gated/version-published steps.
struct GpuSparseAdamDispatchCounters {
    std::uint64_t adamw_device_commits=0;
    std::uint64_t adamw_fused_epilogue_commits=0;
    std::uint64_t muon_adam_commits=0;
    std::uint64_t muon_matrix_directions=0;
    std::uint64_t fused_gradient_clear_commits=0;
};
GpuSparseAdamDispatchCounters gpu_sparse_adam_dispatch_counters() noexcept;

// Per-Trainer owner, not thread_local/global. Registry order is immutable after
// first configure; domains and dense activity refresh per accumulation group.
// Owns reserved moments, logical presence, stable descriptors, rollback, and
// device versions. Exports ONLY logically present moments for checkpoints.
class GpuSparseAdam {
public:
    GpuSparseAdam();
    ~GpuSparseAdam();
    GpuSparseAdam(const GpuSparseAdam&) = delete;
    GpuSparseAdam& operator=(const GpuSparseAdam&) = delete;
    // Callback must OR upstream sticky status into the borrowed device int on
    // this owner's lane. Never clear it, retain it, or build a host cohort.
    void configure(const std::vector<GpuSparseAdamSlot>& slots,
                   std::function<void(int*)> merge_upstream_status = {});
    GpuSparseAdamResult step(const GpuSparseAdamOptions& options);
    // Consume a one-use proof from a successful clearing epilogue. The caller
    // retains exclusive gradient ownership until this call, on the owning lane.
    // Revalidates registry/storage/version, resets host activity only, and lets
    // Trainer omit its next full-bank zero launch. Returns false after abort,
    // restore or storage change; inactive tracked slots overwrite on first VJP.
    bool consume_fused_gradient_reset(const std::vector<Parameter*>& parameters);
    void abort();
    std::vector<GpuSparseAdamState> snapshot() const;
    void restore(const std::vector<GpuSparseAdamState>& states);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace nsos
