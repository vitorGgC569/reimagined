#pragma once
#include "autograd.h"
#include "optimizer_runtime_policy.h"
#include <vector>
#include <memory>
#include <functional>

namespace nsos {
struct GpuSparseAdamSlot {
    Parameter* parameter = nullptr;
    float learning_rate = 0; // absolute LR, caller applies schedule/criticality
    bool weight_decay = false;
    bool dense_contributed = false; // ignored for explicitly device-bound slots
};
struct GpuSparseAdamState {
    std::string name;
    uint64_t version = 0;
    bool initialized = false;
    Tensor m, v; // absent iff !initialized, FP32 otherwise
};
struct GpuSparseAdamOptions {
    float beta1 = 0.9f, beta2 = 0.999f, bc1 = 0, bc2 = 0, eps = 1e-8f;
    float weight_decay = 0, max_norm = 1;
    int accumulation_steps = 1;
    bool deterministic = true;
};
struct GpuSparseAdamResult { bool committed = false; double norm = 0; };

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
    void abort();
    std::vector<GpuSparseAdamState> snapshot() const;
    void restore(const std::vector<GpuSparseAdamState>& states);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace nsos
