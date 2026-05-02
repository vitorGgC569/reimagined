#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nsos {

struct TensorAuditStats {
    std::vector<int> shape;
    int64_t elements = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double l2_norm = 0.0;
    double max_abs = 0.0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    bool finite = true;
};

struct RouterAuditStats {
    int rows = 0;
    int num_experts = 0;
    int top_k = 0;
    std::vector<int> topk_counts;
    std::vector<float> expert_loads;
    double entropy = 0.0;
};

struct LayerAuditRecord {
    uint64_t sequence = 0;
    std::string run_id;
    std::string phase;
    std::string pass;
    std::string block_type;
    std::string tensor_role;
    int step = 0;
    int layer_index = -1;
    TensorAuditStats input;
    TensorAuditStats output;
    double latency_ms = 0.0;
    double grad_l2_norm = 0.0;
    bool has_router = false;
    RouterAuditStats router;
};

struct TokenContextAuditRecord {
    uint64_t sequence = 0;
    std::string run_id;
    std::string phase;
    int step = 0;
    size_t batch_size = 1;
    size_t prompt_tokens_total = 0;
    size_t prompt_tokens_used = 0;
    int context_limit = 0;
    bool truncated = false;
    std::vector<int> token_ids_sample;
};

struct TrainingStepAuditRecord {
    uint64_t sequence = 0;
    std::string run_id;
    std::string phase;
    int step = 0;
    double loss = 0.0;
    double grad_l2_norm = 0.0;
    size_t parameter_count = 0;
};

struct LayerAuditSummary {
    std::string phase;
    size_t records = 0;
    size_t forward_records = 0;
    size_t backward_records = 0;
    size_t router_records = 0;
    size_t token_contexts = 0;
    size_t training_steps = 0;
    size_t total_nan = 0;
    size_t total_inf = 0;
    double max_latency_ms = 0.0;
    double max_l2_norm = 0.0;
    std::vector<int> layers_seen;

    bool healthy() const { return total_nan == 0 && total_inf == 0; }
};

class LayerAuditCollector {
public:
    void set_enabled(bool enabled);
    bool enabled() const;
    void reset();

    void begin_run(const std::string& run_id);
    void set_phase(const std::string& phase);
    void set_step(int step);

    void record_token_context(const std::vector<int>& token_ids_sample,
                              size_t batch_size,
                              size_t prompt_tokens_total,
                              size_t prompt_tokens_used,
                              int context_limit,
                              bool truncated);

    void record_forward(int layer_index,
                        const std::string& block_type,
                        const std::string& tensor_role,
                        const Tensor& input,
                        const Tensor& output,
                        double latency_ms);

    void record_backward(int layer_index,
                         const std::string& block_type,
                         const Tensor& grad_output,
                         const Tensor& grad_input,
                         double latency_ms);

    void record_router(int layer_index,
                       const std::string& block_type,
                       int rows,
                       int num_experts,
                       int top_k,
                       const std::vector<int>& topk_counts,
                       const std::vector<float>& expert_loads);

    void record_training_step(int step,
                              double loss,
                              double grad_l2_norm,
                              size_t parameter_count);

    std::vector<LayerAuditRecord> records() const;
    std::vector<TokenContextAuditRecord> token_contexts() const;
    std::vector<TrainingStepAuditRecord> training_steps() const;

    LayerAuditSummary summarize_phase(const std::string& phase) const;
    bool compare_phase_health(const std::string& lhs_phase,
                              const std::string& rhs_phase,
                              std::string* reason = nullptr) const;
    void write_json(const std::string& path) const;

    static TensorAuditStats summarize_tensor(const Tensor& tensor);

private:
    uint64_t next_sequence_unlocked();

    bool enabled_ = false;
    uint64_t next_sequence_ = 1;
    std::string run_id_ = "default";
    std::string phase_ = "default";
    int step_ = 0;
    std::vector<LayerAuditRecord> records_;
    std::vector<TokenContextAuditRecord> token_contexts_;
    std::vector<TrainingStepAuditRecord> training_steps_;
};

} // namespace nsos
