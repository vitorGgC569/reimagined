#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace nsos {

class Parameter;

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
    size_t zero_count = 0;
    size_t subnormal_count = 0;
    size_t positive_count = 0;
    size_t negative_count = 0;
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

struct ParameterAuditRecord {
    uint64_t sequence = 0;
    std::string run_id;
    std::string phase;
    int step = 0;
    std::string name;
    std::string base_name;
    int layer_index = -1;
    std::string component;
    std::string role;
    bool trainable = true;
    bool optimizer_applied = false;
    uint64_t version_before = 0;
    uint64_t version_after = 0;
    TensorAuditStats weight_before;
    TensorAuditStats gradient;
    TensorAuditStats update;
    TensorAuditStats weight_after;
    double grad_to_weight_ratio = 0.0;
    double update_to_weight_ratio = 0.0;
    double update_to_grad_ratio = 0.0;
    double gradient_update_cosine = 0.0;
    size_t changed_elements = 0;
    size_t ternary_elements_before = 0;
    size_t ternary_elements_after = 0;
    std::string weight_sha256_before;
    std::string gradient_sha256;
    std::string update_sha256;
    std::string weight_sha256_after;
};

// Diagnoses whether the Mamba, Attention and FFN branches complement,
// duplicate or cancel one another. On forward, signal=raw branch activation and
// contribution=post-gate residual contribution.  On backward, signal=gradient
// entering each branch and contribution=that branch's gradient contribution to
// the shared block input after its own pre-norm VJP.
struct HybridInteractionAuditRecord {
    uint64_t sequence = 0;
    std::string run_id;
    std::string phase;
    std::string pass;
    int step = 0;
    int layer_index = -1;
    TensorAuditStats mamba_signal;
    TensorAuditStats attention_signal;
    TensorAuditStats ffn_signal;
    TensorAuditStats mamba_contribution;
    TensorAuditStats attention_contribution;
    TensorAuditStats ffn_contribution;
    TensorAuditStats combined_contribution;
    // Backward-compatible Mamba/Attention pair metrics.
    double signal_cosine = 0.0;
    double contribution_cosine = 0.0;
    double mamba_ffn_signal_cosine = 0.0;
    double attention_ffn_signal_cosine = 0.0;
    double mamba_ffn_contribution_cosine = 0.0;
    double attention_ffn_contribution_cosine = 0.0;
    double attention_to_mamba_signal_ratio = 0.0;
    double attention_to_mamba_contribution_ratio = 0.0;
    double ffn_to_mamba_signal_ratio = 0.0;
    double ffn_to_mamba_contribution_ratio = 0.0;
    // 0 means no vector cancellation; 1 means the three contributions cancel
    // completely. Pairwise alignment/redundancy is represented by cosines.
    double cancellation_fraction = 0.0;
};

struct LayerAuditSummary {
    std::string phase;
    size_t records = 0;
    size_t forward_records = 0;
    size_t backward_records = 0;
    size_t router_records = 0;
    size_t token_contexts = 0;
    size_t training_steps = 0;
    size_t parameter_records = 0;
    size_t changed_parameter_records = 0;
    size_t hybrid_interaction_records = 0;
    size_t parameter_nan = 0;
    size_t parameter_inf = 0;
    size_t total_nan = 0;
    size_t total_inf = 0;
    double max_latency_ms = 0.0;
    double max_l2_norm = 0.0;
    std::vector<int> layers_seen;
    size_t stored_records = 0;
    size_t dropped_records = 0;
    size_t truncated_contexts = 0;
    size_t router_entropy_count = 0;
    double router_entropy_min = 0.0;
    double router_entropy_max = 0.0;
    double router_entropy_mean = 0.0;
    int router_num_experts_max = 0;
    double hybrid_signal_cosine_mean = 0.0;
    double hybrid_contribution_cosine_mean = 0.0;
    double hybrid_attention_to_mamba_mean = 0.0;
    double hybrid_mamba_ffn_contribution_cosine_mean = 0.0;
    double hybrid_attention_ffn_contribution_cosine_mean = 0.0;
    double hybrid_ffn_to_mamba_mean = 0.0;
    double hybrid_cancellation_max = 0.0;
    size_t hybrid_forward_interaction_records = 0;
    double hybrid_forward_signal_cosine_mean = 0.0;
    double hybrid_forward_contribution_cosine_mean = 0.0;
    double hybrid_forward_attention_to_mamba_mean = 0.0;
    double hybrid_forward_cancellation_max = 0.0;
    size_t hybrid_backward_interaction_records = 0;
    double hybrid_backward_signal_cosine_mean = 0.0;
    double hybrid_backward_contribution_cosine_mean = 0.0;
    double hybrid_backward_attention_to_mamba_mean = 0.0;
    double hybrid_backward_cancellation_max = 0.0;

    bool healthy() const {
        return total_nan == 0 && total_inf == 0 &&
               parameter_nan == 0 && parameter_inf == 0;
    }
};

class LayerAuditCollector {
public:
    void set_enabled(bool enabled);
    bool enabled() const;
    void reset();

    void begin_run(const std::string& run_id);
    void set_phase(const std::string& phase);
    void set_step(int step);
    void set_storage_policy(bool summary_only,
                            int record_sample_rate,
                            size_t max_records_per_phase,
                            bool store_token_contexts);
    void set_parameter_audit_policy(bool enabled,
                                    int step_sample_rate,
                                    size_t max_records,
                                    size_t max_snapshot_bytes);
    bool summary_only() const;
    int record_sample_rate() const;
    size_t max_records_per_phase() const;
    bool store_token_contexts() const;
    bool parameter_audit_enabled() const;
    int parameter_step_sample_rate() const;
    size_t max_parameter_records() const;
    size_t max_parameter_snapshot_bytes() const;

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

    void record_hybrid_interaction(
        int layer_index,
        const std::string& pass,
        const Tensor& mamba_signal,
        const Tensor& attention_signal,
        const Tensor& ffn_signal,
        const Tensor& mamba_contribution,
        const Tensor& attention_contribution,
        const Tensor& ffn_contribution);

    // Captures exact host snapshots only when both the collector and parameter
    // auditing are enabled and the step passes the sampling policy. The pair is
    // transactional: complete emits before/gradient/update/after records;
    // abort drops pending snapshots after an optimizer exception.
    void begin_parameter_step(int step,
                              const std::vector<Parameter*>& parameters);
    void complete_parameter_step(int step, bool optimizer_applied);
    void abort_parameter_step() noexcept;

    std::vector<LayerAuditRecord> records() const;
    std::vector<TokenContextAuditRecord> token_contexts() const;
    std::vector<TrainingStepAuditRecord> training_steps() const;
    std::vector<ParameterAuditRecord> parameter_records() const;
    std::vector<HybridInteractionAuditRecord>
    hybrid_interaction_records() const;

    LayerAuditSummary summarize_phase(const std::string& phase) const;
    bool compare_phase_health(const std::string& lhs_phase,
                              const std::string& rhs_phase,
                              std::string* reason = nullptr) const;
    void write_json(const std::string& path) const;

    static TensorAuditStats summarize_tensor(const Tensor& tensor);

private:
    struct PhaseAuditAccumulator {
        LayerAuditSummary summary;
    };

    uint64_t next_sequence_unlocked();
    PhaseAuditAccumulator& phase_accumulator_unlocked(const std::string& phase);
    const PhaseAuditAccumulator* find_phase_accumulator_unlocked(const std::string& phase) const;
    bool should_collect_layer_stats_unlocked(const PhaseAuditAccumulator& accumulator,
                                             const std::string& pass,
                                             int layer_index) const;
    bool should_store_layer_record_unlocked(const PhaseAuditAccumulator& accumulator,
                                            const std::string& pass) const;
    void update_layer_summary_unlocked(PhaseAuditAccumulator& accumulator,
                                       const LayerAuditRecord& record,
                                       bool has_tensor_stats);
    void mark_layer_record_storage_unlocked(PhaseAuditAccumulator& accumulator,
                                            bool stored);
    void update_training_summary_unlocked(PhaseAuditAccumulator& accumulator,
                                          const TrainingStepAuditRecord& record);
    void update_parameter_summary_unlocked(
        PhaseAuditAccumulator& accumulator,
        const ParameterAuditRecord& record);
    void update_hybrid_summary_unlocked(
        PhaseAuditAccumulator& accumulator,
        const HybridInteractionAuditRecord& record);

    struct PendingParameterAudit {
        Parameter* parameter = nullptr;
        std::string name;
        std::string base_name;
        uint64_t version_before = 0;
        Tensor weight_before;
        Tensor gradient;
    };

    // Collector-local synchronization: independent models/audit streams must
    // not serialize through a process-global mutex.
    mutable std::mutex mutex_;
    bool enabled_ = false;
    uint64_t next_sequence_ = 1;
    std::string run_id_ = "default";
    std::string phase_ = "default";
    int step_ = 0;
    bool summary_only_ = false;
    int record_sample_rate_ = 1;
    size_t max_records_per_phase_ = 0;
    bool store_token_contexts_ = true;
    bool parameter_audit_enabled_ = true;
    int parameter_step_sample_rate_ = 1;
    size_t max_parameter_records_ = 0;
    size_t max_parameter_snapshot_bytes_ =
        static_cast<size_t>(1024) * 1024 * 1024;
    int pending_parameter_step_ = -1;
    std::string pending_parameter_run_id_;
    std::string pending_parameter_phase_;
    std::vector<PendingParameterAudit> pending_parameters_;
    std::vector<LayerAuditRecord> records_;
    std::vector<TokenContextAuditRecord> token_contexts_;
    std::vector<TrainingStepAuditRecord> training_steps_;
    std::vector<ParameterAuditRecord> parameter_records_;
    std::vector<HybridInteractionAuditRecord>
        hybrid_interaction_records_;
    std::map<std::string, PhaseAuditAccumulator> phase_summaries_;
};

} // namespace nsos
