#include "../include/layer_audit.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nsos {

namespace {

std::mutex& audit_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (unsigned char ch : input) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(ch) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}

template <typename T>
void write_numeric_array(std::ostream& out, const std::vector<T>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
}

void write_tensor_stats(std::ostream& out, const TensorAuditStats& stats) {
    out << "{";
    out << "\"shape\":";
    write_numeric_array(out, stats.shape);
    out << ",\"elements\":" << stats.elements;
    out << ",\"min\":" << stats.min;
    out << ",\"max\":" << stats.max;
    out << ",\"mean\":" << stats.mean;
    out << ",\"stddev\":" << stats.stddev;
    out << ",\"l2_norm\":" << stats.l2_norm;
    out << ",\"max_abs\":" << stats.max_abs;
    out << ",\"nan_count\":" << stats.nan_count;
    out << ",\"inf_count\":" << stats.inf_count;
    out << ",\"finite\":" << (stats.finite ? "true" : "false");
    out << "}";
}

void write_layer_summary(std::ostream& out, const LayerAuditSummary& summary) {
    out << "{";
    out << "\"phase\":\"" << json_escape(summary.phase) << "\"";
    out << ",\"records\":" << summary.records;
    out << ",\"forward_records\":" << summary.forward_records;
    out << ",\"backward_records\":" << summary.backward_records;
    out << ",\"router_records\":" << summary.router_records;
    out << ",\"token_contexts\":" << summary.token_contexts;
    out << ",\"training_steps\":" << summary.training_steps;
    out << ",\"total_nan\":" << summary.total_nan;
    out << ",\"total_inf\":" << summary.total_inf;
    out << ",\"max_latency_ms\":" << summary.max_latency_ms;
    out << ",\"max_l2_norm\":" << summary.max_l2_norm;
    out << ",\"layers_seen\":";
    write_numeric_array(out, summary.layers_seen);
    out << ",\"stored_records\":" << summary.stored_records;
    out << ",\"dropped_records\":" << summary.dropped_records;
    out << ",\"truncated_contexts\":" << summary.truncated_contexts;
    out << ",\"router_entropy_count\":" << summary.router_entropy_count;
    out << ",\"router_entropy_min\":" << summary.router_entropy_min;
    out << ",\"router_entropy_max\":" << summary.router_entropy_max;
    out << ",\"router_entropy_mean\":" << summary.router_entropy_mean;
    out << ",\"router_num_experts_max\":" << summary.router_num_experts_max;
    out << "}";
}

double router_entropy(const std::vector<float>& loads,
                      const std::vector<int>& topk_counts) {
    double total = 0.0;
    for (float value : loads) {
        if (std::isfinite(value) && value > 0.0f) {
            total += static_cast<double>(value);
        }
    }
    if (total <= 0.0) {
        for (int value : topk_counts) {
            if (value > 0) {
                total += static_cast<double>(value);
            }
        }
        if (total <= 0.0) {
            return 0.0;
        }
        double entropy = 0.0;
        for (int value : topk_counts) {
            if (value <= 0) {
                continue;
            }
            const double p = static_cast<double>(value) / total;
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    double entropy = 0.0;
    for (float value : loads) {
        if (!std::isfinite(value) || value <= 0.0f) {
            continue;
        }
        const double p = static_cast<double>(value) / total;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

void add_layer_seen(std::set<int>& seen, int layer_index) {
    if (layer_index >= 0) {
        seen.insert(layer_index);
    }
}

void add_layer_seen(std::vector<int>& seen, int layer_index) {
    if (layer_index < 0) {
        return;
    }
    if (std::find(seen.begin(), seen.end(), layer_index) == seen.end()) {
        seen.push_back(layer_index);
        std::sort(seen.begin(), seen.end());
    }
}

} // namespace

void LayerAuditCollector::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    enabled_ = enabled;
}

bool LayerAuditCollector::enabled() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return enabled_;
}

void LayerAuditCollector::reset() {
    std::lock_guard<std::mutex> lock(audit_mutex());
    next_sequence_ = 1;
    phase_ = "default";
    step_ = 0;
    records_.clear();
    token_contexts_.clear();
    training_steps_.clear();
    phase_summaries_.clear();
}

void LayerAuditCollector::begin_run(const std::string& run_id) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    run_id_ = run_id.empty() ? "default" : run_id;
    phase_ = "default";
    step_ = 0;
    next_sequence_ = 1;
    records_.clear();
    token_contexts_.clear();
    training_steps_.clear();
    phase_summaries_.clear();
}

void LayerAuditCollector::set_phase(const std::string& phase) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    phase_ = phase.empty() ? "default" : phase;
}

void LayerAuditCollector::set_step(int step) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    step_ = step;
}

void LayerAuditCollector::set_storage_policy(bool summary_only,
                                             int record_sample_rate,
                                             size_t max_records_per_phase,
                                             bool store_token_contexts) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    summary_only_ = summary_only;
    record_sample_rate_ = std::max(record_sample_rate, 1);
    max_records_per_phase_ = max_records_per_phase;
    store_token_contexts_ = store_token_contexts;
}

bool LayerAuditCollector::summary_only() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return summary_only_;
}

int LayerAuditCollector::record_sample_rate() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return record_sample_rate_;
}

size_t LayerAuditCollector::max_records_per_phase() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return max_records_per_phase_;
}

bool LayerAuditCollector::store_token_contexts() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return store_token_contexts_;
}

uint64_t LayerAuditCollector::next_sequence_unlocked() {
    return next_sequence_++;
}

LayerAuditCollector::PhaseAuditAccumulator&
LayerAuditCollector::phase_accumulator_unlocked(const std::string& phase) {
    auto [it, inserted] = phase_summaries_.try_emplace(phase);
    if (inserted) {
        it->second.summary.phase = phase;
    }
    return it->second;
}

const LayerAuditCollector::PhaseAuditAccumulator*
LayerAuditCollector::find_phase_accumulator_unlocked(const std::string& phase) const {
    const auto it = phase_summaries_.find(phase);
    if (it == phase_summaries_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool LayerAuditCollector::should_collect_layer_stats_unlocked(
    const PhaseAuditAccumulator& accumulator,
    const std::string& pass,
    int layer_index) const {
    if (pass == "router" || record_sample_rate_ <= 1) {
        return true;
    }
    if (layer_index >= 0 &&
        std::find(accumulator.summary.layers_seen.begin(),
                  accumulator.summary.layers_seen.end(),
                  layer_index) == accumulator.summary.layers_seen.end()) {
        return true;
    }
    const size_t observed = accumulator.summary.forward_records +
                            accumulator.summary.backward_records;
    return (observed % static_cast<size_t>(record_sample_rate_)) == 0;
}

bool LayerAuditCollector::should_store_layer_record_unlocked(
    const PhaseAuditAccumulator& accumulator,
    const std::string& pass) const {
    if (summary_only_) {
        return false;
    }
    if (pass == "router") {
        return max_records_per_phase_ == 0 ||
               accumulator.summary.stored_records < max_records_per_phase_;
    }
    if (max_records_per_phase_ > 0 &&
        accumulator.summary.stored_records >= max_records_per_phase_) {
        return false;
    }
    return true;
}

void LayerAuditCollector::update_layer_summary_unlocked(
    PhaseAuditAccumulator& accumulator,
    const LayerAuditRecord& record,
    bool has_tensor_stats) {
    auto& summary = accumulator.summary;
    ++summary.records;
    if (record.pass == "forward") {
        ++summary.forward_records;
    } else if (record.pass == "backward") {
        ++summary.backward_records;
    } else if (record.pass == "router") {
        ++summary.router_records;
        const double entropy = record.router.entropy;
        if (summary.router_entropy_count == 0) {
            summary.router_entropy_min = entropy;
            summary.router_entropy_max = entropy;
            summary.router_entropy_mean = entropy;
        } else {
            summary.router_entropy_min = std::min(summary.router_entropy_min, entropy);
            summary.router_entropy_max = std::max(summary.router_entropy_max, entropy);
            summary.router_entropy_mean +=
                (entropy - summary.router_entropy_mean) /
                static_cast<double>(summary.router_entropy_count + 1);
        }
        ++summary.router_entropy_count;
        summary.router_num_experts_max =
            std::max(summary.router_num_experts_max, record.router.num_experts);
    }
    add_layer_seen(summary.layers_seen, record.layer_index);
    summary.max_latency_ms = std::max(summary.max_latency_ms, record.latency_ms);
    if (has_tensor_stats) {
        summary.total_nan += record.input.nan_count + record.output.nan_count;
        summary.total_inf += record.input.inf_count + record.output.inf_count;
        summary.max_l2_norm =
            std::max(summary.max_l2_norm,
                     std::max(record.input.l2_norm, record.output.l2_norm));
    }
}

void LayerAuditCollector::mark_layer_record_storage_unlocked(
    PhaseAuditAccumulator& accumulator,
    bool stored) {
    if (stored) {
        ++accumulator.summary.stored_records;
    } else {
        ++accumulator.summary.dropped_records;
    }
}

void LayerAuditCollector::update_training_summary_unlocked(
    PhaseAuditAccumulator& accumulator,
    const TrainingStepAuditRecord&) {
    ++accumulator.summary.training_steps;
}

TensorAuditStats LayerAuditCollector::summarize_tensor(const Tensor& tensor) {
    TensorAuditStats stats;
    stats.shape = tensor.shape.dims;
    stats.elements = tensor.size;
    if (tensor.size <= 0 || tensor.data() == nullptr) {
        return stats;
    }

    Tensor host = tensor.get_device() == Device::GPU ? tensor.cpu() : tensor;
    const float* ptr = host.data();
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    size_t finite_count = 0;

    for (int i = 0; i < host.size; ++i) {
        const float value = ptr[i];
        if (std::isnan(value)) {
            ++stats.nan_count;
            continue;
        }
        if (std::isinf(value)) {
            ++stats.inf_count;
            continue;
        }
        const double d = static_cast<double>(value);
        sum += d;
        sum_sq += d * d;
        min_value = std::min(min_value, d);
        max_value = std::max(max_value, d);
        max_abs = std::max(max_abs, std::abs(d));
        ++finite_count;
    }

    stats.finite = stats.nan_count == 0 && stats.inf_count == 0;
    if (finite_count == 0) {
        return stats;
    }

    stats.min = min_value;
    stats.max = max_value;
    stats.mean = sum / static_cast<double>(finite_count);
    const double mean_sq = sum_sq / static_cast<double>(finite_count);
    const double variance = std::max(0.0, mean_sq - stats.mean * stats.mean);
    stats.stddev = std::sqrt(variance);
    stats.l2_norm = std::sqrt(sum_sq);
    stats.max_abs = max_abs;
    return stats;
}

void LayerAuditCollector::record_token_context(const std::vector<int>& token_ids_sample,
                                               size_t batch_size,
                                               size_t prompt_tokens_total,
                                               size_t prompt_tokens_used,
                                               int context_limit,
                                               bool truncated) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    if (!enabled_) {
        return;
    }
    TokenContextAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.step = step_;
    record.batch_size = std::max<size_t>(batch_size, 1);
    record.prompt_tokens_total = prompt_tokens_total;
    record.prompt_tokens_used = prompt_tokens_used;
    record.context_limit = context_limit;
    record.truncated = truncated;
    const size_t sample_limit = std::min<size_t>(token_ids_sample.size(), 128);
    record.token_ids_sample.assign(token_ids_sample.begin(),
                                   token_ids_sample.begin() + sample_limit);
    auto& accumulator = phase_accumulator_unlocked(record.phase);
    ++accumulator.summary.token_contexts;
    if (truncated) {
        ++accumulator.summary.truncated_contexts;
    }
    if (store_token_contexts_ && !summary_only_) {
        token_contexts_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_forward(int layer_index,
                                         const std::string& block_type,
                                         const std::string& tensor_role,
                                         const Tensor& input,
                                         const Tensor& output,
                                         double latency_ms) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    if (!enabled_) {
        return;
    }
    auto& accumulator = phase_accumulator_unlocked(phase_);
    const bool collect_stats =
        should_collect_layer_stats_unlocked(accumulator, "forward", layer_index);
    LayerAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = "forward";
    record.block_type = block_type;
    record.tensor_role = tensor_role;
    record.step = step_;
    record.layer_index = layer_index;
    if (collect_stats) {
        record.input = summarize_tensor(input);
        record.output = summarize_tensor(output);
    }
    record.latency_ms = latency_ms;
    update_layer_summary_unlocked(accumulator, record, collect_stats);
    const bool store_record =
        collect_stats && should_store_layer_record_unlocked(accumulator, record.pass);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_backward(int layer_index,
                                          const std::string& block_type,
                                          const Tensor& grad_output,
                                          const Tensor& grad_input,
                                          double latency_ms) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    if (!enabled_) {
        return;
    }
    auto& accumulator = phase_accumulator_unlocked(phase_);
    const bool collect_stats =
        should_collect_layer_stats_unlocked(accumulator, "backward", layer_index);
    LayerAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = "backward";
    record.block_type = block_type;
    record.tensor_role = "gradient";
    record.step = step_;
    record.layer_index = layer_index;
    if (collect_stats) {
        record.input = summarize_tensor(grad_output);
        record.output = summarize_tensor(grad_input);
    }
    record.latency_ms = latency_ms;
    record.grad_l2_norm = record.output.l2_norm;
    update_layer_summary_unlocked(accumulator, record, collect_stats);
    const bool store_record =
        collect_stats && should_store_layer_record_unlocked(accumulator, record.pass);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_router(int layer_index,
                                        const std::string& block_type,
                                        int rows,
                                        int num_experts,
                                        int top_k,
                                        const std::vector<int>& topk_counts,
                                        const std::vector<float>& expert_loads) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    if (!enabled_) {
        return;
    }
    LayerAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = "router";
    record.block_type = block_type;
    record.tensor_role = "moe_router";
    record.step = step_;
    record.layer_index = layer_index;
    record.has_router = true;
    record.router.rows = rows;
    record.router.num_experts = num_experts;
    record.router.top_k = top_k;
    record.router.topk_counts = topk_counts;
    record.router.expert_loads = expert_loads;
    record.router.entropy = router_entropy(expert_loads, topk_counts);
    auto& accumulator = phase_accumulator_unlocked(record.phase);
    update_layer_summary_unlocked(accumulator, record, true);
    const bool store_record = should_store_layer_record_unlocked(accumulator, record.pass);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_training_step(int step,
                                               double loss,
                                               double grad_l2_norm,
                                               size_t parameter_count) {
    std::lock_guard<std::mutex> lock(audit_mutex());
    if (!enabled_) {
        return;
    }
    TrainingStepAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.step = step;
    record.loss = loss;
    record.grad_l2_norm = grad_l2_norm;
    record.parameter_count = parameter_count;
    auto& accumulator = phase_accumulator_unlocked(record.phase);
    update_training_summary_unlocked(accumulator, record);
    training_steps_.push_back(std::move(record));
    step_ = step;
}

std::vector<LayerAuditRecord> LayerAuditCollector::records() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return records_;
}

std::vector<TokenContextAuditRecord> LayerAuditCollector::token_contexts() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return token_contexts_;
}

std::vector<TrainingStepAuditRecord> LayerAuditCollector::training_steps() const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    return training_steps_;
}

LayerAuditSummary LayerAuditCollector::summarize_phase(const std::string& phase) const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    if (const auto* accumulator = find_phase_accumulator_unlocked(phase)) {
        return accumulator->summary;
    }

    LayerAuditSummary summary;
    summary.phase = phase;
    std::set<int> layers_seen;

    for (const auto& record : records_) {
        if (record.phase != phase) {
            continue;
        }
        ++summary.records;
        if (record.pass == "forward") {
            ++summary.forward_records;
        } else if (record.pass == "backward") {
            ++summary.backward_records;
        } else if (record.pass == "router") {
            ++summary.router_records;
        }
        summary.total_nan += record.input.nan_count + record.output.nan_count;
        summary.total_inf += record.input.inf_count + record.output.inf_count;
        summary.max_latency_ms = std::max(summary.max_latency_ms, record.latency_ms);
        summary.max_l2_norm =
            std::max(summary.max_l2_norm,
                     std::max(record.input.l2_norm, record.output.l2_norm));
        add_layer_seen(layers_seen, record.layer_index);
    }

    for (const auto& context : token_contexts_) {
        if (context.phase == phase) {
            ++summary.token_contexts;
        }
    }
    for (const auto& step : training_steps_) {
        if (step.phase == phase) {
            ++summary.training_steps;
        }
    }

    summary.layers_seen.assign(layers_seen.begin(), layers_seen.end());
    return summary;
}

bool LayerAuditCollector::compare_phase_health(const std::string& lhs_phase,
                                               const std::string& rhs_phase,
                                               std::string* reason) const {
    const LayerAuditSummary lhs = summarize_phase(lhs_phase);
    const LayerAuditSummary rhs = summarize_phase(rhs_phase);
    auto fail = [&](const std::string& message) {
        if (reason) {
            *reason = message;
        }
        return false;
    };

    if (!lhs.healthy()) {
        return fail("left phase has NaN/Inf: " + lhs_phase);
    }
    if (!rhs.healthy()) {
        return fail("right phase has NaN/Inf: " + rhs_phase);
    }
    if (lhs.forward_records == 0 || rhs.forward_records == 0) {
        return fail("both phases must contain forward records");
    }
    if (lhs.layers_seen.empty() || rhs.layers_seen.empty()) {
        return fail("both phases must see at least one layer");
    }
    if (lhs.layers_seen != rhs.layers_seen) {
        return fail("phase layer coverage differs");
    }
    if (reason) {
        reason->clear();
    }
    return true;
}

void LayerAuditCollector::write_json(const std::string& path) const {
    std::lock_guard<std::mutex> lock(audit_mutex());
    namespace fs = std::filesystem;
    const fs::path output_path(path);
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream out(output_path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Could not write audit JSON: " + path);
    }
    out << std::setprecision(10);
    out << "{\n";
    out << "  \"run_id\":\"" << json_escape(run_id_) << "\",\n";
    out << "  \"storage_policy\":{";
    out << "\"summary_only\":" << (summary_only_ ? "true" : "false");
    out << ",\"record_sample_rate\":" << record_sample_rate_;
    out << ",\"max_records_per_phase\":" << max_records_per_phase_;
    out << ",\"store_token_contexts\":" << (store_token_contexts_ ? "true" : "false");
    out << "},\n";
    out << "  \"phase_summaries\":[\n";
    size_t summary_index = 0;
    for (const auto& [_, accumulator] : phase_summaries_) {
        out << "    ";
        write_layer_summary(out, accumulator.summary);
        if (++summary_index < phase_summaries_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"records\":[\n";
    for (size_t i = 0; i < records_.size(); ++i) {
        const auto& record = records_[i];
        out << "    {";
        out << "\"sequence\":" << record.sequence;
        out << ",\"phase\":\"" << json_escape(record.phase) << "\"";
        out << ",\"pass\":\"" << json_escape(record.pass) << "\"";
        out << ",\"step\":" << record.step;
        out << ",\"layer_index\":" << record.layer_index;
        out << ",\"block_type\":\"" << json_escape(record.block_type) << "\"";
        out << ",\"tensor_role\":\"" << json_escape(record.tensor_role) << "\"";
        out << ",\"latency_ms\":" << record.latency_ms;
        out << ",\"grad_l2_norm\":" << record.grad_l2_norm;
        out << ",\"input\":";
        write_tensor_stats(out, record.input);
        out << ",\"output\":";
        write_tensor_stats(out, record.output);
        if (record.has_router) {
            out << ",\"router\":{";
            out << "\"rows\":" << record.router.rows;
            out << ",\"num_experts\":" << record.router.num_experts;
            out << ",\"top_k\":" << record.router.top_k;
            out << ",\"topk_counts\":";
            write_numeric_array(out, record.router.topk_counts);
            out << ",\"expert_loads\":";
            write_numeric_array(out, record.router.expert_loads);
            out << ",\"entropy\":" << record.router.entropy;
            out << "}";
        }
        out << "}";
        if (i + 1 < records_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"token_contexts\":[\n";
    for (size_t i = 0; i < token_contexts_.size(); ++i) {
        const auto& context = token_contexts_[i];
        out << "    {";
        out << "\"sequence\":" << context.sequence;
        out << ",\"phase\":\"" << json_escape(context.phase) << "\"";
        out << ",\"step\":" << context.step;
        out << ",\"batch_size\":" << context.batch_size;
        out << ",\"prompt_tokens_total\":" << context.prompt_tokens_total;
        out << ",\"prompt_tokens_used\":" << context.prompt_tokens_used;
        out << ",\"context_limit\":" << context.context_limit;
        out << ",\"truncated\":" << (context.truncated ? "true" : "false");
        out << ",\"token_ids_sample\":";
        write_numeric_array(out, context.token_ids_sample);
        out << "}";
        if (i + 1 < token_contexts_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"training_steps\":[\n";
    for (size_t i = 0; i < training_steps_.size(); ++i) {
        const auto& step = training_steps_[i];
        out << "    {";
        out << "\"sequence\":" << step.sequence;
        out << ",\"phase\":\"" << json_escape(step.phase) << "\"";
        out << ",\"step\":" << step.step;
        out << ",\"loss\":" << step.loss;
        out << ",\"grad_l2_norm\":" << step.grad_l2_norm;
        out << ",\"parameter_count\":" << step.parameter_count;
        out << "}";
        if (i + 1 < training_steps_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    if (!out) {
        throw std::runtime_error("Could not flush audit JSON: " + path);
    }
}

} // namespace nsos
