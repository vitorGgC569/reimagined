#include "../include/trainer.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include "../include/layer_audit.h"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

namespace {

#ifdef USE_CUDA
bool trainer_force_cuda_sync() {
    static const bool force_sync = [] {
        const char* env = std::getenv("NSOS_CUDA_SYNC");
        return env != nullptr && std::string(env) == "1";
    }();
    return force_sync;
}

void trainer_check_cuda(const char* op) {
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + " failed: " +
                                 cudaGetErrorString(launch_status));
    }
    if (trainer_force_cuda_sync()) {
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            throw std::runtime_error(std::string(op) + " synchronize failed: " +
                                     cudaGetErrorString(sync_status));
        }
    }
}

void scale_tensor_inplace(Tensor& tensor, float scale) {
    if (tensor.size == 0 || std::abs(scale - 1.0f) <= 1e-6f) {
        return;
    }
    if (tensor.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
        launch_scale_inplace_kernel(tensor.data(), scale, tensor.size);
        trainer_check_cuda("launch_scale_inplace_kernel");
        return;
    }
    float* ptr = tensor.data();
    for (int i = 0; i < tensor.size; ++i) {
        ptr[i] *= scale;
    }
}

void zero_tensor_inplace(Tensor& tensor) {
    if (tensor.size == 0) {
        return;
    }
    if (tensor.get_device() == Device::GPU) {
        cudaMemset(tensor.data(), 0, static_cast<size_t>(tensor.size) * sizeof(float));
        trainer_check_cuda("cudaMemset");
        return;
    }
    std::fill_n(tensor.data(), tensor.size, 0.0f);
}

void copy_tensor_bytes(float* dst,
                       Device dst_device,
                       const float* src,
                       Device src_device,
                       size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (dst_device == Device::GPU || src_device == Device::GPU) {
        cudaMemcpyKind kind = cudaMemcpyDefault;
        if (dst_device == Device::GPU && src_device == Device::GPU) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (dst_device == Device::GPU && src_device == Device::CPU) {
            kind = cudaMemcpyHostToDevice;
        } else if (dst_device == Device::CPU && src_device == Device::GPU) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyHostToHost;
        }
        cudaMemcpy(dst, src, bytes, kind);
        trainer_check_cuda("cudaMemcpy");
        return;
    }
    std::memcpy(dst, src, bytes);
}

bool can_use_gpu_optimizer(const Parameter& parameter, const Tensor& m, const Tensor& v) {
    return parameter.data.get_device() == Device::GPU &&
           parameter.grad.get_device() == Device::GPU &&
           m.get_device() == Device::GPU &&
           v.get_device() == Device::GPU &&
           gpu_custom_kernels_supported();
}
#else
void scale_tensor_inplace(Tensor& tensor, float scale) {
    if (tensor.size == 0 || std::abs(scale - 1.0f) <= 1e-6f) {
        return;
    }
    float* ptr = tensor.data();
    for (int i = 0; i < tensor.size; ++i) {
        ptr[i] *= scale;
    }
}

void zero_tensor_inplace(Tensor& tensor) {
    if (tensor.size == 0) {
        return;
    }
    std::fill_n(tensor.data(), tensor.size, 0.0f);
}

void copy_tensor_bytes(float* dst,
                       Device dst_device,
                       const float* src,
                       Device src_device,
                       size_t bytes) {
    (void)dst_device;
    (void)src_device;
    if (bytes == 0) {
        return;
    }
    std::memcpy(dst, src, bytes);
}
#endif

void restore_staged_tensor(Tensor& dst, const Tensor& staged) {
    const Device dst_device = dst.get_device();
    if (dst.size == staged.size && dst.shape == staged.shape && dst.size > 0) {
        copy_tensor_bytes(dst.data(),
                          dst_device,
                          staged.data(),
                          staged.get_device(),
                          static_cast<size_t>(dst.size) * sizeof(float));
        return;
    }
    dst = (staged.get_device() == dst_device) ? staged.clone() : staged.to(dst_device);
}

bool trainer_prefers_reference_training_path(const Trainer& trainer) {
    return trainer.model != nullptr;
}

bool auxiliary_stack_requested(const Trainer& trainer) {
    return trainer.phase_scheduler.auxiliary_stack_enabled &&
           (trainer.phase_scheduler.auxiliary_session_adapt_enabled ||
            trainer.phase_scheduler.auxiliary_reasoning_enabled ||
            trainer.phase_scheduler.auxiliary_memory_enabled);
}

bool auxiliary_stack_due_this_step(const Trainer& trainer) {
    const int every = std::max(trainer.phase_scheduler.auxiliary_every_steps, 1);
    return (trainer.global_step_count % every) == 0;
}

float tensor_mean_row_norm(const Tensor& tensor) {
    if (tensor.size == 0 || tensor.shape.size() != 2) {
        return 0.0f;
    }
    Tensor host = tensor.get_device() == Device::GPU ? tensor.cpu() : tensor;
    const int rows = host.shape[0];
    const int dim = host.shape[1];
    if (rows <= 0 || dim <= 0) {
        return 0.0f;
    }
    const float* data = host.data();
    float total = 0.0f;
    for (int row = 0; row < rows; ++row) {
        float norm_sq = 0.0f;
        const float* row_ptr = data + static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int col = 0; col < dim; ++col) {
            norm_sq += row_ptr[col] * row_ptr[col];
        }
        total += std::sqrt(norm_sq);
    }
    return total / static_cast<float>(rows);
}

float tensor_mean_row_delta_norm(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.size == 0 || lhs.shape != rhs.shape || lhs.shape.size() != 2) {
        return 0.0f;
    }
    Tensor lhs_host = lhs.get_device() == Device::GPU ? lhs.cpu() : lhs;
    Tensor rhs_host = rhs.get_device() == Device::GPU ? rhs.cpu() : rhs;
    const int rows = lhs_host.shape[0];
    const int dim = lhs_host.shape[1];
    if (rows <= 0 || dim <= 0) {
        return 0.0f;
    }
    const float* lhs_ptr = lhs_host.data();
    const float* rhs_ptr = rhs_host.data();
    float total = 0.0f;
    for (int row = 0; row < rows; ++row) {
        float norm_sq = 0.0f;
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int col = 0; col < dim; ++col) {
            const float diff = rhs_ptr[base + col] - lhs_ptr[base + col];
            norm_sq += diff * diff;
        }
        total += std::sqrt(norm_sq);
    }
    return total / static_cast<float>(rows);
}

float tensor_mean_row_cosine(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.size == 0 || lhs.shape != rhs.shape || lhs.shape.size() != 2) {
        return 0.0f;
    }
    Tensor lhs_host = lhs.get_device() == Device::GPU ? lhs.cpu() : lhs;
    Tensor rhs_host = rhs.get_device() == Device::GPU ? rhs.cpu() : rhs;
    const int rows = lhs_host.shape[0];
    const int dim = lhs_host.shape[1];
    if (rows <= 0 || dim <= 0) {
        return 0.0f;
    }
    const float* lhs_ptr = lhs_host.data();
    const float* rhs_ptr = rhs_host.data();
    float total = 0.0f;
    for (int row = 0; row < rows; ++row) {
        float dot = 0.0f;
        float lhs_norm_sq = 0.0f;
        float rhs_norm_sq = 0.0f;
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int col = 0; col < dim; ++col) {
            const float a = lhs_ptr[base + col];
            const float b = rhs_ptr[base + col];
            dot += a * b;
            lhs_norm_sq += a * a;
            rhs_norm_sq += b * b;
        }
        const float denom = std::sqrt(std::max(lhs_norm_sq, 1e-8f) * std::max(rhs_norm_sq, 1e-8f));
        total += (denom > 0.0f) ? (dot / denom) : 0.0f;
    }
    return total / static_cast<float>(rows);
}

void accumulate_auxiliary_stats(AuxiliaryStackStats& total, const AuxiliaryStackStats& bucket) {
    total.bucket_count += bucket.bucket_count;
    total.due_count += bucket.due_count;
    total.applied_count += bucket.applied_count;
    total.reasoning_count += bucket.reasoning_count;
    total.memory_count += bucket.memory_count;
    total.session_adapt_count += bucket.session_adapt_count;
    total.sample_count += bucket.sample_count;
    total.prompt_tokens += bucket.prompt_tokens;
    total.answer_tokens += bucket.answer_tokens;
    total.prompt_state_norm += bucket.prompt_state_norm;
    total.target_state_norm += bucket.target_state_norm;
    total.reason_delta_norm += bucket.reason_delta_norm;
    total.reason_cosine += bucket.reason_cosine;
    total.memory_delta_norm += bucket.memory_delta_norm;
    total.memory_cosine += bucket.memory_cosine;
    total.final_target_delta_norm += bucket.final_target_delta_norm;
}

void finalize_auxiliary_stats(AuxiliaryStackStats& stats) {
    if (stats.applied_count > 0) {
        const float denom = static_cast<float>(stats.applied_count);
        stats.prompt_state_norm /= denom;
        stats.target_state_norm /= denom;
        stats.final_target_delta_norm /= denom;
    }
    if (stats.reasoning_count > 0) {
        const float denom = static_cast<float>(stats.reasoning_count);
        stats.reason_delta_norm /= denom;
        stats.reason_cosine /= denom;
    }
    if (stats.memory_count > 0) {
        const float denom = static_cast<float>(stats.memory_count);
        stats.memory_delta_norm /= denom;
        stats.memory_cosine /= denom;
    }
}

MemorySystem& shared_training_memory_store(int dim, int scope) {
    static std::mutex store_mutex;
    static std::unordered_map<long long, std::unique_ptr<MemorySystem>> stores;
    const int resolved_dim = std::max(dim, 1);
    const long long resolved_scope = static_cast<long long>(std::max(scope, 0));
    const long long key =
        (resolved_scope << 32) ^ static_cast<unsigned long long>(resolved_dim);
    std::lock_guard<std::mutex> lock(store_mutex);
    auto& slot = stores[key];
    if (!slot) {
        slot = std::make_unique<MemorySystem>(resolved_dim);
    }
    return *slot;
}

Tensor extract_last_token_states(const Tensor& trunk, const std::vector<int>& valid_lengths) {
    if (trunk.shape.size() != 3) {
        throw std::runtime_error("extract_last_token_states expects [batch, seq, dim]");
    }

    const int batch = trunk.shape[0];
    const int seq_len = trunk.shape[1];
    const int dim = trunk.shape[2];
    if (static_cast<int>(valid_lengths.size()) != batch) {
        throw std::runtime_error("extract_last_token_states requires valid lengths for each batch item");
    }

    Tensor states({batch, dim}, trunk.get_device());
    float* dst = states.data();
    const float* src = trunk.data();
    for (int row = 0; row < batch; ++row) {
        const int token_index =
            std::clamp(valid_lengths[static_cast<size_t>(row)] - 1, 0, std::max(seq_len - 1, 0));
        const size_t src_offset =
            (static_cast<size_t>(row) * static_cast<size_t>(seq_len) +
             static_cast<size_t>(token_index)) *
            static_cast<size_t>(dim);
        const size_t dst_offset = static_cast<size_t>(row) * static_cast<size_t>(dim);
        copy_tensor_bytes(dst + dst_offset,
                          states.get_device(),
                          src + src_offset,
                          trunk.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
    }
    return states;
}

std::vector<int> trim_suffix_tokens(const std::vector<int>& tokens, int max_tokens) {
    if (max_tokens <= 0 || static_cast<int>(tokens.size()) <= max_tokens) {
        return tokens;
    }
    return std::vector<int>(tokens.end() - max_tokens, tokens.end());
}

Tensor blend_state_tensors(const Tensor& base, const Tensor& aux, float aux_mix) {
    if (base.size == 0) {
        return base;
    }
    if (aux.size == 0 || aux.shape != base.shape) {
        return base.clone();
    }

    const float mix = std::clamp(aux_mix, 0.0f, 1.0f);
    if (mix <= 1e-6f) {
        return base.clone();
    }

    Tensor base_host = base.get_device() == Device::GPU ? base.cpu() : base;
    Tensor aux_host = aux.get_device() == Device::GPU ? aux.cpu() : aux;
    Tensor blended = base_host.clone();
    float* out = blended.data();
    const float* aux_ptr = aux_host.data();
    for (int i = 0; i < blended.size; ++i) {
        out[i] = out[i] * (1.0f - mix) + aux_ptr[i] * mix;
    }
    return base.get_device() == Device::GPU ? blended.to(Device::GPU) : blended;
}

Tensor reason_state_batch(Trainer& trainer, const Tensor& states) {
    if (!trainer.model || states.size == 0 || states.shape.size() != 2) {
        return states.clone();
    }

    const int batch = states.shape[0];
    const int dim = states.shape[1];
    Tensor refined({batch, dim}, states.get_device());
    float* refined_ptr = refined.data();
    const float* states_ptr = states.data();
    const int iterations = std::max(trainer.phase_scheduler.auxiliary_reasoning_iterations, 1);
    const int simulations = std::max(trainer.phase_scheduler.auxiliary_reasoning_simulations, 1);

    for (int row = 0; row < batch; ++row) {
        Tensor seed({dim}, states.get_device());
        copy_tensor_bytes(seed.data(),
                          seed.get_device(),
                          states_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          states.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
        Tensor current = seed;
        for (int iter = 0; iter < iterations; ++iter) {
            current = trainer.model->reason(current, simulations);
        }
        copy_tensor_bytes(refined_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          refined.get_device(),
                          current.data(),
                          current.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
    }
    return refined;
}

Tensor recall_memory_batch(const Tensor& states, int memory_scope) {
    if (states.size == 0 || states.shape.size() != 2) {
        return states.clone();
    }

    const int batch = states.shape[0];
    const int dim = states.shape[1];
    MemorySystem& memory = shared_training_memory_store(dim, memory_scope);
    Tensor recalled({batch, dim}, states.get_device());
    float* recalled_ptr = recalled.data();
    const float* states_ptr = states.data();
    for (int row = 0; row < batch; ++row) {
        Tensor query({dim}, states.get_device());
        copy_tensor_bytes(query.data(),
                          query.get_device(),
                          states_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          states.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
        Tensor value = memory.retrieve(query);
        copy_tensor_bytes(recalled_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          recalled.get_device(),
                          value.data(),
                          value.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
    }
    return recalled;
}

void store_memory_batch(const Tensor& states, int memory_scope) {
    if (states.size == 0 || states.shape.size() != 2) {
        return;
    }
    const int batch = states.shape[0];
    const int dim = states.shape[1];
    MemorySystem& memory = shared_training_memory_store(dim, memory_scope);
    const float* states_ptr = states.data();
    for (int row = 0; row < batch; ++row) {
        Tensor value({dim}, states.get_device());
        copy_tensor_bytes(value.data(),
                          value.get_device(),
                          states_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          states.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
        memory.store_episodic(value);
    }
}

AuxiliaryStackStats apply_auxiliary_stack_before_forward(
    Trainer& trainer,
    const std::vector<std::vector<int>>& prompt_batch,
    const std::vector<std::vector<int>>& answer_batch) {
    AuxiliaryStackStats stats;
    stats.bucket_count = 1;
    stats.sample_count = static_cast<int>(prompt_batch.size());
    if (!trainer.model || !auxiliary_stack_requested(trainer) || prompt_batch.empty() ||
        prompt_batch.size() != answer_batch.size() || !auxiliary_stack_due_this_step(trainer)) {
        return stats;
    }
    stats.due_count = 1;

    auto* model = trainer.model;
    const ModelConfig& config = model->model_config();
    if (!config.use_ttt && !trainer.phase_scheduler.auxiliary_memory_enabled &&
        !trainer.phase_scheduler.auxiliary_reasoning_enabled) {
        return stats;
    }

    std::vector<int> prompt_lengths;
    std::vector<int> answer_lengths;
    std::vector<std::vector<int>> trimmed_prompts;
    std::vector<std::vector<int>> trimmed_answers;
    prompt_lengths.reserve(prompt_batch.size());
    answer_lengths.reserve(answer_batch.size());
    trimmed_prompts.reserve(prompt_batch.size());
    trimmed_answers.reserve(answer_batch.size());
    const int prompt_max_tokens = std::max(trainer.phase_scheduler.auxiliary_prompt_max_tokens, 1);
    const int answer_max_tokens = std::max(trainer.phase_scheduler.auxiliary_answer_max_tokens, 1);
    for (size_t index = 0; index < prompt_batch.size(); ++index) {
        if (prompt_batch[index].empty() || answer_batch[index].empty()) {
            return stats;
        }
        trimmed_prompts.push_back(trim_suffix_tokens(prompt_batch[index], prompt_max_tokens));
        trimmed_answers.push_back(trim_suffix_tokens(answer_batch[index], answer_max_tokens));
        prompt_lengths.push_back(static_cast<int>(trimmed_prompts.back().size()));
        answer_lengths.push_back(static_cast<int>(trimmed_answers.back().size()));
        stats.prompt_tokens += static_cast<int>(trimmed_prompts.back().size());
        stats.answer_tokens += static_cast<int>(trimmed_answers.back().size());
    }

    const bool previous_training_mode = model->training_mode();
    model->set_training_mode(false);
    Context aux_ctx;
    Tensor prompt_trunk = model->forward_trunk_batch(trimmed_prompts, &aux_ctx);
    Tensor answer_trunk = model->forward_trunk_batch(trimmed_answers, &aux_ctx);
    Tensor prompt_states = extract_last_token_states(prompt_trunk, prompt_lengths);
    Tensor answer_states = extract_last_token_states(answer_trunk, answer_lengths);
    Tensor target_states = prompt_states.clone();
    const Tensor original_target_states = prompt_states.clone();
    stats.applied_count = 1;
    stats.prompt_state_norm = tensor_mean_row_norm(prompt_states);
    stats.target_state_norm = tensor_mean_row_norm(answer_states);

    if (trainer.phase_scheduler.auxiliary_reasoning_enabled) {
        stats.reasoning_count = 1;
        Tensor reasoned_states = reason_state_batch(trainer, prompt_states);
        stats.reason_delta_norm = tensor_mean_row_delta_norm(prompt_states, reasoned_states);
        stats.reason_cosine = tensor_mean_row_cosine(prompt_states, reasoned_states);
        target_states = blend_state_tensors(target_states, reasoned_states, 0.5f);
    }

    if (trainer.phase_scheduler.auxiliary_memory_enabled) {
        stats.memory_count = 1;
        Tensor target_before_memory = target_states.clone();
        Tensor recalled_states =
            recall_memory_batch(prompt_states, trainer.phase_scheduler.auxiliary_memory_scope);
        stats.memory_cosine = tensor_mean_row_cosine(prompt_states, recalled_states);
        target_states = blend_state_tensors(
            target_states, recalled_states, trainer.phase_scheduler.auxiliary_memory_blend);
        stats.memory_delta_norm = tensor_mean_row_delta_norm(target_before_memory, target_states);
        store_memory_batch(prompt_states, trainer.phase_scheduler.auxiliary_memory_scope);
    }
    stats.final_target_delta_norm = tensor_mean_row_delta_norm(original_target_states, target_states);

    model->set_training_mode(previous_training_mode);
    if (trainer.phase_scheduler.auxiliary_session_adapt_enabled && config.use_ttt) {
        stats.session_adapt_count = 1;
        model->session_adapt(prompt_states, target_states);
    }
    return stats;
}

void zero_model_gradients(const std::vector<Parameter*>& params) {
    for (auto* param : params) {
        if (param) {
            param->zero_grad();
        }
    }
}

struct EffectiveQatSchedule {
    int semantic_warmup_steps = 0;
    int qat_start_step = 0;
};

EffectiveQatSchedule resolve_effective_qat_schedule(const Trainer& trainer) {
    EffectiveQatSchedule resolved;
    resolved.semantic_warmup_steps = std::max(trainer.phase_scheduler.semantic_warmup_steps, 0);
    resolved.qat_start_step = std::max(trainer.phase_scheduler.qat_start_step,
                                       resolved.semantic_warmup_steps + 1);

    const int total_steps = std::max(trainer.total_training_steps, 1);
    if (resolved.semantic_warmup_steps >= total_steps ||
        resolved.qat_start_step > total_steps) {
        const int scaled_warmup = std::max(8, static_cast<int>(std::round(total_steps * 0.25f)));
        const int scaled_qat = std::max(
            scaled_warmup + 1,
            static_cast<int>(std::round(total_steps * 0.60f)));
        resolved.semantic_warmup_steps =
            std::min(resolved.semantic_warmup_steps, std::max(scaled_warmup, 1));
        resolved.qat_start_step =
            std::min(resolved.qat_start_step, std::max(scaled_qat, resolved.semantic_warmup_steps + 1));
    }

    resolved.semantic_warmup_steps =
        std::clamp(resolved.semantic_warmup_steps, 0, std::max(total_steps - 1, 0));
    resolved.qat_start_step =
        std::clamp(resolved.qat_start_step, resolved.semantic_warmup_steps + 1, total_steps);
    return resolved;
}

void scale_gradients(const std::vector<Parameter*>& params, float scale) {
    if (std::abs(scale - 1.0f) <= 1e-6f) {
        return;
    }
    for (auto* p : params) {
        if (!p || p->grad.size == 0) continue;
        scale_tensor_inplace(p->grad, scale);
    }
}

float clip_gradients(const std::vector<Parameter*>& params, float max_norm) {
    float total_norm_sq = 0.0f;
    for (auto* p : params) {
        if (!p || p->grad.size == 0) continue;
        const float norm = p->grad.norm();
        total_norm_sq += norm * norm;
    }
    const float total_norm = std::sqrt(total_norm_sq);

    if (total_norm > max_norm) {
        const float coeff = max_norm / (total_norm + 1e-6f);
        for (auto* p : params) {
            if (!p || p->grad.size == 0) continue;
            scale_tensor_inplace(p->grad, coeff);
        }
    }
    return total_norm;
}

std::vector<int> make_targets(const std::vector<int>& tokens,
                              const std::vector<int>& explicit_targets) {
    if (!explicit_targets.empty()) return explicit_targets;
    if (tokens.size() < 2) throw std::runtime_error("Need at least 2 tokens");
    return std::vector<int>(tokens.begin() + 1, tokens.end());
}

std::vector<int> make_inputs(const std::vector<int>& tokens,
                             const std::vector<int>& explicit_targets) {
    if (!explicit_targets.empty()) return tokens;
    return std::vector<int>(tokens.begin(), tokens.end() - 1);
}

bool should_apply_weight_decay(const Parameter& parameter) {
    const std::string& name = parameter.base_name.empty() ? parameter.name : parameter.base_name;
    return name.find("bias") == std::string::npos &&
           name.find("norm") == std::string::npos &&
           name.find("magnitude") == std::string::npos &&
           name.find("flat_") == std::string::npos;
}

void apply_progressive_qat_phase(Trainer& trainer) {
    if (!trainer.model) {
        return;
    }

    auto bitlinear_layers = trainer.model->collect_bitlinear_layers();
    const auto effective = resolve_effective_qat_schedule(trainer);
    const bool scheduler_enabled = trainer.phase_scheduler.progressive_qat_enabled;
    const bool quantized_active =
        scheduler_enabled &&
        trainer.global_step_count >= effective.qat_start_step;
    const bool quantized_runtime_active =
        quantized_active && !trainer_prefers_reference_training_path(trainer);
    const bool transition_active =
        scheduler_enabled &&
        trainer.global_step_count >= effective.semantic_warmup_steps &&
        trainer.global_step_count < effective.qat_start_step;

    int precision_bits = 8;
    if (quantized_active) {
        precision_bits = std::max(trainer.phase_scheduler.quantized_precision_bits, 2);
    } else if (transition_active) {
        precision_bits = 4;
    }

    for (BitLinear* layer : bitlinear_layers) {
        if (!layer) {
            continue;
        }
        layer->set_precision_mode(precision_bits);
        layer->set_reference_path(!quantized_runtime_active);
    }
}

void apply_qat_regularization(Trainer& trainer) {
    if (!trainer.model || !trainer.phase_scheduler.progressive_qat_enabled) {
        return;
    }
    const auto effective = resolve_effective_qat_schedule(trainer);
    if (trainer.global_step_count < effective.semantic_warmup_steps) {
        return;
    }

    const float ramp_denom = static_cast<float>(std::max(
        effective.qat_start_step - effective.semantic_warmup_steps, 1));
    const float ramp =
        std::clamp(static_cast<float>(trainer.global_step_count -
                                      effective.semantic_warmup_steps) /
                       ramp_denom,
                   0.0f, 1.0f);
    const float regularization =
        trainer.phase_scheduler.ternary_regularization * std::max(ramp, 0.0f);
    if (regularization <= 0.0f) {
        return;
    }

    for (BitLinear* layer : trainer.model->collect_bitlinear_layers()) {
        if (!layer || !layer->has_full_precision_weight()) {
            continue;
        }
        Tensor ternary_target = layer->quantize_weights(layer->weight.data);
        Tensor penalty_grad = layer->weight.data.sub(ternary_target).mul(regularization);
        layer->weight.add_grad(penalty_grad);
    }
}

void apply_moe_aux_regularization(Trainer& trainer) {
    if (!trainer.model || trainer.moe_aux_loss_scale <= 0.0f) {
        return;
    }

    for (auto& layer : trainer.model->layers) {
        if (!layer || !layer->router || !layer->router->gate) {
            continue;
        }
        auto& loads = layer->router->expert_loads;
        if (loads.empty()) {
            continue;
        }
        const float mean_load =
            std::accumulate(loads.begin(), loads.end(), 0.0f) /
            static_cast<float>(loads.size());
        Parameter& gate_weight = layer->router->gate->weight;
        if (gate_weight.grad.size == 0) {
            gate_weight.grad = Tensor::zeros(gate_weight.data.shape.dims, gate_weight.data.get_device());
        }
        if (gate_weight.grad.shape.size() != 2 ||
            gate_weight.grad.shape[0] != static_cast<int>(loads.size())) {
            continue;
        }

        Tensor grad_host =
            gate_weight.grad.get_device() == Device::GPU ? gate_weight.grad.cpu() : gate_weight.grad;
        float* grad_ptr = grad_host.data();
        const int row_width = grad_host.shape[1];
        for (size_t expert = 0; expert < loads.size(); ++expert) {
            const float imbalance =
                (loads[expert] - mean_load) * layer->router->aux_loss_coef * trainer.moe_aux_loss_scale;
            for (int col = 0; col < row_width; ++col) {
                grad_ptr[static_cast<int>(expert) * row_width + col] += imbalance;
            }
        }
        if (gate_weight.grad.get_device() == Device::GPU) {
            gate_weight.grad.copy_from(grad_host.to(Device::GPU));
        } else {
            gate_weight.grad = grad_host;
        }
    }
}

float compute_current_lr(const Trainer& trainer) {
    const float warmup_steps = static_cast<float>(std::max(trainer.warmup_steps, 1));
    if (trainer.global_step_count <= trainer.warmup_steps) {
        return trainer.learning_rate *
               static_cast<float>(trainer.global_step_count) / warmup_steps;
    }

    const float total_steps = static_cast<float>(
        std::max(trainer.total_training_steps, trainer.warmup_steps + 1));
    const float decay_span = std::max(total_steps - warmup_steps, 1.0f);
    float progress =
        static_cast<float>(trainer.global_step_count - trainer.warmup_steps) / decay_span;
    progress = std::clamp(progress, 0.0f, 1.0f);

    const float cosine = 0.5f * (1.0f + std::cos(3.14159265f * progress));
    const float floor = trainer.learning_rate * trainer.min_learning_rate_scale;
    return floor + (trainer.learning_rate - floor) * cosine;
}

void apply_optimizer_step(Trainer& trainer,
                          const std::vector<Parameter*>& params,
                          int accumulation_steps,
                          float* grad_norm_out = nullptr) {
    scale_gradients(params, 1.0f / std::max(accumulation_steps, 1));
    const float grad_norm = clip_gradients(params, trainer.max_grad_norm);
    if (grad_norm_out) {
        *grad_norm_out = grad_norm;
    }

    trainer.global_step_count++;
    const float cur_lr = compute_current_lr(trainer);
    const float bc1 = 1.0f - std::pow(trainer.beta1, trainer.global_step_count);
    const float bc2 = 1.0f - std::pow(trainer.beta2, trainer.global_step_count);

    for (auto* p : params) {
        if (!p || p->grad.size == 0) continue;

        if (trainer.m_state.find(p) == trainer.m_state.end()) {
            trainer.m_state[p] = Tensor::zeros(p->data.shape.dims, p->data.get_device());
            trainer.v_state[p] = Tensor::zeros(p->data.shape.dims, p->data.get_device());
        }

        Tensor& m_tensor = trainer.m_state[p];
        Tensor& v_tensor = trainer.v_state[p];

#ifdef USE_CUDA
        if (can_use_gpu_optimizer(*p, m_tensor, v_tensor)) {
            launch_adamw_update_kernel(
                p->data.data(),
                p->grad.data(),
                m_tensor.data(),
                v_tensor.data(),
                p->data.size,
                trainer.beta1,
                trainer.beta2,
                bc1,
                bc2,
                cur_lr,
                trainer.eps,
                trainer.weight_decay,
                should_apply_weight_decay(*p) ? 1 : 0);
            trainer_check_cuda("launch_adamw_update_kernel");
            p->mark_updated();
            continue;
        }
#endif

        float* w = p->data.data();
        const float* g = p->grad.data();
        float* m = m_tensor.data();
        float* v = v_tensor.data();

        for (int i = 0; i < p->data.size; ++i) {
            m[i] = trainer.beta1 * m[i] + (1.0f - trainer.beta1) * g[i];
            v[i] = trainer.beta2 * v[i] + (1.0f - trainer.beta2) * g[i] * g[i];

            const float m_hat = m[i] / bc1;
            const float v_hat = v[i] / bc2;

            if (trainer.weight_decay > 0.0f && should_apply_weight_decay(*p)) {
                w[i] -= cur_lr * trainer.weight_decay * w[i];
            }
            w[i] -= cur_lr * m_hat / (std::sqrt(v_hat) + trainer.eps);
        }
        p->mark_updated();
    }
}

void record_training_audit_step(Trainer& trainer,
                                double loss,
                                double grad_norm,
                                size_t parameter_count) {
    if (!trainer.model) {
        return;
    }
    LayerAuditCollector* audit = trainer.model->audit_collector();
    if (audit && audit->enabled()) {
        audit->record_training_step(trainer.global_step_count,
                                    loss,
                                    grad_norm,
                                    parameter_count);
    }
}

void apply_supervised_gradient_weights(const Trainer& trainer,
                                       const std::vector<int>& answer_tokens,
                                       int vocab,
                                       Tensor& answer_grad) {
    if (answer_tokens.empty() || vocab <= 0 || answer_grad.size == 0) {
        return;
    }

    float* grad_ptr = answer_grad.data();
    const int last_row = static_cast<int>(answer_tokens.size()) - 1;
    for (int row = 0; row <= last_row; ++row) {
        float scale = 1.0f;
        if (row == 0) {
            scale *= std::max(trainer.first_token_loss_scale, 0.0f);
        }
        if (row == last_row && answer_tokens[row] == trainer.eos_token_id) {
            scale *= std::max(trainer.eos_loss_scale, 0.0f);
        }
        if (std::abs(scale - 1.0f) <= 1e-6f) {
            continue;
        }
        float* row_ptr = grad_ptr + row * vocab;
        for (int col = 0; col < vocab; ++col) {
            row_ptr[col] *= scale;
        }
    }
}

void apply_repetition_unlikelihood(const Trainer& trainer,
                                   const std::vector<int>& answer_tokens,
                                   const Tensor& answer_logits,
                                   Tensor& answer_grad) {
    const float scale = std::max(trainer.repetition_unlikelihood_scale, 0.0f);
    if (scale <= 0.0f || answer_tokens.size() < 2 || answer_grad.size == 0) {
        return;
    }

    Tensor probs = answer_logits.softmax(-1);
    Tensor probs_host = probs.get_device() == Device::GPU ? probs.cpu() : probs;
    Tensor grad_host = answer_grad.get_device() == Device::GPU ? answer_grad.cpu() : answer_grad.clone();

    const int vocab = answer_logits.shape.back();
    const float* prob_ptr = probs_host.data();
    float* grad_ptr = grad_host.data();

    for (int row = 1; row < static_cast<int>(answer_tokens.size()); ++row) {
        const int target_token = answer_tokens[row];
        int negative_ids[4];
        int negative_count = 0;
        const int window_start = std::max(0, row - 4);
        for (int prev = row - 1; prev >= window_start; --prev) {
            const int candidate = answer_tokens[prev];
            if (candidate == target_token || candidate == trainer.eos_token_id) {
                continue;
            }
            bool seen = false;
            for (int idx = 0; idx < negative_count; ++idx) {
                if (negative_ids[idx] == candidate) {
                    seen = true;
                    break;
                }
            }
            if (!seen && negative_count < 4) {
                negative_ids[negative_count++] = candidate;
            }
        }
        if (negative_count == 0) {
            continue;
        }

        const float* row_probs = prob_ptr + row * vocab;
        float* row_grad = grad_ptr + row * vocab;

        for (int neg_index = 0; neg_index < negative_count; ++neg_index) {
            const int neg_token = negative_ids[neg_index];
            if (neg_token < 0 || neg_token >= vocab) {
                continue;
            }
            const float p_neg = row_probs[neg_token];
            if (p_neg <= 1e-6f || p_neg >= 1.0f - 1e-6f) {
                continue;
            }
            const float denom = std::max(1.0f - p_neg, 1e-6f);
            const float factor = scale * p_neg / denom;
            for (int col = 0; col < vocab; ++col) {
                row_grad[col] -= factor * row_probs[col];
            }
            row_grad[neg_token] += factor;
        }
    }

    restore_staged_tensor(answer_grad, grad_host);
}

void apply_supervised_gradient_weights_batch(const Trainer& trainer,
                                             const std::vector<std::vector<int>>& answer_batch,
                                             int vocab,
                                             Tensor& answer_grad) {
    if (answer_batch.empty() || vocab <= 0 || answer_grad.size == 0) {
        return;
    }
    if (answer_grad.shape.size() != 3 ||
        answer_grad.shape[0] != static_cast<int>(answer_batch.size())) {
        throw std::runtime_error(
            "apply_supervised_gradient_weights_batch expects [batch, seq, vocab]");
    }

    const int batch_size = answer_grad.shape[0];
    const int seq_len = answer_grad.shape[1];
    float* grad_ptr = answer_grad.data();
    for (int batch = 0; batch < batch_size; ++batch) {
        const auto& answer_tokens = answer_batch[static_cast<size_t>(batch)];
        if (static_cast<int>(answer_tokens.size()) != seq_len) {
            throw std::runtime_error(
                "apply_supervised_gradient_weights_batch requires equal answer lengths");
        }
        const int last_row = static_cast<int>(answer_tokens.size()) - 1;
        for (int row = 0; row <= last_row; ++row) {
            float scale = 1.0f;
            if (row == 0) {
                scale *= std::max(trainer.first_token_loss_scale, 0.0f);
            }
            if (row == last_row && answer_tokens[static_cast<size_t>(row)] == trainer.eos_token_id) {
                scale *= std::max(trainer.eos_loss_scale, 0.0f);
            }
            if (std::abs(scale - 1.0f) <= 1e-6f) {
                continue;
            }
            float* row_ptr = grad_ptr + ((batch * seq_len + row) * vocab);
            for (int col = 0; col < vocab; ++col) {
                row_ptr[col] *= scale;
            }
        }
    }
}

void apply_repetition_unlikelihood_batch(const Trainer& trainer,
                                         const std::vector<std::vector<int>>& answer_batch,
                                         const Tensor& answer_logits,
                                         Tensor& answer_grad) {
    const float scale = std::max(trainer.repetition_unlikelihood_scale, 0.0f);
    if (scale <= 0.0f || answer_batch.empty() || answer_grad.size == 0) {
        return;
    }
    if (answer_logits.shape.size() != 3 || answer_grad.shape.size() != 3 ||
        answer_logits.shape[0] != static_cast<int>(answer_batch.size()) ||
        answer_grad.shape[0] != static_cast<int>(answer_batch.size())) {
        throw std::runtime_error(
            "apply_repetition_unlikelihood_batch expects [batch, seq, vocab] tensors");
    }

    Tensor probs = answer_logits.softmax(-1);
    Tensor probs_host = probs.get_device() == Device::GPU ? probs.cpu() : probs;
    Tensor grad_host = answer_grad.get_device() == Device::GPU ? answer_grad.cpu() : answer_grad.clone();

    const int batch_size = answer_logits.shape[0];
    const int seq_len = answer_logits.shape[1];
    const int vocab = answer_logits.shape[2];
    const float* prob_ptr = probs_host.data();
    float* grad_ptr = grad_host.data();

    for (int batch = 0; batch < batch_size; ++batch) {
        const auto& answer_tokens = answer_batch[static_cast<size_t>(batch)];
        if (static_cast<int>(answer_tokens.size()) != seq_len || seq_len < 2) {
            continue;
        }

        for (int row = 1; row < seq_len; ++row) {
            const int target_token = answer_tokens[static_cast<size_t>(row)];
            int negative_ids[4];
            int negative_count = 0;
            const int window_start = std::max(0, row - 4);
            for (int prev = row - 1; prev >= window_start; --prev) {
                const int candidate = answer_tokens[static_cast<size_t>(prev)];
                if (candidate == target_token || candidate == trainer.eos_token_id) {
                    continue;
                }
                bool seen = false;
                for (int idx = 0; idx < negative_count; ++idx) {
                    if (negative_ids[idx] == candidate) {
                        seen = true;
                        break;
                    }
                }
                if (!seen && negative_count < 4) {
                    negative_ids[negative_count++] = candidate;
                }
            }
            if (negative_count == 0) {
                continue;
            }

            const float* row_probs = prob_ptr + ((batch * seq_len + row) * vocab);
            float* row_grad = grad_ptr + ((batch * seq_len + row) * vocab);

            for (int neg_index = 0; neg_index < negative_count; ++neg_index) {
                const int neg_token = negative_ids[neg_index];
                if (neg_token < 0 || neg_token >= vocab) {
                    continue;
                }
                const float p_neg = row_probs[neg_token];
                if (p_neg <= 1e-6f || p_neg >= 1.0f - 1e-6f) {
                    continue;
                }
                const float denom = std::max(1.0f - p_neg, 1e-6f);
                const float factor = scale * p_neg / denom;
                for (int col = 0; col < vocab; ++col) {
                    row_grad[col] -= factor * row_probs[col];
                }
                row_grad[neg_token] += factor;
            }
        }
    }

    restore_staged_tensor(answer_grad, grad_host);
}

float train_supervised_batch_impl(Trainer& trainer,
                                  const std::vector<std::vector<int>>& prompt_batch,
                                  const std::vector<std::vector<int>>& answer_batch) {
    if (!trainer.model) throw std::runtime_error("Trainer requires model");
    if (prompt_batch.empty() || prompt_batch.size() != answer_batch.size()) {
        throw std::runtime_error("train_supervised_batch requires aligned prompt/answer batches");
    }

    auto params = trainer.model->parameters();
    apply_progressive_qat_phase(trainer);
    zero_model_gradients(params);

    float total_loss = 0.0f;
    int sample_count = 0;
    AuxiliaryStackStats auxiliary_total;
    std::vector<size_t> sample_order(prompt_batch.size());
    std::iota(sample_order.begin(), sample_order.end(), 0);
    std::sort(sample_order.begin(), sample_order.end(), [&](size_t lhs, size_t rhs) {
        const auto lhs_total = prompt_batch[lhs].size() + answer_batch[lhs].size();
        const auto rhs_total = prompt_batch[rhs].size() + answer_batch[rhs].size();
        if (lhs_total != rhs_total) {
            return lhs_total > rhs_total;
        }
        if (prompt_batch[lhs].size() != prompt_batch[rhs].size()) {
            return prompt_batch[lhs].size() > prompt_batch[rhs].size();
        }
        return answer_batch[lhs].size() > answer_batch[rhs].size();
    });
    std::unordered_map<uint64_t, Tensor> full_grad_buffers;

    for (size_t order_index = 0; order_index < sample_order.size();) {
        const size_t first_index = sample_order[order_index];
        const size_t first_prompt_len = prompt_batch[first_index].size();
        const size_t first_answer_len = answer_batch[first_index].size();
        if (first_prompt_len == 0 || first_answer_len == 0) {
            throw std::runtime_error("train_supervised_batch received empty prompt/answer");
        }

        size_t max_input_len = first_prompt_len + first_answer_len - 1;
        size_t min_input_len = max_input_len;
        size_t max_answer_len = first_answer_len;
        size_t min_answer_len = first_answer_len;
        std::vector<std::vector<int>> grouped_inputs;
        std::vector<std::vector<int>> grouped_prompts;
        std::vector<std::vector<int>> grouped_answers;
        std::vector<size_t> grouped_prompt_lengths;
        while (order_index < sample_order.size()) {
            const size_t batch_index = sample_order[order_index];
            const size_t prompt_len = prompt_batch[batch_index].size();
            const size_t answer_len = answer_batch[batch_index].size();
            if (prompt_len == 0 || answer_len == 0) {
                throw std::runtime_error("train_supervised_batch received empty prompt/answer");
            }

            const size_t input_len = prompt_len + answer_len - 1;
            const size_t candidate_max_input = std::max(max_input_len, input_len);
            const size_t candidate_min_input = std::min(min_input_len, input_len);
            const size_t candidate_max_answer = std::max(max_answer_len, answer_len);
            const size_t candidate_min_answer = std::min(min_answer_len, answer_len);
            const bool compatible_bucket =
                grouped_inputs.empty() ||
                ((candidate_max_input - candidate_min_input) <=
                     std::max<size_t>(12, candidate_max_input / 3) &&
                 (candidate_max_answer - candidate_min_answer) <=
                     std::max<size_t>(8, candidate_max_answer / 2));
            if (!compatible_bucket) {
                break;
            }
            max_input_len = candidate_max_input;
            min_input_len = candidate_min_input;
            max_answer_len = candidate_max_answer;
            min_answer_len = candidate_min_answer;

            std::vector<int> inputs = prompt_batch[batch_index];
            if (answer_len > 1) {
                inputs.insert(inputs.end(),
                              answer_batch[batch_index].begin(),
                              answer_batch[batch_index].end() - 1);
            }
            grouped_inputs.push_back(std::move(inputs));
            grouped_prompts.push_back(prompt_batch[batch_index]);
            grouped_answers.push_back(answer_batch[batch_index]);
            grouped_prompt_lengths.push_back(prompt_len);
            ++order_index;
        }

        trainer.model->set_training_mode(true);
        trainer.model->reset_session();
        AuxiliaryStackStats bucket_aux =
            apply_auxiliary_stack_before_forward(trainer, grouped_prompts, grouped_answers);
        accumulate_auxiliary_stats(auxiliary_total, bucket_aux);
        Context ctx;
        Tensor logits = trainer.model->forward_ids_batch(grouped_inputs, &ctx);
        if (logits.shape.size() != 3) {
            throw std::runtime_error("train_supervised_batch expects rank-3 logits from batched forward");
        }

        const int batch_size = logits.shape[0];
        const int rows = logits.shape[1];
        const int vocab = logits.shape[2];

        const uint64_t grad_buffer_key =
            (static_cast<uint64_t>(batch_size) << 42) ^
            (static_cast<uint64_t>(rows) << 21) ^
            static_cast<uint64_t>(vocab);
        Tensor& full_grad_buffer = full_grad_buffers[grad_buffer_key];
        if (full_grad_buffer.shape != logits.shape ||
            full_grad_buffer.get_device() != logits.get_device()) {
            full_grad_buffer = Tensor::zeros(logits.shape.dims, logits.get_device());
        } else {
            zero_tensor_inplace(full_grad_buffer);
        }
        Tensor full_grad = full_grad_buffer;
        float* full_ptr = full_grad.data();
        const Device grad_device = full_grad.get_device();
        for (int batch = 0; batch < batch_size; ++batch) {
            const int answer_start = static_cast<int>(grouped_prompt_lengths[static_cast<size_t>(batch)]) - 1;
            const int answer_rows = static_cast<int>(grouped_answers[static_cast<size_t>(batch)].size());
            const int answer_end = answer_start + answer_rows;
            if (answer_start < 0 || answer_end > rows) {
                throw std::runtime_error("train_supervised answer window out of range");
            }

            Tensor sample_logits = logits.slice(0, batch, batch + 1).reshape({rows, vocab});
            Tensor answer_logits = sample_logits.slice(0, answer_start, answer_end);
            auto [loss, answer_grad] =
                answer_logits.cross_entropy(grouped_answers[static_cast<size_t>(batch)]);
            apply_supervised_gradient_weights(
                trainer,
                grouped_answers[static_cast<size_t>(batch)],
                vocab,
                answer_grad);
            apply_repetition_unlikelihood(
                trainer,
                grouped_answers[static_cast<size_t>(batch)],
                answer_logits,
                answer_grad);

            const float* answer_ptr = answer_grad.data();
            const Device answer_device = answer_grad.get_device();
            for (int row = 0; row < answer_rows; ++row) {
                const size_t dst_offset =
                    ((static_cast<size_t>(batch) * rows) + (answer_start + row)) *
                    static_cast<size_t>(vocab);
                const size_t src_offset =
                    static_cast<size_t>(row) * static_cast<size_t>(vocab);
                copy_tensor_bytes(full_ptr + dst_offset,
                                  grad_device,
                                  answer_ptr + src_offset,
                                  answer_device,
                                  static_cast<size_t>(vocab) * sizeof(float));
            }
            total_loss += loss;
            ++sample_count;
        }

        trainer.model->backward_external(full_grad, ctx);
    }

    apply_qat_regularization(trainer);
    apply_moe_aux_regularization(trainer);
    float grad_norm = 0.0f;
    apply_optimizer_step(trainer, params, std::max(sample_count, 1), &grad_norm);
    finalize_auxiliary_stats(auxiliary_total);
    trainer.last_auxiliary_stats = auxiliary_total;
    const float mean_loss = total_loss / static_cast<float>(std::max(sample_count, 1));
    record_training_audit_step(trainer, mean_loss, grad_norm, params.size());
    return mean_loss;
}

} // namespace

Trainer::Trainer(JambaModel* m, float lr) : model(m), learning_rate(lr) {
    if (model) {
        model->set_training_mode(true);
    }
    apply_progressive_qat_phase(*this);
}

Trainer::~Trainer() = default;

void Trainer::configure_progressive_qat(const TrainPhaseScheduler& scheduler) {
    phase_scheduler = scheduler;
    apply_progressive_qat_phase(*this);
}

bool Trainer::progressive_qat_active() const {
    const auto effective = resolve_effective_qat_schedule(*this);
    return phase_scheduler.progressive_qat_enabled &&
           global_step_count >= effective.qat_start_step;
}

float Trainer::train_step(const std::vector<int>& tokens,
                          const std::vector<int>& targets) {
    if (!model) throw std::runtime_error("Trainer requires model");
    model->set_training_mode(true);

    const std::vector<int> inputs = make_inputs(tokens, targets);
    const std::vector<int> resolved_targets = make_targets(tokens, targets);

    auto params = model->parameters();
    apply_progressive_qat_phase(*this);
    zero_model_gradients(params);

    model->reset_session();
    Context ctx;
    Tensor logits = model->forward_ids(inputs, &ctx);
    auto [loss, grad] = logits.cross_entropy(resolved_targets);
    model->backward_external(grad, ctx);
    apply_qat_regularization(*this);
    apply_moe_aux_regularization(*this);
    float grad_norm = 0.0f;
    apply_optimizer_step(*this, params, 1, &grad_norm);
    record_training_audit_step(*this, loss, grad_norm, params.size());

    return loss;
}

float Trainer::train_supervised(const std::vector<int>& prompt_tokens,
                                const std::vector<int>& answer_tokens) {
    return train_supervised_batch({prompt_tokens}, {answer_tokens});
}

float Trainer::train_supervised_batch(
    const std::vector<std::vector<int>>& prompt_batch,
    const std::vector<std::vector<int>>& answer_batch) {
    return train_supervised_batch_impl(*this, prompt_batch, answer_batch);
}

void Trainer::train_loop(const std::vector<int>& tokens, int epochs, int batch_size,
                         int seq_len, std::function<void(int, float)> callback,
                         int max_steps) {
    if (tokens.size() <= static_cast<size_t>(seq_len)) {
        throw std::runtime_error("Dataset small");
    }

    const int effective_batch = std::max(batch_size, 1);
    const int est_total_steps = static_cast<int>(tokens.size() / (seq_len * effective_batch)) * epochs;
    const int desired_total_steps = (max_steps > 0) ? max_steps : std::max(est_total_steps, 1);
    if (total_training_steps <= 0) {
        total_training_steps = desired_total_steps;
    } else {
        total_training_steps = std::max(total_training_steps, desired_total_steps);
    }

    int internal_global_step = 0;
    auto params = model->parameters();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t start = 0; start + seq_len < tokens.size();
             start += static_cast<size_t>(seq_len * effective_batch)) {
            apply_progressive_qat_phase(*this);
            zero_model_gradients(params);
            model->set_training_mode(true);

            std::vector<std::vector<int>> batch_inputs;
            std::vector<int> flat_targets;
            batch_inputs.reserve(static_cast<size_t>(effective_batch));
            flat_targets.reserve(static_cast<size_t>(effective_batch) *
                                 static_cast<size_t>(seq_len));

            for (int batch_index = 0; batch_index < effective_batch; ++batch_index) {
                const size_t offset = start + static_cast<size_t>(batch_index * seq_len);
                if (offset + seq_len >= tokens.size()) break;

                std::vector<int> window(tokens.begin() + offset,
                                        tokens.begin() + offset + seq_len + 1);
                const std::vector<int> inputs = make_inputs(window, {});
                const std::vector<int> resolved_targets = make_targets(window, {});
                batch_inputs.push_back(inputs);
                flat_targets.insert(flat_targets.end(),
                                    resolved_targets.begin(),
                                    resolved_targets.end());
            }

            const int samples = static_cast<int>(batch_inputs.size());
            if (samples == 0) continue;

            model->reset_session();
            Context ctx;
            Tensor logits = model->forward_ids_batch(batch_inputs, &ctx);
            auto [loss, grad] = logits.cross_entropy(flat_targets);
            model->backward_external(grad, ctx);

            apply_qat_regularization(*this);
            apply_moe_aux_regularization(*this);
            float grad_norm = 0.0f;
            apply_optimizer_step(*this, params, samples, &grad_norm);
            record_training_audit_step(*this, loss, grad_norm, params.size());

            ++internal_global_step;
            if (callback) {
                callback(internal_global_step, loss);
            }

            if (max_steps > 0 && internal_global_step >= max_steps) return;
        }
    }
}

} // namespace nsos
