#include "../include/trainer.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include "../include/layer_audit.h"
#include <algorithm>
#include <chrono>
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
        launch_scale_inplace_kernel(tensor.raw_data(), scale, tensor.size);
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
        cudaMemset(tensor.raw_data(), 0,
                   static_cast<size_t>(tensor.size) * sizeof(float));
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

#ifdef USE_CUDA
// One reusable device scalar (4 bytes, process lifetime) for the fused global
// gradient-norm reduction.  Allocated once so clip_gradients does not cudaMalloc
// per step.
float* clip_norm_accumulator() {
    static float* d_accum = [] {
        float* p = nullptr;
        if (cudaMalloc(&p, sizeof(float)) != cudaSuccess) {
            p = nullptr;
        }
        (void)cudaGetLastError();
        return p;
    }();
    return d_accum;
}
#endif

float clip_gradients(const std::vector<Parameter*>& params, float max_norm) {
    float total_norm = 0.0f;
    bool fused_done = false;
#ifdef USE_CUDA
    float* d_accum = gpu_custom_kernels_supported() ? clip_norm_accumulator() : nullptr;
    if (d_accum) {
        // FUSED global grad-norm: accumulate every GPU gradient's sum-of-squares
        // into ONE device scalar (norm_kernel uses atomicAdd) and read it back
        // with a SINGLE D2H per step.  Previously this called Tensor::norm()
        // once per parameter, each issuing its own blocking D2H -> N pipeline
        // drains per step (a dominant cause of low GPU utilization in the
        // optimizer phase).  norm_kernel already used atomicAdd, so the global
        // sum's low-bit ordering is no less deterministic than before.
        cudaMemsetAsync(d_accum, 0, sizeof(float), 0);
        double host_sq = 0.0;
        for (auto* p : params) {
            if (!p || p->grad.size == 0) continue;
            if (p->grad.get_device() == Device::GPU) {
                launch_norm_kernel(d_accum, p->grad.raw_data(), p->grad.size);
            } else {
                const float n = p->grad.norm();  // CPU path: no device sync
                host_sq += static_cast<double>(n) * static_cast<double>(n);
            }
        }
        trainer_check_cuda("launch_norm_kernel(clip)");
        float gpu_sq = 0.0f;
        cudaMemcpy(&gpu_sq, d_accum, sizeof(float), cudaMemcpyDeviceToHost);  // single sync
        total_norm = std::sqrt(static_cast<float>(host_sq) + gpu_sq);
        fused_done = true;
    }
#endif
    if (!fused_done) {
        float total_norm_sq = 0.0f;
        for (auto* p : params) {
            if (!p || p->grad.size == 0) continue;
            const float norm = p->grad.norm();
            total_norm_sq += norm * norm;
        }
        total_norm = std::sqrt(total_norm_sq);
    }

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

    for (BitLinear* layer : bitlinear_layers) {
        if (!layer) {
            continue;
        }
        // BitNet b1.58 = ternary weights + int8 activations.  The packed CPU
        // kernel derives the activation range from precision_bits; packed
        // inference uses 8-bit, so quantized training must too (train ==
        // inference numerics).  Weights are ternary via the packed kernel
        // regardless of this value.
        layer->set_precision_mode(8);

        // True quantized training: once the quantized phase is active, route
        // the forward through the REAL packed ternary kernel and back-propagate
        // with a straight-through estimator onto the FP32 latent weights.  GPU
        // layers keep the float reference path (there is no packed CPU kernel
        // for device tensors; the dp4a path is inference-only), so QAT on GPU
        // still trains in float.
        const bool on_gpu = layer->has_full_precision_weight() &&
                            layer->weight.data.get_device() == Device::GPU;
        // Sensitive projections (e.g. Mamba dt/B/C) stay on the float path to
        // preserve the mixed-precision design.
        const bool train_quantized =
            quantized_active && !on_gpu && !layer->quantization_sensitive();
        layer->set_reference_path(!train_quantized);
        if (train_quantized) {
            // Re-quantize the current latent weights so the forward multiplies
            // up-to-date ternary codes (the optimizer just updated them).
            layer->repack_weights();
        }
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

// 4-bit Adam step (Li et al. 2023, "Memory Efficient Optimizers with 4-bit
// States").  Dequantises the packed m/v state into FP32 scratch, runs the
// identical Adam update used by the FP32 path below, then re-quantises.  Trades
// a little compute per step for ~8x less optimizer-state memory.  Engaged only
// for CPU parameters (the GPU 4-bit kernel is a Phase-2 item); GPU parameters
// keep the FP32 launch_adamw_update_kernel path.
static void apply_adam_step_4bit(Trainer& trainer,
                                 Parameter* p,
                                 float cur_lr,
                                 float bc1,
                                 float bc2) {
    const int n = p->data.size;
    Quant4OptState& st = trainer.quant_state[p];

    static thread_local std::vector<float> m_buf;
    static thread_local std::vector<float> v_buf;
    m_buf.resize(static_cast<size_t>(n));
    v_buf.resize(static_cast<size_t>(n));

    if (st.n != n) {
        // First visit (or a reshape under us): start from zeroed moments.
        for (int i = 0; i < n; ++i) {
            m_buf[i] = 0.0f;
            v_buf[i] = 0.0f;
        }
    } else {
        quant4_load_m(st, m_buf.data(), n);
        quant4_load_v(st, v_buf.data(), n);
    }

    float* w = p->data.data();
    const float* g = p->grad.data();
    const bool apply_wd =
        trainer.weight_decay > 0.0f && should_apply_weight_decay(*p);

    for (int i = 0; i < n; ++i) {
        m_buf[i] = trainer.beta1 * m_buf[i] + (1.0f - trainer.beta1) * g[i];
        v_buf[i] = trainer.beta2 * v_buf[i] + (1.0f - trainer.beta2) * g[i] * g[i];

        const float m_hat = m_buf[i] / bc1;
        const float v_hat = v_buf[i] / bc2;

        if (apply_wd) {
            w[i] -= cur_lr * trainer.weight_decay * w[i];
        }
        w[i] -= cur_lr * m_hat / (std::sqrt(v_hat) + trainer.eps);
    }

    // Treat 2-D states as matrices so v gets the paper's rank-1 normalisation;
    // fold higher-rank tensors to 2-D by their last dimension (rows<=0 disables
    // rank-1 and the v path falls back to block-wise abs-max).
    int rows = 0;
    int cols = 0;
    const std::vector<int>& dims = p->data.shape.dims;
    if (dims.size() >= 2) {
        cols = dims.back();
        if (cols > 0 && n % cols == 0) {
            rows = n / cols;
        } else {
            rows = 0;
            cols = 0;
        }
    }

    quant4_store_m(m_buf.data(), n, st);
    quant4_store_v(v_buf.data(), n, rows, cols, st);
    p->mark_updated();
}

// OXTA-CRIT Lei 1 (docs/OXTA_CRIT_THEORY.md) — controlador de criticalidade.
// E4+controle mediu que o QAT progressivo — e só ele — tira as camadas
// lineares da banda crítica (frac g em [0.5,2]: 0.976 -> 0.214 em 240 steps),
// onde g = gamma^2*(1-p0)*fan_in é o ganho de ramo ternário (regra absmean).
// Este controlador aplica, a cada K steps, uma correção multiplicativa pequena
// puxando cada peso rank-2 de volta ao seu ganho INICIAL g0 (capturado na 1a
// visita): w *= exp(-(eta/2)*log(g/g0)), clampado a ±5% por aplicação.
// Rescale puro: o PADRÃO ternário (sinais de round(w/gamma)) é invariante de
// escala — só o balanço escala/esparsidade é restaurado (lei de covariação).
// mark_updated() invalida os caches packed (repack-once permanece correto).
// Opt-in experimental: NSOS_CRIT_REG=1 [NSOS_CRIT_REG_ETA=0.2]
// [NSOS_CRIT_REG_EVERY=10]. Estado g0 é process-wide por Parameter*.
void apply_criticality_regularization(Trainer& trainer,
                                      const std::vector<Parameter*>& params) {
    static const bool enabled = [] {
        const char* e = std::getenv("NSOS_CRIT_REG");
        return e != nullptr && e[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static const float eta = [] {
        const char* e = std::getenv("NSOS_CRIT_REG_ETA");
        const float v = e ? std::strtof(e, nullptr) : 0.2f;
        return (v > 0.0f && v <= 1.0f) ? v : 0.2f;
    }();
    static const int every = [] {
        const char* e = std::getenv("NSOS_CRIT_REG_EVERY");
        const int v = e ? std::atoi(e) : 10;
        return v > 0 ? v : 10;
    }();
    if (trainer.global_step_count % every != 0) {
        return;
    }
    static std::unordered_map<Parameter*, float> g0_map;

    for (auto* p : params) {
        if (!p || p->data.size == 0 || p->data.shape.size() != 2) continue;
        const int rows = p->data.shape[0];
        const int cols = p->data.shape[1];
        if (rows <= 1 || cols <= 1) continue;

        Tensor host = p->data.get_device() == Device::GPU ? p->data.cpu() : p->data;
        const float* w = host.data();
        const int n = host.size;
        double abs_sum = 0.0;
        for (int i = 0; i < n; ++i) abs_sum += std::fabs(w[i]);
        const float gamma = static_cast<float>(abs_sum / std::max(n, 1));
        if (gamma <= 0.0f) continue;
        int zeros = 0;
        const float half_gamma = 0.5f * gamma;
        for (int i = 0; i < n; ++i) zeros += (std::fabs(w[i]) < half_gamma) ? 1 : 0;
        const float p0 = static_cast<float>(zeros) / static_cast<float>(n);
        const float g = gamma * gamma * (1.0f - p0) * static_cast<float>(cols);
        if (g <= 0.0f) continue;

        auto it = g0_map.find(p);
        if (it == g0_map.end()) {
            g0_map.emplace(p, g);  // baseline = ganho na 1a visita (init saudável)
            continue;
        }
        const float log_ratio = std::log(g / it->second);
        float log_c = -0.5f * eta * log_ratio;
        log_c = std::clamp(log_c, -0.05f, 0.05f);
        if (std::fabs(log_c) < 1e-5f) continue;
        scale_tensor_inplace(p->data, std::exp(log_c));
        p->mark_updated();
    }
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

    // AUDIT #10 (2026-05-16): hoist Adam state allocation out of the
    // hot per-step loop.  Previously each step did a find() check per
    // parameter and lazy-initialized.  Even with std::unordered_map's
    // O(1) find, this is wasted work after the first step: every
    // subsequent step hits the same "already present" branch on every
    // parameter.  We now pre-allocate state on first visit per param
    // and rely on operator[] returning an existing reference cheaply.
    //
    // The shape-mismatch guard is an extra safety net: if a parameter
    // is added or resized between calls (e.g., during architecture
    // surgery in tests), we re-init that state to avoid silent shape
    // bugs.  In production training the shapes are stable so this
    // branch never fires after warmup.
    for (auto* p : params) {
        if (!p || p->grad.size == 0) continue;

        // 4-bit optimizer states (CPU path).  GPU params fall through to the
        // FP32 launch_adamw_update_kernel path below until the 4-bit CUDA
        // kernel lands.
        if (trainer.optimizer_state_bits == 4 &&
            p->data.get_device() == Device::CPU) {
            apply_adam_step_4bit(trainer, p, cur_lr, bc1, bc2);
            continue;
        }

        auto m_it = trainer.m_state.find(p);
        if (m_it == trainer.m_state.end()) {
            trainer.m_state.emplace(p, Tensor::zeros(p->data.shape.dims,
                                                      p->data.get_device()));
            trainer.v_state.emplace(p, Tensor::zeros(p->data.shape.dims,
                                                      p->data.get_device()));
        } else if (m_it->second.shape != p->data.shape ||
                   m_it->second.get_device() != p->data.get_device()) {
            // Architecture changed under us — re-init this entry.
            m_it->second = Tensor::zeros(p->data.shape.dims, p->data.get_device());
            trainer.v_state[p] =
                Tensor::zeros(p->data.shape.dims, p->data.get_device());
        }

        Tensor& m_tensor = trainer.m_state[p];
        Tensor& v_tensor = trainer.v_state[p];

#ifdef USE_CUDA
        if (can_use_gpu_optimizer(*p, m_tensor, v_tensor)) {
            launch_adamw_update_kernel(
                p->data.raw_data(),
                p->grad.raw_data(),
                m_tensor.raw_data(),
                v_tensor.raw_data(),
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

    apply_criticality_regularization(trainer, params);
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

#ifdef USE_CUDA
// NSOS_RUL_HOST=1 forces the host repetition-unlikelihood path (for A/B parity
// against the GPU kernel).  Default: GPU when tensors are device-resident.
bool rul_force_host() {
    static const bool force = [] {
        const char* e = std::getenv("NSOS_RUL_HOST");
        return e != nullptr && e[0] == '1';
    }();
    return force;
}

// Persistent device buffer for per-sample answer-token uploads (grows on demand)
// so the GPU loss-adjustment path never cudaMalloc's per call.
int* loss_token_device_buffer(int count) {
    static int* buf = nullptr;
    static int cap = 0;
    if (count <= 0) return nullptr;
    if (count > cap) {
        if (buf) cudaFree(buf);
        buf = nullptr;
        if (cudaMalloc(&buf, static_cast<size_t>(count) * sizeof(int)) != cudaSuccess) {
            (void)cudaGetLastError();
            cap = 0;
            return nullptr;
        }
        cap = count;
    }
    return buf;
}
#endif

void apply_supervised_gradient_weights(const Trainer& trainer,
                                       const std::vector<int>& answer_tokens,
                                       int vocab,
                                       Tensor& answer_grad) {
    if (answer_tokens.empty() || vocab <= 0 || answer_grad.size == 0) {
        return;
    }

    // PERF: short-circuit BEFORE the sync_host_access barrier when no
    // row will actually be rescaled.  The old code called
    // sync_host_access (= cudaDeviceSynchronize on Pascal+Windows /
    // first-touch on any platform) every training step even when both
    // scales were effectively 1.0 and the inner loop was a no-op.  By
    // computing the gate up front we skip the GPU drain for free
    // whenever first_token_loss_scale and eos_loss_scale are 1.0.
    const float ft_scale  = std::max(trainer.first_token_loss_scale, 0.0f);
    const float eos_scale = std::max(trainer.eos_loss_scale, 0.0f);
    const int last_row = static_cast<int>(answer_tokens.size()) - 1;
    const bool needs_first_token = std::abs(ft_scale - 1.0f) > 1e-6f && last_row >= 0;
    const bool needs_eos = std::abs(eos_scale - 1.0f) > 1e-6f && last_row >= 0 &&
                           answer_tokens[static_cast<size_t>(last_row)] == trainer.eos_token_id;
    if (!needs_first_token && !needs_eos) {
        return;
    }

#ifdef USE_CUDA
    // GPU path: scale the affected row(s) in place on the device — no host
    // round-trip (the previous sync_host_access + host write drained the
    // pipeline and forced a per-sample GPU->CPU sync every step).
    if (answer_grad.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
        if (needs_first_token) {
            launch_scale_inplace_kernel(answer_grad.raw_data(), ft_scale, vocab);
        }
        if (needs_eos) {
            launch_scale_inplace_kernel(
                answer_grad.raw_data() + static_cast<size_t>(last_row) * vocab,
                eos_scale, vocab);
        }
        trainer_check_cuda("launch_scale_inplace_kernel(grad_weights)");
        return;
    }
#endif

    // Drain pending GPU work before host write — Pascal+Windows UM has
    // no demand paging.  No-op when answer_grad is host-resident.
    answer_grad.sync_host_access();

    float* grad_ptr = answer_grad.data();
    for (int row = 0; row <= last_row; ++row) {
        float scale = 1.0f;
        if (row == 0) {
            scale *= ft_scale;
        }
        if (row == last_row && answer_tokens[static_cast<size_t>(row)] == trainer.eos_token_id) {
            scale *= eos_scale;
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
    const int vocab = answer_logits.shape.back();

#ifdef USE_CUDA
    // GPU path: do the whole repetition-unlikelihood adjustment on the device
    // using the on-GPU softmax — eliminates the per-sample probs.cpu()/grad.cpu()
    // D2H + host loop that dominated the training step (it scales with
    // answer_len * batch).  NSOS_RUL_HOST=1 forces the host path for A/B parity.
    if (!rul_force_host() && answer_grad.get_device() == Device::GPU &&
        probs.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
        const int rows = static_cast<int>(answer_tokens.size());
        int* d_tokens = loss_token_device_buffer(rows);
        if (d_tokens) {
            cudaMemcpy(d_tokens, answer_tokens.data(),
                       static_cast<size_t>(rows) * sizeof(int), cudaMemcpyHostToDevice);
            launch_repetition_unlikelihood_kernel(
                answer_grad.raw_data(), probs.raw_data(), d_tokens, rows, vocab,
                scale, trainer.eos_token_id);
            trainer_check_cuda("launch_repetition_unlikelihood_kernel");
            return;
        }
    }
#endif

    Tensor probs_host = probs.get_device() == Device::GPU ? probs.cpu() : probs;
    Tensor grad_host = answer_grad.get_device() == Device::GPU ? answer_grad.cpu() : answer_grad.clone();
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

    // Same Pascal+Windows UM hazard as the non-batched variant above.
    answer_grad.sync_host_access();

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
    // Âncora do wall C++ (NSOS_TRAIN_TIMING): a diferença entre o wall do
    // chamador Python e este wall expõe o custo de binding/conversão de listas.
    const auto tm_call0 = std::chrono::steady_clock::now();

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

    // Diagnostic step-timing (NSOS_TRAIN_TIMING=1): localizes where the step
    // time goes (forward / per-sample loss loop / backward / optimizer), with a
    // device sync around each section so the wall-times are accurate.  Off by
    // default (zero overhead); prints one line per step to stderr.
    const bool nsos_step_timing = [] {
        const char* e = std::getenv("NSOS_TRAIN_TIMING");
        return e != nullptr && e[0] == '1';
    }();
    double tm_fwd = 0.0, tm_loss = 0.0, tm_bwd = 0.0, tm_opt = 0.0, tm_gap = 0.0;
    int tm_buckets = 0;
    auto tm_now = [&]() {
#ifdef USE_CUDA
        if (nsos_step_timing) cudaDeviceSynchronize();
#endif
        return std::chrono::steady_clock::now();
    };
    using tm_ms = std::chrono::duration<double, std::milli>;
    // prep = parameters() + QAT phase + zero-grads + sort (sincronizado quando
    // timing on, p/ drenar os memsets enfileirados).  gap = custo entre buckets
    // (agrupamento, aux stack, zeragem do full_grad) — v1 do instrumento não o
    // media e deixava ~60-80% do wall sem dono.
    const auto tm_prep_end = tm_now();
    auto tm_last = tm_prep_end;

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
        const auto _tm_fwd0 = tm_now();
        tm_gap += tm_ms(_tm_fwd0 - tm_last).count();
        Tensor logits = trainer.model->forward_ids_batch(grouped_inputs, &ctx);
        const auto _tm_fwd1 = tm_now();
        tm_fwd += tm_ms(_tm_fwd1 - _tm_fwd0).count();
        ++tm_buckets;
        if (logits.shape.size() != 3) {
            throw std::runtime_error("train_supervised_batch expects rank-3 logits from batched forward");
        }

        // SSA learned block-selector: distill this forward's dense per-block
        // attention mass into ssa_wsel_ (self-contained SGD; no-op unless sparse
        // attention is enabled and per-head Q/K were saved by the exact path).
        trainer.model->accumulate_sparse_selector_grads();

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
        float* full_ptr = full_grad.raw_data();  // GPU→GPU copies below; no host sync needed
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

            // Single contiguous copy of this sample's answer-grad block into
            // full_grad.  In [batch, rows, vocab] layout the answer rows
            // [answer_start, answer_end) are contiguous, and answer_grad is
            // [answer_rows, vocab] contiguous — so the whole block moves in ONE
            // copy.  The old per-row loop issued answer_rows copy_tensor_bytes
            // calls, each with its own cudaStreamSynchronize, i.e.
            // ~answer_rows*batch stream syncs/step (the real cost that scaled
            // with answer_len and dominated the T4 step time).
            const size_t dst_offset =
                (static_cast<size_t>(batch) * rows +
                 static_cast<size_t>(answer_start)) *
                static_cast<size_t>(vocab);
            copy_tensor_bytes(full_ptr + dst_offset, grad_device,
                              answer_grad.raw_data(), answer_grad.get_device(),
                              static_cast<size_t>(answer_rows) *
                                  static_cast<size_t>(vocab) * sizeof(float));
            total_loss += loss;
            ++sample_count;
        }

        const auto _tm_loss1 = tm_now();
        tm_loss += tm_ms(_tm_loss1 - _tm_fwd1).count();
        trainer.model->backward_external(full_grad, ctx);
        const auto _tm_bwd1 = tm_now();
        tm_bwd += tm_ms(_tm_bwd1 - _tm_loss1).count();
        tm_last = _tm_bwd1;
    }

    apply_qat_regularization(trainer);
    apply_moe_aux_regularization(trainer);
    float grad_norm = 0.0f;
    const auto _tm_opt0 = tm_now();
    apply_optimizer_step(trainer, params, std::max(sample_count, 1), &grad_norm);
    if (nsos_step_timing) {
        const auto _tm_opt1 = tm_now();
        tm_opt = tm_ms(_tm_opt1 - _tm_opt0).count();
        const double wall = tm_ms(_tm_opt1 - tm_call0).count();
        const double prep = tm_ms(tm_prep_end - tm_call0).count();
        const double gap_tail = tm_ms(_tm_opt0 - tm_last).count();
        const double unacc =
            wall - (prep + tm_gap + tm_fwd + tm_loss + tm_bwd + gap_tail + tm_opt);
        std::cerr << "[timing] buckets=" << tm_buckets << " wall=" << wall
                  << "ms prep=" << prep << " gap=" << tm_gap << " fwd=" << tm_fwd
                  << " loss=" << tm_loss << " bwd=" << tm_bwd
                  << " tail=" << gap_tail << " opt=" << tm_opt
                  << " unacc=" << unacc << std::endl;
    }
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
    // SSA learned block-selector distillation (no-op unless sparse attention is on).
    model->accumulate_sparse_selector_grads();
    auto [loss, grad] = logits.cross_entropy(resolved_targets);

    // ── Pantheon VIB-style L2 regularizer on logits ──────────────────────
    // When pantheon_vib_beta > 0, add beta * 0.5 * mean(logits^2) to the
    // loss and the corresponding gradient term (beta * logits / N) to the
    // grad tensor before backward.  This is a degenerate VIB compression
    // (variational layer not needed); pulls logits toward zero while CE
    // still pulls them toward correct targets.  See PANTHEON_VALIDATION_REPORT.
    if (this->pantheon_vib_beta > 0.0f && logits.size > 0) {
        const float beta = this->pantheon_vib_beta;
        const int N = logits.size;
        const Device dev = logits.get_device();
        // CPU copy for scalar reduction (safe regardless of device).
        Tensor logits_cpu = (dev == Device::CPU) ? logits : logits.to(Device::CPU);
        const float* lh = logits_cpu.data();
        double sumsq = 0.0;
        for (int i = 0; i < N; ++i) sumsq += (double)lh[i] * lh[i];
        const float l2_term = beta * 0.5f *
            static_cast<float>(sumsq / static_cast<double>(std::max(N, 1)));
        loss += l2_term;
        // Build l2_grad on CPU, ship to grad's device, then add element-wise.
        Tensor l2_grad_cpu = Tensor::zeros(logits.shape.dims, Device::CPU);
        float* lg = l2_grad_cpu.data();
        const float scale = beta / static_cast<float>(std::max(N, 1));
        for (int i = 0; i < N; ++i) lg[i] = scale * lh[i];
        Tensor l2_grad = (grad.get_device() == Device::CPU)
                            ? l2_grad_cpu
                            : l2_grad_cpu.to(grad.get_device());
        grad = grad.add(l2_grad);
    }
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

            // Chunked forward + cross_entropy + backward.
            //
            // The original implementation issued ONE forward over the
            // full batch and ONE cross_entropy launch of size
            // (batch_size × seq_len × vocab) — for a typical phase-3
            // shape this is 10M+ entries in a single sustained kernel.
            // On underprovisioned hosts (e.g. GTX 1050 Ti on a marginal
            // PSU) this monolithic launch keeps the GPU at peak draw
            // long enough to trip the supply, causing host shutdowns
            // exactly at the start of phase3 (Wikipedia).
            //
            // Splitting into micro-chunks of `kCrossEntropyChunkSize`
            // samples gives the host brief recovery windows between
            // launches without changing the math: gradients accumulate
            // through Parameter::add_grad (which does grad += new_grad),
            // and we fire the optimizer step exactly once at the end
            // with `samples` as the accumulation_steps divisor — same
            // as before.  Loss is reported as the per-sample mean.
            //
            // ── Chunk size policy ──────────────────────────────────────
            // PSU-marginal hosts (GTX 1050 Ti + 500W PSU): set
            //   NSOS_TRAIN_CHUNK_SIZE=2 in the environment.  This was
            //   the default until 2026-05-16, when chunking-as-default
            //   was found to leave 5/6 of T4/A100 throughput on the
            //   floor in Colab (low PSU risk).
            // Data-center hosts (T4 / A100 / H100 / well-provisioned
            //   workstations): leave NSOS_TRAIN_CHUNK_SIZE unset OR
            //   set to 0.  We then run the full batch in a single
            //   forward+backward pass and the optimizer step still
            //   divides gradients by `samples` exactly as before.
            // Custom values: any positive integer is honored verbatim.
            //   E.g. NSOS_TRAIN_CHUNK_SIZE=8 is a middle-ground that
            //   keeps GPU utilization high while still giving brief
            //   power-recovery windows for borderline-PSU hosts.
            int chunk_size_env = 0;
            if (const char* env = std::getenv("NSOS_TRAIN_CHUNK_SIZE")) {
                if (*env != '\0') {
                    chunk_size_env = std::atoi(env);
                }
            }
            const int kCrossEntropyChunkSize =
                (chunk_size_env > 0) ? chunk_size_env : samples;

            float aggregate_loss = 0.0f;
            int aggregate_samples = 0;
            for (int chunk_start = 0; chunk_start < samples;
                 chunk_start += kCrossEntropyChunkSize) {
                const int chunk_end =
                    std::min(chunk_start + kCrossEntropyChunkSize, samples);
                const int chunk_samples = chunk_end - chunk_start;

                std::vector<std::vector<int>> chunk_inputs(
                    batch_inputs.begin() + chunk_start,
                    batch_inputs.begin() + chunk_end);
                std::vector<int> chunk_targets(
                    flat_targets.begin() +
                        static_cast<size_t>(chunk_start * seq_len),
                    flat_targets.begin() +
                        static_cast<size_t>(chunk_end * seq_len));

                model->reset_session();
                Context ctx;
                Tensor logits = model->forward_ids_batch(chunk_inputs, &ctx);
                // SSA learned block-selector distillation (no-op unless sparse on).
                model->accumulate_sparse_selector_grads();
                auto [loss, grad] = logits.cross_entropy(chunk_targets);
                model->backward_external(grad, ctx);

                aggregate_loss += loss * static_cast<float>(chunk_samples);
                aggregate_samples += chunk_samples;
            }

            const float mean_loss =
                aggregate_loss / static_cast<float>(std::max(aggregate_samples, 1));

            apply_qat_regularization(*this);
            apply_moe_aux_regularization(*this);
            float grad_norm = 0.0f;
            apply_optimizer_step(*this, params, samples, &grad_norm);
            record_training_audit_step(*this, mean_loss, grad_norm, params.size());

            ++internal_global_step;
            if (callback) {
                callback(internal_global_step, mean_loss);
            }

            if (max_steps > 0 && internal_global_step >= max_steps) return;
        }
    }
}

} // namespace nsos
