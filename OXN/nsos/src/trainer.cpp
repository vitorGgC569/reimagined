#include "../include/trainer.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include "../include/layer_audit.h"
#include "../include/nsos_serializer.h"
#include "../include/nsos/determinism.h"  // K4: ordered reductions under NSOS_DETERMINISTIC
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

namespace {

constexpr uint32_t kTrainingStateMagic = 0x4E535452u;  // NSTR
constexpr uint32_t kTrainingStateVersion = 2u;
constexpr uint32_t kTrainingStateLegacyVersion = 1u;

template <typename T>
void write_training_pod(std::ostream& output, const T& value,
                        const char* label) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error(std::string("Training-state write failed: ") +
                                 label);
    }
}

template <typename T>
T read_training_pod(std::istream& input, const char* label) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error(std::string("Training-state truncated at ") +
                                 label);
    }
    return value;
}

void write_training_string(std::ostream& output, const std::string& value) {
    if (value.size() > 4096) {
        throw std::runtime_error("Training-state parameter name is too long");
    }
    write_training_pod(output, static_cast<uint32_t>(value.size()),
                       "string length");
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) throw std::runtime_error("Training-state string write failed");
}

std::string read_training_string(std::istream& input) {
    const uint32_t length = read_training_pod<uint32_t>(input, "string length");
    if (length > 4096) {
        throw std::runtime_error("Training-state parameter name exceeds limit");
    }
    std::string value(length, '\0');
    if (length > 0) {
        input.read(value.data(), static_cast<std::streamsize>(length));
        if (!input) throw std::runtime_error("Training-state truncated in string");
    }
    return value;
}

uint64_t training_state_file_hash(const std::filesystem::path& path,
                                  uint64_t& byte_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot hash model checkpoint: " + path.string());
    }
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    byte_count = 0;
    std::array<char, 1 << 16> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        byte_count += static_cast<uint64_t>(count);
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
            hash *= kPrime;
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while hashing model checkpoint");
    }
    return hash;
}

void replace_training_state_file(const std::filesystem::path& temporary,
                                 const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        std::filesystem::remove(temporary);
        throw std::runtime_error(
            "Could not atomically replace training state (Win32 error " +
            std::to_string(error) + ")");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("Could not atomically replace training state: " +
                                 error.message());
    }
#endif
}

std::vector<std::pair<Parameter*, std::string>> stable_training_parameters(
    JambaModel* model) {
    std::vector<std::pair<Parameter*, std::string>> result;
    std::unordered_map<std::string, size_t> counts;
    for (Parameter* parameter : model->parameters()) {
        if (!parameter) continue;
        const std::string identity = !parameter->base_name.empty()
                                         ? parameter->base_name
                                         : parameter->name;
        const size_t occurrence = counts[identity]++;
        result.emplace_back(parameter,
                            identity + "#" + std::to_string(occurrence));
    }
    return result;
}

struct TrainingStateMetadata {
    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float eps = 0.0f;
    float weight_decay = 0.0f;
    float max_grad_norm = 0.0f;
    float min_learning_rate_scale = 0.0f;
    float first_token_loss_scale = 0.0f;
    float eos_loss_scale = 0.0f;
    float repetition_unlikelihood_scale = 0.0f;
    float moe_aux_loss_scale = 0.0f;
    float pantheon_vib_beta = 0.0f;
    float logit_l2_beta = 0.0f;
    int32_t warmup_steps = 0;
    int32_t global_step_count = 0;
    int32_t total_training_steps = 0;
    int32_t eos_token_id = 0;
    int32_t optimizer_state_bits = 32;
    TrainPhaseScheduler phase_scheduler;
};

struct TrainingParameterRecord {
    Parameter* parameter = nullptr;
    std::vector<float> m;
    std::vector<float> v;
    bool has_moments = false;
    bool has_criticality = false;
    bool has_external_lr_scale = false;
    bool has_criticality_lr_scale = false;
    float criticality = 0.0f;
    float external_lr_scale = 1.0f;
    float criticality_lr_scale = 1.0f;
};

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

// (auditoria #8) cópia local removida — usa copy_tensor_bytes unificada (tensor.h).

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

// (auditoria #8) cópia local removida — usa copy_tensor_bytes unificada (tensor.h).
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
    // thread_local: per-thread device scalar so concurrent optimizer/clip paths
    // never share one accumulator (replica-safety; matches the tensor.cpp
    // gemm/CE scratch rationale).
    thread_local float* d_accum = [] {
        float* p = nullptr;
        if (cudaMalloc(&p, sizeof(float)) != cudaSuccess) {
            p = nullptr;
        }
        (void)cudaGetLastError();
        return p;
    }();
    return d_accum;
}

int* finite_issue_accumulator() {
    thread_local int* device_flag = [] {
        int* value = nullptr;
        if (cudaMalloc(&value, sizeof(int)) != cudaSuccess) {
            value = nullptr;
            (void)cudaGetLastError();
        }
        return value;
    }();
    return device_flag;
}
#endif

void ensure_finite_optimizer_inputs(const std::vector<Parameter*>& params) {
    bool host_issue = false;
#ifdef USE_CUDA
    int* device_issue =
        gpu_custom_kernels_supported() ? finite_issue_accumulator() : nullptr;
    if (device_issue) cudaMemset(device_issue, 0, sizeof(int));
#endif
    auto inspect = [&](const Tensor& tensor) {
        if (tensor.size == 0) return;
#ifdef USE_CUDA
        if (device_issue && tensor.get_device() == Device::GPU) {
            launch_check_stability_kernel(
                device_issue, tensor.raw_data(),
                std::numeric_limits<float>::max(), tensor.size);
            return;
        }
#endif
        Tensor host = tensor.get_device() == Device::GPU ? tensor.cpu() : tensor;
        const float* values = host.data();
        for (int index = 0; index < host.size; ++index) {
            if (!std::isfinite(values[index])) {
                host_issue = true;
                return;
            }
        }
    };
    for (Parameter* parameter : params) {
        if (!parameter || parameter->grad.size == 0) continue;
        inspect(parameter->data);
        inspect(parameter->grad);
        if (host_issue) break;
    }
#ifdef USE_CUDA
    if (device_issue) {
        trainer_check_cuda("launch_check_stability_kernel(optimizer_gate)");
        int gpu_issue = 0;
        cudaMemcpy(&gpu_issue, device_issue, sizeof(int), cudaMemcpyDeviceToHost);
        host_issue = host_issue || gpu_issue != 0;
    }
#endif
    if (host_issue) {
        throw std::runtime_error(
            "optimizer step rejected: a parameter or gradient contains NaN/Inf");
    }
}

float clip_gradients(const std::vector<Parameter*>& params, float max_norm) {
    float total_norm = 0.0f;
    bool fused_done = false;
    // K4: deterministic mode -> ordered host reduction of the global grad norm.
    // The GPU norm_kernel (and Tensor::norm()) accumulate via atomicAdd, whose
    // low-bit order is run-to-run nondeterministic; summing per parameter on the
    // host in a fixed order (double accumulator) is bit-reproducible.  Opt-in
    // (NSOS_DETERMINISTIC); the fast atomic path stays the default.
    if (determinism::deterministic_reductions_enabled()) {
        double total_sq = 0.0;
        for (auto* p : params) {
            if (!p || p->grad.size == 0) continue;
            Tensor g = (p->grad.get_device() == Device::GPU) ? p->grad.cpu() : p->grad;
            const float* gp = g.data();
            const int n = static_cast<int>(g.size);
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += static_cast<double>(gp[i]) * static_cast<double>(gp[i]);
            total_sq += s;
        }
        total_norm = static_cast<float>(std::sqrt(total_sq));
        fused_done = true;
    }
#ifdef USE_CUDA
    float* d_accum = (!fused_done && gpu_custom_kernels_supported()) ? clip_norm_accumulator() : nullptr;
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
        // K3: QAT now runs on the GPU too — BitLinear::forward has a fake-quant
        // STE path (ternary weights + int8 activations, dequantized so it stays
        // differentiable) that straight-throughs onto the FP32 latent weights.
        // So we no longer force the GPU float reference path; train numerics ==
        // ternary-inference numerics on device.  Sensitive projections (Mamba
        // dt/B/C, FP router) still stay on the float reference path to preserve
        // the mixed-precision design.
        const bool train_quantized =
            quantized_active && !layer->quantization_sensitive();
        layer->set_reference_path(!train_quantized);
        // Only the CPU packed kernel needs current packed codes; the GPU QAT
        // path recomputes the fake-quant from the latent weight each forward.
        if (train_quantized && !on_gpu) {
            // Re-quantize the current latent weights so the forward multiplies
            // up-to-date ternary codes (the optimizer just updated them).
            layer->repack_weights();
        }
    }
}

float apply_qat_regularization(Trainer& trainer, int accumulation_steps) {
    if (!trainer.model || !trainer.phase_scheduler.progressive_qat_enabled) {
        return 0.0f;
    }
    const auto effective = resolve_effective_qat_schedule(trainer);
    if (trainer.global_step_count < effective.semantic_warmup_steps) {
        return 0.0f;
    }

    const float ramp_denom = static_cast<float>(std::max(
        effective.qat_start_step - effective.semantic_warmup_steps, 1));
    const float ramp =
        std::clamp(static_cast<float>(trainer.global_step_count -
                                      effective.semantic_warmup_steps) /
                       ramp_denom,
                   0.0f, 1.0f);
    // Task gradients are accumulated as a sum and divided by
    // accumulation_steps in apply_optimizer_step.  Scale this once-per-step
    // regularizer by the same count so its effective coefficient is invariant
    // to batch size / gradient accumulation.
    const float base_regularization =
        trainer.phase_scheduler.ternary_regularization *
        std::max(ramp, 0.0f);
    const float regularization =
        base_regularization *
        static_cast<float>(std::max(accumulation_steps, 1));
    if (regularization <= 0.0f) {
        return 0.0f;
    }

    double host_penalty_grad_sq = 0.0;
#ifdef USE_CUDA
    float* device_penalty_grad_sq =
        gpu_custom_kernels_supported() ? clip_norm_accumulator() : nullptr;
    if (device_penalty_grad_sq) {
        cudaMemset(device_penalty_grad_sq, 0, sizeof(float));
    }
#endif

    for (BitLinear* layer : trainer.model->collect_bitlinear_layers()) {
        if (!layer || !layer->has_full_precision_weight()) {
            continue;
        }
        // Skip layers kept on the float path (dt/B/C, MoE router): they are never
        // ternarized in the forward, so pulling their latent weights toward
        // ternary codes is spurious pressure on the SSM/routing gain.
        if (layer->quantization_sensitive()) {
            continue;
        }
        // Pull the latent weight toward its ACTUAL fake-quantized value,
        // scale * code, NOT the bare code {-1,0,+1}.  The BitLinear helper
        // reuses the QAT forward's GPU-resident scaled ternary tensor when
        // available; the old trainer-side quantize_weights() call was a CPU
        // loop over GPU managed memory and made ternary training unusably slow.
        Tensor penalty_grad =
            layer->add_qat_regularization_grad(regularization);
        if (penalty_grad.size == 0) continue;
#ifdef USE_CUDA
        if (device_penalty_grad_sq &&
            penalty_grad.get_device() == Device::GPU) {
            launch_norm_kernel(device_penalty_grad_sq,
                               penalty_grad.raw_data(), penalty_grad.size);
            continue;
        }
#endif
        Tensor host = penalty_grad.get_device() == Device::GPU
                          ? penalty_grad.cpu()
                          : penalty_grad;
        const float* values = host.data();
        for (int index = 0; index < host.size; ++index) {
            host_penalty_grad_sq +=
                static_cast<double>(values[index]) * values[index];
        }
    }

#ifdef USE_CUDA
    if (device_penalty_grad_sq) {
        trainer_check_cuda("launch_norm_kernel(qat_regularization)");
        float gpu_penalty_grad_sq = 0.0f;
        cudaMemcpy(&gpu_penalty_grad_sq, device_penalty_grad_sq, sizeof(float),
                   cudaMemcpyDeviceToHost);
        host_penalty_grad_sq += gpu_penalty_grad_sq;
    }
#endif
    // penalty_grad = regularization * diff.  Report the post-accumulation
    // objective whose gradient remains after apply_optimizer_step divides by
    // accumulation_steps: 0.5 * base_regularization * ||diff||^2.
    const double inverse_regularization_sq =
        1.0 / (static_cast<double>(regularization) * regularization);
    return static_cast<float>(
        0.5 * static_cast<double>(base_regularization) *
        host_penalty_grad_sq * inverse_regularization_sq);
}

void begin_moe_aux_accumulation(Trainer& trainer) {
    if (!trainer.model) return;
    for (auto& layer : trainer.model->layers) {
        if (layer && layer->router) {
            layer->router->begin_aux_accumulation();
        }
    }
}

float apply_moe_aux_regularization(Trainer& trainer, int accumulation_steps) {
    if (!trainer.model) {
        return 0.0f;
    }

    if (trainer.moe_aux_loss_scale <= 0.0f) {
        for (auto& layer : trainer.model->layers) {
            if (layer && layer->router) layer->router->cancel_aux_accumulation();
        }
        return 0.0f;
    }

    // Opt-in: differentiable Switch-Transformer aux loss instead of the legacy
    // constant-per-row heuristic.  The heuristic is not the gradient of any loss
    // and is largely a no-op once the gate is ternary; the Switch path computes
    // the exact gradient w.r.t. the router logits and backprops it through the
    // gate.  Default OFF preserves the historical behavior byte-for-byte.
    // K2: differentiable Switch-Transformer aux loss is ON by default (the
    // legacy constant-per-row heuristic is not the gradient of any loss and is
    // ~a no-op once the gate is ternary).  NSOS_MOE_SWITCH_AUX=0 restores the
    // legacy heuristic for comparison.
    // The historical heuristic is not the derivative of any scalar objective.
    // Production training therefore always uses the exact Switch objective;
    // keeping a runtime switch here would make reported loss and gradients
    // diverge again.
    constexpr bool switch_aux = true;
    const int objective_divisor = std::max(accumulation_steps, 1);
    float total_effective_loss = 0.0f;
    const float effective_aux_scale =
        trainer.moe_aux_loss_scale *
        static_cast<float>(std::max(accumulation_steps, 1));

    for (auto& layer : trainer.model->layers) {
        if (!layer || !layer->router || !layer->router->gate) {
            continue;
        }
        if (switch_aux) {
            total_effective_loss +=
                layer->router->accumulate_switch_aux_grad(
                    layer->router->aux_loss_coef * effective_aux_scale) /
                static_cast<float>(objective_divisor);
            continue;
        }
        layer->router->finalize_aux_accumulation();
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

#ifdef USE_CUDA
        if (gate_weight.grad.get_device() == Device::GPU &&
            gpu_custom_kernels_supported()) {
            // (auditoria #6) device-side: sobe E floats e soma por linha no
            // kernel — elimina o cpu()/copy_from de [E,d] por camada por step.
            std::vector<float> imbalance(loads.size());
            for (size_t expert = 0; expert < loads.size(); ++expert) {
                imbalance[expert] = (loads[expert] - mean_load) *
                                    layer->router->aux_loss_coef *
                                    effective_aux_scale;
            }
            thread_local float* d_imb = nullptr;   // K6: race-free persistent buffer
            thread_local size_t d_imb_cap = 0;
            if (loads.size() > d_imb_cap) {
                if (d_imb) cudaFree(d_imb);
                d_imb = nullptr;
                if (cudaMalloc(&d_imb, loads.size() * sizeof(float)) != cudaSuccess) {
                    (void)cudaGetLastError();
                    d_imb_cap = 0;
                }
                else { d_imb_cap = loads.size(); }
            }
            if (d_imb) {
                cudaMemcpy(d_imb, imbalance.data(), loads.size() * sizeof(float),
                           cudaMemcpyHostToDevice);
                launch_add_row_broadcast(gate_weight.grad.raw_data(), d_imb,
                                         static_cast<int>(loads.size()),
                                         gate_weight.grad.shape[1]);
                trainer_check_cuda("launch_add_row_broadcast");
                continue;
            }
        }
#endif
        Tensor grad_host =
            gate_weight.grad.get_device() == Device::GPU ? gate_weight.grad.cpu() : gate_weight.grad;
        float* grad_ptr = grad_host.data();
        const int row_width = grad_host.shape[1];
        for (size_t expert = 0; expert < loads.size(); ++expert) {
            const float imbalance =
                (loads[expert] - mean_load) * layer->router->aux_loss_coef *
                effective_aux_scale;
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
    return total_effective_loss;
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

struct CriticalityMetric {
    Parameter* parameter = nullptr;
    float gamma = 0.0f;
    float gain = 0.0f;
    int fan_in = 0;
};

bool strict_env_flag(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw) return false;
    const std::string value(raw);
    if (value == "1") return true;
    if (value == "0") return false;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

float strict_env_float(const char* name, float fallback, float lo, float hi) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    char* end = nullptr;
    const float value = std::strtof(raw, &end);
    if (!end || end == raw || *end != '\0' || !std::isfinite(value) ||
        value < lo || value > hi) {
        throw std::invalid_argument(std::string(name) + " is outside its valid range");
    }
    return value;
}

int strict_env_int(const char* name, int fallback, int lo) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (!end || end == raw || *end != '\0' || value < lo ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " is outside its valid range");
    }
    return static_cast<int>(value);
}

bool criticality_regularizer_enabled() {
    return strict_env_flag("NSOS_CRIT_REG");
}

bool criticality_lr_enabled() {
    return strict_env_flag("NSOS_CRIT_LR");
}

std::vector<Parameter*> active_ternary_weights(Trainer& trainer) {
    std::vector<Parameter*> result;
    if (!trainer.model) return result;
    for (BitLinear* layer : trainer.model->collect_bitlinear_layers()) {
        if (!layer || layer->quantization_sensitive() ||
            layer->reference_path_enabled() || !layer->has_full_precision_weight()) {
            continue;
        }
        Parameter* parameter = &layer->weight;
        if (parameter->data.shape.size() != 2 || parameter->data.shape[0] <= 1 ||
            parameter->data.shape[1] <= 1 ||
            parameter->grad.size != parameter->data.size) {
            continue;
        }
        result.push_back(parameter);
    }
    return result;
}

#ifdef USE_CUDA
unsigned char* criticality_device_buffer(size_t bytes) {
    thread_local unsigned char* buffer = nullptr;
    thread_local size_t capacity = 0;
    if (bytes == 0) return nullptr;
    if (bytes > capacity) {
        if (buffer) cudaFree(buffer);
        buffer = nullptr;
        const cudaError_t status = cudaMalloc(&buffer, bytes);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            capacity = 0;
            return nullptr;
        }
        capacity = bytes;
    }
    return buffer;
}

size_t align_buffer_offset(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

void check_cuda_copy(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}
#endif

std::vector<CriticalityMetric> measure_active_criticality(Trainer& trainer) {
    const std::vector<Parameter*> weights = active_ternary_weights(trainer);
    std::vector<CriticalityMetric> metrics(weights.size());
    std::vector<size_t> gpu_indices;
    for (size_t index = 0; index < weights.size(); ++index) {
        Parameter* p = weights[index];
        metrics[index].parameter = p;
        metrics[index].fan_in = p->data.shape[1];
#ifdef USE_CUDA
        if (p->data.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
            gpu_indices.push_back(index);
            continue;
        }
#endif
        Tensor host = p->data.get_device() == Device::GPU ? p->data.cpu() : p->data;
        const float* values = host.data();
        double abs_sum = 0.0;
        for (int i = 0; i < host.size; ++i) abs_sum += std::fabs(values[i]);
        const float gamma = static_cast<float>(
            abs_sum / static_cast<double>(std::max<int64_t>(host.size, 1)));
        int zeros = 0;
        const float threshold = 0.5f * gamma;
        for (int i = 0; i < host.size; ++i) {
            zeros += std::fabs(values[i]) < threshold ? 1 : 0;
        }
        metrics[index].gamma = gamma;
        metrics[index].gain = gamma * gamma *
            (1.0f - static_cast<float>(zeros) /
                        static_cast<float>(std::max<int64_t>(host.size, 1))) *
            static_cast<float>(metrics[index].fan_in);
    }
#ifdef USE_CUDA
    if (!gpu_indices.empty()) {
        const int count = static_cast<int>(gpu_indices.size());
        std::vector<float*> host_weights(static_cast<size_t>(count));
        std::vector<unsigned long long> host_offsets(static_cast<size_t>(count + 1));
        std::vector<int> host_fan_in(static_cast<size_t>(count));
        std::vector<float> host_gammas(static_cast<size_t>(count));
        std::vector<float> host_gains(static_cast<size_t>(count));
        unsigned long long total = 0;
        for (int i = 0; i < count; ++i) {
            const size_t metric_index = gpu_indices[static_cast<size_t>(i)];
            Parameter* p = metrics[metric_index].parameter;
            host_weights[static_cast<size_t>(i)] = p->data.raw_data();
            host_offsets[static_cast<size_t>(i)] = total;
            total += static_cast<unsigned long long>(p->data.size);
            host_fan_in[static_cast<size_t>(i)] = metrics[metric_index].fan_in;
        }
        host_offsets[static_cast<size_t>(count)] = total;
        size_t offset = sizeof(float*) * static_cast<size_t>(count);
        offset = align_buffer_offset(offset, alignof(unsigned long long));
        const size_t offsets_offset = offset;
        offset += sizeof(unsigned long long) * static_cast<size_t>(count + 1);
        offset = align_buffer_offset(offset, alignof(int));
        const size_t fan_offset = offset;
        offset += sizeof(int) * static_cast<size_t>(count);
        offset = align_buffer_offset(offset, alignof(float));
        const size_t gamma_offset = offset;
        offset += sizeof(float) * static_cast<size_t>(count);
        const size_t gain_offset = offset;
        offset += sizeof(float) * static_cast<size_t>(count);
        unsigned char* buffer = criticality_device_buffer(offset);
        if (!buffer) throw std::runtime_error("Could not allocate criticality GPU metadata");
        auto* device_weights = reinterpret_cast<float* const*>(buffer);
        auto* device_offsets = reinterpret_cast<unsigned long long*>(buffer + offsets_offset);
        auto* device_fan_in = reinterpret_cast<int*>(buffer + fan_offset);
        auto* device_gammas = reinterpret_cast<float*>(buffer + gamma_offset);
        auto* device_gains = reinterpret_cast<float*>(buffer + gain_offset);
        check_cuda_copy(cudaMemcpy(buffer, host_weights.data(),
                                   sizeof(float*) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality pointer upload");
        check_cuda_copy(cudaMemcpy(device_offsets, host_offsets.data(),
                                   sizeof(unsigned long long) * static_cast<size_t>(count + 1),
                                   cudaMemcpyHostToDevice), "criticality offset upload");
        check_cuda_copy(cudaMemcpy(device_fan_in, host_fan_in.data(),
                                   sizeof(int) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality fan-in upload");
        launch_multi_tensor_criticality_metrics(
            device_weights, device_offsets, device_fan_in, count,
            device_gammas, device_gains);
        trainer_check_cuda("launch_multi_tensor_criticality_metrics");
        check_cuda_copy(cudaMemcpy(host_gammas.data(), device_gammas,
                                   sizeof(float) * static_cast<size_t>(count),
                                   cudaMemcpyDeviceToHost), "criticality gamma download");
        check_cuda_copy(cudaMemcpy(host_gains.data(), device_gains,
                                   sizeof(float) * static_cast<size_t>(count),
                                   cudaMemcpyDeviceToHost), "criticality gain download");
        for (int i = 0; i < count; ++i) {
            CriticalityMetric& metric = metrics[gpu_indices[static_cast<size_t>(i)]];
            metric.gamma = host_gammas[static_cast<size_t>(i)];
            metric.gain = host_gains[static_cast<size_t>(i)];
        }
    }
#endif
    return metrics;
}

void add_criticality_gradient(
    const std::vector<CriticalityMetric>& metrics,
    const std::vector<float>& coefficients) {
#ifdef USE_CUDA
    std::vector<size_t> gpu_indices;
#endif
    for (size_t index = 0; index < metrics.size(); ++index) {
        Parameter* p = metrics[index].parameter;
        const float coefficient = coefficients[index];
        if (!p || coefficient == 0.0f) continue;
#ifdef USE_CUDA
        if (p->data.get_device() == Device::GPU &&
            p->grad.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
            gpu_indices.push_back(index);
            continue;
        }
#endif
        Tensor host_weight = p->data.get_device() == Device::GPU
                                 ? p->data.cpu()
                                 : p->data;
        Tensor host_grad = p->grad.get_device() == Device::GPU
                               ? p->grad.cpu()
                               : p->grad;
        const float* weight = host_weight.data();
        float* grad = host_grad.data();
        for (int i = 0; i < host_grad.size; ++i) {
            const float sign =
                weight[i] > 0.0f ? 1.0f : (weight[i] < 0.0f ? -1.0f : 0.0f);
            grad[i] += coefficient * sign;
        }
        if (p->grad.get_device() == Device::GPU) {
            p->grad.copy_from(host_grad.to(Device::GPU));
        }
    }
#ifdef USE_CUDA
    if (!gpu_indices.empty()) {
        const int count = static_cast<int>(gpu_indices.size());
        std::vector<float*> host_weights(static_cast<size_t>(count));
        std::vector<float*> host_grads(static_cast<size_t>(count));
        std::vector<unsigned long long> host_offsets(static_cast<size_t>(count + 1));
        std::vector<float> host_coefficients(static_cast<size_t>(count));
        unsigned long long total = 0;
        for (int i = 0; i < count; ++i) {
            const size_t index = gpu_indices[static_cast<size_t>(i)];
            Parameter* p = metrics[index].parameter;
            host_weights[static_cast<size_t>(i)] = p->data.raw_data();
            host_grads[static_cast<size_t>(i)] = p->grad.raw_data();
            host_offsets[static_cast<size_t>(i)] = total;
            total += static_cast<unsigned long long>(p->data.size);
            host_coefficients[static_cast<size_t>(i)] = coefficients[index];
        }
        host_offsets[static_cast<size_t>(count)] = total;
        size_t offset = sizeof(float*) * static_cast<size_t>(count) * 2;
        offset = align_buffer_offset(offset, alignof(unsigned long long));
        const size_t offsets_offset = offset;
        offset += sizeof(unsigned long long) * static_cast<size_t>(count + 1);
        offset = align_buffer_offset(offset, alignof(float));
        const size_t coefficients_offset = offset;
        offset += sizeof(float) * static_cast<size_t>(count);
        unsigned char* buffer = criticality_device_buffer(offset);
        if (!buffer) throw std::runtime_error("Could not allocate criticality GPU gradient metadata");
        auto* device_weights = reinterpret_cast<float* const*>(buffer);
        auto* device_grads = device_weights + count;
        auto* device_offsets = reinterpret_cast<unsigned long long*>(buffer + offsets_offset);
        auto* device_coefficients = reinterpret_cast<float*>(buffer + coefficients_offset);
        check_cuda_copy(cudaMemcpy(buffer, host_weights.data(),
                                   sizeof(float*) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality weight pointer upload");
        check_cuda_copy(cudaMemcpy(buffer + sizeof(float*) * static_cast<size_t>(count),
                                   host_grads.data(),
                                   sizeof(float*) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality grad pointer upload");
        check_cuda_copy(cudaMemcpy(device_offsets, host_offsets.data(),
                                   sizeof(unsigned long long) * static_cast<size_t>(count + 1),
                                   cudaMemcpyHostToDevice), "criticality grad offset upload");
        check_cuda_copy(cudaMemcpy(device_coefficients, host_coefficients.data(),
                                   sizeof(float) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality coefficient upload");
        launch_multi_tensor_criticality_grad(
            device_weights, device_grads, device_offsets, device_coefficients,
            count, total);
        trainer_check_cuda("launch_multi_tensor_criticality_grad");
    }
#endif
}

// R = eta/2 * mean_l(log(g_l/g0_l)^2), with the discrete zero mask held
// constant. SymPy gives dR/dw_i = 2*eta*log(g/g0)*sign(w_i)/(L*n*gamma).
// The term is accumulated before clipping and Adam, so weights and moments stay
// in the same coordinate system.
float apply_criticality_regularization_gradient(
    Trainer& trainer, const std::vector<CriticalityMetric>& metrics,
    int accumulation_steps) {
    if (!criticality_regularizer_enabled()) return 0.0f;
    const float eta = strict_env_float(
        "NSOS_CRIT_REG_ETA", 0.2f, std::numeric_limits<float>::min(), 1.0f);
    const int every = strict_env_int("NSOS_CRIT_REG_EVERY", 10, 1);
    if (trainer.global_step_count % every != 0) return 0.0f;

    std::vector<float> log_ratios(metrics.size(), 0.0f);
    size_t active = 0;
    for (size_t index = 0; index < metrics.size(); ++index) {
        const CriticalityMetric& metric = metrics[index];
        if (!metric.parameter || !std::isfinite(metric.gamma) || metric.gamma <= 0.0f ||
            !std::isfinite(metric.gain) || metric.gain <= 0.0f) {
            continue;
        }
        auto baseline = trainer.crit_g0_state.find(metric.parameter);
        if (baseline == trainer.crit_g0_state.end()) {
            trainer.crit_g0_state.emplace(metric.parameter, metric.gain);
            continue;
        }
        if (!std::isfinite(baseline->second) || baseline->second <= 0.0f) {
            throw std::runtime_error("Invalid criticality baseline");
        }
        log_ratios[index] = std::log(metric.gain / baseline->second);
        if (!std::isfinite(log_ratios[index])) {
            throw std::runtime_error("Non-finite criticality regularizer");
        }
        ++active;
    }
    if (active == 0) return 0.0f;
    const float inverse_active = 1.0f / static_cast<float>(active);
    std::vector<float> coefficients(metrics.size(), 0.0f);
    double loss = 0.0;
    for (size_t index = 0; index < metrics.size(); ++index) {
        if (log_ratios[index] == 0.0f || metrics[index].gamma <= 0.0f) continue;
        loss += 0.5 * static_cast<double>(eta) * log_ratios[index] *
                log_ratios[index] * inverse_active;
        coefficients[index] =
            2.0f * eta * log_ratios[index] * inverse_active /
            (static_cast<float>(metrics[index].parameter->data.size) *
             metrics[index].gamma) *
            static_cast<float>(std::max(accumulation_steps, 1));
    }
    add_criticality_gradient(metrics, coefficients);
    if (!std::isfinite(loss)) throw std::runtime_error("Non-finite criticality loss");
    return static_cast<float>(loss);
}

void apply_composed_criticality_lr_control(
    Trainer& trainer, const std::vector<CriticalityMetric>& metrics) {
    trainer.criticality_lr_scale.clear();
    if (!criticality_lr_enabled()) return;
    const float eta = strict_env_float(
        "NSOS_CRIT_LR_ETA", 0.25f, std::numeric_limits<float>::min(), 1.0f);
    for (const CriticalityMetric& metric : metrics) {
        if (!metric.parameter || !std::isfinite(metric.gain) || metric.gain <= 0.0f) {
            continue;
        }
        trainer.criticality_lr_scale.emplace(
            metric.parameter,
            std::clamp(std::pow(1.0f / metric.gain, eta), 0.5f, 2.0f));
    }
}

#ifdef USE_CUDA
namespace {

// NSOS_FUSED_OPT=0 desliga o passo fundido (braço A/B; default ON).
bool fused_optimizer_enabled() {
    const char* e = std::getenv("NSOS_FUSED_OPT");
    return e == nullptr || e[0] != '0';
}

// Buffer device persistente para os metadados do passo fundido
// (ponteiros w/g/m/v + offsets + flags de weight-decay).  Cresce sob demanda;
// re-upload por step é obrigatório porque add_grad recria o tensor de grad
// (ponteiro muda a cada backward) — ~30 KB H2D, custo ~µs.
unsigned char* fused_opt_meta_buffer(size_t bytes) {
    thread_local unsigned char* buf = nullptr;
    thread_local size_t cap = 0;
    if (bytes == 0) return nullptr;
    if (bytes > cap) {
        if (buf) cudaFree(buf);
        buf = nullptr;
        if (cudaMalloc(&buf, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            cap = 0;
            return nullptr;
        }
        cap = bytes;
    }
    return buf;
}

}  // namespace

// Item #4 da auditoria: o passo do otimizador inteiro (scale de acumulação +
// clip global + AdamW) em 2 kernels multi-tensor + 1 D2H de 4 bytes, no lugar
// de ~3-4 launches POR PARÂMETRO (~1850-2450/step no v11; opt medido em
// 226-490 ms).  Equivalência exata com o caminho antigo:
//   norm(g·s) = s·norm(g)  e  adamw(g·s·coeff) == adamw(g, gscale=s·coeff)
// — mesma ordem de operações (norm antes do step++, lr depois), mesmo guard
// do clip (só quando total>max, com o mesmo +1e-6).  Diferença observável
// única: os tensores .grad ficam CRUS após o passo (antes ficavam
// escalados+clipados) — estado interno entre steps, zerado no passo seguinte.
// Retorna false SEM efeitos colaterais se qualquer parâmetro não for elegível
// (qualquer coisa fora da GPU) — o caminho antigo roda por inteiro.
static bool apply_optimizer_step_fused(Trainer& trainer,
                                       const std::vector<Parameter*>& params,
                                       int accumulation_steps,
                                       float* grad_norm_out) {
    if (!fused_optimizer_enabled() || !gpu_custom_kernels_supported()) {
        return false;
    }
    // K4: the fused step folds the atomicAdd clip-norm + multi-tensor AdamW; in
    // deterministic mode fall back to the ordered per-parameter path (with the
    // ordered host clip above) for bit-reproducibility.
    if (determinism::deterministic_reductions_enabled()) {
        return false;
    }
    float* d_accum = clip_norm_accumulator();
    if (!d_accum) {
        return false;
    }

    std::vector<Parameter*> active;
    active.reserve(params.size());
    for (auto* p : params) {
        if (!p || p->grad.size == 0) continue;
        if (p->data.get_device() != Device::GPU ||
            p->grad.get_device() != Device::GPU) {
            return false;
        }
        active.push_back(p);
    }
    if (active.empty()) {
        return false;
    }

    for (auto* p : active) {
        auto m_it = trainer.m_state.find(p);
        if (m_it == trainer.m_state.end()) {
            trainer.m_state.emplace(p, Tensor::zeros(p->data.shape.dims,
                                                     p->data.get_device()));
            trainer.v_state.emplace(p, Tensor::zeros(p->data.shape.dims,
                                                     p->data.get_device()));
        } else if (m_it->second.shape != p->data.shape ||
                   m_it->second.get_device() != p->data.get_device()) {
            m_it->second = Tensor::zeros(p->data.shape.dims, p->data.get_device());
            trainer.v_state[p] =
                Tensor::zeros(p->data.shape.dims, p->data.get_device());
        }
        if (!can_use_gpu_optimizer(*p, trainer.m_state[p], trainer.v_state[p])) {
            return false;
        }
    }

    const int n = static_cast<int>(active.size());
    thread_local std::vector<unsigned char> staging;  // K6: replica-safe
    const size_t ptr_bytes = sizeof(float*) * static_cast<size_t>(n) * 4;
    const size_t off_bytes =
        sizeof(unsigned long long) * static_cast<size_t>(n + 1);
    const size_t wd_bytes = static_cast<size_t>(n);
    const size_t lr_offset =
        (ptr_bytes + off_bytes + wd_bytes + alignof(float) - 1) &
        ~(alignof(float) - 1);
    const size_t lr_bytes = sizeof(float) * static_cast<size_t>(n);
    staging.resize(lr_offset + lr_bytes);
    float** h_w = reinterpret_cast<float**>(staging.data());
    float** h_g = h_w + n;
    float** h_m = h_w + 2 * n;
    float** h_v = h_w + 3 * n;
    auto* h_off =
        reinterpret_cast<unsigned long long*>(staging.data() + ptr_bytes);
    unsigned char* h_wd = staging.data() + ptr_bytes + off_bytes;
    float* h_lr = reinterpret_cast<float*>(staging.data() + lr_offset);
    unsigned long long total = 0;
    for (int i = 0; i < n; ++i) {
        Parameter* p = active[static_cast<size_t>(i)];
        h_w[i] = p->data.raw_data();
        h_g[i] = p->grad.raw_data();
        h_m[i] = trainer.m_state[p].raw_data();
        h_v[i] = trainer.v_state[p].raw_data();
        h_off[i] = total;
        total += static_cast<unsigned long long>(p->data.size);
        h_wd[i] = should_apply_weight_decay(*p) ? 1 : 0;
        h_lr[i] = trainer.lr_scale_for(p);
    }
    h_off[n] = total;
    if (total == 0) {
        return false;
    }

    unsigned char* d_meta = fused_opt_meta_buffer(staging.size());
    if (!d_meta) {
        return false;
    }
    cudaMemcpy(d_meta, staging.data(), staging.size(), cudaMemcpyHostToDevice);
    auto* d_w = reinterpret_cast<float* const*>(d_meta);
    auto* d_g = d_w + n;
    auto* d_m = d_w + 2 * n;
    auto* d_v = d_w + 3 * n;
    const auto* d_off =
        reinterpret_cast<const unsigned long long*>(d_meta + ptr_bytes);
    const unsigned char* d_wd = d_meta + ptr_bytes + off_bytes;
    const float* d_lr = reinterpret_cast<const float*>(d_meta + lr_offset);

    cudaMemsetAsync(d_accum, 0, sizeof(float), 0);
    launch_multi_tensor_sqsum(d_accum, d_w, d_g, d_m, d_v, d_off, d_wd, n, total);
    trainer_check_cuda("launch_multi_tensor_sqsum");
    float sq = 0.0f;
    cudaMemcpy(&sq, d_accum, sizeof(float), cudaMemcpyDeviceToHost);

    const float acc_scale =
        1.0f / static_cast<float>(std::max(accumulation_steps, 1));
    const float total_norm = acc_scale * std::sqrt(std::max(sq, 0.0f));
    if (grad_norm_out) {
        *grad_norm_out = total_norm;
    }
    float coeff = 1.0f;
    if (total_norm > trainer.max_grad_norm) {
        coeff = trainer.max_grad_norm / (total_norm + 1e-6f);
    }
    const float gscale = acc_scale * coeff;

    trainer.global_step_count++;
    const float cur_lr = compute_current_lr(trainer);
    const float bc1 = 1.0f - std::pow(trainer.beta1, trainer.global_step_count);
    const float bc2 = 1.0f - std::pow(trainer.beta2, trainer.global_step_count);

    launch_multi_tensor_adamw(d_w, d_g, d_m, d_v, d_off, d_wd, d_lr, n,
                              total, gscale, trainer.beta1, trainer.beta2,
                              bc1, bc2, cur_lr, trainer.eps,
                              trainer.weight_decay);
    trainer_check_cuda("launch_multi_tensor_adamw");
    for (auto* p : active) {
        p->mark_updated();
    }
    return true;
}
#endif  // USE_CUDA

float apply_optimizer_step(Trainer& trainer,
                           const std::vector<Parameter*>& params,
                           int accumulation_steps,
                           float* grad_norm_out = nullptr) {
    if (!std::isfinite(trainer.learning_rate) || trainer.learning_rate < 0.0f ||
        !std::isfinite(trainer.beta1) || trainer.beta1 < 0.0f || trainer.beta1 >= 1.0f ||
        !std::isfinite(trainer.beta2) || trainer.beta2 < 0.0f || trainer.beta2 >= 1.0f ||
        !std::isfinite(trainer.eps) || trainer.eps <= 0.0f ||
        !std::isfinite(trainer.weight_decay) || trainer.weight_decay < 0.0f ||
        !std::isfinite(trainer.max_grad_norm) || trainer.max_grad_norm <= 0.0f) {
        throw std::invalid_argument("Trainer optimizer hyperparameters are invalid");
    }
    std::vector<CriticalityMetric> criticality_metrics;
    if (criticality_regularizer_enabled() || criticality_lr_enabled()) {
        criticality_metrics = measure_active_criticality(trainer);
    }
    apply_composed_criticality_lr_control(trainer, criticality_metrics);
    const float criticality_loss =
        apply_criticality_regularization_gradient(
            trainer, criticality_metrics, accumulation_steps);
    for (Parameter* parameter : params) {
        if (!parameter) continue;
        const float scale = trainer.lr_scale_for(parameter);
        if (!std::isfinite(scale) || scale <= 0.0f) {
            throw std::invalid_argument(
                "Effective per-parameter learning-rate scale is invalid");
        }
    }
    ensure_finite_optimizer_inputs(params);
#ifdef USE_CUDA
    if (apply_optimizer_step_fused(trainer, params, accumulation_steps,
                                   grad_norm_out)) {
        return criticality_loss;
    }
#endif
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

        // §6 closed loop: effective lr = global lr * per-parameter scale
        // (1.0 when no controller is active).
        const float p_lr = cur_lr * trainer.lr_scale_for(p);

        // 4-bit optimizer states (CPU path).  GPU params fall through to the
        // FP32 launch_adamw_update_kernel path below until the 4-bit CUDA
        // kernel lands.
        if (trainer.optimizer_state_bits == 4 &&
            p->data.get_device() == Device::CPU) {
            apply_adam_step_4bit(trainer, p, p_lr, bc1, bc2);
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
                p_lr,
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
                w[i] -= p_lr * trainer.weight_decay * w[i];
            }
            w[i] -= p_lr * m_hat / (std::sqrt(v_hat) + trainer.eps);
        }
        p->mark_updated();
    }

    return criticality_loss;
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
    // K6: thread_local persistent device buffer (race-free by construction).
    thread_local int* buf = nullptr;
    thread_local int cap = 0;
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

float* repetition_loss_device_buffer() {
    thread_local float* buffer = [] {
        float* value = nullptr;
        if (cudaMalloc(&value, sizeof(float)) != cudaSuccess) {
            value = nullptr;
            (void)cudaGetLastError();
        }
        return value;
    }();
    return buffer;
}
#endif

std::vector<float> supervised_row_weights(
    const Trainer& trainer, const std::vector<int>& answer_tokens) {
    std::vector<float> weights(answer_tokens.size(), 1.0f);
    if (answer_tokens.empty()) return weights;
    weights.front() *= std::max(trainer.first_token_loss_scale, 0.0f);
    if (answer_tokens.back() == trainer.eos_token_id) {
        weights.back() *= std::max(trainer.eos_loss_scale, 0.0f);
    }
    return weights;
}

float effective_logit_l2_beta(const Trainer& trainer) {
    if (trainer.logit_l2_beta != 0.0f &&
        trainer.pantheon_vib_beta != 0.0f &&
        trainer.logit_l2_beta != trainer.pantheon_vib_beta) {
        throw std::invalid_argument(
            "logit_l2_beta conflicts with deprecated pantheon_vib_beta");
    }
    const float beta = trainer.logit_l2_beta != 0.0f
                           ? trainer.logit_l2_beta
                           : trainer.pantheon_vib_beta;
    if (!std::isfinite(beta) || beta < 0.0f) {
        throw std::invalid_argument("logit_l2_beta must be finite and non-negative");
    }
    return beta;
}

float add_logit_l2_objective(const Tensor& logits, float beta, Tensor& grad) {
    if (beta <= 0.0f || logits.size == 0) return 0.0f;
    if (grad.shape != logits.shape || grad.get_device() != logits.get_device()) {
        throw std::invalid_argument("logit L2 requires logits/gradient shape and device parity");
    }
    const float inverse_count = 1.0f / static_cast<float>(logits.size);
    const float norm = logits.norm();
    grad = grad.add(logits.mul(beta * inverse_count));
    return 0.5f * beta * norm * norm * inverse_count;
}

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

float apply_repetition_unlikelihood(const Trainer& trainer,
                                    const std::vector<int>& answer_tokens,
                                    const Tensor& answer_logits,
                                    Tensor& answer_grad) {
    const float scale = std::max(trainer.repetition_unlikelihood_scale, 0.0f);
    if (scale <= 0.0f || answer_tokens.size() < 2 || answer_grad.size == 0) {
        return 0.0f;
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
        float* d_loss = repetition_loss_device_buffer();
        if (d_tokens && d_loss) {
            cudaMemset(d_loss, 0, sizeof(float));
            cudaMemcpy(d_tokens, answer_tokens.data(),
                       static_cast<size_t>(rows) * sizeof(int), cudaMemcpyHostToDevice);
            launch_repetition_unlikelihood_kernel(
                d_loss, answer_grad.raw_data(), probs.raw_data(), d_tokens,
                rows, vocab, scale, trainer.eos_token_id);
            trainer_check_cuda("launch_repetition_unlikelihood_kernel");
            float loss = 0.0f;
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            return loss;
        }
    }
#endif

    Tensor probs_host = probs.get_device() == Device::GPU ? probs.cpu() : probs;
    Tensor grad_host = answer_grad.get_device() == Device::GPU ? answer_grad.cpu() : answer_grad.clone();
    const float* prob_ptr = probs_host.data();
    float* grad_ptr = grad_host.data();
    double loss = 0.0;

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
            loss += -static_cast<double>(scale) *
                    std::log1p(-static_cast<double>(p_neg));
            for (int col = 0; col < vocab; ++col) {
                row_grad[col] -= factor * row_probs[col];
            }
            row_grad[neg_token] += factor;
        }
    }

    restore_staged_tensor(answer_grad, grad_host);
    return static_cast<float>(loss);
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

float apply_repetition_unlikelihood_batch(
    const Trainer& trainer,
    const std::vector<std::vector<int>>& answer_batch,
    const Tensor& answer_logits,
    Tensor& answer_grad) {
    const float scale = std::max(trainer.repetition_unlikelihood_scale, 0.0f);
    if (scale <= 0.0f || answer_batch.empty() || answer_grad.size == 0) {
        return 0.0f;
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
    const float inverse_batch = 1.0f / static_cast<float>(std::max(batch_size, 1));
    const float* prob_ptr = probs_host.data();
    float* grad_ptr = grad_host.data();
    double loss = 0.0;

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
                const float factor = scale * inverse_batch * p_neg / denom;
                loss += -static_cast<double>(scale * inverse_batch) *
                        std::log1p(-static_cast<double>(p_neg));
                for (int col = 0; col < vocab; ++col) {
                    row_grad[col] -= factor * row_probs[col];
                }
                row_grad[neg_token] += factor;
            }
        }
    }

    restore_staged_tensor(answer_grad, grad_host);
    return static_cast<float>(loss);
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
    begin_moe_aux_accumulation(trainer);

    double supervised_loss_sum = 0.0;
    double repetition_loss_sum = 0.0;
    double logit_l2_loss_sum = 0.0;
    double sparse_selector_weighted_sum = 0.0;
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
        const float selector_loss =
            trainer.model->accumulate_sparse_selector_grads(
                static_cast<float>(grouped_inputs.size()));
        sparse_selector_weighted_sum +=
            static_cast<double>(selector_loss) * grouped_inputs.size();

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
            const auto& answer_tokens = grouped_answers[static_cast<size_t>(batch)];
            const std::vector<float> row_weights =
                supervised_row_weights(trainer, answer_tokens);
            auto [supervised_loss, answer_grad] =
                answer_logits.cross_entropy_weighted(answer_tokens, row_weights);
            const float repetition_loss = apply_repetition_unlikelihood(
                trainer,
                grouped_answers[static_cast<size_t>(batch)],
                answer_logits,
                answer_grad);
            const float logit_l2_loss = add_logit_l2_objective(
                answer_logits, effective_logit_l2_beta(trainer), answer_grad);

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
            supervised_loss_sum += supervised_loss;
            repetition_loss_sum += repetition_loss;
            logit_l2_loss_sum += logit_l2_loss;
            ++sample_count;
        }

        const auto _tm_loss1 = tm_now();
        tm_loss += tm_ms(_tm_loss1 - _tm_fwd1).count();
        trainer.model->backward_external(full_grad, ctx);
        const auto _tm_bwd1 = tm_now();
        tm_bwd += tm_ms(_tm_bwd1 - _tm_loss1).count();
        tm_last = _tm_bwd1;
    }

    const int objective_samples = std::max(sample_count, 1);
    const float qat_loss =
        apply_qat_regularization(trainer, objective_samples);
    const float moe_aux_loss =
        apply_moe_aux_regularization(trainer, objective_samples);
    float grad_norm = 0.0f;
    const auto _tm_opt0 = tm_now();
    const float criticality_loss = apply_optimizer_step(
        trainer, params, std::max(sample_count, 1), &grad_norm);
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
    TrainingObjectiveStats objective;
    objective.supervised_cross_entropy = static_cast<float>(
        supervised_loss_sum / static_cast<double>(objective_samples));
    objective.repetition_unlikelihood = static_cast<float>(
        repetition_loss_sum / static_cast<double>(objective_samples));
    objective.logit_l2 = static_cast<float>(
        logit_l2_loss_sum / static_cast<double>(objective_samples));
    objective.sparse_selector = static_cast<float>(
        sparse_selector_weighted_sum / static_cast<double>(objective_samples));
    objective.qat_regularization = qat_loss;
    objective.moe_auxiliary = moe_aux_loss;
    objective.criticality_regularization = criticality_loss;
    objective.total = objective.supervised_cross_entropy +
                      objective.repetition_unlikelihood + objective.logit_l2 +
                      objective.sparse_selector + objective.qat_regularization +
                      objective.moe_auxiliary +
                      objective.criticality_regularization;
    trainer.last_objective_stats = objective;
    record_training_audit_step(trainer, objective.total, grad_norm, params.size());
    return objective.total;
}

} // namespace

Trainer::Trainer(JambaModel* m, float lr) : model(m), learning_rate(lr) {
    if (model) {
        model->set_training_mode(true);
    }
    apply_progressive_qat_phase(*this);
}

Trainer::~Trainer() = default;

void Trainer::save_training_state(const std::string& state_path,
                                  const std::string& model_path) const {
    if (!model) throw std::runtime_error("Trainer has no model");
    const std::filesystem::path model_file(model_path);
    uint64_t model_bytes = 0;
    const uint64_t model_hash =
        training_state_file_hash(model_file, model_bytes);

    const std::filesystem::path destination(state_path);
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const std::filesystem::path temporary =
        destination.parent_path() /
        (destination.filename().string() + ".tmp." +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));

    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open training state for writing");
        }
        write_training_pod(output, kTrainingStateMagic, "magic");
        write_training_pod(output, kTrainingStateVersion, "version");
        write_training_pod(
            output, ModelSerializer::architecture_fingerprint(model),
            "architecture fingerprint");
        write_training_pod(output, model_bytes, "model byte count");
        write_training_pod(output, model_hash, "model hash");
        write_training_pod(output, model->training_rng_sequence(),
                           "dropout RNG sequence");

        for (float value : {learning_rate, beta1, beta2, eps, weight_decay,
                            max_grad_norm, min_learning_rate_scale,
                            first_token_loss_scale, eos_loss_scale,
                            repetition_unlikelihood_scale, moe_aux_loss_scale,
                            pantheon_vib_beta, logit_l2_beta}) {
            write_training_pod(output, value, "trainer float");
        }
        for (int32_t value : {warmup_steps, global_step_count,
                              total_training_steps, eos_token_id,
                              optimizer_state_bits}) {
            write_training_pod(output, value, "trainer integer");
        }

        auto write_bool = [&](bool value) {
            write_training_pod(output, static_cast<uint8_t>(value ? 1 : 0),
                               "scheduler boolean");
        };
        const auto& scheduler = phase_scheduler;
        write_bool(scheduler.progressive_qat_enabled);
        write_training_pod(output,
                           static_cast<int32_t>(scheduler.semantic_warmup_steps),
                           "semantic warmup");
        write_training_pod(output,
                           static_cast<int32_t>(scheduler.qat_start_step),
                           "QAT start");
        write_training_pod(
            output, static_cast<int32_t>(scheduler.quantized_precision_bits),
            "QAT precision");
        write_training_pod(output, scheduler.ternary_regularization,
                           "ternary regularization");
        write_bool(scheduler.auxiliary_stack_enabled);
        write_bool(scheduler.auxiliary_session_adapt_enabled);
        write_bool(scheduler.auxiliary_reasoning_enabled);
        write_bool(scheduler.auxiliary_memory_enabled);
        for (int32_t value : {
                 scheduler.auxiliary_reasoning_iterations,
                 scheduler.auxiliary_reasoning_simulations}) {
            write_training_pod(output, value, "auxiliary scheduler integer");
        }
        write_training_pod(output, scheduler.auxiliary_memory_blend,
                           "auxiliary memory blend");
        for (int32_t value : {
                 scheduler.auxiliary_every_steps,
                 scheduler.auxiliary_prompt_max_tokens,
                 scheduler.auxiliary_answer_max_tokens,
                 scheduler.auxiliary_memory_scope}) {
            write_training_pod(output, value, "auxiliary scheduler integer");
        }

        const auto parameters = stable_training_parameters(model);
        write_training_pod(output, static_cast<uint32_t>(parameters.size()),
                           "parameter count");
        for (const auto& [parameter, stable_name] : parameters) {
            write_training_string(output, stable_name);
            write_training_pod(output,
                               static_cast<uint64_t>(parameter->data.size),
                               "parameter elements");

            std::vector<float> m_values;
            std::vector<float> v_values;
            bool has_moments = false;
            const auto quant_it = quant_state.find(parameter);
            if (quant_it != quant_state.end()) {
                if (quant_it->second.n != parameter->data.size) {
                    throw std::runtime_error(
                        "Quantized optimizer state shape mismatch for " +
                        stable_name);
                }
                m_values.resize(static_cast<size_t>(parameter->data.size));
                v_values.resize(static_cast<size_t>(parameter->data.size));
                quant4_load_m(quant_it->second, m_values.data(),
                              parameter->data.size);
                quant4_load_v(quant_it->second, v_values.data(),
                              parameter->data.size);
                has_moments = true;
            } else {
                const auto m_it = m_state.find(parameter);
                const auto v_it = v_state.find(parameter);
                if ((m_it == m_state.end()) != (v_it == v_state.end())) {
                    throw std::runtime_error(
                        "Incomplete Adam state for " + stable_name);
                }
                if (m_it != m_state.end()) {
                    if (m_it->second.shape != parameter->data.shape ||
                        v_it->second.shape != parameter->data.shape) {
                        throw std::runtime_error(
                            "Adam state shape mismatch for " + stable_name);
                    }
                    Tensor m_host = m_it->second.cpu();
                    Tensor v_host = v_it->second.cpu();
                    m_values.assign(m_host.data(),
                                    m_host.data() + m_host.size);
                    v_values.assign(v_host.data(),
                                    v_host.data() + v_host.size);
                    has_moments = true;
                }
            }

            const auto criticality_it = crit_g0_state.find(parameter);
            const auto external_lr_it = external_lr_scale.find(parameter);
            const auto criticality_lr_it = criticality_lr_scale.find(parameter);
            if ((criticality_it != crit_g0_state.end() &&
                 (!std::isfinite(criticality_it->second) ||
                  criticality_it->second <= 0.0f)) ||
                (external_lr_it != external_lr_scale.end() &&
                 (!std::isfinite(external_lr_it->second) ||
                  external_lr_it->second <= 0.0f)) ||
                (criticality_lr_it != criticality_lr_scale.end() &&
                 (!std::isfinite(criticality_lr_it->second) ||
                  criticality_lr_it->second <= 0.0f))) {
                throw std::runtime_error(
                    "Invalid per-parameter controller state for " + stable_name);
            }
            uint8_t flags = has_moments ? 1u : 0u;
            if (criticality_it != crit_g0_state.end()) flags |= 2u;
            if (external_lr_it != external_lr_scale.end()) flags |= 4u;
            if (criticality_lr_it != criticality_lr_scale.end()) flags |= 8u;
            write_training_pod(output, flags, "parameter flags");
            write_training_pod(
                output,
                criticality_it == crit_g0_state.end() ? 0.0f
                                                       : criticality_it->second,
                "criticality baseline");
            write_training_pod(
                output,
                external_lr_it == external_lr_scale.end()
                    ? 1.0f
                    : external_lr_it->second,
                "external parameter LR scale");
            write_training_pod(
                output,
                criticality_lr_it == criticality_lr_scale.end()
                    ? 1.0f
                    : criticality_lr_it->second,
                "criticality parameter LR scale");
            if (has_moments) {
                const std::streamsize bytes = static_cast<std::streamsize>(
                    static_cast<uint64_t>(parameter->data.size) * sizeof(float));
                output.write(reinterpret_cast<const char*>(m_values.data()), bytes);
                output.write(reinterpret_cast<const char*>(v_values.data()), bytes);
                if (!output) {
                    throw std::runtime_error(
                        "Training-state moment write failed for " + stable_name);
                }
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("Training-state flush failed");
        output.close();
        if (!output) throw std::runtime_error("Training-state close failed");
        replace_training_state_file(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void Trainer::load_training_state(const std::string& state_path,
                                  const std::string& model_path) {
    if (!model) throw std::runtime_error("Trainer has no model");
    uint64_t actual_model_bytes = 0;
    const uint64_t actual_model_hash = training_state_file_hash(
        std::filesystem::path(model_path), actual_model_bytes);

    std::ifstream input(state_path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open training state");
    const uint32_t magic = read_training_pod<uint32_t>(input, "magic");
    const uint32_t version = read_training_pod<uint32_t>(input, "version");
    if (magic != kTrainingStateMagic ||
        (version != kTrainingStateVersion &&
         version != kTrainingStateLegacyVersion)) {
        throw std::runtime_error("Unsupported or corrupt training-state header");
    }
    const uint32_t fingerprint =
        read_training_pod<uint32_t>(input, "architecture fingerprint");
    if (fingerprint != ModelSerializer::architecture_fingerprint(model)) {
        throw std::runtime_error(
            "Training-state architecture fingerprint does not match model");
    }
    const uint64_t expected_model_bytes =
        read_training_pod<uint64_t>(input, "model byte count");
    const uint64_t expected_model_hash =
        read_training_pod<uint64_t>(input, "model hash");
    if (expected_model_bytes != actual_model_bytes ||
        expected_model_hash != actual_model_hash) {
        throw std::runtime_error(
            "Training state belongs to a different model checkpoint");
    }
    const uint64_t rng_sequence =
        read_training_pod<uint64_t>(input, "dropout RNG sequence");

    TrainingStateMetadata metadata;
    float* metadata_floats[] = {
        &metadata.learning_rate,
        &metadata.beta1,
        &metadata.beta2,
        &metadata.eps,
        &metadata.weight_decay,
        &metadata.max_grad_norm,
        &metadata.min_learning_rate_scale,
        &metadata.first_token_loss_scale,
        &metadata.eos_loss_scale,
        &metadata.repetition_unlikelihood_scale,
        &metadata.moe_aux_loss_scale,
        &metadata.pantheon_vib_beta,
        &metadata.logit_l2_beta};
    for (float* value : metadata_floats) {
        *value = read_training_pod<float>(input, "trainer float");
        if (!std::isfinite(*value)) {
            throw std::runtime_error("Non-finite trainer metadata");
        }
    }
    metadata.warmup_steps =
        read_training_pod<int32_t>(input, "warmup steps");
    metadata.global_step_count =
        read_training_pod<int32_t>(input, "global step");
    metadata.total_training_steps =
        read_training_pod<int32_t>(input, "total training steps");
    metadata.eos_token_id =
        read_training_pod<int32_t>(input, "EOS token");
    metadata.optimizer_state_bits =
        read_training_pod<int32_t>(input, "optimizer state bits");
    if (metadata.global_step_count < 0 || metadata.warmup_steps < 0 ||
        metadata.total_training_steps < 0 ||
        (metadata.optimizer_state_bits != 4 &&
         metadata.optimizer_state_bits != 32)) {
        throw std::runtime_error("Invalid trainer integer metadata");
    }

    auto read_bool = [&]() {
        const uint8_t value =
            read_training_pod<uint8_t>(input, "scheduler boolean");
        if (value > 1) throw std::runtime_error("Invalid scheduler boolean");
        return value != 0;
    };
    auto& scheduler = metadata.phase_scheduler;
    scheduler.progressive_qat_enabled = read_bool();
    scheduler.semantic_warmup_steps =
        read_training_pod<int32_t>(input, "semantic warmup");
    scheduler.qat_start_step =
        read_training_pod<int32_t>(input, "QAT start");
    scheduler.quantized_precision_bits =
        read_training_pod<int32_t>(input, "QAT precision");
    scheduler.ternary_regularization =
        read_training_pod<float>(input, "ternary regularization");
    scheduler.auxiliary_stack_enabled = read_bool();
    scheduler.auxiliary_session_adapt_enabled = read_bool();
    scheduler.auxiliary_reasoning_enabled = read_bool();
    scheduler.auxiliary_memory_enabled = read_bool();
    scheduler.auxiliary_reasoning_iterations =
        read_training_pod<int32_t>(input, "auxiliary reasoning iterations");
    scheduler.auxiliary_reasoning_simulations =
        read_training_pod<int32_t>(input, "auxiliary reasoning simulations");
    scheduler.auxiliary_memory_blend =
        read_training_pod<float>(input, "auxiliary memory blend");
    scheduler.auxiliary_every_steps =
        read_training_pod<int32_t>(input, "auxiliary cadence");
    scheduler.auxiliary_prompt_max_tokens =
        read_training_pod<int32_t>(input, "auxiliary prompt limit");
    scheduler.auxiliary_answer_max_tokens =
        read_training_pod<int32_t>(input, "auxiliary answer limit");
    scheduler.auxiliary_memory_scope =
        read_training_pod<int32_t>(input, "auxiliary memory scope");
    if (!std::isfinite(scheduler.ternary_regularization) ||
        !std::isfinite(scheduler.auxiliary_memory_blend) ||
        scheduler.semantic_warmup_steps < 0 || scheduler.qat_start_step < 0 ||
        scheduler.auxiliary_every_steps <= 0) {
        throw std::runtime_error("Invalid phase scheduler metadata");
    }

    const auto parameters = stable_training_parameters(model);
    std::unordered_map<std::string, Parameter*> parameter_by_name;
    parameter_by_name.reserve(parameters.size());
    for (const auto& [parameter, stable_name] : parameters) {
        parameter_by_name.emplace(stable_name, parameter);
    }
    const uint32_t record_count =
        read_training_pod<uint32_t>(input, "parameter count");
    if (record_count != parameters.size()) {
        throw std::runtime_error(
            "Training-state parameter count does not match model");
    }
    std::vector<TrainingParameterRecord> records;
    records.reserve(record_count);
    std::unordered_map<Parameter*, bool> seen;
    for (uint32_t index = 0; index < record_count; ++index) {
        const std::string stable_name = read_training_string(input);
        const auto found = parameter_by_name.find(stable_name);
        if (found == parameter_by_name.end()) {
            throw std::runtime_error(
                "Training-state parameter not found: " + stable_name);
        }
        Parameter* parameter = found->second;
        if (!seen.emplace(parameter, true).second) {
            throw std::runtime_error(
                "Duplicate training-state parameter: " + stable_name);
        }
        const uint64_t elements =
            read_training_pod<uint64_t>(input, "parameter elements");
        if (elements != static_cast<uint64_t>(parameter->data.size)) {
            throw std::runtime_error(
                "Training-state shape mismatch for " + stable_name);
        }
        const uint8_t flags =
            read_training_pod<uint8_t>(input, "parameter flags");
        const uint8_t known_flags =
            version == kTrainingStateLegacyVersion ? uint8_t{7} : uint8_t{15};
        if ((flags & ~known_flags) != 0) {
            throw std::runtime_error("Unknown training-state parameter flags");
        }
        TrainingParameterRecord record;
        record.parameter = parameter;
        record.has_moments = (flags & 1u) != 0;
        record.has_criticality = (flags & 2u) != 0;
        record.has_external_lr_scale = (flags & 4u) != 0;
        record.has_criticality_lr_scale =
            version >= 2u && (flags & 8u) != 0;
        record.criticality =
            read_training_pod<float>(input, "criticality baseline");
        record.external_lr_scale =
            read_training_pod<float>(input, "external parameter LR scale");
        if (version >= 2u) {
            record.criticality_lr_scale = read_training_pod<float>(
                input, "criticality parameter LR scale");
        }
        if (!std::isfinite(record.criticality) ||
            !std::isfinite(record.external_lr_scale) ||
            record.external_lr_scale <= 0.0f ||
            !std::isfinite(record.criticality_lr_scale) ||
            record.criticality_lr_scale <= 0.0f) {
            throw std::runtime_error(
                "Invalid per-parameter training metadata");
        }
        if (record.has_moments) {
            record.m.resize(static_cast<size_t>(elements));
            record.v.resize(static_cast<size_t>(elements));
            const std::streamsize bytes = static_cast<std::streamsize>(
                elements * sizeof(float));
            input.read(reinterpret_cast<char*>(record.m.data()), bytes);
            input.read(reinterpret_cast<char*>(record.v.data()), bytes);
            if (!input) {
                throw std::runtime_error(
                    "Training-state truncated in optimizer moments");
            }
            for (size_t i = 0; i < record.m.size(); ++i) {
                if (!std::isfinite(record.m[i]) ||
                    !std::isfinite(record.v[i]) || record.v[i] < 0.0f) {
                    throw std::runtime_error(
                        "Invalid optimizer moment in training state");
                }
            }
        }
        records.push_back(std::move(record));
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("Training state has trailing payload");
    }
    if (!input.eof()) {
        throw std::runtime_error("Training-state read failed before EOF");
    }

    // Commit only after the whole sidecar, model digest, and every tensor have
    // validated.  A corrupt state therefore cannot leave a half-mutated Trainer.
    learning_rate = metadata.learning_rate;
    beta1 = metadata.beta1;
    beta2 = metadata.beta2;
    eps = metadata.eps;
    weight_decay = metadata.weight_decay;
    max_grad_norm = metadata.max_grad_norm;
    min_learning_rate_scale = metadata.min_learning_rate_scale;
    first_token_loss_scale = metadata.first_token_loss_scale;
    eos_loss_scale = metadata.eos_loss_scale;
    repetition_unlikelihood_scale = metadata.repetition_unlikelihood_scale;
    moe_aux_loss_scale = metadata.moe_aux_loss_scale;
    pantheon_vib_beta = metadata.pantheon_vib_beta;
    logit_l2_beta = metadata.logit_l2_beta;
    warmup_steps = metadata.warmup_steps;
    global_step_count = metadata.global_step_count;
    total_training_steps = metadata.total_training_steps;
    eos_token_id = metadata.eos_token_id;
    optimizer_state_bits = metadata.optimizer_state_bits;
    phase_scheduler = metadata.phase_scheduler;
    m_state.clear();
    v_state.clear();
    quant_state.clear();
    crit_g0_state.clear();
    external_lr_scale.clear();
    criticality_lr_scale.clear();
    for (auto& record : records) {
        Parameter* parameter = record.parameter;
        if (record.has_moments) {
            if (optimizer_state_bits == 4 &&
                parameter->data.get_device() == Device::CPU) {
                Quant4OptState state;
                quant4_store_m(record.m.data(), parameter->data.size, state);
                const int rows = parameter->data.shape.size() == 2
                                     ? parameter->data.shape[0]
                                     : 0;
                const int cols = parameter->data.shape.size() == 2
                                     ? parameter->data.shape[1]
                                     : 0;
                quant4_store_v(record.v.data(), parameter->data.size,
                               rows, cols, state);
                quant_state.emplace(parameter, std::move(state));
            } else {
                Tensor m_tensor = Tensor::zeros(parameter->data.shape.dims,
                                                parameter->data.get_device());
                Tensor v_tensor = Tensor::zeros(parameter->data.shape.dims,
                                                parameter->data.get_device());
                Tensor m_host = Tensor::from_blob(
                    record.m.data(), parameter->data.shape.dims, Device::CPU);
                Tensor v_host = Tensor::from_blob(
                    record.v.data(), parameter->data.shape.dims, Device::CPU);
                m_tensor.copy_from(m_host.to(parameter->data.get_device()));
                v_tensor.copy_from(v_host.to(parameter->data.get_device()));
                m_state.emplace(parameter, std::move(m_tensor));
                v_state.emplace(parameter, std::move(v_tensor));
            }
        }
        if (record.has_criticality) {
            crit_g0_state.emplace(parameter, record.criticality);
        }
        if (record.has_external_lr_scale) {
            external_lr_scale.emplace(parameter, record.external_lr_scale);
        }
        if (record.has_criticality_lr_scale) {
            criticality_lr_scale.emplace(
                parameter, record.criticality_lr_scale);
        }
    }
    model->set_training_rng_sequence(rng_sequence);
}

void Trainer::configure_progressive_qat(const TrainPhaseScheduler& scheduler) {
    phase_scheduler = scheduler;
    apply_progressive_qat_phase(*this);
}

bool Trainer::progressive_qat_active() const {
    const auto effective = resolve_effective_qat_schedule(*this);
    return phase_scheduler.progressive_qat_enabled &&
           global_step_count >= effective.qat_start_step;
}

void Trainer::set_lr_scale_by_name(const std::string& name, float scale) {
    if (!model) throw std::runtime_error("Trainer has no model");
    if (name.empty() || !std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument(
            "Parameter LR scale requires a non-empty name and finite scale > 0");
    }
    bool matched = false;
    for (auto* p : model->parameters()) {
        if (p && p->name == name) {
            external_lr_scale[p] = scale;
            matched = true;
        }
    }
    if (!matched) {
        throw std::invalid_argument("Unknown parameter LR-scale name: " + name);
    }
}

float Trainer::accumulate_gradients(const std::vector<int>& tokens,
                                    const std::vector<int>& targets) {
    if (!model) throw std::runtime_error("Trainer requires model");
    model->set_training_mode(true);

    const std::vector<int> inputs = make_inputs(tokens, targets);
    const std::vector<int> resolved_targets = make_targets(tokens, targets);

    auto params = model->parameters();
    apply_progressive_qat_phase(*this);
    zero_model_gradients(params);
    begin_moe_aux_accumulation(*this);

    model->reset_session();
    Context ctx;
    Tensor logits = model->forward_ids(inputs, &ctx);
    // SSA learned block-selector distillation (no-op unless sparse attention is on).
    const float selector_loss = model->accumulate_sparse_selector_grads(1.0f);
    auto [supervised_loss, grad] = logits.cross_entropy(resolved_targets);

    // ── Pantheon VIB-style L2 regularizer on logits ──────────────────────
    // When logit_l2_beta > 0, add beta * 0.5 * mean(logits^2) to the
    // loss and the corresponding gradient term (beta * logits / N) to the
    // grad tensor before backward.  This is a degenerate VIB compression
    // (variational layer not needed); pulls logits toward zero while CE
    // still pulls them toward correct targets.  See PANTHEON_VALIDATION_REPORT.
    const float logit_l2_loss = add_logit_l2_objective(
        logits, effective_logit_l2_beta(*this), grad);
    model->backward_external(grad, ctx);
    const float qat_loss = apply_qat_regularization(*this, 1);
    const float moe_aux_loss = apply_moe_aux_regularization(*this, 1);
    last_objective_stats = TrainingObjectiveStats{};
    last_objective_stats.supervised_cross_entropy = supervised_loss;
    last_objective_stats.logit_l2 = logit_l2_loss;
    last_objective_stats.sparse_selector = selector_loss;
    last_objective_stats.qat_regularization = qat_loss;
    last_objective_stats.moe_auxiliary = moe_aux_loss;
    last_objective_stats.total = supervised_loss + logit_l2_loss +
                                 selector_loss + qat_loss + moe_aux_loss;
    return last_objective_stats.total;
}

float Trainer::train_step(const std::vector<int>& tokens,
                          const std::vector<int>& targets) {
    // Gradients only (no weight update), then the optimizer step + audit.
    // Factored so the criticality instrument can read gradients without
    // mutating weights (Trainer::accumulate_gradients).  Behavior identical
    // to the previous monolithic train_step.
    (void)accumulate_gradients(tokens, targets);
    auto params = model->parameters();
    float grad_norm = 0.0f;
    const float criticality_loss =
        apply_optimizer_step(*this, params, 1, &grad_norm);
    last_objective_stats.criticality_regularization = criticality_loss;
    last_objective_stats.total += criticality_loss;
    record_training_audit_step(*this, last_objective_stats.total, grad_norm,
                               params.size());
    return last_objective_stats.total;
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
            begin_moe_aux_accumulation(*this);
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

            double aggregate_supervised_loss = 0.0;
            double aggregate_repetition_loss = 0.0;
            double aggregate_logit_l2_loss = 0.0;
            double aggregate_selector_loss = 0.0;
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
                std::vector<std::vector<int>> chunk_target_batch;
                chunk_target_batch.reserve(static_cast<size_t>(chunk_samples));
                for (int sample = 0; sample < chunk_samples; ++sample) {
                    const auto begin =
                        chunk_targets.begin() + static_cast<size_t>(sample * seq_len);
                    chunk_target_batch.emplace_back(begin, begin + seq_len);
                }

                model->reset_session();
                Context ctx;
                Tensor logits = model->forward_ids_batch(chunk_inputs, &ctx);
                // SSA learned block-selector distillation (no-op unless sparse on).
                const float chunk_weight =
                    static_cast<float>(chunk_samples) /
                    static_cast<float>(std::max(samples, 1));
                const float selector_loss =
                    model->accumulate_sparse_selector_grads(chunk_weight);
                auto [supervised_loss, grad] = logits.cross_entropy(chunk_targets);
                const float repetition_loss = apply_repetition_unlikelihood_batch(
                    *this, chunk_target_batch, logits, grad);
                const float logit_l2_loss = add_logit_l2_objective(
                    logits, effective_logit_l2_beta(*this), grad);
                // cross_entropy returns a mean over the chunk's token rows.
                // Weight each chunk by its sample fraction so accumulated
                // gradients equal the full-batch mean regardless of chunking.
                scale_tensor_inplace(grad, chunk_weight);
                model->backward_external(grad, ctx);

                aggregate_supervised_loss +=
                    static_cast<double>(supervised_loss) * chunk_samples;
                aggregate_repetition_loss +=
                    static_cast<double>(repetition_loss) * chunk_samples;
                aggregate_logit_l2_loss +=
                    static_cast<double>(logit_l2_loss) * chunk_samples;
                aggregate_selector_loss +=
                    static_cast<double>(selector_loss) * chunk_samples;
                aggregate_samples += chunk_samples;
            }

            const int objective_samples = std::max(aggregate_samples, 1);
            const float qat_loss = apply_qat_regularization(*this, 1);
            const float moe_aux_loss = apply_moe_aux_regularization(*this, 1);
            last_objective_stats = TrainingObjectiveStats{};
            last_objective_stats.supervised_cross_entropy = static_cast<float>(
                aggregate_supervised_loss / objective_samples);
            last_objective_stats.repetition_unlikelihood = static_cast<float>(
                aggregate_repetition_loss / objective_samples);
            last_objective_stats.logit_l2 = static_cast<float>(
                aggregate_logit_l2_loss / objective_samples);
            last_objective_stats.sparse_selector = static_cast<float>(
                aggregate_selector_loss / objective_samples);
            last_objective_stats.qat_regularization = qat_loss;
            last_objective_stats.moe_auxiliary = moe_aux_loss;
            last_objective_stats.total =
                last_objective_stats.supervised_cross_entropy +
                last_objective_stats.repetition_unlikelihood +
                last_objective_stats.logit_l2 +
                last_objective_stats.sparse_selector +
                last_objective_stats.qat_regularization +
                last_objective_stats.moe_auxiliary;
            float grad_norm = 0.0f;
            const float criticality_loss =
                apply_optimizer_step(*this, params, 1, &grad_norm);
            last_objective_stats.criticality_regularization = criticality_loss;
            last_objective_stats.total += criticality_loss;
            const float mean_loss = last_objective_stats.total;
            record_training_audit_step(*this, mean_loss, grad_norm,
                                       params.size());

            ++internal_global_step;
            if (callback) {
                callback(internal_global_step, mean_loss);
            }

            if (max_steps > 0 && internal_global_step >= max_steps) return;
        }
    }
}

} // namespace nsos
