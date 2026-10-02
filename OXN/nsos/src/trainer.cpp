#include "../include/trainer.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include "../include/cuda/device_buffer.h"
#include "../include/cuda/pinned_buffer.h"
#include "../include/cuda/sparse_optimizer_activity.cuh"
#include "../include/layer_audit.h"
#include "../include/nsos_serializer.h"
#include "../include/nsos/determinism.h"  // K4: ordered reductions under NSOS_DETERMINISTIC
#include "../include/optimizer_runtime_policy.h"
#include "../include/muon_math.h"
#include "../include/training_runtime_policy.h"
#include "../include/gpu_attention_training.h"
#include "../include/checkpoint_io.h"
#include <algorithm>
#include <array>
#include <atomic>
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
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

bool trainer_step_timing_enabled() {
    const char* env = std::getenv("NSOS_TRAIN_TIMING");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

#ifdef USE_CUDA
#include "../include/gpu_backend.h"
#endif

namespace nsos {

namespace {

constexpr uint32_t kTrainingStateMagic = 0x4E535452u;  // NSTR
constexpr uint32_t kTrainingStateVersion = 11u;
constexpr uint32_t kTrainingStateLegacyVersion = 1u;
constexpr uint32_t kTrainingStateTrailerMagic = 0x3553544Eu;  // NTS5
constexpr uint32_t kTrainingStateSha256TrailerMagic =
    0x3953544Eu;  // NTS9
constexpr uint64_t kTrainingStateLegacyTrailerBytes =
    sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t);
constexpr uint64_t kTrainingStateSha256TrailerBytes =
    sizeof(uint32_t) + sizeof(uint64_t) + 64u;
constexpr uint32_t kMaxTrainingMemoryStores = 4096u;
constexpr uint32_t kMaxTrainingMemoryClusters = 4096u;
constexpr uint32_t kMaxTrainingMemoryItemsPerCluster = 1024u;
constexpr uint64_t kMaxTrainingMemoryItems = 16384u;
constexpr uint64_t kMaxTrainingMemoryCompressedBytes =
    64ull * 1024ull * 1024ull;

size_t checked_size_multiply(size_t lhs, size_t rhs,
                             const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

size_t checked_size_add(size_t lhs, size_t rhs,
                        const char* message) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        throw std::overflow_error(message);
    }
    return lhs + rhs;
}

size_t checked_size_align(size_t value, size_t alignment,
                          const char* message) {
    if (alignment == 0 || (alignment & (alignment - 1u)) != 0u) {
        throw std::logic_error("Internal metadata alignment is not a power of two");
    }
    const size_t padded =
        checked_size_add(value, alignment - 1u, message);
    return padded & ~(alignment - 1u);
}

#ifdef NSOS_ENABLE_TEST_HOOKS
std::atomic<long long>
    training_state_stage_failure_countdown{-1};
std::atomic<bool>
    training_state_save_failure_before_replace{false};
std::atomic<bool>
    training_nan_before_optimizer{false};
std::atomic<long long>
    optimizer_commit_failure_countdown{-1};

void training_state_stage_fault_point() {
    long long remaining =
        training_state_stage_failure_countdown.load(
            std::memory_order_relaxed);
    while (remaining >= 0) {
        if (remaining == 0) {
            throw std::bad_alloc();
        }
        if (training_state_stage_failure_countdown
                .compare_exchange_weak(
                    remaining, remaining - 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
            return;
        }
    }
}

void training_state_save_fault_point() {
    if (training_state_save_failure_before_replace.load(
            std::memory_order_relaxed)) {
        throw std::runtime_error(
            "Injected training-state save interruption");
    }
}

void training_nan_fault_point(
    const std::vector<Parameter*>& parameters) {
    if (!training_nan_before_optimizer.exchange(
            false, std::memory_order_relaxed)) {
        return;
    }
    for (Parameter* parameter : parameters) {
        if (!parameter || parameter->has_device_gradient_activity() || !parameter->has_gradient()) {
            continue;
        }
        const Device device = parameter->grad.get_device();
        Tensor poisoned = parameter->grad.cpu();
        poisoned.data()[0] =
            std::numeric_limits<float>::quiet_NaN();
        parameter->grad =
            device == Device::CPU
                ? std::move(poisoned)
                : poisoned.to(device);
        return;
    }
    throw std::runtime_error(
        "Injected training NaN found no optimizer gradient");
}

void optimizer_commit_fault_point() {
    long long remaining =
        optimizer_commit_failure_countdown.load(
            std::memory_order_relaxed);
    while (remaining >= 0) {
        if (remaining == 0) {
            optimizer_commit_failure_countdown.store(
                -1, std::memory_order_relaxed);
            throw std::runtime_error(
                "Injected optimizer commit interruption");
        }
        if (optimizer_commit_failure_countdown
                .compare_exchange_weak(
                    remaining, remaining - 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
            return;
        }
    }
}
#else
void training_state_stage_fault_point() noexcept {}
void training_state_save_fault_point() noexcept {}
void training_nan_fault_point(
    const std::vector<Parameter*>&) noexcept {}
void optimizer_commit_fault_point() noexcept {}
#endif

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
    if (value.size() > 32768) {
        throw std::runtime_error("Training-state string is too long");
    }
    write_training_pod(output, static_cast<uint32_t>(value.size()),
                       "string length");
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) throw std::runtime_error("Training-state string write failed");
}

std::string read_training_string(std::istream& input) {
    const uint32_t length = read_training_pod<uint32_t>(input, "string length");
    if (length > 32768) {
        throw std::runtime_error("Training-state string exceeds limit");
    }
    std::string value(length, '\0');
    if (length > 0) {
        input.read(value.data(), static_cast<std::streamsize>(length));
        if (!input) throw std::runtime_error("Training-state truncated in string");
    }
    return value;
}

void write_training_sha256(std::ostream& output,
                           const std::string& digest,
                           const char* label) {
    if (digest.size() != 64 ||
        !std::all_of(
            digest.begin(), digest.end(),
            [](unsigned char value) {
                return (value >= '0' && value <= '9') ||
                       (value >= 'a' && value <= 'f');
            })) {
        throw std::logic_error(
            std::string("Invalid internal SHA-256 digest for ") + label);
    }
    output.write(digest.data(),
                 static_cast<std::streamsize>(digest.size()));
    if (!output) {
        throw std::runtime_error(
            std::string("Training-state SHA-256 write failed: ") + label);
    }
}

std::string read_training_sha256(std::istream& input,
                                 const char* label) {
    std::string digest(64, '\0');
    input.read(digest.data(),
               static_cast<std::streamsize>(digest.size()));
    if (!input ||
        !std::all_of(
            digest.begin(), digest.end(),
            [](unsigned char value) {
                return (value >= '0' && value <= '9') ||
                       (value >= 'a' && value <= 'f');
            })) {
        throw std::runtime_error(
            std::string("Training-state SHA-256 is invalid at ") + label);
    }
    return digest;
}

void write_runtime_execution_identity(
    std::ostream& output,
    const RuntimeExecutionIdentity& identity) {
    if (identity.fields.empty() || identity.fields.size() > 128u) {
        throw std::logic_error(
            "Runtime execution identity has an invalid field count");
    }
    write_training_pod(
        output, static_cast<uint32_t>(identity.fields.size()),
        "runtime identity field count");
    for (const auto& [key, value] : identity.fields) {
        write_training_string(output, key);
        write_training_string(output, value);
    }
    write_training_sha256(
        output, identity.digest(), "runtime execution identity");
}

RuntimeExecutionIdentity read_runtime_execution_identity(
    std::istream& input) {
    const uint32_t count = read_training_pod<uint32_t>(
        input, "runtime identity field count");
    if (count == 0 || count > 128u) {
        throw std::runtime_error(
            "Training-state runtime identity field count is invalid");
    }
    std::vector<RuntimeExecutionIdentity::Field> fields;
    fields.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        // Function-argument evaluation order is not a serialization order:
        // Clang is allowed to read the value before the key here.  Keep the
        // two stream reads explicitly sequenced so a saved (key,value) pair
        // cannot be restored as (value,key), which previously collapsed many
        // boolean values into duplicate keys such as "0".
        std::string key = read_training_string(input);
        std::string value = read_training_string(input);
        fields.emplace_back(std::move(key), std::move(value));
    }
    RuntimeExecutionIdentity identity =
        validated_runtime_execution_identity(std::move(fields));
    const std::string stored_digest = read_training_sha256(
        input, "runtime execution identity");
    if (identity.digest() != stored_digest) {
        throw std::runtime_error(
            "Training-state runtime execution identity digest mismatch");
    }
    return identity;
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

uint64_t training_state_prefix_hash(const std::filesystem::path& path,
                                    uint64_t byte_limit) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Cannot hash training-state payload: " + path.string());
    }
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    uint64_t consumed = 0;
    std::array<char, 1 << 16> buffer{};
    while (consumed < byte_limit) {
        const uint64_t remaining = byte_limit - consumed;
        const std::streamsize wanted = static_cast<std::streamsize>(
            std::min<uint64_t>(remaining, buffer.size()));
        input.read(buffer.data(), wanted);
        const std::streamsize count = input.gcount();
        if (count != wanted) {
            throw std::runtime_error(
                "Training-state payload is truncated while hashing");
        }
        consumed += static_cast<uint64_t>(count);
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(
                buffer[static_cast<size_t>(index)]);
            hash *= kPrime;
        }
    }
    return hash;
}

void verify_training_state_integrity(
    const std::filesystem::path& path, uint32_t version) {
    if (version < 5u) {
        return;
    }
    const bool sha256_trailer = version >= 9u;
    const uint64_t trailer_bytes =
        sha256_trailer ? kTrainingStateSha256TrailerBytes
                       : kTrainingStateLegacyTrailerBytes;
    const uint64_t total_bytes = std::filesystem::file_size(path);
    if (total_bytes < trailer_bytes) {
        throw std::runtime_error("Training-state integrity trailer is missing");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open training-state integrity trailer");
    }
    input.seekg(
        static_cast<std::streamoff>(total_bytes -
                                    trailer_bytes),
        std::ios::beg);
    const uint32_t magic =
        read_training_pod<uint32_t>(input, "integrity magic");
    const uint64_t payload_bytes =
        read_training_pod<uint64_t>(input, "integrity payload bytes");
    const bool valid =
        sha256_trailer
            ? magic == kTrainingStateSha256TrailerMagic &&
                  payload_bytes == total_bytes - trailer_bytes &&
                  integrity::sha256_file_prefix(path, payload_bytes) ==
                      read_training_sha256(input, "integrity SHA-256")
            : magic == kTrainingStateTrailerMagic &&
                  payload_bytes == total_bytes - trailer_bytes &&
                  training_state_prefix_hash(path, payload_bytes) ==
                      read_training_pod<uint64_t>(
                          input, "integrity hash");
    if (!valid) {
        throw std::runtime_error(
            "Training-state checksum validation failed");
    }
}

bool is_fixed_parameter(const Parameter* parameter) {
    return parameter != nullptr && !parameter->trainable;
}

std::vector<Parameter*> trainable_model_parameters(JambaModel* model) {
    std::vector<Parameter*> result;
    for (Parameter* parameter : model->parameters()) {
        if (parameter && !is_fixed_parameter(parameter)) {
            result.push_back(parameter);
        }
    }
    return result;
}

std::vector<std::pair<Parameter*, std::string>> stable_training_parameters(
    JambaModel* model, bool include_legacy_fixed = false) {
    std::vector<std::pair<Parameter*, std::string>> result;
    std::unordered_map<std::string, size_t> counts;
    for (Parameter* parameter : model->parameters()) {
        if (!parameter ||
            (!include_legacy_fixed &&
             is_fixed_parameter(parameter))) {
            continue;
        }
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
    bool dynamic_loss_scaling_enabled = true;
    float loss_scale = 1024.0f;
    float min_loss_scale = 1.0f;
    float max_loss_scale = 65536.0f;
    float loss_scale_growth_factor = 2.0f;
    float loss_scale_backoff_factor = 0.5f;
    int32_t loss_scale_growth_interval = 2000;
    int32_t loss_scale_growth_tracker = 0;
    TrainPhaseScheduler phase_scheduler;
};

// v11 appends an explicit progress record to the existing payload. Never
// serialize C++ struct padding or infer historical token counts from steps.
constexpr uint32_t kTrainingProgressMagic = 0x31525054u; // TPR1
struct TrainingProgressState {
    Trainer::SchedulerUnit scheduler_unit = Trainer::SchedulerUnit::Steps;
    int64_t warmup_tokens = 0;
    int64_t training_tokens = 0;
    int64_t decay_tokens = 0;
    int64_t tokens_processed = 0;
    int64_t tokens_committed = 0;
    bool token_counters_complete = true;
    int32_t gradient_accumulation_steps = 1;
    float last_grad_norm_pre_clip = 0;
    float last_grad_norm_post_clip = 0;
    bool last_update_was_clipped = false;
    int32_t last_accumulation_steps = 1;
    bool last_optimizer_step_skipped = false;
    AuxiliaryStackStats last_auxiliary_stats;
    TrainingObjectiveStats last_objective_stats;
    TrainingStepTelemetry last_step_telemetry;
};

void require_checkpoint_boundary(const Trainer& trainer) {
    if (trainer.pending_accumulation_microbatches != 0 ||
        trainer.pending_accumulated_tokens != 0 ||
        trainer.pending_supervised_loss_sum.size != 0 ||
        trainer.device_sparse_group_open || !trainer.device_moe_groups.empty())
        throw std::logic_error("Training checkpoint/clone requires a closed accumulation boundary");
}

TrainingProgressState capture_training_progress(const Trainer& trainer) {
    TrainingProgressState state;
    state.scheduler_unit = trainer.scheduler_unit;
    state.warmup_tokens = trainer.warmup_tokens;
    state.training_tokens = trainer.training_tokens;
    state.decay_tokens = trainer.decay_tokens;
    state.tokens_processed = trainer.tokens_processed;
    state.tokens_committed = trainer.tokens_committed;
    state.token_counters_complete = trainer.token_counters_complete;
    state.gradient_accumulation_steps = trainer.gradient_accumulation_steps;
    state.last_grad_norm_pre_clip = trainer.last_grad_norm_pre_clip;
    state.last_grad_norm_post_clip = trainer.last_grad_norm_post_clip;
    state.last_update_was_clipped = trainer.last_update_was_clipped;
    state.last_accumulation_steps = trainer.last_accumulation_steps;
    state.last_optimizer_step_skipped = trainer.last_optimizer_step_skipped;
    state.last_auxiliary_stats = trainer.last_auxiliary_stats;
    state.last_objective_stats = trainer.last_objective_stats;
    state.last_step_telemetry = trainer.last_step_telemetry;
    return state;
}

void publish_training_progress(Trainer& trainer, const TrainingProgressState& state) noexcept {
    trainer.scheduler_unit = state.scheduler_unit;
    trainer.warmup_tokens = state.warmup_tokens;
    trainer.training_tokens = state.training_tokens;
    trainer.decay_tokens = state.decay_tokens;
    trainer.tokens_processed = state.tokens_processed;
    trainer.tokens_committed = state.tokens_committed;
    trainer.token_counters_complete = state.token_counters_complete;
    trainer.gradient_accumulation_steps = state.gradient_accumulation_steps;
    trainer.last_grad_norm_pre_clip = state.last_grad_norm_pre_clip;
    trainer.last_grad_norm_post_clip = state.last_grad_norm_post_clip;
    trainer.last_update_was_clipped = state.last_update_was_clipped;
    trainer.last_accumulation_steps = state.last_accumulation_steps;
    trainer.last_optimizer_step_skipped = state.last_optimizer_step_skipped;
    trainer.last_auxiliary_stats = state.last_auxiliary_stats;
    trainer.last_objective_stats = state.last_objective_stats;
    trainer.last_step_telemetry = state.last_step_telemetry;
}

void validate_training_progress(const TrainingProgressState& state, int global_step) {
    const auto unit = static_cast<uint32_t>(state.scheduler_unit);
    const auto limit = std::numeric_limits<int64_t>::max();
    if (unit > 1 || state.gradient_accumulation_steps <= 0 || state.last_accumulation_steps <= 0 ||
        state.warmup_tokens < 0 || state.warmup_tokens == limit ||
        state.training_tokens < 0 || state.training_tokens == limit || state.decay_tokens < 0 ||
        state.decay_tokens > state.training_tokens || state.tokens_processed < 0 ||
        state.tokens_processed == limit || state.tokens_committed < 0 ||
        state.tokens_committed > state.tokens_processed ||
        (state.scheduler_unit == Trainer::SchedulerUnit::Tokens &&
         (!state.token_counters_complete || state.training_tokens <= state.warmup_tokens ||
          state.decay_tokens > state.training_tokens - state.warmup_tokens)))
        throw std::runtime_error("Invalid training progress scheduler/accumulation/token metadata");
    for (float value : {state.last_grad_norm_pre_clip, state.last_grad_norm_post_clip})
        if (std::isnan(value) || value < 0 || (!state.last_optimizer_step_skipped && !std::isfinite(value)))
            throw std::runtime_error("Invalid training progress clipping telemetry");
    if (state.last_auxiliary_stats.bucket_count < 0) throw std::runtime_error("Invalid auxiliary counter: bucket_count");
    if (state.last_auxiliary_stats.due_count < 0) throw std::runtime_error("Invalid auxiliary counter: due_count");
    if (state.last_auxiliary_stats.applied_count < 0) throw std::runtime_error("Invalid auxiliary counter: applied_count");
    if (state.last_auxiliary_stats.reasoning_count < 0) throw std::runtime_error("Invalid auxiliary counter: reasoning_count");
    if (state.last_auxiliary_stats.memory_count < 0) throw std::runtime_error("Invalid auxiliary counter: memory_count");
    if (state.last_auxiliary_stats.session_adapt_count < 0) throw std::runtime_error("Invalid auxiliary counter: session_adapt_count");
    if (state.last_auxiliary_stats.sample_count < 0) throw std::runtime_error("Invalid auxiliary counter: sample_count");
    if (state.last_auxiliary_stats.prompt_tokens < 0) throw std::runtime_error("Invalid auxiliary counter: prompt_tokens");
    if (state.last_auxiliary_stats.answer_tokens < 0) throw std::runtime_error("Invalid auxiliary counter: answer_tokens");
    if (!std::isfinite(state.last_auxiliary_stats.prompt_state_norm)) throw std::runtime_error("Invalid auxiliary telemetry: prompt_state_norm");
    if (!std::isfinite(state.last_auxiliary_stats.target_state_norm)) throw std::runtime_error("Invalid auxiliary telemetry: target_state_norm");
    if (!std::isfinite(state.last_auxiliary_stats.reason_delta_norm)) throw std::runtime_error("Invalid auxiliary telemetry: reason_delta_norm");
    if (!std::isfinite(state.last_auxiliary_stats.reason_cosine)) throw std::runtime_error("Invalid auxiliary telemetry: reason_cosine");
    if (!std::isfinite(state.last_auxiliary_stats.memory_delta_norm)) throw std::runtime_error("Invalid auxiliary telemetry: memory_delta_norm");
    if (!std::isfinite(state.last_auxiliary_stats.memory_cosine)) throw std::runtime_error("Invalid auxiliary telemetry: memory_cosine");
    if (!std::isfinite(state.last_auxiliary_stats.final_target_delta_norm)) throw std::runtime_error("Invalid auxiliary telemetry: final_target_delta_norm");
    if (!std::isfinite(state.last_objective_stats.supervised_cross_entropy)) throw std::runtime_error("Invalid objective telemetry: supervised_cross_entropy");
    if (!std::isfinite(state.last_objective_stats.repetition_unlikelihood)) throw std::runtime_error("Invalid objective telemetry: repetition_unlikelihood");
    if (!std::isfinite(state.last_objective_stats.logit_l2)) throw std::runtime_error("Invalid objective telemetry: logit_l2");
    if (!std::isfinite(state.last_objective_stats.sparse_selector)) throw std::runtime_error("Invalid objective telemetry: sparse_selector");
    if (!std::isfinite(state.last_objective_stats.qat_regularization)) throw std::runtime_error("Invalid objective telemetry: qat_regularization");
    if (!std::isfinite(state.last_objective_stats.moe_auxiliary)) throw std::runtime_error("Invalid objective telemetry: moe_auxiliary");
    if (!std::isfinite(state.last_objective_stats.criticality_regularization)) throw std::runtime_error("Invalid objective telemetry: criticality_regularization");
    if (!std::isfinite(state.last_objective_stats.total)) throw std::runtime_error("Invalid objective telemetry: total");
    if (state.last_step_telemetry.global_step < 0 || state.last_step_telemetry.global_step > global_step ||
        state.last_step_telemetry.bucket_count < 0)
        throw std::runtime_error("Invalid step telemetry counters");
    if (!std::isfinite(state.last_step_telemetry.wall_ms) || state.last_step_telemetry.wall_ms < 0) throw std::runtime_error("Invalid timing telemetry: wall_ms");
    if (!std::isfinite(state.last_step_telemetry.preparation_ms) || state.last_step_telemetry.preparation_ms < 0) throw std::runtime_error("Invalid timing telemetry: preparation_ms");
    if (!std::isfinite(state.last_step_telemetry.inter_bucket_ms) || state.last_step_telemetry.inter_bucket_ms < 0) throw std::runtime_error("Invalid timing telemetry: inter_bucket_ms");
    if (!std::isfinite(state.last_step_telemetry.forward_ms) || state.last_step_telemetry.forward_ms < 0) throw std::runtime_error("Invalid timing telemetry: forward_ms");
    if (!std::isfinite(state.last_step_telemetry.loss_ms) || state.last_step_telemetry.loss_ms < 0) throw std::runtime_error("Invalid timing telemetry: loss_ms");
    if (!std::isfinite(state.last_step_telemetry.backward_ms) || state.last_step_telemetry.backward_ms < 0) throw std::runtime_error("Invalid timing telemetry: backward_ms");
    if (!std::isfinite(state.last_step_telemetry.optimizer_ms) || state.last_step_telemetry.optimizer_ms < 0) throw std::runtime_error("Invalid timing telemetry: optimizer_ms");
    if (!std::isfinite(state.last_step_telemetry.unaccounted_ms)) throw std::runtime_error("Invalid timing telemetry: unaccounted_ms");
}

void write_training_progress(std::ostream& output, const TrainingProgressState& state) {
    write_training_pod(output, kTrainingProgressMagic, "progress magic");
    write_training_pod(output, static_cast<uint32_t>(state.scheduler_unit), "progress scheduler_unit");
    write_training_pod(output, static_cast<int32_t>(state.gradient_accumulation_steps), "progress gradient_accumulation_steps");
    write_training_pod(output, static_cast<int64_t>(state.warmup_tokens), "progress warmup_tokens");
    write_training_pod(output, static_cast<int64_t>(state.training_tokens), "progress training_tokens");
    write_training_pod(output, static_cast<int64_t>(state.decay_tokens), "progress decay_tokens");
    write_training_pod(output, static_cast<int64_t>(state.tokens_processed), "progress tokens_processed");
    write_training_pod(output, static_cast<int64_t>(state.tokens_committed), "progress tokens_committed");
    write_training_pod(output, static_cast<uint8_t>(state.token_counters_complete), "progress token_counters_complete");
    write_training_pod(output, static_cast<float>(state.last_grad_norm_pre_clip), "progress last_grad_norm_pre_clip");
    write_training_pod(output, static_cast<float>(state.last_grad_norm_post_clip), "progress last_grad_norm_post_clip");
    write_training_pod(output, static_cast<uint8_t>(state.last_update_was_clipped), "progress last_update_was_clipped");
    write_training_pod(output, static_cast<int32_t>(state.last_accumulation_steps), "progress last_accumulation_steps");
    write_training_pod(output, static_cast<uint8_t>(state.last_optimizer_step_skipped), "progress last_optimizer_step_skipped");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.bucket_count), "progress last_auxiliary_stats.bucket_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.due_count), "progress last_auxiliary_stats.due_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.applied_count), "progress last_auxiliary_stats.applied_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.reasoning_count), "progress last_auxiliary_stats.reasoning_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.memory_count), "progress last_auxiliary_stats.memory_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.session_adapt_count), "progress last_auxiliary_stats.session_adapt_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.sample_count), "progress last_auxiliary_stats.sample_count");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.prompt_tokens), "progress last_auxiliary_stats.prompt_tokens");
    write_training_pod(output, static_cast<int32_t>(state.last_auxiliary_stats.answer_tokens), "progress last_auxiliary_stats.answer_tokens");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.prompt_state_norm), "progress last_auxiliary_stats.prompt_state_norm");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.target_state_norm), "progress last_auxiliary_stats.target_state_norm");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.reason_delta_norm), "progress last_auxiliary_stats.reason_delta_norm");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.reason_cosine), "progress last_auxiliary_stats.reason_cosine");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.memory_delta_norm), "progress last_auxiliary_stats.memory_delta_norm");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.memory_cosine), "progress last_auxiliary_stats.memory_cosine");
    write_training_pod(output, static_cast<float>(state.last_auxiliary_stats.final_target_delta_norm), "progress last_auxiliary_stats.final_target_delta_norm");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.supervised_cross_entropy), "progress last_objective_stats.supervised_cross_entropy");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.repetition_unlikelihood), "progress last_objective_stats.repetition_unlikelihood");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.logit_l2), "progress last_objective_stats.logit_l2");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.sparse_selector), "progress last_objective_stats.sparse_selector");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.qat_regularization), "progress last_objective_stats.qat_regularization");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.moe_auxiliary), "progress last_objective_stats.moe_auxiliary");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.criticality_regularization), "progress last_objective_stats.criticality_regularization");
    write_training_pod(output, static_cast<float>(state.last_objective_stats.total), "progress last_objective_stats.total");
    write_training_pod(output, static_cast<uint8_t>(state.last_step_telemetry.enabled), "progress last_step_telemetry.enabled");
    write_training_pod(output, static_cast<int32_t>(state.last_step_telemetry.global_step), "progress last_step_telemetry.global_step");
    write_training_pod(output, static_cast<int32_t>(state.last_step_telemetry.bucket_count), "progress last_step_telemetry.bucket_count");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.wall_ms), "progress last_step_telemetry.wall_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.preparation_ms), "progress last_step_telemetry.preparation_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.inter_bucket_ms), "progress last_step_telemetry.inter_bucket_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.forward_ms), "progress last_step_telemetry.forward_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.loss_ms), "progress last_step_telemetry.loss_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.backward_ms), "progress last_step_telemetry.backward_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.optimizer_ms), "progress last_step_telemetry.optimizer_ms");
    write_training_pod(output, static_cast<double>(state.last_step_telemetry.unaccounted_ms), "progress last_step_telemetry.unaccounted_ms");
}

TrainingProgressState read_training_progress(std::istream& input) {
    if (read_training_pod<uint32_t>(input, "progress magic") != kTrainingProgressMagic)
        throw std::runtime_error("Invalid training progress record magic");
    TrainingProgressState state;
    state.scheduler_unit = static_cast<Trainer::SchedulerUnit>(read_training_pod<uint32_t>(input, "progress scheduler unit"));
    state.gradient_accumulation_steps = read_training_pod<int32_t>(input, "progress gradient_accumulation_steps");
    state.warmup_tokens = read_training_pod<int64_t>(input, "progress warmup_tokens");
    state.training_tokens = read_training_pod<int64_t>(input, "progress training_tokens");
    state.decay_tokens = read_training_pod<int64_t>(input, "progress decay_tokens");
    state.tokens_processed = read_training_pod<int64_t>(input, "progress tokens_processed");
    state.tokens_committed = read_training_pod<int64_t>(input, "progress tokens_committed");
    { const auto value = read_training_pod<uint8_t>(input, "progress token_counters_complete");
      if (value > 1) throw std::runtime_error("Invalid training progress boolean: token_counters_complete");
      state.token_counters_complete = value != 0; }
    state.last_grad_norm_pre_clip = read_training_pod<float>(input, "progress last_grad_norm_pre_clip");
    state.last_grad_norm_post_clip = read_training_pod<float>(input, "progress last_grad_norm_post_clip");
    { const auto value = read_training_pod<uint8_t>(input, "progress last_update_was_clipped");
      if (value > 1) throw std::runtime_error("Invalid training progress boolean: last_update_was_clipped");
      state.last_update_was_clipped = value != 0; }
    state.last_accumulation_steps = read_training_pod<int32_t>(input, "progress last_accumulation_steps");
    { const auto value = read_training_pod<uint8_t>(input, "progress last_optimizer_step_skipped");
      if (value > 1) throw std::runtime_error("Invalid training progress boolean: last_optimizer_step_skipped");
      state.last_optimizer_step_skipped = value != 0; }
    state.last_auxiliary_stats.bucket_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.bucket_count");
    state.last_auxiliary_stats.due_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.due_count");
    state.last_auxiliary_stats.applied_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.applied_count");
    state.last_auxiliary_stats.reasoning_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.reasoning_count");
    state.last_auxiliary_stats.memory_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.memory_count");
    state.last_auxiliary_stats.session_adapt_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.session_adapt_count");
    state.last_auxiliary_stats.sample_count = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.sample_count");
    state.last_auxiliary_stats.prompt_tokens = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.prompt_tokens");
    state.last_auxiliary_stats.answer_tokens = read_training_pod<int32_t>(input, "progress last_auxiliary_stats.answer_tokens");
    state.last_auxiliary_stats.prompt_state_norm = read_training_pod<float>(input, "progress last_auxiliary_stats.prompt_state_norm");
    state.last_auxiliary_stats.target_state_norm = read_training_pod<float>(input, "progress last_auxiliary_stats.target_state_norm");
    state.last_auxiliary_stats.reason_delta_norm = read_training_pod<float>(input, "progress last_auxiliary_stats.reason_delta_norm");
    state.last_auxiliary_stats.reason_cosine = read_training_pod<float>(input, "progress last_auxiliary_stats.reason_cosine");
    state.last_auxiliary_stats.memory_delta_norm = read_training_pod<float>(input, "progress last_auxiliary_stats.memory_delta_norm");
    state.last_auxiliary_stats.memory_cosine = read_training_pod<float>(input, "progress last_auxiliary_stats.memory_cosine");
    state.last_auxiliary_stats.final_target_delta_norm = read_training_pod<float>(input, "progress last_auxiliary_stats.final_target_delta_norm");
    state.last_objective_stats.supervised_cross_entropy = read_training_pod<float>(input, "progress last_objective_stats.supervised_cross_entropy");
    state.last_objective_stats.repetition_unlikelihood = read_training_pod<float>(input, "progress last_objective_stats.repetition_unlikelihood");
    state.last_objective_stats.logit_l2 = read_training_pod<float>(input, "progress last_objective_stats.logit_l2");
    state.last_objective_stats.sparse_selector = read_training_pod<float>(input, "progress last_objective_stats.sparse_selector");
    state.last_objective_stats.qat_regularization = read_training_pod<float>(input, "progress last_objective_stats.qat_regularization");
    state.last_objective_stats.moe_auxiliary = read_training_pod<float>(input, "progress last_objective_stats.moe_auxiliary");
    state.last_objective_stats.criticality_regularization = read_training_pod<float>(input, "progress last_objective_stats.criticality_regularization");
    state.last_objective_stats.total = read_training_pod<float>(input, "progress last_objective_stats.total");
    { const auto value = read_training_pod<uint8_t>(input, "progress last_step_telemetry.enabled");
      if (value > 1) throw std::runtime_error("Invalid training progress boolean: last_step_telemetry.enabled");
      state.last_step_telemetry.enabled = value != 0; }
    state.last_step_telemetry.global_step = read_training_pod<int32_t>(input, "progress last_step_telemetry.global_step");
    state.last_step_telemetry.bucket_count = read_training_pod<int32_t>(input, "progress last_step_telemetry.bucket_count");
    state.last_step_telemetry.wall_ms = read_training_pod<double>(input, "progress last_step_telemetry.wall_ms");
    state.last_step_telemetry.preparation_ms = read_training_pod<double>(input, "progress last_step_telemetry.preparation_ms");
    state.last_step_telemetry.inter_bucket_ms = read_training_pod<double>(input, "progress last_step_telemetry.inter_bucket_ms");
    state.last_step_telemetry.forward_ms = read_training_pod<double>(input, "progress last_step_telemetry.forward_ms");
    state.last_step_telemetry.loss_ms = read_training_pod<double>(input, "progress last_step_telemetry.loss_ms");
    state.last_step_telemetry.backward_ms = read_training_pod<double>(input, "progress last_step_telemetry.backward_ms");
    state.last_step_telemetry.optimizer_ms = read_training_pod<double>(input, "progress last_step_telemetry.optimizer_ms");
    state.last_step_telemetry.unaccounted_ms = read_training_pod<double>(input, "progress last_step_telemetry.unaccounted_ms");
    return state;
}

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

struct TrainingMemoryRecord {
    int64_t key = 0;
    int32_t dimension = 0;
    std::vector<MemorySystem::Cluster> clusters;
};

void write_training_memory_tensor(std::ostream& output,
                                  const Tensor& tensor,
                                  int32_t expected_dimension) {
    Tensor host =
        tensor.get_device() == Device::GPU ? tensor.cpu() : tensor;
    if (host.size != expected_dimension || host.shape.size() == 0) {
        throw std::runtime_error(
            "Auxiliary-memory tensor shape mismatch");
    }
    const float* values = host.data();
    for (int32_t index = 0; index < expected_dimension; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error(
                "Auxiliary-memory tensor contains NaN or Inf");
        }
    }
    output.write(
        reinterpret_cast<const char*>(values),
        static_cast<std::streamsize>(
            static_cast<uint64_t>(expected_dimension) * sizeof(float)));
    if (!output) {
        throw std::runtime_error(
            "Training-state auxiliary-memory tensor write failed");
    }
}

Tensor read_training_memory_tensor(std::istream& input,
                                   int32_t dimension) {
    if (dimension <= 0 || dimension > 1'048'576) {
        throw std::runtime_error(
            "Invalid auxiliary-memory tensor dimension");
    }
    Tensor tensor({dimension}, Device::CPU);
    input.read(
        reinterpret_cast<char*>(tensor.data()),
        static_cast<std::streamsize>(
            static_cast<uint64_t>(dimension) * sizeof(float)));
    if (!input) {
        throw std::runtime_error(
            "Training-state truncated in auxiliary-memory tensor");
    }
    for (int32_t index = 0; index < dimension; ++index) {
        if (!std::isfinite(tensor.data()[index])) {
            throw std::runtime_error(
                "Invalid auxiliary-memory tensor value");
        }
    }
    return tensor;
}

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
        record_gpu_device_synchronization();
    }
}

// Opt-in diagnostics must not drain unrelated streams on the whole device.
// Reusing one event also avoids per-boundary event allocation.  Each wait is a
// default-stream fence, which preserves the historical wall-clock stage
// interpretation while remaining valid under both CUDA and HIP.
class DiagnosticStreamFence {
public:
    explicit DiagnosticStreamFence(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            const cudaError_t status = cudaEventCreate(&event_);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("training timing event creation failed on ") +
                    NSOS_GPU_BACKEND_NAME + ": " +
                    cudaGetErrorString(status));
            }
        }
    }

    DiagnosticStreamFence(const DiagnosticStreamFence&) = delete;
    DiagnosticStreamFence& operator=(const DiagnosticStreamFence&) = delete;

    ~DiagnosticStreamFence() {
        if (event_ != nullptr) {
            gpu::report_cleanup_status(
                cudaEventDestroy(event_),
                "training timing event destruction");
        }
    }

    void wait() {
        if (!enabled_) {
            return;
        }
        cudaError_t status = cudaEventRecord(event_, nsos::gpu::current_stream());
        if (status == cudaSuccess) {
            status = cudaEventSynchronize(event_);
        }
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("training timing stream fence failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
        record_gpu_stream_synchronization();
    }

private:
    bool enabled_ = false;
    cudaEvent_t event_ = nullptr;
};

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
        const cudaError_t status = cudaMemset(
            tensor.raw_data(), 0,
            static_cast<size_t>(tensor.size) * sizeof(float));
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemset(zero_tensor_inplace) failed: ") +
                cudaGetErrorString(status));
        }
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

void validate_auxiliary_consumer(const Trainer& trainer) {
    const auto& scheduler = trainer.phase_scheduler;
    if (!scheduler.auxiliary_stack_enabled) return;
    if (!scheduler.auxiliary_session_adapt_enabled || !trainer.model ||
        !trainer.model->model_config().use_ttt ||
        std::none_of(trainer.model->layers.begin(), trainer.model->layers.end(),
                     [](const auto& layer) { return layer && layer->ttt_layer; })) {
        throw std::invalid_argument(
            "Auxiliary stack requires session_adapt and an active TTT consumer; "
            "memory/reasoning-only training would discard the refined state");
    }
    if (scheduler.auxiliary_memory_enabled && scheduler.auxiliary_memory_blend <= 0) {
        throw std::invalid_argument("Auxiliary memory requires a positive consumed blend");
    }
    if (scheduler.auxiliary_reasoning_enabled && !trainer.model->has_reasoning_policy()) {
        throw std::invalid_argument(
            "Auxiliary reasoning requires an explicitly registered correctness-verifier policy");
    }
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
    float* dst = states.raw_data();
    const float* src = trunk.raw_data();
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

class ModelTrainingModeGuard {
public:
    ModelTrainingModeGuard(
        JambaModel* model, bool temporary_mode, Trainer* trainer)
        : model_(model),
          previous_(model ? model->training_mode() : false),
          trainer_(trainer) {
        if (model_) model_->set_training_mode(temporary_mode);
    }
    ModelTrainingModeGuard(const ModelTrainingModeGuard&) = delete;
    ModelTrainingModeGuard& operator=(
        const ModelTrainingModeGuard&) = delete;
    ~ModelTrainingModeGuard() {
        if (!restored_ && model_) {
            try {
                model_->set_training_mode(previous_);
            } catch (...) {
                if (trainer_) trainer_->mark_optimizer_state_poisoned();
            }
        }
    }
    void restore() {
        if (!restored_ && model_) {
            model_->set_training_mode(previous_);
            restored_ = true;
        }
    }

private:
    JambaModel* model_ = nullptr;
    bool previous_ = false;
    Trainer* trainer_ = nullptr;
    bool restored_ = false;
};

Tensor reason_state_batch(Trainer& trainer, const Tensor& states) {
    if (!trainer.model || states.size == 0 || states.shape.size() != 2) {
        return states.clone();
    }

    const int batch = states.shape[0];
    const int dim = states.shape[1];
    Tensor refined({batch, dim}, states.get_device());
    float* refined_ptr = refined.raw_data();
    const float* states_ptr = states.raw_data();
    const int iterations = std::max(trainer.phase_scheduler.auxiliary_reasoning_iterations, 1);
    const int simulations = std::max(trainer.phase_scheduler.auxiliary_reasoning_simulations, 1);

    for (int row = 0; row < batch; ++row) {
        Tensor seed({dim}, states.get_device());
        copy_tensor_bytes(seed.raw_data(),
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
                          current.raw_data(),
                          current.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
    }
    return refined;
}

Tensor recall_memory_batch(Trainer& trainer, const Tensor& states,
                           int memory_scope) {
    if (states.size == 0 || states.shape.size() != 2) {
        return states.clone();
    }

    const int batch = states.shape[0];
    const int dim = states.shape[1];
    MemorySystem& memory =
        trainer.auxiliary_memory_store(dim, memory_scope);
    Tensor recalled({batch, dim}, states.get_device());
    float* recalled_ptr = recalled.raw_data();
    const float* states_ptr = states.raw_data();
    for (int row = 0; row < batch; ++row) {
        Tensor query({dim}, states.get_device());
        copy_tensor_bytes(query.raw_data(),
                          query.get_device(),
                          states_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          states.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
        Tensor value = memory.retrieve(query);
        copy_tensor_bytes(recalled_ptr + static_cast<size_t>(row) * static_cast<size_t>(dim),
                          recalled.get_device(),
                          value.raw_data(),
                          value.get_device(),
                          static_cast<size_t>(dim) * sizeof(float));
    }
    return recalled;
}

void store_memory_batch(Trainer& trainer, const Tensor& states,
                        int memory_scope) {
    if (states.size == 0 || states.shape.size() != 2) {
        return;
    }
    const int batch = states.shape[0];
    const int dim = states.shape[1];
    MemorySystem& memory =
        trainer.auxiliary_memory_store(dim, memory_scope);
    const float* states_ptr = states.raw_data();
    for (int row = 0; row < batch; ++row) {
        Tensor value({dim}, states.get_device());
        copy_tensor_bytes(value.raw_data(),
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
    validate_auxiliary_consumer(trainer);
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
    std::vector<std::vector<int>> trimmed_prompts;
    prompt_lengths.reserve(prompt_batch.size());
    trimmed_prompts.reserve(prompt_batch.size());
    const int prompt_max_tokens = std::max(trainer.phase_scheduler.auxiliary_prompt_max_tokens, 1);
    for (size_t index = 0; index < prompt_batch.size(); ++index) {
        if (prompt_batch[index].empty() || answer_batch[index].empty()) {
            return stats;
        }
        trimmed_prompts.push_back(trim_suffix_tokens(prompt_batch[index], prompt_max_tokens));
        prompt_lengths.push_back(static_cast<int>(trimmed_prompts.back().size()));
        stats.prompt_tokens += static_cast<int>(trimmed_prompts.back().size());
    }

    ModelTrainingModeGuard training_mode_guard(model, false, &trainer);
    Context aux_ctx;
    Tensor prompt_trunk = model->forward_trunk_batch(trimmed_prompts, &aux_ctx);
    Tensor prompt_states = extract_last_token_states(prompt_trunk, prompt_lengths);
    Tensor target_states = prompt_states.clone();
    const Tensor original_target_states = prompt_states.clone();
    stats.applied_count = 1;
    stats.prompt_state_norm = tensor_mean_row_norm(prompt_states);
    // Prompt-only session adaptation. Answers are consumed by masked CE, not
    // by a redundant diagnostic forward or leaked into session fast weights.

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
            recall_memory_batch(trainer, prompt_states,
                                trainer.phase_scheduler.auxiliary_memory_scope);
        stats.memory_cosine = tensor_mean_row_cosine(prompt_states, recalled_states);
        target_states = blend_state_tensors(
            target_states, recalled_states, trainer.phase_scheduler.auxiliary_memory_blend);
        stats.memory_delta_norm = tensor_mean_row_delta_norm(target_before_memory, target_states);
        store_memory_batch(trainer, prompt_states,
                           trainer.phase_scheduler.auxiliary_memory_scope);
    }
    stats.final_target_delta_norm = tensor_mean_row_delta_norm(original_target_states, target_states);
    stats.target_state_norm = tensor_mean_row_norm(target_states);

    training_mode_guard.restore();
    if (trainer.phase_scheduler.auxiliary_session_adapt_enabled && config.use_ttt) {
        stats.session_adapt_count = 1;
        model->session_adapt(prompt_states, target_states);
    }
    return stats;
}

void zero_model_gradients(const std::vector<Parameter*>& params) {
    // Device sparse groups own this reset in begin/finish; legacy groups own
    // it here. Neither path clears sticky status between microbatches.
    if (!optimizer_policy::device_sparse_adam_enabled())
        attention_training::reset_status(params);
#ifdef USE_CUDA
    if (gpu_custom_kernels_supported()) {
        std::vector<float*> pointers;
        std::vector<unsigned long long> offsets{0};
        pointers.reserve(params.size());
        offsets.reserve(params.size() + 1);
        for (auto* param : params) {
            if (!param || param->has_device_gradient_activity() || param->grad.size == 0) continue;
            if (param->grad.get_device() == Device::GPU) {
                pointers.push_back(param->grad.raw_data());
                offsets.push_back(
                    offsets.back() +
                    static_cast<unsigned long long>(param->grad.size));
            } else {
                param->zero_grad();
            }
        }
        if (!pointers.empty()) {
            const size_t pointer_bytes =
                pointers.size() * sizeof(float*);
            const size_t offset_start =
                (pointer_bytes + alignof(unsigned long long) - 1) &
                ~(alignof(unsigned long long) - 1);
            const size_t offset_bytes =
                offsets.size() * sizeof(unsigned long long);
            thread_local std::vector<unsigned char> staging;
            staging.assign(offset_start + offset_bytes, 0);
            std::memcpy(staging.data(), pointers.data(), pointer_bytes);
            std::memcpy(staging.data() + offset_start, offsets.data(),
                        offset_bytes);

            thread_local cuda_detail::DeviceBuffer<unsigned char> metadata;
            unsigned char* device_metadata =
                metadata.ensure(staging.size());
            if (!device_metadata) {
                throw std::runtime_error(
                    std::string("GPU gradient-zero metadata allocation "
                                "failed on ") +
                    NSOS_GPU_BACKEND_NAME);
            }
            thread_local std::vector<unsigned char> uploaded;
            thread_local unsigned char* uploaded_device = nullptr;
            if (uploaded_device != device_metadata ||
                uploaded.size() != staging.size() ||
                !std::equal(staging.begin(), staging.end(),
                            uploaded.begin())) {
                const cudaError_t status =
                    cudaMemcpy(device_metadata, staging.data(),
                               staging.size(), cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("GPU gradient-zero metadata upload "
                                    "failed on ") +
                        NSOS_GPU_BACKEND_NAME + ": " +
                        cudaGetErrorString(status));
                }
                record_gpu_transfer(
                    Device::GPU, Device::CPU, staging.size());
                uploaded = staging;
                uploaded_device = device_metadata;
            }
            auto* device_pointers =
                reinterpret_cast<float* const*>(device_metadata);
            const auto* device_offsets =
                reinterpret_cast<const unsigned long long*>(
                    device_metadata + offset_start);
            if (!launch_multi_tensor_zero(
                device_pointers, device_offsets,
                static_cast<int>(pointers.size()), offsets.back())) {
                throw std::runtime_error(
                    "GPU gradient-zero launcher rejected invalid arguments");
            }
            trainer_check_cuda("launch_multi_tensor_zero(gradients)");
        }
        for (auto* param : params) {
            if (param && !param->has_device_gradient_activity()) param->reset_gradient_activity();
        }
        return;
    }
#endif
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
        if (!p || p->has_device_gradient_activity() || !p->has_gradient()) continue;
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
    thread_local cuda_detail::DeviceBuffer<float> buffer;
    return buffer.ensure(1);
}

struct DeterministicClipWorkspace {
    cuda_detail::DeviceBuffer<float*> gradients;
    cuda_detail::DeviceBuffer<unsigned long long> sizes;
    cuda_detail::DeviceBuffer<double> partials;
    cuda_detail::DeviceBuffer<double> total;
    cuda_detail::DeviceBuffer<float> coefficient;
    cuda_detail::DeviceBuffer<NsosMultiTensorChunk> chunks;
    std::vector<NsosMultiTensorChunk> host_chunks;
    const NsosMultiTensorChunk* uploaded_chunks_device = nullptr;
    std::vector<float*> uploaded_gradients;
    std::vector<unsigned long long> uploaded_sizes;
    float** uploaded_gradient_device = nullptr;
    unsigned long long* uploaded_size_device = nullptr;
};

DeterministicClipWorkspace& deterministic_clip_workspace() {
    // A training thread owns one complete reduction cohort. This avoids both
    // per-step cudaMalloc churn and cross-replica sharing of mutable metadata.
    thread_local DeterministicClipWorkspace workspace;
    return workspace;
}

int* finite_issue_accumulator() {
    thread_local cuda_detail::DeviceBuffer<int> buffer;
    return buffer.ensure(1);
}

unsigned char* finite_metadata_buffer(size_t bytes) {
    thread_local cuda_detail::DeviceBuffer<unsigned char> buffer;
    return buffer.ensure(bytes);
}
#endif

bool optimizer_inputs_are_finite(
    const std::vector<Parameter*>& params,
    const Trainer* trainer = nullptr,
    bool defer_gpu_result = false) {
    bool host_issue = false;
#ifdef USE_CUDA
    int* device_issue =
        gpu_custom_kernels_supported() ? finite_issue_accumulator() : nullptr;
    if (device_issue) {
        const cudaError_t clear_status =
            cudaMemsetAsync(device_issue, 0, sizeof(int), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("optimizer finite-gate clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(clear_status));
        }
    }
    if (device_issue) attention_training::merge_status(params, device_issue);
    std::vector<const Tensor*> gpu_tensors;
    std::vector<unsigned char> gpu_require_nonnegative;
    gpu_tensors.reserve(params.size() * 4);
    gpu_require_nonnegative.reserve(params.size() * 4);
#endif
    auto inspect = [&](const Tensor& tensor,
                       bool require_nonnegative = false) {
        if (tensor.size == 0) return;
#ifdef USE_CUDA
        if (device_issue && tensor.get_device() == Device::GPU) {
            gpu_tensors.push_back(&tensor);
            gpu_require_nonnegative.push_back(
                require_nonnegative ? 1u : 0u);
            return;
        }
#endif
        Tensor host = tensor.get_device() == Device::GPU ? tensor.cpu() : tensor;
        const float* values = host.data();
        for (int index = 0; index < host.size; ++index) {
            if (!std::isfinite(values[index]) ||
                (require_nonnegative && values[index] < 0.0f)) {
                host_issue = true;
                return;
            }
        }
    };
    for (Parameter* parameter : params) {
        if (!parameter || !parameter->has_gradient()) continue;
        inspect(parameter->data);
        inspect(parameter->grad);
        if (trainer) {
            const auto m =
                trainer->m_state.find(parameter);
            const auto v =
                trainer->v_state.find(parameter);
            if ((m == trainer->m_state.end()) !=
                (v == trainer->v_state.end())) {
                return false;
            }
            if (m != trainer->m_state.end()) {
                inspect(m->second);
                inspect(v->second, true);
            }
        }
        if (host_issue) break;
    }
#ifdef USE_CUDA
    if (device_issue && !host_issue && gpu_tensors.empty()) {
        int issue = 0;
        trainer_check_cuda("attention empty-cohort finite gate");
        const auto fence = cudaStreamSynchronize(nsos::gpu::current_stream());
        if (fence != cudaSuccess) throw std::runtime_error("Attention empty-cohort finite fence failed");
        record_gpu_stream_synchronization();
        const auto copy = cudaMemcpy(&issue, device_issue, sizeof(int), cudaMemcpyDeviceToHost);
        if (copy != cudaSuccess) throw std::runtime_error("Attention empty-cohort finite download failed");
        record_gpu_transfer(Device::CPU, Device::GPU, sizeof(int));
        host_issue = issue != 0;
    }
    if (device_issue && !host_issue && !gpu_tensors.empty()) {
        if (gpu_tensors.size() >= static_cast<size_t>(
                                      std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Optimizer finite-gate tensor count exceeds indexing");
        }
        const int count = static_cast<int>(gpu_tensors.size());
        const bool chunked =
            optimizer_policy::optimizer_finite_chunked_enabled();
        const uint32_t chunk_elements =
            chunked
                ? optimizer_policy::optimizer_finite_chunk_elements()
                : 0u;
        thread_local std::vector<const float*> host_pointers;
        thread_local std::vector<unsigned long long> host_offsets;
        thread_local std::vector<NsosMultiTensorChunk> host_chunks;
        host_pointers.resize(static_cast<size_t>(count));
        host_offsets.assign(static_cast<size_t>(count) + 1u, 0u);
        host_chunks.clear();
        unsigned long long total = 0;
        size_t chunk_count_size = 0;
        for (int index = 0; index < count; ++index) {
            const Tensor* tensor =
                gpu_tensors[static_cast<size_t>(index)];
            const auto elements =
                static_cast<unsigned long long>(tensor->size);
            if (total >
                std::numeric_limits<unsigned long long>::max() - elements) {
                throw std::overflow_error(
                    "Optimizer finite-gate element count overflow");
            }
            host_pointers[static_cast<size_t>(index)] =
                tensor->raw_data();
            host_offsets[static_cast<size_t>(index)] = total;
            total += elements;
            if (chunked) {
                const unsigned long long chunks_for_tensor =
                    1u + (elements - 1u) / chunk_elements;
                if (chunks_for_tensor >
                        static_cast<unsigned long long>(
                            std::numeric_limits<int>::max()) ||
                    chunk_count_size >
                        static_cast<size_t>(
                            std::numeric_limits<int>::max()) -
                            static_cast<size_t>(chunks_for_tensor)) {
                    throw std::overflow_error(
                        "Optimizer finite-gate chunk count exceeds grid "
                        "indexing");
                }
                chunk_count_size +=
                    static_cast<size_t>(chunks_for_tensor);
            }
        }
        host_offsets[static_cast<size_t>(count)] = total;
        if (total == 0) {
            throw std::logic_error(
                "Optimizer finite-gate received an empty GPU cohort");
        }
        const int chunk_count = static_cast<int>(chunk_count_size);
        if (chunked) {
            host_chunks.resize(chunk_count_size);
            size_t next_chunk = 0;
            for (int index = 0; index < count; ++index) {
                const unsigned long long elements =
                    host_offsets[static_cast<size_t>(index + 1)] -
                    host_offsets[static_cast<size_t>(index)];
                for (unsigned long long element_offset = 0;
                     element_offset < elements;
                     element_offset += chunk_elements) {
                    const auto remaining = elements - element_offset;
                    host_chunks[next_chunk++] = NsosMultiTensorChunk{
                        element_offset,
                        static_cast<uint32_t>(
                            std::min<unsigned long long>(
                                remaining, chunk_elements)),
                        static_cast<uint32_t>(index)};
                }
            }
            if (next_chunk != chunk_count_size) {
                throw std::logic_error(
                    "Optimizer finite-gate chunk metadata is inconsistent");
            }
        }

        const size_t pointer_bytes = checked_size_multiply(
            sizeof(float*), static_cast<size_t>(count),
            "Optimizer finite-gate pointer metadata overflow");
        const size_t offsets_offset = checked_size_align(
            pointer_bytes, alignof(unsigned long long),
            "Optimizer finite-gate offset alignment overflow");
        const size_t offset_bytes = checked_size_multiply(
            sizeof(unsigned long long),
            static_cast<size_t>(count) + 1u,
            "Optimizer finite-gate offset metadata overflow");
        const size_t flags_offset = checked_size_add(
            offsets_offset, offset_bytes,
            "Optimizer finite-gate metadata size overflow");
        const size_t flags_bytes = static_cast<size_t>(count);
        const size_t flags_end = checked_size_add(
            flags_offset, flags_bytes,
            "Optimizer finite-gate metadata size overflow");
        const size_t chunks_offset = checked_size_align(
            flags_end, alignof(NsosMultiTensorChunk),
            "Optimizer finite-gate chunk alignment overflow");
        const size_t chunk_bytes = checked_size_multiply(
            chunk_count_size, sizeof(NsosMultiTensorChunk),
            "Optimizer finite-gate chunk metadata overflow");
        const size_t metadata_bytes = checked_size_add(
            chunks_offset, chunk_bytes,
            "Optimizer finite-gate metadata size overflow");
        thread_local std::vector<unsigned char> staging;
        staging.assign(metadata_bytes, 0);
        std::memcpy(staging.data(), host_pointers.data(),
                    pointer_bytes);
        std::memcpy(
            staging.data() + offsets_offset,
            host_offsets.data(),
            offset_bytes);
        std::memcpy(
            staging.data() + flags_offset,
            gpu_require_nonnegative.data(),
            flags_bytes);
        if (chunk_bytes != 0) {
            std::memcpy(
                staging.data() + chunks_offset,
                host_chunks.data(), chunk_bytes);
        }

        unsigned char* device_metadata =
            finite_metadata_buffer(metadata_bytes);
        if (!device_metadata) {
            throw std::runtime_error(
                "optimizer finite-gate metadata allocation failed");
        }
        thread_local std::vector<unsigned char> uploaded_metadata;
        thread_local unsigned char* uploaded_device = nullptr;
        const bool metadata_changed =
            uploaded_device != device_metadata ||
            uploaded_metadata != staging;
        if (metadata_changed) {
            const cudaError_t upload_status =
                cudaMemcpy(device_metadata, staging.data(),
                           staging.size(), cudaMemcpyHostToDevice);
            if (upload_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "optimizer finite-gate metadata upload failed on ") +
                    NSOS_GPU_BACKEND_NAME + ": " +
                    cudaGetErrorString(upload_status));
            }
            record_gpu_transfer(
                Device::GPU, Device::CPU, staging.size());
            uploaded_metadata = staging;
            uploaded_device = device_metadata;
        }
        auto* device_pointers =
            reinterpret_cast<const float* const*>(
                device_metadata);
        const auto* device_offsets =
            reinterpret_cast<const unsigned long long*>(
                device_metadata + offsets_offset);
        const auto* device_require_nonnegative =
            device_metadata + flags_offset;
        const auto* device_chunks = chunked
            ? reinterpret_cast<const NsosMultiTensorChunk*>(
                  device_metadata + chunks_offset)
            : nullptr;
        if (!launch_multi_tensor_check_finite(
            device_issue, device_pointers, device_offsets,
            count, total, device_require_nonnegative,
            device_chunks, chunk_count)) {
            throw std::runtime_error(
                "Optimizer finite-gate launcher rejected invalid arguments");
        }
        trainer_check_cuda(
            "launch_multi_tensor_check_finite(optimizer_gate)");
        if (defer_gpu_result) {
            // The selected optimizer lane has a proven consumer for this same
            // device-side gate. Ordinary fused AdamW reads it before writes;
            // deterministic AdamW folds it into the ordered norm status.
            return true;
        }
        int gpu_issue = 0;
        const cudaError_t copy_status =
            cudaMemcpy(&gpu_issue, device_issue, sizeof(int),
                       cudaMemcpyDeviceToHost);
        record_gpu_stream_synchronization();
        if (copy_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("optimizer finite-gate download failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(copy_status));
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(int));
        host_issue = host_issue || gpu_issue != 0;
    }
#endif
    return !host_issue;
}

float active_loss_scale(const Trainer& trainer) {
    if (!trainer.dynamic_loss_scaling_enabled ||
        matmul_precision_mode() != 2) {
        return 1.0f;
    }
    if (!std::isfinite(trainer.loss_scale) ||
        !std::isfinite(trainer.min_loss_scale) ||
        !std::isfinite(trainer.max_loss_scale) ||
        trainer.min_loss_scale < 1.0f ||
        trainer.max_loss_scale < trainer.min_loss_scale ||
        trainer.loss_scale < trainer.min_loss_scale ||
        trainer.loss_scale > trainer.max_loss_scale) {
        throw std::invalid_argument(
            "Trainer dynamic loss-scale bounds are invalid");
    }
    return trainer.loss_scale;
}

void validate_phase_scheduler_configuration(
    const TrainPhaseScheduler& scheduler) {
    if (scheduler.semantic_warmup_steps < 0 ||
        scheduler.qat_start_step < 0 ||
        scheduler.quantized_precision_bits != 2 ||
        scheduler.activation_precision_bits < 2 ||
        scheduler.activation_precision_bits > 8 ||
        !std::isfinite(scheduler.ternary_regularization) ||
        scheduler.ternary_regularization < 0.0f ||
        scheduler.auxiliary_reasoning_iterations <= 0 ||
        scheduler.auxiliary_reasoning_simulations <= 0 ||
        !std::isfinite(scheduler.auxiliary_memory_blend) ||
        scheduler.auxiliary_memory_blend < 0.0f ||
        scheduler.auxiliary_memory_blend > 1.0f ||
        scheduler.auxiliary_every_steps <= 0 ||
        scheduler.auxiliary_prompt_max_tokens <= 0 ||
        scheduler.auxiliary_answer_max_tokens <= 0 ||
        scheduler.auxiliary_memory_scope < 0 ||
        scheduler.auxiliary_oxtamem_size_mb < 1 ||
        scheduler.auxiliary_oxtamem_size_mb > 32768 ||
        (scheduler.auxiliary_oxtamem_enabled &&
         scheduler.auxiliary_oxtamem_store_path.empty())) {
        throw std::invalid_argument(
            "Invalid training schedule: steps and auxiliary limits must be "
            "positive, ternary weights require 2 bits, activation precision "
            "must be in [2, 8], memory blend must be in [0, 1], and OxtaMem "
            "requires a store path and size in [1, 32768] MiB");
    }
}

void validate_trainer_configuration(const Trainer& trainer) {
    if (!trainer.model) {
        throw std::runtime_error("Trainer requires model");
    }
    validate_auxiliary_consumer(trainer);
    if (!std::isfinite(trainer.learning_rate) ||
        trainer.learning_rate < 0.0f ||
        !std::isfinite(trainer.beta1) ||
        trainer.beta1 < 0.0f || trainer.beta1 >= 1.0f ||
        !std::isfinite(trainer.beta2) ||
        trainer.beta2 < 0.0f || trainer.beta2 >= 1.0f ||
        !std::isfinite(trainer.eps) || trainer.eps <= 0.0f ||
        !std::isfinite(trainer.weight_decay) ||
        trainer.weight_decay < 0.0f ||
        !std::isfinite(trainer.max_grad_norm) ||
        trainer.max_grad_norm <= 0.0f ||
        !std::isfinite(trainer.min_learning_rate_scale) ||
        trainer.min_learning_rate_scale < 0.0f ||
        trainer.min_learning_rate_scale > 1.0f) {
        throw std::invalid_argument(
            "Trainer optimizer and learning-rate configuration is invalid");
    }
    for (const float scale : {
             trainer.first_token_loss_scale,
             trainer.eos_loss_scale,
             trainer.repetition_unlikelihood_scale,
             trainer.moe_aux_loss_scale,
             trainer.logit_l2_beta,
             trainer.pantheon_vib_beta}) {
        if (!std::isfinite(scale) || scale < 0.0f) {
            throw std::invalid_argument(
                "Trainer objective scales must be finite and non-negative");
        }
    }
    if (trainer.logit_l2_beta != 0.0f &&
        trainer.pantheon_vib_beta != 0.0f &&
        trainer.logit_l2_beta != trainer.pantheon_vib_beta) {
        throw std::invalid_argument(
            "logit_l2_beta conflicts with deprecated pantheon_vib_beta");
    }
    if (trainer.warmup_steps < 0 ||
        trainer.global_step_count < 0 ||
        trainer.global_step_count == std::numeric_limits<int>::max() ||
        trainer.total_training_steps < 0 ||
        (trainer.optimizer_state_bits != 4 &&
         trainer.optimizer_state_bits != 32)) {
        throw std::invalid_argument(
            "Trainer step counters or optimizer-state precision are invalid");
    }
    const int vocab_size = trainer.model->model_config().vocab_size;
    if (trainer.eos_token_id < 0 ||
        trainer.eos_token_id >= vocab_size) {
        throw std::invalid_argument(
            "Trainer EOS token is outside the model vocabulary");
    }
    if (!std::isfinite(trainer.loss_scale) ||
        !std::isfinite(trainer.min_loss_scale) ||
        !std::isfinite(trainer.max_loss_scale) ||
        !std::isfinite(trainer.loss_scale_growth_factor) ||
        !std::isfinite(trainer.loss_scale_backoff_factor) ||
        trainer.min_loss_scale < 1.0f ||
        trainer.max_loss_scale < trainer.min_loss_scale ||
        trainer.loss_scale < trainer.min_loss_scale ||
        trainer.loss_scale > trainer.max_loss_scale ||
        trainer.loss_scale_growth_factor < 1.0f ||
        trainer.loss_scale_backoff_factor <= 0.0f ||
        trainer.loss_scale_backoff_factor >= 1.0f ||
        trainer.loss_scale_growth_interval <= 0 ||
        trainer.loss_scale_growth_tracker < 0 ||
        trainer.loss_scale_growth_tracker >=
            trainer.loss_scale_growth_interval) {
        throw std::invalid_argument(
            "Trainer dynamic loss-scale configuration is invalid");
    }
    if (trainer.scheduler_unit == Trainer::SchedulerUnit::Tokens && !trainer.token_counters_complete)
        throw std::invalid_argument("Token scheduler requires complete historical token counters");
    validate_phase_scheduler_configuration(trainer.phase_scheduler);
}

void record_loss_scale_success(Trainer& trainer) {
    trainer.last_optimizer_step_skipped = false;
    if (active_loss_scale(trainer) == 1.0f) return;
    if (!std::isfinite(trainer.loss_scale_growth_factor) ||
        trainer.loss_scale_growth_factor < 1.0f ||
        trainer.loss_scale_growth_interval <= 0) {
        throw std::invalid_argument(
            "Trainer loss-scale growth policy is invalid");
    }
    ++trainer.loss_scale_growth_tracker;
    if (trainer.loss_scale_growth_tracker >=
        trainer.loss_scale_growth_interval) {
        trainer.loss_scale = std::min(
            trainer.max_loss_scale,
            trainer.loss_scale * trainer.loss_scale_growth_factor);
        trainer.loss_scale_growth_tracker = 0;
    }
}

void record_loss_scale_overflow(Trainer& trainer,
                                const std::vector<Parameter*>& params) {
    if (!std::isfinite(trainer.loss_scale_backoff_factor) ||
        trainer.loss_scale_backoff_factor <= 0.0f ||
        trainer.loss_scale_backoff_factor >= 1.0f) {
        throw std::invalid_argument(
            "Trainer loss-scale backoff policy is invalid");
    }
    trainer.loss_scale = std::max(
        trainer.min_loss_scale,
        trainer.loss_scale * trainer.loss_scale_backoff_factor);
    trainer.loss_scale_growth_tracker = 0;
    trainer.last_optimizer_step_skipped = true;
    zero_model_gradients(params);
}

float clip_gradients(const std::vector<Parameter*>& params, float max_norm,
                     const int* deferred_finite_issue = nullptr,
                     bool* deferred_issue_out = nullptr) {
    if ((deferred_finite_issue == nullptr) !=
        (deferred_issue_out == nullptr)) {
        throw std::invalid_argument(
            "Deferred optimizer finite status requires both device and host "
            "outputs");
    }
    if (deferred_issue_out != nullptr) {
        *deferred_issue_out = false;
    }
    float total_norm = 0.0f;
    bool fused_done = false;
    bool clipped_on_device = false;
    bool deferred_issue = false;
    // K4: deterministic mode uses a fixed reduction tree for every GPU tensor,
    // followed by one ordered device-side reduction over parameter partials.
    // The old deterministic implementation copied every complete gradient to
    // the host (tens of gigabytes over a real campaign). Only one final double
    // scalar may now cross the PCIe boundary. The ordinary atomic fast path
    // remains the default when deterministic reductions are disabled.
    if (determinism::deterministic_reductions_enabled()) {
        double total_sq = 0.0;
        bool has_host_gradients = false;
#ifdef USE_CUDA
        std::vector<float*> gpu_gradients;
        std::vector<unsigned long long> gpu_sizes;
        gpu_gradients.reserve(params.size());
        gpu_sizes.reserve(params.size());
#endif
        for (auto* p : params) {
            if (!p || !p->has_gradient()) continue;
            if (p->grad.get_device() == Device::GPU) {
#ifdef USE_CUDA
                if (!gpu_custom_kernels_supported()) {
                    throw std::runtime_error(
                        "Deterministic GPU gradient clipping requires the "
                        "NSOS custom GPU kernels");
                }
                gpu_gradients.push_back(p->grad.raw_data());
                gpu_sizes.push_back(
                    static_cast<unsigned long long>(p->grad.size));
                continue;
#else
                throw std::runtime_error(
                    "A GPU gradient reached a CPU-only deterministic build");
#endif
            }
            const float* gp = p->grad.data();
            has_host_gradients = true;
            const int n = static_cast<int>(p->grad.size);
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                const double value = static_cast<double>(gp[i]);
                s += value * value;
            }
            total_sq += s;
        }
#ifdef USE_CUDA
        if (!gpu_gradients.empty()) {
            DeterministicClipWorkspace& workspace =
                deterministic_clip_workspace();
            const size_t count = gpu_gradients.size();
            float** device_gradients = workspace.gradients.ensure(count);
            unsigned long long* device_sizes =
                workspace.sizes.ensure(count);
            const bool device_clip = optimizer_policy::device_gradient_clip_enabled() &&
                                     !has_host_gradients;
            double* device_partials = device_clip ? nullptr : workspace.partials.ensure(count);
            double* device_total = workspace.total.ensure(1);
            if (!device_gradients || !device_sizes ||
                (!device_clip && !device_partials) || !device_total) {
                throw std::runtime_error(
                    "Deterministic GPU gradient clipping workspace "
                    "allocation failed");
            }

            const bool gradients_changed =
                workspace.uploaded_gradient_device != device_gradients ||
                workspace.uploaded_gradients != gpu_gradients;
            if (gradients_changed) {
                const size_t bytes = count * sizeof(float*);
                const cudaError_t status =
                    cudaMemcpy(device_gradients, gpu_gradients.data(), bytes,
                               cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string(
                            "Deterministic gradient-pointer upload failed on ") +
                        NSOS_GPU_BACKEND_NAME + ": " +
                        cudaGetErrorString(status));
                }
                record_gpu_transfer(Device::GPU, Device::CPU, bytes);
                workspace.uploaded_gradients = gpu_gradients;
                workspace.uploaded_gradient_device = device_gradients;
            }

            const bool sizes_changed =
                workspace.uploaded_size_device != device_sizes ||
                workspace.uploaded_sizes != gpu_sizes;
            if (sizes_changed) {
                const size_t bytes =
                    count * sizeof(unsigned long long);
                const cudaError_t status =
                    cudaMemcpy(device_sizes, gpu_sizes.data(), bytes,
                               cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string(
                            "Deterministic gradient-size upload failed on ") +
                        NSOS_GPU_BACKEND_NAME + ": " +
                        cudaGetErrorString(status));
                }
                record_gpu_transfer(Device::GPU, Device::CPU, bytes);
                workspace.uploaded_sizes = gpu_sizes;
                workspace.uploaded_size_device = device_sizes;
                workspace.host_chunks.clear();
                workspace.uploaded_chunks_device = nullptr;
            }

            bool norm_enqueued = false;
            if (device_clip) {
                if (sizes_changed || workspace.host_chunks.empty()) {
                    workspace.host_chunks.clear();
                    for (size_t tensor = 0; tensor < count; ++tensor) {
                        for (uint64_t offset = 0; offset < gpu_sizes[tensor]; offset += 8192) {
                            workspace.host_chunks.push_back({offset,
                                static_cast<uint32_t>(std::min<uint64_t>(8192, gpu_sizes[tensor] - offset)),
                                static_cast<uint32_t>(tensor)});
                        }
                    }
                }
                if (workspace.host_chunks.size() > static_cast<size_t>(INT_MAX))
                    throw std::overflow_error("Gradient norm chunk count exceeds INT_MAX");
                const size_t chunk_count = workspace.host_chunks.size();
                auto* chunks = workspace.chunks.ensure(chunk_count);
                if (sizes_changed || workspace.uploaded_chunks_device != chunks) {
                    const size_t bytes = chunk_count * sizeof(NsosMultiTensorChunk);
                    const auto status = cudaMemcpy(chunks, workspace.host_chunks.data(), bytes, cudaMemcpyHostToDevice);
                    if (status != cudaSuccess) throw std::runtime_error("Gradient norm chunk upload failed");
                    record_gpu_transfer(Device::GPU, Device::CPU, bytes);
                    workspace.uploaded_chunks_device = chunks;
                }
                device_partials = workspace.partials.ensure(chunk_count);
                norm_enqueued = launch_chunked_norm_device_clip(device_total, device_partials,
                    workspace.coefficient.ensure(1), device_gradients, chunks,
                    static_cast<int>(chunk_count), max_norm, deferred_finite_issue);
                clipped_on_device = norm_enqueued;
            } else {
                norm_enqueued = launch_multi_tensor_sqsum_deterministic(
                device_total, device_partials, device_gradients,
                device_sizes, static_cast<int>(count),
                deferred_finite_issue);
            }
            if (!norm_enqueued) {
                throw std::runtime_error(
                    "Deterministic multi-tensor gradient norm rejected "
                    "invalid arguments");
            }
            trainer_check_cuda(
                "launch_multi_tensor_sqsum_deterministic(clip)");
            double gpu_sq = 0.0;
            const cudaError_t status =
                cudaMemcpy(&gpu_sq, device_total, sizeof(double),
                           cudaMemcpyDeviceToHost);
            record_gpu_stream_synchronization();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Deterministic gradient norm download failed on ") +
                    NSOS_GPU_BACKEND_NAME + ": " +
                    cudaGetErrorString(status));
            }
            record_gpu_transfer(
                Device::CPU, Device::GPU, sizeof(double));
            deferred_issue = gpu_sq < 0.0;
            if (!deferred_issue) {
                total_sq += gpu_sq;
            }
        }
#endif
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
        const cudaError_t clear_status =
            cudaMemsetAsync(d_accum, 0, sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("gradient norm clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(clear_status));
        }
        double host_sq = 0.0;
        for (auto* p : params) {
            if (!p || !p->has_gradient()) continue;
            if (p->grad.get_device() == Device::GPU) {
                launch_norm_kernel(d_accum, p->grad.raw_data(), p->grad.size);
            } else {
                const float n = p->grad.norm();  // CPU path: no device sync
                host_sq += static_cast<double>(n) * static_cast<double>(n);
            }
        }
        trainer_check_cuda("launch_norm_kernel(clip)");
        float gpu_sq = 0.0f;
        const cudaError_t norm_copy_status =
            cudaMemcpy(&gpu_sq, d_accum, sizeof(float),
                       cudaMemcpyDeviceToHost);
        record_gpu_stream_synchronization();
        if (norm_copy_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("gradient norm download failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(norm_copy_status));
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(float));
        total_norm = std::sqrt(static_cast<float>(host_sq) + gpu_sq);
        fused_done = true;
    }
#endif
    if (!fused_done) {
        float total_norm_sq = 0.0f;
        for (auto* p : params) {
            if (!p || !p->has_gradient()) continue;
            const float norm = p->grad.norm();
            total_norm_sq += norm * norm;
        }
        total_norm = std::sqrt(total_norm_sq);
    }

    if (deferred_issue_out != nullptr) {
        *deferred_issue_out = deferred_issue;
    }
    if (!clipped_on_device && !deferred_issue && total_norm > max_norm) {
        const float coeff = max_norm / (total_norm + 1e-6f);
        for (auto* p : params) {
            if (!p || !p->has_gradient()) continue;
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

bool should_apply_weight_decay(const Parameter& parameter, const Trainer& trainer) {
    if (trainer.model) {
        const auto excluded = trainer.model->no_weight_decay_parameters();
        if (std::find(excluded.begin(), excluded.end(), &parameter) != excluded.end()) return false;
    }
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
        layer->set_precision_mode(
            trainer.phase_scheduler.activation_precision_bits);

        // True quantized training: once the quantized phase is active, route
        // the forward through a ternary/int8 path and back-propagate with a
        // straight-through estimator onto the FP32 latent weights.
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
    float sparse_qat_loss = 0.0f;
    std::unique_ptr<gpu::ExecutionContext::Scope> sparse_lane;
    if (trainer.device_sparse_group_open) {
        sparse_lane=std::make_unique<gpu::ExecutionContext::Scope>(trainer.device_sparse_execution_context());
        Tensor grouped_loss;
        for (const auto& group : trainer.device_moe_groups) {
            auto producer = group->device_activity()->producer();
            if (!producer) throw std::logic_error("Missing completed device MoE producer for QAT");
            Tensor loss = producer->add_device_qat_regularization(base_regularization, accumulation_steps);
            if (!grouped_loss.size) grouped_loss = std::move(loss); else grouped_loss.add_inplace_(loss);
        }
        if (grouped_loss.size) sparse_qat_loss = grouped_loss.cpu().data()[0];
    }

    double host_penalty_grad_sq = 0.0;
#ifdef USE_CUDA
    float* device_penalty_grad_sq =
        gpu_custom_kernels_supported() ? clip_norm_accumulator() : nullptr;
    if (device_penalty_grad_sq) {
        const cudaError_t status =
            cudaMemset(device_penalty_grad_sq, 0, sizeof(float));
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemset(qat penalty norm) failed: ") +
                cudaGetErrorString(status));
        }
    }
#endif

    for (BitLinear* layer : trainer.model->collect_bitlinear_layers()) {
        if (!layer || !layer->has_full_precision_weight()) {
            continue;
        }
        // Skip layers kept on the float path (dt/B/C, MoE router): they are never
        // ternarized in the forward, so pulling their latent weights toward
        // ternary codes is spurious pressure on the SSM/routing gain.
        if (layer->quantization_sensitive() || layer->weight.has_device_gradient_activity()) {
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
        const cudaError_t penalty_copy_status =
            cudaMemcpy(&gpu_penalty_grad_sq, device_penalty_grad_sq,
                       sizeof(float), cudaMemcpyDeviceToHost);
        record_gpu_stream_synchronization();
        if (penalty_copy_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "QAT regularization norm download failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(penalty_copy_status));
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(float));
        host_penalty_grad_sq += gpu_penalty_grad_sq;
    }
#endif
    // penalty_grad = regularization * diff.  Report the post-accumulation
    // objective whose gradient remains after apply_optimizer_step divides by
    // accumulation_steps: 0.5 * base_regularization * ||diff||^2.
    const double inverse_regularization_sq =
        1.0 / (static_cast<double>(regularization) * regularization);
    return sparse_qat_loss + static_cast<float>(
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

bool cancel_moe_aux_accumulation(Trainer& trainer) noexcept {
    if (!trainer.model) return true;
    try {
        for (auto& layer : trainer.model->layers) {
            if (layer && layer->router) {
                layer->router->cancel_aux_accumulation();
            }
        }
        return true;
    } catch (...) {
        // Recovery must never replace the original training exception.
        return false;
    }
}

void throw_if_training_cancelled(Trainer& trainer) {
    if (trainer.cancellation_requested()) {
        throw AbortException();
    }
}

class TrainingAttemptGuard {
public:
    TrainingAttemptGuard(
        Trainer& trainer,
        const std::vector<Parameter*>& parameters)
        : trainer_(trainer), parameters_(parameters) {}

    TrainingAttemptGuard(const TrainingAttemptGuard&) = delete;
    TrainingAttemptGuard& operator=(
        const TrainingAttemptGuard&) = delete;

    ~TrainingAttemptGuard() {
        if (committed_) return;
        if (optimizer_committed_) {
            // The weights and optimizer were published, but the enclosing API
            // did not complete (for example telemetry/callback failed). A
            // caller cannot safely infer whether retrying the batch is valid.
            trainer_.mark_optimizer_state_poisoned();
        }
        bool cleanup_succeeded = cancel_moe_aux_accumulation(trainer_);
        try { trainer_.finish_device_sparse_group(true); }
        catch (...) { cleanup_succeeded = false; }
        try {
            zero_model_gradients(parameters_);
        } catch (...) {
            cleanup_succeeded = false;
        }
        try {
            if (trainer_.model) {
                trainer_.model->reset_session();
            }
        } catch (...) {
            cleanup_succeeded = false;
        }
        if (!cleanup_succeeded) {
            trainer_.mark_optimizer_state_poisoned();
        }
    }

    void mark_optimizer_committed() noexcept { optimizer_committed_ = true; }
    void commit() noexcept { committed_ = true; }

private:
    Trainer& trainer_;
    const std::vector<Parameter*>& parameters_;
    bool optimizer_committed_ = false;
    bool committed_ = false;
};

float apply_moe_aux_regularization(Trainer& trainer, int accumulation_steps) {
    if(trainer.device_sparse_group_open)trainer.device_sparse_objectives_finalized=true;
    if (!trainer.model) {
        return 0.0f;
    }

    if (trainer.moe_aux_loss_scale <= 0.0f) {
        for (auto& layer : trainer.model->layers) {
            if (layer && layer->router) layer->router->cancel_aux_accumulation();
        }
        return 0.0f;
    }

    // The historical heuristic is not the derivative of any scalar objective.
    // Production training therefore always uses the exact Switch objective;
    // there is deliberately no runtime rollback that could make reported loss
    // and gradients diverge again.
    const int objective_divisor = std::max(accumulation_steps, 1);
    float total_effective_loss = 0.0f;
    Tensor device_loss;
    const float effective_aux_scale =
        trainer.moe_aux_loss_scale *
        static_cast<float>(std::max(accumulation_steps, 1));

    for (auto& layer : trainer.model->layers) {
        if (!layer || !layer->router || !layer->router->gate) {
            continue;
        }
        Tensor loss = layer->router->accumulate_switch_aux_grad_device(
            layer->router->aux_loss_coef * effective_aux_scale);
        if (loss.get_device() == Device::GPU) {
            // Exact VJPs are already enqueued. Only logging normalization is
            // performed here; no full loads/inputs/probabilities cross PCIe.
            Tensor normalized = loss.mul(1.0f / static_cast<float>(objective_divisor));
            if (device_loss.size == 0) device_loss = normalized;
            else device_loss.add_inplace_(normalized);
        } else {
            total_effective_loss += loss.data()[0] / static_cast<float>(objective_divisor);
        }
    }
    if (device_loss.size != 0) total_effective_loss += device_loss.cpu().data()[0];
    return total_effective_loss;
}

// Posição do scheduler em tokens: os já commitados mais o grupo que está
// prestes a commitar.  Espelha `next_step = global_step_count + 1` do modo
// legado, isto é, a posição depois deste update.
void require_token_capacity(long long value, size_t increment) {
    if (value < 0 || increment > static_cast<unsigned long long>(std::numeric_limits<long long>::max() - value))
        throw std::overflow_error("Training token counter overflow");
}

long long scheduler_token_position(const Trainer& trainer) {
    long long position = trainer.tokens_committed;
    if (trainer.pending_accumulated_tokens > 0) {
        require_token_capacity(position, static_cast<size_t>(trainer.pending_accumulated_tokens));
        position += trainer.pending_accumulated_tokens;
    }
    return position;
}

// Cronograma em tokens.  Warmup linear, depois cosseno até `training_tokens`
// (ou, quando `decay_tokens > 0`, fase estável seguida de decaimento cosseno
// só no trecho final — a forma WSD).
float compute_lr_for_tokens(const Trainer& trainer, long long tokens) {
    const double warmup = static_cast<double>(std::max<long long>(
        trainer.warmup_tokens, 1));
    if (tokens <= trainer.warmup_tokens) {
        return trainer.learning_rate *
               static_cast<float>(static_cast<double>(tokens) / warmup);
    }

    const double total = static_cast<double>(std::max<long long>(
        trainer.training_tokens, trainer.warmup_tokens + 1));
    const float floor_lr =
        trainer.learning_rate * trainer.min_learning_rate_scale;

    double progress = 0.0;
    if (trainer.decay_tokens > 0) {
        // WSD: LR de pico até o início do decaimento, cosseno depois dele.
        const double decay_start =
            std::max(total - static_cast<double>(trainer.decay_tokens),
                     warmup);
        if (static_cast<double>(tokens) <= decay_start) {
            return trainer.learning_rate;
        }
        const double span = std::max(total - decay_start, 1.0);
        progress = (static_cast<double>(tokens) - decay_start) / span;
    } else {
        const double span = std::max(total - warmup, 1.0);
        progress = (static_cast<double>(tokens) - warmup) / span;
    }
    progress = std::clamp(progress, 0.0, 1.0);

    const float cosine =
        0.5f * (1.0f + std::cos(3.14159265f * static_cast<float>(progress)));
    return floor_lr + (trainer.learning_rate - floor_lr) * cosine;
}

float compute_lr_for_step(const Trainer& trainer, int step) {
    if (trainer.scheduler_unit == Trainer::SchedulerUnit::Tokens) {
        return compute_lr_for_tokens(trainer,
                                     scheduler_token_position(trainer));
    }
    const float warmup_steps = static_cast<float>(std::max(trainer.warmup_steps, 1));
    if (step <= trainer.warmup_steps) {
        return trainer.learning_rate *
               static_cast<float>(step) / warmup_steps;
    }

    const float total_steps = static_cast<float>(
        std::max(trainer.total_training_steps, trainer.warmup_steps + 1));
    const float decay_span = std::max(total_steps - warmup_steps, 1.0f);
    float progress =
        static_cast<float>(step - trainer.warmup_steps) / decay_span;
    progress = std::clamp(progress, 0.0f, 1.0f);

    const float cosine = 0.5f * (1.0f + std::cos(3.14159265f * progress));
    const float floor = trainer.learning_rate * trainer.min_learning_rate_scale;
    return floor + (trainer.learning_rate - floor) * cosine;
}

bool finite_values(const std::vector<float>& values,
                   bool require_nonnegative = false) {
    return std::all_of(
        values.begin(), values.end(),
        [&](float value) {
            return std::isfinite(value) &&
                   (!require_nonnegative || value >= 0.0f);
        });
}

void quant4_matrix_shape(const Parameter& parameter,
                         int& rows,
                         int& cols) {
    rows = 0;
    cols = 0;
    const std::vector<int>& dims =
        parameter.data.shape.dims;
    if (dims.size() < 2) {
        return;
    }
    cols = dims.back();
    if (cols <= 0 || parameter.data.size % cols != 0) {
        rows = 0;
        cols = 0;
        return;
    }
    rows = static_cast<int>(parameter.data.size / cols);
}

bool quant4_state_is_valid(const Quant4OptState& state,
                           const Parameter& parameter) {
    const int n = static_cast<int>(parameter.data.size);
    if (n < 0 || state.n != n) {
        return false;
    }
    if (n <= kQuant4MinElems) {
        return !state.quantized &&
               state.m_fp32.size() == static_cast<size_t>(n) &&
               state.v_fp32.size() == static_cast<size_t>(n) &&
               finite_values(state.m_fp32) &&
               finite_values(state.v_fp32, true);
    }
    const size_t packed = static_cast<size_t>(
        n / 2 + static_cast<int>(n % 2 != 0));
    const size_t blocks = static_cast<size_t>(
        n / kQuant4BlockSize +
        static_cast<int>(n % kQuant4BlockSize != 0));
    if (!state.quantized ||
        state.m_codes.size() != packed ||
        state.m_absmax.size() != blocks ||
        state.v_codes.size() != packed ||
        !finite_values(state.m_absmax, true)) {
        return false;
    }

    int rows = 0;
    int cols = 0;
    quant4_matrix_shape(parameter, rows, cols);
    const bool expected_rank1 = rows > 0 && cols > 0;
    if (state.v_rank1 != expected_rank1) {
        return false;
    }
    if (expected_rank1) {
        return state.v_rows == rows &&
               state.v_cols == cols &&
               state.v_row.size() ==
                   static_cast<size_t>(rows) &&
               state.v_col.size() ==
                   static_cast<size_t>(cols) &&
               finite_values(state.v_row, true) &&
               finite_values(state.v_col, true);
    }
    return state.v_rows == 0 && state.v_cols == 0 &&
           state.v_absmax.size() == blocks &&
           finite_values(state.v_absmax, true);
}

bool fp32_state_is_valid(const Trainer& trainer,
                         Parameter* parameter) {
    const auto m = trainer.m_state.find(parameter);
    const auto v = trainer.v_state.find(parameter);
    return m != trainer.m_state.end() &&
           v != trainer.v_state.end() &&
           m->second.shape == parameter->data.shape &&
           v->second.shape == parameter->data.shape &&
           m->second.get_device() ==
               parameter->data.get_device() &&
           v->second.get_device() ==
               parameter->data.get_device();
}

Tensor tensor_from_host_values(
    const std::vector<float>& values,
    const Parameter& parameter) {
    if (values.size() !=
        static_cast<size_t>(parameter.data.size)) {
        throw std::logic_error(
            "Optimizer-state conversion size mismatch");
    }
    Tensor result = Tensor::uninitialized(
        parameter.data.shape.dims,
        parameter.data.get_device());
    copy_tensor_bytes(
        result.raw_data(), result.get_device(),
        values.data(), Device::CPU,
        values.size() * sizeof(float));
    return result;
}

Quant4OptState make_quant4_state(
    const Parameter& parameter,
    const Tensor* source_m,
    const Tensor* source_v) {
    const int n = static_cast<int>(parameter.data.size);
    std::vector<float> m_values(
        static_cast<size_t>(n), 0.0f);
    std::vector<float> v_values(
        static_cast<size_t>(n), 0.0f);
    if (source_m && source_v) {
        const Tensor m_host =
            source_m->get_device() == Device::CPU
                ? *source_m
                : source_m->cpu();
        const Tensor v_host =
            source_v->get_device() == Device::CPU
                ? *source_v
                : source_v->cpu();
        std::copy_n(m_host.data(), n, m_values.data());
        std::copy_n(v_host.data(), n, v_values.data());
        if (!finite_values(m_values) ||
            !finite_values(v_values, true)) {
            throw std::runtime_error(
                "Cannot convert corrupt/non-finite FP32 Adam state to "
                "4-bit state");
        }
    }
    Quant4OptState state;
    int rows = 0;
    int cols = 0;
    quant4_matrix_shape(parameter, rows, cols);
    quant4_store_m(m_values.data(), n, state);
    quant4_store_v(
        v_values.data(), n, rows, cols, state);
    if (!quant4_state_is_valid(state, parameter)) {
        throw std::logic_error(
            "Internal 4-bit optimizer-state preparation failed");
    }
    return state;
}

struct OptimizerPreflight {
    std::vector<float> quant_m_scratch;
    std::vector<float> quant_v_scratch;
};

// Performs every optimizer-state allocation and precision transition before
// the first weight can change. Stable steps only validate existing map nodes
// and resize two thread-local-sized scratch vectors; map copies occur solely
// on first use, architecture surgery, or an explicit 32<->4 bit transition.
OptimizerPreflight prepare_optimizer_state(
    Trainer& trainer,
    const std::vector<Parameter*>& parameters) {
    using TensorStateMap =
        std::unordered_map<Parameter*, Tensor>;
    using QuantStateMap =
        std::unordered_map<Parameter*, Quant4OptState>;
    std::unique_ptr<TensorStateMap> next_m;
    std::unique_ptr<TensorStateMap> next_v;
    std::unique_ptr<QuantStateMap> next_quant;
    auto stage_fp_maps = [&]() {
        if (!next_m) {
            next_m =
                std::make_unique<TensorStateMap>(
                    trainer.m_state);
            next_v =
                std::make_unique<TensorStateMap>(
                    trainer.v_state);
        }
    };
    auto stage_quant_map = [&]() {
        if (!next_quant) {
            next_quant =
                std::make_unique<QuantStateMap>(
                    trainer.quant_state);
        }
    };

    size_t largest_quant_parameter = 0;
    for (Parameter* parameter : parameters) {
        if (!parameter) {
            continue;
        }
        const auto quant_it =
            trainer.quant_state.find(parameter);
        const bool quant_present =
            quant_it != trainer.quant_state.end();
        const bool m_present =
            trainer.m_state.find(parameter) !=
            trainer.m_state.end();
        const bool v_present =
            trainer.v_state.find(parameter) !=
            trainer.v_state.end();
        const bool fp_present = m_present || v_present;
        const bool active = parameter->has_gradient();
        if (!active && !quant_present && !fp_present) {
            continue;
        }
        if (parameter->data.size <= 0 ||
            (active &&
             (parameter->grad.shape != parameter->data.shape ||
              parameter->grad.get_device() !=
                  parameter->data.get_device()))) {
            throw std::logic_error(
                "Optimizer parameter/gradient shape or device invariant "
                "is invalid");
        }
        if (active &&
            parameter->version ==
            std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error(
                "Parameter version counter is exhausted");
        }

        const bool quant_valid =
            quant_present &&
            quant4_state_is_valid(
                quant_it->second, *parameter);
        const bool fp_valid =
            fp32_state_is_valid(trainer, parameter);
        const bool wants_quant =
            trainer.optimizer_state_bits == 4 &&
            parameter->data.get_device() == Device::CPU;

        if (wants_quant) {
            if (active) {
                largest_quant_parameter = std::max(
                    largest_quant_parameter,
                    static_cast<size_t>(
                        parameter->data.size));
            }
            if (!quant_valid) {
                if (quant_present && !fp_valid) {
                    throw std::runtime_error(
                        "Corrupt 4-bit optimizer state has no valid FP32 "
                        "recovery source");
                }
                if (fp_present && !fp_valid) {
                    throw std::runtime_error(
                        "Incomplete or shape-mismatched FP32 optimizer "
                        "state cannot be converted to 4-bit");
                }
                const Tensor* source_m =
                    fp_valid
                        ? &trainer.m_state.at(parameter)
                        : nullptr;
                const Tensor* source_v =
                    fp_valid
                        ? &trainer.v_state.at(parameter)
                        : nullptr;
                Quant4OptState replacement =
                    make_quant4_state(
                        *parameter, source_m, source_v);
                stage_quant_map();
                next_quant->insert_or_assign(
                    parameter, std::move(replacement));
            }
            if (fp_present) {
                if (!fp_valid) {
                    throw std::runtime_error(
                        "Incomplete or corrupt stale FP32 optimizer state");
                }
                stage_fp_maps();
                next_m->erase(parameter);
                next_v->erase(parameter);
            }
            continue;
        }

        if (!fp_valid) {
            if (fp_present && !quant_valid) {
                throw std::runtime_error(
                    "Incomplete or shape-mismatched FP32 optimizer state "
                    "has no valid 4-bit recovery source");
            }
            Tensor replacement_m;
            Tensor replacement_v;
            if (quant_valid) {
                const int n =
                    static_cast<int>(parameter->data.size);
                std::vector<float> m_values(
                    static_cast<size_t>(n));
                std::vector<float> v_values(
                    static_cast<size_t>(n));
                quant4_load_m(
                    quant_it->second, m_values.data(), n);
                quant4_load_v(
                    quant_it->second, v_values.data(), n);
                if (!finite_values(m_values) ||
                    !finite_values(v_values, true)) {
                    throw std::runtime_error(
                        "Decoded 4-bit optimizer state is non-finite");
                }
                replacement_m = tensor_from_host_values(
                    m_values, *parameter);
                replacement_v = tensor_from_host_values(
                    v_values, *parameter);
            } else {
                replacement_m = Tensor::zeros(
                    parameter->data.shape.dims,
                    parameter->data.get_device());
                replacement_v = Tensor::zeros(
                    parameter->data.shape.dims,
                    parameter->data.get_device());
            }
            stage_fp_maps();
            next_m->insert_or_assign(
                parameter, std::move(replacement_m));
            next_v->insert_or_assign(
                parameter, std::move(replacement_v));
        }
        if (quant_present) {
            if (!quant_valid) {
                throw std::runtime_error(
                    "Corrupt stale 4-bit optimizer state");
            }
            stage_quant_map();
            next_quant->erase(parameter);
        }
    }

    OptimizerPreflight prepared;
    prepared.quant_m_scratch.resize(
        largest_quant_parameter);
    prepared.quant_v_scratch.resize(
        largest_quant_parameter);

    // All copies, device allocations, conversions, map nodes and scratch
    // storage have succeeded. These swaps are noexcept and merely canonicalize
    // an absent state to its mathematically equivalent zero representation.
    if (next_m) {
        trainer.m_state.swap(*next_m);
        trainer.v_state.swap(*next_v);
    }
    if (next_quant) {
        trainer.quant_state.swap(*next_quant);
    }
    return prepared;
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
                                 float bc2,
                                 std::vector<float>& m_buf,
                                 std::vector<float>& v_buf) {
    const int n = p->data.size;
    if (m_buf.size() < static_cast<size_t>(n) ||
        v_buf.size() < static_cast<size_t>(n)) {
        throw std::logic_error(
            "4-bit optimizer scratch was not preflighted");
    }
    Quant4OptState& st = trainer.quant_state.at(p);
    quant4_load_m(st, m_buf.data(), n);
    quant4_load_v(st, v_buf.data(), n);

    float* w = p->data.data();
    const float* g = p->grad.data();
    const bool apply_wd =
        trainer.weight_decay > 0.0f && should_apply_weight_decay(*p, trainer);

    for (int i = 0; i < n; ++i) {
        const float updated_m =
            trainer.beta1 * m_buf[i] +
            (1.0f - trainer.beta1) * g[i];
        const float updated_v =
            trainer.beta2 * v_buf[i] +
            (1.0f - trainer.beta2) * g[i] * g[i];
        const float m_hat = updated_m / bc1;
        const float v_hat = updated_v / bc2;
        float updated_weight = w[i];
        if (apply_wd) {
            updated_weight -=
                cur_lr * trainer.weight_decay *
                updated_weight;
        }
        updated_weight -=
            cur_lr * m_hat /
            (std::sqrt(v_hat) + trainer.eps);
        if (!std::isfinite(updated_m) ||
            !std::isfinite(updated_v) ||
            updated_v < 0.0f ||
            !std::isfinite(updated_weight)) {
            throw std::runtime_error(
                "4-bit optimizer produced a non-finite weight or moment");
        }
        m_buf[i] = updated_m;
        v_buf[i] = updated_v;
        w[i] = updated_weight;
    }

    // Treat 2-D states as matrices so v gets the paper's rank-1 normalisation;
    // fold higher-rank tensors to 2-D by their last dimension (rows<=0 disables
    // rank-1 and the v path falls back to block-wise abs-max).
    int rows = 0;
    int cols = 0;
    quant4_matrix_shape(*p, rows, cols);

    quant4_store_m(m_buf.data(), n, st);
    quant4_store_v(v_buf.data(), n, rows, cols, st);
    p->mark_updated();
    optimizer_commit_fault_point();
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
        if ((!parameter->has_device_gradient_activity() && !parameter->has_gradient()) ||
            parameter->data.shape.size() != 2 || parameter->data.shape[0] <= 1 ||
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
    if (bytes == 0) return nullptr;
    thread_local cuda_detail::DeviceBuffer<unsigned char> buffer;
    return buffer.ensure(bytes);
}

size_t align_buffer_offset(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

struct CriticalityActivityTables {
    cuda_detail::DeviceBuffer<const unsigned char*> predicates;
    cuda_detail::DeviceBuffer<const int*> issues;
    explicit CriticalityActivityTables(const std::vector<Parameter*>& parameters) {
        std::vector<const unsigned char*> p;std::vector<const int*> i;
        for(auto* parameter:parameters) {
            const auto& b=parameter->device_gradient_binding();
            p.push_back(b.owner?b.owner->predicate(b.expert):nullptr);
            i.push_back(b.owner?b.owner->issue():nullptr);
        }
        if(!predicates.ensure(p.size())||!issues.ensure(i.size()))throw std::bad_alloc();
        // Caller keeps these tables alive through the metric download or an
        // explicit stream boundary after gradient consumers.
        const auto status=cudaStreamSynchronize(gpu::current_stream());
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));record_gpu_stream_synchronization();
        for(auto entry:{std::make_pair(static_cast<void*>(predicates.get()),static_cast<const void*>(p.data())),
                        std::make_pair(static_cast<void*>(issues.get()),static_cast<const void*>(i.data()))}) {
            const auto copied=cudaMemcpy(entry.first,entry.second,sizeof(void*)*p.size(),cudaMemcpyHostToDevice);
            if(copied!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(copied));
            record_gpu_transfer(Device::GPU,Device::CPU,sizeof(void*)*p.size());
        }
    }
};
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
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(float*) * static_cast<size_t>(count));
        check_cuda_copy(cudaMemcpy(device_offsets, host_offsets.data(),
                                   sizeof(unsigned long long) * static_cast<size_t>(count + 1),
                                   cudaMemcpyHostToDevice), "criticality offset upload");
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(unsigned long long) *
                static_cast<size_t>(count + 1));
        check_cuda_copy(cudaMemcpy(device_fan_in, host_fan_in.data(),
                                   sizeof(int) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality fan-in upload");
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(int) * static_cast<size_t>(count));
        std::vector<Parameter*> candidates;for(size_t index:gpu_indices)candidates.push_back(metrics[index].parameter);
        CriticalityActivityTables activity(candidates);
        if (!launch_activity_criticality_metrics(
            device_weights, device_offsets, device_fan_in, count,activity.predicates.get(),activity.issues.get(),
            device_gammas, device_gains)) {
            throw std::runtime_error(
                "Criticality-metrics launcher rejected invalid arguments");
        }
        trainer_check_cuda("launch_multi_tensor_criticality_metrics");
        check_cuda_copy(cudaMemcpy(host_gammas.data(), device_gammas,
                                   sizeof(float) * static_cast<size_t>(count),
                                   cudaMemcpyDeviceToHost), "criticality gamma download");
        record_gpu_transfer(
            Device::CPU, Device::GPU,
            sizeof(float) * static_cast<size_t>(count));
        check_cuda_copy(cudaMemcpy(host_gains.data(), device_gains,
                                   sizeof(float) * static_cast<size_t>(count),
                                   cudaMemcpyDeviceToHost), "criticality gain download");
        record_gpu_transfer(
            Device::CPU, Device::GPU,
            sizeof(float) * static_cast<size_t>(count));
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
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(float*) * static_cast<size_t>(count));
        check_cuda_copy(cudaMemcpy(buffer + sizeof(float*) * static_cast<size_t>(count),
                                   host_grads.data(),
                                   sizeof(float*) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality grad pointer upload");
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(float*) * static_cast<size_t>(count));
        check_cuda_copy(cudaMemcpy(device_offsets, host_offsets.data(),
                                   sizeof(unsigned long long) * static_cast<size_t>(count + 1),
                                   cudaMemcpyHostToDevice), "criticality grad offset upload");
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(unsigned long long) *
                static_cast<size_t>(count + 1));
        check_cuda_copy(cudaMemcpy(device_coefficients, host_coefficients.data(),
                                   sizeof(float) * static_cast<size_t>(count),
                                   cudaMemcpyHostToDevice), "criticality coefficient upload");
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            sizeof(float) * static_cast<size_t>(count));
        std::vector<Parameter*> candidates;for(size_t index:gpu_indices)candidates.push_back(metrics[index].parameter);
        CriticalityActivityTables activity(candidates);
        if (!launch_activity_criticality_gradient(
            device_weights, device_grads, device_offsets, device_coefficients,
            count,activity.predicates.get(),activity.issues.get())) {
            throw std::runtime_error(
                "Criticality-gradient launcher rejected invalid arguments");
        }
        trainer_check_cuda("launch_multi_tensor_criticality_grad");
        check_cuda_copy(cudaStreamSynchronize(gpu::current_stream()),"criticality activity lifetime");
        record_gpu_stream_synchronization();
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

class OptimizerCommitGuard {
public:
    explicit OptimizerCommitGuard(Trainer& trainer) noexcept
        : trainer_(trainer) {}

    OptimizerCommitGuard(const OptimizerCommitGuard&) = delete;
    OptimizerCommitGuard& operator=(
        const OptimizerCommitGuard&) = delete;

    void arm() noexcept {
        armed_ = true;
    }

    void complete() noexcept {
        armed_ = false;
    }

    ~OptimizerCommitGuard() {
        if (armed_) {
            // No allocation, logging, device call or lock is permitted here:
            // this path also runs while unwinding an OOM/backend exception.
            trainer_.mark_optimizer_state_poisoned();
        }
    }

private:
    Trainer& trainer_;
    bool armed_ = false;
};

#ifdef USE_CUDA
namespace {

// NSOS_FUSED_OPT=0 desliga o passo fundido (braço A/B; default ON).
bool fused_optimizer_enabled() {
    const char* e = std::getenv("NSOS_FUSED_OPT");
    return e == nullptr || e[0] != '0';
}

bool deterministic_multi_tensor_optimizer_enabled() {
    const char* value =
        std::getenv("NSOS_DETERMINISTIC_MULTI_TENSOR_OPT");
    return value == nullptr || value[0] != '0';
}

bool optimizer_update_can_defer_finite_gate(
    Trainer& trainer,
    const std::vector<Parameter*>& params) {
    if (!gpu_custom_kernels_supported()) {
        return false;
    }
    const bool deterministic =
        determinism::deterministic_reductions_enabled();
    if (deterministic) {
        if (!deterministic_multi_tensor_optimizer_enabled() ||
            !optimizer_policy::
                deterministic_finite_gate_deferred_enabled()) {
            return false;
        }
    } else if (!fused_optimizer_enabled() ||
               clip_norm_accumulator() == nullptr) {
        return false;
    }
    bool found_active = false;
    for (Parameter* parameter : params) {
        if (!parameter || !parameter->has_gradient()) {
            continue;
        }
        found_active = true;
        if (parameter->data.get_device() != Device::GPU ||
            parameter->grad.get_device() != Device::GPU) {
            return false;
        }
        const auto moment = trainer.m_state.find(parameter);
        const auto variance = trainer.v_state.find(parameter);
        if (moment == trainer.m_state.end() ||
            variance == trainer.v_state.end() ||
            !can_use_gpu_optimizer(
                *parameter, moment->second, variance->second)) {
            return false;
        }
    }
    return found_active;
}

// Buffer device persistente para os metadados do passo fundido
// (ponteiros w/g/m/v + offsets + flags de weight-decay). Cresce sob demanda;
// a coorte é reenviada somente quando algum endereço/shape/controle muda.
unsigned char* fused_opt_meta_buffer(size_t bytes) {
    if (bytes == 0) return nullptr;
    thread_local cuda_detail::DeviceBuffer<unsigned char> buffer;
    return buffer.ensure(bytes);
}

struct DeterministicOptimizerWorkspace {
    cuda_detail::DeviceBuffer<unsigned char> static_metadata;
    cuda_detail::DeviceBuffer<float> learning_rates;
    std::vector<unsigned char> uploaded_static_metadata;
    std::vector<float> uploaded_learning_rates;
    unsigned char* uploaded_static_device = nullptr;
    float* uploaded_learning_rate_device = nullptr;
};

DeterministicOptimizerWorkspace& deterministic_optimizer_workspace() {
    thread_local DeterministicOptimizerWorkspace workspace;
    return workspace;
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
                                       float* grad_norm_out,
                                       OptimizerCommitGuard& commit_guard,
                                       bool consume_deferred_finite_gate) {
    if (!fused_optimizer_enabled() || !gpu_custom_kernels_supported()) {
        if (consume_deferred_finite_gate) {
            throw std::logic_error(
                "Deferred optimizer finite gate lost its fused consumer");
        }
        return false;
    }
    // K4: the fused step folds the atomicAdd clip-norm + multi-tensor AdamW; in
    // deterministic mode fall back to the ordered per-parameter path (with the
    // ordered host clip above) for bit-reproducibility.
    if (determinism::deterministic_reductions_enabled()) {
        if (consume_deferred_finite_gate) {
            throw std::logic_error(
                "Deferred optimizer finite gate reached deterministic "
                "fallback");
        }
        return false;
    }
    float* d_accum = clip_norm_accumulator();
    if (!d_accum) {
        if (consume_deferred_finite_gate) {
            throw std::runtime_error(
                "Deferred optimizer finite gate lost its norm accumulator");
        }
        return false;
    }

    std::vector<Parameter*> active;
    active.reserve(params.size());
    for (auto* p : params) {
        if (!p || !p->has_gradient()) continue;
        if (p->data.get_device() != Device::GPU ||
            p->grad.get_device() != Device::GPU) {
            if (consume_deferred_finite_gate) {
                throw std::logic_error(
                    "Deferred optimizer finite gate reached a host tensor");
            }
            return false;
        }
        active.push_back(p);
    }
    if (active.empty()) {
        if (consume_deferred_finite_gate) {
            throw std::logic_error(
                "Deferred optimizer finite gate has no active parameters");
        }
        return false;
    }
    if (active.size() >= static_cast<size_t>(
                             std::numeric_limits<int>::max())) {
        if (consume_deferred_finite_gate) {
            throw std::overflow_error(
                "Fused optimizer tensor count exceeds metadata indexing");
        }
        return false;
    }

    for (auto* p : active) {
        const auto m_it = trainer.m_state.find(p);
        const auto v_it = trainer.v_state.find(p);
        if (m_it == trainer.m_state.end() ||
            v_it == trainer.v_state.end() ||
            !can_use_gpu_optimizer(
                *p, m_it->second, v_it->second)) {
            if (consume_deferred_finite_gate) {
                throw std::logic_error(
                    "Deferred optimizer finite gate lost an optimizer "
                    "state tensor");
            }
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
        h_m[i] = trainer.m_state.at(p).raw_data();
        h_v[i] = trainer.v_state.at(p).raw_data();
        h_off[i] = total;
        total += static_cast<unsigned long long>(p->data.size);
        h_wd[i] = should_apply_weight_decay(*p, trainer) ? 1 : 0;
        h_lr[i] = trainer.lr_scale_for(p);
    }
    h_off[n] = total;
    if (total == 0) {
        if (consume_deferred_finite_gate) {
            throw std::logic_error(
                "Deferred optimizer finite gate has an empty cohort");
        }
        return false;
    }

    unsigned char* d_meta = fused_opt_meta_buffer(staging.size());
    if (!d_meta) {
        if (consume_deferred_finite_gate) {
            throw std::runtime_error(
                "Deferred optimizer finite gate could not allocate fused "
                "metadata");
        }
        return false;
    }
    // add_grad keeps each gradient allocation stable. Avoid resending the
    // metadata cohort when pointers, offsets, flags and LR scales are byte-for-
    // byte unchanged; invalidate automatically if the device buffer grows.
    thread_local std::vector<unsigned char> uploaded_meta;
    thread_local unsigned char* uploaded_device = nullptr;
    const bool metadata_changed =
        uploaded_device != d_meta ||
        uploaded_meta.size() != staging.size() ||
        !std::equal(staging.begin(), staging.end(),
                    uploaded_meta.begin());
    if (metadata_changed) {
        const cudaError_t upload_status =
            cudaMemcpy(d_meta, staging.data(), staging.size(),
                       cudaMemcpyHostToDevice);
        if (upload_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("fused optimizer metadata upload failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(upload_status));
        }
        record_gpu_transfer(
            Device::GPU, Device::CPU, staging.size());
        uploaded_meta = staging;
        uploaded_device = d_meta;
    }
    auto* d_w = reinterpret_cast<float* const*>(d_meta);
    auto* d_g = d_w + n;
    auto* d_m = d_w + 2 * n;
    auto* d_v = d_w + 3 * n;
    const auto* d_off =
        reinterpret_cast<const unsigned long long*>(d_meta + ptr_bytes);
    const unsigned char* d_wd = d_meta + ptr_bytes + off_bytes;
    const float* d_lr = reinterpret_cast<const float*>(d_meta + lr_offset);

    const cudaError_t clear_status =
        cudaMemsetAsync(d_accum, 0, sizeof(float), nsos::gpu::current_stream());
    if (clear_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("fused optimizer norm clear failed on ") +
            NSOS_GPU_BACKEND_NAME + ": " +
            cudaGetErrorString(clear_status));
    }
    launch_multi_tensor_sqsum(d_accum, d_w, d_g, d_m, d_v, d_off, d_wd, n, total);
    trainer_check_cuda("launch_multi_tensor_sqsum");

    const float acc_scale =
        1.0f / static_cast<float>(std::max(accumulation_steps, 1));
    const int next_step = trainer.global_step_count + 1;
    const float cur_lr =
        compute_lr_for_step(trainer, next_step);
    const float bc1 =
        1.0f - std::pow(trainer.beta1, next_step);
    const float bc2 =
        1.0f - std::pow(trainer.beta2, next_step);
    int* d_commit_issue = finite_issue_accumulator();
    if (!d_commit_issue) {
        throw std::runtime_error(
            "Fused optimizer commit-status allocation failed");
    }
    if (!consume_deferred_finite_gate) {
        const cudaError_t issue_clear_status =
            cudaMemsetAsync(
                d_commit_issue, 0, sizeof(int), nsos::gpu::current_stream());
        if (issue_clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "fused optimizer status clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(issue_clear_status));
        }
    }

    commit_guard.arm();
    launch_multi_tensor_adamw(d_w, d_g, d_m, d_v, d_off, d_wd, d_lr, n,
                              total, d_accum, acc_scale,
                              trainer.max_grad_norm,
                              trainer.beta1, trainer.beta2,
                              bc1, bc2, cur_lr, trainer.eps,
                              trainer.weight_decay,
                              d_commit_issue);
    trainer_check_cuda("launch_multi_tensor_adamw");
    int commit_issue = 0;
    const cudaError_t issue_copy_status =
        cudaMemcpy(
            &commit_issue, d_commit_issue, sizeof(int),
            cudaMemcpyDeviceToHost);
    record_gpu_stream_synchronization();
    if (issue_copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "fused optimizer status download failed on ") +
            NSOS_GPU_BACKEND_NAME + ": " +
            cudaGetErrorString(issue_copy_status));
    }
    record_gpu_transfer(
        Device::CPU, Device::GPU, sizeof(int));
    if (consume_deferred_finite_gate && commit_issue == 1) {
        // The fused kernel observes the deferred bit before touching weights
        // or moments. This is a cleanly rejected step, not a partial commit.
        commit_guard.complete();
        trainer.last_optimizer_step_skipped = true;
        zero_model_gradients(params);
        throw std::runtime_error(
            "optimizer step rejected: a parameter or gradient contains "
            "NaN/Inf");
    }
    if (commit_issue != 0) {
        throw std::runtime_error(
            "fused optimizer produced a non-finite weight or moment");
    }
    if (grad_norm_out) {
        float sq = 0.0f;
        const cudaError_t status =
            cudaMemcpy(&sq, d_accum, sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("fused optimizer grad-norm download failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(float));
        *grad_norm_out =
            acc_scale * std::sqrt(std::max(sq, 0.0f));
    }
    for (auto* p : active) {
        p->mark_updated();
        optimizer_commit_fault_point();
    }
    trainer.global_step_count = next_step;
    return true;
}

static bool launch_optimizer_step_deterministic_multi_tensor(
    Trainer& trainer, const std::vector<Parameter*>& params,
    float current_learning_rate, float bc1, float bc2,
    int* device_commit_issue, OptimizerCommitGuard& commit_guard,
    std::vector<Parameter*>& gpu_updates) {
    if (!determinism::deterministic_reductions_enabled() ||
        !deterministic_multi_tensor_optimizer_enabled() ||
        !gpu_custom_kernels_supported() || device_commit_issue == nullptr) {
        return false;
    }

    std::vector<Parameter*> active;
    active.reserve(params.size());
    for (Parameter* parameter : params) {
        if (!parameter || !parameter->has_gradient()) {
            continue;
        }
        const auto moment = trainer.m_state.find(parameter);
        const auto variance = trainer.v_state.find(parameter);
        if (parameter->data.get_device() != Device::GPU ||
            parameter->grad.get_device() != Device::GPU ||
            moment == trainer.m_state.end() ||
            variance == trainer.v_state.end() ||
            !can_use_gpu_optimizer(
                *parameter, moment->second, variance->second)) {
            return false;
        }
        active.push_back(parameter);
    }
    if (active.empty() ||
        active.size() >= static_cast<size_t>(
                             std::numeric_limits<int>::max())) {
        return false;
    }

    const int count = static_cast<int>(active.size());
    const bool chunked =
        optimizer_policy::deterministic_adamw_chunked_enabled();
    const uint32_t chunk_elements =
        chunked
            ? optimizer_policy::deterministic_adamw_chunk_elements()
            : 0u;

    thread_local std::vector<unsigned long long> tensor_elements;
    tensor_elements.resize(static_cast<size_t>(count));
    unsigned long long total = 0;
    size_t chunk_count_size = 0;
    for (int index = 0; index < count; ++index) {
        Parameter* parameter = active[static_cast<size_t>(index)];
        const auto elements = static_cast<unsigned long long>(
            parameter->data.size);
        if (total >
            std::numeric_limits<unsigned long long>::max() - elements) {
            throw std::overflow_error(
                "Deterministic optimizer element count overflow");
        }
        tensor_elements[static_cast<size_t>(index)] = elements;
        total += elements;
        if (chunked && elements != 0) {
            const unsigned long long chunks_for_tensor =
                1u + (elements - 1u) / chunk_elements;
            if (chunks_for_tensor >
                    static_cast<unsigned long long>(
                        std::numeric_limits<int>::max()) ||
                chunk_count_size >
                    static_cast<size_t>(std::numeric_limits<int>::max()) -
                        static_cast<size_t>(chunks_for_tensor)) {
                throw std::overflow_error(
                    "Deterministic AdamW chunk count exceeds grid indexing");
            }
            chunk_count_size += static_cast<size_t>(chunks_for_tensor);
        }
    }
    if (total == 0) return false;
    const int chunk_count = static_cast<int>(chunk_count_size);

    const size_t pointer_count = checked_size_multiply(
        static_cast<size_t>(count), 4u,
        "Deterministic optimizer pointer count overflow");
    const size_t pointer_bytes = checked_size_multiply(
        pointer_count, sizeof(float*),
        "Deterministic optimizer pointer metadata overflow");
    const size_t offset_bytes = checked_size_multiply(
        static_cast<size_t>(count) + 1u,
        sizeof(unsigned long long),
        "Deterministic optimizer offset metadata overflow");
    const size_t decay_bytes = static_cast<size_t>(count);
    const size_t offsets_end = checked_size_add(
        pointer_bytes, offset_bytes,
        "Deterministic optimizer metadata size overflow");
    const size_t decay_end = checked_size_add(
        offsets_end, decay_bytes,
        "Deterministic optimizer metadata size overflow");
    const size_t chunk_offset = checked_size_align(
        decay_end, alignof(NsosMultiTensorChunk),
        "Deterministic optimizer chunk alignment overflow");
    const size_t chunk_bytes = checked_size_multiply(
        chunk_count_size, sizeof(NsosMultiTensorChunk),
        "Deterministic optimizer chunk metadata overflow");
    const size_t metadata_bytes = checked_size_add(
        chunk_offset, chunk_bytes,
        "Deterministic optimizer metadata size overflow");

    thread_local std::vector<float*> host_weights;
    thread_local std::vector<float*> host_gradients;
    thread_local std::vector<float*> host_moments;
    thread_local std::vector<float*> host_variances;
    thread_local std::vector<unsigned long long> host_offsets;
    thread_local std::vector<unsigned char> host_decay_flags;
    thread_local std::vector<NsosMultiTensorChunk> host_chunks;
    host_weights.resize(static_cast<size_t>(count));
    host_gradients.resize(static_cast<size_t>(count));
    host_moments.resize(static_cast<size_t>(count));
    host_variances.resize(static_cast<size_t>(count));
    host_offsets.resize(static_cast<size_t>(count) + 1u);
    host_decay_flags.resize(static_cast<size_t>(count));
    host_chunks.resize(chunk_count_size);
    thread_local std::vector<float> host_learning_rates;
    host_learning_rates.resize(static_cast<size_t>(count));

    unsigned long long running_offset = 0;
    size_t next_chunk = 0;
    for (int index = 0; index < count; ++index) {
        Parameter* parameter = active[static_cast<size_t>(index)];
        const unsigned long long elements =
            tensor_elements[static_cast<size_t>(index)];
        host_weights[index] = parameter->data.raw_data();
        host_gradients[index] = parameter->grad.raw_data();
        host_moments[index] = trainer.m_state.at(parameter).raw_data();
        host_variances[index] = trainer.v_state.at(parameter).raw_data();
        host_offsets[index] = running_offset;
        running_offset += elements;
        host_decay_flags[index] =
            should_apply_weight_decay(*parameter, trainer) ? 1u : 0u;
        host_learning_rates[static_cast<size_t>(index)] =
            current_learning_rate * trainer.lr_scale_for(parameter);

        if (chunked) {
            for (unsigned long long element_offset = 0;
                 element_offset < elements;
                 element_offset += chunk_elements) {
                const auto remaining = elements - element_offset;
                const auto count_in_chunk = static_cast<uint32_t>(
                    std::min<unsigned long long>(remaining, chunk_elements));
                host_chunks[next_chunk++] = NsosMultiTensorChunk{
                    element_offset, count_in_chunk,
                    static_cast<uint32_t>(index)};
            }
        }
    }
    host_offsets[count] = running_offset;
    if (running_offset != total || next_chunk != chunk_count_size) {
        throw std::logic_error(
            "Deterministic AdamW metadata construction is inconsistent");
    }

    // Assemble the immutable device blob with byte copies. This avoids
    // creating typed objects inside unsigned-char storage on the host while
    // retaining one aligned H2D transfer for the complete static cohort.
    thread_local std::vector<unsigned char> static_staging;
    static_staging.assign(metadata_bytes, 0u);
    size_t pointer_cursor = 0;
    const auto append_pointer_plane = [&](const std::vector<float*>& plane) {
        const size_t bytes = checked_size_multiply(
            plane.size(), sizeof(float*),
            "Deterministic optimizer pointer-plane size overflow");
        std::memcpy(static_staging.data() + pointer_cursor,
                    plane.data(), bytes);
        pointer_cursor += bytes;
    };
    append_pointer_plane(host_weights);
    append_pointer_plane(host_gradients);
    append_pointer_plane(host_moments);
    append_pointer_plane(host_variances);
    if (pointer_cursor != pointer_bytes) {
        throw std::logic_error(
            "Deterministic optimizer pointer metadata is inconsistent");
    }
    std::memcpy(static_staging.data() + pointer_bytes,
                host_offsets.data(), offset_bytes);
    std::memcpy(static_staging.data() + offsets_end,
                host_decay_flags.data(), decay_bytes);
    if (chunk_bytes != 0) {
        std::memcpy(static_staging.data() + chunk_offset,
                    host_chunks.data(), chunk_bytes);
    }

    DeterministicOptimizerWorkspace& workspace =
        deterministic_optimizer_workspace();
    unsigned char* device_metadata =
        workspace.static_metadata.ensure(metadata_bytes);
    float* device_learning_rates =
        workspace.learning_rates.ensure(static_cast<size_t>(count));
    if (device_metadata == nullptr || device_learning_rates == nullptr) {
        throw std::runtime_error(
            "Deterministic multi-tensor optimizer metadata allocation failed");
    }
    const bool metadata_changed =
        workspace.uploaded_static_device != device_metadata ||
        workspace.uploaded_static_metadata != static_staging;
    if (metadata_changed) {
        const cudaError_t status = cudaMemcpy(
            device_metadata, static_staging.data(), metadata_bytes,
            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "Deterministic optimizer metadata upload failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
        record_gpu_transfer(
            Device::GPU, Device::CPU, metadata_bytes);
        workspace.uploaded_static_metadata = static_staging;
        workspace.uploaded_static_device = device_metadata;
    }
    const bool learning_rates_changed =
        workspace.uploaded_learning_rate_device != device_learning_rates ||
        workspace.uploaded_learning_rates != host_learning_rates;
    if (learning_rates_changed) {
        const size_t bytes = checked_size_multiply(
            host_learning_rates.size(), sizeof(float),
            "Deterministic optimizer learning-rate metadata overflow");
        const cudaError_t status = cudaMemcpyAsync(
            device_learning_rates, host_learning_rates.data(), bytes,
            cudaMemcpyHostToDevice, nsos::gpu::current_stream());
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "Deterministic optimizer learning-rate upload failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
        record_gpu_transfer(Device::GPU, Device::CPU, bytes);
        workspace.uploaded_learning_rates = host_learning_rates;
        workspace.uploaded_learning_rate_device = device_learning_rates;
    }

    auto* device_weights =
        reinterpret_cast<float* const*>(device_metadata);
    auto* device_gradients = device_weights + count;
    auto* device_moments = device_weights + 2 * count;
    auto* device_variances = device_weights + 3 * count;
    const auto* device_offsets =
        reinterpret_cast<const unsigned long long*>(
            device_metadata + pointer_bytes);
    const unsigned char* device_decay_flags =
        device_metadata + offsets_end;
    const auto* device_chunks = chunked
        ? reinterpret_cast<const NsosMultiTensorChunk*>(
              device_metadata + chunk_offset)
        : nullptr;

    // All host allocations and metadata transfers precede the commit point.
    // From the launch onward a backend failure may have published a prefix.
    gpu_updates = active;
    commit_guard.arm();
    if (!launch_multi_tensor_adamw_update_deterministic(
        device_weights, device_gradients, device_moments,
        device_variances, device_offsets, device_decay_flags,
        device_learning_rates, count, total, device_chunks, chunk_count,
        trainer.beta1,
        trainer.beta2, bc1, bc2, trainer.eps,
        trainer.weight_decay, device_commit_issue)) {
        throw std::runtime_error(
            "Deterministic multi-tensor AdamW launcher rejected invalid "
            "arguments");
    }
    trainer_check_cuda(
        "launch_multi_tensor_adamw_update_deterministic");
    return true;
}
#endif  // USE_CUDA

class ParameterAuditStepScope {
public:
    ParameterAuditStepScope(Trainer& trainer,
                            const std::vector<Parameter*>& parameters)
        : step_(trainer.global_step_count + 1) {
        if (!trainer.model) {
            return;
        }
        audit_ = trainer.model->audit_collector();
        if (!audit_ || !audit_->enabled()) {
            audit_ = nullptr;
            return;
        }
        activity_audit_=std::make_unique<DeviceGradientAuditScope>();
        audit_->begin_parameter_step(step_, parameters);
        pending_ = true;
    }

    ParameterAuditStepScope(const ParameterAuditStepScope&) = delete;
    ParameterAuditStepScope& operator=(const ParameterAuditStepScope&) = delete;

    ~ParameterAuditStepScope() {
        if (pending_ && audit_) {
            audit_->abort_parameter_step();
        }
    }

    void complete(bool optimizer_applied) {
        if (!pending_ || !audit_) {
            return;
        }
        audit_->complete_parameter_step(step_, optimizer_applied);
        pending_ = false;
    }

private:
    std::unique_ptr<DeviceGradientAuditScope> activity_audit_;
    LayerAuditCollector* audit_ = nullptr;
    int step_ = 0;
    bool pending_ = false;
};

float apply_optimizer_step(Trainer& trainer,
                           const std::vector<Parameter*>& params,
                           int accumulation_steps,
                           float* grad_norm_out = nullptr) {
    trainer.ensure_optimizer_state_usable();
    if (!std::isfinite(trainer.learning_rate) || trainer.learning_rate < 0.0f ||
        !std::isfinite(trainer.beta1) || trainer.beta1 < 0.0f || trainer.beta1 >= 1.0f ||
        !std::isfinite(trainer.beta2) || trainer.beta2 < 0.0f || trainer.beta2 >= 1.0f ||
        !std::isfinite(trainer.eps) || trainer.eps <= 0.0f ||
        !std::isfinite(trainer.weight_decay) || trainer.weight_decay < 0.0f ||
        !std::isfinite(trainer.max_grad_norm) || trainer.max_grad_norm <= 0.0f) {
        throw std::invalid_argument("Trainer optimizer hyperparameters are invalid");
    }
    trainer.last_optimizer_step_skipped = false;
    if (trainer.device_sparse_group_open) {
        gpu::ExecutionContext::Scope sparse_lane(trainer.device_sparse_execution_context());
        if(!trainer.device_sparse_objectives_finalized) {
            const float qat=apply_qat_regularization(trainer,accumulation_steps);
            const float aux=apply_moe_aux_regularization(trainer,accumulation_steps);
            trainer.last_objective_stats.qat_regularization=qat;trainer.last_objective_stats.moe_auxiliary=aux;
            trainer.last_objective_stats.total+=qat+aux;
        }
        training_nan_fault_point(params);
        std::vector<CriticalityMetric> metrics;
        if (criticality_regularizer_enabled() || criticality_lr_enabled()) metrics=measure_active_criticality(trainer);
        apply_composed_criticality_lr_control(trainer,metrics);
        const float criticality_loss=apply_criticality_regularization_gradient(trainer,metrics,accumulation_steps);
        if (trainer.global_step_count == std::numeric_limits<int>::max())
            throw std::overflow_error("Device sparse Adam global step overflow");
        const int next_step=trainer.global_step_count+1;
        const float lr=compute_lr_for_step(trainer,next_step);
        std::vector<GpuSparseAdamSlot> slots;
        const bool use_muon=optimizer_policy::muon_enabled();
        if(use_muon && trainer.learning_rate<=0)
            throw std::invalid_argument("Muon schedule requires a positive auxiliary Adam base learning rate");
        const float muon_lr=use_muon ? optimizer_policy::muon_learning_rate()*(lr/trainer.learning_rate) : 0;
        for(auto* p:params)if(p && p->trainable && p->data.size) {
            const bool matrix=use_muon && muon::hidden_matrix(p->name,p->data.shape.dims);
            slots.push_back({p,lr*trainer.lr_scale_for(p),should_apply_weight_decay(*p, trainer),
                p->has_device_gradient_activity()?false:p->has_gradient()});
            if(matrix){slots.back().algorithm=GpuSparseAlgorithm::MuonNs5Fp32;slots.back().learning_rate=muon_lr*trainer.lr_scale_for(p);}
        }
        ParameterAuditStepScope parameter_audit(trainer,params);
        const bool restore_moments=!trainer.device_sparse_adam;
        if(restore_moments)trainer.device_sparse_adam=std::make_shared<GpuSparseAdam>();
        trainer.device_sparse_adam->configure(slots);
        if(restore_moments && !trainer.m_state.empty()) {
            std::vector<GpuSparseAdamState> states;
            for(const auto& slot:slots) {
                auto* p=slot.parameter; const auto m=trainer.m_state.find(p),v=trainer.v_state.find(p);
                if((m==trainer.m_state.end())!=(v==trainer.v_state.end()))throw std::logic_error("Sparse Adam incomplete restored moments");
                const bool present=m!=trainer.m_state.end();
                states.push_back({p->name,p->version,present,present?m->second:Tensor(),present?v->second:Tensor(),slot.algorithm});
            }
            trainer.device_sparse_adam->restore(states);
        }
        GpuSparseAdamOptions options;
        options.beta1=trainer.beta1;options.beta2=trainer.beta2;
        options.bc1=1.0f-std::pow(trainer.beta1,next_step);options.bc2=1.0f-std::pow(trainer.beta2,next_step);
        options.eps=trainer.eps;options.weight_decay=trainer.weight_decay;options.max_norm=trainer.max_grad_norm;
        options.accumulation_steps=accumulation_steps;options.deterministic=determinism::deterministic_reductions_enabled();
        options.fused_epilogue=optimizer_policy::fused_optimizer_epilogue_enabled();
        options.clear_gradients=options.fused_epilogue;
        // This lane commits one bank transaction; injected interruption is a
        // group boundary before any update, rather than a partial tensor write.
        optimizer_commit_fault_point();
        const auto result=trainer.device_sparse_adam->step(options);
        parameter_audit.complete(result.committed);
        trainer.finish_device_sparse_group(!result.committed);
        if(!result.committed) {
            trainer.last_optimizer_step_skipped=true;
            if(active_loss_scale(trainer)>1.0f)record_loss_scale_overflow(trainer,params);
            else {zero_model_gradients(params);throw std::runtime_error("Device sparse Adam finite gate/overflow rejected group");}
            if(grad_norm_out)*grad_norm_out=std::numeric_limits<float>::infinity();
            return 0.0f;
        }
        trainer.global_step_count=next_step;record_loss_scale_success(trainer);
        if(grad_norm_out)*grad_norm_out=static_cast<float>(result.norm);
        return criticality_loss;
    }
    training_nan_fault_point(params);
    OptimizerPreflight optimizer_preflight =
        prepare_optimizer_state(trainer, params);
    const bool use_criticality_regularizer =
        criticality_regularizer_enabled();
    const bool use_criticality_lr = criticality_lr_enabled();
#ifdef USE_CUDA
    // In fully resident FP32, leave the finite-gate result on device. The
    // ordinary fused lane consumes it in AdamW; the deterministic lane carries
    // it in the existing ordered-norm scalar. Restrict deterministic deferral
    // to accumulation=1 so a rejected step preserves the historical guarantee
    // that no pre-clip gradient scaling occurred.
    const bool deferred_finite_gate =
        active_loss_scale(trainer) <= 1.0f &&
        (!determinism::deterministic_reductions_enabled() ||
         accumulation_steps == 1) &&
        !use_criticality_regularizer &&
        optimizer_update_can_defer_finite_gate(trainer, params);
    const bool deterministic_deferred_finite_gate =
        deferred_finite_gate &&
        determinism::deterministic_reductions_enabled();
#else
    const bool deferred_finite_gate = false;
    const bool deterministic_deferred_finite_gate = false;
#endif
    if (!optimizer_inputs_are_finite(
            params, &trainer, deferred_finite_gate)) {
        ParameterAuditStepScope rejected_parameter_audit(
            trainer, params);
        if (active_loss_scale(trainer) > 1.0f) {
            record_loss_scale_overflow(trainer, params);
            if (grad_norm_out) {
                *grad_norm_out =
                    std::numeric_limits<float>::infinity();
            }
            rejected_parameter_audit.complete(false);
            return 0.0f;
        }
        trainer.last_optimizer_step_skipped = true;
        rejected_parameter_audit.complete(false);
        zero_model_gradients(params);
        throw std::runtime_error(
            "optimizer step rejected: a parameter or gradient contains NaN/Inf");
    }
    std::vector<CriticalityMetric> criticality_metrics;
    if (use_criticality_regularizer || use_criticality_lr) {
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
    if (use_criticality_regularizer &&
        !optimizer_inputs_are_finite(params, &trainer)) {
        ParameterAuditStepScope rejected_parameter_audit(
            trainer, params);
        trainer.last_optimizer_step_skipped = true;
        rejected_parameter_audit.complete(false);
        zero_model_gradients(params);
        throw std::runtime_error(
            "optimizer step rejected: gradient regularization produced "
            "NaN/Inf");
    }
    // Snapshot only after every objective/regularizer has contributed to the
    // gradient. This is the actual optimizer input, so the recorded
    // gradient-to-update relationship is causally meaningful.
    ParameterAuditStepScope parameter_audit(trainer, params);
    OptimizerCommitGuard commit_guard(trainer);
#ifdef USE_CUDA
    if (apply_optimizer_step_fused(trainer, params, accumulation_steps,
                                   grad_norm_out, commit_guard,
                                   deferred_finite_gate &&
                                       !deterministic_deferred_finite_gate)) {
        record_loss_scale_success(trainer);
        parameter_audit.complete(true);
        commit_guard.complete();
        return criticality_loss;
    }
#endif
    scale_gradients(params, 1.0f / std::max(accumulation_steps, 1));
    bool deferred_finite_issue = false;
#ifdef USE_CUDA
    const int* deferred_device_issue =
        deterministic_deferred_finite_gate
            ? finite_issue_accumulator()
            : nullptr;
#else
    const int* deferred_device_issue = nullptr;
#endif
    const float grad_norm = clip_gradients(
        params, trainer.max_grad_norm, deferred_device_issue,
        deterministic_deferred_finite_gate
            ? &deferred_finite_issue
            : nullptr);
    if (deferred_finite_issue) {
        if (grad_norm_out) {
            *grad_norm_out =
                std::numeric_limits<float>::infinity();
        }
        trainer.last_optimizer_step_skipped = true;
        parameter_audit.complete(false);
        zero_model_gradients(params);
        throw std::runtime_error(
            "optimizer step rejected: a parameter or gradient contains "
            "NaN/Inf");
    }
    if (grad_norm_out) {
        *grad_norm_out = grad_norm;
    }

    const int next_step = trainer.global_step_count + 1;
    const float cur_lr =
        compute_lr_for_step(trainer, next_step);
    const float bc1 =
        1.0f - std::pow(trainer.beta1, next_step);
    const float bc2 =
        1.0f - std::pow(trainer.beta2, next_step);
#ifdef USE_CUDA
    std::vector<Parameter*> gpu_updates;
    gpu_updates.reserve(params.size());
    int* d_commit_issue = nullptr;
    for (Parameter* parameter : params) {
        if (!parameter || !parameter->has_gradient() ||
            (trainer.optimizer_state_bits == 4 &&
             parameter->data.get_device() == Device::CPU)) {
            continue;
        }
        const auto m = trainer.m_state.find(parameter);
        const auto v = trainer.v_state.find(parameter);
        if (m != trainer.m_state.end() &&
            v != trainer.v_state.end() &&
            can_use_gpu_optimizer(
                *parameter, m->second, v->second)) {
            d_commit_issue =
                finite_issue_accumulator();
            if (!d_commit_issue) {
                throw std::runtime_error(
                    "Optimizer commit-status allocation failed");
            }
            break;
        }
    }
    if (d_commit_issue) {
        const cudaError_t issue_clear_status =
            cudaMemsetAsync(
                d_commit_issue, 0, sizeof(int), nsos::gpu::current_stream());
        if (issue_clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "optimizer status clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(issue_clear_status));
        }
    }
    const bool deterministic_multi_tensor_update =
        launch_optimizer_step_deterministic_multi_tensor(
            trainer, params, cur_lr, bc1, bc2, d_commit_issue,
            commit_guard, gpu_updates);
#else
    const bool deterministic_multi_tensor_update = false;
#endif

    // From this point onward, any exception means the backend may have
    // published only a prefix of the optimizer cohort.
    if (!deterministic_multi_tensor_update) {
      commit_guard.arm();
      for (auto* p : params) {
        if (!p || !p->has_gradient()) continue;

        // §6 closed loop: effective lr = global lr * per-parameter scale
        // (1.0 when no controller is active).
        const float p_lr = cur_lr * trainer.lr_scale_for(p);

        // 4-bit optimizer states (CPU path).  GPU params fall through to the
        // FP32 launch_adamw_update_kernel path below until the 4-bit CUDA
        // kernel lands.
        if (trainer.optimizer_state_bits == 4 &&
            p->data.get_device() == Device::CPU) {
            apply_adam_step_4bit(
                trainer, p, p_lr, bc1, bc2,
                optimizer_preflight.quant_m_scratch,
                optimizer_preflight.quant_v_scratch);
            continue;
        }

        Tensor& m_tensor = trainer.m_state.at(p);
        Tensor& v_tensor = trainer.v_state.at(p);

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
                should_apply_weight_decay(*p, trainer) ? 1 : 0,
                d_commit_issue);
            trainer_check_cuda("launch_adamw_update_kernel");
            gpu_updates.push_back(p);
            continue;
        }
#endif

        float* w = p->data.data();
        const float* g = p->grad.data();
        float* m = m_tensor.data();
        float* v = v_tensor.data();
        const bool apply_weight_decay =
            trainer.weight_decay > 0.0f &&
            should_apply_weight_decay(*p, trainer);

        for (int i = 0; i < p->data.size; ++i) {
            const float updated_m =
                trainer.beta1 * m[i] +
                (1.0f - trainer.beta1) * g[i];
            const float updated_v =
                trainer.beta2 * v[i] +
                (1.0f - trainer.beta2) * g[i] * g[i];
            const float m_hat = updated_m / bc1;
            const float v_hat = updated_v / bc2;
            float updated_weight = w[i];
            if (apply_weight_decay) {
                updated_weight -=
                    p_lr * trainer.weight_decay *
                    updated_weight;
            }
            updated_weight -=
                p_lr * m_hat /
                (std::sqrt(v_hat) + trainer.eps);
            if (!std::isfinite(updated_m) ||
                !std::isfinite(updated_v) ||
                updated_v < 0.0f ||
                !std::isfinite(updated_weight)) {
                throw std::runtime_error(
                    "optimizer produced a non-finite CPU weight or moment");
            }
            m[i] = updated_m;
            v[i] = updated_v;
            w[i] = updated_weight;
        }
        p->mark_updated();
        optimizer_commit_fault_point();
      }
    }

#ifdef USE_CUDA
    if (!gpu_updates.empty()) {
        int commit_issue = 0;
        const cudaError_t issue_copy_status =
            cudaMemcpy(
                &commit_issue, d_commit_issue, sizeof(int),
                cudaMemcpyDeviceToHost);
        record_gpu_stream_synchronization();
        if (issue_copy_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "optimizer status download failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(issue_copy_status));
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(int));
        if (commit_issue != 0) {
            throw std::runtime_error(
                "optimizer produced a non-finite GPU weight or moment");
        }
        for (Parameter* parameter : gpu_updates) {
            parameter->mark_updated();
            optimizer_commit_fault_point();
        }
    }
#endif
    trainer.global_step_count = next_step;
    record_loss_scale_success(trainer);
    parameter_audit.complete(true);
    commit_guard.complete();
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
        try {
            audit->record_training_step(
                trainer.global_step_count, loss,
                grad_norm, parameter_count);
        } catch (...) {
            // The optimizer cohort is already committed. An audit sink that
            // cannot record the corresponding step makes this run
            // unauditable, so fail closed instead of inviting a duplicate
            // retry of a step the caller may believe did not happen.
            trainer.mark_optimizer_state_poisoned();
            throw;
        }
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

class OptionalObjectiveGpuWorkspace {
public:
    struct Rows {
        int* targets = nullptr;
        float* weights = nullptr;
        float* losses = nullptr;
    };

    ~OptionalObjectiveGpuWorkspace() {
        if (!wait_for_consumer_noexcept() ||
            awaiting_completion_record_ || poisoned_) {
            device_targets_.abandon();
            device_weights_.abandon();
            device_losses_.abandon();
            host_targets_.abandon();
            host_weights_.abandon();
            copy_complete_ = nullptr;
            return;
        }
        if (copy_complete_ != nullptr) {
            const cudaError_t status =
                cudaEventDestroy(copy_complete_);
            if (status != cudaSuccess) {
                (void)cudaGetLastError();
            }
        }
    }

    Rows stage(const std::vector<int>& targets,
               const std::vector<float>& weights) {
        if (targets.empty() || targets.size() != weights.size()) {
            throw std::invalid_argument(
                "optional objective row staging requires aligned non-empty "
                "target and weight planes");
        }
        wait_for_consumer();
        ensure(targets.size());
        std::copy(
            targets.begin(), targets.end(),
            host_targets_.get());
        std::copy(
            weights.begin(), weights.end(),
            host_weights_.get());
        check(cudaMemcpyAsync(
                  device_targets_.get(), host_targets_.get(),
                  targets.size() * sizeof(int),
                  cudaMemcpyHostToDevice, nsos::gpu::current_stream()),
              "optional-objective target upload");
        awaiting_completion_record_ = true;
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            targets.size() * sizeof(int));
        const cudaError_t weight_status =
            cudaMemcpyAsync(
                device_weights_.get(), host_weights_.get(),
                weights.size() * sizeof(float),
                cudaMemcpyHostToDevice, nsos::gpu::current_stream());
        if (weight_status != cudaSuccess) {
            fail_after_unrecorded_work(
                weight_status,
                "optional-objective weight upload");
        }
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            weights.size() * sizeof(float));
        const cudaError_t event_status =
            cudaEventRecord(copy_complete_, nsos::gpu::current_stream());
        if (event_status != cudaSuccess) {
            fail_after_unrecorded_work(
                event_status,
                "optional-objective staging event record");
        }
        awaiting_completion_record_ = false;
        copy_pending_ = true;
        consumer_pending_ = true;
        return {
            device_targets_.get(),
            device_weights_.get(),
            device_losses_.get()};
    }

    void complete_consumer() {
        if (!consumer_pending_) {
            throw std::logic_error(
                "optional-objective GPU consumer completion was not pending");
        }
        consumer_pending_ = false;
        copy_pending_ = false;
    }

    [[noreturn]] void fail_consumer_completion(
        cudaError_t original_status,
        const char* operation) {
        const cudaError_t sync_status =
            cudaStreamSynchronize( nsos::gpu::current_stream());
        record_gpu_stream_synchronization();
        if (sync_status != cudaSuccess) {
            poisoned_ = true;
            throw std::runtime_error(
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(original_status) +
                "; consumer completion recovery also failed: " +
                cudaGetErrorString(sync_status));
        }
        consumer_pending_ = false;
        copy_pending_ = false;
        check(original_status, operation);
        throw std::logic_error(
            "unreachable optional-objective consumer failure path");
    }

private:
    static void check(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
    }

    void wait_for_consumer() {
        if (poisoned_) {
            throw std::runtime_error(
                "optional-objective GPU staging is poisoned after an "
                "unrecoverable completion error");
        }
        if (consumer_pending_) {
            const cudaError_t status =
                cudaStreamSynchronize( nsos::gpu::current_stream());
            record_gpu_stream_synchronization();
            if (status != cudaSuccess) {
                poisoned_ = true;
                check(
                    status,
                    "optional-objective consumer synchronization");
            }
            consumer_pending_ = false;
            copy_pending_ = false;
            return;
        }
        if (!copy_pending_) return;
        cudaError_t status = cudaEventQuery(copy_complete_);
        if (status == cudaErrorNotReady) {
            status = cudaEventSynchronize(copy_complete_);
            record_gpu_stream_synchronization();
        }
        if (status != cudaSuccess) {
            poisoned_ = true;
            check(
                status,
                "optional-objective staging reuse synchronization");
        }
        copy_pending_ = false;
    }

    bool wait_for_consumer_noexcept() noexcept {
        if (consumer_pending_) {
            const cudaError_t status =
                cudaStreamSynchronize( nsos::gpu::current_stream());
            record_gpu_stream_synchronization();
            if (status != cudaSuccess) {
                poisoned_ = true;
                return false;
            }
            consumer_pending_ = false;
            copy_pending_ = false;
        }
        if (copy_pending_ && copy_complete_ != nullptr) {
            cudaError_t status =
                cudaEventQuery(copy_complete_);
            if (status == cudaErrorNotReady) {
                status =
                    cudaEventSynchronize(copy_complete_);
                record_gpu_stream_synchronization();
            }
            if (status != cudaSuccess) {
                poisoned_ = true;
                return false;
            }
            copy_pending_ = false;
        }
        return !poisoned_ && !awaiting_completion_record_;
    }

    [[noreturn]] void fail_after_unrecorded_work(
        cudaError_t original_status,
        const char* operation) {
        const cudaError_t sync_status =
            cudaStreamSynchronize( nsos::gpu::current_stream());
        record_gpu_stream_synchronization();
        if (sync_status != cudaSuccess) {
            poisoned_ = true;
            throw std::runtime_error(
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(original_status) +
                "; completion recovery also failed: " +
                cudaGetErrorString(sync_status));
        }
        awaiting_completion_record_ = false;
        check(original_status, operation);
        throw std::logic_error(
            "unreachable optional-objective staging failure path");
    }

    void ensure(size_t requested) {
        const size_t minimum =
            std::max<size_t>(requested, 256);
        if (!device_targets_.ensure(minimum)) {
            throw std::runtime_error(
                "optional-objective target allocation failed");
        }
        if (!device_weights_.ensure(minimum)) {
            throw std::runtime_error(
                "optional-objective weight allocation failed");
        }
        if (!device_losses_.ensure(2)) {
            throw std::runtime_error(
                "optional-objective loss allocation failed");
        }
        if (!host_targets_.ensure(minimum)) {
            throw std::runtime_error(
                "optional-objective pinned target allocation failed");
        }
        if (!host_weights_.ensure(minimum)) {
            throw std::runtime_error(
                "optional-objective pinned weight allocation failed");
        }
        if (copy_complete_ == nullptr) {
            const cudaError_t status =
                cudaEventCreate(&copy_complete_);
            if (status != cudaSuccess) {
                check(status,
                      "optional-objective staging event creation");
            }
        }
    }

    cuda_detail::DeviceBuffer<int> device_targets_;
    cuda_detail::DeviceBuffer<float> device_weights_;
    cuda_detail::DeviceBuffer<float> device_losses_;
    cuda_detail::PinnedHostBuffer<int> host_targets_;
    cuda_detail::PinnedHostBuffer<float> host_weights_;
    cudaEvent_t copy_complete_ = nullptr;
    bool copy_pending_ = false;
    bool consumer_pending_ = false;
    bool awaiting_completion_record_ = false;
    bool poisoned_ = false;
};

OptionalObjectiveGpuWorkspace& optional_objective_gpu_workspace() {
    thread_local OptionalObjectiveGpuWorkspace workspace;
    return workspace;
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
    const int batch_size = answer_logits.shape[0];
    const int seq_len = answer_logits.shape[1];
    const int vocab = answer_logits.shape[2];
    const float inverse_batch = 1.0f / static_cast<float>(std::max(batch_size, 1));

#ifdef USE_CUDA
    if (!rul_force_host() &&
        answer_grad.get_device() == Device::GPU &&
        probs.get_device() == Device::GPU &&
        gpu_custom_kernels_supported()) {
        std::vector<int> token_plane;
        token_plane.reserve(static_cast<size_t>(batch_size) * seq_len);
        for (const auto& answer_tokens : answer_batch) {
            if (static_cast<int>(answer_tokens.size()) != seq_len) {
                throw std::invalid_argument(
                    "batched repetition-unlikelihood requires equal token "
                    "sequence lengths");
            }
            token_plane.insert(token_plane.end(),
                               answer_tokens.begin(),
                               answer_tokens.end());
        }
        std::vector<float> unused_weights(token_plane.size(), 0.0f);
        auto& objective_workspace =
            optional_objective_gpu_workspace();
        auto staged = objective_workspace.stage(
            token_plane, unused_weights);
        const cudaError_t clear_status =
            cudaMemsetAsync(staged.losses, 0, sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "batched repetition loss clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(clear_status));
        }
        if (!launch_repetition_unlikelihood_masked_kernel(
            staged.losses, answer_grad.raw_data(), probs.raw_data(),
            staged.targets, batch_size, seq_len, vocab,
            scale * inverse_batch, trainer.eos_token_id)) {
            throw std::runtime_error(
                "Batched repetition-unlikelihood launcher rejected invalid "
                "arguments");
        }
        trainer_check_cuda(
            "launch_repetition_unlikelihood_masked_kernel(batch)");
        float loss = 0.0f;
        const cudaError_t download_status =
            cudaMemcpy(&loss, staged.losses, sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (download_status != cudaSuccess) {
            objective_workspace.fail_consumer_completion(
                download_status,
                "batched repetition loss download");
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(float));
        objective_workspace.complete_consumer();
        return loss;
    }
#endif

    Tensor probs_host = probs.get_device() == Device::GPU ? probs.cpu() : probs;
    Tensor grad_host = answer_grad.get_device() == Device::GPU ? answer_grad.cpu() : answer_grad.clone();
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

struct MaskedOptionalObjectiveLosses {
    float repetition_unlikelihood = 0.0f;
    float logit_l2 = 0.0f;
};

MaskedOptionalObjectiveLosses apply_masked_optional_objectives(
    const Trainer& trainer,
    const Tensor& logits,
    const std::vector<int>& token_plane,
    const std::vector<float>& sample_row_weights,
    Tensor& grad) {
    MaskedOptionalObjectiveLosses losses;
    const float repetition_scale =
        std::max(trainer.repetition_unlikelihood_scale, 0.0f);
    const float l2_beta = effective_logit_l2_beta(trainer);
    if ((repetition_scale <= 0.0f && l2_beta <= 0.0f) ||
        logits.size == 0) {
        return losses;
    }
    if (logits.shape != grad.shape ||
        logits.get_device() != grad.get_device() ||
        logits.shape.size() < 2) {
        throw std::invalid_argument(
            "masked optional objectives require logits/gradient shape and "
            "device parity");
    }
    const int vocab = logits.shape.back();
    if (vocab <= 0 || logits.size % vocab != 0 ||
        logits.size / vocab > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "masked optional objective logits shape is invalid");
    }
    const int rows = static_cast<int>(logits.size / vocab);
    if (static_cast<int>(token_plane.size()) != rows ||
        static_cast<int>(sample_row_weights.size()) != rows) {
        throw std::invalid_argument(
            "masked optional objective row plane mismatch");
    }
    const int batch =
        logits.shape.size() == 3 ? logits.shape[0] : 1;
    const int seq =
        logits.shape.size() == 3 ? logits.shape[1] : rows;
    if (batch <= 0 || seq <= 0 ||
        static_cast<int64_t>(batch) * seq != rows) {
        throw std::invalid_argument(
            "masked optional objectives require [batch, seq, vocab] or "
            "[seq, vocab] logits");
    }
    for (int row = 0; row < rows; ++row) {
        const int token = token_plane[static_cast<size_t>(row)];
        const float row_weight =
            sample_row_weights[static_cast<size_t>(row)];
        if (token < -1 || token >= vocab ||
            !std::isfinite(row_weight) || row_weight < 0.0f ||
            (token < 0 && row_weight != 0.0f)) {
            throw std::invalid_argument(
                "masked optional objective row metadata is invalid");
        }
    }

#ifdef USE_CUDA
    if ((repetition_scale <= 0.0f || !rul_force_host()) &&
        logits.get_device() == Device::GPU &&
        gpu_custom_kernels_supported()) {
        auto& objective_workspace =
            optional_objective_gpu_workspace();
        auto staged = objective_workspace.stage(
            token_plane, sample_row_weights);
        const cudaError_t clear_status =
            cudaMemsetAsync(staged.losses, 0, 2 * sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("optional-objective loss clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(clear_status));
        }
        Tensor probabilities;
        if (repetition_scale > 0.0f) {
            probabilities = logits.softmax(-1);
            if (!launch_repetition_unlikelihood_masked_kernel(
                staged.losses, grad.raw_data(), probabilities.raw_data(),
                staged.targets, batch, seq, vocab, repetition_scale,
                trainer.eos_token_id)) {
                throw std::runtime_error(
                    "Masked repetition-unlikelihood launcher rejected "
                    "invalid arguments");
            }
        }
        if (l2_beta > 0.0f) {
            if (!launch_masked_logit_l2_kernel(
                staged.losses + 1, grad.raw_data(), logits.raw_data(),
                staged.weights, rows, vocab, l2_beta)) {
                throw std::runtime_error(
                    "Masked logit-L2 launcher rejected invalid arguments");
            }
        }
        trainer_check_cuda("launch masked optional objectives");
        float host_losses[2] = {0.0f, 0.0f};
        const cudaError_t download_status =
            cudaMemcpy(host_losses, staged.losses, sizeof(host_losses),
                       cudaMemcpyDeviceToHost);
        if (download_status != cudaSuccess) {
            objective_workspace.fail_consumer_completion(
                download_status,
                "optional-objective loss download");
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(host_losses));
        objective_workspace.complete_consumer();
        losses.repetition_unlikelihood = host_losses[0];
        losses.logit_l2 = host_losses[1];
        return losses;
    }
#endif

    Tensor probability_host;
    if (repetition_scale > 0.0f) {
        Tensor probabilities = logits.softmax(-1);
        probability_host =
            probabilities.get_device() == Device::GPU
                ? probabilities.cpu()
                : probabilities;
    }
    Tensor grad_host =
        grad.get_device() == Device::GPU ? grad.cpu() : grad;
    Tensor logits_host =
        logits.get_device() == Device::GPU ? logits.cpu() : logits;
    float* gradient = grad_host.data();
    const float* logit_values = logits_host.data();
    const float* probabilities =
        probability_host.size > 0 ? probability_host.data() : nullptr;
    double repetition_loss = 0.0;
    double l2_loss = 0.0;
    for (int flat_row = 0; flat_row < rows; ++flat_row) {
        const int token = token_plane[static_cast<size_t>(flat_row)];
        const float row_weight =
            sample_row_weights[static_cast<size_t>(flat_row)];
        const int local_row = flat_row % seq;
        if (l2_beta > 0.0f && row_weight > 0.0f) {
            const float gradient_scale =
                l2_beta * row_weight / static_cast<float>(vocab);
            double row_sq = 0.0;
            for (int column = 0; column < vocab; ++column) {
                const size_t index =
                    static_cast<size_t>(flat_row) * vocab + column;
                const float value = logit_values[index];
                gradient[index] += gradient_scale * value;
                row_sq += static_cast<double>(value) * value;
            }
            l2_loss +=
                0.5 * static_cast<double>(l2_beta) * row_weight *
                row_sq / static_cast<double>(vocab);
        }
        if (repetition_scale <= 0.0f || token < 0 || local_row == 0) {
            continue;
        }
        int negative_ids[4];
        int negative_count = 0;
        const int sample_start = flat_row - local_row;
        const int window_start = std::max(0, local_row - 4);
        for (int previous = local_row - 1;
             previous >= window_start; --previous) {
            const int candidate =
                token_plane[static_cast<size_t>(sample_start + previous)];
            if (candidate < 0 || candidate == token ||
                candidate == trainer.eos_token_id) {
                continue;
            }
            bool seen = false;
            for (int index = 0; index < negative_count; ++index) {
                seen = seen || negative_ids[index] == candidate;
            }
            if (!seen && negative_count < 4) {
                negative_ids[negative_count++] = candidate;
            }
        }
        float* row_gradient =
            gradient + static_cast<size_t>(flat_row) * vocab;
        const float* row_probabilities =
            probabilities + static_cast<size_t>(flat_row) * vocab;
        for (int index = 0; index < negative_count; ++index) {
            const int negative = negative_ids[index];
            const float probability = row_probabilities[negative];
            if (probability <= 1e-6f ||
                probability >= 1.0f - 1e-6f) {
                continue;
            }
            const float factor =
                repetition_scale * probability /
                std::max(1.0f - probability, 1e-6f);
            repetition_loss -=
                static_cast<double>(repetition_scale) *
                std::log1p(-static_cast<double>(probability));
            for (int column = 0; column < vocab; ++column) {
                row_gradient[column] -=
                    factor * row_probabilities[column];
            }
            row_gradient[negative] += factor;
        }
    }
    if (grad.get_device() == Device::GPU) {
        grad.copy_from(grad_host.to(Device::GPU));
    }
    losses.repetition_unlikelihood =
        static_cast<float>(repetition_loss);
    losses.logit_l2 = static_cast<float>(l2_loss);
    return losses;
}

float train_supervised_batch_impl(Trainer& trainer,
                                  const std::vector<std::vector<int>>& prompt_batch,
                                  const std::vector<std::vector<int>>& answer_batch) {
    validate_trainer_configuration(trainer);
    throw_if_training_cancelled(trainer);
    if (prompt_batch.empty() || prompt_batch.size() != answer_batch.size()) {
        throw std::runtime_error("train_supervised_batch requires aligned prompt/answer batches");
    }
    require_checkpoint_boundary(trainer);
    long long batch_tokens = 0;
    for (size_t index = 0; index < prompt_batch.size(); ++index) {
        if (prompt_batch[index].empty() || answer_batch[index].empty())
            throw std::runtime_error("train_supervised_batch received empty prompt/answer");
        require_token_capacity(batch_tokens, prompt_batch[index].size());
        batch_tokens += static_cast<long long>(prompt_batch[index].size());
        require_token_capacity(batch_tokens, answer_batch[index].size() - 1);
        batch_tokens += static_cast<long long>(answer_batch[index].size() - 1);
    }
    require_token_capacity(trainer.tokens_processed, static_cast<size_t>(batch_tokens));
    require_token_capacity(trainer.tokens_committed, static_cast<size_t>(batch_tokens));
    // Âncora do wall C++ (NSOS_TRAIN_TIMING): a diferença entre o wall do
    // chamador Python e este wall expõe o custo de binding/conversão de listas.
    const auto tm_call0 = std::chrono::steady_clock::now();
    trainer.last_step_telemetry = TrainingStepTelemetry{};

    auto params = trainable_model_parameters(trainer.model);
    TrainingAttemptGuard attempt(trainer, params);
    const float loss_scale = active_loss_scale(trainer);
    apply_progressive_qat_phase(trainer);
    zero_model_gradients(params);
    trainer.begin_device_sparse_group();
    begin_moe_aux_accumulation(trainer);

    double supervised_loss_sum = 0.0;
    Tensor supervised_loss_device;
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
    // Diagnostic step-timing (NSOS_TRAIN_TIMING=1): localizes where the step
    // time goes (forward / per-sample loss loop / backward / optimizer).
    // GPU boundaries use a reusable default-stream event instead of a
    // device-wide synchronization, so unrelated streams are never drained.
    // Off by default (zero event/synchronization overhead).
    const bool nsos_step_timing = trainer_step_timing_enabled();
    double tm_fwd = 0.0, tm_loss = 0.0, tm_bwd = 0.0, tm_opt = 0.0, tm_gap = 0.0;
    int tm_buckets = 0;
#ifdef USE_CUDA
    DiagnosticStreamFence timing_fence(nsos_step_timing);
#endif
    auto tm_now = [&]() {
#ifdef USE_CUDA
        timing_fence.wait();
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
        throw_if_training_cancelled(trainer);
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
        throw_if_training_cancelled(trainer);
        accumulate_auxiliary_stats(auxiliary_total, bucket_aux);
        Context ctx;
        ctx.abort_signal = trainer.cancellation_signal();
        const auto _tm_fwd0 = tm_now();
        tm_gap += tm_ms(_tm_fwd0 - tm_last).count();
        const bool cce_head = training_policy::head_cce();
        Tensor logits = cce_head
            ? trainer.model->forward_ids_batch_training_hidden(grouped_inputs, &ctx)
            : trainer.model->forward_ids_batch(grouped_inputs, &ctx);
        throw_if_training_cancelled(trainer);
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
                static_cast<float>(grouped_inputs.size()) * loss_scale);
        sparse_selector_weighted_sum +=
            static_cast<double>(selector_loss) * grouped_inputs.size();

        const int batch_size = logits.shape[0];
        const int rows = logits.shape[1];
        const int vocab = trainer.model->model_config().vocab_size;

        // One masked CE plane for the entire heterogeneous bucket. Prompt and
        // padding rows are target=-1/weight=0. Answer weights carry the former
        // per-sample normalization directly.
        std::vector<int> full_targets(
            static_cast<size_t>(batch_size) * rows, -1);
        std::vector<float> full_weights(
            static_cast<size_t>(batch_size) * rows, 0.0f);
        std::vector<float> sample_row_weights(
            static_cast<size_t>(batch_size) * rows, 0.0f);
        for (int batch = 0; batch < batch_size; ++batch) {
            const int answer_start = static_cast<int>(grouped_prompt_lengths[static_cast<size_t>(batch)]) - 1;
            const int answer_rows = static_cast<int>(grouped_answers[static_cast<size_t>(batch)].size());
            const int answer_end = answer_start + answer_rows;
            if (answer_start < 0 || answer_end > rows) {
                throw std::runtime_error("train_supervised answer window out of range");
            }

            const auto& answer_tokens = grouped_answers[static_cast<size_t>(batch)];
            const std::vector<float> row_weights =
                supervised_row_weights(trainer, answer_tokens);
            const float inverse_answer_rows =
                1.0f / static_cast<float>(std::max(answer_rows, 1));
            for (int answer_row = 0; answer_row < answer_rows; ++answer_row) {
                const size_t flat_row =
                    static_cast<size_t>(batch) * rows +
                    static_cast<size_t>(answer_start + answer_row);
                full_targets[flat_row] =
                    answer_tokens[static_cast<size_t>(answer_row)];
                full_weights[flat_row] =
                    row_weights[static_cast<size_t>(answer_row)] *
                    inverse_answer_rows;
                sample_row_weights[flat_row] = inverse_answer_rows;
            }
        }
        Tensor bucket_supervised_loss, full_grad;
        MaskedOptionalObjectiveLosses optional_losses;
        if (cce_head) {
            TiledCrossEntropyOptions options;
            options.targets = full_targets; options.weights = full_weights;
            options.l2_weights = sample_row_weights; options.sequence_length = rows;
            options.eos_token = trainer.eos_token_id;
            options.repetition_scale = std::max(trainer.repetition_unlikelihood_scale, 0.0f);
            options.l2_beta = effective_logit_l2_beta(trainer); options.gradient_scale = loss_scale;
            auto result = trainer.model->tiled_head_loss(options);
            bucket_supervised_loss = result.losses.slice(0, 0, 1);
            full_grad = std::move(result.input_gradient);
            if (options.repetition_scale > 0 || options.l2_beta > 0) {
                Tensor host = result.losses.cpu();
                optional_losses.repetition_unlikelihood = host.data()[1];
                optional_losses.logit_l2 = host.data()[2];
            }
        } else {
            auto loss_grad = logits.reshape({batch_size * rows, vocab})
                .cross_entropy_weighted_masked_sum_device(full_targets, full_weights);
            bucket_supervised_loss = std::move(loss_grad.first);
            full_grad = loss_grad.second.reshape(logits.shape.dims);
            optional_losses = apply_masked_optional_objectives(
                trainer, logits, full_targets, sample_row_weights, full_grad);
        }
        if (supervised_loss_device.size == 0) {
            supervised_loss_device = std::move(bucket_supervised_loss);
        } else {
            supervised_loss_device.add_inplace_(bucket_supervised_loss);
        }
        sample_count += batch_size;

        // Both optional losses now consume the same masked heterogeneous
        // bucket. On GPU this is two kernels and one two-float readback, with no
        // per-sample slicing, D2H tensor copies, or gradient copy-back.
        repetition_loss_sum +=
            optional_losses.repetition_unlikelihood;
        logit_l2_loss_sum += optional_losses.logit_l2;

        const auto _tm_loss1 = tm_now();
        tm_loss += tm_ms(_tm_loss1 - _tm_fwd1).count();
        if (!cce_head) scale_tensor_inplace(full_grad, loss_scale);
        throw_if_training_cancelled(trainer);
        if (cce_head) trainer.model->backward_training_hidden(full_grad, ctx);
        else trainer.model->backward_external(full_grad, ctx);
        throw_if_training_cancelled(trainer);
        const auto _tm_bwd1 = tm_now();
        tm_bwd += tm_ms(_tm_bwd1 - _tm_loss1).count();
        for (const auto& input : grouped_inputs)
            trainer.tokens_processed += static_cast<long long>(input.size());
        tm_last = _tm_bwd1;
    }

    scale_gradients(params, 1.0f / loss_scale);
    const int objective_samples = std::max(sample_count, 1);
    const float qat_loss =
        apply_qat_regularization(trainer, objective_samples);
    const float moe_aux_loss =
        apply_moe_aux_regularization(trainer, objective_samples);
    float grad_norm = 0.0f;
    LayerAuditCollector* step_audit =
        trainer.model ? trainer.model->audit_collector() : nullptr;
    const auto _tm_opt0 = tm_now();
    throw_if_training_cancelled(trainer);
    trainer.pending_accumulated_tokens = batch_tokens;
    const float criticality_loss = apply_optimizer_step(
        trainer, params, std::max(sample_count, 1),
        step_audit && step_audit->enabled() ? &grad_norm : nullptr);
    if (!trainer.last_optimizer_step_skipped) {
        attempt.mark_optimizer_committed();
        trainer.tokens_committed += batch_tokens;
    }
    trainer.pending_accumulated_tokens = 0;
    const auto _tm_opt1 = tm_now();
    if (supervised_loss_device.size != 1) {
        throw std::logic_error(
            "supervised cross-entropy loss accumulator is invalid");
    }
    Tensor supervised_loss_host =
        supervised_loss_device.get_device() == Device::GPU
            ? supervised_loss_device.cpu()
            : supervised_loss_device;
    supervised_loss_sum = supervised_loss_host.data()[0];
    if (nsos_step_timing) {
        const auto _tm_call1 = tm_now();
        tm_opt = tm_ms(_tm_opt1 - _tm_opt0).count();
        const double wall = tm_ms(_tm_call1 - tm_call0).count();
        const double prep = tm_ms(tm_prep_end - tm_call0).count();
        const double gap_tail = tm_ms(_tm_opt0 - tm_last).count();
        const double unacc =
            wall - (prep + tm_gap + tm_fwd + tm_loss + tm_bwd + gap_tail + tm_opt);
        trainer.last_step_telemetry = TrainingStepTelemetry{
            true,
            trainer.global_step_count,
            tm_buckets,
            wall,
            prep,
            tm_gap + gap_tail,
            tm_fwd,
            tm_loss,
            tm_bwd,
            tm_opt,
            unacc};
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
    attempt.commit();
    return objective.total;
}

} // namespace

#ifdef NSOS_ENABLE_TEST_HOOKS
namespace testing {

void set_training_state_stage_failure_countdown(
    long long countdown) {
    if (countdown < 0) {
        throw std::invalid_argument(
            "Training-state failure countdown must be non-negative");
    }
    training_state_stage_failure_countdown.store(
        countdown, std::memory_order_relaxed);
}

void set_training_state_save_failure_before_replace(bool enabled) {
    training_state_save_failure_before_replace.store(
        enabled, std::memory_order_relaxed);
}

void inject_training_nan_before_optimizer() {
    training_nan_before_optimizer.store(
        true, std::memory_order_relaxed);
}

void set_optimizer_commit_failure_countdown(
    long long countdown) {
    if (countdown < 0) {
        throw std::invalid_argument(
            "Optimizer commit failure countdown must be non-negative");
    }
    optimizer_commit_failure_countdown.store(
        countdown, std::memory_order_relaxed);
}

void clear_training_state_stage_failure() {
    training_state_stage_failure_countdown.store(
        -1, std::memory_order_relaxed);
    training_state_save_failure_before_replace.store(
        false, std::memory_order_relaxed);
    training_nan_before_optimizer.store(
        false, std::memory_order_relaxed);
    optimizer_commit_failure_countdown.store(
        -1, std::memory_order_relaxed);
}

}  // namespace testing
#endif

Trainer::Trainer(JambaModel* m, float lr) : model(m), learning_rate(lr) {
    if (model) {
        validate_trainer_configuration(*this);
        model->set_training_mode(true);
    }
    apply_progressive_qat_phase(*this);
}

Trainer::~Trainer() {
    try { finish_device_sparse_group(true); } catch (...) {}
}

TrainingCheckpointSnapshot::TrainingCheckpointSnapshot(
    std::shared_ptr<JambaModel> model,
    std::unique_ptr<Trainer> trainer,
    int global_step)
    : model_(std::move(model)),
      trainer_(std::move(trainer)),
      global_step_(global_step) {}

TrainingCheckpointSnapshot::~TrainingCheckpointSnapshot() = default;

void TrainingCheckpointSnapshot::write(
    const std::string& model_path,
    const std::string& state_path) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (written_) {
        throw std::logic_error(
            "Training checkpoint snapshot is one-shot and was already written");
    }
    if (!model_ || !trainer_ || trainer_->model != model_.get()) {
        throw std::logic_error(
            "Training checkpoint snapshot ownership is invalid");
    }
    model_->save(model_path);
    trainer_->save_training_state(state_path, model_path);
    written_ = true;
}

const RuntimeExecutionIdentity&
Trainer::ensure_execution_identity_locked() const {
    if (!model) {
        throw std::runtime_error(
            "Runtime execution identity requires a model");
    }
    if (portable_checkpoint_snapshot_) {
        if (!execution_identity_.has_value()) {
            throw std::logic_error(
                "Portable checkpoint snapshot lacks its source runtime "
                "identity");
        }
        return *execution_identity_;
    }
    RuntimeExecutionIdentity current =
        capture_runtime_execution_identity(
            *model, optimizer_state_bits,
            dynamic_loss_scaling_enabled,
            gradient_accumulation_steps,
            scheduler_unit == SchedulerUnit::Tokens);
    if (!execution_identity_.has_value()) {
        execution_identity_.emplace(std::move(current));
    } else if (!runtime_execution_identity_equal(
                   *execution_identity_, current)) {
        throw std::runtime_error(
            runtime_execution_identity_mismatch(
                *execution_identity_, current) +
            "; create a new Trainer/run or restore the original runtime "
            "policy before continuing");
    }
    return *execution_identity_;
}

std::shared_ptr<TrainingCheckpointSnapshot>
Trainer::capture_checkpoint_snapshot() const {
    RuntimeExecutionPolicyLease policy_lease;
    if (!model) {
        throw std::runtime_error(
            "Cannot capture a checkpoint without a model");
    }
    ModelConfig snapshot_config;
    std::vector<Parameter*> source_parameters;
    {
        // Capture immutable construction metadata and force the parameter
        // registry to be materialized under the same transaction used by
        // training. The potentially expensive CPU model construction remains
        // outside this short critical section.
        std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
        std::lock_guard<std::recursive_mutex> model_lock(
            model->execution_mutex_);
        ensure_optimizer_state_usable();
        require_checkpoint_boundary(*this);
        snapshot_config = model->model_config();
        source_parameters = model->parameters();
    }
    // Construction happens outside the full clone transaction so random
    // initialization and host allocations never extend the stop-the-world
    // portion. All authoritative bytes are then overwritten transactionally
    // by clone_runtime_state_to.
    const TensorRandomState tensor_random_state =
        capture_tensor_random_state();
    std::shared_ptr<JambaModel> snapshot_model;
    std::unique_ptr<Trainer> snapshot_trainer;
    try {
        snapshot_model = std::make_shared<JambaModel>(
            snapshot_config, Device::CPU);
        snapshot_trainer = std::make_unique<Trainer>(
            snapshot_model.get(), 0.001f);
    } catch (...) {
        restore_tensor_random_state(tensor_random_state);
        throw;
    }
    restore_tensor_random_state(tensor_random_state);
    snapshot_trainer->portable_checkpoint_snapshot_ = true;

    const std::vector<Parameter*> target_parameters =
        snapshot_model->parameters();
    clone_runtime_state_to(
        *snapshot_trainer, source_parameters, target_parameters);
    const int snapshot_step = snapshot_trainer->global_step_count;
    return std::shared_ptr<TrainingCheckpointSnapshot>(
        new TrainingCheckpointSnapshot(
            std::move(snapshot_model), std::move(snapshot_trainer),
            snapshot_step));
}

std::vector<RuntimeExecutionIdentity::Field>
Trainer::execution_identity_fields() const {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> lock(state_mutex_);
    return ensure_execution_identity_locked().fields;
}

std::string Trainer::execution_identity_digest() const {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> lock(state_mutex_);
    return ensure_execution_identity_locked().digest();
}

void Trainer::validate_execution_identity() const {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> lock(state_mutex_);
    (void)ensure_execution_identity_locked();
}

MemorySystem& Trainer::auxiliary_memory_store(int dim, int scope) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (dim <= 0 || dim > 1'048'576) {
        throw std::invalid_argument(
            "Auxiliary memory dimension must be in [1, 1048576]");
    }
    const uint64_t resolved_scope =
        static_cast<uint64_t>(std::max(scope, 0));
    const uint64_t key =
        (resolved_scope << 32) ^ static_cast<uint32_t>(dim);
    std::lock_guard<std::mutex> lock(auxiliary_memory_mutex_);
    const std::string backend_signature =
        std::string(phase_scheduler.auxiliary_oxtamem_enabled ? "1|" : "0|") +
        phase_scheduler.auxiliary_oxtamem_library_path + "|" +
        phase_scheduler.auxiliary_oxtamem_store_path + "|" +
        std::to_string(phase_scheduler.auxiliary_oxtamem_size_mb);
    if (backend_signature != auxiliary_memory_signature_) {
        auxiliary_memory_stores_.clear();
        auxiliary_memory_backend_ready_.clear();
        auxiliary_memory_signature_ = backend_signature;
    }
    const long long signed_key = static_cast<long long>(key);
    auto& slot = auxiliary_memory_stores_[signed_key];
    auto& backend_ready =
        auxiliary_memory_backend_ready_[signed_key];
    if (!slot) {
        slot = std::make_unique<MemorySystem>(dim);
        backend_ready =
            !phase_scheduler.auxiliary_oxtamem_enabled;
    }
    if (!backend_ready &&
        phase_scheduler.auxiliary_oxtamem_enabled) {
        if (phase_scheduler.auxiliary_oxtamem_store_path.empty()) {
            throw std::invalid_argument(
                "Auxiliary OxtaMem requires a non-empty store path");
        }
        const std::string derived_store =
            phase_scheduler.auxiliary_oxtamem_store_path +
            ".scope-" + std::to_string(resolved_scope) +
            ".dim-" + std::to_string(dim);
        if (!slot->enable_oxtamem_store(
                phase_scheduler.auxiliary_oxtamem_library_path,
                derived_store,
                phase_scheduler.auxiliary_oxtamem_size_mb)) {
            const std::string detail = slot->last_persistence_error();
            throw std::runtime_error(
                "Could not enable auxiliary OxtaMem backend" +
                (detail.empty() ? std::string{} : ": " + detail));
        }
        backend_ready = true;
    }
    return *slot;
}

void Trainer::clear_auxiliary_memory() {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    std::lock_guard<std::mutex> lock(auxiliary_memory_mutex_);
    auxiliary_memory_stores_.clear();
    auxiliary_memory_backend_ready_.clear();
    auxiliary_memory_signature_.clear();
}

void Trainer::clone_runtime_state_to(
    Trainer& target,
    const std::vector<Parameter*>& source_parameters,
    const std::vector<Parameter*>& target_parameters) const {
    RuntimeExecutionPolicyLease policy_lease;
    if (&target == this) {
        throw std::invalid_argument(
            "Trainer runtime state cannot be cloned onto itself");
    }
    std::scoped_lock state_locks(
        state_mutex_, target.state_mutex_);
    if (!model || !target.model) {
        throw std::runtime_error(
            "Trainer runtime-state clone requires two models");
    }
    std::scoped_lock model_locks(
        model->execution_mutex_,
        target.model->execution_mutex_);
    ensure_optimizer_state_usable();
    validate_trainer_configuration(*this);
    require_checkpoint_boundary(*this);
    require_checkpoint_boundary(target);
    synchronize_device_sparse_checkpoint();
    const auto staged_progress = capture_training_progress(*this);
    validate_training_progress(staged_progress, global_step_count);
    const RuntimeExecutionIdentity& source_execution_identity =
        ensure_execution_identity_locked();
    RuntimeExecutionIdentity target_execution_identity =
        target.portable_checkpoint_snapshot_
            ? source_execution_identity
            : capture_runtime_execution_identity(
                  *target.model, optimizer_state_bits,
                  dynamic_loss_scaling_enabled,
                  target.gradient_accumulation_steps,
                  target.scheduler_unit == SchedulerUnit::Tokens);
    if (!target.portable_checkpoint_snapshot_ &&
        !runtime_execution_identity_equal(
            source_execution_identity,
            target_execution_identity)) {
        throw std::runtime_error(
            runtime_execution_identity_mismatch(
                source_execution_identity,
                target_execution_identity));
    }
    if (source_parameters.size() !=
        target_parameters.size()) {
        throw std::runtime_error(
            "Trainer runtime-state clone parameter count mismatch");
    }
    if (phase_scheduler.auxiliary_oxtamem_enabled &&
        !target.portable_checkpoint_snapshot_) {
        throw std::runtime_error(
            "Transactional training clone cannot share a durable OxtaMem "
            "arena; run durable auxiliary-memory training through the "
            "single-owner local training workflow");
    }

    std::unordered_map<Parameter*, Parameter*> mapping;
    mapping.reserve(source_parameters.size());
    std::vector<Tensor> staged_parameter_data;
    staged_parameter_data.reserve(source_parameters.size());
    std::vector<Tensor> staged_gradients;
    staged_gradients.reserve(source_parameters.size());
    for (size_t index = 0;
         index < source_parameters.size(); ++index) {
        Parameter* source = source_parameters[index];
        Parameter* destination =
            target_parameters[index];
        if (!source || !destination ||
            source->name != destination->name ||
            source->data.shape !=
                destination->data.shape ||
            (source->data.get_device() !=
                 destination->data.get_device() &&
             !(target.portable_checkpoint_snapshot_ &&
               destination->data.get_device() == Device::CPU)) ||
            !mapping.emplace(
                source, destination).second) {
            throw std::runtime_error(
                "Trainer runtime-state clone parameter identity mismatch");
        }
        staged_parameter_data.push_back(
            source->data.get_device() ==
                    destination->data.get_device()
                ? source->data.clone()
                : source->data.to(destination->data.get_device()));
        if (source->grad.size == 0 ||
            target.portable_checkpoint_snapshot_) {
            staged_gradients.emplace_back();
        } else {
            if (source->grad.shape !=
                    source->data.shape ||
                source->grad.get_device() !=
                    source->data.get_device()) {
                throw std::runtime_error(
                    "Trainer runtime-state clone found an invalid gradient");
            }
            staged_gradients.push_back(
                source->grad.get_device() ==
                        destination->data.get_device()
                    ? source->grad.clone()
                    : source->grad.to(
                          destination->data.get_device()));
        }
    }
    auto mapped_parameter =
        [&](Parameter* source) -> Parameter* {
            const auto found = mapping.find(source);
            if (found == mapping.end()) {
                throw std::runtime_error(
                    "Trainer runtime state references a parameter outside "
                    "the model registry");
            }
            return found->second;
        };

    std::unordered_map<Parameter*, Tensor> staged_m;
    std::unordered_map<Parameter*, Tensor> staged_v;
    std::unordered_map<Parameter*, Quant4OptState>
        staged_quant;
    std::unordered_map<Parameter*, float> staged_crit;
    std::unordered_map<Parameter*, float>
        staged_external_lr;
    std::unordered_map<Parameter*, float>
        staged_criticality_lr;
    staged_m.reserve(m_state.size());
    staged_v.reserve(v_state.size());
    staged_quant.reserve(quant_state.size());
    staged_crit.reserve(crit_g0_state.size());
    staged_external_lr.reserve(external_lr_scale.size());
    staged_criticality_lr.reserve(
        criticality_lr_scale.size());
    for (const auto& [parameter, state] : m_state) {
        if (!fp32_state_is_valid(*this, parameter)) {
            throw std::runtime_error(
                "Trainer runtime-state clone found incomplete FP32 Adam "
                "state");
        }
        const Tensor host =
            state.get_device() == Device::CPU
                ? state
                : state.cpu();
        const float* values = host.data();
        for (int64_t index = 0;
             index < host.size; ++index) {
            if (!std::isfinite(values[index])) {
                throw std::runtime_error(
                    "Trainer runtime-state clone found a non-finite first "
                    "moment");
            }
        }
        Parameter* destination = mapped_parameter(parameter);
        staged_m.emplace(
            destination,
            target.portable_checkpoint_snapshot_
                ? host
                : state.clone());
    }
    for (const auto& [parameter, state] : v_state) {
        if (!fp32_state_is_valid(*this, parameter)) {
            throw std::runtime_error(
                "Trainer runtime-state clone found incomplete FP32 Adam "
                "state");
        }
        const Tensor host =
            state.get_device() == Device::CPU
                ? state
                : state.cpu();
        const float* values = host.data();
        for (int64_t index = 0;
             index < host.size; ++index) {
            if (!std::isfinite(values[index]) ||
                values[index] < 0.0f) {
                throw std::runtime_error(
                    "Trainer runtime-state clone found an invalid second "
                    "moment");
            }
        }
        Parameter* destination = mapped_parameter(parameter);
        staged_v.emplace(
            destination,
            target.portable_checkpoint_snapshot_
                ? host
                : state.clone());
    }
    for (const auto& [parameter, state] : quant_state) {
        if (!quant4_state_is_valid(state, *parameter)) {
            throw std::runtime_error(
                "Trainer runtime-state clone found corrupt 4-bit state");
        }
        staged_quant.emplace(
            mapped_parameter(parameter), state);
    }
    for (const auto& [parameter, value] : crit_g0_state) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error(
                "Trainer runtime-state clone found invalid criticality "
                "state");
        }
        staged_crit.emplace(
            mapped_parameter(parameter), value);
    }
    for (const auto& [parameter, value] :
         external_lr_scale) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error(
                "Trainer runtime-state clone found invalid external LR "
                "state");
        }
        staged_external_lr.emplace(
            mapped_parameter(parameter), value);
    }
    for (const auto& [parameter, value] :
         criticality_lr_scale) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error(
                "Trainer runtime-state clone found invalid criticality LR "
                "state");
        }
        staged_criticality_lr.emplace(
            mapped_parameter(parameter), value);
    }
    if (staged_m.size() != m_state.size() ||
        staged_v.size() != v_state.size() ||
        staged_quant.size() != quant_state.size() ||
        staged_crit.size() != crit_g0_state.size() ||
        staged_external_lr.size() !=
            external_lr_scale.size() ||
        staged_criticality_lr.size() !=
            criticality_lr_scale.size()) {
        throw std::runtime_error(
            "Trainer runtime-state clone contains duplicate parameter "
            "identities");
    }

    TrainPhaseScheduler staged_scheduler =
        phase_scheduler;
    RuntimeExecutionIdentity staged_execution_identity =
        source_execution_identity;
    std::string staged_memory_signature =
        auxiliary_memory_signature_;
    std::unordered_map<
        long long, std::unique_ptr<MemorySystem>>
        staged_memory_stores;
    std::unordered_map<long long, bool>
        staged_memory_ready;
    {
        std::scoped_lock auxiliary_locks(
            auxiliary_memory_mutex_,
            target.auxiliary_memory_mutex_);
        staged_memory_stores.reserve(
            auxiliary_memory_stores_.size());
        staged_memory_ready.reserve(
            auxiliary_memory_stores_.size());
        for (const auto& [key, source_store] :
             auxiliary_memory_stores_) {
            if (!source_store) {
                throw std::runtime_error(
                    "Trainer auxiliary-memory registry contains a null "
                    "store");
            }
            auto destination_store =
                std::make_unique<MemorySystem>(
                    source_store->chunk_dimension());
            destination_store->restore_runtime_clusters(
                source_store->snapshot_runtime_clusters());
            staged_memory_stores.emplace(
                key, std::move(destination_store));
            staged_memory_ready.emplace(key, true);
        }

        static_assert(
            std::is_nothrow_move_assignable_v<Tensor>);
        static_assert(
            std::is_nothrow_move_assignable_v<
                TrainPhaseScheduler>);
        for (size_t index = 0;
             index < target_parameters.size(); ++index) {
            // Preserve target storage identity so tied embedding/head aliases
            // remain intact on the host snapshot.
            target_parameters[index]->copy_data_from(
                staged_parameter_data[index]);
            target_parameters[index]->grad =
                std::move(staged_gradients[index]);
            target_parameters[index]->copy_gradient_activity_from(
                *source_parameters[index]);
            // Freeze metadata belongs to the cloned optimizer cohort too.
            // Portable snapshots must not revive default-trainable parameters.
            target_parameters[index]->trainable =
                source_parameters[index]->trainable;
        }
        target.learning_rate = learning_rate;
        target.beta1 = beta1;
        target.beta2 = beta2;
        target.eps = eps;
        target.weight_decay = weight_decay;
        target.max_grad_norm = max_grad_norm;
        target.min_learning_rate_scale =
            min_learning_rate_scale;
        target.first_token_loss_scale =
            first_token_loss_scale;
        target.eos_loss_scale = eos_loss_scale;
        target.repetition_unlikelihood_scale =
            repetition_unlikelihood_scale;
        target.moe_aux_loss_scale =
            moe_aux_loss_scale;
        target.pantheon_vib_beta =
            pantheon_vib_beta;
        target.logit_l2_beta = logit_l2_beta;
        target.warmup_steps = warmup_steps;
        target.global_step_count =
            global_step_count;
        target.total_training_steps =
            total_training_steps;
        target.eos_token_id = eos_token_id;
        target.optimizer_state_bits =
            optimizer_state_bits;
        target.dynamic_loss_scaling_enabled =
            dynamic_loss_scaling_enabled;
        target.loss_scale = loss_scale;
        target.min_loss_scale = min_loss_scale;
        target.max_loss_scale = max_loss_scale;
        target.loss_scale_growth_factor =
            loss_scale_growth_factor;
        target.loss_scale_backoff_factor =
            loss_scale_backoff_factor;
        target.loss_scale_growth_interval =
            loss_scale_growth_interval;
        target.loss_scale_growth_tracker =
            loss_scale_growth_tracker;
        target.last_optimizer_step_skipped =
            last_optimizer_step_skipped;
        target.phase_scheduler =
            std::move(staged_scheduler);
        target.execution_identity_ =
            std::move(staged_execution_identity);
        target.last_auxiliary_stats =
            last_auxiliary_stats;
        target.last_objective_stats =
            last_objective_stats;
        target.last_step_telemetry =
            last_step_telemetry;
        publish_training_progress(target, staged_progress);
        target.device_sparse_adam.reset();
        target.m_state.swap(staged_m);
        target.v_state.swap(staged_v);
        target.quant_state.swap(staged_quant);
        target.crit_g0_state.swap(staged_crit);
        target.external_lr_scale.swap(
            staged_external_lr);
        target.criticality_lr_scale.swap(
            staged_criticality_lr);
        target.auxiliary_memory_stores_.swap(
            staged_memory_stores);
        target.auxiliary_memory_backend_ready_.swap(
            staged_memory_ready);
        target.auxiliary_memory_signature_.swap(
            staged_memory_signature);
        target.model->training_rng_sequence_ =
            model->training_rng_sequence_;
        target.cancellation_requested_.store(
            false, std::memory_order_release);
        target.optimizer_state_poisoned_.store(
            false, std::memory_order_release);
    }
}

void Trainer::save_training_state(const std::string& state_path,
                                  const std::string& model_path) const {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer has no model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    ensure_optimizer_state_usable();
    validate_trainer_configuration(*this);
    require_checkpoint_boundary(*this);
    const auto progress = capture_training_progress(*this);
    validate_training_progress(progress, global_step_count);
    const RuntimeExecutionIdentity& execution_identity =
        ensure_execution_identity_locked();
    synchronize_device_sparse_checkpoint();
    const std::filesystem::path model_file(model_path);
    ModelSerializer::verify_checkpoint_matches_live_model(
        model, model_file);
    const uint64_t model_bytes =
        static_cast<uint64_t>(std::filesystem::file_size(model_file));
    const std::string model_sha256 =
        integrity::sha256_file(model_file);

    const std::filesystem::path destination(state_path);
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const std::filesystem::path temporary =
        checkpoint_io::unique_temporary_path(destination);

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
        write_training_sha256(output, model_sha256, "model checkpoint");
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
        write_training_pod(
            output,
            static_cast<uint8_t>(
                dynamic_loss_scaling_enabled ? 1 : 0),
            "dynamic loss scaling enabled");
        for (float value : {
                 loss_scale, min_loss_scale, max_loss_scale,
                 loss_scale_growth_factor,
                 loss_scale_backoff_factor}) {
            write_training_pod(output, value, "loss-scale float");
        }
        write_training_pod(
            output,
            static_cast<int32_t>(loss_scale_growth_interval),
            "loss-scale growth interval");
        write_training_pod(
            output,
            static_cast<int32_t>(loss_scale_growth_tracker),
            "loss-scale growth tracker");

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
        write_training_pod(
            output, static_cast<int32_t>(scheduler.activation_precision_bits),
            "QAT activation precision");
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
        write_bool(scheduler.auxiliary_oxtamem_enabled);
        write_training_pod(
            output, scheduler.auxiliary_oxtamem_size_mb,
            "auxiliary OxtaMem size");
        write_training_string(
            output, scheduler.auxiliary_oxtamem_library_path);
        write_training_string(
            output, scheduler.auxiliary_oxtamem_store_path);

        write_runtime_execution_identity(
            output, execution_identity);

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
            const auto m_it = m_state.find(parameter);
            const auto v_it = v_state.find(parameter);
            const bool wants_quant =
                portable_checkpoint_snapshot_
                    ? quant_it != quant_state.end()
                    : (optimizer_state_bits == 4 &&
                       parameter->data.get_device() == Device::CPU);
            if ((m_it == m_state.end()) !=
                (v_it == v_state.end()) ||
                (quant_it != quant_state.end() &&
                 m_it != m_state.end())) {
                throw std::runtime_error(
                    "Optimizer state is incomplete or has conflicting "
                    "precision representations for " + stable_name);
            }
            if ((wants_quant &&
                 m_it != m_state.end()) ||
                (!wants_quant &&
                 quant_it != quant_state.end())) {
                throw std::runtime_error(
                    "Optimizer-state precision changed without a canonical "
                    "conversion step for " + stable_name);
            }
            if (quant_it != quant_state.end()) {
                if (!quant4_state_is_valid(
                        quant_it->second, *parameter)) {
                    throw std::runtime_error(
                        "Quantized optimizer state is corrupt for " +
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
                if (m_it != m_state.end()) {
                    if (!fp32_state_is_valid(
                            *this, parameter)) {
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
            if (has_moments &&
                (!finite_values(m_values) ||
                 !finite_values(v_values, true))) {
                throw std::runtime_error(
                    "Optimizer state contains a non-finite or negative "
                    "moment for " + stable_name);
            }
            if(has_moments && optimizer_policy::muon_enabled() && muon::hidden_matrix(parameter->name,parameter->data.shape.dims) &&
                std::any_of(v_values.begin(),v_values.end(),[](float value){return value!=0;}))
                throw std::runtime_error("Muon checkpoint contains an Adam second moment for "+stable_name);

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

        std::vector<std::pair<int64_t, const MemorySystem*>>
            memory_stores;
        {
            std::lock_guard<std::mutex> memory_lock(
                auxiliary_memory_mutex_);
            if (auxiliary_memory_stores_.size() >
                kMaxTrainingMemoryStores) {
                throw std::runtime_error(
                    "Too many auxiliary memory stores to checkpoint");
            }
            memory_stores.reserve(auxiliary_memory_stores_.size());
            for (const auto& [key, store] :
                 auxiliary_memory_stores_) {
                if (store) {
                    memory_stores.emplace_back(
                        static_cast<int64_t>(key), store.get());
                }
            }
            std::sort(
                memory_stores.begin(), memory_stores.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.first < rhs.first;
                });
            write_training_pod(
                output,
                static_cast<uint32_t>(memory_stores.size()),
                "auxiliary memory store count");
            uint64_t aggregate_items = 0;
            uint64_t aggregate_compressed_bytes = 0;
            for (const auto& [key, store] : memory_stores) {
                const int32_t dimension =
                    store->chunk_dimension();
                write_training_pod(
                    output, key, "auxiliary memory key");
                write_training_pod(
                    output, dimension,
                    "auxiliary memory dimension");
                const auto clusters =
                    store->snapshot_runtime_clusters();
                if (clusters.size() >
                    kMaxTrainingMemoryClusters) {
                    throw std::runtime_error(
                        "Auxiliary memory exceeds checkpoint "
                        "cluster limit");
                }
                write_training_pod(
                    output,
                    static_cast<uint32_t>(clusters.size()),
                    "auxiliary memory cluster count");
                for (const auto& cluster : clusters) {
                    const uint64_t cluster_items =
                        static_cast<uint64_t>(
                            cluster.items.size()) +
                        static_cast<uint64_t>(
                            cluster.compressed_items.size());
                    if (cluster.items.size() >
                            kMaxTrainingMemoryItemsPerCluster ||
                        cluster.compressed_items.size() >
                            kMaxTrainingMemoryItemsPerCluster ||
                        aggregate_items >
                            kMaxTrainingMemoryItems -
                                cluster_items) {
                        throw std::runtime_error(
                            "Auxiliary memory exceeds checkpoint "
                            "item limit");
                    }
                    aggregate_items += cluster_items;
                    const int64_t access_ns =
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                            cluster.last_access.time_since_epoch())
                            .count();
                    write_training_pod(
                        output, access_ns,
                        "auxiliary memory access time");
                    write_training_memory_tensor(
                        output, cluster.centroid, dimension);
                    write_training_pod(
                        output,
                        static_cast<uint32_t>(
                            cluster.items.size()),
                        "auxiliary memory item count");
                    for (const Tensor& item : cluster.items) {
                        write_training_memory_tensor(
                            output, item, dimension);
                    }
                    write_training_pod(
                        output,
                        static_cast<uint32_t>(
                            cluster.compressed_items.size()),
                        "auxiliary compressed item count");
                    for (const auto& encoded :
                         cluster.compressed_items) {
                        if (encoded.size() >
                            16ull * 1024ull * 1024ull) {
                            throw std::runtime_error(
                                "Auxiliary compressed item exceeds "
                                "checkpoint limit");
                        }
                        if (aggregate_compressed_bytes >
                            kMaxTrainingMemoryCompressedBytes -
                                encoded.size()) {
                            throw std::runtime_error(
                                "Auxiliary compressed payload exceeds "
                                "checkpoint limit");
                        }
                        aggregate_compressed_bytes +=
                            encoded.size();
                        write_training_pod(
                            output,
                            static_cast<uint32_t>(encoded.size()),
                            "auxiliary compressed item bytes");
                        output.write(
                            reinterpret_cast<const char*>(
                                encoded.data()),
                            static_cast<std::streamsize>(
                                encoded.size()));
                        if (!output) {
                            throw std::runtime_error(
                                "Auxiliary compressed item write "
                                "failed");
                        }
                    }
                }
            }
        }
        write_training_progress(output, progress);
        output.flush();
        if (!output) throw std::runtime_error("Training-state flush failed");
        output.close();
        if (!output) throw std::runtime_error("Training-state close failed");

        const uint64_t payload_bytes =
            std::filesystem::file_size(temporary);
        const std::string payload_sha256 =
            integrity::sha256_file_prefix(temporary, payload_bytes);
        std::ofstream trailer(temporary,
                              std::ios::binary | std::ios::app);
        if (!trailer) {
            throw std::runtime_error(
                "Cannot append training-state integrity trailer");
        }
        write_training_pod(trailer, kTrainingStateSha256TrailerMagic,
                           "integrity magic");
        write_training_pod(trailer, payload_bytes,
                           "integrity payload bytes");
        write_training_sha256(
            trailer, payload_sha256, "integrity trailer");
        trailer.flush();
        if (!trailer) {
            throw std::runtime_error(
                "Training-state integrity trailer flush failed");
        }
        trailer.close();
        if (!trailer) {
            throw std::runtime_error(
                "Training-state integrity trailer close failed");
        }
        checkpoint_io::flush_file(temporary);
        if (static_cast<uint64_t>(
                std::filesystem::file_size(model_file)) !=
                model_bytes ||
            integrity::sha256_file(model_file) !=
                model_sha256) {
            throw std::runtime_error(
                "Model checkpoint changed while its training-state sidecar "
                "was being written");
        }
        ModelSerializer::verify_checkpoint_matches_live_model(
            model, model_file);
        training_state_save_fault_point();
        checkpoint_io::atomic_replace(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void Trainer::load_training_state(const std::string& state_path,
                                  const std::string& model_path,
                                  bool allow_legacy_runtime_identity,
                                  bool allow_legacy_progress_state) {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer has no model");
    const bool recovering_poison =
        optimizer_state_poisoned();
    if (recovering_poison &&
        (device_sparse_group_open || !device_moe_groups.empty())) {
        throw OptimizerStatePoisonedException(
            "Poisoned optimizer has an unclosed device owner; recover with "
            "a fresh model and Trainer loaded from the exact model checkpoint "
            "and its matching training-state sidecar");
    }
    require_checkpoint_boundary(*this);
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    std::ifstream input(state_path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open training state");
    const uint32_t magic = read_training_pod<uint32_t>(input, "magic");
    const uint32_t version = read_training_pod<uint32_t>(input, "version");
    if (magic != kTrainingStateMagic ||
        version < kTrainingStateLegacyVersion ||
        version > kTrainingStateVersion) {
        throw std::runtime_error("Unsupported or corrupt training-state header");
    }
    verify_training_state_integrity(
        std::filesystem::path(state_path), version);
    if (version < 11u) {
        if (!allow_legacy_progress_state)
            throw std::runtime_error("Legacy training state has no token/progress record; explicit progress migration required");
        if (scheduler_unit != SchedulerUnit::Steps || (version < 10u && gradient_accumulation_steps != 1))
            throw std::runtime_error("Legacy training state cannot prove token scheduler/accumulation history; resume refused");
    }
    const uint32_t fingerprint =
        read_training_pod<uint32_t>(input, "architecture fingerprint");
    if (fingerprint != ModelSerializer::architecture_fingerprint(model)) {
        throw std::runtime_error(
            "Training-state architecture fingerprint does not match model");
    }
    const uint64_t expected_model_bytes =
        read_training_pod<uint64_t>(input, "model byte count");
    const std::filesystem::path model_file(model_path);
    uint64_t actual_model_bytes =
        static_cast<uint64_t>(std::filesystem::file_size(model_file));
    bool model_digest_matches = false;
    if (version >= 9u) {
        const std::string expected_model_sha256 =
            read_training_sha256(input, "model checkpoint");
        model_digest_matches =
            expected_model_sha256 ==
            integrity::sha256_file(model_file);
    } else {
        const uint64_t expected_model_hash =
            read_training_pod<uint64_t>(input, "model hash");
        uint64_t legacy_model_bytes = 0;
        const uint64_t actual_model_hash =
            training_state_file_hash(
                model_file, legacy_model_bytes);
        actual_model_bytes = legacy_model_bytes;
        model_digest_matches =
            expected_model_hash == actual_model_hash;
    }
    if (expected_model_bytes != actual_model_bytes ||
        !model_digest_matches) {
        throw std::runtime_error(
            "Training state belongs to a different model checkpoint");
    }
    if (version >= 9u || recovering_poison) {
        ModelSerializer::verify_checkpoint_matches_live_model(
            model, model_file);
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
    if (version >= 4u) {
        const uint8_t enabled = read_training_pod<uint8_t>(
            input, "dynamic loss scaling enabled");
        if (enabled > 1) {
            throw std::runtime_error(
                "Invalid dynamic loss-scaling boolean");
        }
        metadata.dynamic_loss_scaling_enabled = enabled != 0;
        metadata.loss_scale =
            read_training_pod<float>(input, "loss scale");
        metadata.min_loss_scale =
            read_training_pod<float>(input, "minimum loss scale");
        metadata.max_loss_scale =
            read_training_pod<float>(input, "maximum loss scale");
        metadata.loss_scale_growth_factor =
            read_training_pod<float>(input, "loss-scale growth factor");
        metadata.loss_scale_backoff_factor =
            read_training_pod<float>(input, "loss-scale backoff factor");
        metadata.loss_scale_growth_interval =
            read_training_pod<int32_t>(
                input, "loss-scale growth interval");
        metadata.loss_scale_growth_tracker =
            read_training_pod<int32_t>(
                input, "loss-scale growth tracker");
    }
    if (metadata.learning_rate < 0.0f ||
        metadata.beta1 < 0.0f || metadata.beta1 >= 1.0f ||
        metadata.beta2 < 0.0f || metadata.beta2 >= 1.0f ||
        metadata.eps <= 0.0f ||
        metadata.weight_decay < 0.0f ||
        metadata.max_grad_norm <= 0.0f ||
        metadata.min_learning_rate_scale < 0.0f ||
        metadata.min_learning_rate_scale > 1.0f ||
        metadata.first_token_loss_scale < 0.0f ||
        metadata.eos_loss_scale < 0.0f ||
        metadata.repetition_unlikelihood_scale < 0.0f ||
        metadata.moe_aux_loss_scale < 0.0f ||
        metadata.pantheon_vib_beta < 0.0f ||
        metadata.logit_l2_beta < 0.0f ||
        (metadata.logit_l2_beta != 0.0f &&
         metadata.pantheon_vib_beta != 0.0f &&
         metadata.logit_l2_beta != metadata.pantheon_vib_beta) ||
        metadata.eos_token_id < 0 ||
        metadata.eos_token_id >= model->model_config().vocab_size ||
        metadata.global_step_count < 0 ||
        metadata.global_step_count == std::numeric_limits<int32_t>::max() ||
        metadata.warmup_steps < 0 ||
        metadata.total_training_steps < 0 ||
        (metadata.optimizer_state_bits != 4 &&
         metadata.optimizer_state_bits != 32) ||
        !std::isfinite(metadata.loss_scale) ||
        !std::isfinite(metadata.min_loss_scale) ||
        !std::isfinite(metadata.max_loss_scale) ||
        !std::isfinite(metadata.loss_scale_growth_factor) ||
        !std::isfinite(metadata.loss_scale_backoff_factor) ||
        metadata.min_loss_scale < 1.0f ||
        metadata.max_loss_scale < metadata.min_loss_scale ||
        metadata.loss_scale < metadata.min_loss_scale ||
        metadata.loss_scale > metadata.max_loss_scale ||
        metadata.loss_scale_growth_factor < 1.0f ||
        metadata.loss_scale_backoff_factor <= 0.0f ||
        metadata.loss_scale_backoff_factor >= 1.0f ||
        metadata.loss_scale_growth_interval <= 0 ||
        metadata.loss_scale_growth_tracker < 0 ||
        metadata.loss_scale_growth_tracker >=
            metadata.loss_scale_growth_interval) {
        throw std::runtime_error(
            "Invalid trainer hyperparameter or loss-scale metadata");
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
    scheduler.activation_precision_bits =
        version >= 3u
            ? read_training_pod<int32_t>(input, "QAT activation precision")
            : 8;
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
    if (version >= 6u) {
        scheduler.auxiliary_oxtamem_enabled = read_bool();
        scheduler.auxiliary_oxtamem_size_mb =
            read_training_pod<uint64_t>(
                input, "auxiliary OxtaMem size");
        scheduler.auxiliary_oxtamem_library_path =
            read_training_string(input);
        scheduler.auxiliary_oxtamem_store_path =
            read_training_string(input);
    }
    try {
        validate_phase_scheduler_configuration(scheduler);
    } catch (const std::invalid_argument& error) {
        throw std::runtime_error(
            std::string("Invalid phase scheduler metadata: ") +
            error.what());
    }

    RuntimeExecutionIdentity current_execution_identity =
        capture_runtime_execution_identity(
            *model, metadata.optimizer_state_bits,
            metadata.dynamic_loss_scaling_enabled,
            gradient_accumulation_steps,
            scheduler_unit == SchedulerUnit::Tokens);
    RuntimeExecutionIdentity staged_execution_identity;
    if (version >= 10u) {
        staged_execution_identity =
            read_runtime_execution_identity(input);
        if (!runtime_execution_identity_equal(
                staged_execution_identity,
                current_execution_identity)) {
            throw std::runtime_error(
                runtime_execution_identity_mismatch(
                    staged_execution_identity,
                    current_execution_identity) +
                "; resume refused before optimizer state staging");
        }
    } else {
        if (!allow_legacy_runtime_identity) {
            throw std::runtime_error(
                "Training state predates the versioned runtime execution "
                "identity. Resume is fail-closed because precision/backend/"
                "optimizer compatibility cannot be proven. Re-run with an "
                "explicit legacy-identity migration only after verifying the "
                "original runtime policy.");
        }
        if(optimizer_policy::muon_enabled())
            throw std::runtime_error("Muon rejects legacy identity migration: Adam moments cannot be reinterpreted as Muon momentum");
        staged_execution_identity =
            std::move(current_execution_identity);
    }
    if (execution_identity_.has_value() &&
        !runtime_execution_identity_equal(
            *execution_identity_, staged_execution_identity)) {
        throw std::runtime_error(
            runtime_execution_identity_mismatch(
                *execution_identity_,
                staged_execution_identity) +
            "; the Trainer was already sealed under another policy");
    }

    const auto parameters =
        stable_training_parameters(model, version < 7u);
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
                if(optimizer_policy::muon_enabled() && muon::hidden_matrix(parameter->name,parameter->data.shape.dims) && record.v[i]!=0)
                    throw std::runtime_error("Muon sidecar contains an Adam second moment");
            }
        }
        records.push_back(std::move(record));
    }
    std::vector<TrainingMemoryRecord> memory_records;
    if (version >= 8u) {
        const uint32_t store_count =
            read_training_pod<uint32_t>(
                input, "auxiliary memory store count");
        if (store_count > kMaxTrainingMemoryStores) {
            throw std::runtime_error(
                "Training state exceeds auxiliary-memory store limit");
        }
        memory_records.reserve(store_count);
        std::unordered_map<int64_t, bool> seen_memory_keys;
        uint64_t aggregate_items = 0;
        uint64_t aggregate_compressed_bytes = 0;
        for (uint32_t store_index = 0;
             store_index < store_count; ++store_index) {
            TrainingMemoryRecord record;
            record.key = read_training_pod<int64_t>(
                input, "auxiliary memory key");
            record.dimension = read_training_pod<int32_t>(
                input, "auxiliary memory dimension");
            if (record.key < 0 || record.dimension <= 0 ||
                record.dimension > 1'048'576 ||
                static_cast<uint32_t>(
                    static_cast<uint64_t>(record.key)) !=
                    static_cast<uint32_t>(record.dimension) ||
                (static_cast<uint64_t>(record.key) >> 32) >
                    static_cast<uint64_t>(
                        std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error(
                    "Invalid auxiliary-memory store identity");
            }
            if (!seen_memory_keys.emplace(record.key, true).second) {
                throw std::runtime_error(
                    "Duplicate auxiliary-memory store identity");
            }
            const uint32_t cluster_count =
                read_training_pod<uint32_t>(
                    input, "auxiliary memory cluster count");
            if (cluster_count > kMaxTrainingMemoryClusters) {
                throw std::runtime_error(
                    "Training state exceeds auxiliary-memory cluster limit");
            }
            record.clusters.reserve(cluster_count);
            for (uint32_t cluster_index = 0;
                 cluster_index < cluster_count; ++cluster_index) {
                MemorySystem::Cluster cluster;
                const int64_t access_ns =
                    read_training_pod<int64_t>(
                        input, "auxiliary memory access time");
                cluster.last_access =
                    std::chrono::system_clock::time_point(
                        std::chrono::duration_cast<
                            std::chrono::system_clock::duration>(
                            std::chrono::nanoseconds(access_ns)));
                cluster.centroid =
                    read_training_memory_tensor(
                        input, record.dimension);
                const uint32_t item_count =
                    read_training_pod<uint32_t>(
                        input, "auxiliary memory item count");
                if (item_count >
                        kMaxTrainingMemoryItemsPerCluster ||
                    aggregate_items >
                        kMaxTrainingMemoryItems - item_count) {
                    throw std::runtime_error(
                        "Training state exceeds auxiliary-memory item limit");
                }
                aggregate_items += item_count;
                cluster.items.reserve(item_count);
                for (uint32_t item_index = 0;
                     item_index < item_count; ++item_index) {
                    cluster.items.push_back(
                        read_training_memory_tensor(
                            input, record.dimension));
                }
                const uint32_t compressed_count =
                    read_training_pod<uint32_t>(
                        input,
                        "auxiliary compressed item count");
                if (compressed_count >
                        kMaxTrainingMemoryItemsPerCluster ||
                    aggregate_items >
                        kMaxTrainingMemoryItems -
                            compressed_count) {
                    throw std::runtime_error(
                        "Training state exceeds auxiliary-memory item limit");
                }
                aggregate_items += compressed_count;
                cluster.compressed_items.reserve(
                    compressed_count);
                for (uint32_t item_index = 0;
                     item_index < compressed_count; ++item_index) {
                    const uint32_t byte_count =
                        read_training_pod<uint32_t>(
                            input,
                            "auxiliary compressed item bytes");
                    if (byte_count >
                            16ull * 1024ull * 1024ull ||
                        aggregate_compressed_bytes >
                            kMaxTrainingMemoryCompressedBytes -
                                byte_count) {
                        throw std::runtime_error(
                            "Training state exceeds auxiliary-memory "
                            "compressed payload limit");
                    }
                    aggregate_compressed_bytes += byte_count;
                    std::vector<uint8_t> encoded(byte_count);
                    if (byte_count > 0) {
                        input.read(
                            reinterpret_cast<char*>(
                                encoded.data()),
                            static_cast<std::streamsize>(
                                byte_count));
                        if (!input) {
                            throw std::runtime_error(
                                "Training-state truncated in "
                                "auxiliary compressed item");
                        }
                    }
                    cluster.compressed_items.push_back(
                        std::move(encoded));
                }
                record.clusters.push_back(std::move(cluster));
            }
            memory_records.push_back(std::move(record));
        }
    }
    TrainingProgressState staged_progress;
    if (version >= 11u) {
        staged_progress = read_training_progress(input);
        validate_training_progress(staged_progress, metadata.global_step_count);
        if (staged_progress.scheduler_unit != scheduler_unit ||
            staged_progress.gradient_accumulation_steps != gradient_accumulation_steps)
            throw std::runtime_error("Training progress declared scheduler/A mismatch; resume refused");
    } else {
        // Explicit step-only migration. Counts below measure only work since
        // migration and are marked incomplete; never fabricate lost history.
        staged_progress.scheduler_unit = SchedulerUnit::Steps;
        staged_progress.gradient_accumulation_steps = gradient_accumulation_steps;
        staged_progress.token_counters_complete = false;
        staged_progress.last_accumulation_steps = gradient_accumulation_steps;
    }

    if (version >= 5u) {
        const bool sha256_trailer = version >= 9u;
        const uint64_t trailer_bytes =
            sha256_trailer ? kTrainingStateSha256TrailerBytes
                           : kTrainingStateLegacyTrailerBytes;
        const uint32_t trailer_magic =
            read_training_pod<uint32_t>(input, "integrity magic");
        const uint64_t payload_bytes =
            read_training_pod<uint64_t>(
                input, "integrity payload bytes");
        const uint64_t total_bytes =
            std::filesystem::file_size(state_path);
        bool valid_trailer = false;
        if (sha256_trailer) {
            const std::string payload_sha256 =
                read_training_sha256(input, "integrity SHA-256");
            valid_trailer =
                trailer_magic ==
                    kTrainingStateSha256TrailerMagic &&
                payload_bytes == total_bytes - trailer_bytes &&
                integrity::sha256_file_prefix(
                    state_path, payload_bytes) == payload_sha256;
        } else {
            const uint64_t payload_hash =
                read_training_pod<uint64_t>(
                    input, "integrity hash");
            valid_trailer =
                trailer_magic == kTrainingStateTrailerMagic &&
                payload_bytes == total_bytes - trailer_bytes &&
                training_state_prefix_hash(
                    state_path, payload_bytes) == payload_hash;
        }
        if (!valid_trailer) {
            throw std::runtime_error(
                "Training-state integrity trailer mismatch");
        }
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error(
            "Training state has trailing payload");
    }
    if (!input.eof()) {
        throw std::runtime_error("Training-state read failed before EOF");
    }

    std::unordered_map<long long, std::unique_ptr<MemorySystem>>
        staged_memory_stores;
    std::unordered_map<long long, bool>
        staged_memory_backend_ready;
    staged_memory_stores.reserve(memory_records.size());
    staged_memory_backend_ready.reserve(memory_records.size());
    for (const TrainingMemoryRecord& record : memory_records) {
        training_state_stage_fault_point();
        auto store =
            std::make_unique<MemorySystem>(record.dimension);
        store->restore_runtime_clusters(record.clusters);
        const long long key =
            static_cast<long long>(record.key);
        staged_memory_stores.emplace(key, std::move(store));
        staged_memory_backend_ready.emplace(
            key,
            !metadata.phase_scheduler.auxiliary_oxtamem_enabled);
    }

    // Stage every optimizer allocation and map node before mutating the live
    // Trainer. This includes device transfers: a host/device OOM or copy
    // failure at any parameter leaves metadata, moments, scheduler, auxiliary
    // memory and RNG byte-for-byte at their pre-load state.
    std::unordered_map<Parameter*, Tensor> staged_m_state;
    std::unordered_map<Parameter*, Tensor> staged_v_state;
    std::unordered_map<Parameter*, Quant4OptState>
        staged_quant_state;
    std::unordered_map<Parameter*, float>
        staged_crit_g0_state;
    std::unordered_map<Parameter*, float>
        staged_external_lr_scale;
    std::unordered_map<Parameter*, float>
        staged_criticality_lr_scale;
    staged_m_state.reserve(records.size());
    staged_v_state.reserve(records.size());
    staged_quant_state.reserve(records.size());
    staged_crit_g0_state.reserve(records.size());
    staged_external_lr_scale.reserve(records.size());
    staged_criticality_lr_scale.reserve(records.size());
    for (const auto& record : records) {
        Parameter* parameter = record.parameter;
        if (record.has_moments) {
            training_state_stage_fault_point();
            if (metadata.optimizer_state_bits == 4 &&
                parameter->data.get_device() == Device::CPU) {
                Quant4OptState state;
                quant4_store_m(
                    record.m.data(), parameter->data.size, state);
                int rows = 0;
                int cols = 0;
                quant4_matrix_shape(
                    *parameter, rows, cols);
                quant4_store_v(
                    record.v.data(), parameter->data.size,
                    rows, cols, state);
                if (!quant4_state_is_valid(
                        state, *parameter)) {
                    throw std::logic_error(
                        "Restored 4-bit optimizer state has an invalid "
                        "layout");
                }
                staged_quant_state.emplace(
                    parameter, std::move(state));
            } else {
                Tensor m_host =
                    Tensor::from_blob(
                        const_cast<float*>(record.m.data()),
                        parameter->data.shape.dims,
                        Device::CPU)
                        .clone();
                Tensor v_host =
                    Tensor::from_blob(
                        const_cast<float*>(record.v.data()),
                        parameter->data.shape.dims,
                        Device::CPU)
                        .clone();
                const Device target =
                    parameter->data.get_device();
                Tensor m_tensor =
                    target == Device::CPU
                        ? std::move(m_host)
                        : m_host.to(target);
                Tensor v_tensor =
                    target == Device::CPU
                        ? std::move(v_host)
                        : v_host.to(target);
                staged_m_state.emplace(
                    parameter, std::move(m_tensor));
                staged_v_state.emplace(
                    parameter, std::move(v_tensor));
            }
        }
        if (record.has_criticality) {
            staged_crit_g0_state.emplace(
                parameter, record.criticality);
        }
        if (record.has_external_lr_scale) {
            staged_external_lr_scale.emplace(
                parameter, record.external_lr_scale);
        }
        if (record.has_criticality_lr_scale) {
            staged_criticality_lr_scale.emplace(
                parameter, record.criticality_lr_scale);
        }
    }
    std::string staged_memory_signature =
        std::string(
            metadata.phase_scheduler.auxiliary_oxtamem_enabled
                ? "1|"
                : "0|") +
        metadata.phase_scheduler.auxiliary_oxtamem_library_path +
        "|" +
        metadata.phase_scheduler.auxiliary_oxtamem_store_path +
        "|" +
        std::to_string(
            metadata.phase_scheduler.auxiliary_oxtamem_size_mb);
    static_assert(
        std::is_nothrow_move_assignable_v<TrainPhaseScheduler>);

    // Commit is allocation-free after the whole sidecar, model digest, every
    // tensor and every container node have validated and staged.
    std::unique_lock<std::mutex> memory_commit_lock(
        auxiliary_memory_mutex_);
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
    dynamic_loss_scaling_enabled =
        metadata.dynamic_loss_scaling_enabled;
    loss_scale = metadata.loss_scale;
    min_loss_scale = metadata.min_loss_scale;
    max_loss_scale = metadata.max_loss_scale;
    loss_scale_growth_factor =
        metadata.loss_scale_growth_factor;
    loss_scale_backoff_factor =
        metadata.loss_scale_backoff_factor;
    loss_scale_growth_interval =
        metadata.loss_scale_growth_interval;
    loss_scale_growth_tracker =
        metadata.loss_scale_growth_tracker;
    publish_training_progress(*this, staged_progress);
    phase_scheduler = std::move(metadata.phase_scheduler);
    execution_identity_ =
        std::move(staged_execution_identity);
    auxiliary_memory_stores_.swap(
        staged_memory_stores);
    auxiliary_memory_backend_ready_.swap(
        staged_memory_backend_ready);
    auxiliary_memory_signature_.swap(
        staged_memory_signature);
    device_sparse_adam.reset();
    m_state.swap(staged_m_state);
    v_state.swap(staged_v_state);
    quant_state.swap(staged_quant_state);
    crit_g0_state.swap(staged_crit_g0_state);
    external_lr_scale.swap(staged_external_lr_scale);
    criticality_lr_scale.swap(
        staged_criticality_lr_scale);
    model->training_rng_sequence_ = rng_sequence;
    optimizer_state_poisoned_.store(
        false, std::memory_order_release);
}

gpu::ExecutionContext& Trainer::device_sparse_execution_context() const {
    if(!model)throw std::logic_error("Device sparse lane requires model ownership");
    return model->execution_context_;
}
void Trainer::begin_device_sparse_group() {
    if(!optimizer_policy::device_sparse_adam_enabled())return;
    if(device_sparse_group_open)throw std::logic_error("Nested device sparse accumulation group");
    const bool allow_dense=optimizer_policy::dense_device_optimizer_enabled();
    const bool has_moe=model && std::any_of(model->layers.begin(),model->layers.end(),[](const auto& block){return block && block->uses_moe();});
    if(optimizer_policy::legacy_device_sparse_adam_enabled() && !has_moe)
        throw std::invalid_argument("NSOS_MOE_DEVICE_ADAM=1 requires at least one MoE block; use the fused epilogue policy for dense Adam");
    if(!model || !gpu_custom_kernels_supported() ||
        ((!allow_dense || has_moe) && (!training_policy::grouped_moe_training() || !training_policy::ordered_moe())))
        throw std::invalid_argument("NSOS_MOE_DEVICE_ADAM requires grouped, ordered GPU MoE");
    if (optimizer_state_bits != 32)
        throw std::invalid_argument("NSOS_MOE_DEVICE_ADAM requires FP32 moments; quantized states use the legacy policy");
    const auto& identity = ensure_execution_identity_locked();
    const auto policy = std::find(identity.fields.begin(), identity.fields.end(),
        RuntimeExecutionIdentity::Field{"optimizer.sparse_gradient_policy", "explicit_group_contribution_device_v1"});
    if (policy == identity.fields.end())
        throw std::logic_error("Integrate device sparse runtime identity before enabling NSOS_MOE_DEVICE_ADAM");
    gpu::ExecutionContext::Scope lane(device_sparse_execution_context());
    model->set_training_mode(true);
    const auto params=trainable_model_parameters(model);
    attention_training::reset_status(model->parameters());
    for(auto* p:params)if(p && p->data.get_device()!=Device::GPU)
        throw std::invalid_argument("Device sparse Adam requires an entirely GPU parameter registry");
    try {
        for(auto& block:model->layers)if(block && block->uses_moe()) {
            std::vector<BitLinear*> up,down;
            for(auto& op:block->expert_gate_up)up.push_back(op.get());
            for(auto& op:block->expert_down)down.push_back(op.get());
            auto group=std::make_shared<GpuMoeTraining>();
            group->begin_device_accumulation(up,down);
            group->device_activity()->gradient_scale=1.0f/active_loss_scale(*this);
            device_moe_groups.push_back(std::move(group));
        }
        if(device_moe_groups.empty() && !allow_dense)throw std::invalid_argument("Device sparse Adam policy requires at least one MoE block");
        device_sparse_group_open=true;device_sparse_objectives_finalized=false;
    } catch(...) {finish_device_sparse_group(true);throw;}
}

void Trainer::finish_device_sparse_group(bool abort, bool materialize_for_audit) {
    if (device_moe_groups.empty()) {
        if(abort && device_sparse_group_open) {
            gpu::ExecutionContext::Scope lane(device_sparse_execution_context());
            if(device_sparse_adam)device_sparse_adam->abort();
            attention_training::reset_status(model->parameters());
        }
        device_sparse_group_open = false;
        if (abort) {
            pending_accumulation_microbatches = 0;
            pending_accumulated_tokens = 0;
            pending_supervised_loss_sum = Tensor{};
        }
        return;
    }
    gpu::ExecutionContext::Scope lane(device_sparse_execution_context());
    if(abort && device_sparse_adam)device_sparse_adam->abort();
    if(abort)attention_training::reset_status(model->parameters());
    // Producer owns the consumed tape; its matching group owns pre-bindings.
    // Keep both alive through ordered finish and detach exactly once.
    for(const auto& group:device_moe_groups) {
        auto producer=group->device_activity()->producer();
        (producer?producer:group)->finish_device_accumulation(abort,materialize_for_audit);
    }
    device_moe_groups.clear();device_sparse_group_open=false;
    if(abort){pending_accumulation_microbatches=0;pending_accumulated_tokens=0;pending_supervised_loss_sum=Tensor{};}
}

void Trainer::synchronize_device_sparse_checkpoint() const {
    if(device_sparse_group_open)throw std::logic_error("Checkpoint/clone requires a finished device sparse group");
    if(!device_sparse_adam)return;
    gpu::ExecutionContext::Scope lane(device_sparse_execution_context());
    const auto states=device_sparse_adam->snapshot();
    auto next_m=m_state,next_v=v_state;
    const auto params=trainable_model_parameters(model);
    for(const auto& state:states) {
        auto found=std::find_if(params.begin(),params.end(),[&](Parameter* p){return p && p->name==state.name;});
        if(found==params.end())throw std::logic_error("Sparse Adam checkpoint registry changed");
        if(state.initialized){next_m.insert_or_assign(*found,state.m);next_v.insert_or_assign(*found,state.v);}
        else {next_m.erase(*found);next_v.erase(*found);}
    }
    m_state.swap(next_m);v_state.swap(next_v);
}

void Trainer::request_cancellation() noexcept {
    cancellation_requested_.store(
        true, std::memory_order_release);
}

void Trainer::clear_cancellation() noexcept {
    cancellation_requested_.store(
        false, std::memory_order_release);
}

bool Trainer::cancellation_requested() const noexcept {
    return cancellation_requested_.load(
        std::memory_order_acquire);
}

bool Trainer::optimizer_state_poisoned() const noexcept {
    return optimizer_state_poisoned_.load(
        std::memory_order_acquire);
}

void Trainer::ensure_optimizer_state_usable() const {
    if (optimizer_state_poisoned()) {
        if (device_sparse_group_open || !device_moe_groups.empty()) {
            throw OptimizerStatePoisonedException(
                "Poisoned optimizer has an unclosed device owner; recover with "
                "a fresh model and Trainer loaded from the exact model checkpoint "
                "and its matching training-state sidecar");
        }
        throw OptimizerStatePoisonedException(
            "Trainer optimizer state is fail-stop poisoned after an "
            "ambiguous commit failure; reload the exact model checkpoint "
            "and then its matching training-state sidecar before continuing");
    }
}

void Trainer::mark_optimizer_state_poisoned() noexcept {
    optimizer_state_poisoned_.store(
        true, std::memory_order_release);
}

void Trainer::configure_progressive_qat(const TrainPhaseScheduler& scheduler) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer has no model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    validate_phase_scheduler_configuration(scheduler);
    phase_scheduler = scheduler;
    apply_progressive_qat_phase(*this);
}

bool Trainer::progressive_qat_active() const {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    const auto effective = resolve_effective_qat_schedule(*this);
    return phase_scheduler.progressive_qat_enabled &&
           global_step_count >= effective.qat_start_step;
}

void Trainer::set_lr_scale_by_name(const std::string& name, float scale) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer has no model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    if (name.empty() || !std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument(
            "Parameter LR scale requires a non-empty name and finite scale > 0");
    }
    bool matched = false;
    for (auto* p : trainable_model_parameters(model)) {
        if (p && p->name == name) {
            external_lr_scale[p] = scale;
            matched = true;
        }
    }
    if (!matched) {
        throw std::invalid_argument("Unknown parameter LR-scale name: " + name);
    }
}

void Trainer::clear_lr_scales() {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    external_lr_scale.clear();
}

float Trainer::accumulate_gradients_impl(
    const std::vector<int>& tokens,
    const std::vector<int>& targets,
    Tensor* deferred_supervised_loss) {
    const auto tm_call0 = std::chrono::steady_clock::now();
    const bool timing_enabled = trainer_step_timing_enabled();
#ifdef USE_CUDA
    DiagnosticStreamFence timing_fence(timing_enabled);
#endif
    auto tm_now = [&]() {
#ifdef USE_CUDA
        timing_fence.wait();
#endif
        return std::chrono::steady_clock::now();
    };
    using tm_ms = std::chrono::duration<double, std::milli>;
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    ensure_optimizer_state_usable();
    validate_trainer_configuration(*this);
    (void)ensure_execution_identity_locked();
    throw_if_training_cancelled(*this);
    model->set_training_mode(true);

    const std::vector<int> inputs = make_inputs(tokens, targets);
    const std::vector<int> resolved_targets = make_targets(tokens, targets);

    auto params = trainable_model_parameters(model);
    TrainingAttemptGuard attempt(*this, params);
    const float loss_scale = active_loss_scale(*this);
    apply_progressive_qat_phase(*this);
    // Zera apenas ao abrir um grupo de acumulação.  Com
    // pending_accumulation_microbatches > 0 este microbatch soma sobre os
    // gradientes já computados, que é o ponto da acumulação; o caminho
    // legado (train_step e accumulate_gradients) sempre chega aqui com o
    // contador em zero e portanto mantém o comportamento anterior.
    if (pending_accumulation_microbatches == 0) {
        bool fused_reset=false;
        if(device_sparse_adam && optimizer_policy::fused_optimizer_epilogue_enabled()) {
            gpu::ExecutionContext::Scope lane(device_sparse_execution_context());
            fused_reset=device_sparse_adam->consume_fused_gradient_reset(params);
        }
        if(!fused_reset)zero_model_gradients(params);
        begin_device_sparse_group();
        begin_moe_aux_accumulation(*this);
    } else if (!device_sparse_group_open) {
        begin_moe_aux_accumulation(*this);
    }
    // Dense stored sums were unscaled at the previous microbatch boundary.
    // Restore their scale before any next forward/selector/backward adds a
    // scaled contribution, so the common unscale preserves both terms.
    // scale_gradients skips device-activity tensors: their producer already
    // unscales each contribution when committing its sparse gradient bank.
    if (pending_accumulation_microbatches > 0) {
        scale_gradients(params, loss_scale);
    }
    last_step_telemetry = TrainingStepTelemetry{};

    model->reset_session();
    Context ctx;
    ctx.abort_signal = cancellation_signal();
    const auto tm_prep_end = tm_now();
    const auto tm_fwd0 = tm_prep_end;
    const bool cce_head = training_policy::head_cce();
    Tensor logits = cce_head ? model->forward_ids_training_hidden(inputs, &ctx)
                             : model->forward_ids(inputs, &ctx);
    throw_if_training_cancelled(*this);
    const auto tm_fwd1 = tm_now();
    // SSA learned block-selector distillation (no-op unless sparse attention is on).
    const float selector_loss =
        model->accumulate_sparse_selector_grads(loss_scale);
    Tensor supervised_loss_device, grad;
    float logit_l2_loss = 0.0f;
    if (cce_head) {
        TiledCrossEntropyOptions options;
        options.targets = resolved_targets;
        options.weights.assign(resolved_targets.size(), 1.0f / resolved_targets.size());
        options.l2_weights = options.weights;
        options.l2_beta = effective_logit_l2_beta(*this); options.gradient_scale = loss_scale;
        auto result = model->tiled_head_loss(options);
        supervised_loss_device = result.losses.slice(0, 0, 1);
        grad = std::move(result.input_gradient);
        if (options.l2_beta > 0) logit_l2_loss = result.losses.cpu().data()[2];
    } else {
        auto loss_grad = logits.cross_entropy_device(resolved_targets);
        supervised_loss_device = std::move(loss_grad.first); grad = std::move(loss_grad.second);
        logit_l2_loss = add_logit_l2_objective(logits, effective_logit_l2_beta(*this), grad);
    }

    // ── Pantheon VIB-style L2 regularizer on logits ──────────────────────
    // When logit_l2_beta > 0, add beta * 0.5 * mean(logits^2) to the
    // loss and the corresponding gradient term (beta * logits / N) to the
    // grad tensor before backward.  This is a degenerate VIB compression
    // (variational layer not needed); pulls logits toward zero while CE
    // still pulls them toward correct targets.  See PANTHEON_VALIDATION_REPORT.
    if (!cce_head) scale_tensor_inplace(grad, loss_scale);
    throw_if_training_cancelled(*this);
    const auto tm_loss1 = tm_now();
    if (cce_head) model->backward_training_hidden(grad, ctx);
    else model->backward_external(grad, ctx);
    throw_if_training_cancelled(*this);
    scale_gradients(params, 1.0f / loss_scale);
    const bool finalize_objectives_now = !device_sparse_group_open || deferred_supervised_loss == nullptr;
    const float qat_loss = finalize_objectives_now ? apply_qat_regularization(*this, 1) : 0.0f;
    const float moe_aux_loss = finalize_objectives_now ? apply_moe_aux_regularization(*this, 1) : 0.0f;
    const auto tm_bwd1 = tm_now();
    float supervised_loss = 0.0f;
    if (deferred_supervised_loss != nullptr) {
        *deferred_supervised_loss = std::move(supervised_loss_device);
    } else {
        Tensor supervised_loss_host =
            supervised_loss_device.get_device() == Device::GPU
                ? supervised_loss_device.cpu()
                : supervised_loss_device;
        if (supervised_loss_host.size != 1) {
            throw std::logic_error(
                "cross_entropy_device returned an invalid loss scalar");
        }
        supervised_loss = supervised_loss_host.data()[0];
    }
    last_objective_stats = TrainingObjectiveStats{};
    last_objective_stats.supervised_cross_entropy = supervised_loss;
    last_objective_stats.logit_l2 = logit_l2_loss;
    last_objective_stats.sparse_selector = selector_loss;
    last_objective_stats.qat_regularization = qat_loss;
    last_objective_stats.moe_auxiliary = moe_aux_loss;
    last_objective_stats.total = supervised_loss + logit_l2_loss +
                                 selector_loss + qat_loss + moe_aux_loss;
    if (timing_enabled) {
        const double wall = tm_ms(tm_bwd1 - tm_call0).count();
        const double prep = tm_ms(tm_prep_end - tm_call0).count();
        const double forward = tm_ms(tm_fwd1 - tm_fwd0).count();
        const double loss = tm_ms(tm_loss1 - tm_fwd1).count();
        const double backward = tm_ms(tm_bwd1 - tm_loss1).count();
        last_step_telemetry = TrainingStepTelemetry{
            true,
            global_step_count,
            1,
            wall,
            prep,
            0.0,
            forward,
            loss,
            backward,
            0.0,
            wall - (prep + forward + loss + backward)};
    }
    attempt.commit();
    return last_objective_stats.total;
}

float Trainer::accumulate_gradients(
    const std::vector<int>& tokens,
    const std::vector<int>& targets) {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(model->execution_mutex_);
    ensure_optimizer_state_usable();
    // Sondagem independente: sempre parte de gradientes limpos.  Reiniciar o
    // contador garante que uma sonda disparada no meio de um grupo não seja
    // confundida com um microbatch dele.
    pending_accumulation_microbatches = 0;
    pending_accumulated_tokens = 0;
    if(device_sparse_group_open)finish_device_sparse_group(true);
    const float result=accumulate_gradients_impl(tokens,targets,nullptr);
    if(device_sparse_group_open)finish_device_sparse_group(false,true);
    return result;
}

float Trainer::accumulate_microbatch(
    const std::vector<int>& tokens,
    const std::vector<int>& targets) {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(model->execution_mutex_);
    if (tokens.empty()) {
        throw std::invalid_argument(
            "accumulate_microbatch requires a non-empty token window");
    }
    if (pending_accumulation_microbatches ==
        std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "gradient accumulation group overflowed its microbatch counter");
    }
    require_token_capacity(tokens_processed, tokens.size());
    require_token_capacity(pending_accumulated_tokens, tokens.size());
    require_token_capacity(tokens_committed, tokens.size());
    // Caminho diferido, idêntico ao de train_step: a loss permanece no device
    // e só é lida no commit.  Ler aqui deslocaria o ponto de sincronização e
    // quebraria a equivalência bit a bit em A=1.
    Tensor deferred_loss;
    const float loss = accumulate_gradients_impl(tokens, targets,
                                                 &deferred_loss);
    if (deferred_loss.size != 1) {
        throw std::logic_error(
            "deferred supervised loss scalar is invalid");
    }
    if (pending_supervised_loss_sum.size == 0) {
        pending_supervised_loss_sum = std::move(deferred_loss);
    } else {
        pending_supervised_loss_sum =
            pending_supervised_loss_sum.add(deferred_loss);
    }
    // Contabilizado depois do forward: um microbatch que lançou não entra no
    // grupo e não avança contador algum.
    pending_accumulation_microbatches += 1;
    pending_accumulated_tokens += static_cast<long long>(tokens.size());
    tokens_processed += static_cast<long long>(tokens.size());
    return loss;
}

void Trainer::abort_gradient_accumulation() {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    // Descarta um grupo parcial. Os gradientes dos microbatches já computados
    // contaminariam a próxima tentativa, e nem global_step_count nem
    // tokens_committed podem avançar por trabalho que nunca foi commitado.
    pending_accumulation_microbatches = 0;
    pending_accumulated_tokens = 0;
    pending_supervised_loss_sum = Tensor{};
    if (!model) return;
    std::lock_guard<std::recursive_mutex> model_lock(model->execution_mutex_);
    auto params = trainable_model_parameters(model);
    bool cleanup_succeeded = cancel_moe_aux_accumulation(*this);
    try { finish_device_sparse_group(true); } catch (...) { cleanup_succeeded = false; }
    try {
        zero_model_gradients(params);
    } catch (...) {
        cleanup_succeeded = false;
    }
    try {
        model->reset_session();
    } catch (...) {
        cleanup_succeeded = false;
    }
    if (!cleanup_succeeded) {
        // Não foi possível provar que o estado voltou a ser utilizável.
        mark_optimizer_state_poisoned();
        throw std::runtime_error(
            "gradient accumulation abort could not restore a clean state");
    }
}

float Trainer::commit_optimizer_step(int accumulation_steps) {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(model->execution_mutex_);
    if (accumulation_steps <= 0) {
        throw std::invalid_argument(
            "commit_optimizer_step requires accumulation_steps >= 1");
    }
    if (accumulation_steps != gradient_accumulation_steps) {
        // O contrato declarado é o que a identidade de runtime grava e o que o
        // checkpoint promete. Commitar com um A diferente produziria uma
        // trajetória que a identidade não descreve.
        throw std::invalid_argument(
            "commit_optimizer_step was asked for accumulation_steps=" +
            std::to_string(accumulation_steps) +
            " but the declared gradient_accumulation_steps is " +
            std::to_string(gradient_accumulation_steps));
    }
    if (pending_accumulated_tokens < 0)
        throw std::invalid_argument("Negative pending token count");
    require_token_capacity(tokens_committed, static_cast<size_t>(pending_accumulated_tokens));
    if (pending_accumulation_microbatches != accumulation_steps) {
        // Fail-closed: commitar com contagem diferente da acumulada aplicaria
        // o divisor errado e falsificaria silenciosamente o batch efetivo.
        throw std::invalid_argument(
            "commit_optimizer_step expected " +
            std::to_string(accumulation_steps) +
            " accumulated microbatches but found " +
            std::to_string(pending_accumulation_microbatches));
    }
    ensure_optimizer_state_usable();

    auto params = trainable_model_parameters(model);
    TrainingAttemptGuard attempt(*this, params);
    throw_if_training_cancelled(*this);

    const Tensor group_supervised_loss = pending_supervised_loss_sum;
    float grad_norm = 0.0f;
    if (device_sparse_group_open) {
        const float qat_loss = apply_qat_regularization(*this, accumulation_steps);
        const float moe_loss = apply_moe_aux_regularization(*this, accumulation_steps);
        last_objective_stats.qat_regularization = qat_loss;
        last_objective_stats.moe_auxiliary = moe_loss;
        last_objective_stats.total += qat_loss + moe_loss;
    }
    const float criticality_loss =
        apply_optimizer_step(*this, params, accumulation_steps, &grad_norm);
    if (!(*this).last_optimizer_step_skipped) attempt.mark_optimizer_committed();

    // Consome a loss do grupo na mesma ordem em que train_step consome a sua:
    // leitura para host só agora, depois do optimizer step.
    if (group_supervised_loss.size != 1) {
        throw std::logic_error(
            "accumulated supervised loss scalar is invalid");
    }
    Tensor loss_host =
        group_supervised_loss.get_device() == Device::GPU
            ? group_supervised_loss.cpu()
            : group_supervised_loss;
    const float supervised_loss =
        loss_host.data()[0] / static_cast<float>(accumulation_steps);
    pending_supervised_loss_sum = Tensor{};
    last_objective_stats.supervised_cross_entropy = supervised_loss;
    last_objective_stats.total += supervised_loss;

    // Regime efetivo do passo: `clip_gradients` devolve a norma medida antes
    // do corte, então a norma aplicada é ela mesma ou o teto.
    last_grad_norm_pre_clip = grad_norm;
    last_update_was_clipped = grad_norm > max_grad_norm;
    last_grad_norm_post_clip =
        last_update_was_clipped ? max_grad_norm : grad_norm;
    last_accumulation_steps = accumulation_steps;

    last_objective_stats.criticality_regularization = criticality_loss;
    last_objective_stats.total += criticality_loss;

    // Só agora o grupo vira trajetória: tokens commitados avançam juntos com o
    // optimizer step, nunca antes dele.
    if (!last_optimizer_step_skipped) tokens_committed += pending_accumulated_tokens;
    pending_accumulated_tokens = 0;
    pending_accumulation_microbatches = 0;

    record_training_audit_step(*this, last_objective_stats.total, grad_norm,
                               params.size());
    attempt.commit();
    return last_objective_stats.total;
}

float Trainer::train_step(const std::vector<int>& tokens,
                          const std::vector<int>& targets) {
    const auto tm_call0 = std::chrono::steady_clock::now();
    const bool timing_enabled = trainer_step_timing_enabled();
#ifdef USE_CUDA
    DiagnosticStreamFence timing_fence(timing_enabled);
#endif
    auto tm_now = [&]() {
#ifdef USE_CUDA
        timing_fence.wait();
#endif
        return std::chrono::steady_clock::now();
    };
    using tm_ms = std::chrono::duration<double, std::milli>;
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    require_token_capacity(tokens_processed, tokens.size());
    require_token_capacity(tokens_committed, tokens.size());
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    ensure_optimizer_state_usable();
    // Gradients only (no weight update), then the optimizer step + audit.
    // Factored so the criticality instrument can read gradients without
    // mutating weights (Trainer::accumulate_gradients).  Behavior identical
    // to the previous monolithic train_step.
    validate_trainer_configuration(*this);
    if (gradient_accumulation_steps != 1) {
        throw std::invalid_argument(
            "train_step requires declared accumulation A=1; use "
            "accumulate_microbatch/commit_optimizer_step for A>1");
    }
    require_checkpoint_boundary(*this);
    Tensor deferred_supervised_loss;
    (void)accumulate_gradients_impl(
        tokens, targets, &deferred_supervised_loss);
    const auto tm_opt0 = tm_now();
    auto params = trainable_model_parameters(model);
    TrainingAttemptGuard attempt(*this, params);
    throw_if_training_cancelled(*this);
    float grad_norm = 0.0f;
    // The token scheduler uses the position after this candidate update.
    // This remains pending until the global optimizer gate commits it.
    pending_accumulated_tokens = static_cast<long long>(tokens.size());
    const float criticality_loss =
        apply_optimizer_step(
            *this, params, 1,
            &grad_norm);
    if (!(*this).last_optimizer_step_skipped) attempt.mark_optimizer_committed();
    // O caminho legado é um grupo de acumulação de tamanho 1: os contadores de
    // token avançam aqui para que scheduler, checkpoint e telemetria enxerguem
    // a mesma grandeza nos dois caminhos.
    tokens_processed += static_cast<long long>(tokens.size());
    if (!last_optimizer_step_skipped) tokens_committed += static_cast<long long>(tokens.size());
    pending_accumulated_tokens = 0;
    last_grad_norm_pre_clip = grad_norm;
    last_update_was_clipped = grad_norm > max_grad_norm;
    last_grad_norm_post_clip =
        last_update_was_clipped ? max_grad_norm : grad_norm;
    last_accumulation_steps = 1;
    const auto tm_opt1 = tm_now();
    if (deferred_supervised_loss.size != 1) {
        throw std::logic_error(
            "deferred supervised loss scalar is invalid");
    }
    Tensor supervised_loss_host =
        deferred_supervised_loss.get_device() == Device::GPU
            ? deferred_supervised_loss.cpu()
            : deferred_supervised_loss;
    const float supervised_loss = supervised_loss_host.data()[0];
    last_objective_stats.supervised_cross_entropy = supervised_loss;
    last_objective_stats.total += supervised_loss;
    last_objective_stats.criticality_regularization = criticality_loss;
    last_objective_stats.total += criticality_loss;
    record_training_audit_step(*this, last_objective_stats.total, grad_norm,
                               params.size());
    if (timing_enabled) {
        const auto tm_call1 = tm_now();
        const double optimizer = tm_ms(tm_opt1 - tm_opt0).count();
        const double wall = tm_ms(tm_call1 - tm_call0).count();
        last_step_telemetry.optimizer_ms = optimizer;
        last_step_telemetry.wall_ms = wall;
        last_step_telemetry.global_step = global_step_count;
        last_step_telemetry.unaccounted_ms =
            wall - (last_step_telemetry.preparation_ms +
                    last_step_telemetry.inter_bucket_ms +
                    last_step_telemetry.forward_ms +
                    last_step_telemetry.loss_ms +
                    last_step_telemetry.backward_ms + optimizer);
    }
    attempt.commit();
    return last_objective_stats.total;
}

float Trainer::train_supervised(const std::vector<int>& prompt_tokens,
                                const std::vector<int>& answer_tokens) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    ensure_optimizer_state_usable();
    return train_supervised_batch({prompt_tokens}, {answer_tokens});
}

float Trainer::train_supervised_batch(
    const std::vector<std::vector<int>>& prompt_batch,
    const std::vector<std::vector<int>>& answer_batch) {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    ensure_optimizer_state_usable();
    (void)ensure_execution_identity_locked();
    return train_supervised_batch_impl(*this, prompt_batch, answer_batch);
}

void Trainer::train_loop(const std::vector<int>& tokens, int epochs, int batch_size,
                         int seq_len, std::function<void(int, float)> callback,
                         int max_steps, int start_step) {
    RuntimeExecutionPolicyLease policy_lease;
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex_);
    if (!model) throw std::runtime_error("Trainer requires model");
    std::lock_guard<std::recursive_mutex> model_lock(
        model->execution_mutex_);
    ensure_optimizer_state_usable();
    validate_trainer_configuration(*this);
    (void)ensure_execution_identity_locked();
    throw_if_training_cancelled(*this);
    require_checkpoint_boundary(*this);
    if (epochs <= 0 || batch_size <= 0 || seq_len <= 0) {
        throw std::invalid_argument(
            "train_loop requires positive epochs, batch_size, and seq_len");
    }
    if (start_step < 0 || (max_steps > 0 && start_step > max_steps)) {
        throw std::invalid_argument(
            "train_loop start_step must be in [0, max_steps]");
    }
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

    int internal_global_step = start_step;
    int encountered_step = 0;
    auto params = trainable_model_parameters(model);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t start = 0; start + seq_len < tokens.size();
             start += static_cast<size_t>(seq_len * effective_batch)) {
            ++encountered_step;
            if (encountered_step <= start_step) {
                continue;
            }
            throw_if_training_cancelled(*this);
            TrainingAttemptGuard attempt(*this, params);
            const float loss_scale = active_loss_scale(*this);
            apply_progressive_qat_phase(*this);
            bool fused_reset=false;
            if(device_sparse_adam && optimizer_policy::fused_optimizer_epilogue_enabled()) {
                gpu::ExecutionContext::Scope lane(device_sparse_execution_context());
                fused_reset=device_sparse_adam->consume_fused_gradient_reset(params);
            }
            if(!fused_reset)zero_model_gradients(params);
            begin_device_sparse_group();
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
            const size_t batch_tokens = flat_targets.size();
            require_token_capacity(tokens_processed, batch_tokens);
            require_token_capacity(tokens_committed, batch_tokens);

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
                throw_if_training_cancelled(*this);
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
                ctx.abort_signal = cancellation_signal();
                const bool cce_head = training_policy::head_cce();
                Tensor logits = cce_head ? model->forward_ids_batch_training_hidden(chunk_inputs, &ctx)
                                         : model->forward_ids_batch(chunk_inputs, &ctx);
                throw_if_training_cancelled(*this);
                // SSA learned block-selector distillation (no-op unless sparse on).
                const float chunk_weight =
                    static_cast<float>(chunk_samples) /
                    static_cast<float>(std::max(samples, 1));
                const float selector_loss =
                    model->accumulate_sparse_selector_grads(
                        chunk_weight * loss_scale);
                float supervised_loss = 0, repetition_loss = 0, logit_l2_loss = 0;
                Tensor grad;
                if (cce_head) {
                    TiledCrossEntropyOptions options;
                    options.targets = chunk_targets;
                    options.weights.assign(chunk_targets.size(), 1.0f / chunk_targets.size());
                    options.l2_weights = options.weights; options.sequence_length = seq_len;
                    options.eos_token = eos_token_id;
                    options.repetition_scale = std::max(repetition_unlikelihood_scale, 0.0f) / chunk_samples;
                    options.l2_beta = effective_logit_l2_beta(*this);
                    options.gradient_scale = chunk_weight * loss_scale;
                    auto result = model->tiled_head_loss(options);
                    Tensor host = result.losses.cpu();
                    supervised_loss = host.data()[0]; repetition_loss = host.data()[1]; logit_l2_loss = host.data()[2];
                    grad = std::move(result.input_gradient);
                } else {
                    auto loss_grad = logits.cross_entropy(chunk_targets);
                    supervised_loss = loss_grad.first; grad = std::move(loss_grad.second);
                    repetition_loss = apply_repetition_unlikelihood_batch(*this, chunk_target_batch, logits, grad);
                    logit_l2_loss = add_logit_l2_objective(logits, effective_logit_l2_beta(*this), grad);
                }
                // cross_entropy returns a mean over the chunk's token rows.
                // Weight each chunk by its sample fraction so accumulated
                // gradients equal the full-batch mean regardless of chunking.
                if (!cce_head) scale_tensor_inplace(grad, chunk_weight * loss_scale);
                throw_if_training_cancelled(*this);
                if (cce_head) model->backward_training_hidden(grad, ctx);
                else model->backward_external(grad, ctx);
                throw_if_training_cancelled(*this);

                tokens_processed += static_cast<long long>(chunk_targets.size());
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

            scale_gradients(params, 1.0f / loss_scale);
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
            LayerAuditCollector* step_audit =
                model ? model->audit_collector() : nullptr;
            throw_if_training_cancelled(*this);
            pending_accumulated_tokens = static_cast<long long>(batch_tokens);
            const float criticality_loss =
                apply_optimizer_step(
                    *this, params, 1,
                    step_audit && step_audit->enabled()
                        ? &grad_norm
                        : nullptr);
            if (!last_optimizer_step_skipped) {
                attempt.mark_optimizer_committed();
                tokens_committed += static_cast<long long>(batch_tokens);
            }
            pending_accumulated_tokens = 0;
            last_objective_stats.criticality_regularization = criticality_loss;
            last_objective_stats.total += criticality_loss;
            const float mean_loss = last_objective_stats.total;
            record_training_audit_step(*this, mean_loss, grad_norm,
                                       params.size());
            if (!last_optimizer_step_skipped) ++internal_global_step;
            if (callback) {
                callback(internal_global_step, mean_loss);
            }
            attempt.commit();

            if (max_steps > 0 && internal_global_step >= max_steps) return;
        }
    }
}

} // namespace nsos
