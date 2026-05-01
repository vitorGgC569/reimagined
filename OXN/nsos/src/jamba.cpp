#include "../include/jamba.h"
#include "../include/nsos_serializer.h"
#include "../include/nsos/determinism.h"
#include "../include/mcts_reasoning.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

namespace {

constexpr int kStableMoEExperts = 8;
constexpr int kStableTopKExperts = 2;
constexpr uint32_t kEdgePackMagic = 0x31454744; // DGE1
constexpr uint32_t kEdgePackVersion = 1;

uint64_t hash_token_sequence(const std::vector<int>& tokens) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (int token : tokens) {
        uint32_t value = static_cast<uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<unsigned char>((value >> shift) & 0xFFu);
            hash *= kPrime;
        }
    }
    return hash;
}

void copy_moe_bytes(float* dst,
                    Device dst_device,
                    const float* src,
                    Device src_device,
                    size_t bytes) {
    if (bytes == 0) {
        return;
    }
#ifdef USE_CUDA
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
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

int sanitize_head_count(int d_model, int requested) {
    int heads = std::max(requested, 1);
    heads = std::min(heads, std::max(d_model, 1));
    while (heads > 1 && (d_model % heads) != 0) {
        --heads;
    }
    return std::max(heads, 1);
}

int select_kv_heads(int n_heads, int requested) {
    if (requested > 0) {
        int chosen = std::min(std::max(requested, 1), std::max(n_heads, 1));
        while (chosen > 1 && (n_heads % chosen) != 0) {
            --chosen;
        }
        return std::max(chosen, 1);
    }

    int chosen = std::max(n_heads / 4, 1);
    while (chosen > 1 && (n_heads % chosen) != 0) {
        --chosen;
    }
    return std::max(chosen, 1);
}

bool layer_matches_schedule(int layer_one_based, int period, int slot) {
    if (period <= 0) {
        return false;
    }
    const int normalized_slot = std::clamp(slot, 0, period - 1);
    return ((layer_one_based - 1) % period) == normalized_slot;
}

void prefix_parameter_names(std::vector<Parameter*>& params, const std::string& prefix) {
    for (auto* param : params) {
        if (!param) {
            continue;
        }
        const std::string current = param->name.empty() ? param->base_name : param->name;
        const std::string leaf = param->base_name.empty() ? current : param->base_name;
        param->base_name = leaf;
        param->name =
            (current.find('.') == std::string::npos) ? (prefix + leaf) : (prefix + current);
    }
}

Tensor make_zero_like(const Tensor& x) {
    return Tensor::zeros(x.shape.dims, x.get_device());
}

std::vector<int> normalize_valid_lengths(const std::vector<int>& lengths,
                                         int batch_size,
                                         int seq_len) {
    std::vector<int> normalized(static_cast<size_t>(batch_size), seq_len);
    if (static_cast<int>(lengths.size()) == batch_size) {
        for (int batch = 0; batch < batch_size; ++batch) {
            normalized[static_cast<size_t>(batch)] =
                std::clamp(lengths[static_cast<size_t>(batch)], 0, seq_len);
        }
    }
    return normalized;
}

void zero_sequence_suffix_inplace(Tensor& tensor, const std::vector<int>& lengths) {
    if (tensor.shape.size() != 3 || lengths.empty()) {
        return;
    }
    const int batch_size = tensor.shape[0];
    const int seq_len = tensor.shape[1];
    const int dim = tensor.shape[2];
    float* ptr = tensor.data();
    for (int batch = 0; batch < batch_size; ++batch) {
        const int valid = std::clamp(lengths[static_cast<size_t>(batch)], 0, seq_len);
        for (int token = valid; token < seq_len; ++token) {
            std::fill_n(ptr + ((batch * seq_len + token) * dim), dim, 0.0f);
        }
    }
}

Tensor apply_training_dropout(const Tensor& input,
                              float dropout_rate,
                              const std::string& scope,
                              int salt) {
    const float rate = std::clamp(dropout_rate, 0.0f, 0.95f);
    if (rate <= 1e-6f || input.size == 0) {
        return input;
    }
    Tensor mask_host(input.shape.dims, Device::CPU);
    auto rng =
        determinism::DeterminismManager::instance().get_rng_for_operation(
            "jamba_dropout", scope,
            static_cast<uint64_t>(std::max(salt, 0)) ^
                static_cast<uint64_t>(input.shape.numel()));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float keep_scale = 1.0f / std::max(1.0f - rate, 1e-6f);
    float* mask_ptr = mask_host.data();
    for (int i = 0; i < mask_host.size; ++i) {
        mask_ptr[i] = dist(rng) >= rate ? keep_scale : 0.0f;
    }
    Tensor mask = input.get_device() == Device::GPU ? mask_host.to(Device::GPU) : mask_host;
    return input.mul(mask);
}

struct LogitDistributionStats {
    std::vector<float> probs;
    float entropy = 0.0f;
    float margin = 0.0f;
};

LogitDistributionStats summarize_logits_distribution(const float* ptr, int vocab) {
    LogitDistributionStats stats;
    if (ptr == nullptr || vocab <= 0) {
        return stats;
    }
    stats.probs.resize(static_cast<size_t>(vocab));
    float max_l = *std::max_element(ptr, ptr + vocab);
    float sum_e = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        stats.probs[static_cast<size_t>(i)] = std::exp(ptr[i] - max_l);
        sum_e += stats.probs[static_cast<size_t>(i)];
    }
    const float inv = 1.0f / std::max(sum_e, 1e-9f);
    float best = 0.0f;
    float second = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        float p = stats.probs[static_cast<size_t>(i)] * inv;
        stats.probs[static_cast<size_t>(i)] = p;
        if (p > 1e-9f) {
            stats.entropy -= p * std::log2(p);
        }
        if (p >= best) {
            second = best;
            best = p;
        } else if (p > second) {
            second = p;
        }
    }
    stats.margin = best - second;
    return stats;
}

float jensen_shannon_divergence(const std::vector<float>& lhs,
                                const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        return 0.0f;
    }
    float divergence = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float p = std::max(lhs[i], 1e-9f);
        const float q = std::max(rhs[i], 1e-9f);
        const float m = 0.5f * (p + q);
        divergence += 0.5f * (p * std::log2(p / m) + q * std::log2(q / m));
    }
    return divergence;
}

void copy_float_bytes_device_safe(float* dst,
                                  Device dst_device,
                                  const float* src,
                                  Device src_device,
                                  size_t bytes) {
    if (bytes == 0 || dst == src) {
        return;
    }
#ifdef USE_CUDA
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
        cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
        }
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed to write edge pack payload");
    }
}

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Failed to read edge pack payload");
    }
    return value;
}

template <typename T>
void write_vector(std::ofstream& out, const std::vector<T>& values) {
    const uint64_t count = static_cast<uint64_t>(values.size());
    write_pod(out, count);
    if (count == 0) {
        return;
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) {
        throw std::runtime_error("Failed to write edge pack vector");
    }
}

template <typename T>
std::vector<T> read_vector(std::ifstream& in, uint64_t max_count, const char* label) {
    const uint64_t count = read_pod<uint64_t>(in);
    if (count > max_count ||
        count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        throw std::runtime_error(std::string("Edge pack vector exceeds limit: ") + label);
    }
    std::vector<T> values(static_cast<size_t>(count));
    if (count == 0) {
        return values;
    }
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw std::runtime_error("Failed to read edge pack vector");
    }
    return values;
}

} // namespace

JambaModel::JambaModel(int nl, int dm, int vs, Device dev)
    : JambaModel([&] {
          ModelConfig cfg;
          cfg.num_layers = nl;
          cfg.d_model = dm;
          cfg.vocab_size = vs;
          cfg.use_cuda = (dev == Device::GPU);
          return cfg;
      }(),
      dev) {}

JambaModel::JambaModel(const ModelConfig& config, Device dev)
    : num_layers(std::max(config.num_layers, 1)),
      d_model(std::max(config.d_model, 8)),
      vocab_size(std::max(config.vocab_size, 2)),
      device(dev),
      model_config_(config) {
    model_config_.num_layers = num_layers;
    model_config_.d_model = d_model;
    model_config_.vocab_size = vocab_size;
    model_config_.use_cuda = (dev == Device::GPU);
    model_config_.n_heads = sanitize_head_count(d_model, model_config_.n_heads);
    model_config_.n_kv_heads =
        select_kv_heads(model_config_.n_heads,
                        sanitize_head_count(d_model, std::max(model_config_.n_kv_heads, 1)));
    model_config_.num_experts = std::max(model_config_.num_experts, 1);
    model_config_.num_experts_per_token =
        std::clamp(model_config_.num_experts_per_token, 1, model_config_.num_experts);
    const bool use_exact_attention_training = model_config_.use_exact_attention_training;

    embedding = std::make_unique<Embedding>(vocab_size, d_model);

    for (int i = 0; i < num_layers; ++i) {
        const int layer_one_based = i + 1;
        const bool use_attention =
            num_layers >= std::max(model_config_.attention_period, 1) &&
            layer_matches_schedule(layer_one_based, model_config_.attention_period,
                                   model_config_.attention_slot);
        const bool use_moe =
            model_config_.use_moe &&
            num_layers >= std::max(model_config_.moe_period, 1) &&
            layer_matches_schedule(layer_one_based, model_config_.moe_period,
                                   model_config_.moe_slot);
        const bool use_ttt =
            model_config_.use_ttt &&
            num_layers >= std::max(model_config_.ttt_period, 1) &&
            layer_matches_schedule(layer_one_based, model_config_.ttt_period,
                                   model_config_.ttt_slot);
        layers.push_back(std::make_unique<JambaBlock>(
            d_model, use_attention, use_moe, use_ttt, i, num_layers,
            model_config_.n_heads, model_config_.n_kv_heads, model_config_.num_experts,
            model_config_.num_experts_per_token,
            use_exact_attention_training,
            model_config_.dropout,
            model_config_.use_gradient_checkpointing));
    }

    value_head = std::make_unique<BitLinear>(d_model, vocab_size);
    to(dev);
}

void JambaModel::save(const std::string& filename) {
    ModelSerializer::save(this, filename);
}

void JambaModel::load(const std::string& filename, bool strict) {
    ModelSerializer::load(this, filename, strict);
}

Tensor JambaModel::forward(const Tensor& x, Context* ctx) {
    if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
        throw AbortException();
    }
    Tensor hidden = x;
    if (hidden.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
        zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
    }
    for (auto& layer : layers) {
        if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
            throw AbortException();
        }
        layer->set_batch_valid_lengths(last_input_batch_lengths_);
        hidden = layer->forward(hidden, ctx);
        if (hidden.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
            zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
        }
    }
    saved_final_hidden_ = hidden;
    saved_final_norm_ = hidden.rmsnorm();
    if (saved_final_norm_.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
        zero_sequence_suffix_inplace(saved_final_norm_, last_input_batch_lengths_);
    }
    Tensor logits = value_head->forward(saved_final_norm_);
    if (logits.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
        zero_sequence_suffix_inplace(logits, last_input_batch_lengths_);
    }
    return logits;
}

Tensor JambaModel::forward(const Tensor& x) {
    return forward(x, nullptr);
}

void JambaModel::to(Device dev) {
    device = dev;
    embedding->to(dev);
    for (auto& layer : layers) {
        layer->to(dev);
    }
    value_head->to(dev);
}

std::vector<Parameter*> JambaModel::parameters() {
    std::vector<Parameter*> params;
    auto embedding_params = embedding->parameters();
    prefix_parameter_names(embedding_params, "embedding.");
    params.insert(params.end(), embedding_params.begin(), embedding_params.end());
    for (size_t index = 0; index < layers.size(); ++index) {
        auto& layer = layers[index];
        auto layer_params = layer->parameters();
        prefix_parameter_names(layer_params, "layers." + std::to_string(index) + ".");
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    auto head_params = value_head->parameters();
    prefix_parameter_names(head_params, "value_head.");
    params.insert(params.end(), head_params.begin(), head_params.end());
    return params;
}

std::vector<BitLinear*> JambaModel::collect_bitlinear_layers() {
    std::vector<BitLinear*> layers_out;
    if (value_head) {
        layers_out.push_back(value_head.get());
    }
    for (auto& layer : layers) {
        layer->collect_bitlinear_layers(layers_out);
    }
    return layers_out;
}

void JambaModel::set_reference_path(bool use_reference_path) {
    for (BitLinear* layer : collect_bitlinear_layers()) {
        if (layer) {
            layer->set_reference_path(use_reference_path);
        }
    }
}

void JambaModel::release_full_precision_linear_weights() {
    for (BitLinear* layer : collect_bitlinear_layers()) {
        if (layer) {
            layer->release_full_precision_weight();
        }
    }
}

void JambaModel::save_edge_linear_pack(const std::string& path) {
    namespace fs = std::filesystem;

    const auto linear_layers = collect_bitlinear_layers();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open edge pack for writing: " + path);
    }

    write_pod(out, kEdgePackMagic);
    write_pod(out, kEdgePackVersion);
    write_pod(out, static_cast<uint32_t>(linear_layers.size()));
    for (BitLinear* layer : linear_layers) {
        if (!layer) {
            throw std::runtime_error("Null BitLinear layer while exporting edge pack");
        }
        const BitLinearPackedState state = layer->export_packed_state();
        write_pod(out, static_cast<int32_t>(state.in_features));
        write_pod(out, static_cast<int32_t>(state.out_features));
        write_pod(out, static_cast<uint8_t>(state.use_bias ? 1 : 0));
        write_pod(out, state.weight_scale);
        write_vector(out, state.packed_weights);
        write_vector(out, state.magnitude);
        write_vector(out, state.bias);
        write_vector(out, state.flat_alpha);
        write_vector(out, state.flat_beta);
    }

    if (!out) {
        throw std::runtime_error("Could not finalize edge pack: " + path);
    }
}

void JambaModel::load_edge_linear_pack(const std::string& path,
                                       bool release_full_precision) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open edge pack for reading: " + path);
    }

    const uint32_t magic = read_pod<uint32_t>(in);
    if (magic != kEdgePackMagic) {
        throw std::runtime_error("Invalid NSOS edge pack header");
    }
    const uint32_t version = read_pod<uint32_t>(in);
    if (version != kEdgePackVersion) {
        throw std::runtime_error("Unsupported NSOS edge pack version");
    }

    const auto linear_layers = collect_bitlinear_layers();
    const uint32_t expected_layers = read_pod<uint32_t>(in);
    if (expected_layers != linear_layers.size()) {
        throw std::runtime_error("Edge pack layer count mismatch");
    }

    for (uint32_t index = 0; index < expected_layers; ++index) {
        BitLinearPackedState state;
        state.in_features = read_pod<int32_t>(in);
        state.out_features = read_pod<int32_t>(in);
        state.use_bias = read_pod<uint8_t>(in) != 0;
        state.weight_scale = read_pod<float>(in);
        if (state.in_features <= 0 || state.out_features <= 0 ||
            state.in_features > 1'000'000 || state.out_features > 1'000'000) {
            throw std::runtime_error("Edge pack layer dimensions are outside configured limits");
        }
        const uint64_t max_weight_words =
            static_cast<uint64_t>(state.out_features) *
            static_cast<uint64_t>((state.in_features + 15) / 16);
        const uint64_t max_out = static_cast<uint64_t>(state.out_features);
        const uint64_t max_in_out =
            static_cast<uint64_t>(state.in_features) * static_cast<uint64_t>(state.out_features);
        state.packed_weights = read_vector<uint32_t>(in, max_weight_words, "packed_weights");
        state.magnitude = read_vector<float>(in, max_out, "magnitude");
        state.bias = read_vector<float>(in, state.use_bias ? max_out : 0, "bias");
        state.flat_alpha = read_vector<float>(in, max_in_out, "flat_alpha");
        state.flat_beta = read_vector<float>(in, max_in_out, "flat_beta");
        linear_layers[index]->import_packed_state(state, device, release_full_precision);
    }

    set_reference_path(!release_full_precision);
}

bool JambaModel::supports_streaming_inference() const {
    return true;
}

bool JambaModel::supports_batched_streaming_inference() const {
    for (const auto& layer : layers) {
        if (!layer) {
            continue;
        }
        if (!layer->mamba_layer && !layer->ttt_layer && !layer->attn_layer) {
            return false;
        }
    }
    return !layers.empty();
}

void JambaModel::set_streaming_inference(bool enabled) {
    if (enabled && training_mode_) {
        set_training_mode(false);
    }
    streaming_inference_enabled_ = enabled;
    for (auto& layer : layers) {
        layer->set_streaming_inference(enabled);
    }
}

void JambaModel::set_training_mode(bool enabled) {
    training_mode_ = enabled;
    for (auto& layer : layers) {
        layer->set_training_mode(enabled);
    }
}

void JambaModel::reset_runtime_telemetry() {
    for (auto& layer : layers) {
        if (layer && layer->mamba_layer) {
            layer->mamba_layer->reset_runtime_telemetry();
        }
    }
}

RuntimeTelemetrySnapshot JambaModel::runtime_telemetry() const {
    RuntimeTelemetrySnapshot snapshot;
    for (const auto& layer : layers) {
        if (!layer || !layer->mamba_layer) {
            continue;
        }
        snapshot.mamba_fast_path_hits += layer->mamba_layer->gpu_fast_path_hits();
        snapshot.mamba_fast_path_fallbacks += layer->mamba_layer->gpu_fast_path_fallbacks();
        if (!layer->mamba_layer->last_fallback_reason().empty()) {
            snapshot.mamba_last_fallback_reason = layer->mamba_layer->last_fallback_reason();
        }
    }
    return snapshot;
}

Tensor JambaModel::forward_ids(const std::vector<int>& ids, Context* ctx) {
    if (streaming_inference_enabled_ && !last_input_ids_.empty() && ids.size() == 1) {
        last_input_ids_.push_back(ids.front());
    } else {
        last_input_ids_ = ids;
    }
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    Tensor x = embedding->forward(ids);
    return forward(x, ctx);
}

Tensor JambaModel::forward_ids_batch(const std::vector<std::vector<int>>& batch_ids, Context* ctx) {
    if (batch_ids.empty()) {
        return Tensor();
    }
    if (streaming_inference_enabled_ &&
        !std::all_of(batch_ids.begin(), batch_ids.end(), [](const std::vector<int>& ids) {
            return ids.size() == 1;
        })) {
        throw std::runtime_error(
            "Streaming inference does not support batched token-id inputs yet");
    }
    int max_seq_len = 0;
    std::vector<std::vector<int>> padded_batch = batch_ids;
    last_input_batch_lengths_.clear();
    last_input_batch_lengths_.reserve(batch_ids.size());
    for (const auto& ids : batch_ids) {
        const int current_len = static_cast<int>(ids.size());
        max_seq_len = std::max(max_seq_len, current_len);
        last_input_batch_lengths_.push_back(current_len);
    }
    for (auto& ids : padded_batch) {
        ids.resize(static_cast<size_t>(max_seq_len), 0);
    }
    last_input_ids_.clear();
    last_input_batches_ = padded_batch;
    Tensor x = embedding->forward_batch(padded_batch);
    zero_sequence_suffix_inplace(x, last_input_batch_lengths_);
    return forward(x, ctx);
}

Tensor JambaModel::forward_trunk(const std::vector<int>& ids, Context* ctx) {
    last_input_ids_ = ids;
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
        throw AbortException();
    }
    Tensor hidden = embedding->forward(ids);
    for (auto& layer : layers) {
        if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
            throw AbortException();
        }
        hidden = layer->forward(hidden, ctx);
    }
    return hidden.rmsnorm();
}

Tensor JambaModel::forward_trunk_batch(const std::vector<std::vector<int>>& batch_ids,
                                       Context* ctx) {
    if (batch_ids.empty()) {
        return Tensor();
    }
    int max_seq_len = 0;
    std::vector<std::vector<int>> padded_batch = batch_ids;
    last_input_batch_lengths_.clear();
    last_input_batch_lengths_.reserve(batch_ids.size());
    for (const auto& ids : batch_ids) {
        const int current_len = static_cast<int>(ids.size());
        max_seq_len = std::max(max_seq_len, current_len);
        last_input_batch_lengths_.push_back(current_len);
    }
    for (auto& ids : padded_batch) {
        ids.resize(static_cast<size_t>(max_seq_len), 0);
    }
    last_input_ids_.clear();
    last_input_batches_ = padded_batch;
    if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
        throw AbortException();
    }
    Tensor hidden = embedding->forward_batch(padded_batch);
    zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
    for (auto& layer : layers) {
        if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
            throw AbortException();
        }
        layer->set_batch_valid_lengths(last_input_batch_lengths_);
        hidden = layer->forward(hidden, ctx);
        zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
    }
    Tensor trunk = hidden.rmsnorm();
    zero_sequence_suffix_inplace(trunk, last_input_batch_lengths_);
    return trunk;
}

Tensor JambaModel::forward_embedding(const Tensor& x, Context* ctx) {
    return forward(x, ctx);
}

Tensor JambaModel::reason(const Tensor& x, int num_simulations) {
    if (x.size == 0) return x;

    // Usa a representação latente como estado raiz do MCTS
    // O evaluator chama forward_embedding para pontuar cada estado explorado
    MCTSConfig cfg;
    cfg.num_simulations = std::max(num_simulations, 1);
    cfg.max_depth       = 8;
    cfg.num_children_per_expansion = 3;
    cfg.c_puct_init     = 1.15f;
    cfg.use_noise       = true;

    // Evaluator: projeta o estado latente para um valor escalar via value_head
    // Usamos a norma L2 negativa como proxy de qualidade (estados mais "ordenados"
    // têm menor norma após normalização RMS)
    auto evaluator = [this](const Tensor& state) -> float {
        if (state.size == 0) return 0.0f;
        try {
            // Projeção do estado pelo value_head → logits → usa a entropia inversa
            // como sinal de qualidade (baixa entropia = distribuição mais concentrada)
            auto summarize_state = [this](const Tensor& latent) {
                Tensor logits = value_head->forward(latent.rmsnorm());
                Tensor host_logits = logits.get_device() == Device::GPU ? logits.cpu() : logits;
                const int vocab = host_logits.shape.back();
                if (vocab <= 0 || host_logits.size < vocab) {
                    return LogitDistributionStats{};
                }
                const float* ptr = host_logits.data() + (host_logits.size - vocab);
                return summarize_logits_distribution(ptr, vocab);
            };
            Tensor host_state = state.get_device() == Device::GPU ? state.cpu() : state;
            const auto base = summarize_state(host_state);
            if (base.probs.empty()) return 0.0f;
            const auto contracted = summarize_state(host_state.mul(0.98f));
            const auto expanded = summarize_state(host_state.mul(1.02f));
            const float stability_penalty =
                0.5f * (jensen_shannon_divergence(base.probs, contracted.probs) +
                        jensen_shannon_divergence(base.probs, expanded.probs));
            return (1.25f * base.margin) - (0.12f * base.entropy) - (0.75f * stability_penalty);
        } catch (...) {
            return 0.0f;
        }
    };

    auto batch_evaluator = [this, &evaluator](const std::vector<Tensor>& states) {
        std::vector<float> values;
        values.reserve(states.size());
        if (states.empty()) {
            return values;
        }

        try {
            const int dim = states.front().shape.back();
            bool compatible = dim > 0;
            for (const Tensor& state : states) {
                if (state.size == 0 || state.shape.back() != dim ||
                    state.shape.size() != states.front().shape.size()) {
                    compatible = false;
                    break;
                }
            }

            if (!compatible) {
                for (const Tensor& state : states) {
                    values.push_back(evaluator(state));
                }
                return values;
            }

            Tensor batch_host({static_cast<int>(states.size()), dim}, Device::CPU);
            float* batch_ptr = batch_host.data();
            for (size_t index = 0; index < states.size(); ++index) {
                Tensor state_2d =
                    states[index].shape.size() == 1 ? states[index].reshape({1, dim}) : states[index];
                Tensor state_cpu =
                    (state_2d.get_device() == Device::GPU) ? state_2d.cpu() : state_2d;
                std::memcpy(batch_ptr + index * dim, state_cpu.data(),
                            static_cast<size_t>(dim) * sizeof(float));
            }

            Tensor batch =
                states.front().get_device() == Device::GPU ? batch_host.to(Device::GPU) : batch_host;
            Tensor normed = batch.rmsnorm();
            Tensor logits = value_head->forward(normed);
            Tensor host_logits = (logits.get_device() == Device::GPU) ? logits.cpu() : logits;
            const int vocab = host_logits.shape.back();
            Tensor contracted_logits =
                value_head->forward(batch_host.mul(0.98f).to(batch.get_device()).rmsnorm());
            Tensor expanded_logits =
                value_head->forward(batch_host.mul(1.02f).to(batch.get_device()).rmsnorm());
            Tensor host_contracted =
                contracted_logits.get_device() == Device::GPU ? contracted_logits.cpu()
                                                              : contracted_logits;
            Tensor host_expanded =
                expanded_logits.get_device() == Device::GPU ? expanded_logits.cpu()
                                                            : expanded_logits;
            for (size_t row = 0; row < states.size(); ++row) {
                const float* ptr = host_logits.data() + static_cast<int>(row) * vocab;
                const float* contracted_ptr = host_contracted.data() + static_cast<int>(row) * vocab;
                const float* expanded_ptr = host_expanded.data() + static_cast<int>(row) * vocab;
                const auto base = summarize_logits_distribution(ptr, vocab);
                const auto contracted = summarize_logits_distribution(contracted_ptr, vocab);
                const auto expanded = summarize_logits_distribution(expanded_ptr, vocab);
                const float stability_penalty =
                    0.5f * (jensen_shannon_divergence(base.probs, contracted.probs) +
                            jensen_shannon_divergence(base.probs, expanded.probs));
                values.push_back((1.25f * base.margin) - (0.12f * base.entropy) -
                                 (0.75f * stability_penalty));
            }
        } catch (...) {
            values.clear();
            for (const Tensor& state : states) {
                values.push_back(evaluator(state));
            }
        }

        return values;
    };

    MCTSReasoning mcts_engine(x, evaluator, cfg);
    mcts_engine.set_batch_evaluator(batch_evaluator);
    mcts_engine.search();
    return mcts_engine.get_best_state();
}

Tensor JambaModel::forward_thought(const Tensor& x, int steps) {
    Tensor hidden = x;
    for (int i = 0; i < std::max(steps, 1); ++i) {
        hidden = forward_embedding(hidden, nullptr);
    }
    return hidden;
}

void JambaModel::run_reasoning_loop(int iterations) {
    // Sem estado de entrada definido, não há o que refinar.
    // Esta sobrecarga existe para compatibilidade; use a versão com Tensor.
    (void)iterations;
    std::cerr << "[JambaModel] run_reasoning_loop(int) chamado sem estado de entrada."
                 " Use run_reasoning_loop(Tensor, int) para MCTS real.\n";
}

Tensor JambaModel::run_reasoning_loop(const Tensor& x, int iterations) {
    if (x.size == 0 || iterations <= 0) return x;

    // Iterative MCTS Refinement:
    // Cada iteração do loop usa o melhor estado MCTS da iteração anterior
    // como novo ponto de partida, criando um refinamento progressivo.
    Tensor current_state = x;
    const int sims_per_iter = std::max(200 / std::max(iterations, 1), 10);

    for (int i = 0; i < iterations; ++i) {
        Tensor refined = reason(current_state, sims_per_iter);
        if (refined.size == 0) break;

        // Passa o estado refinado pelo trunk novamente para atualizar
        // as ativações das camadas com o novo ponto de partida
        current_state = forward_embedding(refined, nullptr);

        std::cout << "[ReasoningLoop] Iteração " << i + 1 << "/" << iterations
                  << " concluída.\n";
    }
    return current_state;
}

void JambaModel::backward_external(const Tensor& grad, Context& ctx) {
    backward(grad, ctx);
}

void JambaModel::backward_embedding(const Tensor& grad, Context& ctx) {
    (void)ctx;
    if (!last_input_batches_.empty()) {
        embedding->backward_batch(grad, last_input_batches_);
    } else if (!last_input_ids_.empty()) {
        embedding->backward(grad, last_input_ids_);
    }
}

void JambaModel::backward(const Tensor& grad, Context& ctx) {
    Tensor dy = value_head->backward(grad);
    if (saved_final_hidden_.size > 0 && saved_final_norm_.size > 0) {
        dy = saved_final_hidden_.rmsnorm_backward(dy, saved_final_norm_);
    }
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        dy = layers[i]->backward(dy, &ctx);
    }
    backward_embedding(dy, ctx);
}

void JambaModel::reset_session() {
    last_input_ids_.clear();
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    saved_final_hidden_ = Tensor();
    saved_final_norm_ = Tensor();
    for (auto& layer : layers) {
        layer->reset();
    }
}

JambaSessionSnapshot JambaModel::fork_session() const {
    JambaSessionSnapshot snapshot;
    snapshot.streaming_enabled = streaming_inference_enabled_;
    snapshot.input_ids = last_input_ids_;
    snapshot.blocks.reserve(layers.size());
    for (const auto& layer : layers) {
        snapshot.blocks.push_back(layer ? layer->snapshot_session_state()
                                        : JambaBlockSessionSnapshot{});
    }
    return snapshot;
}

std::vector<JambaSessionSnapshot> JambaModel::fork_session_batch() const {
    if (!supports_batched_streaming_inference()) {
        throw std::runtime_error("Model does not support batched streaming snapshots");
    }
    std::vector<std::vector<JambaBlockSessionSnapshot>> layer_snapshots;
    layer_snapshots.reserve(layers.size());
    size_t batch_size = 0;
    for (const auto& layer : layers) {
        auto snapshots = layer ? layer->snapshot_session_state_batch()
                               : std::vector<JambaBlockSessionSnapshot>{};
        if (!snapshots.empty()) {
            batch_size = std::max(batch_size, snapshots.size());
        }
        layer_snapshots.push_back(std::move(snapshots));
    }

    std::vector<JambaSessionSnapshot> snapshots(batch_size);
    for (size_t item = 0; item < batch_size; ++item) {
        snapshots[item].streaming_enabled = streaming_inference_enabled_;
        snapshots[item].blocks.reserve(layers.size());
        for (size_t layer_index = 0; layer_index < layer_snapshots.size(); ++layer_index) {
            const auto& layer_batch = layer_snapshots[layer_index];
            if (item < layer_batch.size()) {
                snapshots[item].blocks.push_back(layer_batch[item]);
            } else {
                snapshots[item].blocks.push_back(JambaBlockSessionSnapshot{});
            }
        }
    }
    return snapshots;
}

void JambaModel::restore_session(const JambaSessionSnapshot& snapshot) {
    set_streaming_inference(snapshot.streaming_enabled);
    last_input_ids_ = snapshot.input_ids;
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    saved_final_hidden_ = Tensor();
    saved_final_norm_ = Tensor();

    const size_t shared_layers = std::min(layers.size(), snapshot.blocks.size());
    for (size_t index = 0; index < shared_layers; ++index) {
        layers[index]->restore_session_state(snapshot.blocks[index]);
    }
    for (size_t index = shared_layers; index < layers.size(); ++index) {
        layers[index]->reset();
    }
}

void JambaModel::restore_session_batch(const std::vector<JambaSessionSnapshot>& snapshots) {
    if (snapshots.empty()) {
        reset_session();
        return;
    }
    if (!supports_batched_streaming_inference()) {
        throw std::runtime_error("Model does not support batched streaming restore");
    }

    set_streaming_inference(snapshots.front().streaming_enabled);
    last_input_ids_.clear();
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    saved_final_hidden_ = Tensor();
    saved_final_norm_ = Tensor();

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        std::vector<JambaBlockSessionSnapshot> layer_batch;
        layer_batch.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            if (layer_index < snapshot.blocks.size()) {
                layer_batch.push_back(snapshot.blocks[layer_index]);
            } else {
                layer_batch.push_back(JambaBlockSessionSnapshot{});
            }
        }
        layers[layer_index]->restore_session_state_batch(layer_batch);
    }
}

void JambaModel::set_hamiltonian_mode(bool enabled) {
    for (auto& layer : layers) {
        if (layer->ttt_layer) {
            layer->ttt_layer->set_use_hamiltonian(enabled);
        }
    }
}

void JambaModel::session_adapt(const Tensor& x, const Tensor& y) {
    for (auto& layer : layers) {
        if (layer->ttt_layer) {
            layer->ttt_layer->initialize_from_meta(x, y);
        }
    }
}

Tensor JambaModel::run_simd_inference(const Tensor& x, int steps) {
    return forward_thought(x, steps);
}

JambaBlock::JambaBlock(int dm,
                       bool is_attn,
                       bool is_moe_flag,
                       bool is_ttt_layer,
                       int li,
                       int tl,
                       int query_heads,
                       int kv_heads,
                       int configured_experts,
                       int configured_top_k,
                       bool exact_attention_training,
                       float dropout_rate,
                       bool use_gradient_checkpointing)
    : is_attention(is_attn),
      is_moe(is_moe_flag),
      is_ttt(is_ttt_layer),
      layer_idx(li),
      total_layers(tl),
      d_model(dm),
      num_experts(std::max(configured_experts, 1)),
      dropout_rate_(std::clamp(dropout_rate, 0.0f, 0.95f)) {
    const int hidden_dim = dm * 4;

    if (is_ttt) {
        ttt_layer = std::make_unique<TTTLayer>(dm, hidden_dim);
    } else if (is_attention) {
        attn_layer = std::make_unique<Attention>(dm, std::max(query_heads, 1), 512,
                                                 std::max(kv_heads, 1));
        attn_layer->set_exact_training_path(exact_attention_training);
    } else {
        MambaConfig config;
        config.recompute_ssd = use_gradient_checkpointing;
        config.save_intermediates = !use_gradient_checkpointing;
        config.max_seq_for_storage = use_gradient_checkpointing ? 512 : 2048;
        mamba_layer = std::make_unique<Mamba2SSD>(dm, std::max(dm / 2, 8),
                                                  std::max(dm / 16, 1), config);
    }

    if (is_moe) {
        router = std::make_unique<MoERouter>(
            dm, num_experts, std::clamp(configured_top_k, 1, num_experts));
        for (int j = 0; j < num_experts; ++j) {
            expert_gate_up.push_back(std::make_unique<BitLinear>(dm, hidden_dim));
            expert_down.push_back(std::make_unique<BitLinear>(hidden_dim, dm));
        }
    } else {
        ffn_gate_up = std::make_unique<BitLinear>(dm, hidden_dim);
        ffn_down = std::make_unique<BitLinear>(hidden_dim, dm);
    }
}

JambaBlock::~JambaBlock() = default;

Tensor JambaBlock::forward(const Tensor& x, Context* ctx) {
    last_batch_size_ = x.shape.size() == 3 ? x.shape[0] : 0;
    saved_input_ = x;
    saved_core_norm_ = x.rmsnorm();
    Tensor core;
    if (is_ttt) {
        core = ttt_layer->forward(saved_core_norm_);
    } else if (is_attention) {
        core = attn_layer->forward(saved_core_norm_, ctx);
    } else {
        core = mamba_layer->forward(saved_core_norm_, ctx);
    }
    if (training_mode_ && dropout_rate_ > 1e-6f) {
        core = apply_training_dropout(core, dropout_rate_,
                                      "jamba_core_" + std::to_string(layer_idx),
                                      layer_idx * 17 + 1);
    }

    saved_residual_ = x.add(core);
    saved_ff_norm_ = saved_residual_.rmsnorm();

    if (is_moe) {
        Tensor ff = forward_moe(saved_ff_norm_, ctx, "L" + std::to_string(layer_idx));
        if (training_mode_ && dropout_rate_ > 1e-6f) {
            ff = apply_training_dropout(ff, dropout_rate_,
                                        "jamba_moe_" + std::to_string(layer_idx),
                                        layer_idx * 17 + 2);
        }
        return saved_residual_.add(ff);
    }

    saved_ff_hidden_pre_ = ffn_gate_up->forward(saved_ff_norm_);
    Tensor ff = saved_ff_hidden_pre_.relu();
    if (training_mode_ && dropout_rate_ > 1e-6f) {
        ff = apply_training_dropout(ff, dropout_rate_ * 0.5f,
                                    "jamba_ff_hidden_" + std::to_string(layer_idx),
                                    layer_idx * 17 + 3);
    }
    ff = ffn_down->forward(ff);
    if (training_mode_ && dropout_rate_ > 1e-6f) {
        ff = apply_training_dropout(ff, dropout_rate_,
                                    "jamba_ff_out_" + std::to_string(layer_idx),
                                    layer_idx * 17 + 4);
    }
    return saved_residual_.add(ff);
}

Tensor JambaBlock::forward_moe(const Tensor& x, Context* ctx, const std::string& ln) {
    (void)ctx;
    (void)ln;
    if (!router || expert_gate_up.empty()) {
        return make_zero_like(x);
    }

    auto [unused_logits, weights] = router->forward(x);
    (void)unused_logits;

    const int rows = x.size / x.shape.back();
    const int dim = x.shape.back();
    const Device target_device = x.get_device();
    Tensor weights_host = weights.get_device() == Device::GPU ? weights.cpu() : weights;
    Tensor output_accum = Tensor::zeros({rows, dim}, target_device);
    const float* weight_ptr = weights_host.data();
    const int effective_top_k = std::clamp(router->top_k, 1, num_experts);
    std::vector<std::vector<int>> expert_rows(static_cast<size_t>(num_experts));
    std::vector<int> ranked_experts(static_cast<size_t>(num_experts));
    std::iota(ranked_experts.begin(), ranked_experts.end(), 0);

    for (int row = 0; row < rows; ++row) {
        std::partial_sort(
            ranked_experts.begin(),
            ranked_experts.begin() + effective_top_k,
            ranked_experts.end(),
            [&](int lhs, int rhs) {
                return weight_ptr[row * num_experts + lhs] >
                       weight_ptr[row * num_experts + rhs];
            });
        for (int rank = 0; rank < effective_top_k; ++rank) {
            expert_rows[static_cast<size_t>(ranked_experts[rank])].push_back(row);
        }
    }

    saved_moe_weights_ = weights_host;
    saved_moe_rows_ = expert_rows;

    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const auto& selected_rows = expert_rows[static_cast<size_t>(expert_idx)];
        if (selected_rows.empty()) {
            continue;
        }

        Tensor expert_input({static_cast<int>(selected_rows.size()), dim}, target_device);
        float* expert_input_ptr = expert_input.data();
        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int source_row = selected_rows[local_row];
            copy_moe_bytes(expert_input_ptr + static_cast<int>(local_row) * dim,
                           target_device,
                           x.data() + source_row * dim,
                           x.get_device(),
                           static_cast<size_t>(dim) * sizeof(float));
        }

        Tensor expert_hidden = expert_gate_up[expert_idx]->forward(expert_input).relu();
        Tensor expert_out = expert_down[expert_idx]->forward(expert_hidden);
        std::vector<float> selected_weights(selected_rows.size(), 0.0f);
        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int target_row = selected_rows[local_row];
            selected_weights[local_row] = weight_ptr[target_row * num_experts + expert_idx];
        }
        Tensor scale_tensor({static_cast<int>(selected_rows.size()), 1}, Device::CPU);
        std::memcpy(scale_tensor.data(),
                    selected_weights.data(),
                    selected_weights.size() * sizeof(float));
        Tensor scaled_out = expert_out.mul(scale_tensor.to(target_device));
        Tensor expert_scatter = Tensor::zeros({rows, dim}, target_device);
        float* scatter_ptr = expert_scatter.data();
        const float* scaled_ptr = scaled_out.data();

        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int target_row = selected_rows[local_row];
            copy_moe_bytes(scatter_ptr + target_row * dim,
                           target_device,
                           scaled_ptr + static_cast<int>(local_row) * dim,
                           target_device,
                           static_cast<size_t>(dim) * sizeof(float));
        }
        output_accum = output_accum.add(expert_scatter);
    }

    return output_accum.reshape(x.shape.dims);
}

Tensor JambaBlock::backward_moe(const Tensor& dy, Context* ctx, const std::string& ln,
                                const Tensor& x) {
    (void)ctx;
    (void)ln;
    const Device target_device = dy.get_device();
    Tensor grad_accum = Tensor::zeros(dy.shape.dims, target_device);
    if (saved_moe_rows_.empty() || saved_moe_weights_.size == 0) {
        Tensor grad = Tensor::zeros(dy.shape.dims, dy.get_device());
        for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            Tensor expert_grad = expert_down[expert_idx]->backward(dy);
            expert_grad = expert_gate_up[expert_idx]->backward(expert_grad);
            grad = grad.add(expert_grad);
        }
        return grad;
    }

    const int dim = x.shape.back();
    const float* weight_ptr = saved_moe_weights_.data();
    float* grad_ptr = grad_accum.data();

    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const auto& selected_rows = saved_moe_rows_[static_cast<size_t>(expert_idx)];
        if (selected_rows.empty()) {
            continue;
        }

        Tensor expert_dy({static_cast<int>(selected_rows.size()), dim}, target_device);
        float* expert_dy_ptr = expert_dy.data();

        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int source_row = selected_rows[local_row];
            copy_moe_bytes(expert_dy_ptr + static_cast<int>(local_row) * dim,
                           target_device,
                           dy.data() + source_row * dim,
                           dy.get_device(),
                           static_cast<size_t>(dim) * sizeof(float));
        }
        std::vector<float> selected_weights(selected_rows.size(), 0.0f);
        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int source_row = selected_rows[local_row];
            selected_weights[local_row] = weight_ptr[source_row * num_experts + expert_idx];
        }
        Tensor scale_tensor({static_cast<int>(selected_rows.size()), 1}, Device::CPU);
        std::memcpy(scale_tensor.data(),
                    selected_weights.data(),
                    selected_weights.size() * sizeof(float));
        expert_dy = expert_dy.mul(scale_tensor.to(target_device));

        Tensor expert_grad = expert_down[expert_idx]->backward(expert_dy);
        expert_grad = expert_gate_up[expert_idx]->backward(expert_grad);
        Tensor expert_scatter = Tensor::zeros(dy.shape.dims, target_device);
        float* scatter_ptr = expert_scatter.data();
        const float* expert_grad_ptr = expert_grad.data();

        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int target_row = selected_rows[local_row];
            copy_moe_bytes(scatter_ptr + target_row * dim,
                           target_device,
                           expert_grad_ptr + static_cast<int>(local_row) * dim,
                           expert_grad.get_device(),
                           static_cast<size_t>(dim) * sizeof(float));
        }
        grad_accum = grad_accum.add(expert_scatter);
    }

    return grad_accum;
}

Tensor JambaBlock::backward(const Tensor& dy, Context* ctx) {
    Tensor ff_grad;
    if (is_moe) {
        ff_grad = backward_moe(dy, ctx, "L" + std::to_string(layer_idx), saved_ff_norm_);
        if (saved_residual_.size > 0 && saved_ff_norm_.size > 0) {
            ff_grad = saved_residual_.rmsnorm_backward(ff_grad, saved_ff_norm_);
        }
    } else if (ffn_down && ffn_gate_up) {
        ff_grad = ffn_down->backward(dy);
        if (saved_ff_hidden_pre_.size > 0) {
            float* ff_grad_ptr = ff_grad.data();
            const float* pre_ptr = saved_ff_hidden_pre_.data();
            for (int i = 0; i < ff_grad.size; ++i) {
                if (pre_ptr[i] <= 0.0f) {
                    ff_grad_ptr[i] = 0.0f;
                }
            }
        }
        ff_grad = ffn_gate_up->backward(ff_grad);
        if (saved_residual_.size > 0 && saved_ff_norm_.size > 0) {
            ff_grad = saved_residual_.rmsnorm_backward(ff_grad, saved_ff_norm_);
        }
    } else {
        ff_grad = make_zero_like(dy);
    }

    // Regra da Cadeia para Conexão Residual Superior: d_residual = dy + d_ff
    Tensor residual_grad = dy.add(ff_grad);

    Tensor core_grad;
    if (is_ttt && ttt_layer) {
        core_grad = ttt_layer->backward(residual_grad);
    } else if (is_attention && attn_layer) {
        core_grad = attn_layer->backward(residual_grad, ctx);
    } else if (mamba_layer) {
        Context local_ctx;
        core_grad = mamba_layer->backward(residual_grad, ctx ? *ctx : local_ctx);
    } else {
        core_grad = make_zero_like(dy);
    }

    if (saved_input_.size > 0 && saved_core_norm_.size > 0) {
        core_grad = saved_input_.rmsnorm_backward(core_grad, saved_core_norm_);
    }

    // Regra da Cadeia Inferior: d_input = d_residual + d_core
    return residual_grad.add(core_grad);
}

void JambaBlock::reset() {
    saved_input_ = Tensor();
    saved_core_norm_ = Tensor();
    saved_residual_ = Tensor();
    saved_ff_norm_ = Tensor();
    saved_ff_hidden_pre_ = Tensor();
    saved_moe_weights_ = Tensor();
    saved_moe_rows_.clear();
    if (mamba_layer) {
        mamba_layer->reset();
    }
    if (attn_layer) {
        attn_layer->reset();
    }
    if (ttt_layer) {
        ttt_layer->reset();
    }
}

void JambaBlock::to(Device dev) {
    if (attn_layer) {
        attn_layer->to(dev);
    }
    if (mamba_layer) {
        mamba_layer->to(dev);
    }
    if (ttt_layer) {
        ttt_layer->to(dev);
    }
    if (router) {
        router->to(dev);
    }
    if (ffn_gate_up) {
        ffn_gate_up->to(dev);
    }
    if (ffn_down) {
        ffn_down->to(dev);
    }
    for (auto& expert : expert_gate_up) {
        expert->to(dev);
    }
    for (auto& expert : expert_down) {
        expert->to(dev);
    }
}

std::vector<Parameter*> JambaBlock::parameters() {
    std::vector<Parameter*> params;
    if (attn_layer) {
        auto attn = attn_layer->parameters();
        prefix_parameter_names(attn, "attn.");
        params.insert(params.end(), attn.begin(), attn.end());
    }
    if (mamba_layer) {
        auto mamba = mamba_layer->parameters();
        prefix_parameter_names(mamba, "mamba.");
        params.insert(params.end(), mamba.begin(), mamba.end());
    }
    if (ttt_layer) {
        auto ttt = ttt_layer->parameters();
        prefix_parameter_names(ttt, "ttt.");
        params.insert(params.end(), ttt.begin(), ttt.end());
    }
    if (router) {
        auto router_params = router->parameters();
        prefix_parameter_names(router_params, "router.");
        params.insert(params.end(), router_params.begin(), router_params.end());
    }
    if (ffn_gate_up) {
        auto ff_up = ffn_gate_up->parameters();
        prefix_parameter_names(ff_up, "ffn_gate_up.");
        params.insert(params.end(), ff_up.begin(), ff_up.end());
    }
    if (ffn_down) {
        auto ff_down = ffn_down->parameters();
        prefix_parameter_names(ff_down, "ffn_down.");
        params.insert(params.end(), ff_down.begin(), ff_down.end());
    }
    for (size_t index = 0; index < expert_gate_up.size(); ++index) {
        auto& expert = expert_gate_up[index];
        auto expert_params = expert->parameters();
        prefix_parameter_names(expert_params, "experts." + std::to_string(index) + ".up.");
        params.insert(params.end(), expert_params.begin(), expert_params.end());
    }
    for (size_t index = 0; index < expert_down.size(); ++index) {
        auto& expert = expert_down[index];
        auto expert_params = expert->parameters();
        prefix_parameter_names(expert_params, "experts." + std::to_string(index) + ".down.");
        params.insert(params.end(), expert_params.begin(), expert_params.end());
    }
    return params;
}

void JambaBlock::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (attn_layer) {
        attn_layer->collect_bitlinear_layers(out);
    }
    if (mamba_layer) {
        mamba_layer->collect_bitlinear_layers(out);
    }
    if (ttt_layer) {
        ttt_layer->collect_bitlinear_layers(out);
    }
    if (router) {
        router->collect_bitlinear_layers(out);
    }
    if (ffn_gate_up) {
        out.push_back(ffn_gate_up.get());
    }
    if (ffn_down) {
        out.push_back(ffn_down.get());
    }
    for (auto& expert : expert_gate_up) {
        out.push_back(expert.get());
    }
    for (auto& expert : expert_down) {
        out.push_back(expert.get());
    }
}

void JambaBlock::set_streaming_inference(bool enabled) {
    if (attn_layer) {
        attn_layer->set_streaming_mode(enabled);
    }
    if (mamba_layer) {
        mamba_layer->set_streaming_mode(enabled);
    }
}

void JambaBlock::set_training_mode(bool enabled) {
    training_mode_ = enabled;
    if (attn_layer) {
        attn_layer->set_training_mode(enabled);
    }
    if (ttt_layer) {
        ttt_layer->set_training_mode(enabled);
    }
}

void JambaBlock::set_batch_valid_lengths(const std::vector<int>& lengths) {
    if (attn_layer) {
        attn_layer->set_batch_valid_lengths(lengths);
    }
}

JambaBlockSessionSnapshot JambaBlock::snapshot_session_state() const {
    JambaBlockSessionSnapshot snapshot;
    snapshot.has_attention = static_cast<bool>(attn_layer);
    snapshot.has_mamba = static_cast<bool>(mamba_layer);
    snapshot.has_ttt = static_cast<bool>(ttt_layer);
    if (attn_layer) {
        snapshot.attention = attn_layer->snapshot_cache();
    }
    if (mamba_layer) {
        snapshot.mamba = mamba_layer->snapshot_streaming_state();
    }
    if (ttt_layer) {
        snapshot.ttt = ttt_layer->snapshot_state();
    }
    return snapshot;
}

std::vector<JambaBlockSessionSnapshot> JambaBlock::snapshot_session_state_batch() const {
    if (attn_layer) {
        std::vector<JambaBlockSessionSnapshot> snapshots;
        auto attention_snapshots = attn_layer->snapshot_cache_batch();
        snapshots.reserve(attention_snapshots.size());
        for (const auto& attention_snapshot : attention_snapshots) {
            JambaBlockSessionSnapshot snapshot;
            snapshot.has_attention = true;
            snapshot.attention = attention_snapshot;
            if (ttt_layer) {
                snapshot.has_ttt = true;
                snapshot.ttt = ttt_layer->snapshot_state();
            }
            snapshots.push_back(std::move(snapshot));
        }
        return snapshots;
    }
    if (mamba_layer) {
        std::vector<JambaBlockSessionSnapshot> snapshots;
        auto mamba_snapshots = mamba_layer->snapshot_streaming_state_batch();
        snapshots.reserve(mamba_snapshots.size());
        for (const auto& mamba_snapshot : mamba_snapshots) {
            JambaBlockSessionSnapshot snapshot;
            snapshot.has_mamba = true;
            snapshot.mamba = mamba_snapshot;
            if (ttt_layer) {
                snapshot.has_ttt = true;
                snapshot.ttt = ttt_layer->snapshot_state();
            }
            snapshots.push_back(std::move(snapshot));
        }
        return snapshots;
    }
    if (!ttt_layer || last_batch_size_ <= 0) {
        return {};
    }
    std::vector<JambaBlockSessionSnapshot> snapshots;
    snapshots.reserve(static_cast<size_t>(last_batch_size_));
    const TTTSessionSnapshot shared_ttt = ttt_layer->snapshot_state();
    for (int batch = 0; batch < last_batch_size_; ++batch) {
        JambaBlockSessionSnapshot snapshot;
        snapshot.has_ttt = true;
        snapshot.ttt = shared_ttt;
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void JambaBlock::restore_session_state(const JambaBlockSessionSnapshot& snapshot) {
    if (attn_layer && snapshot.has_attention) {
        attn_layer->restore_cache(snapshot.attention);
    }
    if (mamba_layer && snapshot.has_mamba) {
        mamba_layer->restore_streaming_state(snapshot.mamba);
    }
    if (ttt_layer && snapshot.has_ttt) {
        ttt_layer->restore_state(snapshot.ttt);
    }
}

void JambaBlock::restore_session_state_batch(const std::vector<JambaBlockSessionSnapshot>& snapshots) {
    if (attn_layer) {
        std::vector<AttentionCacheSnapshot> attention_snapshots;
        attention_snapshots.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            attention_snapshots.push_back(snapshot.attention);
        }
        attn_layer->restore_cache_batch(attention_snapshots);
        if (ttt_layer && !snapshots.empty() && snapshots.front().has_ttt) {
            ttt_layer->restore_state(snapshots.front().ttt);
        }
        return;
    }
    if (mamba_layer) {
        std::vector<MambaStreamSnapshot> mamba_snapshots;
        mamba_snapshots.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            mamba_snapshots.push_back(snapshot.mamba);
        }
        mamba_layer->restore_streaming_state_batch(mamba_snapshots);
        if (ttt_layer && !snapshots.empty() && snapshots.front().has_ttt) {
            ttt_layer->restore_state(snapshots.front().ttt);
        }
        return;
    }
    if (ttt_layer) {
        if (!snapshots.empty() && snapshots.front().has_ttt) {
            ttt_layer->restore_state(snapshots.front().ttt);
        }
        return;
    }
    reset();
}

Attention::Attention(int d, int n, int l, int n_kv)
    : d_model(d),
      n_heads(std::max(n, 1)),
      n_kv_heads(select_kv_heads(std::max(n, 1), n_kv)),
      kv_group_size(std::max(std::max(n, 1) /
                                 std::max(select_kv_heads(std::max(n, 1), n_kv), 1),
                             1)),
      head_dim(std::max(d / std::max(n, 1), 1)),
      n_latents(l),
      q_down_proj(std::make_unique<BitLinear>(d, d)),
      kv_down_proj(
          std::make_unique<BitLinear>(d, 2 * n_kv_heads * head_dim)),
      out_proj(std::make_unique<BitLinear>(d, d)),
      max_seq_len(4096),
      theta(10000.0f) {
    precompute_freqs_cis();
}

void Attention::precompute_freqs_cis() {
    // RoPE: pré-computa cos/sin para cada posição e dimensão
    // head_dim/2 pares de dimensões recebem rotações distintas
    const int half_dim = head_dim / 2;
    cos_cached.resize(max_seq_len * half_dim);
    sin_cached.resize(max_seq_len * half_dim);
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, 2.0f * i / head_dim);
            float angle = static_cast<float>(pos) * freq;
            cos_cached[pos * half_dim + i] = std::cos(angle);
            sin_cached[pos * half_dim + i] = std::sin(angle);
        }
    }
}

std::pair<Tensor, Tensor> Attention::apply_rope(const Tensor& q, const Tensor& k, int start_pos) {
    if (q.shape.size() != k.shape.size() || (q.shape.size() != 3 && q.shape.size() != 4)) {
        throw std::runtime_error("Attention::apply_rope expects rank-3 or rank-4 q/k tensors");
    }

    const bool batched = q.shape.size() == 4;
    const int batch = batched ? q.shape[0] : 1;
    const int seq_len = batched ? q.shape[1] : q.shape[0];
    const int q_heads = batched ? q.shape[2] : q.shape[1];
    const int k_heads = batched ? k.shape[2] : k.shape[1];
    const int half_dim = head_dim / 2;

    Tensor q_rot = q.clone();
    Tensor k_rot = k.clone();
    float* qd = q_rot.data();
    float* kd = k_rot.data();

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int pos = std::min(start_pos + s, max_seq_len - 1);
            for (int h = 0; h < q_heads; ++h) {
                const size_t q_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * q_heads + h) * head_dim
                        : (static_cast<size_t>(s) * q_heads + h) * head_dim;
                float* q_head = qd + q_offset;
                for (int i = 0; i < half_dim; ++i) {
                    float cos_v = cos_cached[pos * half_dim + i];
                    float sin_v = sin_cached[pos * half_dim + i];
                    float q0 = q_head[i],           q1 = q_head[i + half_dim];
                    q_head[i]            = q0 * cos_v - q1 * sin_v;
                    q_head[i + half_dim] = q0 * sin_v + q1 * cos_v;
                }
            }
            for (int h = 0; h < k_heads; ++h) {
                const size_t k_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * k_heads + h) * head_dim
                        : (static_cast<size_t>(s) * k_heads + h) * head_dim;
                float* k_head = kd + k_offset;
                for (int i = 0; i < half_dim; ++i) {
                    float cos_v = cos_cached[pos * half_dim + i];
                    float sin_v = sin_cached[pos * half_dim + i];
                    float k0 = k_head[i],           k1 = k_head[i + half_dim];
                    k_head[i]            = k0 * cos_v - k1 * sin_v;
                    k_head[i + half_dim] = k0 * sin_v + k1 * cos_v;
                }
            }
        }
    }
    return {q_rot, k_rot};
}

std::pair<Tensor, Tensor> Attention::apply_rope_backward(const Tensor& grad_q_rot,
                                                         const Tensor& grad_k_rot,
                                                         int start_pos) {
    if ((grad_q_rot.shape.size() != 3 && grad_q_rot.shape.size() != 4) ||
        (grad_k_rot.shape.size() != 3 && grad_k_rot.shape.size() != 4)) {
        throw std::runtime_error("RoPE backward expects rank-3 or rank-4 tensors");
    }

    const bool batched = grad_q_rot.shape.size() == 4;
    const int batch = batched ? grad_q_rot.shape[0] : 1;
    const int seq_len = batched ? grad_q_rot.shape[1] : grad_q_rot.shape[0];
    const int q_heads = batched ? grad_q_rot.shape[2] : grad_q_rot.shape[1];
    const int k_heads = batched ? grad_k_rot.shape[2] : grad_k_rot.shape[1];
    const int half_dim = head_dim / 2;

    Tensor grad_q = grad_q_rot.clone();
    Tensor grad_k = grad_k_rot.clone();
    float* qd = grad_q.data();
    float* kd = grad_k.data();

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int pos = std::min(start_pos + s, max_seq_len - 1);
            for (int h = 0; h < q_heads; ++h) {
                const size_t q_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * q_heads + h) * head_dim
                        : (static_cast<size_t>(s) * q_heads + h) * head_dim;
                float* q_head = qd + q_offset;
                for (int i = 0; i < half_dim; ++i) {
                    const float cos_v = cos_cached[pos * half_dim + i];
                    const float sin_v = sin_cached[pos * half_dim + i];
                    const float qr0 = q_head[i];
                    const float qr1 = q_head[i + half_dim];
                    q_head[i] = qr0 * cos_v + qr1 * sin_v;
                    q_head[i + half_dim] = -qr0 * sin_v + qr1 * cos_v;
                }
            }
            for (int h = 0; h < k_heads; ++h) {
                const size_t k_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * k_heads + h) * head_dim
                        : (static_cast<size_t>(s) * k_heads + h) * head_dim;
                float* k_head = kd + k_offset;
                for (int i = 0; i < half_dim; ++i) {
                    const float cos_v = cos_cached[pos * half_dim + i];
                    const float sin_v = sin_cached[pos * half_dim + i];
                    const float kr0 = k_head[i];
                    const float kr1 = k_head[i + half_dim];
                    k_head[i] = kr0 * cos_v + kr1 * sin_v;
                    k_head[i + half_dim] = -kr0 * sin_v + kr1 * cos_v;
                }
            }
        }
    }

    return {grad_q, grad_k};
}

Tensor Attention::forward(const Tensor& input, Context* ctx) {
    (void)ctx;
    saved_q_rot_ = Tensor();
    saved_k_rot_ = Tensor();
    saved_v_heads_ = Tensor();
    saved_attn_probs_ = Tensor();
    saved_valid_lengths_.clear();
    if (training_mode_ && exact_training_path_ && !streaming_inference_ &&
        (input.shape.size() == 1 || input.shape.size() == 2 || input.shape.size() == 3)) {
        const Device original_device = input.get_device();
        Tensor project_input = input;
        int batch_size = 1;
        int seq_len = 1;
        if (input.shape.size() == 1) {
            project_input = input.reshape({1, 1, d_model});
            batch_size = 1;
            seq_len = 1;
            saved_valid_lengths_ = {1};
        } else if (input.shape.size() == 2) {
            seq_len = input.shape[0];
            project_input = input.reshape({1, seq_len, d_model});
            batch_size = 1;
            saved_valid_lengths_ = {seq_len};
        } else {
            batch_size = input.shape[0];
            seq_len = input.shape[1];
            saved_valid_lengths_ =
                normalize_valid_lengths(active_batch_valid_lengths_, batch_size, seq_len);
        }

        saved_input_rank_ = static_cast<int>(input.shape.size());
        const int kv_dim = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        Tensor q_flat = q_down_proj->forward(project_input);
        Tensor kv_flat = kv_down_proj->forward(project_input);
        Tensor exact_forward_gpu;
        bool used_gpu_exact_forward = false;
#ifdef USE_CUDA
        if (original_device == Device::GPU &&
            q_flat.get_device() == Device::GPU &&
            kv_flat.get_device() == Device::GPU &&
            batch_size > 0 &&
            seq_len > 0 &&
            seq_len <= 4096 &&
            gpu_custom_kernels_supported()) {
            exact_forward_gpu = Tensor({batch_size, seq_len, d_model}, Device::GPU);
            launch_batched_gqa_causal_attention_kernel(q_flat.data(),
                                                       kv_flat.data(),
                                                       exact_forward_gpu.data(),
                                                       batch_size,
                                                       seq_len,
                                                       d_model,
                                                       n_heads,
                                                       n_kv_heads,
                                                       head_dim,
                                                       kv_group_size,
                                                       theta);
            cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Exact batched GQA attention kernel launch failed: ") +
                    cudaGetErrorString(status));
            }
            if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                if (std::string(sync_env) == "1") {
                    status = cudaDeviceSynchronize();
                    if (status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("Exact batched GQA attention kernel sync failed: ") +
                            cudaGetErrorString(status));
                    }
                }
            }
            used_gpu_exact_forward = true;
        }
#endif
        Tensor q_host = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
        Tensor kv_host = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;
        Tensor k_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        Tensor v_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        {
            const float* kv_ptr = kv_host.data();
            float* k_ptr = k_flat.data();
            float* v_ptr = v_flat.data();
            const size_t token_stride = static_cast<size_t>(2 * kv_dim);
            for (int batch = 0; batch < batch_size; ++batch) {
                for (int token = 0; token < seq_len; ++token) {
                    const size_t token_offset =
                        (static_cast<size_t>(batch) * seq_len + token) * token_stride;
                    const size_t out_offset =
                        (static_cast<size_t>(batch) * seq_len + token) *
                        static_cast<size_t>(kv_dim);
                    std::memcpy(
                        k_ptr + out_offset,
                        kv_ptr + token_offset,
                        static_cast<size_t>(kv_dim) * sizeof(float));
                    std::memcpy(
                        v_ptr + out_offset,
                        kv_ptr + token_offset + static_cast<size_t>(kv_dim),
                        static_cast<size_t>(kv_dim) * sizeof(float));
                }
            }
        }

        Tensor q_heads = q_host.reshape({batch_size, seq_len, n_heads, head_dim});
        Tensor k_heads = k_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        Tensor v_heads = v_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        auto [q_rot, k_rot] = apply_rope(q_heads, k_heads, 0);

        Tensor output_heads({batch_size, seq_len, n_heads, head_dim}, q_rot.get_device());
        float* out_ptr = output_heads.data();
        const float* q_ptr = q_rot.data();
        const float* k_ptr = k_rot.data();
        const float* v_ptr = v_heads.data();
        std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);
        std::vector<float> probs(static_cast<size_t>(seq_len), 0.0f);

        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len = std::clamp(saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
            for (int head = 0; head < n_heads; ++head) {
                const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                for (int i = 0; i < seq_len; ++i) {
                    const size_t out_row_offset =
                        (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                        static_cast<size_t>(head_dim);
                    if (i >= valid_len) {
                        std::fill_n(out_ptr + out_row_offset, head_dim, 0.0f);
                        continue;
                    }

                    float max_s = -1e30f;
                    for (int j = 0; j < seq_len; ++j) {
                        float score = -1e9f;
                        if (j < valid_len && j <= i) {
                            float dot = 0.0f;
                            const size_t q_offset =
                                (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                                static_cast<size_t>(head_dim);
                            const size_t k_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                static_cast<size_t>(head_dim);
                            for (int dim = 0; dim < head_dim; ++dim) {
                                dot += q_ptr[q_offset + dim] * k_ptr[k_offset + dim];
                            }
                            score = dot * scale;
                        }
                        scores[static_cast<size_t>(j)] = score;
                        max_s = std::max(max_s, score);
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        float value =
                            (j < valid_len && j <= i) ? std::exp(scores[static_cast<size_t>(j)] - max_s)
                                                      : 0.0f;
                        probs[static_cast<size_t>(j)] = value;
                        sum_exp += value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (int j = 0; j < seq_len; ++j) {
                        probs[static_cast<size_t>(j)] *= inv_sum;
                    }

                    for (int dim = 0; dim < head_dim; ++dim) {
                        float acc = 0.0f;
                        for (int j = 0; j < valid_len; ++j) {
                            const size_t v_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                    static_cast<size_t>(head_dim) +
                                static_cast<size_t>(dim);
                            acc += probs[static_cast<size_t>(j)] * v_ptr[v_offset];
                        }
                        out_ptr[out_row_offset + static_cast<size_t>(dim)] = acc;
                    }
                }
            }
        }

        saved_q_rot_ = q_rot;
        saved_k_rot_ = k_rot;
        saved_v_heads_ = v_heads;
        saved_attn_probs_ = Tensor();

        Tensor output_2d = output_heads.reshape({batch_size, seq_len, d_model});
        Tensor projected_input = used_gpu_exact_forward
                                     ? exact_forward_gpu
                                     : ((original_device == Device::GPU) ? output_2d.to(Device::GPU)
                                                                         : output_2d);
        Tensor projected = out_proj->forward(projected_input);
        if (saved_input_rank_ == 1) {
            return projected.reshape({d_model});
        }
        if (saved_input_rank_ == 2) {
            return projected.reshape({seq_len, d_model});
        }
        return projected;
    }

    if (input.shape.size() == 3) {
        const Device original_device = input.get_device();
        const int batch_size = input.shape[0];
        const int seq_len = input.shape[1];
        const int kv_dim = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        Tensor q_flat = q_down_proj->forward(input);
        Tensor kv_flat = kv_down_proj->forward(input);

        if (streaming_inference_ && seq_len == 1) {
#ifdef USE_CUDA
            if (original_device == Device::GPU &&
                q_flat.get_device() == Device::GPU &&
                kv_flat.get_device() == Device::GPU &&
                batch_size > 0 &&
                gpu_custom_kernels_supported()) {
                ensure_kv_cache_capacity(cached_tokens_ + 1, Device::GPU, batch_size);
                Tensor output_gpu({batch_size, 1, d_model}, Device::GPU);
                const size_t q_row_stride = static_cast<size_t>(d_model);
                const size_t kv_row_stride = static_cast<size_t>(2 * kv_dim);
                const size_t cache_row_stride =
                    static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(kv_dim);
                const size_t out_row_stride = static_cast<size_t>(d_model);
                for (int batch = 0; batch < batch_size; ++batch) {
                    launch_gqa_append_kv_cache_kernel(
                        kv_flat.data() + static_cast<size_t>(batch) * kv_row_stride,
                        key_cache_buffer_.data() + static_cast<size_t>(batch) * cache_row_stride,
                        value_cache_buffer_.data() + static_cast<size_t>(batch) * cache_row_stride,
                        cached_tokens_,
                        n_kv_heads,
                        head_dim,
                        theta);
                    launch_gqa_cached_attention_decode_kernel(
                        q_flat.data() + static_cast<size_t>(batch) * q_row_stride,
                        key_cache_buffer_.data() + static_cast<size_t>(batch) * cache_row_stride,
                        value_cache_buffer_.data() + static_cast<size_t>(batch) * cache_row_stride,
                        output_gpu.data() + static_cast<size_t>(batch) * out_row_stride,
                        cached_tokens_ + 1,
                        d_model,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_group_size,
                        theta);
                }
                ++cached_tokens_;

                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("Batched cached GQA attention kernel launch failed: ") +
                        cudaGetErrorString(status));
                }
                if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                    if (std::string(sync_env) == "1") {
                        status = cudaDeviceSynchronize();
                        if (status != cudaSuccess) {
                            throw std::runtime_error(
                                std::string("Batched cached GQA attention kernel sync failed: ") +
                                cudaGetErrorString(status));
                        }
                    }
                }

                return out_proj->forward(output_gpu);
            }
#endif

            Tensor q_cpu = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
            Tensor kv_cpu = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;
            Tensor k_flat({batch_size, 1, kv_dim}, Device::CPU);
            Tensor v_flat({batch_size, 1, kv_dim}, Device::CPU);
            {
                const float* kv_ptr = kv_cpu.data();
                float* k_ptr = k_flat.data();
                float* v_ptr = v_flat.data();
                const size_t token_stride = static_cast<size_t>(2 * kv_dim);
                for (int batch = 0; batch < batch_size; ++batch) {
                    const size_t token_offset =
                        static_cast<size_t>(batch) * token_stride;
                    const size_t out_offset = static_cast<size_t>(batch) *
                                              static_cast<size_t>(kv_dim);
                    std::memcpy(k_ptr + out_offset,
                                kv_ptr + token_offset,
                                static_cast<size_t>(kv_dim) * sizeof(float));
                    std::memcpy(v_ptr + out_offset,
                                kv_ptr + token_offset + static_cast<size_t>(kv_dim),
                                static_cast<size_t>(kv_dim) * sizeof(float));
                }
            }

            Tensor q4 = q_cpu.reshape({batch_size, 1, n_heads, head_dim});
            Tensor k4 = k_flat.reshape({batch_size, 1, n_kv_heads, head_dim});
            Tensor v4 = v_flat.reshape({batch_size, 1, n_kv_heads, head_dim});
            auto [q_rot, k_rot] = apply_rope(q4, k4, cached_tokens_);
            append_kv_cache_batch_tokens(k_rot.data(), Device::CPU, v4.data(), Device::CPU, batch_size);

            Tensor output_cpu({batch_size, 1, d_model}, Device::CPU);
            float* out_ptr = output_cpu.data();
            const float* q_ptr = q_rot.data();
            for (int batch = 0; batch < batch_size; ++batch) {
                std::vector<float> token_scores(static_cast<size_t>(cached_tokens_), 0.0f);
                for (int head = 0; head < n_heads; ++head) {
                    const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                    float max_s = -1e30f;
                    for (int t = 0; t < cached_tokens_; ++t) {
                        float dot = 0.0f;
                        const float* cached_key_token =
                            kv_cache_token_ptr(key_cache_buffer_, batch, t);
                        const size_t q_offset =
                            (((static_cast<size_t>(batch) * 1) * n_heads) + head) *
                            static_cast<size_t>(head_dim);
                        const size_t kv_offset =
                            static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                        for (int d = 0; d < head_dim; ++d) {
                            dot += q_ptr[q_offset + static_cast<size_t>(d)] *
                                   cached_key_token[kv_offset + static_cast<size_t>(d)];
                        }
                        token_scores[static_cast<size_t>(t)] = dot * scale;
                        max_s = std::max(max_s, token_scores[static_cast<size_t>(t)]);
                    }
                    float sum_exp = 0.0f;
                    for (float& score_value : token_scores) {
                        score_value = std::exp(score_value - max_s);
                        sum_exp += score_value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (float& score_value : token_scores) {
                        score_value *= inv_sum;
                    }
                    for (int d = 0; d < head_dim; ++d) {
                        float acc = 0.0f;
                        for (int t = 0; t < cached_tokens_; ++t) {
                            const float* cached_value_token =
                                kv_cache_token_ptr(value_cache_buffer_, batch, t);
                            const size_t kv_offset =
                                static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                            acc += token_scores[static_cast<size_t>(t)] *
                                   cached_value_token[kv_offset + static_cast<size_t>(d)];
                        }
                        const size_t out_offset =
                            (((static_cast<size_t>(batch) * 1) * n_heads) + head) *
                                static_cast<size_t>(head_dim) +
                            static_cast<size_t>(d);
                        out_ptr[out_offset] = acc;
                    }
                }
            }

            Tensor output =
                (original_device == Device::GPU) ? output_cpu.to(Device::GPU) : output_cpu;
            return out_proj->forward(output);
        }

        if (streaming_inference_) {
            throw std::runtime_error(
                "Attention::forward rank-3 batch path only supports streaming for single-token steps");
        }

#ifdef USE_CUDA
        if (original_device == Device::GPU &&
            q_flat.get_device() == Device::GPU &&
            kv_flat.get_device() == Device::GPU &&
            batch_size > 0 &&
            seq_len > 0 &&
            seq_len <= 4096 &&
            gpu_custom_kernels_supported()) {
            Tensor output_gpu({batch_size, seq_len, d_model}, Device::GPU);
            launch_batched_gqa_causal_attention_kernel(q_flat.data(),
                                                       kv_flat.data(),
                                                       output_gpu.data(),
                                                       batch_size,
                                                       seq_len,
                                                       d_model,
                                                       n_heads,
                                                       n_kv_heads,
                                                       head_dim,
                                                       kv_group_size,
                                                       theta);

            cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Batched GQA causal attention kernel launch failed: ") +
                    cudaGetErrorString(status));
            }
            if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                if (std::string(sync_env) == "1") {
                    status = cudaDeviceSynchronize();
                    if (status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("Batched GQA causal attention kernel sync failed: ") +
                            cudaGetErrorString(status));
                    }
                }
            }

            return out_proj->forward(output_gpu);
        }
#endif

        Tensor q_cpu = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
        Tensor kv_cpu = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;

        Tensor k_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        Tensor v_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        {
            const float* kv_ptr = kv_cpu.data();
            float* k_ptr = k_flat.data();
            float* v_ptr = v_flat.data();
            const size_t token_stride = static_cast<size_t>(2 * kv_dim);
            for (int batch = 0; batch < batch_size; ++batch) {
                for (int token = 0; token < seq_len; ++token) {
                    const size_t token_offset =
                        (static_cast<size_t>(batch) * seq_len + token) * token_stride;
                    const size_t out_offset =
                        (static_cast<size_t>(batch) * seq_len + token) *
                        static_cast<size_t>(kv_dim);
                    std::memcpy(
                        k_ptr + out_offset,
                        kv_ptr + token_offset,
                        static_cast<size_t>(kv_dim) * sizeof(float));
                    std::memcpy(
                        v_ptr + out_offset,
                        kv_ptr + token_offset + static_cast<size_t>(kv_dim),
                        static_cast<size_t>(kv_dim) * sizeof(float));
                }
            }
        }

        Tensor q4 = q_cpu.reshape({batch_size, seq_len, n_heads, head_dim});
        Tensor k4 = k_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        Tensor v4 = v_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        auto [q_rot, k_rot] = apply_rope(q4, k4, 0);

        Tensor output_cpu({batch_size, seq_len, d_model}, Device::CPU);
        float* out_ptr = output_cpu.data();
        const float* q_ptr = q_rot.data();
        const float* k_ptr = k_rot.data();
        const float* v_ptr = v4.data();
        std::vector<float> scores(static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len),
                                  0.0f);

        for (int batch = 0; batch < batch_size; ++batch) {
            for (int head = 0; head < n_heads; ++head) {
                const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                std::fill(scores.begin(), scores.end(), 0.0f);

                for (int i = 0; i < seq_len; ++i) {
                    float max_s = -1e30f;
                    for (int j = 0; j < seq_len; ++j) {
                        float score = -1e9f;
                        if (j <= i) {
                            float dot = 0.0f;
                            const size_t q_offset =
                                (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                                static_cast<size_t>(head_dim);
                            const size_t k_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                static_cast<size_t>(head_dim);
                            for (int dim = 0; dim < head_dim; ++dim) {
                                dot += q_ptr[q_offset + dim] * k_ptr[k_offset + dim];
                            }
                            score = dot * scale;
                        }
                        scores[static_cast<size_t>(i) * seq_len + j] = score;
                        max_s = std::max(max_s, score);
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        float value = std::exp(scores[static_cast<size_t>(i) * seq_len + j] - max_s);
                        scores[static_cast<size_t>(i) * seq_len + j] = value;
                        sum_exp += value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (int j = 0; j < seq_len; ++j) {
                        scores[static_cast<size_t>(i) * seq_len + j] *= inv_sum;
                    }

                    for (int dim = 0; dim < head_dim; ++dim) {
                        float acc = 0.0f;
                        for (int j = 0; j < seq_len; ++j) {
                            const size_t v_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                    static_cast<size_t>(head_dim) +
                                static_cast<size_t>(dim);
                            acc += scores[static_cast<size_t>(i) * seq_len + j] * v_ptr[v_offset];
                        }
                        const size_t out_offset =
                            (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                                static_cast<size_t>(head_dim) +
                            static_cast<size_t>(dim);
                        out_ptr[out_offset] = acc;
                    }
                }
            }
        }

        Tensor output =
            (original_device == Device::GPU) ? output_cpu.to(Device::GPU) : output_cpu;
        return out_proj->forward(output);
    }
    // input shape: [seq_len, d_model]
    // Inferimos seq_len a partir das dimensões do tensor
    const Device original_device = input.get_device();
    Tensor input_2d = input;
    if (input.shape.size() == 1) {
        input_2d = input.reshape({1, d_model});
    }
    const int seq_len = input_2d.shape[0];
    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // 1. Projeções Q, K, V
    Tensor q_flat = q_down_proj->forward(input_2d);          // [seq, d_model]
    Tensor kv_flat = kv_down_proj->forward(input_2d);        // [seq, 2*kv_dim]
#ifdef USE_CUDA
    if (original_device == Device::GPU &&
        q_flat.get_device() == Device::GPU &&
        kv_flat.get_device() == Device::GPU &&
        seq_len > 0 &&
        seq_len <= 4096 &&
        gpu_custom_kernels_supported()) {
        if (streaming_inference_ && seq_len == 1) {
            ensure_kv_cache_capacity(cached_tokens_ + 1, Device::GPU);
            Tensor output_gpu({seq_len, d_model}, Device::GPU);
            launch_gqa_append_kv_cache_kernel(
                kv_flat.data(),
                key_cache_buffer_.data(),
                value_cache_buffer_.data(),
                cached_tokens_,
                n_kv_heads,
                head_dim,
                theta);
            launch_gqa_cached_attention_decode_kernel(
                q_flat.data(),
                key_cache_buffer_.data(),
                value_cache_buffer_.data(),
                output_gpu.data(),
                cached_tokens_ + 1,
                d_model,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_group_size,
                theta);
            ++cached_tokens_;

            cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("GQA cached attention kernel launch failed: ") +
                    cudaGetErrorString(status));
            }
            if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                if (std::string(sync_env) == "1") {
                    status = cudaDeviceSynchronize();
                    if (status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("GQA cached attention kernel sync failed: ") +
                            cudaGetErrorString(status));
                    }
                }
            }

            Tensor projected = out_proj->forward(output_gpu);
            return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
        }

        if (!streaming_inference_) {
        Tensor output_gpu({seq_len, d_model}, Device::GPU);
        launch_gqa_causal_attention_kernel(
            q_flat.data(),
            kv_flat.data(),
            output_gpu.data(),
            seq_len,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_group_size,
            theta);

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("GQA causal attention kernel launch failed: ") +
                cudaGetErrorString(status));
        }
        if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
            if (std::string(sync_env) == "1") {
                status = cudaDeviceSynchronize();
                if (status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("GQA causal attention kernel sync failed: ") +
                        cudaGetErrorString(status));
                }
            }
        }

        Tensor projected = out_proj->forward(output_gpu);
        return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
        }
    }
#endif
    Tensor q_cpu = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
    Tensor kv_cpu = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;

    // 2. Separar K e V do kv_flat
    //    kv_flat layout: [seq, d_model | d_model]
    Tensor k_flat({seq_len, kv_dim}, Device::CPU);
    Tensor v_flat({seq_len, kv_dim}, Device::CPU);
    {
        const float* kv_ptr = kv_cpu.data();
        float* k_ptr = k_flat.data();
        float* v_ptr = v_flat.data();
        for (int s = 0; s < seq_len; ++s) {
            // k é a primeira metade, v é a segunda
            std::memcpy(k_ptr + s * kv_dim, kv_ptr + s * 2 * kv_dim,
                        static_cast<size_t>(kv_dim) * sizeof(float));
            std::memcpy(v_ptr + s * kv_dim, kv_ptr + s * 2 * kv_dim + kv_dim,
                        static_cast<size_t>(kv_dim) * sizeof(float));
        }
    }

    // 3. Reshape para [seq, n_heads, head_dim]
    Tensor q3 = q_cpu.reshape({seq_len, n_heads, head_dim});
    Tensor k3 = k_flat.reshape({seq_len, n_kv_heads, head_dim});
    Tensor v3 = v_flat.reshape({seq_len, n_kv_heads, head_dim});

    // 4. Aplicar RoPE em Q e K
    auto [q_rot, k_rot] = apply_rope(q3, k3, streaming_inference_ ? cached_tokens_ : 0);

    // 5. Scored dot-product attention com máscara causal
    //    Resultado: [seq, d_model]
    Tensor output_cpu({seq_len, d_model}, Device::CPU);
    float* out_ptr = output_cpu.data();

    const float* q_rot_ptr = q_rot.data();
    const float* k_rot_ptr = k_rot.data();
    const float* v_ptr = v3.data();

    if (streaming_inference_ && seq_len == 1) {
        append_kv_cache_token(k_rot_ptr, Device::CPU, v_ptr, Device::CPU);

        std::vector<float> token_scores(static_cast<size_t>(cached_tokens_), 0.0f);
        for (int h = 0; h < n_heads; ++h) {
            const int kv_head = std::min(h / kv_group_size, n_kv_heads - 1);
            float max_s = -1e30f;
            for (int t = 0; t < cached_tokens_; ++t) {
                float dot = 0.0f;
                const float* cached_key_token = kv_cache_token_ptr(key_cache_buffer_, t);
                const size_t kv_offset =
                    static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_rot_ptr[h * head_dim + d] * cached_key_token[kv_offset + d];
                }
                token_scores[static_cast<size_t>(t)] = dot * scale;
                max_s = std::max(max_s, token_scores[static_cast<size_t>(t)]);
            }
            float sum_exp = 0.0f;
            for (float& score_value : token_scores) {
                score_value = std::exp(score_value - max_s);
                sum_exp += score_value;
            }
            const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
            for (float& score_value : token_scores) {
                score_value *= inv_sum;
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < cached_tokens_; ++t) {
                    const float* cached_value_token = kv_cache_token_ptr(value_cache_buffer_, t);
                    const size_t kv_offset =
                        static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                    acc += token_scores[static_cast<size_t>(t)] * cached_value_token[kv_offset + d];
                }
                out_ptr[h * head_dim + d] = acc;
            }
        }
    } else {
        if (streaming_inference_) {
            clear_kv_cache();
            for (int token_index = 0; token_index < seq_len; ++token_index) {
                append_kv_cache_token(k_rot_ptr + token_index * kv_dim,
                                      Device::CPU,
                                      v_ptr + token_index * kv_dim,
                                      Device::CPU);
            }
        }
        std::vector<float> scores(seq_len * seq_len, 0.0f);
        for (int h = 0; h < n_heads; ++h) {
        // Extrair cabeça h de q_rot, k_rot, v3: cada um [seq, head_dim]
        // Calcular scores: [seq, seq] = Q_h @ K_h^T * scale
        std::fill(scores.begin(), scores.end(), 0.0f);
        const float* qh = q_rot.data() + h * head_dim;   // offset na dim de cabeça

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                if (j > i) {
                    // Máscara causal: tokens futuros recebem -inf
                    scores[i * seq_len + j] = -1e9f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    // Stride: [seq, n_heads, head_dim] → posição head h, pos seq i
                    const int kv_head = std::min(h / kv_group_size, n_kv_heads - 1);
                    dot += q_rot_ptr[i * n_heads * head_dim + h * head_dim + d]
                         * k_rot_ptr[j * n_kv_heads * head_dim + kv_head * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }

            // Softmax sobre eixo j (linha i)
            float max_s = *std::max_element(scores.data() + i * seq_len,
                                             scores.data() + i * seq_len + seq_len);
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_s);
                sum_exp += scores[i * seq_len + j];
            }
            float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
            for (int j = 0; j < seq_len; ++j)
                scores[i * seq_len + j] *= inv_sum;

            // Acumular: out[i, h] = sum_j(attn[i,j] * V[j, h])
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    const int kv_head = std::min(h / kv_group_size, n_kv_heads - 1);
                    acc += scores[i * seq_len + j]
                         * v_ptr[j * n_kv_heads * head_dim + kv_head * head_dim + d];
                }
                // Layout de saída: [seq, n_heads, head_dim]
                out_ptr[i * n_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }

    // 6. Reshape [seq, n_heads, head_dim] → [seq, d_model] e projetar
    }
    Tensor output = (original_device == Device::GPU) ? output_cpu.to(Device::GPU) : output_cpu;
    Tensor output2d = output.reshape({seq_len, d_model});
    Tensor projected = out_proj->forward(output2d);
    return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
}

Tensor Attention::backward(const Tensor& dy, Context* ctx) {
    (void)ctx;
    const bool has_exact_cache = saved_q_rot_.size > 0 &&
                                 saved_k_rot_.size > 0 &&
                                 saved_v_heads_.size > 0 &&
                                 !saved_valid_lengths_.empty();
    if (has_exact_cache) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int batch_size = saved_q_rot_.shape[0];
        const int seq_len = saved_q_rot_.shape[1];
        const int kv_dim = n_kv_heads * head_dim;

        Tensor grad_out = out_proj->backward(dy);
        Tensor grad_out_host = (grad_out.get_device() == Device::GPU) ? grad_out.cpu() : grad_out;
        Tensor grad_heads = grad_out_host.reshape({batch_size, seq_len, n_heads, head_dim});

        Tensor q_host =
            (saved_q_rot_.get_device() == Device::GPU) ? saved_q_rot_.cpu() : saved_q_rot_;
        Tensor k_host =
            (saved_k_rot_.get_device() == Device::GPU) ? saved_k_rot_.cpu() : saved_k_rot_;
        Tensor v_host =
            (saved_v_heads_.get_device() == Device::GPU) ? saved_v_heads_.cpu() : saved_v_heads_;
        Tensor grad_q_rot({batch_size, seq_len, n_heads, head_dim}, Device::CPU);
        Tensor grad_k_rot({batch_size, seq_len, n_kv_heads, head_dim}, Device::CPU);
        Tensor grad_v({batch_size, seq_len, n_kv_heads, head_dim}, Device::CPU);

        float* grad_q_ptr = grad_q_rot.data();
        float* grad_k_ptr = grad_k_rot.data();
        float* grad_v_ptr = grad_v.data();
        const float* q_ptr = q_host.data();
        const float* k_ptr = k_host.data();
        const float* v_ptr = v_host.data();
        const float* grad_head_ptr = grad_heads.data();
        std::vector<float> d_probs(static_cast<size_t>(seq_len), 0.0f);
        std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);
        std::vector<float> probs(static_cast<size_t>(seq_len), 0.0f);

        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len =
                std::clamp(saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
            for (int head = 0; head < n_heads; ++head) {
                const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                for (int i = 0; i < valid_len; ++i) {
                    const size_t q_offset =
                        (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                        static_cast<size_t>(head_dim);
                    const size_t grad_out_offset =
                        (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                        static_cast<size_t>(head_dim);

                    float max_s = -1e30f;
                    for (int j = 0; j < seq_len; ++j) {
                        float score = -1e9f;
                        if (j < valid_len && j <= i) {
                            float dot = 0.0f;
                            const size_t k_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                static_cast<size_t>(head_dim);
                            for (int dim = 0; dim < head_dim; ++dim) {
                                dot += q_ptr[q_offset + static_cast<size_t>(dim)] *
                                       k_ptr[k_offset + static_cast<size_t>(dim)];
                            }
                            score = dot * scale;
                        }
                        scores[static_cast<size_t>(j)] = score;
                        max_s = std::max(max_s, score);
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        const float value =
                            (j < valid_len && j <= i)
                                ? std::exp(scores[static_cast<size_t>(j)] - max_s)
                                : 0.0f;
                        probs[static_cast<size_t>(j)] = value;
                        sum_exp += value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (int j = 0; j < seq_len; ++j) {
                        probs[static_cast<size_t>(j)] *= inv_sum;
                    }

                    float row_dot = 0.0f;
                    for (int j = 0; j < valid_len; ++j) {
                        const size_t v_offset =
                            (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) + kv_head) *
                            static_cast<size_t>(head_dim);
                        float dot = 0.0f;
                        for (int dim = 0; dim < head_dim; ++dim) {
                            dot += grad_head_ptr[grad_out_offset + static_cast<size_t>(dim)] *
                                   v_ptr[v_offset + static_cast<size_t>(dim)];
                        }
                        d_probs[static_cast<size_t>(j)] = dot;
                        row_dot += dot * probs[static_cast<size_t>(j)];
                    }

                    for (int j = 0; j < valid_len; ++j) {
                        const float prob = probs[static_cast<size_t>(j)];
                        const float d_score = prob * (d_probs[static_cast<size_t>(j)] - row_dot);
                        const size_t k_offset =
                            (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) + kv_head) *
                            static_cast<size_t>(head_dim);
                        for (int dim = 0; dim < head_dim; ++dim) {
                            const size_t dim_offset = static_cast<size_t>(dim);
                            grad_q_ptr[q_offset + dim_offset] +=
                                scale * d_score * k_ptr[k_offset + dim_offset];
                            grad_k_ptr[k_offset + dim_offset] +=
                                scale * d_score * q_ptr[q_offset + dim_offset];
                            grad_v_ptr[k_offset + dim_offset] +=
                                prob * grad_head_ptr[grad_out_offset + dim_offset];
                        }
                    }
                }
            }
        }

        auto [grad_q_pre_rope, grad_k_pre_rope] = apply_rope_backward(grad_q_rot, grad_k_rot, 0);

        Tensor grad_q_input_cpu({batch_size, seq_len, d_model}, Device::CPU);
        Tensor grad_kv_input_cpu({batch_size, seq_len, 2 * kv_dim}, Device::CPU);
        float* grad_q_input_ptr = grad_q_input_cpu.data();
        float* grad_kv_input_ptr = grad_kv_input_cpu.data();
        const float* grad_q_pre_ptr = grad_q_pre_rope.data();
        const float* grad_k_pre_ptr = grad_k_pre_rope.data();
        const float* grad_v_pre_ptr = grad_v.data();

        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len =
                std::clamp(saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
            for (int token = 0; token < valid_len; ++token) {
                const size_t q_token_offset =
                    (static_cast<size_t>(batch) * seq_len + token) * static_cast<size_t>(d_model);
                for (int head = 0; head < n_heads; ++head) {
                    const size_t head_offset =
                        (((static_cast<size_t>(batch) * seq_len + token) * n_heads) + head) *
                        static_cast<size_t>(head_dim);
                    const size_t out_offset = q_token_offset +
                                              static_cast<size_t>(head) *
                                                  static_cast<size_t>(head_dim);
                    std::memcpy(grad_q_input_ptr + out_offset,
                                grad_q_pre_ptr + head_offset,
                                static_cast<size_t>(head_dim) * sizeof(float));
                }

                const size_t kv_token_offset =
                    (static_cast<size_t>(batch) * seq_len + token) *
                    static_cast<size_t>(2 * kv_dim);
                for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
                    const size_t head_offset =
                        (((static_cast<size_t>(batch) * seq_len + token) * n_kv_heads) + kv_head) *
                        static_cast<size_t>(head_dim);
                    const size_t k_out_offset = kv_token_offset +
                                                static_cast<size_t>(kv_head) *
                                                    static_cast<size_t>(head_dim);
                    const size_t v_out_offset = k_out_offset + static_cast<size_t>(kv_dim);
                    std::memcpy(grad_kv_input_ptr + k_out_offset,
                                grad_k_pre_ptr + head_offset,
                                static_cast<size_t>(head_dim) * sizeof(float));
                    std::memcpy(grad_kv_input_ptr + v_out_offset,
                                grad_v_pre_ptr + head_offset,
                                static_cast<size_t>(head_dim) * sizeof(float));
                }
            }
        }

        Tensor grad_q_input = grad_q_input_cpu;
        Tensor grad_kv_input = grad_kv_input_cpu;
        if (grad_out.get_device() == Device::GPU) {
            grad_q_input = grad_q_input.to(Device::GPU);
            grad_kv_input = grad_kv_input.to(Device::GPU);
        }

        Tensor grad_q = q_down_proj->backward(grad_q_input);
        Tensor grad_kv = kv_down_proj->backward(grad_kv_input);
        Tensor grad_total = grad_q.add(grad_kv);
        if (saved_input_rank_ == 1) {
            return grad_total.reshape({d_model});
        }
        if (saved_input_rank_ == 2) {
            return grad_total.reshape({seq_len, d_model});
        }
        return grad_total;
    }
    {
    Tensor grad_out = out_proj->backward(dy);
    Tensor grad_q = q_down_proj->backward(grad_out);

    const int kv_dim = n_kv_heads * head_dim;
    const int rank = static_cast<int>(grad_out.shape.size());
    const int rows = grad_out.size / std::max(d_model, 1);
    Tensor grad_heads = grad_out.reshape({rows, n_heads, head_dim});
    Tensor grad_kv_proj = Tensor::zeros({rows, 2 * kv_dim}, grad_out.get_device());

    const float* grad_head_ptr = grad_heads.data();
    float* grad_kv_ptr = grad_kv_proj.data();
    for (int row = 0; row < rows; ++row) {
        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            const int start_head = kv_head * kv_group_size;
            const int end_head = std::min(start_head + kv_group_size, n_heads);
            const int group_count = std::max(end_head - start_head, 1);
            for (int dim = 0; dim < head_dim; ++dim) {
                float accum = 0.0f;
                for (int head = start_head; head < end_head; ++head) {
                    const size_t offset =
                        ((static_cast<size_t>(row) * n_heads) + head) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    accum += grad_head_ptr[offset];
                }
                const float shared_grad = accum / static_cast<float>(group_count);
                const size_t k_offset =
                    static_cast<size_t>(row) * static_cast<size_t>(2 * kv_dim) +
                    static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim) +
                    static_cast<size_t>(dim);
                const size_t v_offset = k_offset + static_cast<size_t>(kv_dim);
                grad_kv_ptr[k_offset] = shared_grad;
                grad_kv_ptr[v_offset] = shared_grad;
            }
        }
    }

    Tensor grad_kv_input = grad_kv_proj;
    if (rank == 3) {
        grad_kv_input = grad_kv_proj.reshape({dy.shape[0], dy.shape[1], 2 * kv_dim});
    } else if (rank == 1) {
        grad_kv_input = grad_kv_proj.reshape({2 * kv_dim});
    }

    Tensor grad_kv = kv_down_proj->backward(grad_kv_input);
    return grad_q.add(grad_kv);
    }
    // Gradiente através da projeção de saída
    Tensor grad_out = out_proj->backward(dy);
    // Propagar gradiente através das projeções Q e KV
    // Para MHA completo, o gradiente se divide entre n_heads
    Tensor grad_q  = q_down_proj->backward(grad_out);
    Tensor grad_kv = kv_down_proj->backward(grad_out);
    // Somar contribuições (chain rule simplificada: grad passa pelas duas projeções)
    return grad_q.add(grad_kv.slice(0, 0, grad_q.shape[0]));
}

void Attention::to(Device dev) {
    if (q_down_proj) {
        q_down_proj->to(dev);
    }
    if (kv_down_proj) {
        kv_down_proj->to(dev);
    }
    if (out_proj) {
        out_proj->to(dev);
    }
}

std::vector<Parameter*> Attention::parameters() {
    std::vector<Parameter*> params;
    if (q_down_proj) {
        auto q = q_down_proj->parameters();
        prefix_parameter_names(q, "q_down_proj.");
        params.insert(params.end(), q.begin(), q.end());
    }
    if (kv_down_proj) {
        auto kv = kv_down_proj->parameters();
        prefix_parameter_names(kv, "kv_down_proj.");
        params.insert(params.end(), kv.begin(), kv.end());
    }
    if (out_proj) {
        auto out = out_proj->parameters();
        prefix_parameter_names(out, "out_proj.");
        params.insert(params.end(), out.begin(), out.end());
    }
    return params;
}

void Attention::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (q_down_proj) out.push_back(q_down_proj.get());
    if (kv_down_proj) out.push_back(kv_down_proj.get());
    if (out_proj) out.push_back(out_proj.get());
}

void Attention::clear_kv_cache() {
    cached_tokens_ = 0;
    cached_batch_size_ = 0;
    cache_capacity_tokens_ = 0;
    key_cache_buffer_ = Tensor();
    value_cache_buffer_ = Tensor();
}

void Attention::ensure_kv_cache_capacity(int required_tokens, Device device, int batch_size) {
    const int token_width = n_kv_heads * head_dim;
    const int normalized_batch = std::max(batch_size, 1);
    if (required_tokens <= cache_capacity_tokens_ &&
        cached_batch_size_ == normalized_batch &&
        key_cache_buffer_.size > 0 &&
        value_cache_buffer_.size > 0 &&
        key_cache_buffer_.data_ptr.use_count() == 1 &&
        value_cache_buffer_.data_ptr.use_count() == 1) {
        return;
    }

    const int target_capacity = std::max(
        required_tokens,
        cache_capacity_tokens_ > 0 ? cache_capacity_tokens_ + cache_page_tokens_ : cache_page_tokens_);
    Tensor next_key({normalized_batch, target_capacity, token_width}, device);
    Tensor next_value({normalized_batch, target_capacity, token_width}, device);

    if (cached_tokens_ > 0 && key_cache_buffer_.size > 0 && value_cache_buffer_.size > 0) {
        const int copy_batch = std::min(std::max(cached_batch_size_, 1), normalized_batch);
        const size_t bytes = static_cast<size_t>(copy_batch) *
                             static_cast<size_t>(cached_tokens_) *
                             static_cast<size_t>(token_width) * sizeof(float);
        copy_float_bytes_device_safe(next_key.data(),
                                     next_key.get_device(),
                                     key_cache_buffer_.data(),
                                     key_cache_buffer_.get_device(),
                                     bytes);
        copy_float_bytes_device_safe(next_value.data(),
                                     next_value.get_device(),
                                     value_cache_buffer_.data(),
                                     value_cache_buffer_.get_device(),
                                     bytes);
    }

    key_cache_buffer_ = next_key;
    value_cache_buffer_ = next_value;
    cached_batch_size_ = normalized_batch;
    cache_capacity_tokens_ = target_capacity;
}

void Attention::append_kv_cache_token(const float* key_ptr,
                                      Device key_device,
                                      const float* value_ptr,
                                      Device value_device) {
    const int token_width = n_kv_heads * head_dim;
    if (token_width <= 0) {
        return;
    }

    ensure_kv_cache_capacity(cached_tokens_ + 1,
                             key_cache_buffer_.size > 0 ? key_cache_buffer_.get_device()
                                                        : Device::CPU,
                             1);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    float* key_dst =
        key_cache_buffer_.data() + static_cast<size_t>(cached_tokens_) * token_width;
    float* value_dst =
        value_cache_buffer_.data() + static_cast<size_t>(cached_tokens_) * token_width;
    if (cached_batch_size_ > 1) {
        key_dst = key_cache_buffer_.data() + batch_stride +
                  static_cast<size_t>(cached_tokens_) * token_width;
        value_dst = value_cache_buffer_.data() + batch_stride +
                    static_cast<size_t>(cached_tokens_) * token_width;
    }
    copy_float_bytes_device_safe(key_dst,
                                 key_cache_buffer_.get_device(),
                                 key_ptr,
                                 key_device,
                                 static_cast<size_t>(token_width) * sizeof(float));
    copy_float_bytes_device_safe(value_dst,
                                 value_cache_buffer_.get_device(),
                                 value_ptr,
                                 value_device,
                                 static_cast<size_t>(token_width) * sizeof(float));
    ++cached_tokens_;
}

void Attention::append_kv_cache_batch_tokens(const float* key_ptr,
                                             Device key_device,
                                             const float* value_ptr,
                                             Device value_device,
                                             int batch_size) {
    const int token_width = n_kv_heads * head_dim;
    const int normalized_batch = std::max(batch_size, 1);
    if (token_width <= 0) {
        return;
    }
    ensure_kv_cache_capacity(cached_tokens_ + 1,
                             key_cache_buffer_.size > 0 ? key_cache_buffer_.get_device()
                                                        : Device::CPU,
                             normalized_batch);
    const size_t token_bytes = static_cast<size_t>(token_width) * sizeof(float);
    const size_t src_stride = static_cast<size_t>(token_width);
    const size_t dst_batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    for (int batch = 0; batch < normalized_batch; ++batch) {
        float* key_dst = key_cache_buffer_.data() +
                         static_cast<size_t>(batch) * dst_batch_stride +
                         static_cast<size_t>(cached_tokens_) * static_cast<size_t>(token_width);
        float* value_dst = value_cache_buffer_.data() +
                           static_cast<size_t>(batch) * dst_batch_stride +
                           static_cast<size_t>(cached_tokens_) * static_cast<size_t>(token_width);
        const float* key_src = key_ptr + static_cast<size_t>(batch) * src_stride;
        const float* value_src = value_ptr + static_cast<size_t>(batch) * src_stride;
        copy_float_bytes_device_safe(key_dst,
                                     key_cache_buffer_.get_device(),
                                     key_src,
                                     key_device,
                                     token_bytes);
        copy_float_bytes_device_safe(value_dst,
                                     value_cache_buffer_.get_device(),
                                     value_src,
                                     value_device,
                                     token_bytes);
    }
    ++cached_tokens_;
}

const float* Attention::kv_cache_token_ptr(
    const Tensor& cache,
    int token_index) const {
    const size_t token_width = static_cast<size_t>(n_kv_heads * head_dim);
    return cache.data() + static_cast<size_t>(token_index) * token_width;
}

const float* Attention::kv_cache_token_ptr(
    const Tensor& cache,
    int batch_index,
    int token_index) const {
    const size_t token_width = static_cast<size_t>(n_kv_heads * head_dim);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    return cache.data() + static_cast<size_t>(batch_index) * batch_stride +
           static_cast<size_t>(token_index) * token_width;
}

void Attention::set_streaming_mode(bool enabled) {
    streaming_inference_ = enabled;
    clear_kv_cache();
}

void Attention::reset() {
    clear_kv_cache();
}

AttentionCacheSnapshot Attention::snapshot_cache() const {
    AttentionCacheSnapshot snapshot;
    snapshot.enabled = streaming_inference_;
    snapshot.cached_tokens = cached_tokens_;
    snapshot.cache_page_tokens = cache_page_tokens_;
    snapshot.cache_capacity_tokens = cache_capacity_tokens_;
    if (cached_batch_size_ <= 1) {
        snapshot.key_cache = key_cache_buffer_;
        snapshot.value_cache = value_cache_buffer_;
    } else if (key_cache_buffer_.size > 0 && value_cache_buffer_.size > 0) {
        const int token_width = n_kv_heads * head_dim;
        snapshot.key_cache = Tensor({cache_capacity_tokens_, token_width}, key_cache_buffer_.get_device());
        snapshot.value_cache =
            Tensor({cache_capacity_tokens_, token_width}, value_cache_buffer_.get_device());
        const size_t bytes = static_cast<size_t>(cache_capacity_tokens_) *
                             static_cast<size_t>(token_width) * sizeof(float);
        copy_float_bytes_device_safe(snapshot.key_cache.data(),
                                     snapshot.key_cache.get_device(),
                                     key_cache_buffer_.data(),
                                     key_cache_buffer_.get_device(),
                                     bytes);
        copy_float_bytes_device_safe(snapshot.value_cache.data(),
                                     snapshot.value_cache.get_device(),
                                     value_cache_buffer_.data(),
                                     value_cache_buffer_.get_device(),
                                     bytes);
    }
    return snapshot;
}

std::vector<AttentionCacheSnapshot> Attention::snapshot_cache_batch() const {
    std::vector<AttentionCacheSnapshot> snapshots;
    if (cached_batch_size_ <= 0 || key_cache_buffer_.size == 0 || value_cache_buffer_.size == 0) {
        return snapshots;
    }
    const int token_width = n_kv_heads * head_dim;
    const size_t bytes = static_cast<size_t>(cache_capacity_tokens_) *
                         static_cast<size_t>(token_width) * sizeof(float);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    snapshots.reserve(static_cast<size_t>(cached_batch_size_));
    for (int batch = 0; batch < cached_batch_size_; ++batch) {
        AttentionCacheSnapshot snapshot;
        snapshot.enabled = streaming_inference_;
        snapshot.cached_tokens = cached_tokens_;
        snapshot.cache_page_tokens = cache_page_tokens_;
        snapshot.cache_capacity_tokens = cache_capacity_tokens_;
        snapshot.key_cache =
            Tensor({cache_capacity_tokens_, token_width}, key_cache_buffer_.get_device());
        snapshot.value_cache =
            Tensor({cache_capacity_tokens_, token_width}, value_cache_buffer_.get_device());
        copy_float_bytes_device_safe(snapshot.key_cache.data(),
                                     snapshot.key_cache.get_device(),
                                     key_cache_buffer_.data() +
                                         static_cast<size_t>(batch) * batch_stride,
                                     key_cache_buffer_.get_device(),
                                     bytes);
        copy_float_bytes_device_safe(snapshot.value_cache.data(),
                                     snapshot.value_cache.get_device(),
                                     value_cache_buffer_.data() +
                                         static_cast<size_t>(batch) * batch_stride,
                                     value_cache_buffer_.get_device(),
                                     bytes);
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void Attention::restore_cache(const AttentionCacheSnapshot& snapshot) {
    streaming_inference_ = snapshot.enabled;
    cached_tokens_ = snapshot.cached_tokens;
    cache_page_tokens_ = std::max(snapshot.cache_page_tokens, 1);
    cache_capacity_tokens_ = std::max(snapshot.cache_capacity_tokens, 0);
    cached_batch_size_ = 1;
    key_cache_buffer_ = snapshot.key_cache;
    value_cache_buffer_ = snapshot.value_cache;
}

void Attention::restore_cache_batch(const std::vector<AttentionCacheSnapshot>& snapshots) {
    if (snapshots.empty()) {
        clear_kv_cache();
        return;
    }
    streaming_inference_ = snapshots.front().enabled;
    cached_tokens_ = snapshots.front().cached_tokens;
    cache_page_tokens_ = std::max(snapshots.front().cache_page_tokens, 1);
    cache_capacity_tokens_ = std::max(snapshots.front().cache_capacity_tokens, 0);
    cached_batch_size_ = static_cast<int>(snapshots.size());
    const Device cache_device =
        snapshots.front().key_cache.size > 0 ? snapshots.front().key_cache.get_device() : Device::CPU;
    const int token_width = n_kv_heads * head_dim;
    key_cache_buffer_ = Tensor({cached_batch_size_, cache_capacity_tokens_, token_width}, cache_device);
    value_cache_buffer_ =
        Tensor({cached_batch_size_, cache_capacity_tokens_, token_width}, cache_device);
    const size_t bytes = static_cast<size_t>(cache_capacity_tokens_) *
                         static_cast<size_t>(token_width) * sizeof(float);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    for (size_t batch = 0; batch < snapshots.size(); ++batch) {
        copy_float_bytes_device_safe(key_cache_buffer_.data() + batch * batch_stride,
                                     key_cache_buffer_.get_device(),
                                     snapshots[batch].key_cache.data(),
                                     snapshots[batch].key_cache.get_device(),
                                     bytes);
        copy_float_bytes_device_safe(value_cache_buffer_.data() + batch * batch_stride,
                                     value_cache_buffer_.get_device(),
                                     snapshots[batch].value_cache.data(),
                                     snapshots[batch].value_cache.get_device(),
                                     bytes);
    }
}

MoERouter::MoERouter(int d_model_value, int n, int k)
    : num_experts(n),
      top_k(std::min(k, n)),
      aux_loss_coef(0.01f),
      expert_loads(n, 0.0f),
      gate(std::make_unique<BitLinear>(d_model_value, n, false)) {}

std::pair<Tensor, Tensor> MoERouter::forward(const Tensor& x) {
    Tensor logits = gate->forward(x);
    Tensor weights = logits.softmax(-1);

    std::fill(expert_loads.begin(), expert_loads.end(), 0.0f);
    Tensor weights_host = weights.get_device() == Device::GPU ? weights.cpu() : weights;
    float* weight_ptr = weights_host.data();
    const int rows = x.size / x.shape.back();
    if (top_k > 0 && top_k < num_experts) {
        std::vector<int> ranked(static_cast<size_t>(num_experts));
        std::vector<char> keep(static_cast<size_t>(num_experts), 0);
        std::iota(ranked.begin(), ranked.end(), 0);
        for (int row = 0; row < rows; ++row) {
            std::fill(keep.begin(), keep.end(), 0);
            std::partial_sort(
                ranked.begin(),
                ranked.begin() + top_k,
                ranked.end(),
                [&](int lhs, int rhs) {
                    return weight_ptr[row * num_experts + lhs] >
                           weight_ptr[row * num_experts + rhs];
                });
            float selected_sum = 0.0f;
            for (int rank = 0; rank < top_k; ++rank) {
                keep[static_cast<size_t>(ranked[rank])] = 1;
                selected_sum += weight_ptr[row * num_experts + ranked[rank]];
            }
            const float inv_selected = 1.0f / std::max(selected_sum, 1e-9f);
            for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                float& value = weight_ptr[row * num_experts + expert_idx];
                value = keep[static_cast<size_t>(expert_idx)] ? (value * inv_selected) : 0.0f;
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            expert_loads[expert_idx] += weight_ptr[row * num_experts + expert_idx];
        }
    }

    if (weights.get_device() == Device::GPU) {
        weights.copy_from(weights_host.to(Device::GPU));
    } else {
        weights = weights_host;
    }

    return {logits, weights};
}

Tensor MoERouter::backward(const Tensor& grad_logits) {
    return gate ? gate->backward(grad_logits) : grad_logits;
}

void MoERouter::to(Device dev) {
    if (gate) {
        gate->to(dev);
    }
}

std::vector<Parameter*> MoERouter::parameters() {
    std::vector<Parameter*> params;
    if (gate) {
        auto gate_params = gate->parameters();
        prefix_parameter_names(gate_params, "gate.");
        params.insert(params.end(), gate_params.begin(), gate_params.end());
    }
    if (shared_expert_gate) {
        auto shared_gate_params = shared_expert_gate->parameters();
        prefix_parameter_names(shared_gate_params, "shared_expert_gate.");
        params.insert(params.end(), shared_gate_params.begin(), shared_gate_params.end());
    }
    if (shared_expert_up) {
        auto shared_up_params = shared_expert_up->parameters();
        prefix_parameter_names(shared_up_params, "shared_expert_up.");
        params.insert(params.end(), shared_up_params.begin(), shared_up_params.end());
    }
    if (shared_expert_down) {
        auto shared_down_params = shared_expert_down->parameters();
        prefix_parameter_names(shared_down_params, "shared_expert_down.");
        params.insert(params.end(), shared_down_params.begin(), shared_down_params.end());
    }
    return params;
}

void MoERouter::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (gate) out.push_back(gate.get());
    if (shared_expert_gate) out.push_back(shared_expert_gate.get());
    if (shared_expert_up) out.push_back(shared_expert_up.get());
    if (shared_expert_down) out.push_back(shared_expert_down.get());
}

float MoERouter::compute_aux_loss() {
    if (expert_loads.empty()) {
        return 0.0f;
    }
    const float mean_load =
        std::accumulate(expert_loads.begin(), expert_loads.end(), 0.0f) /
        static_cast<float>(expert_loads.size());
    float variance = 0.0f;
    for (float load : expert_loads) {
        const float diff = load - mean_load;
        variance += diff * diff;
    }
    return aux_loss_coef * variance / static_cast<float>(expert_loads.size());
}

std::vector<int> D2FDecoder::generate(
    const std::vector<int>& prompt,
    int length,
    Context* ctx,
    float temperature,
    float top_p,
    int   top_k,
    int   eos_token_id)
{
    std::vector<int> output = prompt;
    const bool use_streaming =
        model != nullptr && model->supports_streaming_inference() && !prompt.empty();
    const bool restore_training_mode = model != nullptr && model->training_mode();

    // RNG local, seed por tempo para não repetir;
    const uint64_t rng_sequence =
        hash_token_sequence(prompt) ^
        (static_cast<uint64_t>(std::max(length, 0)) << 32) ^
        static_cast<uint64_t>(std::max(eos_token_id, 0));
    auto seeded_rng =
        determinism::DeterminismManager::instance().get_rng_for_operation(
            "jamba_decoder", "generate", rng_sequence);
    std::mt19937 rng(static_cast<uint32_t>(seeded_rng()));

    try {
        if (model != nullptr) {
            model->set_training_mode(false);
        }
        if (use_streaming) {
            model->reset_session();
            model->set_streaming_inference(true);
        }

        Tensor logits;
        if (!output.empty()) {
            logits = model->forward_ids(output, ctx);
        }

        for (int i = 0; i < length; ++i) {
            // Abort check
            if (ctx && ctx->abort_signal &&
                ctx->abort_signal->load(std::memory_order_relaxed)) {
                throw AbortException();
            }

            // 1. Forward pass: obtém logits sobre vocabulário inteiro
            if (i > 0 || output.empty()) {
                const std::vector<int> step_tokens =
                    (use_streaming && !output.empty()) ? std::vector<int>{output.back()} : output;
                logits = model->forward_ids(step_tokens, ctx);
            }
            if (logits.size == 0) break;

            // logits shape: [seq_len, vocab_size] → pegar último token
            Tensor host_logits = logits.get_device() == Device::GPU ? logits.cpu() : logits;
            const int vocab_size = host_logits.shape.back();
            const int last_offset = host_logits.size - vocab_size;
            const float* raw = host_logits.data() + last_offset;

            // 2. Aplicar temperatura (escala os logits)
            const float temp = std::max(temperature, 1e-6f);
            std::vector<float> scaled(vocab_size);
            for (int v = 0; v < vocab_size; ++v)
                scaled[v] = raw[v] / temp;

            // 3. Top-k filtering: manter apenas os k maiores logits
            if (top_k > 0 && top_k < vocab_size) {
                // Encontrar o k-ésimo maior valor
                std::vector<float> sorted_vals = scaled;
                std::nth_element(sorted_vals.begin(),
                                 sorted_vals.begin() + (vocab_size - top_k),
                                 sorted_vals.end());
                float cutoff = sorted_vals[vocab_size - top_k];
                for (int v = 0; v < vocab_size; ++v)
                    if (scaled[v] < cutoff) scaled[v] = -1e9f;
            }

            // 4. Converter para probabilidades (softmax numericamente estável)
            float max_val = *std::max_element(scaled.begin(), scaled.end());
            std::vector<float> probs(vocab_size);
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab_size; ++v) {
                probs[v] = std::exp(scaled[v] - max_val);
                sum_exp += probs[v];
            }
            float inv = 1.0f / std::max(sum_exp, 1e-9f);
            for (int v = 0; v < vocab_size; ++v) probs[v] *= inv;

            // 5. Top-p (nucleus) filtering: manter tokens até p acumulado
            if (top_p < 1.0f && top_p > 0.0f) {
                // Ordenar por probabilidade decrescente
                std::vector<int> indices(vocab_size);
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(),
                          [&](int a, int b){ return probs[a] > probs[b]; });
                float cumulative = 0.0f;
                for (int idx : indices) {
                    if (cumulative >= top_p) probs[idx] = 0.0f;
                    else cumulative += probs[idx];
                }
                // Renormalizar após filtro
                float new_sum = 0.0f;
                for (float p : probs) new_sum += p;
                inv = 1.0f / std::max(new_sum, 1e-9f);
                for (float& p : probs) p *= inv;
            }

            // 6. Sample categórico
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int next_token = dist(rng);
            output.push_back(next_token);

            // 7. Critério de parada: EOS token
            if (next_token == eos_token_id) break;
        }
    } catch (const AbortException& e) {
        std::cerr << "[NSOS:Decoder] " << e.what() << "\n";
        // Retorna tokens gerados até o momento do cancelamento
    }
    if (use_streaming) {
        model->set_streaming_inference(false);
    }
    if (model != nullptr && restore_training_mode) {
        model->set_training_mode(true);
    }
    return output;
}

} // namespace nsos
