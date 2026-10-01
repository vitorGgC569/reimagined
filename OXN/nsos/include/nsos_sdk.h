#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
#include "tokenizer.h"
#include "trainer.h"
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nsos {

// Use ModelConfig from nsos_config.h instead of redefining

struct GenerationOptions {
    // Default raised from 50 -> 512 (2026-05-17).  The old default
    // forced users to override on every call to get a useful chat
    // response (50 tokens is roughly one short sentence).  With the
    // KV cache pre-allocation fix in InferenceEngine::generate and
    // the always-on streaming path, generating 512 tokens is now a
    // 512-step O(1) loop instead of the O(N^2) regression the small
    // default was masking.  Users who want short outputs override
    // explicitly; the new default matches frontier chat APIs
    // (Anthropic / OpenAI default around 1024).
    int max_tokens = 512;
    int min_new_tokens = 1;
    float temperature = 0.7f;
    float top_p = 0.92f;
    int top_k = 0;
    int eos_token_id = 0;
    int max_context_tokens = 4096;
    bool suppress_control_tokens_at_start = true;
    float repetition_penalty = 1.08f;
    int no_repeat_ngram_size = 3;
    bool stream = false;
};

struct GenerationMetrics {
    size_t prompt_tokens_total = 0;
    size_t prompt_tokens_used = 0;
    size_t generated_tokens = 0;
    size_t batch_size = 1;
    double elapsed_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double sampler_ms = 0.0;
    double prompt_tokens_per_sec = 0.0;
    double decode_tokens_per_sec = 0.0;
    double total_tokens_per_sec = 0.0;
    bool used_streaming = false;
    bool loaded_from_pack = false;
    size_t mamba_fast_path_hits = 0;
    size_t mamba_fast_path_fallbacks = 0;
    size_t mamba_stream_priming_gpu_calls = 0;
    size_t mamba_stream_priming_host_fallbacks = 0;
    std::string mamba_last_fallback_reason;
};

// Explicit pack-load policy. Only fields deliberately excluded from the
// checkpoint architecture digest may be overridden at load time. std::nullopt
// means "preserve the value authored in the pack"; unlike ModelConfig default
// comparison, this can represent an explicit false/default value unambiguously.
struct ModelLoadOptions {
    std::optional<bool> use_cuda;
    std::optional<int> default_batch_size;
    std::optional<int> mcts_simulations;
    std::optional<int> mcts_depth;
    std::optional<std::string> checkpoint_path;
};

class InferenceEngine {
public:
    std::unique_ptr<JambaModel> model;
    std::unique_ptr<Trainer> trainer;
    Tokenizer tokenizer;
    ModelConfig config;
    
    InferenceEngine() = default;
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&& other) = delete;
    InferenceEngine& operator=(InferenceEngine&& other) noexcept;
    // One argument: a pack is authoritative and receives no runtime override;
    // a raw checkpoint uses a default ModelConfig.
    bool load_model(const std::string& path);
    // Raw checkpoint construction contract. For a pack, architecture remains
    // authoritative and only the operational fields represented by
    // ModelLoadOptions are taken exactly from this legacy argument.
    bool load_model(const std::string& path, const ModelConfig& config);
    // Unambiguous production pack load. A raw checkpoint cannot be constructed
    // from runtime options alone and is rejected.
    bool load_model(const std::string& path, const ModelLoadOptions& options);
    // Evaluation uses the same authoritative loader and tokenizer as serving.
    void load_tokenizer(const std::string& path);
    Tensor forward_logits(const std::vector<int>& ids);
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& ids) const;
    ModelConfig model_config() const;
    size_t parameter_count() const;
    std::string generate(const std::string& prompt, int max_tokens = 512, float temperature = 0.7f);
    std::string generate(const std::string& prompt, const GenerationOptions& options);
    std::string generate_stream(const std::string& prompt,
                                const GenerationOptions& options,
                                const std::function<void(const std::string&)>& on_chunk);
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts,
                                            const GenerationOptions& options = {});
    float train_step(const std::vector<int>& input, const std::vector<int>& target = {});
    float train_step(const std::string& text);
    void request_training_cancellation() noexcept;
    void clear_training_cancellation() noexcept;
    bool training_cancellation_requested() const noexcept;
    bool training_optimizer_state_poisoned() const noexcept;
    bool inference_state_poisoned() const noexcept;
    bool self_heal(const std::string& prompt, const std::string& response);
    void self_heal();
    bool save_checkpoint(const std::string& path) const;
    bool save_model_pack(const std::string& directory) const;
    std::unique_ptr<InferenceEngine> clone_for_inference() const;
    // Deep clone including optimizer/scheduler state. Used by administrative
    // training endpoints to train transactionally and publish only success.
    std::unique_ptr<InferenceEngine> clone_for_training() const;
    size_t get_memory_usage() const;
    std::vector<int> sanitize_token_ids(const std::vector<int>& ids) const;
    const GenerationMetrics& last_generation_metrics() const { return last_metrics_; }

private:
    void try_load_tokenizer(const std::string& path);
    bool try_load_model_pack(const std::string& path,
                             const ModelLoadOptions& options);
    bool load_model_raw(const std::string& path,
                        const ModelConfig& raw_config);
    void commit_loaded_engine(InferenceEngine&& staged);
    bool try_evaluate_simple_math(const std::string& prompt,
                                  const std::string& response,
                                  bool& is_valid) const;
    GenerationMetrics last_metrics_{};
    bool loaded_from_pack_ = false;
    std::atomic<bool> inference_state_poisoned_{false};
};

} // namespace nsos
