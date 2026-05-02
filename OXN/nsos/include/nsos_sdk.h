#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
#include "tokenizer.h"
#include "trainer.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nsos {

// Use ModelConfig from nsos_config.h instead of redefining

struct GenerationOptions {
    int max_tokens = 50;
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
    std::string mamba_last_fallback_reason;
};

class InferenceEngine {
public:
    std::unique_ptr<JambaModel> model;
    std::unique_ptr<Trainer> trainer;
    Tokenizer tokenizer;
    ModelConfig config;
    
    InferenceEngine() = default;
    bool load_model(const std::string& path, const ModelConfig& config = {});
    std::string generate(const std::string& prompt, int max_tokens = 50, float temperature = 0.7f);
    std::string generate(const std::string& prompt, const GenerationOptions& options);
    std::string generate_stream(const std::string& prompt,
                                const GenerationOptions& options,
                                const std::function<void(const std::string&)>& on_chunk);
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts,
                                            const GenerationOptions& options = {});
    float train_step(const std::vector<int>& input, const std::vector<int>& target = {});
    float train_step(const std::string& text);
    bool self_heal(const std::string& prompt, const std::string& response);
    void self_heal();
    bool save_checkpoint(const std::string& path) const;
    bool save_model_pack(const std::string& directory) const;
    std::unique_ptr<InferenceEngine> clone_for_inference() const;
    size_t get_memory_usage() const;
    std::vector<int> sanitize_token_ids(const std::vector<int>& ids) const;
    const GenerationMetrics& last_generation_metrics() const { return last_metrics_; }

private:
    void try_load_tokenizer(const std::string& path);
    bool try_load_model_pack(const std::string& path, const ModelConfig& runtime_overrides);
    bool try_evaluate_simple_math(const std::string& prompt,
                                  const std::string& response,
                                  bool& is_valid) const;
    GenerationMetrics last_metrics_{};
    bool loaded_from_pack_ = false;
};

} // namespace nsos
