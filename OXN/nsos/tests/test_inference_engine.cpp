#include "nsos_sdk.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace nsos;

int main() {
  try {
    ModelConfig config;
    config.num_layers = 2;
    config.d_model = 32;
    config.vocab_size = 128;
    config.n_heads = 4;
    config.n_kv_heads = 2;

    InferenceEngine engine;
    std::cout << "[InferenceEngineTest] loading model..." << std::endl;
    const bool loaded = engine.load_model("", config);
    assert(loaded);

    std::cout << "[InferenceEngineTest] training..." << std::endl;
    const float loss = engine.train_step(std::string("123456"));
    assert(std::isfinite(loss));

    std::cout << "[InferenceEngineTest] generating..." << std::endl;
    GenerationOptions batch_options;
    batch_options.max_tokens = 2;
    // The SDK requires a valid vocabulary ID; this byte fixture uses NUL.
    batch_options.eos_token_id = 0;
    const std::string generated = engine.generate("123", batch_options);
    (void)generated;
    const auto batch_outputs = engine.generate_batch({"123", "456"}, batch_options);
    assert(batch_outputs.size() == 2);
    const auto& batch_metrics = engine.last_generation_metrics();
    assert(batch_metrics.batch_size == 2);
    assert(batch_metrics.prompt_tokens_used > 0);
    assert(batch_metrics.used_streaming);
    assert(batch_metrics.sampler_ms >= 0.0);

    auto require = [](bool ok, const char* reason) {
      if (!ok) throw std::runtime_error(reason);
    };
    const std::vector<int> prefix{1, 5, 7, 3};
    Tensor full = engine.forward_logits(prefix);
    engine.model->reset_session();
    Tensor last = engine.model->forward_ids_last(prefix);
    require(last.shape.dims == std::vector<int>({1, config.vocab_size}),
            "last-only projection must produce exactly one vocabulary row");
    for (int i = 0; i < config.vocab_size; ++i)
      require(std::abs(last.data()[i] - full.data()[full.size - config.vocab_size + i]) < 1e-4f,
              "last-only logits differ from full projection");
    require(engine.parameter_count() > 0, "SDK parameter inventory is empty");

    GenerationOptions greedy;
    greedy.max_tokens = 5;
    greedy.top_k = 1;
    greedy.temperature = 0;
    greedy.eos_token_id = 0;
    const std::vector<std::string> mixed_prompts{"123", "7", "456", "89", "7"};
    std::vector<std::string> expected;
    for (const auto& prompt : mixed_prompts) expected.push_back(engine.generate(prompt, greedy));
    require(engine.generate_batch(mixed_prompts, greedy) == expected,
            "mixed-length batch must preserve single-sequence greedy results and order");
    require(engine.generate_batch(mixed_prompts, greedy) == expected,
            "batch reuse leaked session state");
    require(engine.generate_batch({}, greedy).empty(), "empty batch must stay empty");
    greedy.max_tokens = 0;
    require(engine.generate("123", greedy).empty(), "zero-token generation is not empty");
    const auto empty_outputs = engine.generate_batch(mixed_prompts, greedy);
    for (const auto& output : empty_outputs) require(output.empty(), "zero-token batch generated output");
    require(engine.last_generation_metrics().generated_tokens == 0,
            "zero-token batch reported generated tokens");

    std::cout << "[InferenceEngineTest] self-heal..." << std::endl;
    assert(engine.self_heal("1+1=", "3") == true);
    assert(engine.self_heal("1+1=", "2") == false);
    assert(engine.get_memory_usage() > 0);

    std::cout << "InferenceEngine smoke test passed!" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "InferenceEngine smoke test failed: " << e.what() << std::endl;
    return 1;
  }
}
