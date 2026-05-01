#include "nsos_sdk.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace nsos;

int main() {
  try {
    ModelConfig config;
    config.num_layers = 2;
    config.d_model = 32;
    config.vocab_size = 128;

    InferenceEngine engine;
    std::cout << "[InferenceEngineTest] loading model..." << std::endl;
    const bool loaded = engine.load_model("", config);
    assert(loaded);

    std::cout << "[InferenceEngineTest] training..." << std::endl;
    const float loss = engine.train_step(std::string("123456"));
    assert(std::isfinite(loss));

    std::cout << "[InferenceEngineTest] generating..." << std::endl;
    const std::string generated = engine.generate("123", 2);
    (void)generated;
    GenerationOptions batch_options;
    batch_options.max_tokens = 2;
    const auto batch_outputs = engine.generate_batch({"123", "456"}, batch_options);
    assert(batch_outputs.size() == 2);
    const auto& batch_metrics = engine.last_generation_metrics();
    assert(batch_metrics.batch_size == 2);
    assert(batch_metrics.prompt_tokens_used > 0);
    assert(batch_metrics.used_streaming);
    assert(batch_metrics.sampler_ms >= 0.0);

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
