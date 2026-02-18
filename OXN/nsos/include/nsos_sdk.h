#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
#include <string>
#include <vector>
#include <memory>

namespace nsos {

// Use ModelConfig from nsos_config.h instead of redefining

class InferenceEngine {
public:
    std::unique_ptr<JambaModel> model;
    
    InferenceEngine() = default;
    void load_model(const std::string& path);
    std::string generate(const std::string& prompt, int max_tokens = 50, float temperature = 0.7f);
    float train_step(const std::vector<int>& input, const std::vector<int>& target);
    void self_heal();
    size_t get_memory_usage();
};

} // namespace nsos
