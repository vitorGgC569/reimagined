#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace nsos {

// Magic Numbers & Constants
constexpr uint32_t NSOS_MODEL_MAGIC = 0x4E534F53; // "NSOS" in ASCII
constexpr uint32_t NSOS_MODEL_VERSION = 1;

constexpr float NSOS_DEFAULT_EPSILON = 1e-6f;
constexpr float NSOS_ROPE_THETA = 10000.0f;
constexpr int NSOS_MAX_SEQ_LEN = 131072;

// Device Type Enum
enum class Device {
    CPU,
    GPU
};

// Unified Model Configuration
struct ModelConfig {
    int num_layers = 12;
    int d_model = 768;
    int vocab_size = 32000;
    int n_heads = 12;
    int n_kv_heads = 4; // Grouped Query Attention
    int sliding_window = 4096;
    
    // MoE Settings
    int num_experts = 8;
    int num_experts_per_token = 2;
    bool use_moe = false;

    // Training Settings
    bool use_gradient_checkpointing = false;
    float dropout = 0.0f;
    
    // System 2 Settings
    int mcts_simulations = 50;
    int mcts_depth = 5;
    
    // Path settings
    std::string checkpoint_path = "checkpoints/";
    
    // Hardware settings
    bool use_cuda = false;
    bool use_flash_attn = false;
};

} // namespace nsos
