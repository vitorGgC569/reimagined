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
    int attention_period = 8;
    int attention_slot = 7;
    
    // MoE Settings
    int num_experts = 8;
    int num_experts_per_token = 2;
    bool use_moe = false;
    int moe_period = 6;
    int moe_slot = 5;
    // ── Nemotron K·m invariant tuning (Cherry-pick #4) ──
    // When > 0, overrides the default expert FFN intermediate dimension
    // (otherwise computed as d_model * 4).  Use this to apply the K·m
    // invariant from Nemotron 3 Super (Sec 2.1.1, Principle 3):
    // increase num_experts_per_token AND decrease moe_expert_hidden_dim
    // proportionally, holding K × m fixed to preserve quality while
    // reducing memory bandwidth in MoE inference.
    // See docs/NEMOTRON_KM_INTEGRATION.md for the principle + tested values.
    // Default 0 preserves the existing dm * 4 behavior exactly.
    int moe_expert_hidden_dim = 0;

    // TTT schedule
    // Defaults stay conservative for product paths; TTT is research-only unless a
    // profile explicitly opts in and validates snapshot/replay behavior.
    bool use_ttt = false;
    int ttt_period = 8;
    int ttt_slot = 3;

    // Training Settings
    bool use_gradient_checkpointing = false;
    float dropout = 0.0f;
    
    // System 2 Settings
    int mcts_simulations = 50;
    int mcts_depth = 5;
    
    // Path settings
    std::string checkpoint_path = "checkpoints/";
    int max_context_tokens = 4096;
    int default_batch_size = 1;
    
    // Hardware settings
    bool use_cuda = false;
    bool use_exact_attention_training = true;
    // Legacy compatibility flag only. Runtime selection should be driven by
    // explicit backend policy, not by this name.
    bool use_flash_attn = false;
};

} // namespace nsos
