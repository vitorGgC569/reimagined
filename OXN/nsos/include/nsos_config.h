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

    // ── CHRASS topological injection (2026-05-25 wiring) ──
    // When enabled, each JambaBlock instantiates a ChrassLayer that runs
    // IN PARALLEL with the FFN/MoE path on the post-norm activations and
    // its output is added to the block residual.  The block becomes:
    //     out = x + drop(mixer(norm(x))) + drop(ffn(norm(x))) + drop(chrass(norm(x)))
    // Adjacency for each block is a random sparse matrix with density
    // `chrass_density` (0..1), seeded by `chrass_seed + layer_idx` for
    // determinism across runs.  Validated standalone at 26/26 tests
    // (see docs/CHRASS_VALIDATION_REPORT.md).  Set use_chrass=false (default)
    // to preserve byte-exact existing behavior.
    bool use_chrass = false;
    float chrass_density = 0.10f;  // 10% nonzero edges in adjacency
    uint32_t chrass_seed = 0x0CDA55u;

    // ── Pantheon VIB-style compression regularizer (2026-05-25 wiring) ──
    // When > 0, Trainer adds beta * 0.5 * mean(logits^2) to the cross-entropy
    // loss, with the corresponding gradient (beta * logits / N) added to the
    // backward grad before model->backward_external.  This is a degenerate
    // case of Variational Information Bottleneck applied directly to logits
    // (no variational layer needed) — pulls logits toward zero, encouraging
    // confident but compressed representations.  Full VIB with per-feature
    // mean+log_var (using pantheon::physics::InformationBottleneck) is a
    // Phase 2 enhancement requiring additional gradient routing.
    // Default 0.0 preserves byte-exact existing behavior.
    // See docs/PANTHEON_VALIDATION_REPORT.md.
    float pantheon_vib_beta = 0.0f;

    // ── Slender embedding head-to-toe quantization (2026-05-25 wiring) ──
    // When true, JambaModel calls embedding->set_slender_quantization(true)
    // after construction.  Forward path then uses slender_forward_cpu_,
    // which ternary-quantizes the embedding lookup with cached weights.
    // Validated standalone via test_slender_embedding (4 tests PASS).
    bool use_slender_embedding = false;

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
