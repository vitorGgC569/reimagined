#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace nsos {

// Magic Numbers & Constants
constexpr uint32_t NSOS_MODEL_MAGIC = 0x4E534F53; // "NSOS" in ASCII
// v4 (2026-07): adds the explicit hybrid-composition contract,
// exact-attention-linear/terminal-layer policy, canonical absolute parameter
// identities, and SHA-256 configuration/payload integrity. v1-v3 remain
// readable through their version-specific compatibility paths.
constexpr uint32_t NSOS_MODEL_VERSION = 4;
// Fingerprint bits (v2+ header, uint32 after version):
constexpr uint32_t NSOS_FP_MAMBA_PROPER    = 1u << 0;
constexpr uint32_t NSOS_FP_STATE_EXPANSION = 1u << 1;
constexpr uint32_t NSOS_FP_TIE_EMBEDDINGS  = 1u << 2;
constexpr uint32_t NSOS_FP_A_LOG_DOMAIN    = 1u << 3;  // always set by v2 saves
constexpr uint32_t NSOS_FP_MAMBA2_FAITHFUL = 1u << 4;
constexpr uint32_t NSOS_FP_HYBRID_PARALLEL = 1u << 5;
constexpr uint32_t NSOS_FP_EXACT_ATTN_LINEAR = 1u << 6;
constexpr uint32_t NSOS_FP_FORCE_MAMBA_LAST = 1u << 7;
constexpr uint32_t NSOS_FP_MAMBA3 = 1u << 8;

constexpr float NSOS_DEFAULT_EPSILON = 1e-6f;
constexpr float NSOS_ROPE_THETA = 10000.0f;
constexpr int NSOS_MAX_SEQ_LEN = 131072;

// Device Type Enum
enum class Device {
    CPU,
    GPU
};

enum class HybridComposition {
    // Checkpoint-compatible historical behavior: an attention slot replaces
    // the Mamba mixer and shares one LayerScale across attention and FFN.
    LegacyReplacement = 0,
    // Production behavior: every scheduled attention slot retains its Mamba
    // branch and adds independently normalized/gated Attention and FFN paths.
    ParallelGated = 1,
};

// Unified Model Configuration
struct ModelConfig {
    int architecture_schema_version = 2;
    int num_layers = 12;
    int d_model = 768;
    int vocab_size = 32000;
    int n_heads = 12;
    int n_kv_heads = 4; // Grouped Query Attention
    int sliding_window = 4096;
    int attention_period = 8;
    int attention_slot = 7;
    HybridComposition hybrid_composition =
        HybridComposition::ParallelGated;
    bool force_mamba_last_layer = false;
    bool faithful_attention_linears = true;
    float hybrid_mamba_gate_init = 1.0f;
    float hybrid_attention_gate_init = 0.01f;
    float hybrid_ffn_gate_init = 0.01f;
    // RoPE base frequency (theta).  Larger theta -> slower rotation -> more
    // position-invariant per-head dims (helps content-based associative recall,
    // at the cost of positional resolution); theta -> inf approaches NoPE.  Was
    // hardcoded 10000 in Attention; now an explicit serialized field so the
    // recall-vs-RoPE hypothesis can be tested reproducibly.
    float rope_theta = 10000.0f;
    
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

    // Exact logit L2 regularizer: beta * 0.5 * mean(logits^2).  This is not a
    // variational information bottleneck: it has no latent posterior, prior,
    // reparameterization, or KL term.
    float logit_l2_beta = 0.0f;
    // Deprecated manifest/API alias retained only for old packs and scripts.
    // If both names are non-zero they must agree exactly.
    float pantheon_vib_beta = 0.0f;

    // ── Slender embedding head-to-toe quantization (2026-05-25 wiring) ──
    // When true, JambaModel calls embedding->set_slender_quantization(true)
    // after construction.  Forward path then uses slender_forward_cpu_,
    // which ternary-quantizes the embedding lookup with cached weights.
    // Validated standalone via test_slender_embedding (4 tests PASS).
    bool use_slender_embedding = false;

    // ── KAN FFN (Kolmogorov-Arnold) ──
    // When true, every non-MoE block replaces its dense gate-up -> squared-ReLU
    // -> down FFN with a single BitFastKANLayer (learnable RBF activations).
    // Default false preserves the dense-FFN behavior exactly.
    bool use_kan = false;

    // ── Corrected selective SSM (Mamba-2 SSD) — DEFAULT ON ──────────────────
    // The legacy diagonal gated-EMA path is degenerate (delta≡C, C applied
    // twice, saturating tanh readout, B=σ(x) self-gate, no input conv).  These
    // enable the validated corrected path: independent x/z/B/C/dt projections +
    // causal depthwise conv1d + single-C LINEAR readout + SiLU gate, with the
    // full N-dimensional SSD state h∈R^{H×P×N} (Gu & Dao 2024).  Hand-derived
    // gradients, gradchecked (test_gradcheck: check_mamba2_proper / _nstate) and
    // GPU-parity validated (test_gpu_parity_mamba_proper / _nstate). Set false
    // only to reload a pre-correction checkpoint trained on the legacy path.
    bool mamba_proper_ssm = true;
    bool mamba_state_expansion = true;
    // Number of SSM states per Mamba head. This is an architectural field,
    // not a boolean feature toggle. CUDA kernels currently support up to 64.
    int  mamba_d_state = 64;
    int  mamba_conv_kernel = 4;
    // Exact state-spaces/mamba Mamba2 graph.  Pure Mamba layers omit the
    // historical dense FFN; set false to load/use the pre-faithful layout.
    bool mamba2_faithful = true;
    int  mamba_expand = 2;
    int  mamba_head_dim = 64;
    int  mamba_n_groups = 1;

    // ── Weight tying (N6) — DEFAULT ON ──────────────────────────────────────
    // Tie the LM head (value_head) to the token embedding matrix (both are
    // [vocab, d_model]).  Sharing one latent weight cuts parameters and
    // typically improves small-model generalization (Press & Wolf 2017).
    bool tie_word_embeddings = true;

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
    // New opt-in architecture; schema v3 records every Mamba3 field.
    // Dense FP32 projections, no automatic Mamba2/checkpoint reinterpretation.
    bool mamba3_enabled = false;
    int mamba3_schema_version = 1;
    int mamba3_state_dim = 128;
    bool mamba3_mimo = false;
    int mamba3_mimo_rank = 4;
    bool mamba3_outproj_norm = false;
    float mamba3_rope_fraction = 0.5f;
    float mamba3_norm_eps = 1e-5f;
    float mamba3_a_floor = 1e-4f;
};

inline void validate_model_config(const ModelConfig& config) {
    auto require_range = [](int value, int minimum, int maximum,
                            const std::string& field) {
        if (value < minimum || value > maximum) {
            throw std::invalid_argument(
                "ModelConfig." + field + " must be in [" +
                std::to_string(minimum) + ", " +
                std::to_string(maximum) + "]");
        }
    };
    auto require_slot = [&](int period, int slot, const std::string& prefix) {
        require_range(period, 1, 1'000'000, prefix + "_period");
        if (slot < 0 || slot >= period) {
            throw std::invalid_argument(
                "ModelConfig." + prefix + "_slot must be in [0, period)");
        }
    };
    require_range(config.num_layers, 1, 512, "num_layers");
    require_range(config.d_model, 8, 65536, "d_model");
    require_range(config.vocab_size, 2, 2'000'000, "vocab_size");
    require_range(config.n_heads, 1, config.d_model, "n_heads");
    if (config.d_model % config.n_heads != 0) {
        throw std::invalid_argument(
            "ModelConfig.d_model must be divisible by n_heads");
    }
    if (((config.d_model / config.n_heads) & 1) != 0) {
        throw std::invalid_argument(
            "ModelConfig attention head dimension must be even for RoPE");
    }
    require_range(config.n_kv_heads, 1, config.n_heads, "n_kv_heads");
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::invalid_argument(
            "ModelConfig.n_heads must be divisible by n_kv_heads for GQA");
    }
    require_range(config.sliding_window, 1, NSOS_MAX_SEQ_LEN,
                  "sliding_window");
    require_range(config.max_context_tokens, 1, NSOS_MAX_SEQ_LEN,
                  "max_context_tokens");
    require_range(config.default_batch_size, 1, 4096,
                  "default_batch_size");
    require_range(config.architecture_schema_version, 1, 3,
                  "architecture_schema_version");
    require_slot(config.attention_period, config.attention_slot, "attention");
    require_slot(config.moe_period, config.moe_slot, "moe");
    require_slot(config.ttt_period, config.ttt_slot, "ttt");
    require_range(config.num_experts, 1, 4096, "num_experts");
    require_range(config.num_experts_per_token, 1, config.num_experts,
                  "num_experts_per_token");
    require_range(config.moe_expert_hidden_dim, 0, 1'048'576,
                  "moe_expert_hidden_dim");
    require_range(config.mcts_simulations, 1, 1'000'000,
                  "mcts_simulations");
    require_range(config.mcts_depth, 1, 4096, "mcts_depth");
    require_range(config.mamba_d_state, 1, 64, "mamba_d_state");
    require_range(config.mamba_conv_kernel, 1, 16, "mamba_conv_kernel");
    require_range(config.mamba_expand, 1, 8, "mamba_expand");
    if (config.mamba2_faithful || config.mamba3_enabled) {
        const long long inner =
            static_cast<long long>(config.mamba_expand) * config.d_model;
        require_range(config.mamba_head_dim, 1,
                      static_cast<int>((std::min)(inner, 4096LL)),
                      "mamba_head_dim");
        if (inner % config.mamba_head_dim != 0) {
            throw std::invalid_argument(
                "ModelConfig mamba_expand*d_model must be divisible by "
                "mamba_head_dim");
        }
        const int mamba_heads = static_cast<int>(inner / config.mamba_head_dim);
        require_range(config.mamba_n_groups, 1, mamba_heads,
                      "mamba_n_groups");
        if (mamba_heads % config.mamba_n_groups != 0) {
            throw std::invalid_argument(
                "ModelConfig Mamba head count must be divisible by "
                "mamba_n_groups");
        }
    }
    if (config.mamba3_enabled) {
        if (config.architecture_schema_version != 3 || config.mamba3_schema_version != 1)
            throw std::invalid_argument("Mamba3 requires architecture schema v3 / block schema v1");
        require_range(config.mamba3_state_dim, 4, 128, "mamba3_state_dim");
        require_range(config.mamba_head_dim, 1, 128, "mamba_head_dim");
        require_range(config.mamba3_mimo_rank, 1, 8, "mamba3_mimo_rank");
        if (config.mamba3_state_dim % 2 ||
            (config.mamba3_rope_fraction != 0.5f && config.mamba3_rope_fraction != 1.0f) ||
            !std::isfinite(config.mamba3_norm_eps) || config.mamba3_norm_eps < 1e-12f || config.mamba3_norm_eps > 1 ||
            !std::isfinite(config.mamba3_a_floor) || config.mamba3_a_floor <= 0 || config.mamba3_a_floor > 64)
            throw std::invalid_argument("Invalid Mamba3 state/rotary/normalization/floor configuration");
    }
    if (!std::isfinite(config.dropout) || config.dropout < 0.0f ||
        config.dropout >= 1.0f) {
        throw std::invalid_argument(
            "ModelConfig.dropout must be finite and in [0, 1)");
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        throw std::invalid_argument(
            "ModelConfig.rope_theta must be finite and positive");
    }
    const int hybrid_mode =
        static_cast<int>(config.hybrid_composition);
    if (hybrid_mode <
            static_cast<int>(HybridComposition::LegacyReplacement) ||
        hybrid_mode >
            static_cast<int>(HybridComposition::ParallelGated)) {
        throw std::invalid_argument(
            "ModelConfig.hybrid_composition is invalid");
    }
    if (config.architecture_schema_version == 1 &&
        (config.hybrid_composition !=
             HybridComposition::LegacyReplacement ||
         !config.force_mamba_last_layer ||
         config.faithful_attention_linears)) {
        throw std::invalid_argument(
            "ModelConfig architecture schema v1 requires the historical "
            "LegacyReplacement hybrid, force_mamba_last_layer=true and "
            "faithful_attention_linears=false");
    }
    for (const auto& [value, field] :
         std::initializer_list<std::pair<float, const char*>>{
             {config.hybrid_mamba_gate_init,
              "hybrid_mamba_gate_init"},
             {config.hybrid_attention_gate_init,
              "hybrid_attention_gate_init"},
             {config.hybrid_ffn_gate_init,
              "hybrid_ffn_gate_init"}}) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument(
                std::string("ModelConfig.") + field +
                " must be finite and in [0, 1]");
        }
    }
    if (!std::isfinite(config.chrass_density) ||
        config.chrass_density < 0.0f || config.chrass_density > 1.0f) {
        throw std::invalid_argument(
            "ModelConfig.chrass_density must be finite and in [0, 1]");
    }
    if (!std::isfinite(config.logit_l2_beta) || config.logit_l2_beta < 0.0f ||
        !std::isfinite(config.pantheon_vib_beta) ||
        config.pantheon_vib_beta < 0.0f) {
        throw std::invalid_argument(
            "ModelConfig logit L2 coefficients must be finite and non-negative");
    }
    if (config.logit_l2_beta != 0.0f && config.pantheon_vib_beta != 0.0f &&
        config.logit_l2_beta != config.pantheon_vib_beta) {
        throw std::invalid_argument(
            "ModelConfig.logit_l2_beta conflicts with deprecated "
            "pantheon_vib_beta");
    }
    if (config.use_flash_attn) {
        throw std::invalid_argument(
            "ModelConfig.use_flash_attn is unsupported; select the validated "
            "exact/sparse attention backend explicitly");
    }
}

} // namespace nsos
