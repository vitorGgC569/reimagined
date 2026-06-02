#pragma once
#include "autograd.h"
#include "jamba.h"
#include "optimizer_4bit.h"
#include <vector>
#include <functional>
#include <unordered_map>

namespace nsos {

struct TrainPhaseScheduler {
    bool progressive_qat_enabled = false;
    int semantic_warmup_steps = 100;
    int qat_start_step = 300;
    int quantized_precision_bits = 2;
    float ternary_regularization = 1e-3f;
    bool auxiliary_stack_enabled = false;
    bool auxiliary_session_adapt_enabled = false;
    bool auxiliary_reasoning_enabled = false;
    bool auxiliary_memory_enabled = false;
    int auxiliary_reasoning_iterations = 1;
    int auxiliary_reasoning_simulations = 24;
    float auxiliary_memory_blend = 0.35f;
    int auxiliary_every_steps = 1;
    int auxiliary_prompt_max_tokens = 96;
    int auxiliary_answer_max_tokens = 24;
    int auxiliary_memory_scope = 0;
};

struct AuxiliaryStackStats {
    int bucket_count = 0;
    int due_count = 0;
    int applied_count = 0;
    int reasoning_count = 0;
    int memory_count = 0;
    int session_adapt_count = 0;
    int sample_count = 0;
    int prompt_tokens = 0;
    int answer_tokens = 0;
    float prompt_state_norm = 0.0f;
    float target_state_norm = 0.0f;
    float reason_delta_norm = 0.0f;
    float reason_cosine = 0.0f;
    float memory_delta_norm = 0.0f;
    float memory_cosine = 0.0f;
    float final_target_delta_norm = 0.0f;
};

class Trainer {
public:
    JambaModel* model;
    float learning_rate;
    
    // Hyperparâmetros do AdamW e Regularização
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    float max_grad_norm = 1.0f;
    float min_learning_rate_scale = 0.1f;
    float first_token_loss_scale = 2.5f;
    float eos_loss_scale = 0.35f;
    float repetition_unlikelihood_scale = 0.0f;
    float moe_aux_loss_scale = 0.01f;
    // Pantheon VIB-style logits L2 regularizer (2026-05-25 wiring).
    // When > 0, adds beta * 0.5 * mean(logits^2) to loss + grad contribution.
    // 0.0 = OFF (default).  Validated standalone Pantheon battery 14/14 PASS.
    float pantheon_vib_beta = 0.0f;
    int warmup_steps = 20;
    int global_step_count = 0;
    int total_training_steps = 1000; // Valor base para o scheduler de LR
    int eos_token_id = 0;

    // Optimizer-state precision.  32 = FP32 m/v (default, current path).  4 =
    // 4-bit packed m/v (Li et al. 2023) — ~8x less optimizer memory.  The 4-bit
    // path engages only for parameters resident on CPU; GPU parameters keep the
    // FP32 path until the CUDA 4-bit kernel lands (Phase 2).
    int optimizer_state_bits = 32;

    // Buffers de Memória AdamW (M = First Moment, V = Second Moment)
    std::unordered_map<Parameter*, Tensor> m_state;
    std::unordered_map<Parameter*, Tensor> v_state;
    // 4-bit packed Adam state (used when optimizer_state_bits == 4).
    std::unordered_map<Parameter*, Quant4OptState> quant_state;
    TrainPhaseScheduler phase_scheduler;
    AuxiliaryStackStats last_auxiliary_stats;
    Trainer(JambaModel* m, float lr = 0.001f);
    ~Trainer();

    void configure_progressive_qat(const TrainPhaseScheduler& scheduler);
    bool progressive_qat_active() const;
    
    float train_step(const std::vector<int>& tokens, const std::vector<int>& targets);
    float train_supervised(const std::vector<int>& prompt_tokens,
                           const std::vector<int>& answer_tokens);
    float train_supervised_batch(const std::vector<std::vector<int>>& prompt_batch,
                                 const std::vector<std::vector<int>>& answer_batch);
    void train_loop(const std::vector<int>& tokens,
                    int epochs,
                    int batch_size,
                    int seq_len,
                    std::function<void(int, float)> callback = nullptr,
                    int max_steps = -1);
};

} // namespace nsos
