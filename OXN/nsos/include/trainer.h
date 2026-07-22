#pragma once
#include "autograd.h"
#include "jamba.h"
#include "optimizer_4bit.h"
#include <vector>
#include <functional>
#include <unordered_map>

namespace nsos {

struct TrainPhaseScheduler {
    // K3: QAT is ON by default now that it runs on the GPU (fake-quant STE in
    // BitLinear::forward).  The model trains against ternary weights + int8
    // activations after qat_start_step, so deployment quantization is no longer
    // a post-hoc cliff.  Set false to train purely in FP32.
    bool progressive_qat_enabled = true;
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

struct TrainingObjectiveStats {
    float supervised_cross_entropy = 0.0f;
    float repetition_unlikelihood = 0.0f;
    float logit_l2 = 0.0f;
    float sparse_selector = 0.0f;
    float qat_regularization = 0.0f;
    float moe_auxiliary = 0.0f;
    float criticality_regularization = 0.0f;
    float total = 0.0f;
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
    // Deprecated compatibility name.  This term is logit L2, not a
    // variational information bottleneck.  New code should use logit_l2_beta.
    float pantheon_vib_beta = 0.0f;
    float logit_l2_beta = 0.0f;
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
    // OXTA-CRIT Lei 1 — baseline do ganho de ramo ternário g0 por Parameter,
    // capturado na 1a visita do controlador de criticalidade
    // (apply_criticality_regularization).  Antes era um std::unordered_map
    // STATIC dentro da função (process-wide): o baseline VAZAVA entre runs e
    // entre instâncias de Trainer no mesmo processo — quebrando braços A/B e
    // qualquer teste que construísse dois modelos.  Mantido per-instância junto
    // de m_state/v_state para que cada Trainer tenha sua própria linha de base.
    std::unordered_map<Parameter*, float> crit_g0_state;

    // OXTA-CRIT §6 closed loop.  Keep independent controller state so update
    // order cannot make one producer silently overwrite the other: external is
    // the Python replica-coherence/SNR controller; criticality is the in-loop
    // branch-gain controller.  AdamW consumes their product.  The fused GPU
    // optimizer accepts this effective scale directly and therefore remains on.
    std::unordered_map<Parameter*, float> external_lr_scale;
    std::unordered_map<Parameter*, float> criticality_lr_scale;
    float lr_scale_for(Parameter* p) const {
        const auto external = external_lr_scale.find(p);
        const auto criticality = criticality_lr_scale.find(p);
        return (external == external_lr_scale.end() ? 1.0f : external->second) *
               (criticality == criticality_lr_scale.end()
                    ? 1.0f
                    : criticality->second);
    }
    void set_lr_scale_by_name(const std::string& name, float scale);
    void clear_lr_scales() { external_lr_scale.clear(); }

    // Versioned, crash-safe optimizer/scheduler/RNG sidecar.  `model_path` is
    // hashed into the state file so a torn pair or an accidentally mixed model
    // and optimizer checkpoint is rejected before any Trainer state mutates.
    void save_training_state(const std::string& state_path,
                             const std::string& model_path) const;
    void load_training_state(const std::string& state_path,
                             const std::string& model_path);

    TrainPhaseScheduler phase_scheduler;
    AuxiliaryStackStats last_auxiliary_stats;
    TrainingObjectiveStats last_objective_stats;
    Trainer(JambaModel* m, float lr = 0.001f);
    ~Trainer();

    void configure_progressive_qat(const TrainPhaseScheduler& scheduler);
    bool progressive_qat_active() const;
    
    float train_step(const std::vector<int>& tokens, const std::vector<int>& targets);
    // Forward + loss + backward WITHOUT the optimizer step.  Leaves the freshly
    // computed gradients in the parameters (zeroes them first, like train_step)
    // so an external instrument can read them — used by the per-layer gradient
    // SNR / backward-Lyapunov probe (criticality instrument, OXTA-CRIT §6).
    // Returns the loss.  Does NOT mutate weights or optimizer (m/v) state.
    float accumulate_gradients(const std::vector<int>& tokens,
                               const std::vector<int>& targets);
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
