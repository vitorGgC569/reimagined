#pragma once
#include "autograd.h"
#include "jamba.h"
#include "optimizer_4bit.h"
#include "gpu_sparse_adam.h"
#include "gpu_moe_training.h"
#include "runtime_execution_identity.h"
#include <atomic>
#include <cstdint>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nsos {

#ifdef NSOS_ENABLE_TEST_HOOKS
namespace testing {
// Allows `countdown` successful optimizer-record staging points, then throws
// std::bad_alloc at the next one. A negative value disables injection.
void set_training_state_stage_failure_countdown(
    long long countdown);
// Simulates a process failure after the temporary sidecar is durable but
// before it atomically replaces an existing destination.
void set_training_state_save_failure_before_replace(bool enabled);
// Poisons one optimizer input immediately before the finite-value gate. The
// injection is one-shot and exists only in test-enabled builds.
void inject_training_nan_before_optimizer();
// Throws after `countdown + 1` CPU parameters have been updated, exercising
// the fail-stop path for an optimizer cohort that may be partially committed.
void set_optimizer_commit_failure_countdown(
    long long countdown);
void clear_training_state_stage_failure();
}  // namespace testing
#endif

class OptimizerStatePoisonedException final
    : public std::runtime_error {
public:
    explicit OptimizerStatePoisonedException(
        const std::string& message)
        : std::runtime_error(message) {}
};

struct TrainPhaseScheduler {
    // K3: QAT is ON by default now that it runs on the GPU (fake-quant STE in
    // BitLinear::forward).  The model trains against ternary weights + int8
    // activations after qat_start_step, so deployment quantization is no longer
    // a post-hoc cliff.  Set false to train purely in FP32.
    bool progressive_qat_enabled = true;
    int semantic_warmup_steps = 100;
    int qat_start_step = 300;
    // BitNet b1.58 weight precision is fixed at two packed ternary bits.
    int quantized_precision_bits = 2;
    // Activations are independently quantized (int8 by default).
    int activation_precision_bits = 8;
    float ternary_regularization = 1e-3f;
    // Active stacks require a real TTT session-adaptation consumer. Reasoning
    // additionally requires JambaModel::set_reasoning_policy. Runtime callbacks
    // must be registered explicitly after load; they are not serialized.
    bool auxiliary_stack_enabled = false;
    bool auxiliary_session_adapt_enabled = false;
    bool auxiliary_reasoning_enabled = false;
    bool auxiliary_memory_enabled = false;
    int auxiliary_reasoning_iterations = 1;
    int auxiliary_reasoning_simulations = 24;
    float auxiliary_memory_blend = 0.35f;
    int auxiliary_every_steps = 1;
    int auxiliary_prompt_max_tokens = 96;
    int auxiliary_answer_max_tokens = 24; // legacy serialized field; no answer forward
    int auxiliary_memory_scope = 0;
    // Optional durable OxtaMem backend for the auxiliary memory path. The
    // store path is treated as a stable prefix; Trainer derives one arena per
    // (scope, hidden dimension) to preserve backend file-lock semantics.
    bool auxiliary_oxtamem_enabled = false;
    std::string auxiliary_oxtamem_library_path;
    std::string auxiliary_oxtamem_store_path;
    uint64_t auxiliary_oxtamem_size_mb = 128;
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
    int answer_tokens = 0; // zero: answers are consumed by supervised CE only
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

// Per-step timings are populated only when NSOS_TRAIN_TIMING=1. GPU phase
// boundaries are stream-fenced by the implementation, so these values are
// diagnostic measurements rather than enqueue-time estimates.
struct TrainingStepTelemetry {
    bool enabled = false;
    int global_step = 0;
    int bucket_count = 0;
    double wall_ms = 0.0;
    double preparation_ms = 0.0;
    double inter_bucket_ms = 0.0;
    double forward_ms = 0.0;
    double loss_ms = 0.0;
    double backward_ms = 0.0;
    double optimizer_ms = 0.0;
    double unaccounted_ms = 0.0;
};

class Trainer;

// Immutable host-owned checkpoint cohort. Capture synchronizes/copies each
// parameter and optimizer state to CPU under the Trainer+model transaction;
// write() performs serialization, hashing, flush and atomic file replacement
// later, safely on a dedicated worker. No duplicate GPU model is created.
class TrainingCheckpointSnapshot {
public:
    ~TrainingCheckpointSnapshot();
    void write(const std::string& model_path,
               const std::string& state_path);
    int global_step() const noexcept { return global_step_; }

private:
    friend class Trainer;
    TrainingCheckpointSnapshot(
        std::shared_ptr<JambaModel> model,
        std::unique_ptr<Trainer> trainer,
        int global_step);

    std::shared_ptr<JambaModel> model_;
    std::unique_ptr<Trainer> trainer_;
    int global_step_ = 0;
    std::mutex write_mutex_;
    bool written_ = false;
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

    // ---- Unidade do scheduler de learning rate --------------------------
    //
    // `Steps` é o contrato legado: warmup_steps e total_training_steps contam
    // optimizer steps.  Isso deixa de ser comparável assim que a acumulação
    // de gradiente entra: para um mesmo volume de dados, o número de
    // optimizer steps é dividido por A, de modo que um braço com A alto
    // gastaria o experimento inteiro dentro do warmup e pareceria péssimo por
    // artefato de configuração, não por mérito.
    //
    // `Tokens` faz o scheduler andar por tokens efetivamente commitados, de
    // modo que todo A recebe a mesma quantidade de dados antes de alcançar o
    // LR principal.
    //
    // Sidecar v11 persists the declared unit and exact token schedule.
    // Older sidecars require explicit step-only progress migration.
    enum class SchedulerUnit { Steps, Tokens };
    SchedulerUnit scheduler_unit = SchedulerUnit::Steps;
    long long warmup_tokens = 0;
    long long training_tokens = 0;
    // WSD: comprimento da fase de decaimento em tokens.  0 mantém o cosseno
    // clássico do warmup até o fim de training_tokens.
    long long decay_tokens = 0;

    // Telemetria versus estado de treino.  `tokens_processed` conta todo token
    // que passou pelo forward; `tokens_committed` conta apenas os tokens de
    // grupos de acumulação que chegaram a commitar um optimizer step.  Uma
    // falha no microbatch 37 de 64 avança o primeiro e não o segundo — e são
    // os tokens commitados que definem a trajetória, o scheduler e o
    // checkpoint.
    long long tokens_processed = 0;
    long long tokens_committed = 0;
    // False after explicit legacy step-only migration: these counts cover
    // work since migration, not the unavailable historical token totals.
    // An incomplete run cannot resume or switch to the token scheduler.
    bool token_counters_complete = true;

    // Regime efetivo do último commit.  Se o clipping estiver ativo na maior
    // parte dos updates, quem governa o passo é o clipper e não o AdamW —
    // um A/B de batch size que ignore isso mede a coisa errada.
    float last_grad_norm_pre_clip = 0.0f;
    float last_grad_norm_post_clip = 0.0f;
    bool last_update_was_clipped = false;
    // Telemetria do último commit. NÃO alimenta a identidade: é resultado, não
    // contrato.
    int last_accumulation_steps = 1;

    // Contrato declarado de acumulação, fixado antes do treino começar e lido
    // pela identidade de runtime.  Precisa ser declaração e não observação:
    // usar o valor do último commit faria a identidade mudar no meio da
    // corrida, e o gate fail-closed — corretamente — rejeitaria a própria
    // execução.  commit_optimizer_step() exige que o A pedido coincida com
    // este valor.
    int gradient_accumulation_steps = 1;

    // Estado transitório do grupo de acumulação em andamento.  Zerado por
    // commit_optimizer_step() e por abort_gradient_accumulation().
    long long pending_accumulated_tokens = 0;
    int pending_accumulation_microbatches = 0;
    // Soma das losses supervisionadas do grupo, mantida no device.  Espelha o
    // caminho diferido de train_step(): com A=1 o tensor é exatamente o que
    // train_step consumiria, o que preserva a equivalência bit a bit entre os
    // dois caminhos.  Ler cada microbatch para o host em vez disso mudaria o
    // ponto de sincronização e, com ele, o resultado.
    Tensor pending_supervised_loss_sum;

    // Optimizer-state precision.  32 = FP32 m/v (default, current path).  4 =
    // 4-bit packed m/v (Li et al. 2023) — ~8x less optimizer memory.  The 4-bit
    // path engages only for parameters resident on CPU; GPU parameters keep the
    // FP32 path until the CUDA 4-bit kernel lands (Phase 2).
    int optimizer_state_bits = 32;

    // Dynamic loss scaling is engaged only for FP16 GEMMs. Master weights,
    // optimizer state, and stored gradients remain FP32.
    bool dynamic_loss_scaling_enabled = true;
    float loss_scale = 1024.0f;
    float min_loss_scale = 1.0f;
    float max_loss_scale = 65536.0f;
    float loss_scale_growth_factor = 2.0f;
    float loss_scale_backoff_factor = 0.5f;
    int loss_scale_growth_interval = 2000;
    int loss_scale_growth_tracker = 0;
    bool last_optimizer_step_skipped = false;
    // Explicit device sparse lane. Group scopes retain domains until every
    // producer and optimizer consumer has completed on the owning lane.
    std::shared_ptr<GpuSparseAdam> device_sparse_adam;
    std::vector<std::shared_ptr<GpuMoeTraining>> device_moe_groups;
    bool device_sparse_group_open = false;
    bool device_sparse_objectives_finalized = false;
    gpu::ExecutionContext& device_sparse_execution_context() const;
    void begin_device_sparse_group();
    void finish_device_sparse_group(bool abort, bool materialize_for_audit = false);
    void synchronize_device_sparse_checkpoint() const;

    // Auxiliary memory belongs to a Trainer run. It must never leak across
    // models, A/B arms, tenants, or tests through process-static storage.
    MemorySystem& auxiliary_memory_store(int dim, int scope);
    void clear_auxiliary_memory();
    // Buffers de Memória AdamW (M = First Moment, V = Second Moment)
    mutable std::unordered_map<Parameter*, Tensor> m_state;
    mutable std::unordered_map<Parameter*, Tensor> v_state;
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
    mutable std::mutex auxiliary_memory_mutex_;
    std::unordered_map<long long, std::unique_ptr<MemorySystem>>
        auxiliary_memory_stores_;
    // A restored store keeps its in-memory clusters immediately, while its
    // optional durable backend is attached lazily on first use. This avoids
    // opening the same OxtaMem arena twice while replacing a checkpoint.
    std::unordered_map<long long, bool> auxiliary_memory_backend_ready_;
    std::string auxiliary_memory_signature_;
    float lr_scale_for(Parameter* p) const {
        const auto external = external_lr_scale.find(p);
        const auto criticality = criticality_lr_scale.find(p);
        return (external == external_lr_scale.end() ? 1.0f : external->second) *
               (criticality == criticality_lr_scale.end()
                    ? 1.0f
                    : criticality->second);
    }
    void set_lr_scale_by_name(const std::string& name, float scale);
    void clear_lr_scales();

    // Versioned, crash-safe optimizer/scheduler/RNG sidecar.  `model_path` is
    // hashed into the state file so a torn pair or an accidentally mixed model
    // and optimizer checkpoint is rejected before any Trainer state mutates.
    void save_training_state(const std::string& state_path,
                             const std::string& model_path) const;
    // The two legacy permissions are independent. Progress migration is
    // step-only and marks token history incomplete; Tokens is always refused.
    void load_training_state(const std::string& state_path,
                             const std::string& model_path,
                             bool allow_legacy_runtime_identity = false,
                             bool allow_legacy_progress_state = false);

    // Seals the structural execution policy on first
    // observation/training/checkpoint operation. Backend, device, precision,
    // reduction order, optimizer implementation, layouts and selected kernel
    // algorithms cannot change afterwards. Scheduled scalar hyperparameters
    // (for example learning_rate and objective weights) remain mutable by
    // design: their exact values/state are versioned in the training sidecar,
    // while the PT-BR pipeline records the complete schedule in its immutable
    // run manifest.
    std::vector<RuntimeExecutionIdentity::Field>
    execution_identity_fields() const;
    std::string execution_identity_digest() const;
    void validate_execution_identity() const;
    std::shared_ptr<TrainingCheckpointSnapshot>
    capture_checkpoint_snapshot() const;

    TrainPhaseScheduler phase_scheduler;
    AuxiliaryStackStats last_auxiliary_stats;
    TrainingObjectiveStats last_objective_stats;
    TrainingStepTelemetry last_step_telemetry;
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

    // ---- Acumulação de gradiente explícita ------------------------------
    //
    // Contrato:
    //
    //     for (i = 0; i < A; ++i) accumulate_microbatch(...);
    //     commit_optimizer_step(A);
    //
    // `accumulate_microbatch` zera os gradientes apenas ao abrir um grupo; as
    // chamadas seguintes somam sobre o que já existe.  Difere de
    // `accumulate_gradients`, que zera sempre e existe para sondagem
    // independente — chamá-la N vezes sobrescreve em vez de acumular.
    //
    // `commit_optimizer_step` divide os gradientes por A ANTES do clipping
    // (contrato `mean_before_clip`) e só então aplica AdamW e zera.  Sem essa
    // normalização o gradiente seria A vezes maior e o clipping passaria a
    // governar o passo, de modo que um A/B de batch mediria o regime do
    // clipper em vez do tamanho de batch.
    //
    // Um grupo interrompido no meio deve chamar
    // `abort_gradient_accumulation()`: os gradientes dos microbatches já
    // computados contaminariam a próxima tentativa.
    float accumulate_microbatch(const std::vector<int>& tokens,
                                const std::vector<int>& targets);
    float commit_optimizer_step(int accumulation_steps);
    void abort_gradient_accumulation();
    float train_supervised(const std::vector<int>& prompt_tokens,
                           const std::vector<int>& answer_tokens);
    float train_supervised_batch(const std::vector<std::vector<int>>& prompt_batch,
                                 const std::vector<std::vector<int>>& answer_batch);
    void train_loop(const std::vector<int>& tokens,
                    int epochs,
                    int batch_size,
                    int seq_len,
                    std::function<void(int, float)> callback = nullptr,
                    int max_steps = -1,
                    int start_step = 0);
    // Cooperative, thread-safe cancellation. The request path deliberately
    // does not acquire state_mutex_: it must remain callable while a training
    // thread owns the full optimizer transaction. Cancellation is observed at
    // model layer/chunk boundaries and before optimizer publication.
    void request_cancellation() noexcept;
    void clear_cancellation() noexcept;
    bool cancellation_requested() const noexcept;
    std::atomic<bool>* cancellation_signal() noexcept {
        return &cancellation_requested_;
    }
    // Once an exception can no longer prove whether every backend update
    // completed, training and sidecar export fail closed. Recovery requires
    // reloading the exact model checkpoint followed by its matching Trainer
    // sidecar; a successful sidecar load clears this flag.
    bool optimizer_state_poisoned() const noexcept;
    void ensure_optimizer_state_usable() const;
    // Internal transaction hook used by the optimizer commit guard. Public so
    // the translation-unit-local guard can remain independent of Trainer's
    // storage layout; callers must never use it as a recovery mechanism.
    void mark_optimizer_state_poisoned() noexcept;

    // Python releases the GIL during training. Every property read/write must
    // therefore participate in the same transaction lock as optimizer,
    // scheduler and checkpoint operations instead of racing those methods.
    template <typename T>
    T synchronized_read(T Trainer::*member) const {
        std::lock_guard<std::recursive_mutex> lock(state_mutex_);
        return this->*member;
    }

    template <typename T>
    void synchronized_write(T Trainer::*member, const T& value) {
        std::lock_guard<std::recursive_mutex> lock(state_mutex_);
        this->*member = value;
    }

private:
    friend class InferenceEngine;
    friend class TrainingCheckpointSnapshot;
    float accumulate_gradients_impl(
        const std::vector<int>& tokens,
        const std::vector<int>& targets,
        Tensor* deferred_supervised_loss);
    // Deep, transactional copy used only by administrative training clones.
    // Keeping this private prevents a caller from supplying parameter vectors
    // captured outside the source model transaction. Durable OxtaMem stores
    // are rejected because mutating the same external arena before clone
    // publication would violate HTTP transaction isolation.
    void clone_runtime_state_to(
        Trainer& target,
        const std::vector<Parameter*>& source_parameters,
        const std::vector<Parameter*>& target_parameters) const;
    // Serializes optimizer, scheduler, RNG, auxiliary-memory, and checkpoint
    // transactions. Recursive because convenience entry points delegate to
    // lower-level training methods.
    mutable std::recursive_mutex state_mutex_;
    mutable std::optional<RuntimeExecutionIdentity> execution_identity_;
    // True only on the CPU clone owned by TrainingCheckpointSnapshot. Its
    // persisted identity intentionally describes the source GPU execution,
    // while its host tensors are merely transport buffers for serialization.
    bool portable_checkpoint_snapshot_ = false;
    const RuntimeExecutionIdentity& ensure_execution_identity_locked() const;
    std::atomic<bool> cancellation_requested_{false};
    std::atomic<bool> optimizer_state_poisoned_{false};
};

} // namespace nsos
