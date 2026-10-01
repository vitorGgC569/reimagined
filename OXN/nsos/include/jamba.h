#ifndef JAMBA_H
#define JAMBA_H

#include "autograd.h"
#include "bitlinear.h"
#include "chrass_layer_v2.h"
#include "embedding.h"
#include "gpu_execution.h"
#include "gpu_attention_training.h"
#include "gpu_moe_training.h"
#include "gpu_projection_group.h"
#include "gpu_kv_cache.h"
#include "kan.h"
#include "mamba2.h"
#include "mamba3_layer.h"
#include "mcts_reasoning.h"
#include "memory_system.h"
#include "module.h"
#include "nsos_config.h"
#include "self_healer.h"
#include "ttt_layer.h"
#ifdef USE_CUDA
#include "cuda/device_buffer.h"
#endif
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace nsos {

class LayerAuditCollector;
class Trainer;

struct AttentionCacheSnapshot {
  bool enabled = false;
  int cached_tokens = 0;
  int cache_page_tokens = 64;
  int cache_capacity_tokens = 0;
  Tensor key_cache;
  Tensor value_cache;
  std::shared_ptr<GpuKvCache> compact;
};

struct JambaBlockSessionSnapshot {
  bool has_attention = false;
  bool has_mamba = false;
  bool has_mamba3 = false;
  bool has_ttt = false;
  AttentionCacheSnapshot attention;
  MambaStreamSnapshot mamba;
  Mamba3SessionSnapshot mamba3;
  TTTSessionSnapshot ttt;
};

struct JambaSessionSnapshot {
  bool streaming_enabled = false;
  std::vector<int> input_ids;
  std::vector<JambaBlockSessionSnapshot> blocks;
};

struct RuntimeTelemetrySnapshot {
  struct MambaLayer {
    int layer_index = -1;
    size_t fast_path_hits = 0;
    size_t fast_path_fallbacks = 0;
    size_t faithful_forward_gpu_calls = 0;
    size_t faithful_forward_host_fallbacks = 0;
    size_t faithful_backward_gpu_calls = 0;
    size_t faithful_backward_host_fallbacks = 0;
    size_t faithful_streaming_gpu_calls = 0;
    size_t faithful_streaming_host_fallbacks = 0;
    size_t stream_priming_gpu_calls = 0;
    size_t stream_priming_host_fallbacks = 0;
    size_t faithful_recompute_forwards = 0;
    size_t faithful_selective_history_recomputes = 0;
    size_t faithful_full_block_recompute_forwards = 0;
    size_t faithful_warp_aggregated_backward_calls = 0;
    size_t faithful_deterministic_backward_calls = 0;
    size_t faithful_scalar_atomic_backward_calls = 0;
    size_t faithful_reduced_conv_backward_calls = 0;
    size_t faithful_generic_atomic_conv_backward_calls = 0;
    size_t faithful_peak_state_history_bytes = 0;
    size_t faithful_grouped_projection_forward_calls = 0;
    size_t faithful_grouped_projection_backward_calls = 0;
    size_t faithful_grouped_projection_cache_rebuilds = 0;
    size_t faithful_grouped_projection_full_forward_calls = 0;
    size_t faithful_grouped_projection_sensitive_forward_calls = 0;
    size_t faithful_grouped_projection_full_backward_calls = 0;
    size_t faithful_grouped_projection_sensitive_backward_calls = 0;
    std::string last_fallback_reason;
  };
  size_t mamba_fast_path_hits = 0;
  size_t mamba_fast_path_fallbacks = 0;
  size_t faithful_forward_gpu_calls = 0;
  size_t faithful_forward_host_fallbacks = 0;
  size_t faithful_backward_gpu_calls = 0;
  size_t faithful_backward_host_fallbacks = 0;
  size_t faithful_streaming_gpu_calls = 0;
  size_t faithful_streaming_host_fallbacks = 0;
  size_t stream_priming_gpu_calls = 0;
  size_t stream_priming_host_fallbacks = 0;
  size_t faithful_recompute_forwards = 0;
  size_t faithful_selective_history_recomputes = 0;
  size_t faithful_full_block_recompute_forwards = 0;
  size_t faithful_warp_aggregated_backward_calls = 0;
  size_t faithful_deterministic_backward_calls = 0;
  size_t faithful_scalar_atomic_backward_calls = 0;
  size_t faithful_reduced_conv_backward_calls = 0;
  size_t faithful_generic_atomic_conv_backward_calls = 0;
  size_t faithful_peak_state_history_bytes = 0;
  size_t faithful_grouped_projection_forward_calls = 0;
  size_t faithful_grouped_projection_backward_calls = 0;
  size_t faithful_grouped_projection_cache_rebuilds = 0;
  size_t faithful_grouped_projection_full_forward_calls = 0;
  size_t faithful_grouped_projection_sensitive_forward_calls = 0;
  size_t faithful_grouped_projection_full_backward_calls = 0;
  size_t faithful_grouped_projection_sensitive_backward_calls = 0;
  std::string mamba_last_fallback_reason;
  std::vector<MambaLayer> mamba_layers;
  std::vector<Mamba3Telemetry> mamba3_layers;
};

class Attention {
public:
  Attention(int d_model, int n_heads, int n_latents = 512, int n_kv_heads = 0,
            float rope_theta = 10000.0f, int sliding_window = 4096);
  Tensor forward(const Tensor &input, Context *ctx);
  Tensor backward(const Tensor &dy, Context *ctx);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void collect_bitlinear_layers(std::vector<BitLinear*>& out);
  void set_streaming_mode(bool enabled);
  void set_training_mode(bool enabled) { training_mode_ = enabled; }
  // Compatibility setter for checkpoints/configurations authored before exact
  // attention became a correctness requirement. Training always uses the
  // exact softmax-Jacobian path; disabling it is intentionally a no-op.
  void set_exact_training_path(bool enabled) { (void)enabled; }
  void set_exact_linear_mode(bool enabled);
  void set_batch_valid_lengths(const std::vector<int>& lengths) {
    active_batch_valid_lengths_ = lengths;
  }
  // SSA (Subquadratic Sparse Attention) opt-in.  When enabled, the CPU prefill
  // attention path routes each head through content-dependent block selection
  // (see sparse_attention.h) instead of full O(n^2) attention.  Default OFF ->
  // byte-identical to the existing path.
  void set_sparse_attention(bool enabled, int block_size = 64,
                            int top_k_blocks = 8, int local_blocks = 1,
                            int sink_blocks = 1) {
    sparse_enabled_ = enabled;
    ssa_block_size_ = block_size > 0 ? block_size : 64;
    ssa_top_k_blocks_ = top_k_blocks;
    ssa_local_blocks_ = local_blocks;
    ssa_sink_blocks_ = sink_blocks;
  }
  bool sparse_attention_enabled() const { return sparse_enabled_; }
  Parameter* sparse_selector_param() { return &ssa_wsel_; }
  // Trains the learned block selector (ssa_wsel_) by distilling the dense
  // attention pattern from the last exact-training forward into the sparse
  // selector.  Accumulates dWsel into ssa_wsel_.grad; returns the distill loss
  // (0 if the per-head tensors from the forward are unavailable).
  float accumulate_selector_distill_grad(float gradient_weight = 1.0f);
  void reset();
  // Public pre-allocation hook for inference paths.  Reserves the KV
  // cache for `total_tokens` slots up-front so the per-token decode
  // loop never triggers the page-growth path (default page size is
  // 64; without pre-alloc we realloc + memcpy every 64 generated
  // tokens, ~8 reallocs for 512 tokens).  Safe to call before any
  // forward; idempotent if already large enough; the rotary frequency
  // cache is recomputed if total_tokens exceeds the current capacity
  // of cos_cached/sin_cached.
  void reserve_kv_cache(int total_tokens, Device device, int batch_size = 1);
  AttentionCacheSnapshot snapshot_cache() const;
  std::vector<AttentionCacheSnapshot> snapshot_cache_batch() const;
  void restore_cache(const AttentionCacheSnapshot& snapshot);
  void restore_cache_batch(const std::vector<AttentionCacheSnapshot>& snapshots);
  int num_query_heads() const { return n_heads; }
  int num_kv_heads() const { return n_kv_heads; }

  // Detach the KV cache to a uniquely-owned buffer if a session snapshot is
  // sharing it (use_count > 1).  ensure_kv_cache_capacity() reallocates
  // copy-on-write when the buffer is shared; calling this BEFORE a CUDA-graph
  // capture (right after fork_session) moves that realloc + D2D copy OUTSIDE
  // the capture, so the recorded decode step sees a stable, owned cache and
  // does not realloc mid-capture (which would corrupt the graph).
  void make_kv_cache_unique();
  // Host mirror advance for graph REPLAYS: a replay executes the captured
  // kernels (which advance the device-side position) without ever entering
  // Attention::forward, so cached_tokens_ must be bumped externally to keep
  // snapshots/session forks truthful.
  void advance_cached_tokens_external() { ++cached_tokens_; }
  int cached_tokens() const { return cached_tokens_; }
  int kv_cache_capacity() const { return cache_capacity_tokens_; }

private:
  int d_model;
  int n_heads;
  int n_kv_heads;
  int kv_group_size;
  int head_dim;
  int n_latents;
  std::unique_ptr<BitLinear> q_down_proj, kv_down_proj, out_proj;
  GpuProjectionGroup decode_projections_;
  std::unique_ptr<attention_training::Provider> rdna_training_;
  bool saved_rdna_training_ = false;
  std::vector<float> cos_cached, sin_cached;
  int max_seq_len;
  int sliding_window_;
  float theta;
  bool streaming_inference_ = false;
  int cached_tokens_ = 0;
  int cached_batch_size_ = 0;
  int cache_page_tokens_ = 64;
  int cache_capacity_tokens_ = 0;
  Tensor key_cache_buffer_;
  std::shared_ptr<GpuKvCache> compact_kv_;
  bool compact_kv_requested() const;
  void ensure_compact_kv_capacity(int tokens, int batch);
  void materialize_compact_kv();
  Tensor value_cache_buffer_;
  bool training_mode_ = true;
  std::vector<int> active_batch_valid_lengths_;
  std::vector<int> saved_valid_lengths_;
  Tensor saved_q_rot_;
  Tensor saved_k_rot_;
  Tensor saved_v_heads_;
  Tensor saved_attn_probs_;
  Tensor saved_attn_lse_;       // tiled exact training: [B,S,H], no S*S cache
  Tensor saved_attn_output_;    // pre-output projection, for softmax VJP delta
  int saved_input_rank_ = 0;
  // GPU training path (src/cuda/attention_train_kernels.cu): device copies of
  // the rotary tables, lazily (re)uploaded when cos_cached/sin_cached grow, and
  // the device-resident exact-cache backward.  Behavior-neutral on CPU builds.
  Tensor rope_cos_gpu_;
  Tensor rope_sin_gpu_;
  size_t rope_gpu_uploaded_ = 0;
  void ensure_rope_gpu_cache();
  Tensor forward_exact_gpu(const Tensor& q_flat, const Tensor& kv_flat,
                           int batch_size, int seq_len, int kv_dim,
                           float scale);
  Tensor backward_exact_gpu(const Tensor& dy, int batch_size, int seq_len,
                             int kv_dim, float scale);
  Tensor backward_sparse_gpu(const Tensor& dy, int batch_size, int seq_len,
                             int kv_dim, float scale);
  // SSA opt-in state (default OFF preserves exact existing behavior).
  bool sparse_enabled_ = false;
  int ssa_block_size_ = 64;
  int ssa_top_k_blocks_ = 8;
  int ssa_local_blocks_ = 1;
  int ssa_sink_blocks_ = 1;
  Parameter ssa_wsel_;  // learned block-selection routing (init identity)
  void precompute_freqs_cis();
  // N7 (RoPE length): grow the cos/sin frequency tables so they cover positions
  // up to `max_pos_exclusive` (doubles max_seq_len until it fits, then
  // recomputes the tables and forces a GPU re-upload).  Replaces the old silent
  // clamp at max_seq_len-1, which made every position >= max_seq_len share one
  // rotation (loss of positional distinction / no length extrapolation).
  void ensure_freqs_capacity(int max_pos_exclusive);
  void clear_kv_cache();
  void ensure_kv_cache_capacity(int required_tokens, Device device, int batch_size = 1);
  void append_kv_cache_token(const float* key_ptr,
                             Device key_device,
                             const float* value_ptr,
                             Device value_device);
  void append_kv_cache_batch_tokens(const float* key_ptr,
                                    Device key_device,
                                    const float* value_ptr,
                                    Device value_device,
                                    int batch_size);
  const float* kv_cache_token_ptr(const Tensor& cache, int token_index) const;
  const float* kv_cache_token_ptr(const Tensor& cache, int batch_index, int token_index) const;
  std::pair<Tensor, Tensor> apply_rope(const Tensor &q, const Tensor &k, int start_pos = 0);
  std::pair<Tensor, Tensor> apply_rope_backward(const Tensor& grad_q_rot,
                                                const Tensor& grad_k_rot,
                                                int start_pos = 0);
  // Dedicated SSA (sparse attention) forward for the non-streaming rank-1/2/3
  // paths.  Routes per-head through sparse_selective_attention (GPU kernel when
  // on device) instead of the dense fast-paths; entered from forward() when
  // sparse attention is enabled.  Keeps the dense maze untouched.
  Tensor sparse_forward(const Tensor &input, Context *ctx);
};

class MoERouter {
public:
  MoERouter(int d_model, int n = 256, int k = 8);
  // valid_rows is a flattened [rows] mask (1 = real token, 0 = padding).
  // Routing still returns a shape-preserving tensor for every row, while all
  // load-balancing statistics and auxiliary-loss gradients exclude padding.
  // An empty mask means every row is valid (rank-1/2 and legacy callers).
  std::pair<Tensor, Tensor> forward(
      const Tensor &x, const std::vector<uint8_t> &valid_rows = {});
  Tensor backward(const Tensor &grad_logits);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void collect_bitlinear_layers(std::vector<BitLinear*>& out);
  int num_experts, top_k;
  float aux_loss_coef;
  std::vector<float> expert_loads;
  // ── Differentiable Switch-Transformer load-balancing aux loss ────────────
  // The legacy load balancing (trainer.cpp apply_moe_aux_regularization) adds a
  // constant per expert row to the gate gradient — NOT the gradient of any loss,
  // and largely a no-op once the gate is ternary.  This computes the proper
  // Switch aux loss  L = coef·N·Σ_e f_e·P_e  (f_e = hard dispatch fraction,
  // P_e = mean routing prob) and its EXACT gradient w.r.t. the router logits,
  // then backpropagates it through the gate (accumulating gate weight grads).
  // Returns the aux loss value (for logging). Requires a prior forward()
  // (uses saved pre-mask probabilities). This is the trainer's objective;
  // the obsolete gradient-only imbalance heuristic has no runtime path.
  float accumulate_switch_aux_grad(float coef);
  // Same exact objective/VJP, with its scalar retained on the gate's device.
  // Does not materialize load telemetry. The trainer reduces these scalars
  // before its single logging/finite-control boundary.
  Tensor accumulate_switch_aux_grad_device(float coef);
  // A training step may contain several length buckets / micro-chunks.  Keep
  // every router forward until the once-per-step Switch objective is formed so
  // its value and gradient are invariant to chunking.
  void begin_aux_accumulation();
  void finalize_aux_accumulation(bool materialize_loads = true);
  void cancel_aux_accumulation();
  // Explicit telemetry boundary; training can keep loads on-device until the
  // step is finalized, while an enabled audit can request the latest forward.
  void materialize_expert_loads();
  // Pure, testable core: given pre-mask softmax probs [T, N] and the active
  // top_k, returns the gradient of the Switch aux loss w.r.t. the logits
  // (same shape) and writes the loss value to *out_loss if non-null.  f_e uses
  // a stop-gradient hard top-k count (standard).  Finite-difference gradchecked
  // in tests/test_gradcheck.cpp.
  static Tensor switch_aux_grad_logits(const Tensor &probs, int top_k,
                                       float coef, float *out_loss);
  static Tensor switch_aux_grad_logits_device(const Tensor &probs, int top_k,
                                              float coef, Tensor *out_loss);
  // ── Task gradient to the router (opt-in NSOS_MOE_ROUTER_GRAD) ─────────────
  // The forward scales each expert output by its routing weight w[r,e], but the
  // legacy backward never propagated dL/dw to the gate — so the router learned
  // ONLY from the (aux) balancing term, never from the task loss.  Given
  // g_w[r,e] = sum_dim(dy[r]·expert_out_e[r]) (the gradient w.r.t. the routing
  // weights, supplied by the block), this backprops through the top-k
  // renormalization and the softmax to the logits and into the gate.  Default
  // OFF preserves the historical behavior.
  Tensor accumulate_task_router_grad(const Tensor &g_w);
  // Pure, testable core: gradient of the routing weights w.r.t. the logits,
  // given pre-mask softmax probs [T,N], the upstream g_w [T,N] and top_k.
  // Backprops renorm(top-k(softmax)).  Finite-difference gradchecked.
  static Tensor router_grad_logits(const Tensor &probs, const Tensor &g_w,
                                   int top_k);
  std::unique_ptr<BitLinear> gate, shared_expert_gate, shared_expert_up, shared_expert_down;

private:
  struct AuxForwardRecord {
    Tensor probs;
    Tensor input;
    std::vector<uint8_t> valid_rows;
  };
  // Pre-mask softmax routing probabilities from the last forward (same device),
  // needed by accumulate_switch_aux_grad.  Empty until the first forward.
  Tensor saved_probs_;
  bool aux_accumulation_active_ = false;
  std::vector<AuxForwardRecord> aux_forward_records_;
  std::vector<float> accumulated_expert_loads_;
  Tensor pending_expert_loads_device_;
  Tensor accumulated_expert_loads_device_;
};

class JambaBlock {
public:
  JambaBlock(int d_model,
             bool is_attn,
             bool is_moe_flag,
             bool is_ttt_layer,
             int layer_idx,
             int total_layers,
             int query_heads,
             int kv_heads,
             int configured_experts,
             int configured_top_k,
             bool exact_attention_training,
             float dropout_rate = 0.0f,
             bool use_gradient_checkpointing = false,
             // ── Nemotron K·m invariant (Cherry-pick #4) ──
             // When > 0, overrides the default expert FFN intermediate
             // dimension (m).  Default 0 means use the historical d_model*4
             // value, preserving byte-for-byte the existing behavior.
             // See OXN/nsos/docs/NEMOTRON_KM_INTEGRATION.md.
             int configured_expert_hidden_dim = 0,
             // ── CHRASS topological injection (2026-05-25) ──
             // When true, instantiate a ChrassLayer of size d_model with a
             // random sparse adjacency at construction.  The layer runs
             // IN PARALLEL with FFN/MoE and its output is added to the
             // block residual.  Validated standalone at 26/26 tests; see
             // OXN/nsos/docs/CHRASS_VALIDATION_REPORT.md.
             bool use_chrass = false,
             float chrass_density = 0.10f,
             uint32_t chrass_seed = 0u,
             // ── KAN FFN ── when true, a non-MoE block uses a BitFastKANLayer
             // in place of the dense gate-up/down FFN.
             bool use_kan = false,
             // ── Corrected Mamba-2 SSD (K1) — defaults ON via ModelConfig.
             // Selects the validated selective-SSM path (independent
             // projections + conv1d + linear readout + SiLU gate) and the full
             // N-dimensional SSD state.
             bool mamba_proper_ssm = true,
             bool mamba_state_expansion = true,
             int mamba_d_state = 64,
             int mamba_conv_kernel = 4,
             bool mamba2_faithful = true,
             int mamba_expand = 2,
             int mamba_head_dim = 64,
             int mamba_n_groups = 1,
             // RoPE base (theta) for the attention layers; larger = more
             // position-invariant (content-recall) dimensions.
             float rope_theta = 10000.0f,
             int sliding_window = 4096,
             int max_context_tokens = 4096,
             HybridComposition hybrid_composition =
                 HybridComposition::LegacyReplacement,
             bool faithful_attention_linears = false,
             float hybrid_mamba_gate_init = 1.0f,
             float hybrid_attention_gate_init = 0.01f,
             float hybrid_ffn_gate_init = 0.01f,
             bool mamba3_enabled = false,
             const Mamba3Config& mamba3_config = {});
  ~JambaBlock();
  Tensor forward(const Tensor &x, Context *ctx);
  Tensor backward(const Tensor &dy, Context *ctx);
  void reset();
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void collect_bitlinear_layers(std::vector<BitLinear*>& out);
  void set_streaming_inference(bool enabled);
  void set_training_mode(bool enabled);
  void set_batch_valid_lengths(const std::vector<int>& lengths);
  void set_dropout_sequence(uint64_t sequence) { dropout_sequence_ = sequence; }
  void set_audit_collector(LayerAuditCollector* collector) { audit_collector_ = collector; }
  // Pacote A.1: when > 0 AND training_mode_ is false, forward_moe uses
  // this top-k instead of router->top_k.  Lets us train with top-2 and
  // decode with top-1 (cuts expert FLOPs ~50% per token).  No effect
  // during training or when the override is 0.  Per-block setting; the
  // JambaModel-level set_moe_inference_top_k iterates and sets each.
  void set_inference_top_k_override(int k) { inference_top_k_override_ = k; }
  int inference_top_k_override() const { return inference_top_k_override_; }
  std::string audit_block_type() const;
  JambaBlockSessionSnapshot snapshot_session_state(bool device_resident = false) const;
  std::vector<JambaBlockSessionSnapshot> snapshot_session_state_batch(bool device_resident = false) const;
  void restore_session_state(const JambaBlockSessionSnapshot& snapshot);
  void restore_session_state_batch(const std::vector<JambaBlockSessionSnapshot>& snapshots);
  bool uses_attention() const { return is_attention; }
  bool uses_moe() const { return is_moe; }
  bool uses_ttt() const { return is_ttt; }

private:
  Tensor forward_moe(const Tensor &x, Context *ctx, const std::string &ln);
  Tensor backward_moe(const Tensor &dy, Context *ctx, const std::string &ln, const Tensor &x);
  // Phase 4-extended: GPU-resident batched MoE forward.  Uses count +
  // exclusive-scan + gather + per-expert forward + scatter-add kernels
  // to replace the per-expert std::memcpy loops in forward_moe.
  // Caller must verify GPU eligibility (target_device, gpu_custom_kernels)
  // before invoking — this function does no fallback.
  Tensor forward_moe_gpu_batched(const Tensor &x, const Tensor &weights,
                                  int rows, int dim, int effective_top_k);
  Tensor forward_moe_gpu_sparse_decode(const Tensor& x, const Tensor& weights,
                                       int dim);
#ifdef USE_CUDA
  cuda_detail::DeviceBuffer<GpuLinearView> moe_decode_views_;
#endif
  std::vector<GpuLinearView> moe_decode_host_views_;

  // GPU-resident batched MoE backward, paired with the forward above.
  // Recomputes the permutation/gather pipeline from saved_moe_weights_
  // (which the forward syncs to host) and dispatches per-expert
  // backward calls on contiguous slices.  Returns the input gradient
  // [rows, dim] in the same device as dy.
  Tensor backward_moe_gpu_batched(const Tensor &dy, const Tensor &x);
  Tensor mamba_forward(const Tensor& input, Context* context);
  Tensor mamba_backward(const Tensor& gradient, Context& context);
  // Releases only per-forward wrapper state after a successful backward.
  // Child modules own and release their own tapes; reusable allocation
  // capacity (for example the MoE permutation buffer) is intentionally kept.
  void release_consumed_backward_state();
  bool is_attention, is_moe, is_ttt;
  bool hybrid_parallel_ = false;
  bool faithful_mamba_core_only_ = false;
  bool learnable_core_norm_ = false;
  bool learnable_attention_norm_ = false;
  bool learnable_ff_norm_ = false;
  float core_norm_eps_ = 1e-6f;
  Parameter core_norm_weight_;
  Parameter attention_norm_weight_;
  Parameter ff_norm_weight_;
  // Deterministic LayerScale (per-channel gamma, init 0.01) on faithful legacy
  // replacement-attention blocks. Parallel hybrid blocks use independent
  // Mamba/Attention/FFN gates instead.
  bool use_attn_layerscale_ = false;
  Parameter attn_layerscale_;
  Parameter mamba_gate_;
  Parameter attention_gate_;
  Parameter ffn_gate_;
  int layer_idx, total_layers, d_model, num_experts;
  float dropout_rate_;
  bool training_mode_ = true;
  uint64_t dropout_sequence_ = 0;
  std::vector<int> active_batch_valid_lengths_;
  int inference_top_k_override_ = 0;  // Pacote A.1; 0 = no override
  int last_batch_size_ = 0;
  Tensor saved_input_;
  Tensor saved_core_rms_;
  Tensor saved_core_norm_;
  Tensor saved_attention_rms_;
  Tensor saved_attention_norm_;
  Tensor saved_ff_rms_;
  Tensor saved_residual_;
  Tensor saved_ff_norm_;
  Tensor saved_ff_hidden_pre_;
  // Pre-scale attention core/ff outputs saved for the LayerScale gamma grad.
  Tensor saved_ls_core_;
  Tensor saved_ls_ff_;
  Tensor saved_mamba_core_;
  Tensor saved_attention_core_;
  // Training dropout masks captured in forward and REAPPLIED in backward (the
  // recurrence has no autograd tape, so the mask must be carried explicitly;
  // omitting it let gradients flow through dropped/rescaled units).  Empty when
  // dropout is off for this forward.
  Tensor saved_drop_core_;
  Tensor saved_drop_mamba_;
  Tensor saved_drop_attention_;
  Tensor saved_drop_moe_;
  Tensor saved_drop_ff_hidden_;
  Tensor saved_drop_ff_out_;
  Tensor saved_moe_weights_;
  std::vector<std::vector<int>> saved_moe_rows_;
  // LEARN S1 (squared ReLU backward): per-expert pre-activation cache.
  // The forward MoE path saves the output of expert_gate_up BEFORE the
  // squared_relu activation, indexed by expert id.  Backward reads
  // these to apply squared_relu_backward correctly (dx = dy * 2 *
  // max(0, pre)).  Pre-existing code went straight from expert_down
  // backward to expert_gate_up backward without ANY activation
  // gradient, which silently broke the chain rule for both ReLU and
  // squared_relu — only the sign was preserved, magnitude was wrong.
  // This vector is sized to num_experts and only the entries that
  // actually got non-empty input during forward are populated.
  std::vector<Tensor> saved_moe_pre_activations_;
  // Task-router-grad: per-expert unscaled output rows for the ordered path.
  std::vector<Tensor> saved_moe_expert_out_;
  // Batched GPU routing state retained per block. Counts/offsets are the only
  // small host metadata needed to delimit expert GEMMs. Per-token permutation,
  // scales and unscaled expert outputs remain device-resident through backward.
#ifdef USE_CUDA
  cuda_detail::DeviceBuffer<int> saved_moe_permutation_device_;
  cuda_detail::DeviceBuffer<int> saved_moe_inverse_device_;
  cuda_detail::DeviceBuffer<int> saved_moe_offsets_device_;
#endif
  bool saved_moe_ordered_device_ = false;
  std::shared_ptr<GpuMoeTraining> grouped_moe_training_;
  bool saved_moe_grouped_training_ = false;
  Tensor saved_moe_scale_device_;
  Tensor saved_moe_permuted_output_;
  std::vector<int> saved_moe_offsets_host_;
  std::vector<int> saved_moe_counts_host_;
  int saved_moe_n_active_ = 0;
  LayerAuditCollector* audit_collector_ = nullptr;
public:
  std::unique_ptr<Attention> attn_layer;
  std::unique_ptr<Mamba2SSD> mamba_layer;
  std::unique_ptr<Mamba3Layer> mamba3_layer;
  std::unique_ptr<TTTLayer> ttt_layer;
  std::unique_ptr<MoERouter> router;
  std::vector<std::unique_ptr<BitLinear>> expert_gate_up, expert_down;
  std::unique_ptr<BitLinear> ffn_gate_up, ffn_down;
  // ── KAN FFN slot — non-null only when use_kan=true (replaces ffn_gate_up/down) ──
  std::unique_ptr<BitFastKANLayer> kan_ffn;
  // ── CHRASS slot (parallel with FFN) — nullopt unless use_chrass=true ──
  std::unique_ptr<ChrassLayer> chrass_layer;
};

class JambaModel : public Module {
public:
  // Provider tapes must release while execution_context_ still owns its stream.
  ~JambaModel() override { layers.clear(); }
  std::vector<std::unique_ptr<JambaBlock>> layers;
  std::unique_ptr<Embedding> embedding;
  std::unique_ptr<BitLinear> value_head;

  JambaModel(int num_layers, int d_model, int vocab_size = 128000, Device device = Device::CPU);
  JambaModel(const ModelConfig& config, Device device = Device::CPU);
  Tensor forward(const Tensor &x) override;
  Tensor forward(const Tensor &x, Context *ctx);
  void to(Device device) override;
  Tensor forward_ids(const std::vector<int> &ids, Context *ctx = nullptr);
  Tensor forward_ids_last(const std::vector<int>& ids, Context* ctx = nullptr);
  Tensor forward_ids_batch(const std::vector<std::vector<int>>& batch_ids,
                           Context* ctx = nullptr);
  Tensor forward_ids_training_hidden(const std::vector<int>& ids, Context* ctx = nullptr);
  Tensor forward_ids_batch_training_hidden(const std::vector<std::vector<int>>& ids, Context* ctx = nullptr);
  TiledCrossEntropyResult tiled_head_loss(const TiledCrossEntropyOptions& options);
  void backward_training_hidden(const Tensor& gradient, Context& ctx);
  Tensor forward_trunk(const std::vector<int> &ids, Context *ctx = nullptr);
  Tensor forward_trunk_batch(const std::vector<std::vector<int>>& batch_ids,
                             Context* ctx = nullptr);
  Tensor forward_embedding(const Tensor &x, Context *ctx = nullptr);
  Tensor reason(const Tensor &x, int num_simulations = 100);
  void set_reasoning_policy(ReasoningPolicy policy);
  void clear_reasoning_policy();
  bool has_reasoning_policy() const;
  VerifiedReasoningReport last_reasoning_report() const;
  std::vector<Parameter *> parameters() override;
  // Canonical optimizer exclusions owned by selected Mamba3 blocks: dt_bias and D.
  std::vector<Parameter *> no_weight_decay_parameters();
  // Logical parameter paths that share a canonical registry tensor. Each pair
  // is {alias, canonical}; aliases are excluded from optimizer traversal so a
  // shared weight is updated exactly once.
  std::vector<std::pair<std::string, std::string>>
  parameter_aliases() const;
  void save(const std::string &filename);
  // Export the full-precision checkpoint and its edge-linear companion under
  // one model execution lock. This prevents a training update from producing
  // artifacts from two different parameter generations.
  ModelConfig save_model_pack_artifacts(
      const std::string& weights_filename,
      const std::string& edge_linear_filename);
  void load(const std::string &filename, bool strict = true);
  Tensor forward_thought(const Tensor &x, int steps);
  void run_reasoning_loop(int iterations);
  Tensor run_reasoning_loop(const Tensor &x, int iterations);
  void backward_external(const Tensor &grad, Context &ctx);
  void backward_embedding(const Tensor &grad, Context &ctx);
  void backward(const Tensor &grad, Context &ctx);
  std::vector<BitLinear*> collect_bitlinear_layers();
  void set_reference_path(bool use_reference_path);
  // Enable SSA (Subquadratic Sparse Attention) on every attention layer.
  void set_sparse_attention(bool enabled, int block_size = 64,
                            int top_k_blocks = 8, int local_blocks = 1,
                            int sink_blocks = 1);
  // Accumulate learned-selector (SSA) distillation grads on every attention
  // layer (call after an exact-training forward).  Returns total distill loss.
  float accumulate_sparse_selector_grads(float gradient_weight = 1.0f);
  // Phase 5b deeper: enable/disable the GPU __dp4a packed-inference
  // fast path on every BitLinear in the model in one call.  Safe to
  // toggle at runtime.  Caller is responsible for ensuring weights
  // have already been packed (via repack_weights / load_edge_linear_pack)
  // when enabling — the BitLinear forward path falls back to float
  // matmul if packed weights are not yet materialized.
  void set_gpu_packed_inference(bool enabled);
  // Pacote A.1: when k >= 1, every MoE block uses this top-k during
  // inference (training_mode_=false).  Pass 0 to clear and fall back to
  // router->top_k.  Typical use: train with top-2, decode with top-1.
  void set_moe_inference_top_k(int k);
  // Pre-allocate KV cache on every attention layer up to `total_tokens`.
  // Eliminates page-growth realloc+memcpy spikes during long
  // generations (default page size is 64; for 512 tokens that's 8
  // reallocs * O(cache_size_so_far) memcpy each — measurable on
  // sustained decode workloads).  Caller passes prompt_len +
  // max_new_tokens before the decode loop.  Idempotent.
  void reserve_kv_cache(int total_tokens, Device device, int batch_size = 1);
  void release_full_precision_linear_weights();
  void save_edge_linear_pack(const std::string& path);
  void load_edge_linear_pack(const std::string& path, bool release_full_precision = true);
  bool supports_streaming_inference() const;
  bool supports_batched_streaming_inference() const;
  void set_streaming_inference(bool enabled);
  void set_training_mode(bool enabled);
  bool training_mode() const;
  uint64_t training_rng_sequence() const;
  void set_training_rng_sequence(uint64_t sequence);
  JambaSessionSnapshot fork_session(bool device_resident = false) const;
  std::vector<JambaSessionSnapshot> fork_session_batch(bool device_resident = false) const;
  void restore_session(const JambaSessionSnapshot& snapshot);
  void restore_session_batch(const std::vector<JambaSessionSnapshot>& snapshots);
  void reset_session();
  // ── HIP/CUDA graph decode (opt-in NSOS_GPU_GRAPH_DECODE=1) ──────────
  // Captures ONE streaming single-token forward into a CUDA graph and replays
  // it per generated token, collapsing the per-token kernel-launch cascade
  // into a single graph launch on the model's explicit execution stream.
  // Requires GPU streaming after prefill, device-only routing, no TTT/sparse
  // attention, no forced synchronization and a pre-reserved KV cache.
  // returns the step's logits, or an EMPTY Tensor whenever the graph path is
  // unavailable/disabled — the caller then falls back to forward_ids({token})
  // with identical results.  1st call runs eagerly (warm-up: memory pool,
  // lazy device state, inference weight caches); 2nd call captures, verifies
  // instantiation and executes; later calls replay. Capture failures restore
  // the snapshot and disable this path. Launch failures throw because device
  // state may have advanced; the caller must reset the session. Logits alias
  // a stable captured output; clone before retaining across later replays.
  Tensor forward_ids_decode_graph(int token);
  bool decode_graph_active() const;
  std::string decode_graph_status() const;
  void set_hamiltonian_mode(bool enabled);
  Tensor run_simd_inference(const Tensor &x, int steps);
  void session_adapt(const Tensor &x, const Tensor &y);
  const ModelConfig& model_config() const { return model_config_; }
  void reset_runtime_telemetry();
  RuntimeTelemetrySnapshot runtime_telemetry() const;
  void set_audit_collector(LayerAuditCollector* collector);
  LayerAuditCollector* audit_collector() const;
  // ── OPT-IN profiler attachment (zero-overhead when null) ────────────
  // Sets a pointer to a profiler::InferenceProfiler instance.  When
  // null (the default and the production case), the model forward
  // path skips all profiling work via a single null check.  When
  // non-null, the model emits begin_event/end_event for every layer
  // and the major ops inside each layer.  The profiler lives in the
  // separate `nsos_profiler` static library which is NOT linked into
  // production builds — calling attach_profiler from a production
  // build (which never sees the profiler header) is impossible.
  //
  // The signature uses `void*` to keep the production `nsos_core`
  // build free of any #include of the profiler header.  Internally,
  // the profiler-aware build path reinterprets back to the typed
  // pointer.  This is a deliberate choice: production headers don't
  // need to know the profiler type exists.
  // Function-pointer + data attach.  We store the callbacks AS
  // POINTERS ON THE INSTANCE (not in globals) so that the Windows
  // DLL boundary between nsos_ext.pyd (which has the model code) and
  // nsos_profiler_ext.pyd (which provides the callbacks) doesn't
  // matter — function-pointer values are uniform across DLLs in the
  // same process, and we never need a shared global symbol.
  //
  // Both callbacks take (void* profiler_opaque, int layer_idx) for
  // begin and (void* profiler_opaque) for end.  The profiler library
  // installs them by passing raw function-pointer addresses cast to
  // uintptr_t (and back through reinterpret_cast inside set_layer_callbacks).
  using ProfilerBeginLayerFn = void (*)(void* profiler, int layer_idx);
  using ProfilerEndLayerFn   = void (*)(void* profiler);
  void attach_profiler(void* profiler);
  void set_profiler_callbacks(ProfilerBeginLayerFn begin_fn,
                              ProfilerEndLayerFn end_fn);
  void* profiler() const;
  ProfilerBeginLayerFn profiler_begin_layer_fn() const;
  ProfilerEndLayerFn profiler_end_layer_fn() const;
  void record_audit_token_context(const std::vector<int>& token_ids_sample,
                                  size_t batch_size,
                                  size_t prompt_tokens_total,
                                  size_t prompt_tokens_used,
                                  int context_limit,
                                  bool truncated);
private:
  friend class Trainer;
  friend class InferenceEngine;
  // A model owns mutable forward state (training RNG sequence, saved
  // activations, streaming caches and audit context).  Concurrent callers of
  // one model instance therefore execute atomically.  A recursive mutex is
  // intentional: convenience entry points such as forward_ids() delegate to
  // forward().  Throughput-oriented serving should use the native batch API;
  // this lock provides a correctness guarantee for accidental shared-model
  // concurrency instead of exposing data races.
  mutable std::recursive_mutex execution_mutex_;
  gpu::ExecutionContext execution_context_;
  ReasoningPolicy reasoning_policy_;
  VerifiedReasoningReport last_reasoning_report_;
  int num_layers, d_model, vocab_size;
  Device device;
  ModelConfig model_config_;
  // CUDA-graph decode state (pimpl; defined in jamba.cpp, USE_CUDA only —
  // stays empty otherwise).  shared_ptr so the incomplete type is fine here.
  std::shared_ptr<struct JambaDecodeGraph> decode_graph_;
  bool decode_graph_disabled_ = false;
  std::string decode_graph_status_ = "not_attempted";
  void invalidate_decode_graph_(const char* reason);
  // N6 (weight tying): when on, value_head (LM head) shares the embedding's
  // weight BUFFER (both are [vocab, d_model]).  apply_weight_tying_() re-points
  // the shared_ptr storage; it is idempotent and re-applied after every to()
  // (device move reallocates) and load() (deserialize reallocates).  The shared
  // matrix is trained as ONE parameter (embedding.weight): value_head.weight is
  // dropped from parameters() and its gradient is folded into embedding.weight
  // at the end of backward().
  bool tie_word_embeddings_ = false;
  bool weight_tied_ = false;
  uint64_t tied_embedding_version_observed_ = 0;
  Parameter final_norm_weight_;
  // Built exactly once after topology construction. Parameter names are then
  // frozen and become durable identities across training, audit and
  // serialization. This removes mutable/global renaming from parameters().
  std::vector<Parameter*> parameter_registry_;
  std::vector<std::string> parameter_registry_names_;
  bool parameter_registry_initialized_ = false;
  void initialize_parameter_registry_();
  void apply_weight_tying_();
  void refresh_weight_tying_cache_version_();
  void validate_token_ids(const std::vector<int>& ids,
                          const char* operation) const;
  bool streaming_inference_enabled_ = false;
  bool training_mode_ = true;
  Tensor forward_impl(const Tensor& x, Context* ctx, bool last_logits_only, bool headless = false);
  Tensor forward_ids_impl(const std::vector<int>& ids, Context* ctx,
                          bool last_logits_only, bool headless = false);
  Tensor forward_ids_batch_impl(const std::vector<std::vector<int>>& ids, Context* ctx, bool headless);
  void backward_impl(const Tensor& grad, Context& ctx, bool headless);
  bool headless_backward_ = false;
  bool tiled_head_ready_ = false;
  // A successful training forward grants exactly one model-level backward.
  // Consuming the ticket before any gradient mutation makes duplicate or
  // failed backwards fail closed instead of partially accumulating updates.
  bool backward_ready_ = false;
  // Sparse-selector distillation has its own one-shot gradient path, executed
  // before the main model backward. Keeping a separate ticket prevents stale
  // attention caches or accidental double accumulation.
  bool sparse_selector_backward_ready_ = false;
  void invalidate_backward_state_() noexcept {
    backward_ready_ = false;
    sparse_selector_backward_ready_ = false;
    headless_backward_ = false;
    tiled_head_ready_ = false;
  }
  void grant_backward_state_() noexcept {
    backward_ready_ = training_mode_;
    sparse_selector_backward_ready_ = training_mode_;
  }
  // Monotonic training-forward id used by stochastic layers.  Session resets
  // deliberately do not rewind it; a full training checkpoint persists it.
  uint64_t training_rng_sequence_ = 0;
  std::vector<int> last_input_ids_;
  std::vector<std::vector<int>> last_input_batches_;
  std::vector<int> last_input_batch_lengths_;
  Tensor saved_final_hidden_;
  Tensor saved_final_rms_;
  Tensor saved_final_norm_;
  LayerAuditCollector* audit_collector_ = nullptr;
  // Opt-in profiler state.  All three default to null; production
  // forward path is a single load + branch when set is empty.
  void*                profiler_              = nullptr;
  ProfilerBeginLayerFn profiler_begin_layer_  = nullptr;
  ProfilerEndLayerFn   profiler_end_layer_    = nullptr;
};

class D2FDecoder {
public:
    JambaModel* model;
    D2FDecoder(JambaModel* m) : model(m) {}
    
    // Geração cancelável via contexto com full sampling pipeline:
    // temperature, top-p (nucleus), top-k, EOS token stopping criterion
    std::vector<int> generate(
        const std::vector<int>& prompt,
        int length,
        Context* ctx = nullptr,
        float temperature  = 0.7f,
        float top_p        = 0.9f,
        int   top_k        = 50,
        int   eos_token_id = 0
    );
};

} // namespace nsos
#endif
