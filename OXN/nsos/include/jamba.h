#ifndef JAMBA_H
#define JAMBA_H

#include "autograd.h"
#include "bitlinear.h"
#include "chrass_layer_v2.h"
#include "embedding.h"
#include "mamba2.h"
#include "mcts_reasoning.h"
#include "memory_system.h"
#include "module.h"
#include "nsos_config.h"
#include "self_healer.h"
#include "ttt_layer.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nsos {

class LayerAuditCollector;

struct AttentionCacheSnapshot {
  bool enabled = false;
  int cached_tokens = 0;
  int cache_page_tokens = 64;
  int cache_capacity_tokens = 0;
  Tensor key_cache;
  Tensor value_cache;
};

struct JambaBlockSessionSnapshot {
  bool has_attention = false;
  bool has_mamba = false;
  bool has_ttt = false;
  AttentionCacheSnapshot attention;
  MambaStreamSnapshot mamba;
  TTTSessionSnapshot ttt;
};

struct JambaSessionSnapshot {
  bool streaming_enabled = false;
  std::vector<int> input_ids;
  std::vector<JambaBlockSessionSnapshot> blocks;
};

struct RuntimeTelemetrySnapshot {
  size_t mamba_fast_path_hits = 0;
  size_t mamba_fast_path_fallbacks = 0;
  std::string mamba_last_fallback_reason;
};

class Attention {
public:
  Attention(int d_model, int n_heads, int n_latents = 512, int n_kv_heads = 0);
  Tensor forward(const Tensor &input, Context *ctx);
  Tensor backward(const Tensor &dy, Context *ctx);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void collect_bitlinear_layers(std::vector<BitLinear*>& out);
  void set_streaming_mode(bool enabled);
  void set_training_mode(bool enabled) { training_mode_ = enabled; }
  void set_exact_training_path(bool enabled) { exact_training_path_ = enabled; }
  void set_batch_valid_lengths(const std::vector<int>& lengths) {
    active_batch_valid_lengths_ = lengths;
  }
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

private:
  int d_model;
  int n_heads;
  int n_kv_heads;
  int kv_group_size;
  int head_dim;
  int n_latents;
  std::unique_ptr<BitLinear> q_down_proj, kv_down_proj, out_proj;
  std::vector<float> cos_cached, sin_cached;
  int max_seq_len;
  float theta;
  bool streaming_inference_ = false;
  int cached_tokens_ = 0;
  int cached_batch_size_ = 0;
  int cache_page_tokens_ = 64;
  int cache_capacity_tokens_ = 0;
  Tensor key_cache_buffer_;
  Tensor value_cache_buffer_;
  bool training_mode_ = true;
  std::vector<int> active_batch_valid_lengths_;
  std::vector<int> saved_valid_lengths_;
  Tensor saved_q_rot_;
  Tensor saved_k_rot_;
  Tensor saved_v_heads_;
  Tensor saved_attn_probs_;
  int saved_input_rank_ = 0;
  bool exact_training_path_ = true;
  void precompute_freqs_cis();
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
};

class MoERouter {
public:
  MoERouter(int d_model, int n = 256, int k = 8);
  std::pair<Tensor, Tensor> forward(const Tensor &x);
  Tensor backward(const Tensor &grad_logits);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void collect_bitlinear_layers(std::vector<BitLinear*>& out);
  int num_experts, top_k;
  float aux_loss_coef;
  std::vector<float> expert_loads;
  float compute_aux_loss();
  std::unique_ptr<BitLinear> gate, shared_expert_gate, shared_expert_up, shared_expert_down;
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
             uint32_t chrass_seed = 0u);
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
  void set_audit_collector(LayerAuditCollector* collector) { audit_collector_ = collector; }
  // Pacote A.1: when > 0 AND training_mode_ is false, forward_moe uses
  // this top-k instead of router->top_k.  Lets us train with top-2 and
  // decode with top-1 (cuts expert FLOPs ~50% per token).  No effect
  // during training or when the override is 0.  Per-block setting; the
  // JambaModel-level set_moe_inference_top_k iterates and sets each.
  void set_inference_top_k_override(int k) { inference_top_k_override_ = k; }
  int inference_top_k_override() const { return inference_top_k_override_; }
  std::string audit_block_type() const;
  JambaBlockSessionSnapshot snapshot_session_state() const;
  std::vector<JambaBlockSessionSnapshot> snapshot_session_state_batch() const;
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

  // GPU-resident batched MoE backward, paired with the forward above.
  // Recomputes the permutation/gather pipeline from saved_moe_weights_
  // (which the forward syncs to host) and dispatches per-expert
  // backward calls on contiguous slices.  Returns the input gradient
  // [rows, dim] in the same device as dy.
  Tensor backward_moe_gpu_batched(const Tensor &dy, const Tensor &x);
  bool is_attention, is_moe, is_ttt;
  int layer_idx, total_layers, d_model, num_experts;
  float dropout_rate_;
  bool training_mode_ = true;
  int inference_top_k_override_ = 0;  // Pacote A.1; 0 = no override
  int last_batch_size_ = 0;
  Tensor saved_input_;
  Tensor saved_core_norm_;
  Tensor saved_residual_;
  Tensor saved_ff_norm_;
  Tensor saved_ff_hidden_pre_;
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
  // AUDIT #4+#5: cache of GPU routing outputs.  Forward populates
  // these from the device buffers (one small D2H copy each).  Backward
  // reuses them instead of re-launching the count/scan/assignment
  // kernels and re-syncing host arrays.  Both are host-side, so they
  // outlive the GpuDeviceBuffer scope from forward_moe_gpu_batched.
  //
  // saved_moe_permutation_host_[k] is the source row id at slot k
  //   (slots are ordered by expert; experts in [0, num_experts)).
  // saved_moe_offsets_host_[e]   is the start slot for expert e
  // saved_moe_offsets_host_[e+1] is the end slot for expert e
  // saved_moe_counts_host_[e]    is offsets[e+1] - offsets[e]
  std::vector<int> saved_moe_permutation_host_;
  std::vector<int> saved_moe_offsets_host_;
  std::vector<int> saved_moe_counts_host_;
  int saved_moe_n_active_ = 0;
  LayerAuditCollector* audit_collector_ = nullptr;
public:
  std::unique_ptr<Attention> attn_layer;
  std::unique_ptr<Mamba2SSD> mamba_layer;
  std::unique_ptr<TTTLayer> ttt_layer;
  std::unique_ptr<MoERouter> router;
  std::vector<std::unique_ptr<BitLinear>> expert_gate_up, expert_down;
  std::unique_ptr<BitLinear> ffn_gate_up, ffn_down;
  // ── CHRASS slot (parallel with FFN) — nullopt unless use_chrass=true ──
  std::unique_ptr<ChrassLayer> chrass_layer;
};

class JambaModel : public Module {
public:
  std::vector<std::unique_ptr<JambaBlock>> layers;
  std::unique_ptr<Embedding> embedding;
  std::unique_ptr<BitLinear> value_head;
  std::unique_ptr<MCTSReasoning> mcts;

  JambaModel(int num_layers, int d_model, int vocab_size = 128000, Device device = Device::CPU);
  JambaModel(const ModelConfig& config, Device device = Device::CPU);
  Tensor forward(const Tensor &x) override;
  Tensor forward(const Tensor &x, Context *ctx);
  void to(Device device) override;
  Tensor forward_ids(const std::vector<int> &ids, Context *ctx = nullptr);
  Tensor forward_ids_batch(const std::vector<std::vector<int>>& batch_ids,
                           Context* ctx = nullptr);
  Tensor forward_trunk(const std::vector<int> &ids, Context *ctx = nullptr);
  Tensor forward_trunk_batch(const std::vector<std::vector<int>>& batch_ids,
                             Context* ctx = nullptr);
  Tensor forward_embedding(const Tensor &x, Context *ctx = nullptr);
  Tensor reason(const Tensor &x, int num_simulations = 100);
  std::vector<Parameter *> parameters() override;
  void save(const std::string &filename);
  void load(const std::string &filename, bool strict = true);
  Tensor forward_thought(const Tensor &x, int steps);
  void run_reasoning_loop(int iterations);
  Tensor run_reasoning_loop(const Tensor &x, int iterations);
  void backward_external(const Tensor &grad, Context &ctx);
  void backward_embedding(const Tensor &grad, Context &ctx);
  void backward(const Tensor &grad, Context &ctx);
  std::vector<BitLinear*> collect_bitlinear_layers();
  void set_reference_path(bool use_reference_path);
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
  bool training_mode() const { return training_mode_; }
  JambaSessionSnapshot fork_session() const;
  std::vector<JambaSessionSnapshot> fork_session_batch() const;
  void restore_session(const JambaSessionSnapshot& snapshot);
  void restore_session_batch(const std::vector<JambaSessionSnapshot>& snapshots);
  void reset_session();
  void set_hamiltonian_mode(bool enabled);
  Tensor run_simd_inference(const Tensor &x, int steps);
  void session_adapt(const Tensor &x, const Tensor &y);
  const ModelConfig& model_config() const { return model_config_; }
  void reset_runtime_telemetry();
  RuntimeTelemetrySnapshot runtime_telemetry() const;
  void set_audit_collector(LayerAuditCollector* collector);
  LayerAuditCollector* audit_collector() const { return audit_collector_; }
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
  void attach_profiler(void* profiler) { profiler_ = profiler; }
  void set_profiler_callbacks(ProfilerBeginLayerFn begin_fn,
                              ProfilerEndLayerFn end_fn) {
      profiler_begin_layer_ = begin_fn;
      profiler_end_layer_   = end_fn;
  }
  void* profiler() const { return profiler_; }
  ProfilerBeginLayerFn profiler_begin_layer_fn() const { return profiler_begin_layer_; }
  ProfilerEndLayerFn   profiler_end_layer_fn()   const { return profiler_end_layer_; }
  void record_audit_token_context(const std::vector<int>& token_ids_sample,
                                  size_t batch_size,
                                  size_t prompt_tokens_total,
                                  size_t prompt_tokens_used,
                                  int context_limit,
                                  bool truncated);
private:
  int num_layers, d_model, vocab_size;
  Device device;
  ModelConfig model_config_;
  bool streaming_inference_enabled_ = false;
  bool training_mode_ = true;
  std::vector<int> last_input_ids_;
  std::vector<std::vector<int>> last_input_batches_;
  std::vector<int> last_input_batch_lengths_;
  Tensor saved_final_hidden_;
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
