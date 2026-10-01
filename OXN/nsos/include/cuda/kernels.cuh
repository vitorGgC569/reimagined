#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <cstddef>
#include "gpu_linear_view.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NsosTensorAuditDeviceStats {
  double min_value;
  double max_value;
  double sum;
  double sum_sq;
  double max_abs;
  uint64_t finite_count;
  uint64_t nan_count;
  uint64_t inf_count;
  uint64_t zero_count;
  uint64_t subnormal_count;
  uint64_t positive_count;
  uint64_t negative_count;
} NsosTensorAuditDeviceStats;

void launch_grouped_decode_projections(const nsos::GpuLinearView* views,
    const float* input, float* prepared, float* scales, float* output,
    int groups, int output_stride);

void launch_gqa_append_decode_batch(const float* query, const float* kv,
    void* keys, void* values, float* output, int batch, int capacity, int position,
    int heads, int kv_heads, int dim, int group, float theta, int window, int fp16);
void launch_kv_half_convert(const void* source, void* target, size_t count, int to_half);

typedef struct NsosHybridAuditDeviceMetrics {
  // 0..5: Mamba/Attention/FFN signals then contributions.
  // 6: sum of all three contributions.
  NsosTensorAuditDeviceStats tensor_stats[7];
  // Signal MA/MF/AF followed by contribution MA/MF/AF dot products.
  double pair_dots[6];
} NsosHybridAuditDeviceMetrics;

// Immutable launch metadata shared by deterministic multi-tensor schedulers.
// Every descriptor covers one non-empty, in-bounds interval of one tensor.
// The fixed-width 16-byte ABI is shared by C++ host and HIP/CUDA device code.
typedef struct NsosMultiTensorChunk {
  uint64_t element_offset;
  uint32_t element_count;
  uint32_t tensor_index;
} NsosMultiTensorChunk;

// Optional device contribution contract. A null table preserves dense legacy
// behavior; a null entry is dense, otherwise the pointed byte is authoritative.
// abort_issue is a stable, caller-owned pre-update gate (zero means valid).
// Predicate/table/storage must outlive every queued consumer on its stream.
typedef struct NsosOptimizerContribution {
  const unsigned char* const* predicates;  // device table [n_tensors]
  const int* abort_issue;                 // nullable device scalar
} NsosOptimizerContribution;

// Sidecar descriptor; does not change the legacy chunk or launch ABIs.
// Inactive tensor entries may be null and must never be dereferenced. Active
// entries must have valid caller-owned weight/gradient/moment storage. No
// allocation, version publication, or cache invalidation is implied here.
typedef struct NsosActivityAwareOptimizerDesc {
  float* const* w;
  float* const* g;
  float* const* m;
  float* const* v;
  const unsigned long long* offsets;
  const unsigned char* wd_flags;
  const float* learning_rates;
  int n_tensors;
  unsigned long long total;
  NsosOptimizerContribution contribution;
} NsosActivityAwareOptimizerDesc;

bool launch_chunked_norm_device_clip(double* total, double* partials,
    float* coefficient, float* const* gradients,
    const NsosMultiTensorChunk* chunks, int chunk_count, float max_norm,
    const int* deferred_finite_issue);

bool launch_ttt_device_forward(const float* input, const float* keys,
    const float* base, float* adaptation, float* momentum, float* history,
    float* errors, float* output, float* scale, int rows, int hidden, int dim,
    float decay, float step, float max_norm, bool hamiltonian,
    int history_chunk = 0, float* momentum_history = nullptr);
bool launch_ttt_full_backward(const float* keys, const float* errors,
    const float* boundaries, const float* momentum_boundaries, const float* grad,
    float* grad_keys, float* grad_base, float* grad_direct, float* adaptation,
    float* momentum, float* local_history, float* adj_a, float* adj_m,
    float* q, double* coefficients, int rows, int hidden, int dim, int chunk,
    float decay, float step, float max_norm, bool hamiltonian);
bool launch_ttt_device_key_backward(const float* grad, const float* history,
    float* grad_keys, int rows, int hidden, int dim);

bool launch_attn_tiled_forward(const float* q, const float* k, const float* v,
    const int* valid, float* out, float* lse, int B, int S, int H, int KV,
    int hd, int group, int window, float scale);
bool launch_attn_tiled_backward(const float* q, const float* k, const float* v,
    const float* out, const float* grad, const float* lse, const int* valid,
    float* delta, float* dq, float* dk, float* dv, int B, int S, int H,
    int KV, int hd, int group, int window, float scale);

void launch_moe_ordered_assign(const float* weights, const int* offsets,
    int* permutation, float* scale, int* inverse, int rows, int experts);
void launch_moe_ordered_combine(const float* values, const int* inverse,
    const float* scales, float* output, int rows, int dim, int experts);

// Compact device-side audit reductions. Only the fixed-size output structs
// cross D2H; full activations and gradients remain device-resident.
void launch_tensor_audit_stats_kernel(
    NsosTensorAuditDeviceStats *out, const float *values,
    long long count);
void launch_hybrid_audit_metrics_kernel(
    NsosHybridAuditDeviceMetrics *out,
    const float *mamba_signal, const float *attention_signal,
    const float *ffn_signal, const float *mamba_contribution,
    const float *attention_contribution,
    const float *ffn_contribution, long long count);

void launch_add_kernel(float *out, const float *a, const float *b, int n);
void launch_sub_kernel(float *out, const float *a, const float *b, int n);
void launch_mul_scalar_kernel(float *out, const float *a, float scalar, int n);
void launch_mul_tensor_kernel(float *out, const float *a, const float *b,
                              int n);
void launch_sigmoid_kernel(float *out, const float *in, int n);
void launch_silu_kernel(float *out, const float *in, int n);
void launch_silu_backward_kernel(float *in_grad, const float *grad_out,
                                 const float *pre_activation, int n);
bool launch_silu_gate_forward_kernel(float *out, const float *value,
                                     const float *gate, int n);
bool launch_silu_gate_backward_kernel(float *value_grad, float *gate_grad,
                                      const float *grad_out,
                                      const float *value,
                                      const float *gate, int n);
void launch_scale_inplace_kernel(float *data, float scale, int n);
void launch_mul_vector_broadcast_kernel(float *out, const float *in,
                                        const float *vec, int rows, int cols);
void launch_embedding_gather_kernel(float *out, const float *weight,
                                    const int *ids, int total_positions,
                                    int vocab_size, int embedding_dim);
void launch_embedding_scatter_add_kernel(float *grad_weight,
                                         const float *grad_output,
                                         const int *ids,
                                         int total_positions,
                                         int vocab_size,
                                         int embedding_dim);
void launch_embedding_scatter_add_deterministic_kernel(
    float *grad_weight, const float *grad_output, const int *ids,
    int total_positions, int vocab_size, int embedding_dim);
// Ordered sparse gather for deterministic embedding gradients. unique_ids is
// [U], offsets is [U+1], and positions is grouped by token with each group in
// strictly ascending original-position order. Absent vocabulary rows remain
// untouched (the caller zero-initializes the dense destination).
bool launch_embedding_scatter_add_deterministic_sparse_kernel(
    float *grad_weight, const float *grad_output,
    const int *unique_ids, const int *offsets, const int *positions,
    int unique_count, int total_positions, int vocab_size,
    int embedding_dim);

// BitNet GEMM
void launch_bitnet_gemm(const int8_t *A, const uint32_t *W, float *C, int M,
                        int K, int N, float scale, int grid_x, int grid_y,
                        int block_dim);
// One native wave per output, with contiguous K reads. Fuses the activation
// scale and optional output affine without materializing a pre-output tensor.
void launch_bitnet_gemv_scaled(const int8_t *A, const uint32_t *W, float *C,
                              int K, int N, float weight_scale,
                              const float *activation_scale,
                              const float *magnitude, const float *bias);

// Standard Math Kernels
void launch_rmsnorm_kernel(float *out, const float *in, int n_rows, int n_cols,
                           float eps);
void launch_layernorm_kernel(float *out, const float *in, int n_rows,
                             int n_cols);
void launch_matmul_kernel(const float *A, const float *B, float *C, int M,
                          int K, int N, int grid_x, int grid_y, int block_dim);
void launch_transpose2d_kernel(float *out, const float *in, int rows,
                               int cols);
void launch_cross_entropy_kernel(float *d_loss, float *grad,
                                 const float *logits, const int *target,
                                 int batch, int vocab, int grid_x,
                                 int block_dim);
void launch_rmsnorm_backward_kernel(float *dx, const float *grad,
                                    const float *x_norm, const float *x,
                                    int outer, int inner, float eps);
void launch_adamw_update_kernel(float *weights, const float *grad, float *m,
                                float *v, int n, float beta1, float beta2,
                                float bc1, float bc2, float lr, float eps,
                                float weight_decay, int apply_weight_decay,
                                int *found_nonfinite);

// New Phase 5 Kernels
void launch_relu_kernel(float *out, const float *in, int n);

// AUDIT #6 + LEARN B3 (2026-05-16): cast FP32 -> low-precision in
// place for mixed-precision matmul.  mode=1 -> BF16, mode=2 -> FP16.
// Output buffer must be `n * 2` bytes pre-allocated by the caller.
// Used by Tensor::matmul to prepare A/B for cublasGemmStridedBatchedEx
// Tensor Cores path.  See the rationale in tensor.cpp::matmul about
// why BF16 is the default (8-bit exponent matches FP32 range,
// avoids BitNet activation overflow that FP16's 5-bit exponent
// can't handle).
void launch_cast_f32_to_lowp_kernel(void *out, const float *in, size_t n, int mode);
// Fused FP32 -> lowp conversion and row-major 2D transpose. The destination
// contains [cols, rows] without an intermediate FP32 transpose. This is the
// HIP-safe operand preparation for lowp GEMMs whose rocBLAS OP_T path is not
// executable on every validated RDNA code-object bundle.
void launch_cast_transpose_f32_to_lowp_kernel(void *out, const float *in,
                                               int rows, int cols, int mode);

// LEARN S1 (BitNet b1.58 2B4T 2026): Squared ReLU activation.
//   forward:  out[i] = max(0, in[i])^2
//   backward: in_grad[i] = grad_out[i] * 2 * max(0, in[i])
// Squared ReLU is the activation BitNet b1.58 ships with because SwiGLU
// in low-precision (ternary, FP8) suffers occasional activation spikes
// that overflow the dynamic range and diverge loss after extended
// training.  Squared ReLU has comparable expressive power to SwiGLU at
// our scale (40M-80M) while remaining numerically stable in quantized
// regimes.  Both forward and backward are pure elementwise so we ship
// dedicated kernels — fusing forward+grad with the BitLinear matmul
// would be the next-level fusion, deferred to a future pass.
void launch_squared_relu_kernel(float *out, const float *in, int n);
void launch_squared_relu_backward_kernel(float *in_grad, const float *grad_out,
                                          const float *pre_activation, int n);

void launch_clamp_kernel(float *out, const float *in, float min_val,
                         float max_val, int n);
void launch_norm_kernel(float *d_sum_sq, const float *in, int n);
// Fixed-order global norm for deterministic training. One block reduces each
// tensor in registry order, then one thread reduces those partials in that same
// order. No atomics and only the final double scalar crosses to the host.
bool launch_multi_tensor_sqsum_deterministic(
    double *total, double *partials, float *const *gradients,
    const unsigned long long *sizes, int tensor_count,
    const int *deferred_finite_issue);
// Repetition-unlikelihood gradient adjustment fully on GPU (mirrors the host
// loop in trainer.cpp::apply_repetition_unlikelihood).  grad/probs are
// [rows, vocab] device pointers; answer_tokens is a [rows] device int buffer.
void launch_repetition_unlikelihood_kernel(
    float *d_loss, float *grad, const float *probs,
    const int *answer_tokens, int rows, int vocab, float scale,
    int eos_token_id);
// Heterogeneous batched variant. token_plane is [batch, seq] with -1 for
// prompt/padding rows; history windows never cross batch boundaries.
bool launch_repetition_unlikelihood_masked_kernel(
    float *d_loss, float *grad, const float *probs,
    const int *token_plane, int batch, int seq, int vocab, float scale,
    int eos_token_id);
// Adds beta * row_weight[row] / vocab * logits to grad and accumulates
// 0.5 * beta * row_weight[row] * mean(logits_row^2) into d_loss.
bool launch_masked_logit_l2_kernel(
    float *d_loss, float *grad, const float *logits,
    const float *row_weights, int rows, int vocab, float beta);

// ── GQA attention TRAINING path (src/cuda/attention_train_kernels.cu) ──────
// Glue kernels keeping Attention forward-saves and the exact-cache backward
// fully device-resident (the GEMMs run through Tensor::matmul's batched
// cuBLAS path).  Layouts identical to jamba.cpp:
//   q/dO heads [B,S,H,hd] · k/v heads [B,S,KV,hd] · permuted [B,H,S,hd]
//   transposed [B,H,hd,S] · scores/P/dS [B*H,S,S]
void launch_attn_gather_heads(float *out, const float *src, int B, int S,
                              int H_out, int hd, int src_heads, int group,
                              int transposed);
void launch_attn_unpermute_heads(float *out, const float *src, int B, int H,
                                 int S, int hd);
void launch_attn_reduce_group(float *out, const float *src, int B, int S,
                              int KV, int hd, int H, int group);
void launch_attn_masked_softmax(float *p, const int *valid, int B, int H,
                                int S, float scale, int sliding_window);
void launch_attn_softmax_backward(float *ds, const float *p, const float *dp,
                                  int B, int H, int S, float scale);
void launch_batched_transpose_last2(float *out, const float *src, int N,
                                    int R, int C);
void launch_rope_apply(float *x, const float *cos_buf, const float *sin_buf,
                       int B, int S, int H, int hd, int start_pos, int max_seq,
                       int dir);
void launch_kv_split(float *k, float *v, const float *kv, long long rows,
                     int kvd);
void launch_kv_concat(float *out, const float *k, const float *v,
                      long long rows, int kvd);

// ── Otimizador multi-tensor (src/cuda/fused_optimizer_kernels.cu) ──────────
// O passo AdamW inteiro (scale de acumulação + clip + update) em 2 kernels
// para TODOS os parâmetros, com o scale e o coeficiente de clip dobrados em
// gscale.  Arrays de ponteiros/offsets/flags residem em buffer device único.
void launch_multi_tensor_sqsum(float *accum, float *const *w, float *const *g,
                               float *const *m, float *const *v,
                               const unsigned long long *offsets,
                               const unsigned char *wd_flags, int n_tensors,
                               unsigned long long total);
void launch_add_row_broadcast(float *grad, const float *row_add, int rows,
                              int cols);
void launch_multi_tensor_adamw(float *const *w, float *const *g,
                               float *const *m, float *const *v,
                               const unsigned long long *offsets,
                               const unsigned char *wd_flags,
                               const float *lr_scales, int n_tensors,
                               unsigned long long total,
                               const float *gradient_sq_sum,
                               float accumulation_scale,
                               float max_grad_norm,
                               float beta1, float beta2, float bc1, float bc2,
                               float lr, float eps, float weight_decay,
                               int *found_nonfinite);
// Update-only deterministic lane. Gradients have already been scaled and
// clipped by the canonical ordered path. Every element follows the exact
// scalar operation order of launch_adamw_update_kernel; tensor scheduling is
// fused without introducing a reduction or atomic arithmetic dependency.
bool launch_multi_tensor_adamw_update_deterministic(
    float *const *w, float *const *g, float *const *m, float *const *v,
    const unsigned long long *offsets, const unsigned char *wd_flags,
    const float *learning_rates, int n_tensors, unsigned long long total,
    const NsosMultiTensorChunk *chunks, int chunk_count,
    float beta1, float beta2, float bc1, float bc2, float eps,
    float weight_decay, int *found_nonfinite);

// OXTA-CRIT: one fixed-order block per tensor computes absmean and the exact
// ternary branch gain without copying full weights to the host.  The companion
// kernel adds a pre-Adam radial regularizer to the already accumulated grads.
bool launch_multi_tensor_criticality_metrics(
    float *const *w, const unsigned long long *offsets, const int *fan_in,
    int n_tensors, float *gammas, float *gains);
bool launch_multi_tensor_criticality_grad(
    float *const *w, float *const *g,
    const unsigned long long *offsets, const float *coefficients,
    int n_tensors, unsigned long long total);
bool launch_multi_tensor_check_finite(
    int *found_issue, const float *const *values,
    const unsigned long long *offsets, int n_tensors,
    unsigned long long total,
    const unsigned char *require_nonnegative,
    const NsosMultiTensorChunk *chunks, int chunk_count);
bool launch_multi_tensor_zero(
    float *const *values, const unsigned long long *offsets,
    int n_tensors, unsigned long long total);

void launch_check_stability_kernel(int *d_found_issue, const float *in,
                                   float max_val, int n);

// Decode-time greedy token selection on-device (#2 GPU sampler, greedy path).
// argmax over allowed tokens with repetition penalty, mirroring the host greedy
// branch in nsos_sdk.cpp.  repeated/seen/control are uint8[vocab] device masks
// (any may be null); out_token is a device int.  Removes the per-token [vocab]
// D2H + host vocab scan.  See src/cuda/kernels.cu for the exact value rule.
void launch_decode_greedy_argmax(const float *raw, int vocab,
                                 const unsigned char *repeated,
                                 const unsigned char *seen,
                                 const unsigned char *control,
                                 int suppress_control, float penalty,
                                 int *out_token);

// =====================================================================
// Generic top-k mask + renormalization for MoE routing.
//
// In-place transformation of `weights[batch * num_experts]`:
//   * For each row, find the top-`k` values.
//   * Set every non-top-k entry to 0.
//   * Renormalize the surviving values so they sum to 1 (within the row).
//
// Matches the CPU implementation in jamba.cpp::MoERouter::forward.  k must
// satisfy 1 <= k <= num_experts.  The implementation has no fixed-size
// per-thread expert array, so expert counts above 64 are safe.  Single launch,
// no internal sync.
// =====================================================================
void launch_moe_topk_mask_kernel(float *weights, int batch, int num_experts,
                                 int k);

// Dense single-row decode accumulation: out[i] += (*scale_dev) * y[i] with
// the scale read from DEVICE memory at execution time (no host copy of the
// routing weight -> no per-token D2H; CUDA-graph capturable).
void launch_moe_scale_accum_row_kernel(float *out, const float *y,
                                       const float *scale_dev, int n);
// Two banks of E descriptors (up, down). Inactive experts perform no weight
// reads or dot products. All routing stays on device; final sum is ordered.
void launch_moe_sparse_decode(const nsos::GpuLinearView* views,
                             const float* routing, const float* input,
                             float* prepared, float* scales, float* hidden,
                             float* contributions, float* output,
    int experts, int dim, int hidden_dim, int rows);

// Accumulates per-expert load (sum of router weights across the batch)
// directly on the device.  `expert_loads[num_experts]` must be
// pre-zeroed by the caller (cudaMemsetAsync recommended) — kernel uses
// atomicAdd for thread-safe accumulation.  No internal sync.
void launch_moe_load_accumulate_kernel(const float *weights,
                                       float *expert_loads, int batch,
                                       int num_experts);
void launch_moe_load_ordered_kernel(const float *weights, float *expert_loads,
                                    int batch, int num_experts);
void launch_moe_zero_invalid_rows_kernel(float *weights,
                                         const uint8_t *valid_rows,
                                         int batch, int num_experts);
// Device-resident Switch auxiliary objective. `counts` and `prob_sums` are
// pre-zeroed [num_experts]; `loss` is pre-zeroed [1]. top_k <= 64.
void launch_moe_switch_aux_stats_kernel(const float *probs, float *counts,
                                        float *prob_sums, int rows,
                                        int num_experts, int top_k,
                                        bool deterministic);
void launch_moe_switch_aux_grad_kernel(const float *probs,
                                       const float *counts, float *grad,
                                       int rows, int num_experts, float coef);
void launch_moe_switch_aux_loss_kernel(const float *counts,
                                       const float *prob_sums, float *loss,
                                       int rows, int num_experts, float coef,
                                       bool deterministic);

// =====================================================================
// Batched MoE dispatch primitives (Phase 4-extended).
//
// These kernels implement the GPU-resident permute-gather-execute-
// scatter pipeline that replaces the per-expert std::memcpy loops in
// JambaBlock::forward_moe.  Recipe:
//   1. count[e]   = number of (row, expert) pairs with mask != 0
//   2. offset[e]  = exclusive_scan(count)
//   3. permutation[k] = source row index, packed by expert
//      assignment[k]  = expert id for slot k
//      scale[k]       = router weight for slot k
//   4. permuted_input[k, :] = x[permutation[k], :]   (contiguous by expert)
//   5. expert e processes permuted_input[offset[e] : offset[e]+count[e], :]
//   6. y[row, :] += scale[k] * permuted_output[k, :]   for every active k
//
// All kernels require pre-allocated, properly-sized output buffers and
// do NOT call cudaDeviceSynchronize internally.  Callers serialize via
// the default stream or explicit sync.
// =====================================================================

// Counts the number of nonzero weights per expert column.  counts[E]
// must be zeroed before launch (uses atomicAdd).  weights layout is
// [batch, num_experts] post-mask (zeros for non-top-k entries).
void launch_moe_count_per_expert_kernel(const float *weights, int *counts,
                                         int batch, int num_experts);

// Exclusive scan over a small array (num_experts ≤ 1024).  Single block,
// shared-memory Hillis-Steele scan.  offsets[num_experts+1] is filled
// such that offsets[0]=0 and offsets[num_experts] = total active slots.
void launch_moe_exclusive_scan_small_kernel(const int *counts, int *offsets,
                                             int num_experts);

// Computes the permutation, expert assignments and scales given the
// post-mask weights and per-expert offsets.  permutation[N_active],
// assignment[N_active], scale[N_active] are output.  N_active equals
// offsets[num_experts] from the previous step.  Internally uses
// atomicAdd against a small workspace counters[num_experts] (the
// caller passes a zeroed buffer of size num_experts).
void launch_moe_compute_assignments_kernel(const float *weights,
                                            const int *offsets,
                                            int *workspace_counters,
                                            int *permutation,
                                            int *assignment, float *scale,
                                            int batch, int num_experts);

// Gathers rows from input[batch, dim] into permuted[N_active, dim] in
// the order defined by permutation[N_active].
void launch_moe_gather_rows_kernel(const float *input, const int *permutation,
                                    float *permuted, int N_active, int dim);

// Scatter-add the per-slot scaled output back into y[batch, dim].
// y MUST be pre-zeroed by the caller (cudaMemsetAsync recommended); the
// kernel uses atomicAdd to accumulate over slots that share a row.
void launch_moe_scatter_add_weighted_kernel(const float *permuted_output,
                                             const int *permutation,
                                             const float *scale, float *y,
                                             int N_active, int dim);
void launch_moe_scale_rows_kernel(const float *input, const float *scale,
                                  float *output, int rows, int dim);
void launch_moe_router_weight_grad_kernel(
    const float *dy, const float *unscaled_expert_output,
    const int *permutation, const int *offsets, float *grad_weights,
    int n_active, int dim, int num_experts);
void launch_moe_router_logits_grad_kernel(
    const float *probs, const float *grad_weights, float *grad_logits,
    int rows, int num_experts, int top_k);

// =====================================================================
// BitNet 1.58-bit GPU dispatch primitives (Phase 5a of the GPU plan).
//
// Per-row activation quantization mirroring
// BitLinear::quantize_activations_bitnet on CPU:
//   * x  : float [M, K]    (input activations, device)
//   * x_q: int8  [M, K]    (output, device, must be allocated)
//   * act_scales: float [M] (output, device, must be allocated)
//   * precision_bits selects q_max (2 → ternary, 8 → INT8, etc.)
// No internal sync.  See src/cuda/bitnet_kernels.cu for details.
// =====================================================================
void launch_quantize_activations_bitnet_kernel(const float *x, int8_t *x_q,
                                               float *act_scales, int M, int K,
                                               int precision_bits);

// In-place per-row scaling: y[row, col] *= act_scales[row].
// Used to fold per-row activation scale into the output of
// `launch_bitnet_gemm` (which only carries the global weight_scale).
void launch_bitnet_apply_act_scales_kernel(float *y, const float *act_scales,
                                           int M, int N);

// Σ|x| into a device scalar (caller pre-zeros).  Absmean weight-scale support.
void launch_abs_sum_kernel(float *d_abs_sum, const float *in, int n);

// ── K3: GPU QAT fake-quant (straight-through) primitives ────────────────────
// Weight fake-quant: out[i] = clamp(round(w[i]/scale), -1, +1) * scale
// (the canonical NSOS ternary rule; identical numerics to
// BitLinear::quantize_weights * weight_scale).
void launch_fake_quant_ternary_kernel(float *out, const float *w, float scale,
                                      int n);
// Same ternary fake-quant, but `scale_out[0] = abs_sum[0] / n + 1e-8` is
// computed/read on device.  Avoids per-layer D2H sync during GPU QAT.
void launch_fake_quant_ternary_absmean_kernel(float *out, float *scale_out,
                                              const float *w,
                                              const float *abs_sum, int n);
// Per-row activation fake-quant (quant→dequant in one pass), returns the
// DEQUANTIZED float activations the matmul should use:
//   s_row = (max_j|x|+1e-8)/q_max ; out[r,j] = clip(round(x/s_row),±q_max)*s_row
// q_max from precision_bits (2 → ±1, else 2^(b-1)-1).  Matches
// BitLinear::quantize_activations_bitnet + dequant.
void launch_fake_quant_activations_kernel(float *out, const float *x, int M,
                                          int K, int precision_bits);
// STE clip: zero grad rows whose latent weight already saturated past the
// ternary band:  dW[i] = (|w[i]/scale| > 1) ? 0 : dW[i].
void launch_ste_clip_weight_grad_kernel(float *dW, const float *w, float scale,
                                        int n);
// Same STE clip, with the scale read from device memory.
void launch_ste_clip_weight_grad_device_scale_kernel(float *dW, const float *w,
                                                     const float *scale,
                                                     int n);

// HPC Fused Cross-Entropy: softmax + log + NLL in single kernel
void launch_fused_cross_entropy(float *d_loss, float *grad, const float *logits,
                                const int *target, int batch, int vocab);
void launch_fused_cross_entropy_weighted(
    float *d_loss, float *grad, const float *logits, const int *target,
    const float *row_weights, int batch, int vocab);
bool launch_fused_cross_entropy_deterministic(
    float *d_loss, float *row_losses, float *grad, const float *logits,
    const int *target, const float *row_weights, int batch, int vocab);

// Broadcast kernels
void launch_add_broadcast_kernel(float *out, const float *a, const float *b,
                                 int n, int stride);
bool launch_add_trailing_broadcast_kernel(float *out, const float *a,
                                          const float *b, int n, int width);
bool launch_sub_broadcast_kernel(float *out, const float *a, const float *b,
                                 int n, int stride, int reverse);
bool launch_sub_trailing_broadcast_kernel(float *out, const float *a,
                                          const float *b, int n, int width,
                                          int reverse);
void launch_mul_broadcast_kernel(float *out, const float *a, const float *b,
                                 int n, int D);

// Contiguous tensor slice for any rank/axis. Layout is represented as
// [outer, axis, inner], and the result copies [start,end) of axis.
bool launch_slice_contiguous_kernel(float *out, const float *in, int outer,
                                    int axis_size, int inner, int start,
                                    int slice_size, int total);
void launch_mean_kernel(float *out, const float *in, int outer, int reduce,
                        int inner);
void launch_softmax_kernel(float *out, const float *in, int outer, int inner);
void launch_gqa_causal_attention_kernel(const float *q_flat, const float *kv_flat,
                                        float *out, int seq_len, int d_model,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int kv_group_size,
                                        float theta, int sliding_window);
void launch_batched_gqa_causal_attention_kernel(const float *q_flat,
                                                const float *kv_flat,
                                                float *out,
                                                int batch_size,
                                                int seq_len,
                                                int d_model,
                                                int n_heads,
                                                int n_kv_heads,
                                                int head_dim,
                                                int kv_group_size,
                                                float theta,
                                                int sliding_window);
void launch_gqa_append_kv_cache_kernel(const float *kv_flat, float *key_cache,
                                       float *value_cache, int cache_row,
                                       int n_kv_heads, int head_dim,
                                       float theta);
void launch_gqa_cached_attention_decode_kernel(const float *q_flat,
                                               const float *key_cache,
                                               const float *value_cache,
                                               float *out, int cached_tokens,
                                               int d_model, int n_heads,
                                               int n_kv_heads, int head_dim,
                                               int kv_group_size,
                                               float theta,
                                               int sliding_window);

// ── CUDA Graphs (opt-in NSOS_CUDA_GRAPH) ─────────────────────────────────────
// Decode-step launch-overhead amortization: capture the per-token kernel sequence
// once, then replay the executable graph each token.  Graph-capture validity is a
// GPU-runtime property (no host build can establish it), so the mechanism ships
// with a runtime self-test:
//   cuda_graphs_supported() -> device/driver advertises graph support (nonzero).
//   cuda_graph_self_test()  -> captures a kernel sequence into a graph, replays
//                              it, and verifies graphed result == eager result;
//                              returns nonzero on success, 0 if graphs are
//                              unavailable/broken.  Run once on the target GPU
//                              before relying on graphed decode.
int cuda_graphs_supported(void);
int cuda_graph_self_test(void);

// D2H micro-benchmark (per-token copy study): times `iters` 4-byte
// device-to-host copies through a PAGEABLE staging int vs a PINNED
// (cudaMallocHost) staging int and writes the average microseconds per copy.
// Returns nonzero on success, 0 on any CUDA error.
int nsos_bench_d2h_copy(int iters, double *pageable_us, double *pinned_us);

#ifdef __cplusplus
}
#endif

#endif
