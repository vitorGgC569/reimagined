#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void launch_add_kernel(float *out, const float *a, const float *b, int n);
void launch_sub_kernel(float *out, const float *a, const float *b, int n);
void launch_mul_scalar_kernel(float *out, const float *a, float scalar, int n);
void launch_mul_tensor_kernel(float *out, const float *a, const float *b,
                              int n);
void launch_sigmoid_kernel(float *out, const float *in, int n);
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

// BitNet GEMM
void launch_bitnet_gemm(const int8_t *A, const uint32_t *W, float *C, int M,
                        int K, int N, float scale, int grid_x, int grid_y,
                        int block_dim);

// Standard Math Kernels
void launch_rmsnorm_kernel(float *out, const float *in, int n_rows, int n_cols,
                           int stride_unused, int block_dim);
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
                                    const float *x_norm, int outer,
                                    int inner);
void launch_adamw_update_kernel(float *weights, const float *grad, float *m,
                                float *v, int n, float beta1, float beta2,
                                float bc1, float bc2, float lr, float eps,
                                float weight_decay, int apply_weight_decay);

// New Phase 5 Kernels
void launch_relu_kernel(float *out, const float *in, int n);

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
void launch_kaiming_uniform_kernel(float *out, int n, float limit,
                                   unsigned long long seed);
void launch_check_stability_kernel(int *d_found_issue, const float *in,
                                   float max_val, int n);
void launch_moe_topk_kernel(const float *logits, float *weights, float *indices,
                            int batch, int num_experts, int k);

// =====================================================================
// Generic top-k mask + renormalization for MoE routing.
//
// In-place transformation of `weights[batch * num_experts]`:
//   * For each row, find the top-`k` values.
//   * Set every non-top-k entry to 0.
//   * Renormalize the surviving values so they sum to 1 (within the row).
//
// Matches the CPU implementation in jamba.cpp::MoERouter::forward.  k must
// satisfy 1 <= k <= num_experts.  num_experts is bounded by
// MOE_TOPK_MASK_MAX_EXPERTS (compile-time constant in mamba_kernels.cu /
// moe_kernels.cu — currently 64).  Single launch, no internal sync.
// =====================================================================
void launch_moe_topk_mask_kernel(float *weights, int batch, int num_experts,
                                 int k);

// Accumulates per-expert load (sum of router weights across the batch)
// directly on the device.  `expert_loads[num_experts]` must be
// pre-zeroed by the caller (cudaMemsetAsync recommended) — kernel uses
// atomicAdd for thread-safe accumulation.  No internal sync.
void launch_moe_load_accumulate_kernel(const float *weights,
                                       float *expert_loads, int batch,
                                       int num_experts);

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

// HPC Fused Cross-Entropy: softmax + log + NLL in single kernel
void launch_fused_cross_entropy(float *d_loss, float *grad, const float *logits,
                                const int *target, int batch, int vocab);

// Broadcast kernels
void launch_add_broadcast_kernel(float *out, const float *a, const float *b,
                                 int n, int stride);
void launch_mul_broadcast_kernel(float *out, const float *a, const float *b,
                                 int n, int D);

// Slice, Mean, Softmax
void launch_slice_kernel_dim2(float *out, const float *in, int d0, int d1,
                              int d2, int start, int end);
void launch_mean_kernel(float *out, const float *in, int outer, int reduce,
                        int inner);
void launch_softmax_kernel(float *out, const float *in, int outer, int inner);
void launch_gqa_causal_attention_kernel(const float *q_flat, const float *kv_flat,
                                        float *out, int seq_len, int d_model,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int kv_group_size,
                                        float theta);
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
                                                float theta);
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
                                               float theta);

#ifdef __cplusplus
}
#endif

#endif
