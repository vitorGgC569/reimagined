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
void launch_clamp_kernel(float *out, const float *in, float min_val,
                         float max_val, int n);
void launch_norm_kernel(float *d_sum_sq, const float *in, int n);
void launch_kaiming_uniform_kernel(float *out, int n, float limit,
                                   unsigned long long seed);
void launch_check_stability_kernel(int *d_found_issue, const float *in,
                                   float max_val, int n);
void launch_moe_topk_kernel(const float *logits, float *weights, float *indices,
                            int batch, int num_experts, int k);

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
