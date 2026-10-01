// =====================================================================
// MoE routing kernels (top-k mask, load accumulation).
//
// Replaces the GPU→CPU→GPU round-trip that MoERouter::forward used to
// pay for every layer × every step.  See jamba.cpp::MoERouter::forward
// for the CPU implementation these kernels mirror token-for-token:
//   * argmax-based top-k mask in [0, num_experts) per row
//   * zero-out non-top-k entries
//   * renormalize the surviving entries so each row sums to 1
//   * accumulate per-expert "load" (sum of post-mask weights).
//
// Target: NVIDIA GTX 1050 Ti (sm_61) and newer.  No __dp4a / tensor-core
// dependencies here — the MoE routing tensor is fp32 and small.
// =====================================================================

#include "cuda/kernels.cuh"

#include <cstdio>
#include "gpu_backend.h"
#if defined(NSOS_GPU_BACKEND_CUDA)
#include <device_launch_parameters.h>
#endif

// =====================================================================
// Top-k mask + renormalize, one CUDA thread per row.
//
// Strategy: selection is performed in-place.  A selected non-negative
// softmax weight v is temporarily marked as -v-1, which cannot collide
// with an unselected probability.  This keeps O(num_experts*k) behavior
// without a fixed-size local array.
// =====================================================================
__global__ void moe_topk_mask_kernel(float *__restrict__ weights, int batch,
                                     int num_experts, int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch) return;
  if (num_experts <= 0 || k <= 0) return;

  float *row_ptr = weights + row * num_experts;

  // Defensive: clamp k to [1, num_experts].  Caller should already do
  // this but the kernel must not write OOB.
  const int eff_k =
      (k > num_experts) ? num_experts : ((k < 1) ? 1 : k);

  if (eff_k == num_experts) {
    // No masking needed; renormalize so the row sums to 1 (matches CPU
    // behavior when top_k == num_experts: the CPU path leaves weights
    // untouched, but they were already softmaxed upstream so the sum
    // is already ~1.  Renormalizing here is cheap and guards against
    // any drift in upstream kernels).
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) sum += row_ptr[e];
    const float inv = 1.0f / fmaxf(sum, 1e-9f);
    for (int e = 0; e < num_experts; ++e) row_ptr[e] *= inv;
    return;
  }

  float selected_sum = 0.0f;
  for (int rank = 0; rank < eff_k; ++rank) {
    int best_index = -1;
    float best_value = -1.0f;
    for (int e = 0; e < num_experts; ++e) {
      const float value = row_ptr[e];
      if (value >= 0.0f &&
          (best_index < 0 || value > best_value)) {
        best_index = e;
        best_value = value;
      }
    }
    if (best_index < 0) {
      break;
    }
    selected_sum += best_value;
    row_ptr[best_index] = -best_value - 1.0f;
  }

  const float inv_selected = 1.0f / fmaxf(selected_sum, 1e-9f);

  // Restore marked survivors and zero every unselected probability.
  for (int e = 0; e < num_experts; ++e) {
    const float value = row_ptr[e];
    row_ptr[e] =
        value < 0.0f ? (-value - 1.0f) * inv_selected : 0.0f;
  }
}

// =====================================================================
// Per-expert load accumulator.  expert_loads[e] += sum_row weights[row, e].
//
// One thread per (row, expert) pair, accumulating into expert_loads
// with atomicAdd.  Caller must pre-zero expert_loads.
// =====================================================================
__global__ void moe_load_accumulate_kernel(const float *__restrict__ weights,
                                           float *__restrict__ expert_loads,
                                           int batch, int num_experts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * num_experts;
  if (idx >= total) return;

  const int row = idx / num_experts;
  const int e = idx % num_experts;
  const float w = weights[row * num_experts + e];
  if (w != 0.0f) {
    atomicAdd(&expert_loads[e], w);
  }
}

__global__ void moe_load_ordered_kernel(const float* weights, float* loads,
                                       int rows, int experts) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts) return;
  float total = 0.0f;
  for (int row = 0; row < rows; ++row)
    total += weights[static_cast<size_t>(row) * experts + expert];
  loads[expert] = total;
}

__global__ void moe_zero_invalid_rows_kernel(
    float *__restrict__ weights, const uint8_t *__restrict__ valid_rows,
    int batch, int num_experts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * num_experts;
  if (idx >= total) return;
  if (valid_rows[idx / num_experts] == 0) weights[idx] = 0.0f;
}

__global__ void moe_switch_aux_stats_kernel(
    const float *__restrict__ probs, float *__restrict__ counts,
    float *__restrict__ prob_sums, int rows, int num_experts, int top_k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float *p = probs + static_cast<size_t>(row) * num_experts;
  for (int expert = 0; expert < num_experts; ++expert) {
    atomicAdd(&prob_sums[expert], p[expert]);
  }
  const int effective_top_k = min(max(top_k, 1), num_experts);
  // Stable top-k membership without a fixed-size local array.  An expert's
  // rank is the number of strictly larger probabilities plus equal-valued
  // experts with a smaller index.  This exactly defines tie-breaking and
  // works for any num_experts/top_k supported by the tensor shape.
  for (int expert = 0; expert < num_experts; ++expert) {
    int rank = 0;
    const float value = p[expert];
    for (int candidate = 0; candidate < num_experts; ++candidate) {
      const float other = p[candidate];
      if (other > value || (other == value && candidate < expert)) ++rank;
    }
    if (rank < effective_top_k) atomicAdd(&counts[expert], 1.0f);
  }
}

__global__ void moe_switch_aux_stats_deterministic_kernel(
    const float *__restrict__ probs, float *__restrict__ counts,
    float *__restrict__ prob_sums, int rows, int num_experts, int top_k) {
  // Independent expert owners preserve the original row-order additions,
  // without serializing the entire [rows,experts] matrix on one GPU thread.
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) return;
  const int effective_top_k = min(max(top_k, 1), num_experts);
  float sum = 0.0f, count = 0.0f;
  for (int row = 0; row < rows; ++row) {
    const float *p = probs + static_cast<size_t>(row) * num_experts;
    sum += p[expert];
    int rank = 0;
    const float value = p[expert];
    for (int candidate = 0; candidate < num_experts; ++candidate) {
      const float other = p[candidate];
      if (other > value || (other == value && candidate < expert)) ++rank;
    }
    if (rank < effective_top_k) count += 1.0f;
  }
  prob_sums[expert] = sum;
  counts[expert] = count;
}

__global__ void moe_switch_aux_grad_kernel(
    const float *__restrict__ probs, const float *__restrict__ counts,
    float *__restrict__ grad, int rows, int num_experts, float coef) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t base = static_cast<size_t>(row) * num_experts;
  double dot_count = 0.0;
  for (int expert = 0; expert < num_experts; ++expert) {
    dot_count += static_cast<double>(counts[expert]) * probs[base + expert];
  }
  const double scale = static_cast<double>(coef) * num_experts /
                       (static_cast<double>(rows) * rows);
  for (int expert = 0; expert < num_experts; ++expert) {
    const double probability = probs[base + expert];
    grad[base + expert] = static_cast<float>(
        scale * probability * (counts[expert] - dot_count));
  }
}

__global__ void moe_switch_aux_loss_kernel(
    const float *__restrict__ counts, const float *__restrict__ prob_sums,
    float *__restrict__ loss, int rows, int num_experts, float coef) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) return;
  const double scale = static_cast<double>(coef) * num_experts /
                       (static_cast<double>(rows) * rows);
  atomicAdd(loss, static_cast<float>(
                      scale * counts[expert] * prob_sums[expert]));
}

__global__ void moe_switch_aux_loss_deterministic_kernel(
    const float *__restrict__ counts, const float *__restrict__ prob_sums,
    float *__restrict__ loss, int rows, int num_experts, float coef) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const double scale = static_cast<double>(coef) * num_experts /
                       (static_cast<double>(rows) * rows);
  double total = 0.0;
  for (int expert = 0; expert < num_experts; ++expert) {
    total += static_cast<double>(counts[expert]) * prob_sums[expert];
  }
  *loss = static_cast<float>(scale * total);
}

// =====================================================================
// Phase 4-extended: batched MoE pipeline kernels.
// See include/cuda/kernels.cuh for the contract / recipe overview.
// =====================================================================

// Per-expert nonzero count.  Each thread reads one (row, expert) pair
// and atomicAdd's the per-expert counter when the weight is nonzero.
__global__ void moe_count_per_expert_kernel(const float *__restrict__ weights,
                                             int *__restrict__ counts,
                                             int batch, int num_experts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * num_experts;
  if (idx >= total) return;
  if (weights[idx] != 0.0f) {
    atomicAdd(&counts[idx % num_experts], 1);
  }
}

// Exclusive scan over a small array (≤ 1024 elements).  Hillis–Steele
// in shared memory; single block.  offsets[num_experts+1] receives:
//   offsets[0] = 0
//   offsets[i] = sum(counts[0..i-1])
//   offsets[num_experts] = total
__global__ void moe_exclusive_scan_small_kernel(
    const int *__restrict__ counts, int *__restrict__ offsets,
    int num_experts) {
  // Buffer sized for up to 1024 elements; cap caller-side enforced by
  // launcher.
  __shared__ int s_data[1024];
  const int tid = threadIdx.x;
  if (tid < num_experts) {
    s_data[tid] = counts[tid];
  } else if (tid < 1024) {  // s_data has 1024 slots (0..1023); `<= 1024` was OOB
    s_data[tid] = 0;
  }
  __syncthreads();

  // Up-sweep / inclusive scan on the loaded values.
  for (int offset = 1; offset < num_experts; offset *= 2) {
    int v = (tid >= offset && tid < num_experts) ? s_data[tid - offset] : 0;
    __syncthreads();
    if (tid < num_experts) {
      s_data[tid] += v;
    }
    __syncthreads();
  }

  // Convert to exclusive scan: offsets[0] = 0, offsets[i] = inclusive[i-1].
  if (tid == 0) {
    offsets[0] = 0;
  }
  if (tid < num_experts) {
    offsets[tid + 1] = s_data[tid];
  }
}

// For each (row, expert) with nonzero weight, write a slot into the
// permutation buffer.  Slot index is computed via atomicAdd against a
// per-expert workspace counter (caller pre-zeros it) plus the expert's
// offset.
__global__ void moe_compute_assignments_kernel(
    const float *__restrict__ weights, const int *__restrict__ offsets,
    int *__restrict__ workspace_counters, int *__restrict__ permutation,
    int *__restrict__ assignment, float *__restrict__ scale, int batch,
    int num_experts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * num_experts;
  if (idx >= total) return;

  const float w = weights[idx];
  if (w == 0.0f) return;

  const int row = idx / num_experts;
  const int expert = idx % num_experts;

  // Slot within this expert's segment, then add the segment's start.
  const int local_slot = atomicAdd(&workspace_counters[expert], 1);
  const int slot = offsets[expert] + local_slot;

  permutation[slot] = row;
  assignment[slot] = expert;
  scale[slot] = w;
}

// Gather rows from input[batch, dim] into permuted[N_active, dim].
// One thread per (slot, dim) element.
__global__ void moe_ordered_assign_kernel(const float* weights, const int* offsets,
    int* permutation, float* scale, int* inverse, int rows, int experts) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts) return;
  int slot = offsets[expert];
  for (int row = 0; row < rows; ++row) {
    const size_t ri = static_cast<size_t>(row) * experts + expert;
    const float weight = weights[ri];
    inverse[ri] = weight == 0.0f ? -1 : slot;
    if (weight != 0.0f) {
      permutation[slot] = row;
      scale[slot++] = weight;
    }
  }
}

__global__ void moe_ordered_combine_kernel(const float* values, const int* inverse,
    const float* scales, float* output, int rows, int dim, int experts) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(rows) * dim) return;
  const size_t row = i / dim;
  const int d = static_cast<int>(i % dim);
  float sum = 0.0f;
  for (int expert = 0; expert < experts; ++expert) {
    const int slot = inverse[row * experts + expert];
    if (slot >= 0) sum += values[static_cast<size_t>(slot) * dim + d] *
        (scales ? scales[slot] : 1.0f);
  }
  output[i] = sum;
}

extern "C" void launch_moe_ordered_assign(const float* weights, const int* offsets,
    int* permutation, float* scale, int* inverse, int rows, int experts) {
  moe_ordered_assign_kernel<<<(experts + 255)/256, 256, 0, nsos::gpu::current_stream()>>>(
      weights, offsets, permutation, scale, inverse, rows, experts);
}

extern "C" void launch_moe_ordered_combine(const float* values, const int* inverse,
    const float* scales, float* output, int rows, int dim, int experts) {
  const size_t count = static_cast<size_t>(rows) * dim;
  moe_ordered_combine_kernel<<<(count + 255)/256, 256, 0, nsos::gpu::current_stream()>>>(
      values, inverse, scales, output, rows, dim, experts);
}

__global__ void moe_gather_rows_kernel(const float *__restrict__ input,
                                        const int *__restrict__ permutation,
                                        float *__restrict__ permuted,
                                        int N_active, int dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N_active * dim;
  if (idx >= total) return;

  const int slot = idx / dim;
  const int d = idx % dim;
  const int src_row = permutation[slot];
  permuted[idx] = input[src_row * dim + d];
}

// Scatter-add scaled rows from permuted_output back into y[batch, dim].
// y MUST be zeroed beforehand.  Uses atomicAdd because multiple slots
// may share the same source row (top-k > 1).
__global__ void moe_scatter_add_weighted_kernel(
    const float *__restrict__ permuted_output,
    const int *__restrict__ permutation, const float *__restrict__ scale,
    float *__restrict__ y, int N_active, int dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N_active * dim;
  if (idx >= total) return;

  const int slot = idx / dim;
  const int d = idx % dim;
  const int dst_row = permutation[slot];
  const float scaled = permuted_output[idx] * scale[slot];
  atomicAdd(&y[dst_row * dim + d], scaled);
}

__global__ void moe_scale_rows_kernel(const float *__restrict__ input,
                                      const float *__restrict__ scale,
                                      float *__restrict__ output, int rows,
                                      int dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * dim;
  if (idx >= total) return;
  output[idx] = input[idx] * scale[idx / dim];
}

__global__ void moe_router_weight_grad_kernel(
    const float *__restrict__ dy,
    const float *__restrict__ unscaled_expert_output,
    const int *__restrict__ permutation, const int *__restrict__ offsets,
    float *__restrict__ grad_weights, int n_active, int dim,
    int num_experts) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n_active) return;
  int expert = 0;
  while (expert + 1 < num_experts && slot >= offsets[expert + 1]) ++expert;
  const int row = permutation[slot];
  double dot = 0.0;
  const size_t dy_base = static_cast<size_t>(row) * dim;
  const size_t output_base = static_cast<size_t>(slot) * dim;
  for (int d = 0; d < dim; ++d) {
    dot += static_cast<double>(dy[dy_base + d]) *
           unscaled_expert_output[output_base + d];
  }
  grad_weights[static_cast<size_t>(row) * num_experts + expert] =
      static_cast<float>(dot);
}

__device__ __forceinline__ bool moe_stable_topk_member(
    const float *probabilities, int expert, int num_experts, int top_k) {
  int rank = 0;
  const float value = probabilities[expert];
  for (int candidate = 0; candidate < num_experts; ++candidate) {
    const float other = probabilities[candidate];
    if (other > value || (other == value && candidate < expert)) ++rank;
  }
  return rank < top_k;
}

__global__ void moe_router_logits_grad_kernel(
    const float *__restrict__ probs,
    const float *__restrict__ grad_weights,
    float *__restrict__ grad_logits, int rows, int num_experts, int top_k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t base = static_cast<size_t>(row) * num_experts;
  const float *p = probs + base;
  const float *gw = grad_weights + base;
  const int effective_top_k = min(max(top_k, 1), num_experts);
  double selected_sum = 0.0;
  for (int expert = 0; expert < num_experts; ++expert) {
    if (moe_stable_topk_member(p, expert, num_experts, effective_top_k)) {
      selected_sum += p[expert];
    }
  }
  if (!(selected_sum > 0.0)) {
    for (int expert = 0; expert < num_experts; ++expert) {
      grad_logits[base + expert] = 0.0f;
    }
    return;
  }
  double grad_dot_weight = 0.0;
  for (int expert = 0; expert < num_experts; ++expert) {
    if (moe_stable_topk_member(p, expert, num_experts, effective_top_k)) {
      grad_dot_weight += static_cast<double>(gw[expert]) * p[expert] /
                         selected_sum;
    }
  }
  // For renormalized top-k, sum_j p_j*g_p_j is analytically zero, so the
  // softmax VJP reduces to w_j*(g_w_j - <g_w,w>) on selected experts.
  for (int expert = 0; expert < num_experts; ++expert) {
    if (moe_stable_topk_member(p, expert, num_experts, effective_top_k)) {
      grad_logits[base + expert] = static_cast<float>(
          (static_cast<double>(p[expert]) / selected_sum) *
          (gw[expert] - grad_dot_weight));
    } else {
      grad_logits[base + expert] = 0.0f;
    }
  }
}

// Dense single-row decode accumulation: out[i] += (*scale_dev) * y[i], with
// the scale read from DEVICE memory (the row's routing weight for one
// expert).  Host never sees the weights -> no per-token D2H, and the read
// happens at kernel EXECUTION, so the op is CUDA-graph capturable (the
// weight changes every replay while the pointer stays fixed).
__global__ void moe_scale_accum_row_kernel(float *__restrict__ out,
                                           const float *__restrict__ y,
                                           const float *__restrict__ scale_dev,
                                           int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] += (*scale_dev) * y[i];
}

extern "C" {

void launch_moe_scale_accum_row_kernel(float *out, const float *y,
                                       const float *scale_dev, int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(n, threads);
  moe_scale_accum_row_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, y, scale_dev, n);
}

void launch_moe_topk_mask_kernel(float *weights, int batch, int num_experts,
                                 int k) {
  if (batch <= 0 || num_experts <= 0) return;
  const int threads = 128;
  const int blocks = nsos::gpu::ceil_div_positive(batch, threads);
  moe_topk_mask_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(weights, batch, num_experts, k);
}

void launch_moe_load_accumulate_kernel(const float *weights,
                                       float *expert_loads, int batch,
                                       int num_experts) {
  if (batch <= 0 || num_experts <= 0) return;
  const int total = batch * num_experts;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_load_accumulate_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(weights, expert_loads, batch,
                                                  num_experts);
}

void launch_moe_load_ordered_kernel(const float* weights, float* loads,
                                    int rows, int experts) {
  if (!weights || !loads || rows <= 0 || experts <= 0) return;
  moe_load_ordered_kernel<<<nsos::gpu::ceil_div_positive(experts, 128), 128, 0,
      nsos::gpu::current_stream()>>>(weights, loads, rows, experts);
}

void launch_moe_zero_invalid_rows_kernel(float *weights,
                                         const uint8_t *valid_rows,
                                         int batch, int num_experts) {
  if (batch <= 0 || num_experts <= 0 || valid_rows == nullptr) return;
  const int total = batch * num_experts;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_zero_invalid_rows_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      weights, valid_rows, batch, num_experts);
}

void launch_moe_switch_aux_stats_kernel(const float *probs, float *counts,
                                        float *prob_sums, int rows,
                                        int num_experts, int top_k,
                                        bool deterministic) {
  if (rows <= 0 || num_experts <= 0 || top_k <= 0) return;
  if (deterministic) {
    moe_switch_aux_stats_deterministic_kernel<<<nsos::gpu::ceil_div_positive(num_experts, 128), 128, 0, nsos::gpu::current_stream()>>>(
        probs, counts, prob_sums, rows, num_experts, top_k);
    return;
  }
  const int threads = 128;
  const int blocks = nsos::gpu::ceil_div_positive(rows, threads);
  moe_switch_aux_stats_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      probs, counts, prob_sums, rows, num_experts, top_k);
}

void launch_moe_switch_aux_grad_kernel(const float *probs,
                                       const float *counts, float *grad,
                                       int rows, int num_experts, float coef) {
  if (rows <= 0 || num_experts <= 0) return;
  const int threads = 128;
  const int blocks = nsos::gpu::ceil_div_positive(rows, threads);
  moe_switch_aux_grad_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      probs, counts, grad, rows, num_experts, coef);
}

void launch_moe_switch_aux_loss_kernel(const float *counts,
                                       const float *prob_sums, float *loss,
                                       int rows, int num_experts, float coef,
                                       bool deterministic) {
  if (rows <= 0 || num_experts <= 0) return;
  if (deterministic) {
    moe_switch_aux_loss_deterministic_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(
        counts, prob_sums, loss, rows, num_experts, coef);
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(num_experts, threads);
  moe_switch_aux_loss_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      counts, prob_sums, loss, rows, num_experts, coef);
}

void launch_moe_count_per_expert_kernel(const float *weights, int *counts,
                                         int batch, int num_experts) {
  if (batch <= 0 || num_experts <= 0) return;
  const int total = batch * num_experts;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_count_per_expert_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(weights, counts, batch,
                                                    num_experts);
}

void launch_moe_exclusive_scan_small_kernel(const int *counts, int *offsets,
                                             int num_experts) {
  if (num_experts <= 0) return;
  // Single block, threads sized to num_experts (rounded up to next pow2
  // via 1024 cap).
  int threads = 1;
  while (threads < num_experts && threads < 1024) threads <<= 1;
  if (threads > 1024) threads = 1024;
  moe_exclusive_scan_small_kernel<<<1, threads, 0, nsos::gpu::current_stream()>>>(counts, offsets, num_experts);
}

void launch_moe_compute_assignments_kernel(const float *weights,
                                            const int *offsets,
                                            int *workspace_counters,
                                            int *permutation, int *assignment,
                                            float *scale, int batch,
                                            int num_experts) {
  if (batch <= 0 || num_experts <= 0) return;
  const int total = batch * num_experts;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_compute_assignments_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      weights, offsets, workspace_counters, permutation, assignment, scale,
      batch, num_experts);
}

void launch_moe_gather_rows_kernel(const float *input, const int *permutation,
                                    float *permuted, int N_active, int dim) {
  if (N_active <= 0 || dim <= 0) return;
  const int total = N_active * dim;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_gather_rows_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(input, permutation, permuted,
                                               N_active, dim);
}

void launch_moe_scatter_add_weighted_kernel(const float *permuted_output,
                                             const int *permutation,
                                             const float *scale, float *y,
                                             int N_active, int dim) {
  if (N_active <= 0 || dim <= 0) return;
  const int total = N_active * dim;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_scatter_add_weighted_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      permuted_output, permutation, scale, y, N_active, dim);
}

void launch_moe_scale_rows_kernel(const float *input, const float *scale,
                                  float *output, int rows, int dim) {
  if (rows <= 0 || dim <= 0) return;
  const int total = rows * dim;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  moe_scale_rows_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(input, scale, output, rows, dim);
}

void launch_moe_router_weight_grad_kernel(
    const float *dy, const float *unscaled_expert_output,
    const int *permutation, const int *offsets, float *grad_weights,
    int n_active, int dim, int num_experts) {
  if (n_active <= 0 || dim <= 0 || num_experts <= 0) return;
  const int threads = 128;
  const int blocks = nsos::gpu::ceil_div_positive(n_active, threads);
  moe_router_weight_grad_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      dy, unscaled_expert_output, permutation, offsets, grad_weights,
      n_active, dim, num_experts);
}

void launch_moe_router_logits_grad_kernel(
    const float *probs, const float *grad_weights, float *grad_logits,
    int rows, int num_experts, int top_k) {
  if (rows <= 0 || num_experts <= 0 || top_k <= 0) return;
  const int threads = 128;
  const int blocks = nsos::gpu::ceil_div_positive(rows, threads);
  moe_router_logits_grad_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      probs, grad_weights, grad_logits, rows, num_experts, top_k);
}

}  // extern "C"

namespace {
// A small fixed reduction avoids architecture-specific assumptions and has
// bounded LDS usage independent of the number of experts or input width.
__device__ float moe_decode_reduce(float value, float* scratch, bool maximum) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = maximum
          ? fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride])
          : scratch[threadIdx.x] + scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();  // all waves consume the reduction before scratch is reused
  return result;
}

__global__ __launch_bounds__(256) void moe_decode_prepare_kernel(
    const nsos::GpuLinearView* views, const float* routing,
    const float* input, float* prepared, float* scales,
    int input_stride, int experts = 0, int input_row_stride = 0) {
  const int e = blockIdx.x;
  const int row = blockIdx.y;
  const int slot = row * experts + e;
  if (routing && routing[slot] == 0.0f) return;
  const auto view = views[e];
  const float* x = input + static_cast<size_t>(e) * input_stride + static_cast<size_t>(row) * input_row_stride;
  float* y = prepared + static_cast<size_t>(slot) * view.inputs;
  __shared__ float scratch[256];
  float squares = 0.0f;
  if (view.rms_input) {
    for (int k = threadIdx.x; k < view.inputs; k += blockDim.x)
      squares += x[k] * x[k];
  }
  const float total = moe_decode_reduce(squares, scratch, false);
  const float inv_norm = view.rms_input ? rsqrtf(total / view.inputs + 1e-6f) : 1.0f;
  float maximum = 0.0f;
  for (int k = threadIdx.x; k < view.inputs; k += blockDim.x)
    maximum = fmaxf(maximum, fabsf(x[k] * inv_norm));
  maximum = moe_decode_reduce(maximum, scratch, true);
  const float qmax = view.activation_bits ? float((1 << (view.activation_bits - 1)) - 1) : 1.0f;
  const float scale = view.activation_bits ? (maximum + 1e-8f) / qmax : 1.0f;
  if (threadIdx.x == 0) scales[slot] = scale;
  for (int k = threadIdx.x; k < view.inputs; k += blockDim.x) {
    float value = x[k] * inv_norm;
    if (view.activation_bits) {
      value = roundf(fminf(qmax, fmaxf(-qmax, value * (qmax / (maximum + 1e-8f)))));
      if (!view.packed) value *= scale;
    }
    y[k] = value;
  }
}

__global__ __launch_bounds__(64) void moe_decode_linear_kernel(
    const nsos::GpuLinearView* views, const float* routing,
    const float* prepared, const float* scales, float* output, bool squared_relu,
    int output_stride = 0, int experts = 0) {
  const int e = blockIdx.y;
  const int slot = blockIdx.z * experts + e;
  const int n = blockIdx.x;
  const auto view = views[e];
  if (n >= view.outputs) return;
  float* destination = output + static_cast<size_t>(slot) * (output_stride ? output_stride : view.outputs) + n;
  if (routing && routing[slot] == 0.0f) {
    if (threadIdx.x == 0) *destination = 0.0f;
    return;
  }
  const float* x = prepared + static_cast<size_t>(slot) * view.inputs;
  float value = 0.0f;
  int integer_value = 0;
  for (int k = threadIdx.x; k < view.inputs; k += blockDim.x) {
    if (view.packed) {
      const uint32_t word = view.packed[static_cast<size_t>(n) * (view.inputs / 16) + k / 16];
      const int weight = int((word >> ((k % 16) * 2)) & 3u) - 1;
      integer_value += int(x[k]) * weight;
    } else {
      value += x[k] * view.weight[static_cast<size_t>(n) * view.inputs + k];
    }
  }
  __shared__ float partials[64];
  __shared__ int integers[64];
  partials[threadIdx.x] = value;
  integers[threadIdx.x] = integer_value;
  __syncthreads();
  for (int stride = 32; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
      integers[threadIdx.x] += integers[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    value = view.packed ? __fmul_rn(__fmul_rn(float(integers[0]), view.weight_scale), scales[slot]) : partials[0];
    if (view.magnitude) value = __fmul_rn(value, view.magnitude[n]);
    if (view.bias) value = __fadd_rn(value, view.bias[n]);
    if (squared_relu) { value = fmaxf(value, 0.0f); value *= value; }
    *destination = value;
  }
}

__global__ void moe_decode_merge_kernel(const float* contributions,
                                       const float* routing, float* output,
                                       int experts, int dim) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= dim) return;
  routing += static_cast<size_t>(blockIdx.y) * experts;
  contributions += static_cast<size_t>(blockIdx.y) * experts * dim;
  output += static_cast<size_t>(blockIdx.y) * dim;
  float sum = 0.0f;
  for (int e = 0; e < experts; ++e) {
    if (routing[e] != 0.0f)
      sum = __fadd_rn(sum, __fmul_rn(routing[e], contributions[static_cast<size_t>(e) * dim + n]));
  }
  output[n] = sum;
}
}  // namespace

extern "C" void launch_grouped_decode_projections(const nsos::GpuLinearView* views,
    const float* input, float* prepared, float* scales, float* output,
    int groups, int output_stride) {
  moe_decode_prepare_kernel<<<groups, 256, 0, nsos::gpu::current_stream()>>>(views, nullptr, input, prepared, scales, 0);
  moe_decode_linear_kernel<<<dim3(output_stride, groups), 64, 0, nsos::gpu::current_stream()>>>(views, nullptr, prepared, scales, output, false, output_stride);
}

extern "C" void launch_moe_sparse_decode(
    const nsos::GpuLinearView* views, const float* routing, const float* input,
    float* prepared, float* scales, float* hidden, float* contributions,
    float* output, int experts, int dim, int hidden_dim, int rows) {
  moe_decode_prepare_kernel<<<dim3(experts, rows), 256, 0, nsos::gpu::current_stream()>>>(views, routing, input, prepared, scales, 0, experts, dim);
  moe_decode_linear_kernel<<<dim3(hidden_dim, experts, rows), 64, 0, nsos::gpu::current_stream()>>>(views, routing, prepared, scales, hidden, true, 0, experts);
  moe_decode_prepare_kernel<<<dim3(experts, rows), 256, 0, nsos::gpu::current_stream()>>>(views + experts, routing, hidden, prepared, scales, hidden_dim, experts, experts * hidden_dim);
  moe_decode_linear_kernel<<<dim3(dim, experts, rows), 64, 0, nsos::gpu::current_stream()>>>(views + experts, routing, prepared, scales, contributions, false, 0, experts);
  moe_decode_merge_kernel<<<dim3(nsos::gpu::ceil_div_positive(dim, 256), rows), 256, 0, nsos::gpu::current_stream()>>>(contributions, routing, output, experts, dim);
}
