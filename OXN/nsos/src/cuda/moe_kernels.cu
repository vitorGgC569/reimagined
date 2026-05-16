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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Per-row stack-allocated arrays cap the supported expert count.  64 is
// far above current configurations (8 in stable_jamba, 32 in scaled-up
// research), but cheap enough to keep without measurable register
// pressure on Pascal.
#define MOE_TOPK_MASK_MAX_EXPERTS 64

// =====================================================================
// Top-k mask + renormalize, one CUDA thread per row.
//
// Strategy: each thread reads its row into a 64-element register array,
// computes a partial sort to find the k-th largest value (used as the
// threshold), zeros entries below the threshold, sums the survivors,
// and divides through.  For typical num_experts (8 or 16) the partial
// sort is essentially free; for 64 it is still O(num_experts * k) which
// is fine because k itself is tiny (typically 2).
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

  // Local copy of row values + their original indices.  We use a
  // selection-sort over k iterations to identify the k largest (and
  // their indices) without a full sort — this is O(num_experts * k)
  // and avoids dynamic shared memory.
  float local_vals[MOE_TOPK_MASK_MAX_EXPERTS];
  int   local_idx[MOE_TOPK_MASK_MAX_EXPERTS];
  const int N =
      (num_experts > MOE_TOPK_MASK_MAX_EXPERTS) ? MOE_TOPK_MASK_MAX_EXPERTS
                                                : num_experts;
  for (int e = 0; e < N; ++e) {
    local_vals[e] = row_ptr[e];
    local_idx[e] = e;
  }

  // Partial selection sort: bring the top-k to the front.  Stable in
  // index order on ties, matching std::partial_sort's predicate-only
  // ordering (CPU code uses ranked indices comparing weights, with no
  // tie-breaker — both implementations may pick different ties; this
  // is acceptable because the renormalization absorbs the choice).
  for (int rank = 0; rank < eff_k; ++rank) {
    int best = rank;
    for (int j = rank + 1; j < N; ++j) {
      if (local_vals[j] > local_vals[best]) {
        best = j;
      }
    }
    if (best != rank) {
      float tv = local_vals[rank];
      local_vals[rank] = local_vals[best];
      local_vals[best] = tv;
      int ti = local_idx[rank];
      local_idx[rank] = local_idx[best];
      local_idx[best] = ti;
    }
  }

  // Sum of the survivors and inverse for renormalization.
  float selected_sum = 0.0f;
  for (int rank = 0; rank < eff_k; ++rank) {
    selected_sum += local_vals[rank];
  }
  const float inv_selected = 1.0f / fmaxf(selected_sum, 1e-9f);

  // Write back: zero everything, then scatter the renormalized
  // survivors into their original positions.
  for (int e = 0; e < num_experts; ++e) {
    row_ptr[e] = 0.0f;
  }
  for (int rank = 0; rank < eff_k; ++rank) {
    row_ptr[local_idx[rank]] = local_vals[rank] * inv_selected;
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
  } else if (tid <= 1024) {
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

extern "C" {

void launch_moe_topk_mask_kernel(float *weights, int batch, int num_experts,
                                 int k) {
  if (batch <= 0 || num_experts <= 0) return;
  const int threads = 128;
  const int blocks = (batch + threads - 1) / threads;
  moe_topk_mask_kernel<<<blocks, threads>>>(weights, batch, num_experts, k);
}

void launch_moe_load_accumulate_kernel(const float *weights,
                                       float *expert_loads, int batch,
                                       int num_experts) {
  if (batch <= 0 || num_experts <= 0) return;
  const int total = batch * num_experts;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  moe_load_accumulate_kernel<<<blocks, threads>>>(weights, expert_loads, batch,
                                                  num_experts);
}

void launch_moe_count_per_expert_kernel(const float *weights, int *counts,
                                         int batch, int num_experts) {
  if (batch <= 0 || num_experts <= 0) return;
  const int total = batch * num_experts;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  moe_count_per_expert_kernel<<<blocks, threads>>>(weights, counts, batch,
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
  moe_exclusive_scan_small_kernel<<<1, threads>>>(counts, offsets, num_experts);
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
  const int blocks = (total + threads - 1) / threads;
  moe_compute_assignments_kernel<<<blocks, threads>>>(
      weights, offsets, workspace_counters, permutation, assignment, scale,
      batch, num_experts);
}

void launch_moe_gather_rows_kernel(const float *input, const int *permutation,
                                    float *permuted, int N_active, int dim) {
  if (N_active <= 0 || dim <= 0) return;
  const int total = N_active * dim;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  moe_gather_rows_kernel<<<blocks, threads>>>(input, permutation, permuted,
                                               N_active, dim);
}

void launch_moe_scatter_add_weighted_kernel(const float *permuted_output,
                                             const int *permutation,
                                             const float *scale, float *y,
                                             int N_active, int dim) {
  if (N_active <= 0 || dim <= 0) return;
  const int total = N_active * dim;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  moe_scatter_add_weighted_kernel<<<blocks, threads>>>(
      permuted_output, permutation, scale, y, N_active, dim);
}

}  // extern "C"
