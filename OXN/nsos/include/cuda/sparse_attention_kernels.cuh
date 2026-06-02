#ifndef SPARSE_ATTENTION_KERNELS_CUH
#define SPARSE_ATTENTION_KERNELS_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {
namespace cuda {

#ifdef USE_CUDA
// =====================================================================
// SSA (Subquadratic Sparse Attention) GPU kernels — match the host
// implementation in src/sparse_attention.cpp 1:1 (parity-tested).
//
// Caps (dispatch falls back to CPU when exceeded): head dim <= 256,
// top_k_blocks <= 64.  All pointers are device pointers; no internal sync.
// =====================================================================

// Per-block mean key: bm[b, c] = mean_j-in-block k[j, c].  bm is [nb, d],
// nb = ceil(n / B).
void launch_sparse_block_means(const float* k, float* bm, int n, int d, int B);

// Subquadratic selective causal attention, one thread per query.  For query i
// the visible blocks are 0..i/B; attention sinks (first `sink_blocks`), the
// local window (last `local_blocks`), and the top-`top_k` content-scored blocks
// (route_i . bm_b) are selected, then exact softmax attention runs over the
// causal positions inside the selected blocks.  `route` is the block-scoring
// query (raw q, or Wsel@q for the learned selector); `out` is [n, d].
void launch_sparse_selective_attention(const float* q, const float* k,
                                       const float* v, const float* route,
                                       const float* bm, float* out, int n, int d,
                                       int B, int top_k, int local_blocks,
                                       int sink_blocks, float scale);
#endif  // USE_CUDA

}  // namespace cuda
}  // namespace nsos

#endif  // SPARSE_ATTENTION_KERNELS_CUH
