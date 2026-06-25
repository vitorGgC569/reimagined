// LEARN C1 (2026-05-16) — Flash Attention v2 forward (FlashAttention-2)
// Reference: Dao 2023 "FlashAttention-2: Faster Attention with Better
// Parallelism and Work Partitioning" (arXiv 2307.08691).
//
// vs the existing gqa_causal_attention_kernel:
//   * Existing kernel materializes the full attention row in shared
//     memory (O(seq_len) shared per block).  For seq_len=512 this is
//     2KB per block — fits but limits occupancy.
//   * Flash Attention tiles K/V into blocks of size Bc (typically 64
//     or 128) and updates the output incrementally using ONLINE
//     SOFTMAX (Milakov & Gimelshein 2018).  Shared memory is now
//     O(Bc) per block regardless of seq_len.  This unlocks long
//     contexts (>4K) and improves SM occupancy.
//
// The online softmax recurrence:
//   For each K/V block i, after computing the per-row scores S_i:
//     m_new = max(m_old, rowmax(S_i))
//     P_i = exp(S_i - m_new)
//     sum_new = exp(m_old - m_new) * sum_old + rowsum(P_i)
//     O_new = exp(m_old - m_new) * O_old + P_i @ V_i
//   At the end: O_final = O_new / sum_new
//   loss-relevant log-sum-exp: lse = m_new + log(sum_new)
//
// Causal masking: for query at position q, K positions k > q are
// masked out (we skip blocks where the entire K block is in the
// future, and within a partial block we mask individual entries).
//
// This implementation is FP32 throughout (no Tensor Core path).  A
// follow-up (LEARN C1.2) will add BF16 Tensor Core via cuBLAS-style
// fragments — for now we keep correctness over peak throughput.
//
// Integration point: Attention::forward in jamba.cpp dispatches to
// launch_flash_attention_kernel when NSOS_USE_FLASH_ATTENTION=1 is
// set in the environment AND head_dim is supported (64, 128 typical).

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

#ifndef NSOS_FLASH_ATTN_BR
#define NSOS_FLASH_ATTN_BR 64   // Query block size (rows per block)
#endif
#ifndef NSOS_FLASH_ATTN_BC
#define NSOS_FLASH_ATTN_BC 64   // Key/Value block size (cols per K/V chunk)
#endif

namespace {

// Block-wide reduce-max using warp shuffles + shared memory.
// Caller guarantees blockDim.x is a multiple of 32 (warp size).
__inline__ __device__ float block_reduce_max_f32(float val,
                                                 float* warp_max_scratch) {
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if (lane == 0) {
        warp_max_scratch[warp_id] = val;
    }
    __syncthreads();
    // First warp reduces the per-warp results
    float merged = (threadIdx.x < warps_per_block)
                       ? warp_max_scratch[threadIdx.x]
                       : -FLT_MAX;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            merged = fmaxf(merged, __shfl_down_sync(0xFFFFFFFF, merged, offset));
        }
        if (threadIdx.x == 0) {
            warp_max_scratch[0] = merged;
        }
    }
    __syncthreads();
    return warp_max_scratch[0];
}

__inline__ __device__ float block_reduce_sum_f32(float val,
                                                 float* warp_sum_scratch) {
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (lane == 0) {
        warp_sum_scratch[warp_id] = val;
    }
    __syncthreads();
    float merged = (threadIdx.x < warps_per_block)
                       ? warp_sum_scratch[threadIdx.x]
                       : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            merged += __shfl_down_sync(0xFFFFFFFF, merged, offset);
        }
        if (threadIdx.x == 0) {
            warp_sum_scratch[0] = merged;
        }
    }
    __syncthreads();
    return warp_sum_scratch[0];
}

}  // namespace

// Flash Attention v2 forward, causal, single batch element, one head.
//
// Grid:
//   gridDim.x = ceil(seq_len / Br)      one block per Br queries
//   gridDim.y = num_heads               one block-row per head
//   gridDim.z = 1                       batch is folded into Q/K/V offsets
// Block:
//   blockDim.x = power-of-2 in [32, 256]
//
// Q, K, V are pre-RoPE'd (rotary already applied) and arranged as:
//   Q: [seq_len, num_heads, head_dim]    row-major
//   K: [seq_len, num_kv_heads, head_dim] row-major (GQA-friendly)
//   V: [seq_len, num_kv_heads, head_dim] row-major
// Out: [seq_len, num_heads, head_dim] row-major
//
// For GQA the kernel maps query head h -> kv head h / group_size.
__global__ void flash_attention_fwd_kernel(const float* __restrict__ Q,
                                            const float* __restrict__ K,
                                            const float* __restrict__ V,
                                            float* __restrict__ O,
                                            int seq_len, int num_heads,
                                            int num_kv_heads, int head_dim,
                                            int kv_group_size,
                                            float scale) {
    extern __shared__ float smem[];
    const int Br = NSOS_FLASH_ATTN_BR;
    const int Bc = NSOS_FLASH_ATTN_BC;

    // smem layout (in floats):
    //   Kj [Bc * head_dim]        — current K block, transposed-friendly
    //   Vj [Bc * head_dim]        — current V block
    //   warp_scratch [32]         — shared scratch for block-reduce
    float* Kj = smem;
    float* Vj = smem + Bc * head_dim;
    float* warp_scratch = Vj + Bc * head_dim;

    const int q_block_id = blockIdx.x;          // which Br block of queries
    const int head_id = blockIdx.y;             // which head
    const int kv_head_id = (kv_group_size > 0)
                              ? min(head_id / kv_group_size, num_kv_heads - 1)
                              : head_id;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    // Bring Q for our Br rows into per-thread registers.
    // We use a "thread per query row" parallelism: each thread is
    // responsible for ONE query row's accumulator (m, l, O).  This is
    // the classic FlashAttention-2 partition (vs v1 which partitioned
    // over K/V chunks).
    //
    // For seq_len/Br threads needed; we spread them across warps.
    const int q_base = q_block_id * Br;
    const int q_local = tid;                    // local query row (0..Br-1)
    const int q_global = q_base + q_local;
    const bool valid_query = (q_global < seq_len) && (q_local < Br);

    // Per-thread accumulators (one query row each).  We store O in
    // registers (small head_dim, e.g. 64).
    float m_i = -FLT_MAX;     // running max
    float l_i = 0.0f;         // running denom (sum of exp)
    // O_i is head_dim floats in thread-local memory; we use the
    // smaller stack-allocated array since head_dim is bounded at
    // compile time for the typical configs (64, 128).
    constexpr int MAX_HEAD_DIM = 128;
    float O_i[MAX_HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < MAX_HEAD_DIM; ++d) {
        O_i[d] = 0.0f;
    }

    // Load this query row into registers.
    float Q_i[MAX_HEAD_DIM];
    if (valid_query) {
        for (int d = 0; d < head_dim; ++d) {
            Q_i[d] = Q[q_global * num_heads * head_dim + head_id * head_dim + d];
        }
    }

    // Number of K/V column blocks to iterate.  Causal: we only need
    // to process columns up to q_block_id's last query, which is
    // q_base + Br - 1.  That's columns 0 .. q_base + Br - 1.
    // Number of blocks = ceil((q_base + Br) / Bc).
    const int last_col = min(q_base + Br, seq_len);
    const int n_col_blocks = (last_col + Bc - 1) / Bc;

    for (int j = 0; j < n_col_blocks; ++j) {
        const int col_base = j * Bc;

        // Cooperative load of K_j and V_j into shared memory.
        // Each thread loads multiple elements.
        const int kv_block_size = Bc * head_dim;
        for (int idx = tid; idx < kv_block_size; idx += threads) {
            const int row = idx / head_dim;
            const int dim_idx = idx % head_dim;
            const int col_global = col_base + row;
            if (col_global < seq_len) {
                Kj[idx] = K[col_global * num_kv_heads * head_dim +
                            kv_head_id * head_dim + dim_idx];
                Vj[idx] = V[col_global * num_kv_heads * head_dim +
                            kv_head_id * head_dim + dim_idx];
            } else {
                Kj[idx] = 0.0f;
                Vj[idx] = 0.0f;
            }
        }
        __syncthreads();

        if (valid_query) {
            // Compute S_ij = Q_i · K_j^T (Br scores per thread).
            // We then do online softmax update on the row.
            float scores[Bc];
            float row_max = -FLT_MAX;

            for (int c = 0; c < Bc; ++c) {
                const int col_global = col_base + c;
                if (col_global > q_global) {
                    // Causal mask: future tokens are -inf
                    scores[c] = -FLT_MAX;
                } else if (col_global >= seq_len) {
                    scores[c] = -FLT_MAX;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += Q_i[d] * Kj[c * head_dim + d];
                    }
                    scores[c] = dot * scale;
                    row_max = fmaxf(row_max, scores[c]);
                }
            }

            const float m_new = fmaxf(m_i, row_max);
            float row_sum = 0.0f;
            for (int c = 0; c < Bc; ++c) {
                if (scores[c] > -FLT_MAX / 2) {
                    scores[c] = expf(scores[c] - m_new);
                    row_sum += scores[c];
                } else {
                    scores[c] = 0.0f;
                }
            }
            const float scale_old = expf(m_i - m_new);
            const float l_new = scale_old * l_i + row_sum;

            // Update O: O_new = scale_old * O_old + scores @ V_j
            for (int d = 0; d < head_dim; ++d) {
                float acc = scale_old * O_i[d];
                for (int c = 0; c < Bc; ++c) {
                    acc += scores[c] * Vj[c * head_dim + d];
                }
                O_i[d] = acc;
            }

            m_i = m_new;
            l_i = l_new;
        }

        __syncthreads();  // before reloading Kj/Vj next iteration
    }

    // Final normalization: O = O / l, write to global memory.
    if (valid_query) {
        const float inv_l = 1.0f / fmaxf(l_i, 1e-9f);
        for (int d = 0; d < head_dim; ++d) {
            O[q_global * num_heads * head_dim + head_id * head_dim + d] =
                O_i[d] * inv_l;
        }
    }
    // Silence unused-warning when the reductions aren't taken.
    (void)warp_scratch;
}

extern "C" void launch_flash_attention_kernel(const float* Q, const float* K,
                                               const float* V, float* O,
                                               int seq_len, int num_heads,
                                               int num_kv_heads, int head_dim,
                                               int kv_group_size, float scale) {
    // OOB guard: the kernel stores Q_i/O_i in fixed-size MAX_HEAD_DIM=128
    // register arrays (and scores[Bc]); a head_dim beyond that overflows the
    // per-thread stack.  Refuse to launch (caller must fall back to the dense
    // gqa_causal_attention path) rather than corrupt memory.  This path is
    // opt-in (NSOS_USE_FLASH_ATTENTION) and currently unwired, so the guard is
    // a landmine-defuse for whoever enables it.
    if (head_dim <= 0 || head_dim > 128) {
        std::fprintf(stderr,
                     "[flash_attn] head_dim=%d unsupported (max 128); skipping "
                     "flash kernel — use dense attention\n",
                     head_dim);
        return;
    }
    const int Br = NSOS_FLASH_ATTN_BR;
    const int Bc = NSOS_FLASH_ATTN_BC;
    const dim3 grid((seq_len + Br - 1) / Br, num_heads, 1);
    const dim3 block(Br, 1, 1);
    // Shared memory: 2 * Bc * head_dim (Kj, Vj) + 32 (warp scratch)
    const size_t shmem_bytes =
        (2 * Bc * head_dim + 32) * sizeof(float);
    flash_attention_fwd_kernel<<<grid, block, shmem_bytes>>>(
        Q, K, V, O, seq_len, num_heads, num_kv_heads, head_dim,
        kv_group_size, scale);
}
