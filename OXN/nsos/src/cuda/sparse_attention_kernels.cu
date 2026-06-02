#include "../../include/cuda/sparse_attention_kernels.cuh"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <math.h>

namespace nsos {
namespace cuda {

namespace {

constexpr int kSparseTopKMax = 64;   // top_k_blocks cap (per-thread local array)
constexpr int kSparseDimMax = 256;   // head dim cap (per-thread accumulator)

// bm[b, c] = mean over the block's keys.  One thread per (block, channel).
__global__ void sparse_block_means_kernel(const float* __restrict__ k,
                                          float* __restrict__ bm, int n, int d,
                                          int B, int nb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * d) return;
    const int b = idx / d;
    const int c = idx % d;
    const int start = b * B;
    const int end = min((b + 1) * B, n);
    const int cnt = end - start;
    float s = 0.0f;
    for (int j = start; j < end; ++j) {
        s += k[static_cast<size_t>(j) * d + c];
    }
    bm[idx] = cnt > 0 ? s / static_cast<float>(cnt) : 0.0f;
}

// One thread per query.  Selects sink + local + top-k blocks (content-scored
// against the per-block mean key), then runs exact online-softmax causal
// attention over the positions in the selected blocks.  Matches the host
// sparse_selective_attention in src/sparse_attention.cpp.
__global__ void sparse_selective_attention_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ route,
    const float* __restrict__ bm, float* __restrict__ out, int n, int d, int B,
    int top_k, int local_blocks, int sink_blocks, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int cur = i / B;
    const int ncand = cur + 1;
    const int sink = min(max(sink_blocks, 0), ncand);
    const int local = max(local_blocks, 0);
    const int lo = max(0, cur - local + 1);

    int tk = top_k;
    if (tk > kSparseTopKMax) tk = kSparseTopKMax;

    // Content selection: keep the tk highest-scoring non-(sink/local) blocks.
    float topsc[kSparseTopKMax];
    int topidx[kSparseTopKMax];
    int kk = 0;
    if (tk > 0) {
        const float* ri = route + static_cast<size_t>(i) * d;
        for (int b = 0; b < ncand; ++b) {
            const bool autosel = (b < sink) || (b >= lo && b <= cur);
            if (autosel) continue;
            const float* bmb = bm + static_cast<size_t>(b) * d;
            float s = 0.0f;
            for (int c = 0; c < d; ++c) s += ri[c] * bmb[c];
            s *= scale;
            if (kk < tk) {
                topsc[kk] = s;
                topidx[kk] = b;
                ++kk;
            } else {
                int mi = 0;
                for (int t = 1; t < tk; ++t) {
                    if (topsc[t] < topsc[mi]) mi = t;
                }
                if (s > topsc[mi]) {
                    topsc[mi] = s;
                    topidx[mi] = b;
                }
            }
        }
    }

    // Exact attention over selected blocks, online softmax (flash-style).
    float acc[kSparseDimMax];
    for (int c = 0; c < d; ++c) acc[c] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    const float* qi = q + static_cast<size_t>(i) * d;

    for (int b = 0; b <= cur; ++b) {
        bool sel = (b < sink) || (b >= lo && b <= cur);
        if (!sel) {
            for (int t = 0; t < kk; ++t) {
                if (topidx[t] == b) {
                    sel = true;
                    break;
                }
            }
        }
        if (!sel) continue;

        const int start = b * B;
        const int end = min((b + 1) * B, i + 1);  // causal: j <= i
        for (int j = start; j < end; ++j) {
            const float* kj = k + static_cast<size_t>(j) * d;
            const float* vj = v + static_cast<size_t>(j) * d;
            float s = 0.0f;
            for (int c = 0; c < d; ++c) s += qi[c] * kj[c];
            s *= scale;
            const float m_new = fmaxf(m, s);
            const float corr = expf(m - m_new);
            const float p = expf(s - m_new);
            l = l * corr + p;
            for (int c = 0; c < d; ++c) acc[c] = acc[c] * corr + p * vj[c];
            m = m_new;
        }
    }

    const float inv = 1.0f / (l + 1e-20f);
    float* oi = out + static_cast<size_t>(i) * d;
    for (int c = 0; c < d; ++c) oi[c] = acc[c] * inv;
}

}  // namespace

void launch_sparse_block_means(const float* k, float* bm, int n, int d, int B) {
    const int nb = (n + B - 1) / B;
    const int total = nb * d;
    if (total <= 0) return;
    const int blocks = (total + 255) / 256;
    sparse_block_means_kernel<<<blocks, 256>>>(k, bm, n, d, B, nb);
}

void launch_sparse_selective_attention(const float* q, const float* k,
                                       const float* v, const float* route,
                                       const float* bm, float* out, int n, int d,
                                       int B, int top_k, int local_blocks,
                                       int sink_blocks, float scale) {
    if (n <= 0) return;
    const int blocks = (n + 127) / 128;
    sparse_selective_attention_kernel<<<blocks, 128>>>(
        q, k, v, route, bm, out, n, d, B, top_k, local_blocks, sink_blocks, scale);
}

}  // namespace cuda
}  // namespace nsos

#endif  // USE_CUDA
