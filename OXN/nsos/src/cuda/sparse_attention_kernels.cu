#include "../../include/cuda/sparse_attention_kernels.cuh"

#ifdef USE_CUDA

#include "gpu_backend.h"
#include <climits>
#include <math.h>
#include <stdexcept>

namespace nsos {
namespace cuda {

namespace {

constexpr int kSparseTopKMax = 64;   // top_k_blocks cap (per-thread local array)
constexpr int kSparseDimMax = 256;   // head dim cap (per-thread accumulator)
constexpr int kSparseBlockMax = 512; // selector distillation local deltas

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

__device__ __forceinline__ int sparse_select_top_blocks(
    int i, const float* route_i, const float* bm, int d, int B, int top_k,
    int local_blocks, int sink_blocks, float scale,
    float (&top_scores)[kSparseTopKMax], int (&top_indices)[kSparseTopKMax]) {
    const int current = i / B;
    const int candidates = current + 1;
    const int sink = min(max(sink_blocks, 0), candidates);
    const int local = max(local_blocks, 0);
    const int local_begin = max(0, current - local + 1);
    const int k_limit = min(max(top_k, 0), kSparseTopKMax);
    int selected_count = 0;
    for (int block = 0; block < candidates; ++block) {
        if (block < sink || (block >= local_begin && block <= current)) continue;
        const float* mean = bm + static_cast<size_t>(block) * d;
        float score = 0.0f;
        for (int c = 0; c < d; ++c) score += route_i[c] * mean[c];
        score *= scale;
        if (selected_count < k_limit) {
            top_scores[selected_count] = score;
            top_indices[selected_count] = block;
            ++selected_count;
        } else if (k_limit > 0) {
            int minimum = 0;
            for (int slot = 1; slot < k_limit; ++slot) {
                if (top_scores[slot] < top_scores[minimum] ||
                    (top_scores[slot] == top_scores[minimum] &&
                     top_indices[slot] > top_indices[minimum])) {
                    minimum = slot;
                }
            }
            if (score > top_scores[minimum] ||
                (score == top_scores[minimum] && block < top_indices[minimum])) {
                top_scores[minimum] = score;
                top_indices[minimum] = block;
            }
        }
    }
    return selected_count;
}

__device__ __forceinline__ bool sparse_block_selected(
    int block, int current, int selected_count, const int* top_indices,
    int local_blocks, int sink_blocks) {
    const int candidates = current + 1;
    const int sink = min(max(sink_blocks, 0), candidates);
    const int local_begin = max(0, current - max(local_blocks, 0) + 1);
    if (block < sink || (block >= local_begin && block <= current)) return true;
    for (int slot = 0; slot < selected_count; ++slot) {
        if (top_indices[slot] == block) return true;
    }
    return false;
}

__device__ void sparse_attention_backward_query(
    int i, const float* q, const float* k, const float* v,
    const float* route, const float* bm, const float* d_out,
    float* d_q, float* d_k, float* d_v, int n, int d, int B, int top_k,
    int local_blocks, int sink_blocks, float scale) {
    (void)n;
    float top_scores[kSparseTopKMax];
    int top_indices[kSparseTopKMax];
    const int selected_count = sparse_select_top_blocks(
        i, route + static_cast<size_t>(i) * d, bm, d, B, top_k,
        local_blocks, sink_blocks, scale,
        top_scores, top_indices);
    const int current = i / B;
    const float* qi = q + static_cast<size_t>(i) * d;
    const float* go = d_out + static_cast<size_t>(i) * d;

    float maximum = -INFINITY;
    for (int block = 0; block <= current; ++block) {
        if (!sparse_block_selected(block, current, selected_count, top_indices,
                                   local_blocks, sink_blocks)) continue;
        const int begin = block * B;
        const int end = min((block + 1) * B, i + 1);
        for (int j = begin; j < end; ++j) {
            const float* kj = k + static_cast<size_t>(j) * d;
            float score = 0.0f;
            for (int c = 0; c < d; ++c) score += qi[c] * kj[c];
            maximum = fmaxf(maximum, score * scale);
        }
    }

    double denominator = 0.0;
    double weighted_upstream = 0.0;
    for (int block = 0; block <= current; ++block) {
        if (!sparse_block_selected(block, current, selected_count, top_indices,
                                   local_blocks, sink_blocks)) continue;
        const int begin = block * B;
        const int end = min((block + 1) * B, i + 1);
        for (int j = begin; j < end; ++j) {
            const float* kj = k + static_cast<size_t>(j) * d;
            const float* vj = v + static_cast<size_t>(j) * d;
            float score = 0.0f;
            float upstream = 0.0f;
            for (int c = 0; c < d; ++c) {
                score += qi[c] * kj[c];
                upstream += go[c] * vj[c];
            }
            const double exponential = exp(static_cast<double>(score * scale - maximum));
            denominator += exponential;
            weighted_upstream += exponential * upstream;
        }
    }
    if (!(denominator > 0.0)) return;
    const double softmax_dot = weighted_upstream / denominator;
    float* dqi = d_q + static_cast<size_t>(i) * d;
    for (int block = 0; block <= current; ++block) {
        if (!sparse_block_selected(block, current, selected_count, top_indices,
                                   local_blocks, sink_blocks)) continue;
        const int begin = block * B;
        const int end = min((block + 1) * B, i + 1);
        for (int j = begin; j < end; ++j) {
            const float* kj = k + static_cast<size_t>(j) * d;
            const float* vj = v + static_cast<size_t>(j) * d;
            float score = 0.0f;
            float upstream = 0.0f;
            for (int c = 0; c < d; ++c) {
                score += qi[c] * kj[c];
                upstream += go[c] * vj[c];
            }
            const double probability =
                exp(static_cast<double>(score * scale - maximum)) / denominator;
            const double d_score =
                probability * (static_cast<double>(upstream) - softmax_dot) * scale;
            for (int c = 0; c < d; ++c) {
                dqi[c] += static_cast<float>(d_score * kj[c]);
                atomicAdd(d_k + static_cast<size_t>(j) * d + c,
                          static_cast<float>(d_score * qi[c]));
                atomicAdd(d_v + static_cast<size_t>(j) * d + c,
                          static_cast<float>(probability * go[c]));
            }
        }
    }
}

__global__ void sparse_selective_attention_decode_kernel(
    const float* q_current, const float* k_cache, const float* v_cache,
    const float* route_current, const float* block_means, float* out,
    int cached_tokens, int d, int B, int top_k, int local_blocks,
    int sink_blocks, float scale) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || cached_tokens <= 0) return;
    const int i = cached_tokens - 1;
    float top_scores[kSparseTopKMax];
    int top_indices[kSparseTopKMax];
    const int selected_count = sparse_select_top_blocks(
        i, route_current, block_means, d, B, top_k, local_blocks,
        sink_blocks, scale, top_scores, top_indices);
    const int current = i / B;
    float accumulator[kSparseDimMax];
    for (int c = 0; c < d; ++c) accumulator[c] = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (int block = 0; block <= current; ++block) {
        if (!sparse_block_selected(block, current, selected_count, top_indices,
                                   local_blocks, sink_blocks)) continue;
        const int begin = block * B;
        const int end = min((block + 1) * B, cached_tokens);
        for (int j = begin; j < end; ++j) {
            const float* kj = k_cache + static_cast<size_t>(j) * d;
            const float* vj = v_cache + static_cast<size_t>(j) * d;
            float score = 0.0f;
            for (int c = 0; c < d; ++c) score += q_current[c] * kj[c];
            score *= scale;
            const float next_maximum = fmaxf(maximum, score);
            const float correction = expf(maximum - next_maximum);
            const float probability = expf(score - next_maximum);
            denominator = denominator * correction + probability;
            for (int c = 0; c < d; ++c) {
                accumulator[c] =
                    accumulator[c] * correction + probability * vj[c];
            }
            maximum = next_maximum;
        }
    }
    const float inverse = 1.0f / (denominator + 1e-20f);
    for (int c = 0; c < d; ++c) out[c] = accumulator[c] * inverse;
}

__global__ void sparse_selective_attention_backward_kernel(
    const float* q, const float* k, const float* v, const float* route,
    const float* bm, const float* d_out, float* d_q, float* d_k, float* d_v,
    int n, int d, int B, int top_k, int local_blocks, int sink_blocks,
    float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sparse_attention_backward_query(
        i, q, k, v, route, bm, d_out, d_q, d_k, d_v, n, d, B, top_k,
        local_blocks, sink_blocks, scale);
}

__global__ void sparse_selective_attention_backward_deterministic_kernel(
    const float* q, const float* k, const float* v, const float* route,
    const float* bm, const float* d_out, float* d_q, float* d_k, float* d_v,
    int n, int d, int B, int top_k, int local_blocks, int sink_blocks,
    float scale) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) sparse_attention_backward_query(
        i, q, k, v, route, bm, d_out, d_q, d_k, d_v, n, d, B, top_k,
        local_blocks, sink_blocks, scale);
}

__device__ void sparse_selector_distill_query(
    int i, const float* q, const float* k, const float* block_means,
    const float* w_selector, float* d_w, float* loss, float* count,
    int n, int d, int B, float scale) {
    (void)n;
    const int candidate_blocks = i / B + 1;
    if (candidate_blocks < 2 || candidate_blocks > kSparseBlockMax) return;
    const float* qi = q + static_cast<size_t>(i) * d;
    float dense_max = -INFINITY;
    for (int j = 0; j <= i; ++j) {
        const float* kj = k + static_cast<size_t>(j) * d;
        float score = 0.0f;
        for (int c = 0; c < d; ++c) score += qi[c] * kj[c];
        dense_max = fmaxf(dense_max, score * scale);
    }
    double dense_denominator = 0.0;
    for (int j = 0; j <= i; ++j) {
        const float* kj = k + static_cast<size_t>(j) * d;
        float score = 0.0f;
        for (int c = 0; c < d; ++c) score += qi[c] * kj[c];
        dense_denominator += exp(static_cast<double>(score * scale - dense_max));
    }

    float projected[kSparseDimMax];
    for (int a = 0; a < d; ++a) {
        double value = 0.0;
        for (int c = 0; c < d; ++c) {
            value += static_cast<double>(w_selector[static_cast<size_t>(a) * d + c]) *
                     qi[c];
        }
        projected[a] = static_cast<float>(value);
    }
    float selector_max = -INFINITY;
    for (int block = 0; block < candidate_blocks; ++block) {
        const float* mean = block_means + static_cast<size_t>(block) * d;
        float score = 0.0f;
        for (int a = 0; a < d; ++a) score += projected[a] * mean[a];
        selector_max = fmaxf(selector_max, score);
    }
    double selector_denominator = 0.0;
    for (int block = 0; block < candidate_blocks; ++block) {
        const float* mean = block_means + static_cast<size_t>(block) * d;
        float score = 0.0f;
        for (int a = 0; a < d; ++a) score += projected[a] * mean[a];
        selector_denominator += exp(static_cast<double>(score - selector_max));
    }
    if (!(dense_denominator > 0.0) || !(selector_denominator > 0.0)) return;

    float delta[kSparseBlockMax];
    double query_loss = 0.0;
    for (int block = 0; block < candidate_blocks; ++block) {
        double target_mass = 0.0;
        const int begin = block * B;
        const int end = min((block + 1) * B, i + 1);
        for (int j = begin; j < end; ++j) {
            const float* kj = k + static_cast<size_t>(j) * d;
            float score = 0.0f;
            for (int c = 0; c < d; ++c) score += qi[c] * kj[c];
            target_mass +=
                exp(static_cast<double>(score * scale - dense_max)) /
                dense_denominator;
        }
        const float* mean = block_means + static_cast<size_t>(block) * d;
        float selector_score = 0.0f;
        for (int a = 0; a < d; ++a) selector_score += projected[a] * mean[a];
        const double prediction =
            exp(static_cast<double>(selector_score - selector_max)) /
            selector_denominator;
        delta[block] = static_cast<float>(prediction - target_mass);
        if (target_mass > 0.0) {
            query_loss -= target_mass * log(fmax(prediction, 1e-20));
        }
    }
    for (int a = 0; a < d; ++a) {
        double projected_grad = 0.0;
        for (int block = 0; block < candidate_blocks; ++block) {
            projected_grad += static_cast<double>(delta[block]) *
                block_means[static_cast<size_t>(block) * d + a];
        }
        for (int c = 0; c < d; ++c) {
            atomicAdd(d_w + static_cast<size_t>(a) * d + c,
                      static_cast<float>(projected_grad * qi[c]));
        }
    }
    atomicAdd(loss, static_cast<float>(query_loss));
    atomicAdd(count, 1.0f);
}

__global__ void sparse_selector_distill_kernel(
    const float* q, const float* k, const float* block_means,
    const float* w_selector, float* d_w, float* loss, float* count,
    int n, int d, int B, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sparse_selector_distill_query(
        i, q, k, block_means, w_selector, d_w, loss, count, n, d, B, scale);
}

__global__ void sparse_selector_distill_deterministic_kernel(
    const float* q, const float* k, const float* block_means,
    const float* w_selector, float* d_w, float* loss, float* count,
    int n, int d, int B, float scale) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) sparse_selector_distill_query(
        i, q, k, block_means, w_selector, d_w, loss, count, n, d, B, scale);
}

}  // namespace

void launch_sparse_block_means(const float* k, float* bm, int n, int d, int B) {
    if (n <= 0 || d <= 0 || B <= 0) return;
    const int nb = gpu::ceil_div_positive(n, B);
    if (nb > INT_MAX / d) {
        throw std::overflow_error(
            "sparse block-means grid exceeds INT_MAX elements");
    }
    const int total = nb * d;
    const int blocks = gpu::ceil_div_positive(total, 256);
    sparse_block_means_kernel<<<blocks, 256, 0, nsos::gpu::current_stream()>>>(k, bm, n, d, B, nb);
}

void launch_sparse_selective_attention(const float* q, const float* k,
                                       const float* v, const float* route,
                                       const float* bm, float* out, int n, int d,
                                       int B, int top_k, int local_blocks,
                                       int sink_blocks, float scale) {
    if (n <= 0) return;
    const int blocks = gpu::ceil_div_positive(n, 128);
    sparse_selective_attention_kernel<<<blocks, 128, 0, nsos::gpu::current_stream()>>>(
        q, k, v, route, bm, out, n, d, B, top_k, local_blocks, sink_blocks, scale);
}

void launch_sparse_selective_attention_decode(
    const float* q_current, const float* k_cache, const float* v_cache,
    const float* route_current, const float* block_means, float* out,
    int cached_tokens, int d, int B, int top_k, int local_blocks,
    int sink_blocks, float scale) {
    if (cached_tokens <= 0 || d <= 0 || d > kSparseDimMax ||
        top_k < 0 || top_k > kSparseTopKMax) return;
    sparse_selective_attention_decode_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(
        q_current, k_cache, v_cache, route_current, block_means, out,
        cached_tokens, d, B, top_k, local_blocks, sink_blocks, scale);
}

void launch_sparse_selective_attention_backward(
    const float* q, const float* k, const float* v, const float* route,
    const float* bm, const float* d_out, float* d_q, float* d_k, float* d_v,
    int n, int d, int B, int top_k, int local_blocks, int sink_blocks,
    float scale, bool deterministic) {
    if (n <= 0 || d <= 0 || d > kSparseDimMax ||
        top_k < 0 || top_k > kSparseTopKMax) return;
    if (deterministic) {
        sparse_selective_attention_backward_deterministic_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(
            q, k, v, route, bm, d_out, d_q, d_k, d_v, n, d, B, top_k,
            local_blocks, sink_blocks, scale);
        return;
    }
    const int blocks = gpu::ceil_div_positive(n, 128);
    sparse_selective_attention_backward_kernel<<<blocks, 128, 0, nsos::gpu::current_stream()>>>(
        q, k, v, route, bm, d_out, d_q, d_k, d_v, n, d, B, top_k,
        local_blocks, sink_blocks, scale);
}

void launch_sparse_selector_distill(
    const float* q, const float* k, const float* block_means,
    const float* w_selector, float* d_w, float* loss, float* count,
    int n, int d, int B, float scale, bool deterministic) {
    if (n <= 0 || d <= 0 || d > kSparseDimMax || B <= 0) return;
    const int blocks_per_sequence = gpu::ceil_div_positive(n, B);
    if (blocks_per_sequence > kSparseBlockMax) return;
    if (deterministic) {
        sparse_selector_distill_deterministic_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(
            q, k, block_means, w_selector, d_w, loss, count, n, d, B, scale);
        return;
    }
    const int blocks = gpu::ceil_div_positive(n, 128);
    sparse_selector_distill_kernel<<<blocks, 128, 0, nsos::gpu::current_stream()>>>(
        q, k, block_means, w_selector, d_w, loss, count, n, d, B, scale);
}

}  // namespace cuda
}  // namespace nsos

#endif  // USE_CUDA
