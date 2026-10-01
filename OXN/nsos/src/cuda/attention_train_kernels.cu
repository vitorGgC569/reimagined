// attention_train_kernels.cu — GPU training path for GQA attention.
//
// WHY: the exact-cache Attention::backward in jamba.cpp ran ENTIRELY on the
// host — a single-threaded O(B·H·S²·hd) scalar loop fed by .cpu() copies of
// Q/K/V/dO (UM page migrations every step).  With 6 attention layers at
// batch 32 / seq 160 that host loop dominated the T4 step time.  The forward
// save path also did a full D2H + host KV-split + host RoPE per layer per
// step even when the fused GQA forward kernel produced the output on GPU.
//
// These kernels keep the WHOLE training path device-resident.  The heavy
// lifting (5 GEMM families) is done by Tensor::matmul's batched cuBLAS path
// (which also inherits the BF16 Tensor-Core mode); the kernels here are the
// cheap glue: head gather/expand (GQA), masked softmax fwd/bwd that mirrors
// the host math EXACTLY (same masks, same 1e-9 guard), RoPE rotation
// (forward and transpose), batched last-2-dim transpose, group reduction,
// and KV split/concat.  Parity vs the host path is gated by the local A/B
// (scripts/attn_bwd_parity.py) at 1e-3; NSOS_ATTN_BWD_HOST=1 restores the
// host path at runtime.
//
// All kernels are layout-exact to jamba.cpp:
//   q/dO heads:  [B, S, H,  hd]   k/v heads: [B, S, KV, hd]
//   permuted:    [B, H, S,  hd]   transposed: [B, H, hd, S]
//   scores/P/dS: [B*H, S, S]

#include "gpu_backend.h"
#include <climits>
#include <cmath>
#include <stdexcept>
#if defined(NSOS_GPU_BACKEND_CUDA)
#include <device_launch_parameters.h>
#endif

namespace {

constexpr int kThreads = 256;

inline int blocks_for(long long n) {
  const long long blocks = nsos::gpu::ceil_div_positive(
      n, static_cast<long long>(kThreads));
  if (blocks > INT_MAX) {
    throw std::overflow_error("attention GPU grid exceeds INT_MAX blocks");
  }
  return static_cast<int>(blocks);
}

// out[b,h,s,d] (or out[b,h,d,s] when transposed) = src[b,s,src_h,d]
// src_h = h when group <= 1 (pure permute), else min(h / group, src_heads-1)
// (GQA expansion: every query head reads its shared KV head).
__global__ void attn_gather_heads_kernel(float* __restrict__ out,
                                         const float* __restrict__ src,
                                         int B, int S, int H_out, int hd,
                                         int src_heads, int group,
                                         int transposed) {
  const long long total = static_cast<long long>(B) * H_out * S * hd;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % hd);
  long long rest = idx / hd;
  const int s = static_cast<int>(rest % S);
  rest /= S;
  const int h = static_cast<int>(rest % H_out);
  const int b = static_cast<int>(rest / H_out);
  int sh = h;
  if (group > 1) {
    sh = h / group;
    if (sh > src_heads - 1) sh = src_heads - 1;
  }
  const float v = src[(((static_cast<long long>(b) * S + s) * src_heads) + sh) * hd + d];
  if (transposed) {
    out[(((static_cast<long long>(b) * H_out + h) * hd) + d) * S + s] = v;
  } else {
    out[(((static_cast<long long>(b) * H_out + h) * S) + s) * hd + d] = v;
  }
}

// out[b,s,h,d] = src[b,h,s,d]
__global__ void attn_unpermute_heads_kernel(float* __restrict__ out,
                                            const float* __restrict__ src,
                                            int B, int H, int S, int hd) {
  const long long total = static_cast<long long>(B) * H * S * hd;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % hd);
  long long rest = idx / hd;
  const int s = static_cast<int>(rest % S);
  rest /= S;
  const int h = static_cast<int>(rest % H);
  const int b = static_cast<int>(rest / H);
  out[(((static_cast<long long>(b) * S + s) * H) + h) * hd + d] =
      src[(((static_cast<long long>(b) * H + h) * S) + s) * hd + d];
}

// out[b,s,kv,d] = sum_{g<group, kv*group+g < H} src[b, kv*group+g, s, d]
__global__ void attn_reduce_group_kernel(float* __restrict__ out,
                                         const float* __restrict__ src,
                                         int B, int S, int KV, int hd,
                                         int H, int group) {
  const long long total = static_cast<long long>(B) * S * KV * hd;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % hd);
  long long rest = idx / hd;
  const int kv = static_cast<int>(rest % KV);
  rest /= KV;
  const int s = static_cast<int>(rest % S);
  const int b = static_cast<int>(rest / S);
  // Mirror of the host mapping kv_head = min(h/group, KV-1): the LAST kv head
  // absorbs any trailing query heads when H is not an exact multiple of group.
  const int h_begin = kv * group;
  int h_end = h_begin + group;
  if (kv == KV - 1 || h_end > H) h_end = H;
  float acc = 0.0f;
  for (int h = h_begin; h < h_end; ++h) {
    acc += src[(((static_cast<long long>(b) * H + h) * S) + s) * hd + d];
  }
  out[idx] = acc;
}

// Masked softmax over scores rows, EXACTLY mirroring the host loop:
// row (b,h,i): score_j = scale * raw_j if (j <= i && j < valid_b && i < valid_b)
// else masked; probs = exp(score - max) / max(sum, 1e-9); masked entries 0.
// Rows with i >= valid_b come out all-zero (matches host rows left untouched).
__global__ void attn_masked_softmax_kernel(float* __restrict__ p,
                                           const int* __restrict__ valid,
                                           int B, int H, int S, float scale,
                                           int sliding_window) {
  const long long row = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long rows = static_cast<long long>(B) * H * S;
  if (row >= rows) return;
  const int i = static_cast<int>(row % S);
  const int b = static_cast<int>(row / (static_cast<long long>(H) * S));
  const int valid_len = valid[b];
  float* prow = p + row * S;
  if (i >= valid_len) {
    for (int j = 0; j < S; ++j) prow[j] = 0.0f;
    return;
  }
  const int limit = (i < valid_len - 1 ? i : valid_len - 1);
  const int begin = max(0, i - sliding_window + 1);
  float max_s = -3.0e38f;
  for (int j = begin; j <= limit; ++j) {
    const float sc = prow[j] * scale;
    if (sc > max_s) max_s = sc;
  }
  float sum = 0.0f;
  for (int j = begin; j <= limit; ++j) {
    const float e = expf(prow[j] * scale - max_s);
    prow[j] = e;
    sum += e;
  }
  const float inv = 1.0f / fmaxf(sum, 1e-9f);
  for (int j = 0; j < begin; ++j) prow[j] = 0.0f;
  for (int j = begin; j <= limit; ++j) prow[j] *= inv;
  for (int j = limit + 1; j < S; ++j) prow[j] = 0.0f;
}

// dS = scale * P .* (dP - rowdot),  rowdot = sum_j P_j * dP_j.
// (Host computes d_score = prob*(dprob - rowdot) and multiplies by `scale`
// inside the dQ/dK accumulation; folding scale here keeps the GEMMs plain.)
__global__ void attn_softmax_backward_kernel(float* __restrict__ ds,
                                             const float* __restrict__ p,
                                             const float* __restrict__ dp,
                                             int B, int H, int S, float scale) {
  const long long row = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long rows = static_cast<long long>(B) * H * S;
  if (row >= rows) return;
  const float* prow = p + row * S;
  const float* dprow = dp + row * S;
  float* dsrow = ds + row * S;
  float rowdot = 0.0f;
  for (int j = 0; j < S; ++j) rowdot += prow[j] * dprow[j];
  for (int j = 0; j < S; ++j) {
    dsrow[j] = scale * prow[j] * (dprow[j] - rowdot);
  }
}

// out[n, c, r] = src[n, r, c]
__global__ void batched_transpose_last2_kernel(float* __restrict__ out,
                                               const float* __restrict__ src,
                                               int N, int R, int C) {
  const long long total = static_cast<long long>(N) * R * C;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int c = static_cast<int>(idx % C);
  long long rest = idx / C;
  const int r = static_cast<int>(rest % R);
  const int n = static_cast<int>(rest / R);
  out[(static_cast<long long>(n) * C + c) * R + r] =
      src[(static_cast<long long>(n) * R + r) * C + c];
}

// In-place RoPE rotation on [B,S,H,hd] (hd even, half = hd/2).
// dir=+1 (forward):  x0' =  x0*cos - x1*sin ; x1' =  x0*sin + x1*cos
// dir=-1 (backward): x0' =  x0*cos + x1*sin ; x1' = -x0*sin + x1*cos
// pos = min(start_pos + s, max_seq - 1)   (identical clamp to the host).
__global__ void rope_apply_kernel(float* __restrict__ x,
                                  const float* __restrict__ cos_buf,
                                  const float* __restrict__ sin_buf,
                                  int B, int S, int H, int hd,
                                  int start_pos, int max_seq, int dir) {
  const int half = hd / 2;
  const long long total = static_cast<long long>(B) * S * H * half;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int i = static_cast<int>(idx % half);
  long long rest = idx / half;
  const int h = static_cast<int>(rest % H);
  rest /= H;
  const int s = static_cast<int>(rest % S);
  const int b = static_cast<int>(rest / S);
  int pos = start_pos + s;
  if (pos > max_seq - 1) pos = max_seq - 1;
  const float c = cos_buf[static_cast<long long>(pos) * half + i];
  const float sn = sin_buf[static_cast<long long>(pos) * half + i];
  float* base = x + (((static_cast<long long>(b) * S + s) * H) + h) * hd;
  const float x0 = base[i];
  const float x1 = base[i + half];
  if (dir >= 0) {
    base[i] = x0 * c - x1 * sn;
    base[i + half] = x0 * sn + x1 * c;
  } else {
    base[i] = x0 * c + x1 * sn;
    base[i + half] = -x0 * sn + x1 * c;
  }
}

// k[(b,s), 0:kvd] = kv[(b,s), 0:kvd] ; v[(b,s), 0:kvd] = kv[(b,s), kvd:2kvd]
__global__ void kv_split_kernel(float* __restrict__ k, float* __restrict__ v,
                                const float* __restrict__ kv,
                                long long rows, int kvd) {
  const long long total = rows * kvd;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int c = static_cast<int>(idx % kvd);
  const long long r = idx / kvd;
  k[idx] = kv[r * (2LL * kvd) + c];
  v[idx] = kv[r * (2LL * kvd) + kvd + c];
}

// out[(b,s), 0:kvd] = k ; out[(b,s), kvd:2kvd] = v
__global__ void kv_concat_kernel(float* __restrict__ out,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 long long rows, int kvd) {
  const long long total = rows * kvd;
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int c = static_cast<int>(idx % kvd);
  const long long r = idx / kvd;
  out[r * (2LL * kvd) + c] = k[idx];
  out[r * (2LL * kvd) + kvd + c] = v[idx];
}

// Logical subwaves of 32 work on both wave32 and wave64 HIP targets. No
// atomics: dQ is query-owned; dK/dV are KV-row-owned and reduce heads in order.
__device__ float tiled_sum32(float value) {
  for (int offset = 16; offset; offset >>= 1) {
#if defined(NSOS_GPU_BACKEND_HIP)
    value += __shfl_down(value, offset, 32);
#else
    value += __shfl_down_sync(0xffffffffu, value, offset, 32);
#endif
  }
  return value;
}

__device__ size_t attn_idx(int b, int s, int h, int S, int H, int hd) {
  return ((static_cast<size_t>(b) * S + s) * H + h) * hd;
}

__global__ void attn_tiled_forward_kernel(const float* q, const float* k,
    const float* v, const int* valid, float* out, float* lse,
    int S, int H, int KV, int hd, int group, int window, float scale) {
  __shared__ float scores[8], weights[8], alpha, maximum, denom;
  const int h = blockIdx.x % H, i = (blockIdx.x / H) % S;
  const int b = blockIdx.x / (H * S), kh = min(h / group, KV - 1);
  const int lane = threadIdx.x % 32, slot = threadIdx.x / 32, d = threadIdx.x;
  const size_t qi = attn_idx(b, i, h, S, H, hd);
  if (i >= valid[b]) {
    if (d < hd) out[qi + d] = 0.0f;
    if (d == 0) lse[blockIdx.x] = -INFINITY;
    return;
  }
  if (d == 0) { maximum = -INFINITY; denom = 0.0f; }
  __syncthreads();
  float accum = 0.0f;
  for (int begin = max(0, i - window + 1); begin <= i; begin += 8) {
    const int j = begin + slot;
    const size_t ki = attn_idx(b, min(j, i), kh, S, KV, hd);
    float dot = 0.0f;
    for (int x = lane; x < hd; x += 32) dot += q[qi + x] * k[ki + x];
    dot = tiled_sum32(dot);
    if (lane == 0) scores[slot] = j <= i ? dot * scale : -INFINITY;
    __syncthreads();
    if (d == 0) {
      float next = maximum;
      for (int z = 0; z < 8; ++z) next = fmaxf(next, scores[z]);
      alpha = expf(maximum - next);
      float sum = 0.0f;
      for (int z = 0; z < 8; ++z) { weights[z] = expf(scores[z] - next); sum += weights[z]; }
      denom = denom * alpha + sum;
      maximum = next;
    }
    __syncthreads();
    if (d < hd) {
      accum *= alpha;
      for (int z = 0; z < 8 && begin + z <= i; ++z)
        accum += weights[z] * v[attn_idx(b, begin + z, kh, S, KV, hd) + d];
    }
    __syncthreads();
  }
  if (d < hd) out[qi + d] = accum / denom;
  if (d == 0) lse[blockIdx.x] = maximum + logf(denom);
}

__global__ void attn_tiled_dq_kernel(const float* q, const float* k,
    const float* v, const float* out, const float* grad, const float* lse,
    const int* valid, float* delta, float* dq,
    int S, int H, int KV, int hd, int group, int window, float scale) {
  __shared__ float ds[8], rowdot;
  const int h = blockIdx.x % H, i = (blockIdx.x / H) % S;
  const int b = blockIdx.x / (H * S), kh = min(h / group, KV - 1);
  const int lane = threadIdx.x % 32, slot = threadIdx.x / 32, d = threadIdx.x;
  const size_t qi = attn_idx(b, i, h, S, H, hd);
  if (i >= valid[b]) {
    if (d < hd) dq[qi + d] = 0.0f;
    if (d == 0) delta[blockIdx.x] = 0.0f;
    return;
  }
  float dot = 0.0f;
  for (int x = lane; x < hd; x += 32) dot += grad[qi + x] * out[qi + x];
  dot = tiled_sum32(dot);
  if (d == 0) { rowdot = dot; delta[blockIdx.x] = dot; }
  __syncthreads();
  float accum = 0.0f;
  for (int begin = max(0, i - window + 1); begin <= i; begin += 8) {
    const int j = begin + slot;
    const size_t ki = attn_idx(b, min(j, i), kh, S, KV, hd);
    float score = 0.0f, dp = 0.0f;
    for (int x = lane; x < hd; x += 32) {
      score += q[qi + x] * k[ki + x];
      dp += grad[qi + x] * v[ki + x];
    }
    score = tiled_sum32(score); dp = tiled_sum32(dp);
    if (lane == 0) ds[slot] = j <= i
        ? expf(score * scale - lse[blockIdx.x]) * (dp - rowdot) * scale : 0.0f;
    __syncthreads();
    if (d < hd)
      for (int z = 0; z < 8 && begin + z <= i; ++z)
        accum += ds[z] * k[attn_idx(b, begin + z, kh, S, KV, hd) + d];
    __syncthreads();
  }
  if (d < hd) dq[qi + d] = accum;
}

__global__ void attn_tiled_dkv_kernel(const float* q, const float* k,
    const float* v, const float* grad, const float* lse, const int* valid,
    const float* delta, float* dk, float* dv,
    int S, int H, int KV, int hd, int group, int window, float scale) {
  __shared__ float ds[8], prob[8];
  const int kh = blockIdx.x % KV, j = (blockIdx.x / KV) % S;
  const int b = blockIdx.x / (KV * S);
  const int lane = threadIdx.x % 32, slot = threadIdx.x / 32, d = threadIdx.x;
  const size_t ki = attn_idx(b, j, kh, S, KV, hd);
  float ak = 0.0f, av = 0.0f;
  const int last = min(valid[b], j + min(window, S - j));
  const int head_end = kh == KV - 1 ? H : min(H, (kh + 1) * group);
  for (int h = kh * group; h < head_end; ++h) {
    for (int begin = j; begin < last; begin += 8) {
      const int i = begin + slot;
      const size_t qi = attn_idx(b, min(i, S - 1), h, S, H, hd);
      float score = 0.0f, dp = 0.0f;
      for (int x = lane; x < hd; x += 32) {
        score += q[qi + x] * k[ki + x];
        dp += grad[qi + x] * v[ki + x];
      }
      score = tiled_sum32(score); dp = tiled_sum32(dp);
      if (lane == 0) {
        const size_t row = (static_cast<size_t>(b) * S + min(i, S - 1)) * H + h;
        prob[slot] = i < last ? expf(score * scale - lse[row]) : 0.0f;
        ds[slot] = i < last ? prob[slot] * (dp - delta[row]) * scale : 0.0f;
      }
      __syncthreads();
      if (d < hd) {
        for (int z = 0; z < 8 && begin + z < last; ++z) {
          const size_t src = attn_idx(b, begin + z, h, S, H, hd) + d;
          ak += ds[z] * q[src];
          av += prob[z] * grad[src];
        }
      }
      __syncthreads();
    }
  }
  if (d < hd) { dk[ki + d] = ak; dv[ki + d] = av; }
}

}  // namespace

extern "C" bool launch_attn_tiled_forward(const float* q, const float* k, const float* v,
    const int* valid, float* out, float* lse, int B, int S, int H, int KV,
    int hd, int group, int window, float scale) {
  if (!q || !k || !v || !valid || !out || !lse || B <= 0 || S <= 0 || H <= 0 ||
      KV <= 0 || KV > H || hd <= 0 || hd > 256 || group <= 0 || group > H ||
      static_cast<long long>(KV - 1)*group > H || window <= 0 || !std::isfinite(scale) ||
      static_cast<long long>(B) * S * H > INT_MAX) return false;
  attn_tiled_forward_kernel<<<B*S*H, 256, 0, nsos::gpu::current_stream()>>>(
      q, k, v, valid, out, lse, S, H, KV, hd, group, window, scale);
  return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_attn_tiled_backward(const float* q, const float* k, const float* v,
    const float* out, const float* grad, const float* lse, const int* valid,
    float* delta, float* dq, float* dk, float* dv, int B, int S, int H,
    int KV, int hd, int group, int window, float scale) {
  if (!q || !k || !v || !out || !grad || !lse || !valid || !delta || !dq || !dk || !dv ||
      B <= 0 || S <= 0 || H <= 0 || KV <= 0 || KV > H || hd <= 0 || hd > 256 ||
      group <= 0 || group > H || static_cast<long long>(KV - 1)*group > H ||
      window <= 0 || !std::isfinite(scale) || static_cast<long long>(B)*S*H > INT_MAX) return false;
  const auto stream = nsos::gpu::current_stream();
  attn_tiled_dq_kernel<<<B*S*H, 256, 0, stream>>>(q, k, v, out, grad, lse, valid,
      delta, dq, S, H, KV, hd, group, window, scale);
  attn_tiled_dkv_kernel<<<B*S*KV, 256, 0, stream>>>(q, k, v, grad, lse, valid,
      delta, dk, dv, S, H, KV, hd, group, window, scale);
  return cudaGetLastError() == cudaSuccess;
}

extern "C" void launch_attn_gather_heads(float* out, const float* src, int B, int S,
                              int H_out, int hd, int src_heads, int group,
                              int transposed) {
  const long long total = static_cast<long long>(B) * H_out * S * hd;
  if (total <= 0) return;
  attn_gather_heads_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(
      out, src, B, S, H_out, hd, src_heads, group, transposed);
}

extern "C" void launch_attn_unpermute_heads(float* out, const float* src, int B, int H,
                                 int S, int hd) {
  const long long total = static_cast<long long>(B) * H * S * hd;
  if (total <= 0) return;
  attn_unpermute_heads_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(out, src, B, H, S, hd);
}

extern "C" void launch_attn_reduce_group(float* out, const float* src, int B, int S,
                              int KV, int hd, int H, int group) {
  const long long total = static_cast<long long>(B) * S * KV * hd;
  if (total <= 0) return;
  attn_reduce_group_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(out, src, B, S, KV, hd, H, group);
}

extern "C" void launch_attn_masked_softmax(float* p, const int* valid, int B, int H,
                                int S, float scale, int sliding_window) {
  const long long rows = static_cast<long long>(B) * H * S;
  if (rows <= 0) return;
  attn_masked_softmax_kernel<<<blocks_for(rows), kThreads, 0, nsos::gpu::current_stream()>>>(
      p, valid, B, H, S, scale, sliding_window);
}

extern "C" void launch_attn_softmax_backward(float* ds, const float* p, const float* dp,
                                  int B, int H, int S, float scale) {
  const long long rows = static_cast<long long>(B) * H * S;
  if (rows <= 0) return;
  attn_softmax_backward_kernel<<<blocks_for(rows), kThreads, 0, nsos::gpu::current_stream()>>>(ds, p, dp, B, H, S, scale);
}

extern "C" void launch_batched_transpose_last2(float* out, const float* src, int N,
                                    int R, int C) {
  const long long total = static_cast<long long>(N) * R * C;
  if (total <= 0) return;
  batched_transpose_last2_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(out, src, N, R, C);
}

extern "C" void launch_rope_apply(float* x, const float* cos_buf, const float* sin_buf,
                       int B, int S, int H, int hd, int start_pos, int max_seq,
                       int dir) {
  const int half = hd / 2;
  const long long total = static_cast<long long>(B) * S * H * half;
  if (total <= 0) return;
  rope_apply_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(x, cos_buf, sin_buf, B, S, H,
                                                     hd, start_pos, max_seq, dir);
}

extern "C" void launch_kv_split(float* k, float* v, const float* kv, long long rows, int kvd) {
  const long long total = rows * kvd;
  if (total <= 0) return;
  kv_split_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(k, v, kv, rows, kvd);
}

extern "C" void launch_kv_concat(float* out, const float* k, const float* v,
                      long long rows, int kvd) {
  const long long total = rows * kvd;
  if (total <= 0) return;
  kv_concat_kernel<<<blocks_for(total), kThreads, 0, nsos::gpu::current_stream()>>>(out, k, v, rows, kvd);
}
