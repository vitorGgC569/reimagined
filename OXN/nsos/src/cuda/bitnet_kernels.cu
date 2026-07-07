// =====================================================================
// BitNet 1.58-bit GPU dispatch: quantization + per-row dequant scaling.
//
// Composes with the existing `launch_bitnet_gemm` (in cuda/kernels.cu)
// to provide a callable GPU `__dp4a` path:
//
//   x (float [M, K])           --quantize_activations_bitnet_kernel-->
//   x_q (int8 [M, K]) + act_scales (float [M])
//                               --launch_bitnet_gemm (scale = weight_scale)-->
//   y_pre (float [M, N])
//   y_pre                       --bitnet_apply_act_scales_kernel-->
//   y (float [M, N])            (each row scaled by act_scales[row])
//
// CPU parity reference: BitLinear::quantize_activations_bitnet +
// BitNetAdapter::gemm_158bit_i8 in src/bitlinear.cpp.
//
// Target: NVIDIA GTX 1050 Ti (sm_61) and newer.  __dp4a path only used
// downstream — these kernels are pure fp32/int8 elementwise/reduction.
// =====================================================================

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BITNET_WARP_SIZE 32
#define BITNET_FULL_MASK 0xFFFFFFFFu

namespace {

__device__ __forceinline__ float warp_reduce_max_abs(float v) {
#pragma unroll
  for (int offset = BITNET_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(BITNET_FULL_MASK, v, offset);
    v = fmaxf(v, other);
  }
  return v;
}

}  // namespace

// =====================================================================
// Per-row activation quantization for BitNet 1.58-bit / INT-N.
//
// For each row of `x[M, K]`:
//   max_val = max_j |x[row, j]|
//   scale   = q_max / (max_val + 1e-8)
//   x_q[row, j] = round(x[row, j] * scale)            (clipped to int8)
//   act_scales[row] = (max_val + 1e-8) / q_max         (== 1/scale)
//
// `precision_bits` selects q_max:
//   * precision_bits == 2 → q_max = 1.0  (ternary -1/0/+1)
//   * otherwise          → q_max = 2^(precision_bits-1) - 1
//
// One thread block per row.  Block uses warp-reduce + __shared__ to
// get the row max, then each thread quantizes its strided slice.
// =====================================================================
__global__ void quantize_activations_bitnet_kernel(const float *__restrict__ x,
                                                   int8_t *__restrict__ x_q,
                                                   float *__restrict__ act_scales,
                                                   int M, int K,
                                                   int precision_bits) {
  const int row = blockIdx.x;
  if (row >= M) return;

  const float *row_in = x + row * K;
  int8_t *row_out = x_q + row * K;

  // Phase 1: row max(|x|)
  float partial = 0.0f;
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    partial = fmaxf(partial, fabsf(row_in[j]));
  }

  // Block-wide reduction (assumes blockDim.x % WARP_SIZE == 0; the
  // launcher guarantees this by clamping to a multiple of 32).
  __shared__ float warp_max[BITNET_WARP_SIZE];
  const int lane = threadIdx.x % BITNET_WARP_SIZE;
  const int warp_id = threadIdx.x / BITNET_WARP_SIZE;

  partial = warp_reduce_max_abs(partial);
  if (lane == 0) warp_max[warp_id] = partial;
  __syncthreads();

  const int num_warps = (blockDim.x + BITNET_WARP_SIZE - 1) / BITNET_WARP_SIZE;
  float max_val = (threadIdx.x < num_warps) ? warp_max[threadIdx.x] : 0.0f;
  if (warp_id == 0) max_val = warp_reduce_max_abs(max_val);

  // Broadcast max_val and the derived scales via shared memory.
  __shared__ float s_inv_scale;  // == 1/scale, written to act_scales[row]
  __shared__ float s_scale;
  if (threadIdx.x == 0) {
    const float q_max =
        (precision_bits <= 2)
            ? 1.0f
            : (powf(2.0f, static_cast<float>(precision_bits - 1)) - 1.0f);
    const float denom = max_val + 1e-8f;
    s_scale = q_max / denom;
    s_inv_scale = denom / q_max;
    act_scales[row] = s_inv_scale;
  }
  __syncthreads();

  const float scale = s_scale;
  const float clip_lo = (precision_bits <= 2) ? -1.0f : -127.0f;
  const float clip_hi = (precision_bits <= 2) ? 1.0f : 127.0f;

  // Phase 2: quantize.  rintf rounds to nearest, ties to even — the
  // CPU reference uses std::round which rounds half-away-from-zero.
  // Drift is bounded by 0.5 * scale per element and absorbed by the
  // per-row act_scales factor on dequant; tests use 2e-3 tolerance.
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    float v = row_in[j] * scale;
    v = fminf(fmaxf(v, clip_lo), clip_hi);
    row_out[j] = static_cast<int8_t>(rintf(v));
  }
}

// =====================================================================
// In-place per-row dequantization scaling.
//
// For each (row, col): y[row, col] *= act_scales[row].
// The bitnet_gemm_kernel emits y_pre = (acc_int32 * weight_scale); this
// kernel folds in the per-row activation scale to obtain the final fp32
// output that matches BitLinear::gemm_158bit_ultra on CPU.
// =====================================================================
__global__ void bitnet_apply_act_scales_kernel(float *__restrict__ y,
                                               const float *__restrict__ act_scales,
                                               int M, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (idx >= total) return;

  const int row = idx / N;
  y[idx] *= act_scales[row];
}

// =====================================================================
// K3: GPU QAT fake-quant (straight-through estimator) kernels.
// =====================================================================

// Weight ternary fake-quant: out = clamp(round(w/scale), -1, +1) * scale.
__global__ void fake_quant_ternary_kernel(float *__restrict__ out,
                                          const float *__restrict__ w,
                                          float scale, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float inv = 1.0f / (scale + 1e-8f);
  const float v = w[i] * inv;
  const float t = (v > 0.5f) ? 1.0f : ((v < -0.5f) ? -1.0f : 0.0f);
  out[i] = t * scale;
}

// Same rule, but the absmean scale is computed from a device-resident
// reduction result.  This keeps QAT on the GPU hot path free of D2H syncs.
__global__ void fake_quant_ternary_absmean_kernel(
    float *__restrict__ out, float *__restrict__ scale_out,
    const float *__restrict__ w, const float *__restrict__ abs_sum, int n) {
  if (n <= 0) return;
  const float scale = abs_sum[0] / static_cast<float>(n) + 1e-8f;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    scale_out[0] = scale;
  }
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float inv = 1.0f / (scale + 1e-8f);
  const float v = w[i] * inv;
  const float t = (v > 0.5f) ? 1.0f : ((v < -0.5f) ? -1.0f : 0.0f);
  out[i] = t * scale;
}

// Per-row activation fake-quant (quant→dequant fused): one block per row.
__global__ void fake_quant_activations_kernel(float *__restrict__ out,
                                              const float *__restrict__ x,
                                              int M, int K, int precision_bits) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const float *row_in = x + row * K;
  float *row_out = out + row * K;

  float partial = 0.0f;
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    partial = fmaxf(partial, fabsf(row_in[j]));
  }
  __shared__ float warp_max[BITNET_WARP_SIZE];
  const int lane = threadIdx.x % BITNET_WARP_SIZE;
  const int warp_id = threadIdx.x / BITNET_WARP_SIZE;
  partial = warp_reduce_max_abs(partial);
  if (lane == 0) warp_max[warp_id] = partial;
  __syncthreads();
  const int num_warps = (blockDim.x + BITNET_WARP_SIZE - 1) / BITNET_WARP_SIZE;
  float max_val = (threadIdx.x < num_warps) ? warp_max[threadIdx.x] : 0.0f;
  if (warp_id == 0) max_val = warp_reduce_max_abs(max_val);

  __shared__ float s_scale;     // q_max / (max+eps)
  __shared__ float s_inv_scale; // (max+eps) / q_max
  if (threadIdx.x == 0) {
    const float q_max = (precision_bits <= 2)
                            ? 1.0f
                            : (powf(2.0f, static_cast<float>(precision_bits - 1)) - 1.0f);
    const float denom = max_val + 1e-8f;
    s_scale = q_max / denom;
    s_inv_scale = denom / q_max;
  }
  __syncthreads();
  const float scale = s_scale;
  const float inv_scale = s_inv_scale;
  const float clip = (precision_bits <= 2)
                         ? 1.0f
                         : (powf(2.0f, static_cast<float>(precision_bits - 1)) - 1.0f);
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    float v = row_in[j] * scale;
    v = fminf(fmaxf(v, -clip), clip);
    row_out[j] = rintf(v) * inv_scale;  // dequantized
  }
}

// STE clip: zero dW where the latent weight already saturated past |w/scale|>1.
__global__ void ste_clip_weight_grad_kernel(float *__restrict__ dW,
                                            const float *__restrict__ w,
                                            float scale, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float inv = 1.0f / (scale + 1e-8f);
  if (fabsf(w[i] * inv) > 1.0f) dW[i] = 0.0f;
}

__global__ void ste_clip_weight_grad_device_scale_kernel(
    float *__restrict__ dW, const float *__restrict__ w,
    const float *__restrict__ scale_dev, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float scale = scale_dev[0];
  const float inv = 1.0f / (scale + 1e-8f);
  if (fabsf(w[i] * inv) > 1.0f) dW[i] = 0.0f;
}

extern "C" {

void launch_fake_quant_ternary_kernel(float *out, const float *w, float scale,
                                      int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fake_quant_ternary_kernel<<<blocks, threads>>>(out, w, scale, n);
}

void launch_fake_quant_ternary_absmean_kernel(float *out, float *scale_out,
                                              const float *w,
                                              const float *abs_sum, int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fake_quant_ternary_absmean_kernel<<<blocks, threads>>>(out, scale_out, w,
                                                         abs_sum, n);
}

void launch_fake_quant_activations_kernel(float *out, const float *x, int M,
                                          int K, int precision_bits) {
  if (M <= 0 || K <= 0) return;
  int threads = (K + BITNET_WARP_SIZE - 1) / BITNET_WARP_SIZE * BITNET_WARP_SIZE;
  if (threads < BITNET_WARP_SIZE) threads = BITNET_WARP_SIZE;
  if (threads > 256) threads = 256;
  fake_quant_activations_kernel<<<M, threads>>>(out, x, M, K, precision_bits);
}

void launch_ste_clip_weight_grad_kernel(float *dW, const float *w, float scale,
                                        int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  ste_clip_weight_grad_kernel<<<blocks, threads>>>(dW, w, scale, n);
}

void launch_ste_clip_weight_grad_device_scale_kernel(float *dW, const float *w,
                                                     const float *scale,
                                                     int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  ste_clip_weight_grad_device_scale_kernel<<<blocks, threads>>>(dW, w, scale,
                                                                n);
}

void launch_quantize_activations_bitnet_kernel(const float *x, int8_t *x_q,
                                               float *act_scales, int M, int K,
                                               int precision_bits) {
  if (M <= 0 || K <= 0) return;
  // Round threads up to a multiple of WARP_SIZE so the warp reduction
  // is well-formed; cap at 256 so shared/register pressure stays small
  // on Pascal SMs.
  int threads = (K + BITNET_WARP_SIZE - 1) / BITNET_WARP_SIZE * BITNET_WARP_SIZE;
  if (threads < BITNET_WARP_SIZE) threads = BITNET_WARP_SIZE;
  if (threads > 256) threads = 256;
  quantize_activations_bitnet_kernel<<<M, threads>>>(x, x_q, act_scales, M, K,
                                                     precision_bits);
}

void launch_bitnet_apply_act_scales_kernel(float *y, const float *act_scales,
                                           int M, int N) {
  if (M <= 0 || N <= 0) return;
  const int total = M * N;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  bitnet_apply_act_scales_kernel<<<blocks, threads>>>(y, act_scales, M, N);
}

}  // extern "C"
