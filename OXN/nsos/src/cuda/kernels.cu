#include "cuda/kernels.cuh"
#include "tensor.h"
#include "gpu_execution.h"

#include <cmath>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "gpu_backend.h"
#if defined(NSOS_GPU_BACKEND_CUDA)
#include <device_launch_parameters.h>
#endif
#include <vector>

// =========================================================================
// HPC-Optimized CUDA Kernels for NSOS/OXN
// Target: NVIDIA GTX 1050 Ti (SM 6.1, Pascal)
// All kernels use warp-level primitives and shared memory tiling.
// =========================================================================

// Compile-time constants
#define TILE_DIM 16

namespace {

// Reductions use a full 256-thread block.  Besides matching the hot NSOS
// widths (d_model/vocabulary), this guarantees at least one complete native
// wavefront on HIP wave32 and wave64.  The device primitives below reduce over
// the backend's native `warpSize`; no CUDA warp-32 assumption is made.
constexpr int kReductionThreads = 256;

bool checked_positive_product_to_int(int lhs, int rhs, int *result) {
  if (result == nullptr || lhs <= 0 || rhs <= 0 || lhs > INT_MAX / rhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_positive_product_to_int(int first, int second, int third,
                                     int *result) {
  int partial = 0;
  return checked_positive_product_to_int(first, second, &partial) &&
         checked_positive_product_to_int(partial, third, result);
}

__device__ __forceinline__ float nsos_shfl_down(float value, int offset) {
#if defined(NSOS_GPU_BACKEND_HIP)
  return __shfl_down(value, offset);
#else
  return __shfl_down_sync(0xFFFFFFFFu, value, offset);
#endif
}

}  // namespace

template <bool CombineThree>
__global__ void tensor_audit_stats_kernel(
    NsosTensorAuditDeviceStats *out, const float *first,
    const float *second, const float *third, long long count) {
  constexpr int kThreads = 256;
  __shared__ double minimum[kThreads];
  __shared__ double maximum[kThreads];
  __shared__ double sums[kThreads];
  __shared__ double sum_squares[kThreads];
  __shared__ double max_absolute[kThreads];
  __shared__ uint64_t finite_counts[kThreads];
  __shared__ uint64_t nan_counts[kThreads];
  __shared__ uint64_t inf_counts[kThreads];
  __shared__ uint64_t zero_counts[kThreads];
  __shared__ uint64_t subnormal_counts[kThreads];
  __shared__ uint64_t positive_counts[kThreads];
  __shared__ uint64_t negative_counts[kThreads];

  const int lane = threadIdx.x;
  double local_minimum = INFINITY;
  double local_maximum = -INFINITY;
  double local_sum = 0.0;
  double local_sum_squares = 0.0;
  double local_max_absolute = 0.0;
  uint64_t local_finite = 0;
  uint64_t local_nan = 0;
  uint64_t local_inf = 0;
  uint64_t local_zero = 0;
  uint64_t local_subnormal = 0;
  uint64_t local_positive = 0;
  uint64_t local_negative = 0;

  for (long long index = lane; index < count;
       index += blockDim.x) {
    float value = first[index];
    if constexpr (CombineThree) {
      value += second[index] + third[index];
    }
    if (isnan(value)) {
      ++local_nan;
      continue;
    }
    if (isinf(value)) {
      ++local_inf;
      continue;
    }
    if (value == 0.0f) {
      ++local_zero;
    } else {
      if (value > 0.0f) {
        ++local_positive;
      } else {
        ++local_negative;
      }
      if (fabsf(value) < FLT_MIN) {
        ++local_subnormal;
      }
    }
    const double converted = static_cast<double>(value);
    local_sum += converted;
    local_sum_squares += converted * converted;
    local_minimum = fmin(local_minimum, converted);
    local_maximum = fmax(local_maximum, converted);
    local_max_absolute =
        fmax(local_max_absolute, fabs(converted));
    ++local_finite;
  }

  minimum[lane] = local_minimum;
  maximum[lane] = local_maximum;
  sums[lane] = local_sum;
  sum_squares[lane] = local_sum_squares;
  max_absolute[lane] = local_max_absolute;
  finite_counts[lane] = local_finite;
  nan_counts[lane] = local_nan;
  inf_counts[lane] = local_inf;
  zero_counts[lane] = local_zero;
  subnormal_counts[lane] = local_subnormal;
  positive_counts[lane] = local_positive;
  negative_counts[lane] = local_negative;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      minimum[lane] = fmin(minimum[lane], minimum[lane + stride]);
      maximum[lane] = fmax(maximum[lane], maximum[lane + stride]);
      sums[lane] += sums[lane + stride];
      sum_squares[lane] += sum_squares[lane + stride];
      max_absolute[lane] =
          fmax(max_absolute[lane], max_absolute[lane + stride]);
      finite_counts[lane] += finite_counts[lane + stride];
      nan_counts[lane] += nan_counts[lane + stride];
      inf_counts[lane] += inf_counts[lane + stride];
      zero_counts[lane] += zero_counts[lane + stride];
      subnormal_counts[lane] += subnormal_counts[lane + stride];
      positive_counts[lane] += positive_counts[lane + stride];
      negative_counts[lane] += negative_counts[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    out->min_value = minimum[0];
    out->max_value = maximum[0];
    out->sum = sums[0];
    out->sum_sq = sum_squares[0];
    out->max_abs = max_absolute[0];
    out->finite_count = finite_counts[0];
    out->nan_count = nan_counts[0];
    out->inf_count = inf_counts[0];
    out->zero_count = zero_counts[0];
    out->subnormal_count = subnormal_counts[0];
    out->positive_count = positive_counts[0];
    out->negative_count = negative_counts[0];
  }
}

__global__ void hybrid_audit_pair_dots_kernel(
    double *out, const float *mamba_signal,
    const float *attention_signal, const float *ffn_signal,
    const float *mamba_contribution,
    const float *attention_contribution,
    const float *ffn_contribution, long long count) {
  constexpr int kThreads = 256;
  __shared__ double partial[6][kThreads];
  const int lane = threadIdx.x;
  double values[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (long long index = lane; index < count;
       index += blockDim.x) {
    const double ms = static_cast<double>(mamba_signal[index]);
    const double as = static_cast<double>(attention_signal[index]);
    const double fs = static_cast<double>(ffn_signal[index]);
    const double mc =
        static_cast<double>(mamba_contribution[index]);
    const double ac =
        static_cast<double>(attention_contribution[index]);
    const double fc =
        static_cast<double>(ffn_contribution[index]);
    values[0] += ms * as;
    values[1] += ms * fs;
    values[2] += as * fs;
    values[3] += mc * ac;
    values[4] += mc * fc;
    values[5] += ac * fc;
  }
  for (int metric = 0; metric < 6; ++metric) {
    partial[metric][lane] = values[metric];
  }
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      for (int metric = 0; metric < 6; ++metric) {
        partial[metric][lane] +=
            partial[metric][lane + stride];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    for (int metric = 0; metric < 6; ++metric) {
      out[metric] = partial[metric][0];
    }
  }
}

extern "C" void launch_tensor_audit_stats_kernel(
    NsosTensorAuditDeviceStats *out, const float *values,
    long long count) {
  tensor_audit_stats_kernel<false><<<1, 256, 0, nsos::gpu::current_stream()>>>(
      out, values, nullptr, nullptr, count);
}

extern "C" void launch_hybrid_audit_metrics_kernel(
    NsosHybridAuditDeviceMetrics *out,
    const float *mamba_signal, const float *attention_signal,
    const float *ffn_signal, const float *mamba_contribution,
    const float *attention_contribution,
    const float *ffn_contribution, long long count) {
  const float *values[6] = {
      mamba_signal, attention_signal, ffn_signal,
      mamba_contribution, attention_contribution,
      ffn_contribution};
  for (int index = 0; index < 6; ++index) {
    tensor_audit_stats_kernel<false><<<1, 256, 0, nsos::gpu::current_stream()>>>(
        &out->tensor_stats[index], values[index],
        nullptr, nullptr, count);
  }
  tensor_audit_stats_kernel<true><<<1, 256, 0, nsos::gpu::current_stream()>>>(
      &out->tensor_stats[6], mamba_contribution,
      attention_contribution, ffn_contribution, count);
  hybrid_audit_pair_dots_kernel<<<1, 256, 0, nsos::gpu::current_stream()>>>(
      out->pair_dots, mamba_signal, attention_signal,
      ffn_signal, mamba_contribution,
      attention_contribution, ffn_contribution, count);
}

// -------------------------------------------------------------------------
// Warp-level reduction primitives
// -------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += nsos_shfl_down(val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    float other = nsos_shfl_down(val, offset);
    val = fmaxf(val, other);
  }
  return val;
}

// Block-wide reduction using shared memory (for blocks larger than 1 warp)
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[32]; // one per warp
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  val = warp_reduce_sum(val);

  if (lane == 0)
    shared[warp_id] = val;
  __syncthreads();

  // First warp reduces across warp results
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
  if (warp_id == 0)
    val = warp_reduce_sum(val);

  return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  val = warp_reduce_max(val);

  if (lane == 0)
    shared[warp_id] = val;
  __syncthreads();

  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -1e30f;
  if (warp_id == 0)
    val = warp_reduce_max(val);

  return val;
}

// -------------------------------------------------------------------------
// Element-wise Kernels (unchanged — already efficient)
// -------------------------------------------------------------------------

__global__ void add_kernel(float *out, const float *a, const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

extern "C" void launch_add_kernel(float *out, const float *a, const float *b,
                                  int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  add_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, a, b, n);
}

__global__ void add_broadcast_kernel(float *out, const float *a, const float *b,
                                     int n, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx % stride];
  }
}

extern "C" void launch_add_broadcast_kernel(float *out, const float *a,
                                            const float *b, int n, int stride) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  add_broadcast_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, a, b, n, stride);
}

__global__ void add_trailing_broadcast_kernel(
    float *out, const float *a, const float *b, int n, int width) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx / width];
  }
}

extern "C" bool launch_add_trailing_broadcast_kernel(
    float *out, const float *a, const float *b, int n, int width) {
  if (out == nullptr || a == nullptr || b == nullptr || n <= 0 ||
      width <= 0 || n % width != 0) {
    return false;
  }
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  add_trailing_broadcast_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      out, a, b, n, width);
  return true;
}

__global__ void sub_kernel(float *out, const float *a, const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] - b[idx];
  }
}

extern "C" void launch_sub_kernel(float *out, const float *a, const float *b,
                                  int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  sub_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, a, b, n);
}

__global__ void sub_broadcast_kernel(float *out, const float *a,
                                     const float *b, int n, int stride,
                                     bool reverse) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float broadcast = b[idx % stride];
    out[idx] = reverse ? broadcast - a[idx] : a[idx] - broadcast;
  }
}

extern "C" bool launch_sub_broadcast_kernel(
    float *out, const float *a, const float *b, int n, int stride,
    int reverse) {
  if (out == nullptr || a == nullptr || b == nullptr || n <= 0 ||
      stride <= 0 || n % stride != 0 || (reverse != 0 && reverse != 1)) {
    return false;
  }
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  sub_broadcast_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      out, a, b, n, stride, reverse);
  return true;
}

__global__ void sub_trailing_broadcast_kernel(
    float *out, const float *a, const float *b, int n, int width,
    bool reverse) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float broadcast = b[idx / width];
    out[idx] = reverse ? broadcast - a[idx] : a[idx] - broadcast;
  }
}

extern "C" bool launch_sub_trailing_broadcast_kernel(
    float *out, const float *a, const float *b, int n, int width,
    int reverse) {
  if (out == nullptr || a == nullptr || b == nullptr || n <= 0 ||
      width <= 0 || n % width != 0 || (reverse != 0 && reverse != 1)) {
    return false;
  }
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  sub_trailing_broadcast_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      out, a, b, n, width, reverse);
  return true;
}

__global__ void mul_scalar_kernel(float *out, const float *a, float scalar,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] * scalar;
  }
}

extern "C" void launch_mul_scalar_kernel(float *out, const float *a,
                                         float scalar, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  mul_scalar_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, a, scalar, n);
}

__global__ void mul_tensor_kernel(float *out, const float *a, const float *b,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] * b[idx];
  }
}

extern "C" void launch_mul_tensor_kernel(float *out, const float *a,
                                         const float *b, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  mul_tensor_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, a, b, n);
}

__device__ __forceinline__ float stable_sigmoid_device(float value) {
  if (value >= 0.0f) {
    const float exponential = expf(-value);
    return 1.0f / (1.0f + exponential);
  }
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

__global__ void sigmoid_kernel(float *out, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = stable_sigmoid_device(in[idx]);
  }
}

extern "C" void launch_sigmoid_kernel(float *out, const float *in, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  sigmoid_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, in, n);
}

__global__ void silu_kernel(float *out, const float *in, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float value = in[index];
    out[index] = value * stable_sigmoid_device(value);
  }
}

extern "C" void launch_silu_kernel(float *out, const float *in, int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(n, threads);
  silu_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, in, n);
}

__global__ void silu_backward_kernel(
    float *in_grad, const float *grad_out,
    const float *pre_activation, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float value = pre_activation[index];
    const float sigmoid = stable_sigmoid_device(value);
    in_grad[index] =
        grad_out[index] * sigmoid *
        (1.0f + value * (1.0f - sigmoid));
  }
}

extern "C" void launch_silu_backward_kernel(
    float *in_grad, const float *grad_out,
    const float *pre_activation, int n) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(n, threads);
  silu_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      in_grad, grad_out, pre_activation, n);
}

__global__ void silu_gate_forward_kernel(
    float *out, const float *value, const float *gate, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  const float gate_value = gate[index];
  const float activated_gate =
      gate_value * stable_sigmoid_device(gate_value);
  out[index] = value[index] * activated_gate;
}

extern "C" bool launch_silu_gate_forward_kernel(
    float *out, const float *value, const float *gate, int n) {
  if (out == nullptr || value == nullptr || gate == nullptr || n <= 0) {
    return false;
  }
  constexpr int threads = 256;
  silu_gate_forward_kernel<<<nsos::gpu::ceil_div_positive(n, threads), threads, 0, nsos::gpu::current_stream()>>>(
      out, value, gate, n);
  return true;
}

__global__ void silu_gate_backward_kernel(
    float *value_grad, float *gate_grad, const float *grad_out,
    const float *value, const float *gate, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  const float gate_value = gate[index];
  const float sigmoid = stable_sigmoid_device(gate_value);
  const float activated_gate = gate_value * sigmoid;
  const float upstream_gate = grad_out[index] * value[index];
  value_grad[index] = grad_out[index] * activated_gate;
  gate_grad[index] =
      upstream_gate * sigmoid *
      (1.0f + gate_value * (1.0f - sigmoid));
}

extern "C" bool launch_silu_gate_backward_kernel(
    float *value_grad, float *gate_grad, const float *grad_out,
    const float *value, const float *gate, int n) {
  if (value_grad == nullptr || gate_grad == nullptr || grad_out == nullptr ||
      value == nullptr || gate == nullptr || n <= 0) {
    return false;
  }
  constexpr int threads = 256;
  silu_gate_backward_kernel<<<nsos::gpu::ceil_div_positive(n, threads), threads, 0, nsos::gpu::current_stream()>>>(
      value_grad, gate_grad, grad_out, value, gate, n);
  return true;
}

__global__ void scale_inplace_kernel(float *data, float scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

extern "C" void launch_scale_inplace_kernel(float *data, float scale, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  scale_inplace_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(data, scale, n);
}

// Broadcast Mul: [..., D] * [..., 1]
__global__ void mul_broadcast_kernel(float *out, const float *a, const float *b,
                                     int n, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] * b[idx / D];
  }
}

extern "C" void launch_mul_broadcast_kernel(float *out, const float *a,
                                            const float *b, int n, int D) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  mul_broadcast_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, a, b, n, D);
}

__global__ void mul_vector_broadcast_kernel(float *out, const float *in,
                                            const float *vec, int rows,
                                            int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx < total) {
    out[idx] = in[idx] * vec[idx % cols];
  }
}

extern "C" void launch_mul_vector_broadcast_kernel(float *out, const float *in,
                                                   const float *vec, int rows,
                                                   int cols) {
  int total = rows * cols;
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mul_vector_broadcast_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, in, vec, rows, cols);
}

__global__ void embedding_gather_kernel(float *out, const float *weight,
                                        const int *ids, int total_positions,
                                        int vocab_size, int embedding_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = total_positions * embedding_dim;
  if (idx >= total) {
    return;
  }

  int position = idx / embedding_dim;
  int dim = idx % embedding_dim;
  int token_id = ids[position];
  if (token_id < 0 || token_id >= vocab_size) {
    out[idx] = 0.0f;
    return;
  }

  out[idx] = weight[token_id * embedding_dim + dim];
}

extern "C" void launch_embedding_gather_kernel(float *out, const float *weight,
                                               const int *ids,
                                               int total_positions,
                                               int vocab_size,
                                               int embedding_dim) {
  int total = total_positions * embedding_dim;
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(total, threads);
  embedding_gather_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, weight, ids, total_positions,
                                               vocab_size, embedding_dim);
}

__global__ void embedding_scatter_add_kernel(float *grad_weight,
                                             const float *grad_output,
                                             const int *ids,
                                             int total_positions,
                                             int vocab_size,
                                             int embedding_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = total_positions * embedding_dim;
  if (idx >= total) {
    return;
  }

  int position = idx / embedding_dim;
  int dim = idx % embedding_dim;
  int token_id = ids[position];
  if (token_id < 0 || token_id >= vocab_size) {
    return;
  }

  atomicAdd(&grad_weight[token_id * embedding_dim + dim], grad_output[idx]);
}

extern "C" void launch_embedding_scatter_add_kernel(float *grad_weight,
                                                    const float *grad_output,
                                                    const int *ids,
                                                    int total_positions,
                                                    int vocab_size,
                                                    int embedding_dim) {
  int total = total_positions * embedding_dim;
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(total, threads);
  embedding_scatter_add_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      grad_weight, grad_output, ids, total_positions, vocab_size, embedding_dim);
}

// One thread owns one output element and visits positions in a fixed order.
// Unlike the faster scatter kernel above, this gather-style reduction uses no
// atomics, so repeated token ids produce bitwise-stable gradients.
__global__ void embedding_scatter_add_deterministic_kernel(
    float *grad_weight, const float *grad_output, const int *ids,
    int total_positions, int vocab_size, int embedding_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = vocab_size * embedding_dim;
  if (idx >= total) {
    return;
  }

  const int token_id = idx / embedding_dim;
  const int dim = idx % embedding_dim;
  float sum = 0.0f;
  for (int position = 0; position < total_positions; ++position) {
    if (ids[position] == token_id) {
      sum += grad_output[position * embedding_dim + dim];
    }
  }
  grad_weight[idx] = sum;
}

extern "C" void launch_embedding_scatter_add_deterministic_kernel(
    float *grad_weight, const float *grad_output, const int *ids,
    int total_positions, int vocab_size, int embedding_dim) {
  const int total = vocab_size * embedding_dim;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  embedding_scatter_add_deterministic_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      grad_weight, grad_output, ids, total_positions, vocab_size, embedding_dim);
}

__global__ void embedding_scatter_add_deterministic_sparse_kernel(
    float *grad_weight, const float *grad_output,
    const int *unique_ids, const int *offsets, const int *positions,
    int unique_count, int total_positions, int vocab_size,
    int embedding_dim) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total =
      static_cast<long long>(unique_count) * embedding_dim;
  if (idx >= total) {
    return;
  }

  const int unique_index = static_cast<int>(idx / embedding_dim);
  const int dim = static_cast<int>(idx % embedding_dim);
  const int token_id = unique_ids[unique_index];
  const int begin = offsets[unique_index];
  const int end = offsets[unique_index + 1];
  if (token_id < 0 || token_id >= vocab_size || begin < 0 || end < begin ||
      end > total_positions) {
    return;
  }

  float sum = 0.0f;
  for (int item = begin; item < end; ++item) {
    const int position = positions[item];
    if (position < 0 || position >= total_positions) {
      return;
    }
    sum += grad_output[
        static_cast<long long>(position) * embedding_dim + dim];
  }
  grad_weight[static_cast<long long>(token_id) * embedding_dim + dim] = sum;
}

extern "C" bool launch_embedding_scatter_add_deterministic_sparse_kernel(
    float *grad_weight, const float *grad_output,
    const int *unique_ids, const int *offsets, const int *positions,
    int unique_count, int total_positions, int vocab_size,
    int embedding_dim) {
  if (grad_weight == nullptr || grad_output == nullptr ||
      unique_ids == nullptr || offsets == nullptr || positions == nullptr ||
      unique_count <= 0 || total_positions <= 0 || vocab_size <= 0 ||
      embedding_dim <= 0) {
    return false;
  }
  const long long total =
      static_cast<long long>(unique_count) * embedding_dim;
  constexpr int threads = 256;
  const long long block_count = nsos::gpu::ceil_div_positive(
      total, static_cast<long long>(threads));
  if (block_count > static_cast<long long>(INT_MAX)) {
    return false;
  }
  embedding_scatter_add_deterministic_sparse_kernel
      <<<static_cast<int>(block_count), threads>>>(
          grad_weight, grad_output, unique_ids, offsets, positions,
          unique_count, total_positions, vocab_size, embedding_dim);
  return true;
}

__global__ void slice_contiguous_kernel(
    float *out, const float *in, int axis_size, int inner, int start,
    int slice_size, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int inner_index = idx % inner;
  const int axis_index = (idx / inner) % slice_size;
  const int outer_index = idx / (inner * slice_size);
  const int source =
      ((outer_index * axis_size + start + axis_index) * inner) +
      inner_index;
  out[idx] = in[source];
}

extern "C" bool launch_slice_contiguous_kernel(
    float *out, const float *in, int outer, int axis_size, int inner,
    int start, int slice_size, int total) {
  if (out == nullptr || in == nullptr || outer <= 0 || axis_size <= 0 ||
      inner <= 0 || start < 0 || slice_size <= 0 ||
      start > axis_size - slice_size || total <= 0) {
    return false;
  }
  int expected_total = 0;
  int source_total = 0;
  if (!checked_positive_product_to_int(outer, slice_size, inner,
                                       &expected_total) ||
      !checked_positive_product_to_int(outer, axis_size, inner,
                                       &source_total) ||
      total != expected_total) {
    return false;
  }
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(total, threads);
  slice_contiguous_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      out, in, axis_size, inner, start, slice_size, total);
  return true;
}

// =========================================================================
// HPC RMSNorm — Warp-Reduction Kernel
// One block per row, threads cooperatively reduce sum-of-squares.
// Supports arbitrary n_cols via strided thread access.
// =========================================================================

__global__ void rmsnorm_kernel(float *out, const float *in, int n_rows,
                               int n_cols, float eps) {
  int row = blockIdx.x;
  if (row >= n_rows)
    return;

  const float *row_in = in + row * n_cols;
  float *row_out = out + row * n_cols;

  // Phase 1: Each thread accumulates partial sum-of-squares
  float partial_sq = 0.0f;
  for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
    float val = row_in[i];
    partial_sq += val * val;
  }

  // Phase 2: Block-wide reduction for total sum-of-squares
  float total_sq = block_reduce_sum(partial_sq);

  // Broadcast the result to all threads
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(total_sq / n_cols + eps);
  }
  __syncthreads();

  float rms_scale = s_rms;

  // Phase 3: Normalize
  for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
    row_out[i] = row_in[i] * rms_scale;
  }
}

extern "C" void launch_rmsnorm_kernel(float *out, const float *in, int n_rows,
                                      int n_cols, float eps) {
  rmsnorm_kernel<<<n_rows, kReductionThreads, 0, nsos::gpu::current_stream()>>>(out, in, n_rows, n_cols, eps);
}

// =========================================================================
// HPC LayerNorm — Warp-Reduction Kernel
// Two-pass reduction: mean, then variance. One block per row.
// =========================================================================

__global__ void layernorm_kernel(float *out, const float *in, int n_rows,
                                 int n_cols) {
  int row = blockIdx.x;
  if (row >= n_rows)
    return;

  const float *row_in = in + row * n_cols;
  float *row_out = out + row * n_cols;

  // Phase 1: Compute mean via block-wide reduction
  float partial_sum = 0.0f;
  for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
    partial_sum += row_in[i];
  }

  float total_sum = block_reduce_sum(partial_sum);

  __shared__ float s_mean;
  __shared__ float s_inv_std;

  if (threadIdx.x == 0) {
    s_mean = total_sum / n_cols;
  }
  __syncthreads();

  float mean = s_mean;

  // Phase 2: Compute variance via block-wide reduction
  float partial_var = 0.0f;
  for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
    float diff = row_in[i] - mean;
    partial_var += diff * diff;
  }

  float total_var = block_reduce_sum(partial_var);

  if (threadIdx.x == 0) {
    s_inv_std = rsqrtf(total_var / n_cols + 1e-6f);
  }
  __syncthreads();

  float inv_std = s_inv_std;

  // Phase 3: Normalize
  for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
    row_out[i] = (row_in[i] - mean) * inv_std;
  }
}

extern "C" void launch_layernorm_kernel(float *out, const float *in, int n_rows,
                                        int n_cols) {
  layernorm_kernel<<<n_rows, kReductionThreads, 0, nsos::gpu::current_stream()>>>(out, in, n_rows, n_cols);
}

// =========================================================================
// HPC Tiled SGEMM — Shared Memory Tiling
// Each thread block computes a TILE_DIM x TILE_DIM tile of the output C.
// Tiles of A and B are loaded cooperatively into shared memory to maximize
// data reuse and minimize global memory traffic.
// =========================================================================

// (matmul_kernel + launch_matmul_kernel removidos — zero callers; GEMMs densos
// vão por cuBLAS em Tensor::matmul/matmul_nt.)

__global__ void transpose2d_kernel(float *out, const float *in, int rows,
                                   int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x < cols && y < rows) {
    tile[threadIdx.y][threadIdx.x] = in[static_cast<size_t>(y) * cols + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < rows && y < cols) {
    out[static_cast<size_t>(y) * rows + x] = tile[threadIdx.x][threadIdx.y];
  }
}

extern "C" void launch_transpose2d_kernel(float *out, const float *in, int rows,
                                          int cols) {
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid(nsos::gpu::ceil_div_positive(cols, TILE_DIM),
            nsos::gpu::ceil_div_positive(rows, TILE_DIM));
  transpose2d_kernel<<<grid, block, 0, nsos::gpu::current_stream()>>>(out, in, rows, cols);
}

// =========================================================================
// HPC BitNet 1.58-bit GEMM — __dp4a Accelerated (SM 6.1+)
// Uses __dp4a for 4-element INT8 dot products per clock cycle.
// Unpacks 2-bit ternary weights {-1,0,+1} into int8 vectors and uses
// __dp4a for 4x throughput on the inner product.
// =========================================================================

__global__ void bitnet_gemm_kernel(const int8_t *A, const uint32_t *W, float *C,
                                   int M, int K, int N, float scale) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    int acc = 0;
    int k_blocks =
        (K + 15) / 16; // Each uint32 packs 16 ternary weights (2 bits each)

    for (int kb = 0; kb < k_blocks; kb++) {
      uint32_t w_pack = W[static_cast<size_t>(col) * k_blocks + kb];
      int base_k = kb * 16;

// Process 16 weights in groups of 4 using __dp4a
// __dp4a computes: acc += dot(a_vec, b_vec) where a,b are int8x4
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int k_start = base_k + g * 4;
        if (k_start >= K)
          break;

        // Unpack 4 ternary weights from the packed uint32
        // Each weight is 2 bits: 00 -> -1, 01 -> 0, 10 -> +1
        int shift = g * 8; // 4 weights * 2 bits = 8 bits per group
        int8_t w0 = (int8_t)(((w_pack >> (shift + 0)) & 0x3) - 1);
        int8_t w1 = (int8_t)(((w_pack >> (shift + 2)) & 0x3) - 1);
        int8_t w2 = (int8_t)(((w_pack >> (shift + 4)) & 0x3) - 1);
        int8_t w3 = (int8_t)(((w_pack >> (shift + 6)) & 0x3) - 1);

        // Pack weights into int32 for __dp4a
        int w_packed = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
                       ((w3 & 0xFF) << 24);

        // Load 4 activation values and pack into int32
        int8_t a0 = (k_start + 0 < K) ? A[static_cast<size_t>(row) * K + k_start + 0] : 0;
        int8_t a1 = (k_start + 1 < K) ? A[static_cast<size_t>(row) * K + k_start + 1] : 0;
        int8_t a2 = (k_start + 2 < K) ? A[static_cast<size_t>(row) * K + k_start + 2] : 0;
        int8_t a3 = (k_start + 3 < K) ? A[static_cast<size_t>(row) * K + k_start + 3] : 0;
        int a_packed = (a0 & 0xFF) | ((a1 & 0xFF) << 8) | ((a2 & 0xFF) << 16) |
                       ((a3 & 0xFF) << 24);

        // __dp4a: acc += a0*w0 + a1*w1 + a2*w2 + a3*w3
        acc = __dp4a(a_packed, w_packed, acc);
      }
    }
    C[static_cast<size_t>(row) * N + col] = (float)acc * scale;
  }
}

extern "C" void launch_bitnet_gemm(const int8_t *A, const uint32_t *W, float *C,
                                   int M, int K, int N, float scale, int grid_x,
                                   int grid_y, int block_dim) {
  dim3 grid(grid_x, grid_y);
  dim3 block(block_dim, block_dim);
  bitnet_gemm_kernel<<<grid, block, 0, nsos::gpu::current_stream()>>>(A, W, C, M, K, N, scale);
}

// Each block owns one output and one complete wave on either backend. Using
// 64 threads keeps the launch valid for both native wave sizes; wave32 reduces
// two partials through LDS. Integer reduction preserves the packed dot exactly.
__global__ __launch_bounds__(64) void bitnet_gemv_scaled_kernel(
    const int8_t *A, const uint32_t *W, float *C, int K, int N,
    float weight_scale, const float *activation_scale,
    const float *magnitude, const float *bias) {
  for (int col = blockIdx.x; col < N; col += gridDim.x) {
  int acc = 0;
  const int words = K / 16;
  for (int kb = threadIdx.x; kb < words; kb += blockDim.x) {
    const uint32_t packed = W[static_cast<size_t>(col) * words + kb];
    for (int g = 0; g < 4; ++g) {
      const int offset = kb * 16 + g * 4;
      const uint32_t codes = packed >> (g * 8);
      uint32_t w = 0;
      uint32_t a = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int value = static_cast<int>((codes >> (2 * b)) & 3u) - 1;
        w |= (static_cast<uint32_t>(value) & 255u) << (8 * b);
        a |= static_cast<uint32_t>(static_cast<uint8_t>(A[offset + b])) << (8 * b);
      }
      acc = __dp4a(static_cast<int>(a), static_cast<int>(w), acc);
    }
  }
  __shared__ int sums[64];
  sums[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = 32; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float value = __fmul_rn(static_cast<float>(sums[0]), weight_scale);
    value = __fmul_rn(value, activation_scale[0]);
    if (magnitude) value = __fmul_rn(value, magnitude[col]);
    if (bias) value = __fadd_rn(value, bias[col]);
    C[col] = value;
  }
  __syncthreads();
  }
}

// Dynamic activation LUT: 3^4 sums per four activations, rebuilt each token.
// Existing two-bit weight storage is unchanged; this experiment substitutes
// table reads for integer multiplies, not an additional weight compression.
__global__ void bitnet_lut_build_kernel(const int8_t* x, int16_t* table, int groups) {
  const int group = blockIdx.x, code = threadIdx.x;
  if (group >= groups || code >= 81) return;
  int rest = code, sum = 0;
  for (int i = 0; i < 4; ++i) { sum += int(x[group * 4 + i]) * (rest % 3 - 1); rest /= 3; }
  table[group * 81 + code] = int16_t(sum);
}
__global__ __launch_bounds__(64) void bitnet_lut_gemv_kernel(const int8_t* x,
    const uint32_t* weights, const int16_t* table, float* output, int K,
    float weight_scale, const float* scale, const float* magnitude, const float* bias) {
  const int row = blockIdx.x;
  int sum = 0;
  for (int group = threadIdx.x; group < K / 4; group += 64) {
    const unsigned code = (weights[size_t(row) * (K / 16) + group / 4] >> ((group % 4) * 8)) & 255u;
    int index = 0, power = 1, direct = 0;
    bool valid = true;
    for (int i = 0; i < 4; ++i) {
      const int w = (code >> (i * 2)) & 3u;
      valid = valid && w != 3;
      index += power * w; power *= 3;
      direct += int(x[group * 4 + i]) * (w - 1);
    }
    sum += valid ? int(table[group * 81 + index]) : direct;
  }
  __shared__ int scratch[64];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 32; stride; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float value = __fmul_rn(__fmul_rn(float(scratch[0]), weight_scale), scale[0]);
    if (magnitude) value = __fmul_rn(value, magnitude[row]);
    if (bias) value = __fadd_rn(value, bias[row]);
    output[row] = value;
  }
}

extern "C" void launch_bitnet_gemv_scaled(
    const int8_t *A, const uint32_t *W, float *C, int K, int N,
    float weight_scale, const float *activation_scale,
    const float *magnitude, const float *bias) {
  const char* experiment = std::getenv("NSOS_GPU_EXPERIMENT");
  const bool persistent = experiment && std::strcmp(experiment, "persistent_gemv") == 0;
  const bool lut = experiment && std::strcmp(experiment, "ternary_lut") == 0;
  if (experiment && experiment[0] && std::strcmp(experiment, "none") != 0 && !persistent && !lut)
    throw std::invalid_argument("NSOS_GPU_EXPERIMENT must be none, persistent_gemv or ternary_lut");
  if (lut) {
    auto* table = static_cast<int16_t*>(nsos::gpu::current_execution_context().reserve(
        nsos::gpu::WorkspaceSlot::BitnetLut, nsos::gpu::StorageType::Int16, size_t(K / 4) * 81));
    bitnet_lut_build_kernel<<<K / 4, 128, 0, nsos::gpu::current_stream()>>>(A, table, K / 4);
    bitnet_lut_gemv_kernel<<<N, 64, 0, nsos::gpu::current_stream()>>>(A, W, table, C, K, weight_scale,
        activation_scale, magnitude, bias);
    return;
  }
  const int blocks = persistent ? (std::min)(N, 32) : N;
  bitnet_gemv_scaled_kernel<<<blocks, 64, 0, nsos::gpu::current_stream()>>>(A, W, C, K, N, weight_scale,
                                     activation_scale, magnitude, bias);
}

// =========================================================================
// Mean (Reduction) Kernel (unchanged — already reasonable)
// =========================================================================

__global__ void mean_kernel(float *out, const float *in, int outer, int reduce,
                            int inner) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < outer && k < inner) {
    float sum = 0;
    for (int j = 0; j < reduce; ++j) {
      sum += in[i * reduce * inner + j * inner + k];
    }
    out[i * inner + k] = sum / reduce;
  }
}

extern "C" void launch_mean_kernel(float *out, const float *in, int outer,
                                   int reduce, int inner) {
  dim3 block(16, 16);
  dim3 grid(nsos::gpu::ceil_div_positive(outer, block.x),
            nsos::gpu::ceil_div_positive(inner, block.y));
  mean_kernel<<<grid, block, 0, nsos::gpu::current_stream()>>>(out, in, outer, reduce, inner);
}

// =========================================================================
// HPC Softmax — Warp/Block-Reduction Kernel
// One block per row. Uses block-wide reductions for max and sum-exp.
// Three cooperative phases: find max, compute exp+sum, normalize.
// =========================================================================

__global__ void softmax_kernel(float *out, const float *in, int outer,
                               int inner) {
  int row = blockIdx.x;
  if (row >= outer)
    return;

  const float *row_in = in + row * inner;
  float *row_out = out + row * inner;

  // Phase 1: Find max value via block-wide reduction
  float partial_max = -1e30f;
  for (int j = threadIdx.x; j < inner; j += blockDim.x) {
    partial_max = fmaxf(partial_max, row_in[j]);
  }

  float max_val = block_reduce_max(partial_max);

  // Broadcast max to all threads
  __shared__ float s_max;
  __shared__ float s_sum;
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();
  max_val = s_max;

  // Phase 2: Compute exp and sum via block-wide reduction
  float partial_sum = 0.0f;
  for (int j = threadIdx.x; j < inner; j += blockDim.x) {
    float v = expf(row_in[j] - max_val);
    row_out[j] = v;
    partial_sum += v;
  }

  float total_sum = block_reduce_sum(partial_sum);

  if (threadIdx.x == 0)
    s_sum = total_sum + 1e-9f;
  __syncthreads();
  float inv_sum = 1.0f / s_sum;

  // Phase 3: Normalize
  for (int j = threadIdx.x; j < inner; j += blockDim.x) {
    row_out[j] *= inv_sum;
  }
}

extern "C" void launch_softmax_kernel(float *out, const float *in, int outer,
                                      int inner) {
  softmax_kernel<<<outer, kReductionThreads, 0, nsos::gpu::current_stream()>>>(out, in, outer, inner);
}

// =========================================================================
// HPC Fused Cross-Entropy Kernel
// Fused softmax + log + NLL in a single kernel launch.
// One block per batch element. Block-wide reductions for max and sum-exp.
// Computes both loss and gradient in a single pass.
// =========================================================================

__global__ void fused_cross_entropy_kernel(float *total_loss, float *grad,
                                           const float *logits,
                                           const int *target,
                                           const float *row_weights,
                                           int batch, int vocab,
                                           bool ordered_loss) {
  int b = blockIdx.x;
  if (b >= batch)
    return;

  int t = target[b];
  if (t < 0 || t >= vocab) {
    if (ordered_loss && threadIdx.x == 0) {
      total_loss[b] = 0.0f;
    }
    return;
  }

  const float *row_logits = logits + b * vocab;
  float *row_grad = grad + b * vocab;
  const float row_weight = row_weights != nullptr ? row_weights[b] : 1.0f;

  // Phase 1: Find max logit via block-wide reduction
  float partial_max = -1e30f;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    partial_max = fmaxf(partial_max, row_logits[v]);
  }

  float max_val = block_reduce_max(partial_max);

  __shared__ float s_max;
  __shared__ float s_sum;
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();
  max_val = s_max;

  // Phase 2: Compute exp and sum via block-wide reduction
  float partial_sum = 0.0f;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    const float exponential = expf(row_logits[v] - max_val);
    // The gradient buffer is the final output and also a safe per-element
    // staging area. Reusing this exponential removes the second expf pass
    // without changing reduction order or FP32 arithmetic.
    row_grad[v] = exponential;
    partial_sum += exponential;
  }

  float total_sum = block_reduce_sum(partial_sum);

  if (threadIdx.x == 0) {
    s_sum = total_sum;
    // Compute loss for this batch element
    float log_sum_exp = max_val + logf(total_sum);
    const float row_loss =
        row_weight * (log_sum_exp - row_logits[t]);
    if (ordered_loss) {
      total_loss[b] = row_loss;
    } else {
      atomicAdd(total_loss, row_loss);
    }
  }
  __syncthreads();

  // max-subtraction guarantees at least one exp(0)=1 term, so s_sum is
  // strictly positive.  Adding epsilon here would make the returned gradient
  // no longer the exact derivative of the log-sum-exp loss above.
  float inv_sum = 1.0f / s_sum;

  // Phase 3: Compute gradient = softmax(logits) - one_hot(target)
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    float prob = row_grad[v] * inv_sum;
    row_grad[v] = row_weight * (prob - (v == t ? 1.0f : 0.0f));
  }
}

extern "C" void launch_fused_cross_entropy(float *d_loss, float *grad,
                                           const float *logits,
                                           const int *target, int batch,
                                           int vocab) {
  fused_cross_entropy_kernel<<<batch, kReductionThreads, 0, nsos::gpu::current_stream()>>>(
      d_loss, grad, logits, target, nullptr, batch, vocab, false);
}

extern "C" void launch_fused_cross_entropy_weighted(
    float *d_loss, float *grad, const float *logits, const int *target,
    const float *row_weights, int batch, int vocab) {
  fused_cross_entropy_kernel<<<batch, kReductionThreads, 0, nsos::gpu::current_stream()>>>(
      d_loss, grad, logits, target, row_weights, batch, vocab, false);
}

__global__ void cross_entropy_ordered_loss_sum_kernel(
    float *total_loss, const float *row_losses, int batch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float sum = 0.0f;
  for (int row = 0; row < batch; ++row) {
    sum += row_losses[row];
  }
  total_loss[0] = sum;
}

extern "C" bool launch_fused_cross_entropy_deterministic(
    float *d_loss, float *row_losses, float *grad, const float *logits,
    const int *target, const float *row_weights, int batch, int vocab) {
  int elements = 0;
  if (d_loss == nullptr || row_losses == nullptr || grad == nullptr ||
      logits == nullptr || target == nullptr ||
      !checked_positive_product_to_int(batch, vocab, &elements)) {
    return false;
  }
  fused_cross_entropy_kernel<<<batch, kReductionThreads, 0, nsos::gpu::current_stream()>>>(
      row_losses, grad, logits, target, row_weights, batch, vocab, true);
  cross_entropy_ordered_loss_sum_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(
      d_loss, row_losses, batch);
  return true;
}

// Legacy cross-entropy launcher (preserved for backward compatibility)
__global__ void cross_entropy_kernel(float *total_loss, float *grad,
                                     const float *logits, const int *target,
                                     int batch, int vocab) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch) {
    int t = target[b];
    if (t < 0 || t >= vocab)
      return;

    float max_logit = -1e9f;
    for (int v = 0; v < vocab; v++) {
      if (logits[b * vocab + v] > max_logit)
        max_logit = logits[b * vocab + v];
    }
    float sum_exp = 0;
    for (int v = 0; v < vocab; v++) {
      sum_exp += expf(logits[b * vocab + v] - max_logit);
    }

    float log_sum_exp = max_logit + logf(sum_exp);
    float logit_t = logits[b * vocab + t];
    atomicAdd(total_loss, log_sum_exp - logit_t);

    for (int v = 0; v < vocab; v++) {
      float prob = expf(logits[b * vocab + v] - log_sum_exp);
      grad[b * vocab + v] = prob - (v == t ? 1.0f : 0.0f);
    }
  }
}

extern "C" void launch_cross_entropy_kernel(float *d_loss, float *grad,
                                            const float *logits,
                                            const int *target, int batch,
                                            int vocab, int grid_x,
                                            int block_dim) {
  cross_entropy_kernel<<<grid_x, block_dim, 0, nsos::gpu::current_stream()>>>(d_loss, grad, logits, target,
                                              batch, vocab);
}

__global__ void rmsnorm_backward_kernel(float *dx, const float *grad,
                                        const float *x_norm, const float *x,
                                        int outer, int inner, float eps) {
  int row = blockIdx.x;
  if (row >= outer) {
    return;
  }

  const float *row_grad = grad + row * inner;
  const float *row_norm = x_norm + row * inner;
  const float *row_x = x + row * inner;
  float *row_dx = dx + row * inner;

  float partial_dot = 0.0f;
  float partial_sq = 0.0f;
  for (int col = threadIdx.x; col < inner; col += blockDim.x) {
    partial_dot += row_grad[col] * row_norm[col];
    partial_sq += row_x[col] * row_x[col];  // sum of squares of the pre-norm input
  }

  float dot = block_reduce_sum(partial_dot);
  __syncthreads();  // reuse of shared reduction buffer between the two reduces
  float sum_sq = block_reduce_sum(partial_sq);
  __shared__ float s_dot;
  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_dot = dot / max(inner, 1);
    // Exact RMSNorm Jacobian: dx = (1/rms) * (g - y * mean(g.y)), with
    // rms = sqrt(mean(x^2) + eps).  Mirrors the CPU path in tensor.cpp.
    s_inv_rms = rsqrtf(sum_sq / (float)max(inner, 1) + eps);
  }
  __syncthreads();

  for (int col = threadIdx.x; col < inner; col += blockDim.x) {
    row_dx[col] = (row_grad[col] - row_norm[col] * s_dot) * s_inv_rms;
  }
}

extern "C" void launch_rmsnorm_backward_kernel(float *dx, const float *grad,
                                               const float *x_norm,
                                               const float *x, int outer,
                                               int inner, float eps) {
  rmsnorm_backward_kernel<<<outer, kReductionThreads, 0, nsos::gpu::current_stream()>>>(
      dx, grad, x_norm, x, outer, inner, eps);
}

__global__ void adamw_update_kernel(float *weights, const float *grad, float *m,
                                    float *v, int n, float beta1, float beta2,
                                    float bc1, float bc2, float lr, float eps,
                                    float weight_decay,
                                    int apply_weight_decay,
                                    int *found_nonfinite) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float g = grad[idx];
  float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
  float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
  m[idx] = m_new;
  v[idx] = v_new;

  float m_hat = m_new / bc1;
  float v_hat = v_new / bc2;
  float w = weights[idx];
  if (apply_weight_decay) {
    w -= lr * weight_decay * w;
  }
  w -= lr * m_hat / (sqrtf(v_hat) + eps);
  weights[idx] = w;
  if (!isfinite(m_new) || !isfinite(v_new) || !isfinite(w)) {
    atomicExch(found_nonfinite, 1);
  }
}

extern "C" void launch_adamw_update_kernel(float *weights, const float *grad,
                                           float *m, float *v, int n,
                                           float beta1, float beta2, float bc1,
                                           float bc2, float lr, float eps,
                                           float weight_decay,
                                           int apply_weight_decay,
                                           int *found_nonfinite) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  adamw_update_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(weights, grad, m, v, n, beta1, beta2,
                                           bc1, bc2, lr, eps, weight_decay,
                                           apply_weight_decay,
                                           found_nonfinite);
}

// -------------------------------------------------------------------------
// Utility Kernels (unchanged — already simple/efficient)
// -------------------------------------------------------------------------

// ── Mixed-precision casts (AUDIT #6 + LEARN B3) ─────────────────────────
// Cast FP32 -> BF16 or FP16 for Tensor Core matmul inputs.  Done as
// a dedicated kernel rather than via cuBLAS's auto-conversion because
// (a) we want explicit control over rounding (BF16 here uses
// round-to-nearest-even which matches IEEE 754) and (b) doing the cast
// in-place on the same stream as the GEMM avoids implicit syncs.
//
// __nv_bfloat16 is in <cuda_bf16.h>; __half is in <cuda_fp16.h>.
// Both ship with the CUDA toolkit since CUDA 11 (Ampere).
__global__ void cast_f32_to_bf16_kernel(__nv_bfloat16 *out, const float *in, size_t n) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
#if defined(NSOS_GPU_BACKEND_HIP)
    out[idx] = hip_bfloat16(in[idx]);
#else
    out[idx] = __float2bfloat16(in[idx]);
#endif
  }
}

__global__ void cast_f32_to_fp16_kernel(__half *out, const float *in, size_t n) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = __float2half(in[idx]);
  }
}

extern "C" void launch_cast_f32_to_lowp_kernel(void *out, const float *in,
                                                size_t n, int mode) {
  if (out == nullptr || in == nullptr || n == 0) {
    throw std::invalid_argument(
        "lowp cast requires non-null buffers and a non-empty tensor");
  }
  if (mode != 1 && mode != 2) {
    throw std::invalid_argument("lowp cast mode must be BF16 or FP16");
  }
  // mode=1 -> BF16; mode=2 -> FP16; validated above.
  const int threads = 256;
  const size_t blocks_sz = nsos::gpu::ceil_div_positive(
      n, static_cast<size_t>(threads));
  // A requested low-precision contract must fail closed on an invalid grid;
  // returning here would feed an uninitialized staging buffer to GEMM.
  if (blocks_sz > static_cast<size_t>(INT_MAX)) {
    throw std::overflow_error("lowp cast exceeds the GPU grid limit");
  }
  const int blocks = static_cast<int>(blocks_sz);
  if (mode == 1) {
    cast_f32_to_bf16_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
        reinterpret_cast<__nv_bfloat16 *>(out), in, n);
  } else if (mode == 2) {
    cast_f32_to_fp16_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
        reinterpret_cast<__half *>(out), in, n);
  }
}

template <typename Lowp, bool BFloat16>
__global__ void cast_transpose_f32_to_lowp_kernel(
    Lowp *out, const float *in, int rows, int cols, int tiles_x) {
  constexpr int kTile = 32;
  constexpr int kRowsPerIteration = 8;
  __shared__ float tile[kTile][kTile + 1];

  const int tile_index = static_cast<int>(blockIdx.x);
  const int tile_x = tile_index % tiles_x;
  const int tile_y = tile_index / tiles_x;
  const int input_col = tile_x * kTile + static_cast<int>(threadIdx.x);
  const int input_row_base =
      tile_y * kTile + static_cast<int>(threadIdx.y);

#pragma unroll
  for (int offset = 0; offset < kTile; offset += kRowsPerIteration) {
    const int input_row = input_row_base + offset;
    if (input_row < rows && input_col < cols) {
      tile[threadIdx.y + offset][threadIdx.x] =
          in[static_cast<size_t>(input_row) * cols + input_col];
    }
  }
  __syncthreads();

  const int output_col =
      tile_y * kTile + static_cast<int>(threadIdx.x);
  const int output_row_base =
      tile_x * kTile + static_cast<int>(threadIdx.y);
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kRowsPerIteration) {
    const int output_row = output_row_base + offset;
    if (output_row < cols && output_col < rows) {
      const float value = tile[threadIdx.x][threadIdx.y + offset];
      if constexpr (BFloat16) {
#if defined(NSOS_GPU_BACKEND_HIP)
        out[static_cast<size_t>(output_row) * rows + output_col] =
            hip_bfloat16(value);
#else
        out[static_cast<size_t>(output_row) * rows + output_col] =
            __float2bfloat16(value);
#endif
      } else {
        out[static_cast<size_t>(output_row) * rows + output_col] =
            __float2half(value);
      }
    }
  }
}

extern "C" void launch_cast_transpose_f32_to_lowp_kernel(
    void *out, const float *in, int rows, int cols, int mode) {
  if (out == nullptr || in == nullptr || rows <= 0 || cols <= 0) {
    throw std::invalid_argument(
        "lowp cast-transpose requires non-null buffers and positive shape");
  }
  if (mode != 1 && mode != 2) {
    throw std::invalid_argument(
        "lowp cast-transpose mode must be BF16 or FP16");
  }
  constexpr int kTile = 32;
  constexpr int kRowsPerIteration = 8;
  const size_t tiles_x = nsos::gpu::ceil_div_positive(
      static_cast<size_t>(cols), static_cast<size_t>(kTile));
  const size_t tiles_y = nsos::gpu::ceil_div_positive(
      static_cast<size_t>(rows), static_cast<size_t>(kTile));
  if (tiles_x > static_cast<size_t>(INT_MAX) ||
      tiles_y > static_cast<size_t>(INT_MAX) ||
      tiles_x > static_cast<size_t>(INT_MAX) / tiles_y) {
    throw std::overflow_error(
        "lowp cast-transpose exceeds the GPU grid limit");
  }
  const int blocks = static_cast<int>(tiles_x * tiles_y);
  const dim3 threads(kTile, kRowsPerIteration);
  if (mode == 1) {
    cast_transpose_f32_to_lowp_kernel<__nv_bfloat16, true>
        <<<blocks, threads, 0, nsos::gpu::current_stream()>>>(reinterpret_cast<__nv_bfloat16 *>(out), in,
                              rows, cols, static_cast<int>(tiles_x));
  } else {
    cast_transpose_f32_to_lowp_kernel<__half, false>
        <<<blocks, threads, 0, nsos::gpu::current_stream()>>>(reinterpret_cast<__half *>(out), in,
                              rows, cols, static_cast<int>(tiles_x));
  }
}

__global__ void relu_kernel(float *out, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = fmaxf(0.0f, in[idx]);
}

extern "C" void launch_relu_kernel(float *out, const float *in, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  relu_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, in, n);
}

// ── Squared ReLU (LEARN S1) ──────────────────────────────────────────────
// Activation:  f(x) = max(0, x)^2
// Used in BitNet b1.58 2B4T as the FFN activation because SwiGLU in
// low-precision regimes (ternary, FP8) suffers occasional activation
// spikes that overflow dynamic range and diverge loss after extended
// training (~hundreds of billions of tokens).  Squared ReLU has
// comparable expressive power to SwiGLU at moderate scale (1B-2B
// params) and is numerically stable in quantized regimes.
//
// The forward computes y = relu(x) * relu(x) but more cheaply: one
// load, one max, one multiply.  The result is in [0, +infty) and is
// monotonically increasing for x > 0 with slope 2x (vs ReLU's slope 1).
__global__ void squared_relu_kernel(float *out, const float *in, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = fmaxf(0.0f, in[idx]);
    out[idx] = v * v;
  }
}

extern "C" void launch_squared_relu_kernel(float *out, const float *in, int n) {
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(n, threads);
  squared_relu_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, in, n);
}

// Squared ReLU backward.
//
// Given pre_activation x and grad_out dL/dy where y = squared_relu(x):
//   dL/dx = dL/dy * dy/dx
//   dy/dx = d(max(0,x)^2)/dx
//         = 2 * max(0, x)   for x > 0
//         = 0               for x <= 0
//
// We multiply IN PLACE into in_grad (which on entry holds dL/dy from
// the next layer's backward; on exit holds dL/dx for this activation).
__global__ void squared_relu_backward_kernel(float *in_grad,
                                              const float *grad_out,
                                              const float *pre_activation,
                                              int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float x = pre_activation[idx];
    const float relu_x = fmaxf(0.0f, x);
    // grad_out may alias in_grad — read it first, write last.
    const float dy = grad_out[idx];
    in_grad[idx] = dy * 2.0f * relu_x;
  }
}

extern "C" void launch_squared_relu_backward_kernel(float *in_grad,
                                                     const float *grad_out,
                                                     const float *pre_activation,
                                                     int n) {
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(n, threads);
  squared_relu_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(in_grad, grad_out,
                                                     pre_activation, n);
}

__global__ void clamp_kernel(float *out, const float *in, float min_val,
                             float max_val, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = fminf(fmaxf(in[idx], min_val), max_val);
}

extern "C" void launch_clamp_kernel(float *out, const float *in, float min_val,
                                    float max_val, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  clamp_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(out, in, min_val, max_val, n);
}

__global__ void norm_kernel(float *d_sum_sq, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    atomicAdd(d_sum_sq, val * val);
  }
}

extern "C" void launch_norm_kernel(float *d_sum_sq, const float *in, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  norm_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(d_sum_sq, in, n);
}

__global__ void multi_tensor_sqsum_deterministic_partials_kernel(
    double *__restrict__ partials,
    float *const *__restrict__ gradients,
    const unsigned long long *__restrict__ sizes, int tensor_count) {
  constexpr int kThreads = 256;
  __shared__ double reduction[kThreads];
  const int tensor_index = blockIdx.x;
  if (tensor_index >= tensor_count) {
    return;
  }
  const float *gradient = gradients[tensor_index];
  const unsigned long long count = sizes[tensor_index];
  double local = 0.0;
  for (unsigned long long index = threadIdx.x; index < count;
       index += kThreads) {
    const double value = static_cast<double>(gradient[index]);
    local += value * value;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[tensor_index] = reduction[0];
  }
}

__global__ void multi_tensor_sqsum_deterministic_finalize_kernel(
    double *__restrict__ total, const double *__restrict__ partials,
    int tensor_count, const int *__restrict__ deferred_finite_issue) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  double sum = 0.0;
  for (int tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
    sum += partials[tensor_index];
  }
  // Sum-of-squares cannot be negative. Encode a deferred finite-gate failure
  // as -1.0 so the existing eight-byte D2H scalar carries both results without
  // changing the ordered FP64 reduction or adding a second barrier.
  total[0] = deferred_finite_issue != nullptr &&
                     deferred_finite_issue[0] != 0
                 ? -1.0
                 : sum;
}

extern "C" bool launch_multi_tensor_sqsum_deterministic(
    double *total, double *partials, float *const *gradients,
    const unsigned long long *sizes, int tensor_count,
    const int *deferred_finite_issue) {
  if (total == nullptr || partials == nullptr || gradients == nullptr ||
      sizes == nullptr || tensor_count <= 0) {
    return false;
  }
  constexpr int kThreads = 256;
  multi_tensor_sqsum_deterministic_partials_kernel<<<tensor_count, kThreads, 0, nsos::gpu::current_stream()>>>(
      partials, gradients, sizes, tensor_count);
  multi_tensor_sqsum_deterministic_finalize_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(
      total, partials, tensor_count, deferred_finite_issue);
  return true;
}

__global__ void ttt_row_error_kernel(const float* input, const float* keys,
    const float* base, const float* adaptation, float* errors, float* output,
    float* scale, int hidden, int dim, float max_norm) {
  __shared__ double norm[256];
  double sum = 0.0;
  for (int d = threadIdx.x; d < dim; d += 256) {
    float correction = 0.0f;
    for (int h = 0; h < hidden; ++h)
      correction += keys[h] * adaptation[static_cast<size_t>(h) * dim + d];
    const float out = base[d] + correction;
    const float error = out - input[d];
    output[d] = out;
    errors[d] = error;
    sum += static_cast<double>(error) * error;
  }
  norm[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) norm[threadIdx.x] += norm[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float value = static_cast<float>(sqrt(norm[0]));
    scale[0] = max_norm > 0 && value > max_norm ? max_norm / (value + 1e-6f) : 1.0f;
  }
}

__global__ void ttt_adapt_kernel(const float* key, const float* error,
    const float* scale, float* adaptation, float* momentum, float* history,
    int hidden, int dim, float decay, float step, bool hamiltonian,
    float* momentum_history = nullptr) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(hidden) * dim) return;
  const float old = adaptation[i];
  const float g = (key[i / dim] * error[i % dim]) * scale[0];
  const float m = momentum[i] * decay + (hamiltonian ? g * (1.0f - decay) : g);
  if (history) history[i] = old;
  if (momentum_history) momentum_history[i] = momentum[i];
  momentum[i] = m;
  adaptation[i] = hamiltonian ? old - (g + m * 0.25f) * step
                              : old * decay - g * step;
}

__global__ void ttt_key_backward_kernel(const float* grad, const float* history,
    float* grad_keys, int rows, int hidden, int dim) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(rows) * hidden) return;
  const int row = static_cast<int>(i / hidden);
  float sum = 0.0f;
  for (int d = 0; d < dim; ++d)
    sum += grad[static_cast<size_t>(row) * dim + d] * history[i * dim + d];
  grad_keys[i] = sum;
}

extern "C" bool launch_ttt_device_forward(const float* input, const float* keys,
    const float* base, float* adaptation, float* momentum, float* history,
    float* errors, float* output, float* scale, int rows, int hidden, int dim,
    float decay, float step, float max_norm, bool hamiltonian,
    int history_chunk, float* momentum_history) {
  if (!input || !keys || !base || !adaptation || !momentum || !history ||
      !errors || !output || !scale || rows <= 0 || hidden <= 0 || dim <= 0 ||
      !std::isfinite(decay) || decay < 0 || decay >= 1 || !std::isfinite(step) || step < 0 ||
      !std::isfinite(max_norm) || max_norm < 0 ||
      (history_chunk != 0 && (history_chunk != 32 || !momentum_history))) return false;
  const size_t cells = static_cast<size_t>(hidden) * dim;
  const auto stream = nsos::gpu::current_stream();
  for (int r = 0; r < rows; ++r) {
    const size_t row = static_cast<size_t>(r) * dim;
    ttt_row_error_kernel<<<1, 256, 0, stream>>>(input + row, keys + static_cast<size_t>(r) * hidden,
        base + row, adaptation, errors + row, output + row, scale, hidden, dim, max_norm);
    float* saved_a = history_chunk == 0 ? history + r * cells
        : r % history_chunk == 0 ? history + (r / history_chunk) * cells : nullptr;
    float* saved_m = history_chunk > 0 && r % history_chunk == 0
        ? momentum_history + (r / history_chunk) * cells : nullptr;
    ttt_adapt_kernel<<<(cells + 255) / 256, 256, 0, stream>>>(
        keys + static_cast<size_t>(r) * hidden, errors + row, scale,
        adaptation, momentum, saved_a, hidden, dim, decay, step, hamiltonian, saved_m);
  }
  return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_ttt_device_key_backward(const float* grad, const float* history,
    float* grad_keys, int rows, int hidden, int dim) {
  if (!grad || !history || !grad_keys || rows <= 0 || hidden <= 0 || dim <= 0) return false;
  const size_t count = static_cast<size_t>(rows) * hidden;
  ttt_key_backward_kernel<<<(count + 255) / 256, 256, 0, nsos::gpu::current_stream()>>>(
      grad, history, grad_keys, rows, hidden, dim);
  return cudaGetLastError() == cudaSuccess;
}

// Coefficients for reconstructing the original FP32 update, not a second
// forward pass with current projection weights or recomputed errors.
__global__ void ttt_error_scale_kernel(const float* error, double* coeff,
                                     int dim, float max_norm) {
  __shared__ double norm[256];
  double sum = 0;
  for (int d = threadIdx.x; d < dim; d += 256) sum += static_cast<double>(error[d]) * error[d];
  norm[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) norm[threadIdx.x] += norm[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float n = static_cast<float>(sqrt(norm[0]));
    const float scale = max_norm > 0 && n > max_norm ? max_norm / (n + 1e-6f) : 1;
    // ttt_adapt_kernel takes a float coefficient. Keep a separate aligned slot.
    reinterpret_cast<float*>(coeff)[0] = scale;
  }
}

__device__ float ttt_local_gbar(float a, float m, float decay, float step, bool ham) {
  const float next_m_bar = ham ? m - (0.25f * step) * a : m;
  return -step * a + (ham ? (1.0f - decay) * next_m_bar : next_m_bar);
}

__global__ void ttt_full_q_kernel(const float* key, const float* error,
    const float* adj_a, const float* adj_m, float* q, double* coeff,
    int hidden, int dim, float decay, float step, float max_norm, bool ham) {
  __shared__ double norm[256], dot[256];
  double ns = 0, ds = 0;
  for (int d = threadIdx.x; d < dim; d += 256) {
    float value = 0;
    for (int h = 0; h < hidden; ++h) {
      const size_t i = static_cast<size_t>(h) * dim + d;
      value += key[h] * ttt_local_gbar(adj_a[i], adj_m[i], decay, step, ham);
    }
    q[d] = value;
    ns += static_cast<double>(error[d]) * error[d];
    ds += static_cast<double>(value) * error[d];
  }
  norm[threadIdx.x] = ns; dot[threadIdx.x] = ds;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      norm[threadIdx.x] += norm[threadIdx.x + stride];
      dot[threadIdx.x] += dot[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float n = static_cast<float>(sqrt(norm[0]));
    const bool clipped = max_norm > 0 && n > max_norm;
    const float scale = clipped ? max_norm / (n + 1e-6f) : 1;
    coeff[0] = scale;
    coeff[1] = clipped ? static_cast<double>(scale) * dot[0] /
        (static_cast<double>(n) * (n + static_cast<double>(1e-6f))) : 0;
  }
}

// The reference kernel above puts all columns on one CU. At large state
// widths, independent column owners distribute the same ordered hidden sum
// over wave-sized CTAs. No atomics, hidden-axis tree or new summation order.
__global__ void ttt_full_q_columns_kernel(const float* key, const float* adj_a,
    const float* adj_m, float* q, int hidden, int dim,
    float decay, float step, bool ham) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  float value = 0;
  for (int h = 0; h < hidden; ++h) {
    const size_t i = static_cast<size_t>(h) * dim + d;
    value += key[h] * ttt_local_gbar(adj_a[i], adj_m[i], decay, step, ham);
  }
  q[d] = value;
}

__global__ void ttt_full_q_coefficients_kernel(const float* error, const float* q,
    double* coeff, int dim, float max_norm) {
  __shared__ double norm[256], dot[256];
  double ns = 0, ds = 0;
  for (int d = threadIdx.x; d < dim; d += 256) {
    ns += static_cast<double>(error[d]) * error[d];
    ds += static_cast<double>(q[d]) * error[d];
  }
  norm[threadIdx.x] = ns; dot[threadIdx.x] = ds;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      norm[threadIdx.x] += norm[threadIdx.x + stride];
      dot[threadIdx.x] += dot[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float n = static_cast<float>(sqrt(norm[0]));
    const bool clipped = max_norm > 0 && n > max_norm;
    const float scale = clipped ? max_norm / (n + 1e-6f) : 1;
    coeff[0] = scale;
    coeff[1] = clipped ? static_cast<double>(scale) * dot[0] /
        (static_cast<double>(n) * (n + static_cast<double>(1e-6f))) : 0;
  }
}

__global__ void ttt_full_error_vjp_kernel(const float* grad, const float* error,
    const float* q, const double* coeff, float* grad_base, float* grad_direct, int dim) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  const double de = coeff[0] * q[d] - coeff[1] * error[d];
  grad_base[d] = static_cast<float>(grad[d] + de);
  grad_direct[d] = static_cast<float>(-de);
}

__global__ void ttt_full_state_vjp_kernel(const float* key, const float* error,
    const float* state, const float* total, const double* coeff,
    float* adj_a, float* adj_m, float* grad_keys,
    int hidden, int dim, float decay, float step, bool ham) {
  const int lane = threadIdx.x % 32;
  const int h = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (h >= hidden) return;
  float dk = 0;
  for (int d = lane; d < dim; d += 32) {
    const size_t i = static_cast<size_t>(h) * dim + d;
    const float a_bar = adj_a[i], m_bar = adj_m[i];
    const float gbar = ttt_local_gbar(a_bar, m_bar, decay, step, ham);
    dk += total[d] * state[i] + static_cast<float>(coeff[0]) * gbar * error[d];
    adj_a[i] = (ham ? a_bar : decay * a_bar) + key[h] * total[d];
    adj_m[i] = decay * (ham ? m_bar - (0.25f * step) * a_bar : m_bar);
  }
  for (int delta = 16; delta; delta >>= 1) {
#if defined(NSOS_GPU_BACKEND_HIP)
    dk += __shfl_down(dk, delta, 32);
#else
    dk += __shfl_down_sync(0xffffffffu, dk, delta, 32);
#endif
  }
  if (lane == 0) grad_keys[h] = dk;
}

extern "C" bool launch_ttt_full_backward(const float* keys, const float* errors,
    const float* boundaries, const float* momentum_boundaries, const float* grad,
    float* grad_keys, float* grad_base, float* grad_direct, float* adaptation,
    float* momentum, float* local_history, float* adj_a, float* adj_m,
    float* q, double* coefficients, int rows, int hidden, int dim, int chunk,
    float decay, float step, float max_norm, bool hamiltonian) {
  if (!keys || !errors || !boundaries || !momentum_boundaries || !grad ||
      !grad_keys || !grad_base || !grad_direct || !adaptation || !momentum ||
      !local_history || !adj_a || !adj_m || !q || !coefficients ||
      rows <= 0 || hidden <= 0 || dim <= 0 || chunk != 32 ||
      !std::isfinite(decay) || decay < 0 || decay >= 1 ||
      !std::isfinite(step) || step < 0 || !std::isfinite(max_norm) || max_norm < 0) return false;
  const size_t cells = static_cast<size_t>(hidden) * dim;
  const auto stream = nsos::gpu::current_stream();
  if (cudaMemsetAsync(adj_a, 0, cells * sizeof(float), stream) != cudaSuccess ||
      cudaMemsetAsync(adj_m, 0, cells * sizeof(float), stream) != cudaSuccess) return false;
  for (int c = (rows - 1) / chunk; c >= 0; --c) {
    const int start = c * chunk, stop = start + std::min(chunk, rows - start);
    if (cudaMemcpyAsync(adaptation, boundaries + c * cells, cells * sizeof(float),
        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(momentum, momentum_boundaries + c * cells, cells * sizeof(float),
        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return false;
    nsos::record_gpu_transfer(nsos::Device::GPU, nsos::Device::GPU, cells * sizeof(float));
    nsos::record_gpu_transfer(nsos::Device::GPU, nsos::Device::GPU, cells * sizeof(float));
    for (int r = start; r < stop; ++r) {
      const float* e = errors + static_cast<size_t>(r) * dim;
      ttt_error_scale_kernel<<<1, 256, 0, stream>>>(e, coefficients, dim, max_norm);
      ttt_adapt_kernel<<<(cells + 255) / 256, 256, 0, stream>>>(
          keys + static_cast<size_t>(r) * hidden, e, reinterpret_cast<float*>(coefficients),
          adaptation, momentum, local_history + (r - start) * cells,
          hidden, dim, decay, step, hamiltonian);
    }
    for (int r = stop - 1; r >= start; --r) {
      const size_t row = static_cast<size_t>(r) * dim;
      const float* k = keys + static_cast<size_t>(r) * hidden;
      if (hidden >= 256 && dim >= 256) {
        ttt_full_q_columns_kernel<<<(dim + 31) / 32, 32, 0, stream>>>(
            k, adj_a, adj_m, q, hidden, dim, decay, step, hamiltonian);
        ttt_full_q_coefficients_kernel<<<1, 256, 0, stream>>>(
            errors + row, q, coefficients, dim, max_norm);
      } else {
        // Small states keep the single-launch reference geometry.
        ttt_full_q_kernel<<<1, 256, 0, stream>>>(k, errors + row, adj_a, adj_m, q,
            coefficients, hidden, dim, decay, step, max_norm, hamiltonian);
      }
      ttt_full_error_vjp_kernel<<<(dim + 255) / 256, 256, 0, stream>>>(
          grad + row, errors + row, q, coefficients, grad_base + row, grad_direct + row, dim);
      ttt_full_state_vjp_kernel<<<(static_cast<size_t>(hidden) * 32 + 127) / 128, 128, 0, stream>>>(
          k, errors + row, local_history + (r - start) * cells, grad_base + row, coefficients,
          adj_a, adj_m, grad_keys + static_cast<size_t>(r) * hidden,
          hidden, dim, decay, step, hamiltonian);
    }
  }
  return cudaGetLastError() == cudaSuccess;
}

__global__ void chunked_norm_partials_kernel(double* partials,
    float* const* gradients, const NsosMultiTensorChunk* chunks) {
  __shared__ double tree[256];
  const auto chunk = chunks[blockIdx.x];
  const float* g = gradients[chunk.tensor_index] + chunk.element_offset;
  double sum = 0.0;
  for (unsigned int i = threadIdx.x; i < chunk.element_count; i += 256) {
    const double value = static_cast<double>(g[i]);
    sum += value * value;
  }
  tree[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) tree[threadIdx.x] += tree[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = tree[0];
}

__global__ void norm_clip_coefficient_kernel(const double* total,
    float* coefficient, float max_norm) {
  const float norm = static_cast<float>(sqrt(total[0]));
  coefficient[0] = total[0] >= 0.0 && isfinite(norm) && norm > max_norm
      ? max_norm / (norm + 1e-6f) : 1.0f;
}

__global__ void chunked_device_clip_kernel(float* const* gradients,
    const NsosMultiTensorChunk* chunks, const float* coefficient) {
  const float scale = coefficient[0];
  if (scale == 1.0f) return;
  const auto chunk = chunks[blockIdx.x];
  float* g = gradients[chunk.tensor_index] + chunk.element_offset;
  for (unsigned int i = threadIdx.x; i < chunk.element_count; i += blockDim.x)
    g[i] *= scale;
}

extern "C" bool launch_chunked_norm_device_clip(double* total, double* partials,
    float* coefficient, float* const* gradients,
    const NsosMultiTensorChunk* chunks, int chunk_count, float max_norm,
    const int* deferred_finite_issue) {
  if (!total || !partials || !coefficient || !gradients || !chunks ||
      chunk_count <= 0 || !std::isfinite(max_norm) || max_norm <= 0) return false;
  const auto stream = nsos::gpu::current_stream();
  chunked_norm_partials_kernel<<<chunk_count, 256, 0, stream>>>(partials, gradients, chunks);
  multi_tensor_sqsum_deterministic_finalize_kernel<<<1, 1, 0, stream>>>(
      total, partials, chunk_count, deferred_finite_issue);
  norm_clip_coefficient_kernel<<<1, 1, 0, stream>>>(total, coefficient, max_norm);
  chunked_device_clip_kernel<<<chunk_count, 256, 0, stream>>>(gradients, chunks, coefficient);
  return cudaGetLastError() == cudaSuccess;
}

// Σ|x| — block-partial + single atomicAdd per block (same nondeterminism class
// as norm_kernel).  Powers the BitNet b1.58 absmean weight scale on device.
__global__ void abs_sum_kernel(float *d_abs_sum, const float *in, int n) {
  __shared__ float partial[256];
  float local = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    local += fabsf(in[i]);
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(d_abs_sum, partial[0]);
  }
}

extern "C" void launch_abs_sum_kernel(float *d_abs_sum, const float *in, int n) {
  const int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  if (blocks > 4096) blocks = 4096;  // grid-stride loop covers the rest
  abs_sum_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(d_abs_sum, in, n);
}

// Repetition-unlikelihood (Welleck et al. 2020) gradient adjustment, on GPU.
// Mirrors the host loop in trainer.cpp::apply_repetition_unlikelihood EXACTLY,
// eliminating the per-sample probs.cpu()/grad.cpu() D2H + host loop that was the
// dominant training-step cost (it scales with answer_len*batch).
//
// One block per answer row (rows 1..rows-1; row 0 has no history window).
// Thread 0 reconstructs the up-to-4 unique "negative" tokens from the answer
// token window [row-4, row-1] (excluding the row's target and EOS), computes
// each factor = scale * p_neg/(1-p_neg) from the on-GPU softmax, sums them, then
// all threads apply grad[row,:] -= total_factor*probs[row,:] and thread 0 adds
// each factor back at its negative-token column.  Net result is identical to the
// host's per-negative subtract+scatter (the subtracts are linear and sum).
__global__ void repetition_unlikelihood_kernel(
    float *__restrict__ total_loss, float *__restrict__ grad,
    const float *__restrict__ probs,
    const int *__restrict__ answer_tokens, int rows, int vocab, float scale,
    int eos_token_id) {
  const int row = blockIdx.x + 1;  // rows 1..rows-1
  if (row >= rows) return;

  __shared__ int s_neg[4];
  __shared__ float s_factor[4];
  __shared__ int s_count;
  __shared__ float s_total_factor;

  if (threadIdx.x == 0) {
    const int target_token = answer_tokens[row];
    int neg_ids[4];
    int neg_count = 0;
    const int window_start = max(0, row - 4);
    for (int prev = row - 1; prev >= window_start; --prev) {
      const int cand = answer_tokens[prev];
      if (cand == target_token || cand == eos_token_id) continue;
      bool seen = false;
      for (int i = 0; i < neg_count; ++i) {
        if (neg_ids[i] == cand) { seen = true; break; }
      }
      if (!seen && neg_count < 4) neg_ids[neg_count++] = cand;
    }
    int kept = 0;
    float total = 0.0f;
    for (int i = 0; i < neg_count; ++i) {
      const int neg = neg_ids[i];
      if (neg < 0 || neg >= vocab) continue;
      const float p_neg = probs[row * vocab + neg];
      if (p_neg <= 1e-6f || p_neg >= 1.0f - 1e-6f) continue;
      const float denom = fmaxf(1.0f - p_neg, 1e-6f);
      const float factor = scale * p_neg / denom;
      atomicAdd(total_loss, -scale * log1pf(-p_neg));
      s_neg[kept] = neg;
      s_factor[kept] = factor;
      total += factor;
      ++kept;
    }
    s_count = kept;
    s_total_factor = total;
  }
  __syncthreads();

  const int count = s_count;
  if (count == 0) return;
  const float total_factor = s_total_factor;
  float *row_grad = grad + row * vocab;
  const float *row_probs = probs + row * vocab;
  for (int col = threadIdx.x; col < vocab; col += blockDim.x) {
    row_grad[col] -= total_factor * row_probs[col];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < count; ++i) {
      row_grad[s_neg[i]] += s_factor[i];
    }
  }
}

extern "C" void launch_repetition_unlikelihood_kernel(
    float *d_loss, float *grad, const float *probs,
    const int *answer_tokens, int rows, int vocab, float scale,
    int eos_token_id) {
  if (rows <= 1 || vocab <= 0) return;
  const int threads = 256;
  const int blocks = rows - 1;  // one block per row 1..rows-1
  repetition_unlikelihood_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      d_loss, grad, probs, answer_tokens, rows, vocab, scale, eos_token_id);
}

__global__ void repetition_unlikelihood_masked_kernel(
    float *__restrict__ total_loss, float *__restrict__ grad,
    const float *__restrict__ probs, const int *__restrict__ token_plane,
    int batch, int seq, int vocab, float scale, int eos_token_id) {
  const int flat_row = blockIdx.x;
  const int total_rows = batch * seq;
  if (flat_row >= total_rows) return;
  const int row = flat_row % seq;
  if (row == 0 || token_plane[flat_row] < 0) return;

  __shared__ int s_neg[4];
  __shared__ float s_factor[4];
  __shared__ int s_count;
  __shared__ float s_total_factor;
  if (threadIdx.x == 0) {
    const int target_token = token_plane[flat_row];
    int neg_ids[4];
    int neg_count = 0;
    const int window_start = max(0, row - 4);
    const int sample_start = flat_row - row;
    for (int previous = row - 1; previous >= window_start; --previous) {
      const int candidate = token_plane[sample_start + previous];
      if (candidate < 0 || candidate == target_token ||
          candidate == eos_token_id) {
        continue;
      }
      bool seen = false;
      for (int index = 0; index < neg_count; ++index) {
        if (neg_ids[index] == candidate) {
          seen = true;
          break;
        }
      }
      if (!seen && neg_count < 4) {
        neg_ids[neg_count++] = candidate;
      }
    }
    int kept = 0;
    float total_factor = 0.0f;
    for (int index = 0; index < neg_count; ++index) {
      const int negative = neg_ids[index];
      if (negative < 0 || negative >= vocab) continue;
      const float probability = probs[flat_row * vocab + negative];
      if (probability <= 1e-6f || probability >= 1.0f - 1e-6f) {
        continue;
      }
      const float denominator = fmaxf(1.0f - probability, 1e-6f);
      const float factor = scale * probability / denominator;
      atomicAdd(total_loss, -scale * log1pf(-probability));
      s_neg[kept] = negative;
      s_factor[kept] = factor;
      total_factor += factor;
      ++kept;
    }
    s_count = kept;
    s_total_factor = total_factor;
  }
  __syncthreads();

  if (s_count == 0) return;
  float *row_grad = grad + flat_row * vocab;
  const float *row_probs = probs + flat_row * vocab;
  for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
    row_grad[column] -= s_total_factor * row_probs[column];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int index = 0; index < s_count; ++index) {
      row_grad[s_neg[index]] += s_factor[index];
    }
  }
}

extern "C" bool launch_repetition_unlikelihood_masked_kernel(
    float *d_loss, float *grad, const float *probs,
    const int *token_plane, int batch, int seq, int vocab, float scale,
    int eos_token_id) {
  int rows = 0;
  int elements = 0;
  if (d_loss == nullptr || grad == nullptr || probs == nullptr ||
      token_plane == nullptr || seq <= 1 || vocab <= 0 ||
      !std::isfinite(scale) || scale <= 0.0f ||
      !checked_positive_product_to_int(batch, seq, &rows) ||
      !checked_positive_product_to_int(rows, vocab, &elements)) {
    return false;
  }
  repetition_unlikelihood_masked_kernel<<<rows, 256, 0, nsos::gpu::current_stream()>>>(
      d_loss, grad, probs, token_plane, batch, seq, vocab, scale,
      eos_token_id);
  return true;
}

__global__ void masked_logit_l2_kernel(
    float *__restrict__ total_loss, float *__restrict__ grad,
    const float *__restrict__ logits,
    const float *__restrict__ row_weights, int rows, int vocab, float beta) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float row_weight = row_weights[row];
  if (row_weight == 0.0f) return;
  __shared__ float partial[256];
  float local_sq = 0.0f;
  const float gradient_scale =
      beta * row_weight / static_cast<float>(vocab);
  const int offset = row * vocab;
  for (int column = threadIdx.x; column < vocab; column += blockDim.x) {
    const float value = logits[offset + column];
    local_sq += value * value;
    grad[offset + column] += gradient_scale * value;
  }
  partial[threadIdx.x] = local_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(total_loss,
              0.5f * beta * row_weight * partial[0] /
                  static_cast<float>(vocab));
  }
}

extern "C" bool launch_masked_logit_l2_kernel(
    float *d_loss, float *grad, const float *logits,
    const float *row_weights, int rows, int vocab, float beta) {
  int elements = 0;
  if (d_loss == nullptr || grad == nullptr || logits == nullptr ||
      row_weights == nullptr || !std::isfinite(beta) || beta <= 0.0f ||
      !checked_positive_product_to_int(rows, vocab, &elements)) {
    return false;
  }
  masked_logit_l2_kernel<<<rows, 256, 0, nsos::gpu::current_stream()>>>(
      d_loss, grad, logits, row_weights, rows, vocab, beta);
  return true;
}

__global__ void check_stability_kernel(int *d_found, const float *in,
                                       float max_val, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = in[idx];
    if (isnan(v) || isinf(v) || fabsf(v) > max_val) {
      atomicExch(d_found, 1);
    }
  }
}

extern "C" void launch_check_stability_kernel(int *d_found, const float *in,
                                              float max_val, int n) {
  int threads = 256;
  int blocks = nsos::gpu::ceil_div_positive(n, threads);
  check_stability_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(d_found, in, max_val, n);
}

// -------------------------------------------------------------------------
// GQA Causal Attention Kernel
// -------------------------------------------------------------------------
__global__ void gqa_causal_attention_kernel(const float *q_flat,
                                            const float *kv_flat,
                                            float *out, int seq_len,
                                            int d_model, int n_heads,
                                            int n_kv_heads, int head_dim,
                                            int kv_group_size, float theta,
                                            int sliding_window) {
  const int token_index = blockIdx.x;
  const int head_index = blockIdx.y;
  const int thread_index = threadIdx.x;
  const int kv_dim = n_kv_heads * head_dim;
  const int kv_head = min(head_index / max(kv_group_size, 1), n_kv_heads - 1);
  const int half_dim = head_dim / 2;
  const float scale = rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
  const int first_key = max(0, token_index - sliding_window + 1);

  extern __shared__ float shared_scores[];

  for (int source_index = first_key + thread_index;
       source_index <= token_index;
       source_index += blockDim.x) {
    float dot = 0.0f;

    for (int pair = 0; pair < half_dim; ++pair) {
      const float exponent = (2.0f * static_cast<float>(pair)) /
                             fmaxf(static_cast<float>(head_dim), 1.0f);
      const float frequency = powf(theta, -exponent);
      const float query_angle = static_cast<float>(token_index) * frequency;
      const float key_angle = static_cast<float>(source_index) * frequency;
      const float query_cos = cosf(query_angle);
      const float query_sin = sinf(query_angle);
      const float key_cos = cosf(key_angle);
      const float key_sin = sinf(key_angle);

      const int query_base = token_index * d_model + head_index * head_dim;
      const int key_base = source_index * 2 * kv_dim + kv_head * head_dim;

      const float q0 = q_flat[query_base + pair];
      const float q1 = q_flat[query_base + half_dim + pair];
      const float k0 = kv_flat[key_base + pair];
      const float k1 = kv_flat[key_base + half_dim + pair];

      const float q_rot0 = q0 * query_cos - q1 * query_sin;
      const float q_rot1 = q0 * query_sin + q1 * query_cos;
      const float k_rot0 = k0 * key_cos - k1 * key_sin;
      const float k_rot1 = k0 * key_sin + k1 * key_cos;

      dot += q_rot0 * k_rot0 + q_rot1 * k_rot1;
    }

    if ((head_dim & 1) != 0) {
      const int last_dim = head_dim - 1;
      const int query_index =
          token_index * d_model + head_index * head_dim + last_dim;
      const int key_index =
          source_index * 2 * kv_dim + kv_head * head_dim + last_dim;
      dot += q_flat[query_index] * kv_flat[key_index];
    }

    shared_scores[source_index] = dot * scale;
  }

  __syncthreads();

  // Block-parallel softmax + output (replaces the single-thread serial path so
  // the whole block, not thread 0 alone, does the O(n_keys*head_dim) work).
  // Math is equivalent: max is order-independent; the per-dim output is summed
  // over sources in the same order as before (bit-identical per dim); only the
  // softmax denominator's reduction order changes (within parity tolerance).
  // redbuf is static shared sized to the launcher's fixed 128-thread block.
  __shared__ float redbuf[128];
  const int n_keys = token_index + 1;

  float local_max = -1e30f;
  for (int s = first_key + thread_index; s < n_keys; s += blockDim.x) {
    local_max = fmaxf(local_max, shared_scores[s]);
  }
  redbuf[thread_index] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      redbuf[thread_index] =
          fmaxf(redbuf[thread_index], redbuf[thread_index + stride]);
    }
    __syncthreads();
  }
  const float max_score = redbuf[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int s = first_key + thread_index; s < n_keys; s += blockDim.x) {
    const float stabilized = expf(shared_scores[s] - max_score);
    shared_scores[s] = stabilized;
    local_sum += stabilized;
  }
  redbuf[thread_index] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      redbuf[thread_index] += redbuf[thread_index + stride];
    }
    __syncthreads();
  }
  const float denom = fmaxf(redbuf[0], 1e-9f);
  __syncthreads();

  for (int dim = thread_index; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int source_index = first_key; source_index < n_keys; ++source_index) {
      const int value_base =
          source_index * 2 * kv_dim + kv_dim + kv_head * head_dim;
      acc += (shared_scores[source_index] / denom) * kv_flat[value_base + dim];
    }
    out[token_index * d_model + head_index * head_dim + dim] = acc;
  }
}

extern "C" void launch_gqa_causal_attention_kernel(const float *q_flat,
                                                   const float *kv_flat,
                                                   float *out, int seq_len,
                                                   int d_model, int n_heads,
                                                   int n_kv_heads,
                                                   int head_dim,
                                                   int kv_group_size,
                                                   float theta,
                                                   int sliding_window) {
  const int threads = 128;
  const dim3 grid(seq_len, n_heads);
  const size_t shared_bytes = static_cast<size_t>(seq_len) * sizeof(float);
  gqa_causal_attention_kernel<<<grid, threads, shared_bytes, nsos::gpu::current_stream()>>>(
      q_flat, kv_flat, out, seq_len, d_model, n_heads, n_kv_heads, head_dim,
      kv_group_size, theta, sliding_window);
}

__global__ void batched_gqa_causal_attention_kernel(const float *q_flat,
                                                    const float *kv_flat,
                                                    float *out,
                                                    int seq_len,
                                                    int d_model,
                                                    int n_heads,
                                                    int n_kv_heads,
                                                    int head_dim,
                                                    int kv_group_size,
                                                    float theta,
                                                    int sliding_window,
                                                    size_t q_batch_stride,
                                                    size_t kv_batch_stride) {
  const int batch_index = blockIdx.z;
  const int token_index = blockIdx.x;
  const int head_index = blockIdx.y;
  const int thread_index = threadIdx.x;
  const int kv_dim = n_kv_heads * head_dim;
  const int kv_head = min(head_index / max(kv_group_size, 1), n_kv_heads - 1);
  const int half_dim = head_dim / 2;
  const float scale = rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
  const int first_key = max(0, token_index - sliding_window + 1);

  const float *q_batch = q_flat + static_cast<size_t>(batch_index) * q_batch_stride;
  const float *kv_batch = kv_flat + static_cast<size_t>(batch_index) * kv_batch_stride;
  float *out_batch = out + static_cast<size_t>(batch_index) * q_batch_stride;

  extern __shared__ float shared_scores[];

  for (int source_index = first_key + thread_index;
       source_index <= token_index;
       source_index += blockDim.x) {
    float dot = 0.0f;

    for (int pair = 0; pair < half_dim; ++pair) {
      const float exponent = (2.0f * static_cast<float>(pair)) /
                             fmaxf(static_cast<float>(head_dim), 1.0f);
      const float frequency = powf(theta, -exponent);
      const float query_angle = static_cast<float>(token_index) * frequency;
      const float key_angle = static_cast<float>(source_index) * frequency;
      const float query_cos = cosf(query_angle);
      const float query_sin = sinf(query_angle);
      const float key_cos = cosf(key_angle);
      const float key_sin = sinf(key_angle);

      const int query_base = token_index * d_model + head_index * head_dim;
      const int key_base = source_index * 2 * kv_dim + kv_head * head_dim;

      const float q0 = q_batch[query_base + pair];
      const float q1 = q_batch[query_base + half_dim + pair];
      const float k0 = kv_batch[key_base + pair];
      const float k1 = kv_batch[key_base + half_dim + pair];

      const float q_rot0 = q0 * query_cos - q1 * query_sin;
      const float q_rot1 = q0 * query_sin + q1 * query_cos;
      const float k_rot0 = k0 * key_cos - k1 * key_sin;
      const float k_rot1 = k0 * key_sin + k1 * key_cos;

      dot += q_rot0 * k_rot0 + q_rot1 * k_rot1;
    }

    if ((head_dim & 1) != 0) {
      const int last_dim = head_dim - 1;
      const int query_index =
          token_index * d_model + head_index * head_dim + last_dim;
      const int key_index =
          source_index * 2 * kv_dim + kv_head * head_dim + last_dim;
      dot += q_batch[query_index] * kv_batch[key_index];
    }

    shared_scores[source_index] = dot * scale;
  }

  __syncthreads();

  // Block-parallel softmax + output (see the single-batch kernel above for the
  // equivalence argument).  redbuf is static shared sized to the 128-thread block.
  __shared__ float redbuf[128];
  const int n_keys = token_index + 1;

  float local_max = -1e30f;
  for (int s = first_key + thread_index; s < n_keys; s += blockDim.x) {
    local_max = fmaxf(local_max, shared_scores[s]);
  }
  redbuf[thread_index] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      redbuf[thread_index] =
          fmaxf(redbuf[thread_index], redbuf[thread_index + stride]);
    }
    __syncthreads();
  }
  const float max_score = redbuf[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int s = first_key + thread_index; s < n_keys; s += blockDim.x) {
    const float stabilized = expf(shared_scores[s] - max_score);
    shared_scores[s] = stabilized;
    local_sum += stabilized;
  }
  redbuf[thread_index] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      redbuf[thread_index] += redbuf[thread_index + stride];
    }
    __syncthreads();
  }
  const float denom = fmaxf(redbuf[0], 1e-9f);
  __syncthreads();

  for (int dim = thread_index; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int source_index = first_key; source_index < n_keys; ++source_index) {
      const int value_base =
          source_index * 2 * kv_dim + kv_dim + kv_head * head_dim;
      acc += (shared_scores[source_index] / denom) * kv_batch[value_base + dim];
    }
    out_batch[token_index * d_model + head_index * head_dim + dim] = acc;
  }
}

extern "C" void launch_batched_gqa_causal_attention_kernel(const float *q_flat,
                                                           const float *kv_flat,
                                                           float *out,
                                                           int batch_size,
                                                           int seq_len,
                                                           int d_model,
                                                           int n_heads,
                                                           int n_kv_heads,
                                                           int head_dim,
                                                           int kv_group_size,
                                                           float theta,
                                                           int sliding_window) {
  const int threads = 128;
  const dim3 grid(seq_len, n_heads, batch_size);
  const size_t shared_bytes = static_cast<size_t>(seq_len) * sizeof(float);
  const size_t q_batch_stride =
      static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
  const size_t kv_batch_stride =
      static_cast<size_t>(seq_len) * static_cast<size_t>(2 * n_kv_heads * head_dim);
  batched_gqa_causal_attention_kernel<<<grid, threads, shared_bytes, nsos::gpu::current_stream()>>>(
      q_flat, kv_flat, out, seq_len, d_model, n_heads, n_kv_heads, head_dim,
      kv_group_size, theta, sliding_window, q_batch_stride, kv_batch_stride);
}

template <typename KV>
__global__ void gqa_append_kv_cache_kernel(const float *kv_flat, KV *key_cache,
                                           KV *value_cache, int cache_row,
                                           int n_kv_heads, int head_dim,
                                           float theta, const int* position, int capacity = 0) {
  if (position) cache_row = *position;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int kv_dim = n_kv_heads * head_dim;
  kv_flat += static_cast<size_t>(blockIdx.y) * 2 * kv_dim;
  key_cache += static_cast<size_t>(blockIdx.y) * capacity * kv_dim;
  value_cache += static_cast<size_t>(blockIdx.y) * capacity * kv_dim;
  if (idx >= kv_dim) {
    return;
  }

  const int head = idx / head_dim;
  const int dim = idx % head_dim;
  const int half_dim = head_dim / 2;
  const int src_base = head * head_dim;
  const int dst_base = cache_row * kv_dim + src_base;

  float rotated_value = kv_flat[src_base + dim];
  if (dim < half_dim) {
    const float exponent = (2.0f * static_cast<float>(dim)) /
                           fmaxf(static_cast<float>(head_dim), 1.0f);
    const float frequency = powf(theta, -exponent);
    const float angle = static_cast<float>(cache_row) * frequency;
    const float cos_v = cosf(angle);
    const float sin_v = sinf(angle);
    const float x0 = kv_flat[src_base + dim];
    const float x1 = kv_flat[src_base + half_dim + dim];
    rotated_value = x0 * cos_v - x1 * sin_v;
  } else if (dim < 2 * half_dim) {
    const int pair = dim - half_dim;
    const float exponent = (2.0f * static_cast<float>(pair)) /
                           fmaxf(static_cast<float>(head_dim), 1.0f);
    const float frequency = powf(theta, -exponent);
    const float angle = static_cast<float>(cache_row) * frequency;
    const float cos_v = cosf(angle);
    const float sin_v = sinf(angle);
    const float x0 = kv_flat[src_base + pair];
    const float x1 = kv_flat[src_base + half_dim + pair];
    rotated_value = x0 * sin_v + x1 * cos_v;
  }

  key_cache[dst_base + dim] = rotated_value;
  value_cache[dst_base + dim] = kv_flat[kv_dim + src_base + dim];
}

extern "C" void launch_gqa_append_kv_cache_kernel(const float *kv_flat,
                                                  float *key_cache,
                                                  float *value_cache,
                                                  int cache_row,
                                                  int n_kv_heads,
                                                  int head_dim,
                                                  float theta) {
  const int kv_dim = n_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(kv_dim, threads);
  gqa_append_kv_cache_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      kv_flat, key_cache, value_cache, cache_row, n_kv_heads, head_dim, theta,
      nsos::gpu::decode_position());
}

__global__ void gqa_cached_attention_decode_kernel(const float *q_flat,
                                                   const float *key_cache,
                                                   const float *value_cache,
                                                   float *out,
                                                   int cached_tokens,
                                                   int d_model,
                                                   int n_heads,
                                                   int n_kv_heads,
                                                   int head_dim,
                                                   int kv_group_size,
                                                   float theta,
                                                   int sliding_window) {
  const int head_index = blockIdx.x;
  const int thread_index = threadIdx.x;
  const int kv_dim = n_kv_heads * head_dim;
  const int kv_head = min(head_index / max(kv_group_size, 1), n_kv_heads - 1);
  const int half_dim = head_dim / 2;
  const float scale = rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
  const int query_pos = max(cached_tokens - 1, 0);
  const int first_key = max(0, cached_tokens - sliding_window);

  extern __shared__ float shared_scores[];

  for (int source_index = first_key + thread_index;
       source_index < cached_tokens;
       source_index += blockDim.x) {
    float dot = 0.0f;
    for (int pair = 0; pair < half_dim; ++pair) {
      const float exponent = (2.0f * static_cast<float>(pair)) /
                             fmaxf(static_cast<float>(head_dim), 1.0f);
      const float frequency = powf(theta, -exponent);
      const float angle = static_cast<float>(query_pos) * frequency;
      const float cos_v = cosf(angle);
      const float sin_v = sinf(angle);

      const int query_base = head_index * head_dim;
      const int key_base = source_index * kv_dim + kv_head * head_dim;
      const float q0 = q_flat[query_base + pair];
      const float q1 = q_flat[query_base + half_dim + pair];
      const float q_rot0 = q0 * cos_v - q1 * sin_v;
      const float q_rot1 = q0 * sin_v + q1 * cos_v;

      dot += q_rot0 * key_cache[key_base + pair] +
             q_rot1 * key_cache[key_base + half_dim + pair];
    }

    if ((head_dim & 1) != 0) {
      const int last_dim = head_dim - 1;
      dot += q_flat[head_index * head_dim + last_dim] *
             key_cache[source_index * kv_dim + kv_head * head_dim + last_dim];
    }

    shared_scores[source_index] = dot * scale;
  }

  __syncthreads();

  // Block-parallel softmax + output (was a single-thread serial path).  Same math
  // as the prefill kernel's parallelization (commit ca7c4a9): block-reduce max,
  // exp in place + block-reduce denom, then per-dim parallel output.  The denom
  // reduction reorders the sum (within softmax parity tolerance); the per-dim
  // accumulation keeps the source order, so outputs match to fp tolerance.
  __shared__ float redbuf[128];
  __shared__ float s_max;
  __shared__ float s_denom;

  float local_max = -1e30f;
  for (int source_index = first_key + thread_index;
       source_index < cached_tokens;
       source_index += blockDim.x) {
    local_max = fmaxf(local_max, shared_scores[source_index]);
  }
  redbuf[thread_index] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      redbuf[thread_index] =
          fmaxf(redbuf[thread_index], redbuf[thread_index + stride]);
    }
    __syncthreads();
  }
  if (thread_index == 0) s_max = redbuf[0];
  __syncthreads();
  const float max_score = s_max;

  float local_sum = 0.0f;
  for (int source_index = first_key + thread_index;
       source_index < cached_tokens;
       source_index += blockDim.x) {
    const float stabilized = expf(shared_scores[source_index] - max_score);
    shared_scores[source_index] = stabilized;
    local_sum += stabilized;
  }
  redbuf[thread_index] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      redbuf[thread_index] += redbuf[thread_index + stride];
    }
    __syncthreads();
  }
  if (thread_index == 0) s_denom = fmaxf(redbuf[0], 1e-9f);
  __syncthreads();
  const float denom = s_denom;

  for (int dim = thread_index; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int source_index = first_key; source_index < cached_tokens; ++source_index) {
      const int value_base = source_index * kv_dim + kv_head * head_dim;
      acc +=
          (shared_scores[source_index] / denom) * value_cache[value_base + dim];
    }
    out[head_index * head_dim + dim] = acc;
  }
}

// Bounded LDS online softmax. Keys are already rotated when appended. Each
// partition owns its output and publishes (max, denominator, numerator), which
// permits a stable, deterministic merge without attention-score storage O(T).
template <typename KV>
__global__ __launch_bounds__(128) void gqa_decode_tiled_kernel(
    const float* query, const KV* keys, const KV* values, float* partials,
    int tokens, int heads, int kv_heads, int dim, int group,
    float theta, int window, int partitions, const int* position, int capacity = 0) {
  if (position) tokens = *position + 1;
  const int h = blockIdx.x % heads, batch = blockIdx.x / heads;
  const int part = blockIdx.y, tid = threadIdx.x;
  const int kvh = min(h / max(group, 1), kv_heads - 1);
  const int width = kv_heads * dim;
  query += static_cast<size_t>(batch) * heads * dim;
  keys += static_cast<size_t>(batch) * capacity * width;
  values += static_cast<size_t>(batch) * capacity * width;
  const int first = max(0, tokens - max(window, 0));
  const int span = (tokens - first + partitions - 1) / partitions;
  const int begin = min(tokens, first + part * span);
  const int end = min(tokens, begin + span);
  float* dst = partials + (static_cast<size_t>(blockIdx.x) * partitions + part) * (dim + 2);
  extern __shared__ float rotated[];
  __shared__ float scores[128], reduction[128], maximum, denominator, alpha, next_max;
  for (int d = tid; d < dim; d += 128) {
    const int half = dim / 2;
    float q = query[h * dim + d];
    if (d < half * 2) {
      const int pair = d % half;
      const float angle = float(tokens - 1) * powf(theta, -2.0f * pair / dim);
      const float c = cosf(angle), s = sinf(angle);
      const float q0 = query[h * dim + pair], q1 = query[h * dim + half + pair];
      q = d < half ? q0 * c - q1 * s : q0 * s + q1 * c;
    }
    rotated[d] = q;
    dst[d] = 0.0f;
  }
  if (tid == 0) { maximum = -INFINITY; denominator = 0.0f; }
  __syncthreads();
  for (int tile = begin; tile < end; tile += 128) {
    const int token = tile + tid;
    float score = -INFINITY;
    if (token < end) {
      float dot = 0.0f;
      const size_t base = static_cast<size_t>(token) * width + kvh * dim;
      for (int d = 0; d < dim; ++d) dot += rotated[d] * float(keys[base + d]);
      score = dot * rsqrtf(float(dim));
    }
    reduction[tid] = score;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
      if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      __syncthreads();
    }
    if (tid == 0) {
      next_max = fmaxf(maximum, reduction[0]);
      alpha = denominator == 0.0f ? 0.0f : expf(maximum - next_max);
    }
    __syncthreads();
    const float probability = token < end ? expf(score - next_max) : 0.0f;
    scores[tid] = probability;
    reduction[tid] = probability;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
      if (tid < stride) reduction[tid] += reduction[tid + stride];
      __syncthreads();
    }
    if (tid == 0) { denominator = denominator * alpha + reduction[0]; maximum = next_max; }
    for (int d = tid; d < dim; d += 128) {
      float sum = dst[d] * alpha;
      for (int j = 0; j < min(128, end - tile); ++j)
        sum += scores[j] * float(values[static_cast<size_t>(tile + j) * width + kvh * dim + d]);
      dst[d] = sum;
    }
    __syncthreads();
  }
  if (tid == 0) { dst[dim] = maximum; dst[dim + 1] = denominator; }
}

__global__ void gqa_decode_merge_kernel(const float* partials, float* output,
                                      int dim, int partitions) {
  const int h = blockIdx.x;
  const float* row = partials + static_cast<size_t>(h) * partitions * (dim + 2);
  float maximum = -INFINITY;
  for (int p = 0; p < partitions; ++p) maximum = fmaxf(maximum, row[p * (dim + 2) + dim]);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float numerator = 0.0f, denominator = 0.0f;
    for (int p = 0; p < partitions; ++p) {
      const float* part = row + p * (dim + 2);
      if (part[dim + 1] == 0.0f) continue;
      const float scale = expf(part[dim] - maximum);
      numerator += part[d] * scale;
      denominator += part[dim + 1] * scale;
    }
    output[h * dim + d] = denominator > 0.0f ? numerator / denominator : 0.0f;
  }
}

extern "C" void launch_gqa_cached_attention_decode_kernel(
    const float *q_flat, const float *key_cache, const float *value_cache,
    float *out, int cached_tokens, int d_model, int n_heads, int n_kv_heads,
    int head_dim, int kv_group_size, float theta, int sliding_window) {
  const char* reference = std::getenv("NSOS_GPU_ATTENTION_REFERENCE");
  if (reference && reference[0] == '1' && !nsos::gpu::decode_position()) {
    if (cached_tokens > 12000) throw std::runtime_error("Reference attention exceeds bounded LDS budget");
    gqa_cached_attention_decode_kernel<<<n_heads, 128, size_t(cached_tokens) * sizeof(float), nsos::gpu::current_stream()>>>(
        q_flat, key_cache, value_cache, out, cached_tokens, d_model, n_heads,
        n_kv_heads, head_dim, kv_group_size, theta, sliding_window);
    return;
  }
  const int parts = (std::min)(32, (std::max)(1, (cached_tokens + 1023) / 1024));
  nsos::gpu::record_dispatch(nsos::gpu::DispatchPath::TiledAttention);
  auto* partials = static_cast<float*>(nsos::gpu::current_execution_context().reserve(
      nsos::gpu::WorkspaceSlot::AttentionPartials, nsos::gpu::StorageType::Float32,
      size_t(n_heads) * parts * (head_dim + 2)));
  gqa_decode_tiled_kernel<<<dim3(n_heads, parts), 128, size_t(head_dim) * sizeof(float), nsos::gpu::current_stream()>>>(
      q_flat, key_cache, value_cache, partials, cached_tokens, n_heads, n_kv_heads,
      head_dim, kv_group_size, theta, sliding_window, parts, nsos::gpu::decode_position());
  gqa_decode_merge_kernel<<<n_heads, 128, 0, nsos::gpu::current_stream()>>>(partials, out, head_dim, parts);
}

template <typename KV>
void gqa_append_decode_batch(const float* query, const float* kv, KV* keys, KV* values,
    float* output, int batch, int capacity, int position, int heads, int kv_heads,
    int dim, int group, float theta, int window) {
  if (batch < 1 || capacity <= position || position < 0 || dim < 1 || dim > 8192)
    throw std::invalid_argument("Invalid batched attention cache geometry");
  const int parts = (std::min)(32, (position + 1024) / 1024);
  const int all_heads = batch * heads;
  auto* partials = static_cast<float*>(nsos::gpu::current_execution_context().reserve(
      nsos::gpu::WorkspaceSlot::AttentionPartials, nsos::gpu::StorageType::Float32,
      size_t(all_heads) * parts * (dim + 2)));
  gqa_append_kv_cache_kernel<<<dim3(nsos::gpu::ceil_div_positive(kv_heads * dim, 256), batch), 256, 0, nsos::gpu::current_stream()>>>(
      kv, keys, values, position, kv_heads, dim, theta, nsos::gpu::decode_position(), capacity);
  gqa_decode_tiled_kernel<<<dim3(all_heads, parts), 128, size_t(dim) * sizeof(float), nsos::gpu::current_stream()>>>(
      query, keys, values, partials, position + 1, heads, kv_heads, dim, group,
      theta, window, parts, nsos::gpu::decode_position(), capacity);
  gqa_decode_merge_kernel<<<all_heads, 128, 0, nsos::gpu::current_stream()>>>(partials, output, dim, parts);
}

extern "C" void launch_gqa_append_decode_batch(const float* query, const float* kv,
    void* keys, void* values, float* output, int batch, int capacity, int position,
    int heads, int kv_heads, int dim, int group, float theta, int window, int fp16) {
  nsos::gpu::record_dispatch(fp16 ? nsos::gpu::DispatchPath::CompactAttention : nsos::gpu::DispatchPath::TiledAttention);
  if (fp16)
    gqa_append_decode_batch(query, kv, static_cast<__half*>(keys), static_cast<__half*>(values),
        output, batch, capacity, position, heads, kv_heads, dim, group, theta, window);
  else
    gqa_append_decode_batch(query, kv, static_cast<float*>(keys), static_cast<float*>(values),
        output, batch, capacity, position, heads, kv_heads, dim, group, theta, window);
}

__global__ void kv_half_convert_kernel(const float* source_f32, const __half* source_f16,
    float* target_f32, __half* target_f16, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (target_f16) target_f16[i] = __float2half(source_f32[i]);
  else target_f32[i] = __half2float(source_f16[i]);
}
extern "C" void launch_kv_half_convert(const void* source, void* target, size_t count, int to_half) {
  if (!count) return;
  const size_t blocks = (count + 255) / 256;
  if (blocks > size_t(INT_MAX)) throw std::overflow_error("KV conversion grid overflow");
  kv_half_convert_kernel<<<int(blocks), 256, 0, nsos::gpu::current_stream()>>>(
      to_half ? static_cast<const float*>(source) : nullptr,
      to_half ? nullptr : static_cast<const __half*>(source),
      to_half ? nullptr : static_cast<float*>(target),
      to_half ? static_cast<__half*>(target) : nullptr, count);
}

// =====================================================================
// Decode-time greedy token selection ON-DEVICE (#2 GPU sampler, greedy path).
// Mirrors the host greedy branch in nsos_sdk.cpp::sample_from_host_logits_row:
//   value(t) = banned(t)   ? -inf
//            : repeated(t)  ? (raw[t] >= 0 ? raw[t]/penalty : raw[t]*penalty)
//            : raw[t]
//   banned(t) = seen[t]  (no-repeat-ngram bans + blocked-EOS, scattered by host)
//             | (suppress_control && control[t])  (piece starts with "<|")
// argmax with ties -> LOWEST index (host uses strict '>', keeping the first max).
// repeated/seen/control are uint8[vocab] device masks; any may be null.
// Single block; shared-memory (value,index) reduction.  Removes the per-token
// [vocab] D2H + host scan and keeps selection on-device (graph-capturable).
// =====================================================================
__global__ void decode_greedy_argmax_kernel(const float *raw, int vocab,
                                            const unsigned char *repeated,
                                            const unsigned char *seen,
                                            const unsigned char *control,
                                            int suppress_control, float penalty,
                                            int *out_token) {
  extern __shared__ unsigned char gsa_smem[];
  float *sval = reinterpret_cast<float *>(gsa_smem);
  int *sidx = reinterpret_cast<int *>(sval + blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  float best = -3.0e38f;
  int best_i = -1;
  for (int t = tid; t < vocab; t += static_cast<int>(blockDim.x)) {
    if (seen && seen[t]) continue;
    if (suppress_control && control && control[t]) continue;
    float v = raw[t];
    if (repeated && repeated[t] && penalty > 1.0f) {
      v = (v >= 0.0f) ? (v / penalty) : (v * penalty);
    }
    if (v > best) {
      best = v;
      best_i = t;
    }
  }
  sval[tid] = best;
  sidx[tid] = best_i;
  __syncthreads();
  for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const float ov = sval[tid + s];
      const int oi = sidx[tid + s];
      const bool take =
          (oi >= 0) && (ov > sval[tid] ||
                        (ov == sval[tid] && (sidx[tid] < 0 || oi < sidx[tid])));
      if (take) {
        sval[tid] = ov;
        sidx[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    // sidx[0] == -1 when every candidate is masked: emit the -1 sentinel so the
    // caller falls back to the host path (which scans for the first allowed token
    // then EOS/0), instead of silently forcing token 0.
    *out_token = sidx[0];
  }
}

extern "C" void launch_decode_greedy_argmax(const float *raw, int vocab,
                                            const unsigned char *repeated,
                                            const unsigned char *seen,
                                            const unsigned char *control,
                                            int suppress_control, float penalty,
                                            int *out_token) {
  const int block = 256;
  const size_t smem =
      static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
  decode_greedy_argmax_kernel<<<1, block, smem, nsos::gpu::current_stream()>>>(
      raw, vocab, repeated, seen, control, suppress_control, penalty, out_token);
}

// =====================================================================
// CUDA Graphs (opt-in NSOS_CUDA_GRAPH): decode-step launch-overhead amortization.
// Capture a per-token kernel sequence once, then replay the executable graph each
// token.  Graph-capture validity is a GPU-runtime property, so the mechanism is
// shipped with a runtime self-test that captures + replays a known sequence and
// checks the graphed result against an eager run.
// =====================================================================
__global__ void graph_selftest_add_one_kernel(float *buf, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += 1.0f;
}

extern "C" int cuda_graphs_supported(void) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
      cudaSuccess) {
    return 0;
  }
  // CUDA Graphs are supported on the entire graph runtime API (CUDA 10+); a
  // compute-capable device is the practical gate.
  return major >= 3 ? 1 : 0;
}

extern "C" int cuda_graph_self_test(void) {
  const int n = 1024;
  const int block = 256;
  const int grid = nsos::gpu::ceil_div_positive(n, block);
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float *d_buf = nullptr;
  float *d_eager = nullptr;
  float *d_mid = nullptr;  // allocated DURING capture (capture-safe-alloc proof)
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  int ok = 0;

  if (cudaMalloc(&d_buf, bytes) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_eager, bytes) != cudaSuccess) goto cleanup;
  // Capture on a dedicated NON-BLOCKING stream (CuPy does the same): the legacy
  // default stream cannot be captured, and a non-blocking stream won't serialize
  // against unrelated default-stream work.
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    goto cleanup;
  }

  // Eager reference: zero, then +1 three times -> 3.0 everywhere.
  if (cudaMemsetAsync(d_eager, 0, bytes, stream) != cudaSuccess) {
    goto cleanup;
  }
  for (int k = 0; k < 3; ++k) {
    graph_selftest_add_one_kernel<<<grid, block, 0, stream>>>(d_eager, n);
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) goto cleanup;
  nsos::record_gpu_stream_synchronization();

  // Capture the identical sequence into a graph.  Use the RELAXED capture mode:
  // CuPy defaults to it (cupy/cuda/stream.pyx begin_capture) because a memory
  // pool backed by cudaMalloc/cudaMallocManaged may need to grow mid-capture,
  // which stricter modes forbid.  The real decode graph will allocate from the
  // Tensor pool during capture, so the mechanism must be proven under Relaxed.
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) !=
      cudaSuccess) {
    goto cleanup;
  }
  if (cudaMemsetAsync(d_buf, 0, bytes, stream) != cudaSuccess) {
    goto cleanup;
  }
  for (int k = 0; k < 3; ++k) {
    graph_selftest_add_one_kernel<<<grid, block, 0, stream>>>(d_buf, n);
  }
  // Capture-safe allocation: under RELAXED mode a (synchronous) allocation during
  // capture is permitted instead of aborting the capture.  This is the exact
  // pattern the real decode graph relies on when the Tensor pool must grow
  // mid-capture.  Allocate here, then operate on the new buffer in the same graph.
  if (cudaMallocManaged(&d_mid, bytes) != cudaSuccess) {
    (void)cudaStreamEndCapture(stream, &graph);  // abandon the in-flight capture
    goto cleanup;
  }
  if (cudaMemsetAsync(d_mid, 0, bytes, stream) != cudaSuccess) {
    (void)cudaStreamEndCapture(stream, &graph);
    goto cleanup;
  }
  for (int k = 0; k < 3; ++k) {
    graph_selftest_add_one_kernel<<<grid, block, 0, stream>>>(d_mid, n);
  }
  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) {
    (void)cudaGetLastError();
    goto cleanup;
  }
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) goto cleanup;
  // Pre-upload the executable graph (CuPy Graph.upload) so the first launch does
  // not pay the upload cost — relevant when the decode graph replays every token.
  if (cudaGraphUpload(exec, stream) != cudaSuccess) goto cleanup;

  // Replay twice; the in-graph memset makes each replay independent, so the
  // result equals a single eager pass (3.0).  Exercises capture + repeated replay.
  if (cudaGraphLaunch(exec, stream) != cudaSuccess) goto cleanup;
  if (cudaGraphLaunch(exec, stream) != cudaSuccess) goto cleanup;
  if (cudaStreamSynchronize(stream) != cudaSuccess) goto cleanup;
  nsos::record_gpu_stream_synchronization();

  {
    std::vector<float> h_buf(static_cast<size_t>(n));
    std::vector<float> h_eager(static_cast<size_t>(n));
    std::vector<float> h_mid(static_cast<size_t>(n));
    if (cudaMemcpy(h_buf.data(), d_buf, bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      goto cleanup;
    }
    nsos::record_gpu_transfer(nsos::Device::CPU, nsos::Device::GPU, bytes);
    if (cudaMemcpy(h_eager.data(), d_eager, bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      goto cleanup;
    }
    nsos::record_gpu_transfer(nsos::Device::CPU, nsos::Device::GPU, bytes);
    if (cudaMemcpy(h_mid.data(), d_mid, bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      goto cleanup;
    }
    nsos::record_gpu_transfer(nsos::Device::CPU, nsos::Device::GPU, bytes);
    ok = 1;
    for (int i = 0; i < n; ++i) {
      const size_t idx = static_cast<size_t>(i);
      if (fabsf(h_buf[idx] - h_eager[idx]) > 1e-5f ||
          fabsf(h_buf[idx] - 3.0f) > 1e-5f ||
          fabsf(h_mid[idx] - 3.0f) > 1e-5f) {  // buffer allocated mid-capture
        ok = 0;
        break;
      }
    }
  }

cleanup:
  if (exec && cudaGraphExecDestroy(exec) != cudaSuccess) ok = 0;
  if (graph && cudaGraphDestroy(graph) != cudaSuccess) ok = 0;
  if (stream && cudaStreamDestroy(stream) != cudaSuccess) ok = 0;
  if (d_buf && cudaFree(d_buf) != cudaSuccess) ok = 0;
  if (d_eager && cudaFree(d_eager) != cudaSuccess) ok = 0;
  if (d_mid && cudaFree(d_mid) != cudaSuccess) ok = 0;
  (void)cudaGetLastError();
  return ok;
}

// =====================================================================
// D2H per-token copy micro-benchmark: pageable vs pinned staging.
// The decode loop moves ONE int per token device->host (the sampled token).
// Pageable host memory forces the driver through an intermediate staging
// buffer; pinned (cudaMallocHost) memory DMAs directly.  This measures the
// real per-copy latency of both so the pinned-staging decision in
// GpuGreedySampler is backed by a number from the target GPU, not theory.
// =====================================================================
extern "C" int nsos_bench_d2h_copy(int iters, double *pageable_us,
                                   double *pinned_us) {
  if (iters <= 0 || pageable_us == nullptr || pinned_us == nullptr) return 0;
  int ok = 0;
  int *d_val = nullptr;
  int *h_pinned = nullptr;
  int h_pageable = 0;
  cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
  if (cudaMalloc(&d_val, sizeof(int)) != cudaSuccess) goto cleanup;
  if (cudaMallocHost(&h_pinned, sizeof(int)) != cudaSuccess) goto cleanup;
  if (cudaMemset(d_val, 0x2a, sizeof(int)) != cudaSuccess) goto cleanup;
  if (cudaEventCreate(&ev_start) != cudaSuccess) goto cleanup;
  if (cudaEventCreate(&ev_stop) != cudaSuccess) goto cleanup;

  {
    // Warm both paths once so lazy driver setup is outside the timing.
    if (cudaMemcpy(&h_pageable, d_val, sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      goto cleanup;
    }
    nsos::record_gpu_transfer(
        nsos::Device::CPU, nsos::Device::GPU, sizeof(int));
    if (cudaMemcpy(h_pinned, d_val, sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      goto cleanup;
    }
    nsos::record_gpu_transfer(
        nsos::Device::CPU, nsos::Device::GPU, sizeof(int));
    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    nsos::record_gpu_device_synchronization();

    float ms = 0.0f;
    if (cudaEventRecord(ev_start, nsos::gpu::current_stream()) != cudaSuccess) goto cleanup;
    for (int i = 0; i < iters; ++i) {
      if (cudaMemcpy(&h_pageable, d_val, sizeof(int),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto cleanup;
      }
      nsos::record_gpu_transfer(
          nsos::Device::CPU, nsos::Device::GPU, sizeof(int));
    }
    if (cudaEventRecord(ev_stop, nsos::gpu::current_stream()) != cudaSuccess) goto cleanup;
    if (cudaEventSynchronize(ev_stop) != cudaSuccess) goto cleanup;
    nsos::record_gpu_stream_synchronization();
    if (cudaEventElapsedTime(&ms, ev_start, ev_stop) != cudaSuccess) goto cleanup;
    *pageable_us = static_cast<double>(ms) * 1000.0 / iters;

    if (cudaEventRecord(ev_start, nsos::gpu::current_stream()) != cudaSuccess) goto cleanup;
    for (int i = 0; i < iters; ++i) {
      if (cudaMemcpy(h_pinned, d_val, sizeof(int), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        goto cleanup;
      }
      nsos::record_gpu_transfer(
          nsos::Device::CPU, nsos::Device::GPU, sizeof(int));
    }
    if (cudaEventRecord(ev_stop, nsos::gpu::current_stream()) != cudaSuccess) goto cleanup;
    if (cudaEventSynchronize(ev_stop) != cudaSuccess) goto cleanup;
    nsos::record_gpu_stream_synchronization();
    if (cudaEventElapsedTime(&ms, ev_start, ev_stop) != cudaSuccess) goto cleanup;
    *pinned_us = static_cast<double>(ms) * 1000.0 / iters;
    ok = 1;
  }

cleanup:
  if (ev_start && cudaEventDestroy(ev_start) != cudaSuccess) ok = 0;
  if (ev_stop && cudaEventDestroy(ev_stop) != cudaSuccess) ok = 0;
  if (h_pinned && cudaFreeHost(h_pinned) != cudaSuccess) ok = 0;
  if (d_val && cudaFree(d_val) != cudaSuccess) ok = 0;
  (void)cudaGetLastError();
  return ok;
}
