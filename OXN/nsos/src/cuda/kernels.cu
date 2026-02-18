#include "cuda/bitnet_math.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// =========================================================================
// HPC-Optimized CUDA Kernels for NSOS/OXN
// Target: NVIDIA GTX 1050 Ti (SM 6.1, Pascal)
// All kernels use warp-level primitives and shared memory tiling.
// =========================================================================

// Compile-time constants
#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF
#define TILE_DIM 16

// -------------------------------------------------------------------------
// Warp-level reduction primitives
// -------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(FULL_MASK, val, offset);
    val = fmaxf(val, other);
  }
  return val;
}

// Block-wide reduction using shared memory (for blocks larger than 1 warp)
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[32]; // one per warp
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  val = warp_reduce_sum(val);

  if (lane == 0)
    shared[warp_id] = val;
  __syncthreads();

  // First warp reduces across warp results
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
  if (warp_id == 0)
    val = warp_reduce_sum(val);

  return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  val = warp_reduce_max(val);

  if (lane == 0)
    shared[warp_id] = val;
  __syncthreads();

  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
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
  int blocks = (n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(out, a, b, n);
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
  int blocks = (n + threads - 1) / threads;
  add_broadcast_kernel<<<blocks, threads>>>(out, a, b, n, stride);
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
  int blocks = (n + threads - 1) / threads;
  sub_kernel<<<blocks, threads>>>(out, a, b, n);
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
  int blocks = (n + threads - 1) / threads;
  mul_scalar_kernel<<<blocks, threads>>>(out, a, scalar, n);
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
  int blocks = (n + threads - 1) / threads;
  mul_tensor_kernel<<<blocks, threads>>>(out, a, b, n);
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
  int blocks = (n + threads - 1) / threads;
  mul_broadcast_kernel<<<blocks, threads>>>(out, a, b, n, D);
}

// Slice Kernel (3D, Dim=2)
__global__ void slice_kernel_dim2(float *out, const float *in, int d0, int d1,
                                  int d2, int start, int end) {
  int out_d2 = end - start;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = d0 * d1 * out_d2;

  if (idx < total) {
    int k = idx % out_d2;
    int temp = idx / out_d2;
    int j = temp % d1;
    int i = temp / d1;

    int in_idx = i * (d1 * d2) + j * d2 + (start + k);
    out[idx] = in[in_idx];
  }
}

extern "C" void launch_slice_kernel_dim2(float *out, const float *in, int d0,
                                         int d1, int d2, int start, int end) {
  int out_d2 = end - start;
  int total = d0 * d1 * out_d2;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  slice_kernel_dim2<<<blocks, threads>>>(out, in, d0, d1, d2, start, end);
}

// =========================================================================
// HPC RMSNorm — Warp-Reduction Kernel
// One block per row, threads cooperatively reduce sum-of-squares.
// Supports arbitrary n_cols via strided thread access.
// =========================================================================

__global__ void rmsnorm_kernel(float *out, const float *in, int n_rows,
                               int n_cols) {
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
    s_rms = rsqrtf(total_sq / n_cols + 1e-6f);
  }
  __syncthreads();

  float rms_scale = s_rms;

  // Phase 3: Normalize
  for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
    row_out[i] = row_in[i] * rms_scale;
  }
}

extern "C" void launch_rmsnorm_kernel(float *out, const float *in, int n_rows,
                                      int n_cols, int stride_unused,
                                      int block_dim) {
  // One block per row, block_dim threads cooperatively process the row
  int threads_per_block = min(
      256, max(WARP_SIZE, ((n_cols + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
  rmsnorm_kernel<<<n_rows, threads_per_block>>>(out, in, n_rows, n_cols);
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
  int threads_per_block = min(
      256, max(WARP_SIZE, ((n_cols + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
  layernorm_kernel<<<n_rows, threads_per_block>>>(out, in, n_rows, n_cols);
}

// =========================================================================
// HPC Tiled SGEMM — Shared Memory Tiling
// Each thread block computes a TILE_DIM x TILE_DIM tile of the output C.
// Tiles of A and B are loaded cooperatively into shared memory to maximize
// data reuse and minimize global memory traffic.
// =========================================================================

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N) {
  __shared__ float As[TILE_DIM][TILE_DIM];
  __shared__ float Bs[TILE_DIM][TILE_DIM];

  int bx = blockIdx.x;  // Column tile index
  int by = blockIdx.y;  // Row tile index
  int tx = threadIdx.x; // Thread column within tile
  int ty = threadIdx.y; // Thread row within tile

  int row = by * TILE_DIM + ty;
  int col = bx * TILE_DIM + tx;

  float sum = 0.0f;

  // Iterate over tiles along the K dimension
  int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
  for (int t = 0; t < num_tiles; ++t) {
    // Load tile of A into shared memory
    int a_col = t * TILE_DIM + tx;
    if (row < M && a_col < K) {
      As[ty][tx] = A[row * K + a_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    // Load tile of B into shared memory
    int b_row = t * TILE_DIM + ty;
    if (b_row < K && col < N) {
      Bs[ty][tx] = B[b_row * N + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

// Compute partial dot product for this tile
#pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

extern "C" void launch_matmul_kernel(const float *A, const float *B, float *C,
                                     int M, int K, int N, int grid_x,
                                     int grid_y, int block_dim) {
  // Override caller's grid/block with optimal tiled configuration
  dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
  dim3 block(TILE_DIM, TILE_DIM);
  matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
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
      uint32_t w_pack = W[col * k_blocks + kb];
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
        int8_t a0 = (k_start + 0 < K) ? A[row * K + k_start + 0] : 0;
        int8_t a1 = (k_start + 1 < K) ? A[row * K + k_start + 1] : 0;
        int8_t a2 = (k_start + 2 < K) ? A[row * K + k_start + 2] : 0;
        int8_t a3 = (k_start + 3 < K) ? A[row * K + k_start + 3] : 0;
        int a_packed = (a0 & 0xFF) | ((a1 & 0xFF) << 8) | ((a2 & 0xFF) << 16) |
                       ((a3 & 0xFF) << 24);

        // __dp4a: acc += a0*w0 + a1*w1 + a2*w2 + a3*w3
        acc = __dp4a(a_packed, w_packed, acc);
      }
    }
    C[row * N + col] = (float)acc * scale;
  }
}

extern "C" void launch_bitnet_gemm(const int8_t *A, const uint32_t *W, float *C,
                                   int M, int K, int N, float scale, int grid_x,
                                   int grid_y, int block_dim) {
  dim3 grid(grid_x, grid_y);
  dim3 block(block_dim, block_dim);
  bitnet_gemm_kernel<<<grid, block>>>(A, W, C, M, K, N, scale);
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
  dim3 grid((outer + block.x - 1) / block.x, (inner + block.y - 1) / block.y);
  mean_kernel<<<grid, block>>>(out, in, outer, reduce, inner);
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
  int threads_per_block = min(
      256, max(WARP_SIZE, ((inner + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
  softmax_kernel<<<outer, threads_per_block>>>(out, in, outer, inner);
}

// =========================================================================
// HPC Fused Cross-Entropy Kernel
// Fused softmax + log + NLL in a single kernel launch.
// One block per batch element. Block-wide reductions for max and sum-exp.
// Computes both loss and gradient in a single pass.
// =========================================================================

__global__ void fused_cross_entropy_kernel(float *total_loss, float *grad,
                                           const float *logits,
                                           const int *target, int batch,
                                           int vocab) {
  int b = blockIdx.x;
  if (b >= batch)
    return;

  int t = target[b];
  if (t < 0 || t >= vocab)
    return;

  const float *row_logits = logits + b * vocab;
  float *row_grad = grad + b * vocab;

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
    partial_sum += expf(row_logits[v] - max_val);
  }

  float total_sum = block_reduce_sum(partial_sum);

  if (threadIdx.x == 0) {
    s_sum = total_sum;
    // Compute loss for this batch element
    float log_sum_exp = max_val + logf(total_sum);
    atomicAdd(total_loss, log_sum_exp - row_logits[t]);
  }
  __syncthreads();

  float inv_sum = 1.0f / (s_sum + 1e-9f);

  // Phase 3: Compute gradient = softmax(logits) - one_hot(target)
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    float prob = expf(row_logits[v] - max_val) * inv_sum;
    row_grad[v] = prob - (v == t ? 1.0f : 0.0f);
  }
}

extern "C" void launch_fused_cross_entropy(float *d_loss, float *grad,
                                           const float *logits,
                                           const int *target, int batch,
                                           int vocab) {
  int threads_per_block = min(
      256, max(WARP_SIZE, ((vocab + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE));
  fused_cross_entropy_kernel<<<batch, threads_per_block>>>(
      d_loss, grad, logits, target, batch, vocab);
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
  cross_entropy_kernel<<<grid_x, block_dim>>>(d_loss, grad, logits, target,
                                              batch, vocab);
}

// -------------------------------------------------------------------------
// Utility Kernels (unchanged — already simple/efficient)
// -------------------------------------------------------------------------

__global__ void relu_kernel(float *out, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = fmaxf(0.0f, in[idx]);
}

extern "C" void launch_relu_kernel(float *out, const float *in, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  relu_kernel<<<blocks, threads>>>(out, in, n);
}

__global__ void kaiming_uniform_kernel(float *data, int n, float limit,
                                       unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    unsigned long long x = seed + idx;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    float r = (float)(x % 1000000) / 1000000.0f;
    data[idx] = -limit + 2.0f * limit * r;
  }
}

extern "C" void launch_kaiming_uniform_kernel(float *out, int n, float limit,
                                              unsigned long long seed) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kaiming_uniform_kernel<<<blocks, threads>>>(out, n, limit, seed);
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
  int blocks = (n + threads - 1) / threads;
  clamp_kernel<<<blocks, threads>>>(out, in, min_val, max_val, n);
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
  int blocks = (n + threads - 1) / threads;
  norm_kernel<<<blocks, threads>>>(d_sum_sq, in, n);
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
  int blocks = (n + threads - 1) / threads;
  check_stability_kernel<<<blocks, threads>>>(d_found, in, max_val, n);
}

// -------------------------------------------------------------------------
// MoE Top-K Kernel (Simplified for k=2)
// -------------------------------------------------------------------------
__global__ void moe_topk_kernel(const float *logits, float *weights,
                                float *indices, int batch, int num_experts,
                                int k) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch) {
    float max1 = -1e9f;
    float max2 = -1e9f;
    int idx1 = 0;
    int idx2 = 1;

    for (int e = 0; e < num_experts; ++e) {
      float val = logits[b * num_experts + e];
      if (val > max1) {
        max2 = max1;
        idx2 = idx1;
        max1 = val;
        idx1 = e;
      } else if (val > max2) {
        max2 = val;
        idx2 = e;
      }
    }

    // Softmax on top-2
    float sum = expf(max1 - max1) + expf(max2 - max1);
    weights[b * k + 0] = expf(max1 - max1) / sum;
    weights[b * k + 1] = expf(max2 - max1) / sum;
    indices[b * k + 0] = (float)idx1;
    indices[b * k + 1] = (float)idx2;
  }
}

extern "C" void launch_moe_topk_kernel(const float *logits, float *weights,
                                       float *indices, int batch,
                                       int num_experts, int k) {
  int threads = 256;
  int blocks = (batch + threads - 1) / threads;
  moe_topk_kernel<<<blocks, threads>>>(logits, weights, indices, batch,
                                       num_experts, k);
}
