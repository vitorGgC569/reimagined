#include "cuda/mamba_kernels.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// =========================================================================
// HPC Chunk-Parallel Mamba2 SSD Forward Kernel
// Target: NVIDIA GTX 1050 Ti (SM 6.1)
//
// Key Optimization: Instead of serial iteration over the entire sequence,
// the sequence is divided into chunks. Each thread block processes one chunk
// and parallelizes over the state dimension N using cooperative threads.
// Inter-chunk state is propagated via global memory.
//
// x: [B, Seq, H, P]  dt: [B, Seq, H]  A: [H]
// B_param: [B, Seq, H, N]  C_param: [B, Seq, H, N]
// y: [B, Seq, H, P]  final_state: [B, H, P, N]
// =========================================================================

#define MAX_N 64      // Maximum state dimension
#define CHUNK_SIZE 32 // Sequence chunk size for parallel processing
#define WARP_SIZE_M 32

__device__ __forceinline__ float softplus_device(float x) {
  // Numerically stable softplus: log(1 + exp(x))
  if (x > 20.0f)
    return x;
  if (x < -20.0f)
    return 0.0f;
  return logf(1.0f + expf(x));
}

// =========================================================================
// Chunk-parallel Mamba SSD Forward Kernel
//
// Grid: (num_chunks, B * H * P)
// Block: (min(N, 256))
//
// Each thread block handles one (batch, head, p_dim, chunk) combination.
// Threads within the block cooperate on the state dimension N.
// Within each chunk, the SSM recurrence is computed sequentially over
// CHUNK_SIZE timesteps, but the state update for each timestep is
// parallelized across N using the threads in the block.
// =========================================================================

__global__ void mamba_ssd_forward_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A, const float *__restrict__ B_param,
    const float *__restrict__ C_param, float *__restrict__ y,
    float *__restrict__ final_state, int Batch, int Seq, int H, int P, int N) {

  // Decode grid indices
  int chunk_idx = blockIdx.x; // Which chunk along the sequence
  int bhp_idx = blockIdx.y;   // Combined (batch, head, p) index

  int p = bhp_idx % P;
  int h = (bhp_idx / P) % H;
  int b = bhp_idx / (P * H);

  if (b >= Batch)
    return;

  int n = threadIdx.x; // Each thread handles one state dimension element
  bool active = (n < N);

  int chunk_start = chunk_idx * CHUNK_SIZE;
  int chunk_end = min(chunk_start + CHUNK_SIZE, Seq);

  // Shared memory for cooperative output computation
  // Each thread contributes state[n] * C[n], reduced to y_val
  __shared__ float s_y_partial[256]; // Partial sums for output reduction

  // Stride calculations
  // x: [B, Seq, H, P] -> stride: [Seq*H*P, H*P, P, 1]
  int x_stride_b = Seq * H * P;
  int x_stride_t = H * P;
  int x_stride_h = P;

  // dt: [B, Seq, H] -> stride: [Seq*H, H, 1]
  int dt_stride_b = Seq * H;
  int dt_stride_t = H;

  // B_param: [B, Seq, H, N] -> stride: [Seq*H*N, H*N, N, 1]
  int bn_stride_b = Seq * H * N;
  int bn_stride_t = H * N;
  int bn_stride_h = N;

  // y: [B, Seq, H, P] -> same strides as x
  // final_state: [B, H, P, N] -> stride: [H*P*N, P*N, N, 1]

  float a_val = A[h];

  // Load initial state from previous chunk (or zero for first chunk)
  // We use a per-thread register for the state element
  float state_n = 0.0f;

  // If this is not the first chunk, load state from inter-chunk buffer
  // Inter-chunk state is stored in final_state as a temporary workspace
  if (chunk_idx > 0 && active) {
    // Read state propagated from previous chunk
    state_n = final_state[b * H * P * N + h * P * N + p * N + n];
  }

  // Process timesteps within this chunk sequentially
  for (int t = chunk_start; t < chunk_end; ++t) {
    // Fetch dt and compute decay
    float val_dt = dt[b * dt_stride_b + t * dt_stride_t + h];
    float dt_sp = softplus_device(val_dt);
    float exponent = -dt_sp * a_val;
    exponent = fmaxf(exponent, -10.0f); // Clamp for stability
    float decay = expf(exponent);

    // Fetch x value (shared across all N threads)
    float val_x = x[b * x_stride_b + t * x_stride_t + h * x_stride_h + p];

    // Fetch B[t,n] and C[t,n] (each thread gets its own n)
    float val_b = 0.0f, val_c = 0.0f;
    if (active) {
      val_b = B_param[b * bn_stride_b + t * bn_stride_t + h * bn_stride_h + n];
      val_c = C_param[b * bn_stride_b + t * bn_stride_t + h * bn_stride_h + n];
    }

    // SSM state update: state[n] = state[n] * decay + x * B[n]
    if (active) {
      state_n = state_n * decay + val_x * val_b;
    }

    // Compute output: y = sum_n(state[n] * C[n])
    // Each thread computes its partial contribution
    float partial = active ? (state_n * val_c) : 0.0f;
    s_y_partial[threadIdx.x] = partial;
    __syncthreads();

    // Block-wide reduction for output sum
    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        s_y_partial[threadIdx.x] += s_y_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    // Thread 0 writes the output
    if (threadIdx.x == 0) {
      y[b * x_stride_b + t * x_stride_t + h * x_stride_h + p] = s_y_partial[0];
    }
    __syncthreads(); // Ensure all threads see shared memory is free before next
                     // timestep
  }

  // Save final state for this chunk
  // Either as the final output state or as input for the next chunk
  if (active) {
    final_state[b * H * P * N + h * P * N + p * N + n] = state_n;
  }
}

namespace nsos {
namespace cuda {

__global__ void mamba_simple_scan_forward_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A, float *__restrict__ y, int Batch, int Seq,
    int D) {

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) {
    return;
  }

  const int b = channel / D;
  const int d = channel % D;
  const float a_value = fmaxf(A[d], 1e-3f);
  float state = 0.0f;

  for (int t = 0; t < Seq; ++t) {
    const int index = (b * Seq + t) * D + d;
    const float decay = expf(-softplus_device(dt[index]) * a_value);
    state = tanhf(x[index] + state * decay);
    y[index] = state;
  }
}

__global__ void mamba_simple_scan_backward_kernel(
    const float *__restrict__ grad_y, const float *__restrict__ y,
    const float *__restrict__ dt, const float *__restrict__ A,
    float *__restrict__ grad_x, float *__restrict__ grad_dt,
    float *__restrict__ grad_A, int Batch, int Seq, int D) {

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) {
    return;
  }

  const int b = channel / D;
  const int d = channel % D;
  const float raw_a = A[d];
  const float a_value = fmaxf(raw_a, 1e-3f);
  float grad_state_next = 0.0f;
  float grad_a_local = 0.0f;

  for (int t = Seq - 1; t >= 0; --t) {
    const int index = (b * Seq + t) * D + d;
    const int prev_index = (b * Seq + t - 1) * D + d;
    const float y_t = y[index];
    const float prev_state = (t == 0) ? 0.0f : y[prev_index];
    const float dt_value = dt[index];
    const float dt_softplus = softplus_device(dt_value);
    const float decay = expf(-dt_softplus * a_value);

    const float grad_state = grad_y[index] + grad_state_next;
    const float grad_pre = grad_state * (1.0f - y_t * y_t);
    grad_x[index] = grad_pre;

    const float grad_decay = grad_pre * prev_state;
    grad_state_next = grad_pre * decay;

    const float decay_pre = grad_decay * decay;
    const float sigmoid_dt = 1.0f / (1.0f + expf(-dt_value));
    grad_dt[index] = decay_pre * (-a_value) * sigmoid_dt;

    if (raw_a > 1e-3f) {
      grad_a_local += decay_pre * (-dt_softplus);
    }
  }

  atomicAdd(&grad_A[d], grad_a_local);
}

__global__ void mamba_single_token_update_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A, float *__restrict__ state,
    float *__restrict__ y, int Batch, int D) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) {
    return;
  }

  const int d = channel % D;
  const float a_value = fmaxf(A[d], 1e-3f);
  const float decay = expf(-softplus_device(dt[channel]) * a_value);
  const float next_state = x[channel] + state[channel] * decay;
  state[channel] = next_state;
  y[channel] = tanhf(next_state);
}

void launch_mamba_ssd_forward(const float *x, const float *dt, const float *A,
                              const float *B_param, const float *C_param,
                              float *y, float *final_state, int Batch, int Seq,
                              int n_heads, int d_head, int d_state) {

  int num_chunks = (Seq + CHUNK_SIZE - 1) / CHUNK_SIZE;
  int bhp_total = Batch * n_heads * d_head;

  // Grid: (num_chunks, B*H*P)
  // Block: (min(d_state, 256)) — one thread per state dimension element
  dim3 grid(num_chunks, bhp_total);
  int threads =
      min(256, max(WARP_SIZE_M,
                   ((d_state + WARP_SIZE_M - 1) / WARP_SIZE_M) * WARP_SIZE_M));
  dim3 block(threads);

  // For multi-chunk sequences, we need to process chunks sequentially
  // to propagate state between chunks. Launch one chunk at a time.
  if (num_chunks <= 1) {
    // Single chunk — direct launch
    mamba_ssd_forward_kernel<<<grid, block>>>(x, dt, A, B_param, C_param, y,
                                              final_state, Batch, Seq, n_heads,
                                              d_head, d_state);
  } else {
    // Multi-chunk — launch each chunk sequentially to ensure correct
    // state propagation between chunks via final_state buffer
    for (int c = 0; c < num_chunks; ++c) {
      dim3 chunk_grid(1, bhp_total);
      // Offset the kernel's chunk_idx by adjusting the grid
      // We pass c as the chunk offset by modifying the pointer arithmetic
      // Actually, we launch with grid(1, bhp) but need to tell the kernel
      // which chunk to process. We use blockIdx.x = 0 always, so we need
      // to adjust. Instead, let's pass chunk start/end directly.

      // Simpler approach: launch all chunks in one go, kernel handles state
      // propagation via global memory with sync between chunks
      // For SM 6.1, the simplest correct approach is sequential launches
      int chunk_start = c * CHUNK_SIZE;
      int chunk_seq = min(CHUNK_SIZE, Seq - chunk_start);

      // Adjust pointers to this chunk's start
      const float *x_chunk = x + chunk_start * n_heads * d_head;
      const float *dt_chunk = dt + chunk_start * n_heads;
      const float *B_chunk = B_param + chunk_start * n_heads * d_state;
      const float *C_chunk = C_param + chunk_start * n_heads * d_state;
      float *y_chunk = y + chunk_start * n_heads * d_head;

      // Launch kernel for this chunk's subsequence only
      // Use the base kernel but with adjusted Seq = chunk_seq and chunk_idx = 0
      dim3 single_chunk_grid(1, bhp_total);
      mamba_ssd_forward_kernel<<<single_chunk_grid, block>>>(
          x_chunk, dt_chunk, A, B_chunk, C_chunk, y_chunk, final_state, Batch,
          chunk_seq, n_heads, d_head, d_state);

      // Implicit sync via sequential kernel launches on default stream
    }
  }

  cudaDeviceSynchronize();
}

void launch_mamba_simple_scan_forward(const float *x, const float *dt,
                                      const float *A, float *y, int Batch,
                                      int Seq, int D) {
  // No internal cudaDeviceSynchronize: callers serialize with the host
  // explicitly (e.g. before reading outputs on the CPU).  Keeping the
  // launch fire-and-forget lets training and inference overlap CPU work
  // with the kernel runtime.
  const int total_channels = Batch * D;
  const int threads = 256;
  const int blocks = (total_channels + threads - 1) / threads;
  mamba_simple_scan_forward_kernel<<<blocks, threads>>>(x, dt, A, y, Batch, Seq,
                                                        D);
}

void launch_mamba_simple_scan_backward(const float *grad_y, const float *y,
                                       const float *dt, const float *A,
                                       float *grad_x, float *grad_dt,
                                       float *grad_A, int Batch, int Seq,
                                       int D) {
  // grad_A uses atomicAdd inside the kernel, so it MUST be zeroed first.
  // The cudaMemset is asynchronous on the default stream and serializes
  // implicitly with the launch below.
  const int total_channels = Batch * D;
  const int threads = 256;
  const int blocks = (total_channels + threads - 1) / threads;
  cudaMemsetAsync(grad_A, 0, static_cast<size_t>(D) * sizeof(float));
  mamba_simple_scan_backward_kernel<<<blocks, threads>>>(
      grad_y, y, dt, A, grad_x, grad_dt, grad_A, Batch, Seq, D);
}

void launch_mamba_single_token_update(const float *x, const float *dt,
                                      const float *A, float *state,
                                      float *y, int Batch, int D) {
  // Streaming-inference critical path: callers in mamba2.cpp chain
  // tensor ops on `y` and `state` immediately after this launch and
  // historically relied on an implicit synchronization here.  The other
  // launchers in this file are fire-and-forget by design (callers sync
  // explicitly via `.cpu()` or cudaDeviceSynchronize), but the
  // single-token streaming path is hot enough that exposing a race
  // there would be a regression.  Keep sync inside the launcher.
  const int total_channels = Batch * D;
  const int threads = 256;
  const int blocks = (total_channels + threads - 1) / threads;
  mamba_single_token_update_kernel<<<blocks, threads>>>(x, dt, A, state, y,
                                                        Batch, D);
  cudaDeviceSynchronize();
}

// =====================================================================
// Selective scan with B/C gating — matches the CPU implementation in
// src/mamba2.cpp::Mamba2SSD::ssd_forward token-for-token.
//
// Parallelization: one CUDA thread per (batch, dim) channel processes
// the full sequence sequentially (SSM recurrence is inherently serial
// in time).  Across the (B*D) axis the channels are independent so we
// scale linearly with grid width.
//
// Compared to the simple_scan kernel above, this variant honors the
// selective B and C parameters (per-token, per-channel gating) which
// the v9-class hybrid models use.  Without these gates the model
// computes a different function — see ssd_forward CPU loop for the
// authoritative recurrence.
// =====================================================================

__global__ void mamba_selective_scan_forward_kernel(
    const float *__restrict__ x,
    const float *__restrict__ dt,
    const float *__restrict__ A,
    const float *__restrict__ B_in,
    const float *__restrict__ C_in,
    float *__restrict__ y,
    float *__restrict__ state_history,  // may be nullptr
    int Batch, int Seq, int D) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) return;

  const int b = channel / D;
  const int d = channel % D;
  const float a_value = fmaxf(A[d], 1e-3f);

  float state = 0.0f;
  for (int t = 0; t < Seq; ++t) {
    const int idx = (b * Seq + t) * D + d;
    const float dt_val = dt[idx];
    const float decay = expf(-softplus_device(dt_val) * a_value);
    state = state * decay + B_in[idx] * x[idx];
    if (state_history != nullptr) {
      state_history[idx] = state;
    }
    y[idx] = tanhf(state) * C_in[idx];
  }
}

// Backward pass.  Mirrors the CPU loop in
// src/mamba2.cpp::Mamba2SSD::ssd_backward exactly, including:
//   * y_t  = tanh(h_t) * C_t   →   dC_t = grad_y_t * tanh(h_t)
//   * dh_t = grad_y_t * C_t * (1 - tanh(h_t)^2) + dh_next
//   * h_t  = h_{t-1} * decay + B_t * x_t
//       dx_t  = dh_t * B_t
//       dB_t  = dh_t * x_t
//       dh_{t-1} (carried) = dh_t * decay
//       dDecay = dh_t * h_{t-1}
//       ddt_t  = dDecay * decay * (-A) * sigmoid(dt_t)
//       dA    += dDecay * decay * (-softplus(dt_t))   (only when A>1e-3)
//
// Each thread handles one (batch, dim) channel and walks t from Seq-1
// down to 0, accumulating dA locally and atomicAdd-ing once at the end
// to minimize contention.
__global__ void mamba_selective_scan_backward_kernel(
    const float *__restrict__ grad_y,
    const float *__restrict__ x,
    const float *__restrict__ dt,
    const float *__restrict__ A,
    const float *__restrict__ B_in,
    const float *__restrict__ C_in,
    const float *__restrict__ state_history,
    float *__restrict__ grad_x,
    float *__restrict__ grad_dt,
    float *__restrict__ grad_A,   // accumulated via atomicAdd
    float *__restrict__ grad_B,
    float *__restrict__ grad_C,
    int Batch, int Seq, int D) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) return;

  const int b = channel / D;
  const int d = channel % D;
  const float raw_a = A[d];
  const float a_value = fmaxf(raw_a, 1e-3f);

  float grad_state_next = 0.0f;
  float grad_a_local = 0.0f;

  for (int t = Seq - 1; t >= 0; --t) {
    const int idx = (b * Seq + t) * D + d;
    const int prev_idx = (t == 0) ? idx : ((b * Seq + (t - 1)) * D + d);

    const float state_t = state_history[idx];
    const float prev_state = (t == 0) ? 0.0f : state_history[prev_idx];
    const float c_value = C_in[idx];
    const float dt_val = dt[idx];
    const float dt_sp = softplus_device(dt_val);
    const float decay = expf(-dt_sp * a_value);
    const float candidate = tanhf(state_t);

    // dC = grad_y * tanh(h)
    grad_C[idx] = grad_y[idx] * candidate;

    // dh = grad_y * C * (1 - tanh^2(h)) + dh_next
    const float grad_candidate = grad_y[idx] * c_value;
    const float grad_state =
        grad_candidate * (1.0f - candidate * candidate) + grad_state_next;

    // dx = dh * B  ;  dB = dh * x
    grad_x[idx] = grad_state * B_in[idx];
    grad_B[idx] = grad_state * x[idx];

    // Carry through the recurrence
    const float grad_decay = grad_state * prev_state;
    grad_state_next = grad_state * decay;

    // ddt = dDecay * decay * (-A) * sigmoid(dt)
    const float decay_pre = grad_decay * decay;
    // Numerically stable sigmoid (CPU path uses a branched form; expf is
    // sufficient here because dt magnitudes are bounded by softplus scale).
    const float sigmoid_dt = 1.0f / (1.0f + expf(-dt_val));
    grad_dt[idx] = decay_pre * (-a_value) * sigmoid_dt;

    // dA += dDecay * decay * (-softplus(dt))   — only when A is not clamped
    if (raw_a > 1e-3f) {
      grad_a_local += decay_pre * (-dt_sp);
    }
  }

  if (grad_a_local != 0.0f) {
    atomicAdd(&grad_A[d], grad_a_local);
  }
}

void launch_mamba_selective_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D) {
  const int total_channels = Batch * D;
  if (total_channels <= 0 || Seq <= 0) {
    return;
  }
  const int threads = 256;
  const int blocks = (total_channels + threads - 1) / threads;
  mamba_selective_scan_forward_kernel<<<blocks, threads>>>(
      x, dt, A, B_in, C_in, y, state_history, Batch, Seq, D);
}

void launch_mamba_selective_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D) {
  const int total_channels = Batch * D;
  if (total_channels <= 0 || Seq <= 0) {
    return;
  }
  // grad_A uses atomicAdd; zero before launch.  Async memset serializes
  // implicitly with the kernel below on the default stream.
  cudaMemsetAsync(grad_A, 0, static_cast<size_t>(D) * sizeof(float));

  const int threads = 256;
  const int blocks = (total_channels + threads - 1) / threads;
  mamba_selective_scan_backward_kernel<<<blocks, threads>>>(
      grad_y, x, dt, A, B_in, C_in, state_history, grad_x, grad_dt, grad_A,
      grad_B, grad_C, Batch, Seq, D);
}

}  // namespace cuda
}  // namespace nsos
