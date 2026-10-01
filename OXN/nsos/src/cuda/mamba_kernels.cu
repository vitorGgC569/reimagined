#include "cuda/mamba_kernels.cuh"
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "gpu_backend.h"
#include "runtime_execution_identity.h"
#include "training_runtime_policy.h"
#if defined(NSOS_GPU_BACKEND_CUDA)
#include <device_launch_parameters.h>
#endif
#if defined(NSOS_GPU_BACKEND_CUDA)
#include <math_constants.h>
#endif

// Shared CUDA/HIP Mamba kernels. The former experimental chunk-parallel
// implementation was removed because it did not propagate the recurrence
// carry correctly; the live scans below are the correctness baseline on both
// NVIDIA and AMD backends.

#define MAX_N 64      // Maximum state dimension
constexpr int MAMBA_DECAY_TERMS_WIDTH = 5;
// Keep the deterministic nstate carry below the conservative 32 KiB dynamic
// shared-memory floor supported by every production CUDA/HIP target.
constexpr int MAMBA_NSTATE_DETERMINISTIC_MAX_CARRY_FLOATS = 8192;

void mamba_gpu_runtime_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed on " + NSOS_GPU_BACKEND_NAME +
        ": " + cudaGetErrorString(status));
  }
}

__device__ __forceinline__ float mamba_shfl_down(float value, int offset) {
#if defined(NSOS_GPU_BACKEND_HIP)
  return __shfl_down(value, offset);
#else
  return __shfl_down_sync(0xFFFFFFFFu, value, offset);
#endif
}

// Faithful Mamba shares B/C across every P-channel in a head (and, when
// n_groups < n_heads, across multiple heads).  Issuing one global atomic per
// channel caused extreme contention: the production P=64 configuration
// performed 64 atomics to the same address for each (B,t,h,n) contribution.
//
// Aggregate a complete hardware warp/wave first whenever P is aligned to the
// active warpSize, then publish only one partial per warp.  CUDA is warp32;
// RDNA HIP is wave32 and CDNA HIP may be wave64, so using the device builtin
// instead of a vendor constant keeps the same source correct on both stacks.
// The generic atomic path remains available for non-aligned head widths.
__device__ __forceinline__ float mamba_warp_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += mamba_shfl_down(value, offset);
  }
  return value;
}

__device__ __forceinline__ void mamba_channel_reduction_add(
    float *destination, float value, int p, bool warp_aggregate) {
  if (!warp_aggregate) {
    atomicAdd(destination, value);
    return;
  }
  value = mamba_warp_sum(value);
  if (p % warpSize == 0) {
    atomicAdd(destination, value);
  }
}

__device__ __forceinline__ float softplus_device(float x) {
  // Numerically stable softplus: log(1 + exp(x))
  if (x > 20.0f)
    return x;
  if (x < -20.0f)
    return expf(x);
  return logf(1.0f + expf(x));
}

// N1 (must match host nsos::mamba_a_eff in src/mamba2.cpp): A is stored in the
// LOG domain; the effective decay rate is A_eff = exp(A_log) > 0, so the
// recurrence is unconditionally stable and the gradient flows for every channel
// (no 1e-3 clamp / mask).  A_log = 0 â‡’ A_eff = 1 (old A = ones default).
struct MambaDecayTermsDevice {
  float delta;
  float delta_grad;
  float decay;
  float decay_dt_factor;
  float decay_alog_factor;
};

__device__ __forceinline__ MambaDecayTermsDevice mamba_decay_terms_dev(
    float dt_raw, float a_log) {
  const float delta = softplus_device(dt_raw);
  const float delta_grad = dt_raw >= 0.0f
                               ? 1.0f / (1.0f + expf(-dt_raw))
                               : expf(dt_raw) / (1.0f + expf(dt_raw));
  if (!isfinite(dt_raw) || !isfinite(a_log)) {
    return {delta, delta_grad, CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F};
  }
  const float log_delta = dt_raw < -20.0f ? dt_raw : logf(delta);
  const float log_q = log_delta + a_log;
  if (log_q > 50.0f) {
    return {delta, delta_grad, 0.0f, 0.0f, 0.0f};
  }
  const float q = expf(log_q);
  const float decay = expf(-q);
  const float log_sigmoid = dt_raw >= 0.0f
                                ? -log1pf(expf(-dt_raw))
                                : dt_raw - log1pf(expf(dt_raw));
  const float dt_log = log_sigmoid + a_log - q;
  const float dt_factor = dt_log < -110.0f ? 0.0f : expf(dt_log);
  return {delta, delta_grad, decay, dt_factor, q * decay};
}

__device__ __forceinline__ MambaDecayTermsDevice
mamba_load_decay_terms_dev(const float *packed, size_t index) {
  const float *terms =
      packed + index * MAMBA_DECAY_TERMS_WIDTH;
  return {terms[0], terms[1], terms[2], terms[3], terms[4]};
}

__device__ __forceinline__ size_t mamba_faithful_state_index(
    int row, int chan, int n, int inner, int N,
    bool state_major_history) {
  const size_t row_base =
      static_cast<size_t>(row) * inner * N;
  return state_major_history
             ? row_base + static_cast<size_t>(n) * inner + chan
             : row_base + static_cast<size_t>(chan) * N + n;
}

// (mamba_ssd_forward_kernel chunked removido â€” carry inter-chunk quebrado,
// zero callers; os kernels vivos sao selective_scan_* e nstate_*.)

namespace nsos {
namespace cuda {

namespace {

bool environment_toggle_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr || value[0] != '0';
}

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

bool checked_faithful_convdim(int inner, int group_state, int *convdim) {
  if (convdim == nullptr || inner <= 0 || group_state <= 0 ||
      group_state > (INT_MAX - inner) / 2) {
    return false;
  }
  *convdim = inner + 2 * group_state;
  return true;
}

}  // namespace

bool faithful_warp_aggregation_enabled() {
  static const bool enabled =
      environment_toggle_enabled("NSOS_MAMBA_WARP_AGGREGATE");
  return enabled;
}

bool faithful_reduced_conv_enabled() {
  static const bool enabled =
      environment_toggle_enabled("NSOS_MAMBA_REDUCED_CONV");
  return enabled;
}

bool faithful_k4_conv_enabled() {
  static const bool enabled =
      environment_toggle_enabled("NSOS_MAMBA_CONV_K4");
  return enabled;
}

bool faithful_head_channel_geometry_enabled(int channels_per_head) {
  static const bool force_linear = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY");
    return value != nullptr && value[0] == '1';
  }();
  return !force_linear && channels_per_head > 0 &&
         channels_per_head <= 256;
}

bool faithful_deterministic_head_wave_geometry_enabled(
    int channels_per_head, int runtime_warp_size) {
  static const bool enabled =
      environment_toggle_enabled(
          "NSOS_MAMBA_DETERMINISTIC_HEAD_WAVE_GEOMETRY");
  return enabled &&
         faithful_head_channel_geometry_enabled(channels_per_head) &&
         runtime_warp_size > 0 &&
         channels_per_head % runtime_warp_size == 0;
}

bool faithful_deterministic_shared_carry_enabled() {
  static const bool enabled = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_DETERMINISTIC_SHARED_CARRY");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool faithful_precomputed_decay_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("NSOS_MAMBA_PRECOMPUTE_DECAY");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool faithful_state_parallel_backward_enabled() {
  static const bool enabled = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_STATE_PARALLEL_BACKWARD");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool faithful_chunked_backward_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("NSOS_MAMBA_CHUNKED_BACKWARD");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

int faithful_backward_chunk_size() {
  static const int chunk_size = [] {
    const char *value = std::getenv("NSOS_MAMBA_BACKWARD_CHUNK_SIZE");
    if (value == nullptr || value[0] == '\0') return 128;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || end == nullptr || end[0] != '\0' || parsed < 8 ||
        parsed > 256 || (parsed & (parsed - 1)) != 0) {
      throw std::invalid_argument(
          "NSOS_MAMBA_BACKWARD_CHUNK_SIZE must be a power of two in "
          "[8, 256]");
    }
    return static_cast<int>(parsed);
  }();
  return chunk_size;
}

bool faithful_chunk_lds_state_major_enabled() {
  static const bool enabled = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_CHUNK_LDS_STATE_MAJOR");
    if (value == nullptr || value[0] == '\0') return true;
    if (value[0] == '0' && value[1] == '\0') return false;
    if (value[0] == '1' && value[1] == '\0') return true;
    throw std::invalid_argument(
        "NSOS_MAMBA_CHUNK_LDS_STATE_MAJOR must be exactly 0 or 1");
  }();
  return enabled;
}

bool faithful_state_major_history_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("NSOS_MAMBA_STATE_MAJOR_HISTORY");
    if (value == nullptr || value[0] == '\0') return true;
    if (value[0] == '0' && value[1] == '\0') return false;
    if (value[0] == '1' && value[1] == '\0') return true;
    throw std::invalid_argument(
        "NSOS_MAMBA_STATE_MAJOR_HISTORY must be exactly 0 or 1");
  }();
  return enabled;
}

// (mamba_simple_scan_forward_kernel/backward_kernel removidos â€” mortos; a
// variante aplicava tanh DENTRO da recorrÃªncia, divergindo de todos os
// caminhos vivos que carregam estado linear com tanh sÃ³ no readout.)

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
  const MambaDecayTermsDevice terms =
      mamba_decay_terms_dev(dt[channel], A[d]);
  const float dt_scale = terms.delta;
  const float decay = terms.decay;
  const float next_state =
      dt_scale * x[channel] + state[channel] * decay;
  state[channel] = next_state;
  y[channel] = tanhf(next_state);
}

// (launch_mamba_ssd_forward [chunked, carry inter-chunk quebrado] e
// launch_mamba_simple_scan_forward/backward [variante tanh-dentro-da-
// recorrÃªncia, divergente do resto] removidos â€” zero callers; os caminhos
// vivos sÃ£o launch_mamba_selective_scan_* e launch_mamba_nstate_*.)

void launch_mamba_single_token_update(const float *x, const float *dt,
                                      const float *A, float *state,
                                      float *y, int Batch, int D) {
  // Device consumers launched on the same stream are ordered without a
  // device-wide barrier. The former cudaDeviceSynchronize here serialized
  // every decoded token and prevented overlap across otherwise independent
  // host work.
  int total_channels = 0;
  if (!checked_positive_product_to_int(Batch, D, &total_channels)) {
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total_channels, threads);
  mamba_single_token_update_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(x, dt, A, state, y,
                                                        Batch, D);
}

// =====================================================================
// Selective scan with B/C gating â€” matches the CPU implementation in
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
// computes a different function â€” see ssd_forward CPU loop for the
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
    int Batch, int Seq, int D,
    bool linear_readout) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) return;

  const int b = channel / D;
  const int d = channel % D;
  float state = 0.0f;
  for (int t = 0; t < Seq; ++t) {
    const int idx = (b * Seq + t) * D + d;
    const float dt_val = dt[idx];
    const MambaDecayTermsDevice terms =
        mamba_decay_terms_dev(dt_val, A[d]);
    const float dt_scale = terms.delta;
    const float decay = terms.decay;
    state = state * decay + dt_scale * B_in[idx] * x[idx];
    if (state_history != nullptr) {
      state_history[idx] = state;
    }
    // linear_readout=true -> proper diagonal SSM (y = h*C); false -> legacy
    // tanh readout (y = tanh(h)*C).  The recurrence (state) is identical and
    // linear, so the parallel-prefix path stays valid for both.
    y[idx] = (linear_readout ? state : tanhf(state)) * C_in[idx];
  }
}

// Backward pass.  Mirrors the CPU loop in
// src/mamba2.cpp::Mamba2SSD::ssd_backward exactly, including:
//   * y_t  = tanh(h_t) * C_t   â†’   dC_t = grad_y_t * tanh(h_t)
//   * dh_t = grad_y_t * C_t * (1 - tanh(h_t)^2) + dh_next
//   * h_t  = h_{t-1} * decay + softplus(dt_t) * B_t * x_t
//       dx_t  = dh_t * softplus(dt_t) * B_t
//       dB_t  = dh_t * softplus(dt_t) * x_t
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
    int Batch, int Seq, int D,
    bool linear_readout) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_channels = Batch * D;
  if (channel >= total_channels) return;

  const int b = channel / D;
  const int d = channel % D;
  float grad_state_next = 0.0f;
  float grad_a_local = 0.0f;

  for (int t = Seq - 1; t >= 0; --t) {
    const int idx = (b * Seq + t) * D + d;
    const int prev_idx = (t == 0) ? idx : ((b * Seq + (t - 1)) * D + d);

    const float state_t = state_history[idx];
    const float prev_state = (t == 0) ? 0.0f : state_history[prev_idx];
    const float c_value = C_in[idx];
    const float dt_val = dt[idx];
    const MambaDecayTermsDevice terms =
        mamba_decay_terms_dev(dt_val, A[d]);
    const float dt_sp = terms.delta;
    const float decay = terms.decay;
    // Readout candidate: linear (h) for the proper diagonal SSM, tanh(h) for
    // the legacy path.  dcandidate/dh is 1 (linear) or (1 - tanh^2) (legacy).
    const float candidate = linear_readout ? state_t : tanhf(state_t);
    const float dcand = linear_readout ? 1.0f : (1.0f - candidate * candidate);

    // dC = grad_y * candidate
    grad_C[idx] = grad_y[idx] * candidate;

    // dh = grad_y * C * dcandidate + dh_next
    const float grad_candidate = grad_y[idx] * c_value;
    const float grad_state = grad_candidate * dcand + grad_state_next;

    // dx = dh * dt_scale * B  ;  dB = dh * dt_scale * x
    grad_x[idx] = grad_state * dt_sp * B_in[idx];
    grad_B[idx] = grad_state * dt_sp * x[idx];

    // Carry through the recurrence
    const float grad_decay = grad_state * prev_state;
    grad_state_next = grad_state * decay;

    // ddt receives both the discretized input and decay contributions.
    grad_dt[idx] = grad_state * B_in[idx] * x[idx] * terms.delta_grad -
                   grad_decay * terms.decay_dt_factor;

    // N1: dA_log += dDecayÂ·decayÂ·(-softplus(dt))Â·A_eff  (unconditional).
    grad_a_local -= grad_decay * terms.decay_alog_factor;
  }

  if (grad_a_local != 0.0f) {
    atomicAdd(&grad_A[d], grad_a_local);
  }
}

// â”€â”€ Parallel-prefix (associative) selective scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// The forward recurrence h_t = decay_t * h_{t-1}
//                              + softplus(dt_t) * B_t * x_t
// is a first-order AFFINE recurrence, i.e. an associative scan with operator
//   (a_l,b_l) o (a_r,b_r) = (a_l*a_r,  a_r*b_l + b_r),  identity (1,0),
// applied to h_{-1}=0 so that h_t is the b-component of the inclusive scan.
// This kernel does ONE block per (batch,dim) channel and a Hillis-Steele
// inclusive scan over time in shared memory -> O(log Seq) depth instead of the
// O(Seq) per-thread loop of mamba_selective_scan_forward_kernel.
//
// Why a separate, OPT-IN path: the sequential kernel is already parallel over
// Batch*D channels, so for TRAINING batches it saturates the GPU and this adds
// nothing.  The win is the FEW-CHANNEL / LONG-SEQUENCE regime (inference,
// batch=1) where most SMs would sit idle.  Default OFF (NSOS_MAMBA_PARALLEL_SCAN)
// keeps the validated sequential kernel as the bit-reference; the reassociated
// FP summation here differs only in low bits and must clear the 1e-4 parity gate
// (validate on a real GPU per docs/COLAB_GPU_VALIDATION.md) before promotion.
// Requires Seq to fit one block (<= 1024 threads) + 2*Seq floats of shared mem;
// the launcher falls back to the sequential kernel otherwise.
__global__ void mamba_selective_scan_forward_parallel_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A, const float *__restrict__ B_in,
    const float *__restrict__ C_in, float *__restrict__ y,
    float *__restrict__ state_history, int Batch, int Seq, int D,
    bool linear_readout) {
  extern __shared__ float smem[];   // [0,Seq) = a (decay prod), [Seq,2Seq) = b
  float *sa = smem;
  float *sb = smem + Seq;

  const int channel = blockIdx.x;   // one block per (batch,dim) channel
  if (channel >= Batch * D) return; // whole block returns together â€” no divergence
  const int b = channel / D;
  const int d = channel % D;
  const int t = threadIdx.x;        // blockDim.x == Seq, so t in [0,Seq)

  const int idx = (b * Seq + t) * D + d;
  const MambaDecayTermsDevice terms =
      mamba_decay_terms_dev(dt[idx], A[d]);
  const float dt_scale = terms.delta;
  sa[t] = terms.decay;                                 // decay_t
  sb[t] = dt_scale * B_in[idx] * x[idx];               // input term
  __syncthreads();

  // Hillis-Steele inclusive scan with the affine operator.  Double-buffer via
  // two barriers so every thread reads the previous step's values.
  for (int off = 1; off < Seq; off <<= 1) {
    float a_new = sa[t];
    float b_new = sb[t];
    if (t >= off) {
      const float a_prev = sa[t - off];
      const float b_prev = sb[t - off];
      a_new = a_prev * sa[t];           // a_left * a_right
      b_new = sa[t] * b_prev + sb[t];   // a_right * b_left + b_right
    }
    __syncthreads();
    sa[t] = a_new;
    sb[t] = b_new;
    __syncthreads();
  }

  const float h_t = sb[t];  // inclusive-scan b-component == h_t (h_{-1}=0)
  if (state_history != nullptr) {
    state_history[idx] = h_t;
  }
  y[idx] = (linear_readout ? h_t : tanhf(h_t)) * C_in[idx];
}

// Runtime toggle (default from NSOS_MAMBA_PARALLEL_SCAN).  Atomic so a parity
// test or the Python A/B can switch paths within one process; external linkage
// (declared in the .cuh) so those callers can reach it.
static std::atomic<bool> &mamba_parallel_scan_flag() {
  static std::atomic<bool> flag{[] {
    const char *e = std::getenv("NSOS_MAMBA_PARALLEL_SCAN");
    return e != nullptr && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
  }()};
  return flag;
}

void set_mamba_parallel_scan(bool enabled) {
  RuntimeExecutionPolicyMutationGuard mutation;
  mamba_parallel_scan_flag().store(enabled, std::memory_order_relaxed);
}

bool mamba_parallel_scan_enabled() {
  return mamba_parallel_scan_flag().load(std::memory_order_relaxed);
}

void launch_mamba_selective_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D) {
  int total_channels = 0;
  if (Seq <= 0 ||
      !checked_positive_product_to_int(Batch, D, &total_channels)) {
    return;
  }
  // Opt-in O(log Seq) parallel-prefix scan (default OFF -> sequential kernel).
  // Needs Seq to fit in one block; shared mem = 2*Seq floats.
  if (mamba_parallel_scan_enabled() && Seq <= 1024) {
    const int threads = Seq;            // one thread per timestep
    const int blocks = total_channels;  // one block per channel
    const size_t shmem = static_cast<size_t>(2) * static_cast<size_t>(Seq) *
                         sizeof(float);
    mamba_selective_scan_forward_parallel_kernel<<<blocks, threads, shmem, nsos::gpu::current_stream()>>>(
        x, dt, A, B_in, C_in, y, state_history, Batch, Seq, D,
        /*linear_readout=*/false);
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total_channels, threads);
  mamba_selective_scan_forward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      x, dt, A, B_in, C_in, y, state_history, Batch, Seq, D,
      /*linear_readout=*/false);
}

void launch_mamba_selective_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D) {
  int total_channels = 0;
  if (Seq <= 0 ||
      !checked_positive_product_to_int(Batch, D, &total_channels)) {
    return;
  }
  // grad_A uses atomicAdd; zero before launch.  Async memset serializes
  // implicitly with the kernel below on the default stream.
  mamba_gpu_runtime_check(
      cudaMemsetAsync(grad_A, 0, static_cast<size_t>(D) * sizeof(float), nsos::gpu::current_stream()),
      "mamba selective-scan grad_A clear");

  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total_channels, threads);
  mamba_selective_scan_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      grad_y, x, dt, A, B_in, C_in, state_history, grad_x, grad_dt, grad_A,
      grad_B, grad_C, Batch, Seq, D, /*linear_readout=*/false);
}

// â”€â”€ Proper diagonal SSM (linear readout y = h*C) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// GPU-resident path for Mamba2SSD::forward_proper/backward_proper.  Same affine
// recurrence + state_history contract as the legacy selective scan, but with the
// LINEAR readout (no tanh), so the corrected diagonal SSM runs on device instead
// of falling back to host.
void launch_mamba_proper_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D) {
  int total_channels = 0;
  if (Seq <= 0 ||
      !checked_positive_product_to_int(Batch, D, &total_channels)) {
    return;
  }
  // Opt-in O(log Seq) parallel-prefix scan (NSOS_MAMBA_PARALLEL_SCAN), same
  // associative affine recurrence as the diagonal proper path but with the
  // LINEAR readout.  Default OFF keeps the validated sequential kernel.
  if (mamba_parallel_scan_enabled() && Seq <= 1024) {
    const int threads = Seq;
    const int blocks = total_channels;
    const size_t shmem = static_cast<size_t>(2) * static_cast<size_t>(Seq) *
                         sizeof(float);
    mamba_selective_scan_forward_parallel_kernel<<<blocks, threads, shmem, nsos::gpu::current_stream()>>>(
        x, dt, A, B_in, C_in, y, state_history, Batch, Seq, D,
        /*linear_readout=*/true);
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total_channels, threads);
  mamba_selective_scan_forward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      x, dt, A, B_in, C_in, y, state_history, Batch, Seq, D,
      /*linear_readout=*/true);
}

void launch_mamba_proper_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D) {
  int total_channels = 0;
  if (Seq <= 0 ||
      !checked_positive_product_to_int(Batch, D, &total_channels)) {
    return;
  }
  mamba_gpu_runtime_check(
      cudaMemsetAsync(grad_A, 0, static_cast<size_t>(D) * sizeof(float), nsos::gpu::current_stream()),
      "mamba proper-scan grad_A clear");
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total_channels, threads);
  mamba_selective_scan_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      grad_y, x, dt, A, B_in, C_in, state_history, grad_x, grad_dt, grad_A,
      grad_B, grad_C, Batch, Seq, D, /*linear_readout=*/true);
}

// â”€â”€ Causal depthwise conv1d (proper path local token mixing) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// in/out: [Batch, Seq, D] ; weight: [D, K].
//   out[b,t,c] = sum_{j=0..K-1} weight[c,j] * in[b, t-(K-1)+j, c]   (in[<0]=0)
__global__ void conv1d_causal_forward_kernel(
    const float *__restrict__ in, const float *__restrict__ weight,
    float *__restrict__ out, int batch, int seq, int dim, int K) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * seq * dim;
  if (tid >= total) return;
  const int c = tid % dim;
  const int t = (tid / dim) % seq;
  const int b = tid / (dim * seq);
  float acc = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int st = t - (K - 1) + j;
    if (st < 0) continue;
    acc += weight[c * K + j] * in[(b * seq + st) * dim + c];
  }
  out[tid] = acc;
}

__device__ __forceinline__ float mamba_stable_sigmoid(float value) {
  if (value >= 0.0f) {
    const float exponential = expf(-value);
    return 1.0f / (1.0f + exponential);
  }
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

__global__ void conv1d_causal_bias_silu_forward_kernel(
    const float *__restrict__ in, const float *__restrict__ weight,
    const float *__restrict__ bias, float *__restrict__ pre_activation,
    float *__restrict__ activated, int batch, int seq, int dim, int K) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * seq * dim;
  if (tid >= total) return;
  const int c = tid % dim;
  const int t = (tid / dim) % seq;
  const int b = tid / (dim * seq);
  float value = bias[c];
  for (int j = 0; j < K; ++j) {
    const int st = t - (K - 1) + j;
    if (st < 0) continue;
    value += weight[c * K + j] * in[(b * seq + st) * dim + c];
  }
  pre_activation[tid] = value;
  activated[tid] = value * mamba_stable_sigmoid(value);
}

__global__ void conv1d_causal_backward_kernel(
    const float *__restrict__ grad_out, const float *__restrict__ in,
    const float *__restrict__ weight, float *__restrict__ grad_in,
    float *__restrict__ grad_weight, int batch, int seq, int dim, int K) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * seq * dim;
  if (tid >= total) return;
  const int c = tid % dim;
  const int t = (tid / dim) % seq;
  const int b = tid / (dim * seq);
  const float go = grad_out[tid];
  for (int j = 0; j < K; ++j) {
    const int st = t - (K - 1) + j;
    if (st < 0) continue;
    const int sidx = (b * seq + st) * dim + c;
    atomicAdd(&grad_in[sidx], go * weight[c * K + j]);
    atomicAdd(&grad_weight[c * K + j], go * in[sidx]);
  }
}

__global__ void mamba2_faithful_conv_forward_kernel(
    const float *__restrict__ x_in, const float *__restrict__ b_in,
    const float *__restrict__ c_in, const float *__restrict__ weight,
    const float *__restrict__ bias, float *__restrict__ x_pre,
    float *__restrict__ b_pre, float *__restrict__ c_pre,
    float *__restrict__ x_out, float *__restrict__ b_out,
    float *__restrict__ c_out, int seq, int inner, int group_state, int K,
    int convdim, int total) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  const int global_c = tid % convdim;
  const int row = tid / convdim;
  const int t = row % seq;
  const int batch_base = row - t;

  const float *source = x_in;
  float *pre = x_pre;
  float *out = x_out;
  int local_c = global_c;
  int width = inner;
  if (global_c >= inner) {
    local_c = global_c - inner;
    width = group_state;
    if (local_c < group_state) {
      source = b_in;
      pre = b_pre;
      out = b_out;
    } else {
      local_c -= group_state;
      source = c_in;
      pre = c_pre;
      out = c_out;
    }
  }

  float value = bias[global_c];
  for (int j = 0; j < K; ++j) {
    const int source_t = t - (K - 1) + j;
    if (source_t < 0) continue;
    value +=
        weight[static_cast<size_t>(global_c) * K + j] *
        source[static_cast<size_t>(batch_base + source_t) * width + local_c];
  }
  const size_t destination =
      static_cast<size_t>(row) * width + local_c;
  pre[destination] = value;
  out[destination] = value * mamba_stable_sigmoid(value);
}

__global__ void mamba2_faithful_conv_backward_kernel(
    const float *__restrict__ gx_out, const float *__restrict__ gb_out,
    const float *__restrict__ gc_out, const float *__restrict__ x_in,
    const float *__restrict__ b_in, const float *__restrict__ c_in,
    const float *__restrict__ weight, float *__restrict__ gx_in,
    float *__restrict__ gb_in, float *__restrict__ gc_in,
    float *__restrict__ grad_weight, int seq, int inner, int group_state,
    int K, int convdim, int total) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  const int global_c = tid % convdim;
  const int row = tid / convdim;
  const int t = row % seq;
  const int batch_base = row - t;

  const float *grad_output = gx_out;
  const float *source = x_in;
  float *grad_input = gx_in;
  int local_c = global_c;
  int width = inner;
  if (global_c >= inner) {
    local_c = global_c - inner;
    width = group_state;
    if (local_c < group_state) {
      grad_output = gb_out;
      source = b_in;
      grad_input = gb_in;
    } else {
      local_c -= group_state;
      grad_output = gc_out;
      source = c_in;
      grad_input = gc_in;
    }
  }

  const float go =
      grad_output[static_cast<size_t>(row) * width + local_c];
  for (int j = 0; j < K; ++j) {
    const int source_t = t - (K - 1) + j;
    if (source_t < 0) continue;
    const size_t source_index =
        static_cast<size_t>(batch_base + source_t) * width + local_c;
    atomicAdd(&grad_input[source_index],
              go * weight[static_cast<size_t>(global_c) * K + j]);
    atomicAdd(&grad_weight[static_cast<size_t>(global_c) * K + j],
              go * source[source_index]);
  }
}

// One block owns one depthwise channel.  Re-indexing the convolution backward
// by source row makes grad_input exclusive to the current thread, while the
// same (source,output,tap) traversal accumulates grad_weight in registers and
// then in a fixed-order block reduction.  This removes every global atomic and
// every pre-zero memset from the production K<=16 faithful convolution.
__global__ void mamba2_faithful_conv_backward_reduced_kernel(
    const float *__restrict__ gx_out, const float *__restrict__ gb_out,
    const float *__restrict__ gc_out, const float *__restrict__ x_in,
    const float *__restrict__ b_in, const float *__restrict__ c_in,
    const float *__restrict__ weight, float *__restrict__ gx_in,
    float *__restrict__ gb_in, float *__restrict__ gc_in,
    float *__restrict__ grad_weight, int rows, int seq, int inner,
    int group_state, int K, int convdim) {
  const int global_c = blockIdx.x;
  if (global_c >= convdim) return;

  const float *grad_output = gx_out;
  const float *source = x_in;
  float *grad_input = gx_in;
  int local_c = global_c;
  int width = inner;
  if (global_c >= inner) {
    local_c = global_c - inner;
    width = group_state;
    if (local_c < group_state) {
      grad_output = gb_out;
      source = b_in;
      grad_input = gb_in;
    } else {
      local_c -= group_state;
      grad_output = gc_out;
      source = c_in;
      grad_input = gc_in;
    }
  }

  float local_weight[kFaithfulReducedConvMaxKernel];
#pragma unroll
  for (int tap = 0; tap < kFaithfulReducedConvMaxKernel; ++tap) {
    local_weight[tap] = 0.0f;
  }
  for (int source_row = threadIdx.x; source_row < rows;
       source_row += blockDim.x) {
    const int source_t = source_row % seq;
    const int batch_base = source_row - source_t;
    const size_t source_index =
        static_cast<size_t>(source_row) * width + local_c;
    const float source_value = source[source_index];
    float input_gradient = 0.0f;
    for (int tap = 0; tap < K; ++tap) {
      const int output_t = source_t + (K - 1) - tap;
      if (output_t >= seq) continue;
      const size_t output_index =
          static_cast<size_t>(batch_base + output_t) * width + local_c;
      const float go = grad_output[output_index];
      input_gradient +=
          go * weight[static_cast<size_t>(global_c) * K + tap];
      local_weight[tap] += go * source_value;
    }
    grad_input[source_index] = input_gradient;
  }

  __shared__ float reductions[kFaithfulReducedConvMaxKernel][256];
  for (int tap = 0; tap < K; ++tap) {
    reductions[tap][threadIdx.x] = local_weight[tap];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (int tap = 0; tap < K; ++tap) {
        reductions[tap][threadIdx.x] +=
            reductions[tap][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int tap = 0; tap < K; ++tap) {
      grad_weight[static_cast<size_t>(global_c) * K + tap] =
          reductions[tap][0];
    }
  }
}

// Production models use K=4. Keeping a separate compile-time specialization
// reduces the generic K<=16 register array and shared reduction footprint from
// 16 KiB to 4 KiB per block while preserving the exact source-row, tap and
// tree-reduction order of the generic deterministic kernel.
template <int Kernel>
__global__ void mamba2_faithful_conv_backward_fixed_kernel(
    const float *__restrict__ gx_out, const float *__restrict__ gb_out,
    const float *__restrict__ gc_out, const float *__restrict__ x_in,
    const float *__restrict__ b_in, const float *__restrict__ c_in,
    const float *__restrict__ weight, float *__restrict__ gx_in,
    float *__restrict__ gb_in, float *__restrict__ gc_in,
    float *__restrict__ grad_weight, int rows, int seq, int inner,
    int group_state, int convdim) {
  const int global_c = blockIdx.x;
  if (global_c >= convdim) return;

  const float *grad_output = gx_out;
  const float *source = x_in;
  float *grad_input = gx_in;
  int local_c = global_c;
  int width = inner;
  if (global_c >= inner) {
    local_c = global_c - inner;
    width = group_state;
    if (local_c < group_state) {
      grad_output = gb_out;
      source = b_in;
      grad_input = gb_in;
    } else {
      local_c -= group_state;
      grad_output = gc_out;
      source = c_in;
      grad_input = gc_in;
    }
  }

  float local_weight[Kernel];
#pragma unroll
  for (int tap = 0; tap < Kernel; ++tap) local_weight[tap] = 0.0f;
  for (int source_row = threadIdx.x; source_row < rows;
       source_row += blockDim.x) {
    const int source_t = source_row % seq;
    const int batch_base = source_row - source_t;
    const size_t source_index =
        static_cast<size_t>(source_row) * width + local_c;
    const float source_value = source[source_index];
    float input_gradient = 0.0f;
#pragma unroll
    for (int tap = 0; tap < Kernel; ++tap) {
      const int output_t = source_t + (Kernel - 1) - tap;
      if (output_t >= seq) continue;
      const size_t output_index =
          static_cast<size_t>(batch_base + output_t) * width + local_c;
      const float go = grad_output[output_index];
      input_gradient +=
          go * weight[static_cast<size_t>(global_c) * Kernel + tap];
      local_weight[tap] += go * source_value;
    }
    grad_input[source_index] = input_gradient;
  }

  __shared__ float reductions[Kernel][256];
#pragma unroll
  for (int tap = 0; tap < Kernel; ++tap) {
    reductions[tap][threadIdx.x] = local_weight[tap];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int tap = 0; tap < Kernel; ++tap) {
        reductions[tap][threadIdx.x] +=
            reductions[tap][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int tap = 0; tap < Kernel; ++tap) {
      grad_weight[static_cast<size_t>(global_c) * Kernel + tap] =
          reductions[tap][0];
    }
  }
}

__global__ void mamba2_faithful_bias_backward_kernel(
    const float *__restrict__ gx_pre, const float *__restrict__ gb_pre,
    const float *__restrict__ gc_pre, float *__restrict__ grad_bias,
    int rows, int inner, int group_state, int convdim) {
  const int global_c = blockIdx.x;
  if (global_c >= convdim) return;
  const float *source = gx_pre;
  int local_c = global_c;
  int width = inner;
  if (global_c >= inner) {
    local_c = global_c - inner;
    width = group_state;
    if (local_c < group_state) {
      source = gb_pre;
    } else {
      local_c -= group_state;
      source = gc_pre;
    }
  }
  float partial = 0.0f;
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    partial += source[static_cast<size_t>(row) * width + local_c];
  }
  __shared__ float reductions[256];
  reductions[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grad_bias[global_c] = reductions[0];
  }
}

void launch_conv1d_causal_forward(const float *in, const float *weight,
                                  float *out, int batch, int seq, int dim,
                                  int K) {
  int total = 0;
  if (K <= 0 ||
      !checked_positive_product_to_int(batch, seq, dim, &total)) {
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  conv1d_causal_forward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(in, weight, out, batch, seq,
                                                    dim, K);
}

bool launch_conv1d_causal_bias_silu_forward(
    const float *in, const float *weight, const float *bias,
    float *pre_activation, float *activated, int batch, int seq, int dim,
    int K) {
  int total = 0;
  int weight_values = 0;
  if (in == nullptr || weight == nullptr || bias == nullptr ||
      pre_activation == nullptr || activated == nullptr || K <= 0 ||
      !checked_positive_product_to_int(batch, seq, dim, &total) ||
      !checked_positive_product_to_int(dim, K, &weight_values)) {
    return false;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  conv1d_causal_bias_silu_forward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      in, weight, bias, pre_activation, activated, batch, seq, dim, K);
  return true;
}

bool launch_mamba2_faithful_conv_forward(
    const float *x_in, const float *b_in, const float *c_in,
    const float *weight, const float *bias, float *x_pre, float *b_pre,
    float *c_pre, float *x_out, float *b_out, float *c_out, int batch,
    int seq, int inner, int group_state, int K) {
  int convdim = 0;
  int total = 0;
  int weight_values = 0;
  if (x_in == nullptr || b_in == nullptr || c_in == nullptr ||
      weight == nullptr || bias == nullptr || x_pre == nullptr ||
      b_pre == nullptr || c_pre == nullptr || x_out == nullptr ||
      b_out == nullptr || c_out == nullptr || K <= 0 ||
      !checked_faithful_convdim(inner, group_state, &convdim) ||
      !checked_positive_product_to_int(batch, seq, convdim, &total) ||
      !checked_positive_product_to_int(convdim, K, &weight_values)) {
    return false;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba2_faithful_conv_forward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      x_in, b_in, c_in, weight, bias, x_pre, b_pre, c_pre, x_out, b_out,
      c_out, seq, inner, group_state, K, convdim, total);
  return true;
}

void launch_conv1d_causal_backward(const float *grad_out, const float *in,
                                   const float *weight, float *grad_in,
                                   float *grad_weight, int batch, int seq,
                                   int dim, int K) {
  int total = 0;
  if (K <= 0 ||
      !checked_positive_product_to_int(batch, seq, dim, &total)) {
    return;
  }
  // grad_in / grad_weight accumulate via atomicAdd; zero them first (async on
  // the default stream, serializes with the kernel below).
  mamba_gpu_runtime_check(
      cudaMemsetAsync(
          grad_in, 0, static_cast<size_t>(total) * sizeof(float), nsos::gpu::current_stream()),
      "mamba conv grad_input clear");
  mamba_gpu_runtime_check(
      cudaMemsetAsync(
          grad_weight, 0,
          static_cast<size_t>(dim) * static_cast<size_t>(K) * sizeof(float), nsos::gpu::current_stream()),
      "mamba conv grad_weight clear");
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  conv1d_causal_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      grad_out, in, weight, grad_in, grad_weight, batch, seq, dim, K);
}

bool launch_conv1d_causal_backward_deterministic(
    const float *grad_out, const float *in, const float *weight,
    float *grad_in, float *grad_weight, int batch, int seq, int dim, int K) {
  int rows = 0;
  int weight_values = 0;
  if (grad_out == nullptr || in == nullptr || weight == nullptr ||
      grad_in == nullptr || grad_weight == nullptr || dim <= 0 || K <= 0 ||
      K > kFaithfulReducedConvMaxKernel ||
      !checked_positive_product_to_int(batch, seq, &rows) ||
      !checked_positive_product_to_int(dim, K, &weight_values)) {
    return false;
  }
  constexpr int threads = 256;
  if (K == 4 && faithful_k4_conv_enabled()) {
    mamba2_faithful_conv_backward_fixed_kernel<4><<<dim, threads, 0, nsos::gpu::current_stream()>>>(
        grad_out, nullptr, nullptr, in, nullptr, nullptr, weight, grad_in,
        nullptr, nullptr, grad_weight, rows, seq, dim, 0,
        dim);
    return true;
  }
  // The reduced faithful kernel's first channel group is exactly the generic
  // causal depthwise convolution. With convdim==inner, the nullable B/C
  // pointers are structurally unreachable.
  mamba2_faithful_conv_backward_reduced_kernel<<<dim, threads, 0, nsos::gpu::current_stream()>>>(
      grad_out, nullptr, nullptr, in, nullptr, nullptr, weight, grad_in,
      nullptr, nullptr, grad_weight, rows, seq, dim, 0, K,
      dim);
  return true;
}

bool launch_mamba2_faithful_conv_backward(
    const float *gx_out, const float *gb_out, const float *gc_out,
    const float *x_in, const float *b_in, const float *c_in,
    const float *weight, float *gx_in, float *gb_in, float *gc_in,
    float *grad_weight, int batch, int seq, int inner, int group_state,
    int K) {
  int convdim = 0;
  int rows = 0;
  int total = 0;
  int weight_values = 0;
  if (gx_out == nullptr || gb_out == nullptr || gc_out == nullptr ||
      x_in == nullptr || b_in == nullptr || c_in == nullptr ||
      weight == nullptr || gx_in == nullptr || gb_in == nullptr ||
      gc_in == nullptr || grad_weight == nullptr || K <= 0 ||
      !checked_positive_product_to_int(batch, seq, &rows) ||
      !checked_faithful_convdim(inner, group_state, &convdim) ||
      !checked_positive_product_to_int(rows, convdim, &total) ||
      !checked_positive_product_to_int(convdim, K, &weight_values)) {
    return false;
  }
  if (K <= kFaithfulReducedConvMaxKernel &&
      faithful_reduced_conv_enabled()) {
    constexpr int threads = 256;
    if (K == 4 && faithful_k4_conv_enabled()) {
      mamba2_faithful_conv_backward_fixed_kernel<4><<<convdim, threads, 0, nsos::gpu::current_stream()>>>(
          gx_out, gb_out, gc_out, x_in, b_in, c_in, weight, gx_in, gb_in,
          gc_in, grad_weight, rows, seq, inner,
           group_state, convdim);
      return true;
    }
    mamba2_faithful_conv_backward_reduced_kernel<<<convdim, threads, 0, nsos::gpu::current_stream()>>>(
        gx_out, gb_out, gc_out, x_in, b_in, c_in, weight, gx_in, gb_in,
        gc_in, grad_weight, rows, seq, inner,
        group_state, K, convdim);
    return true;
  }
  // Direct Mamba2SSD users are not constrained by ModelConfig's K<=16
  // contract. Preserve a correct generic fallback for those configurations.
  if (cudaMemsetAsync(
          gx_in, 0, static_cast<size_t>(rows) * inner * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(
          gb_in, 0,
          static_cast<size_t>(rows) * group_state * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(
          gc_in, 0,
          static_cast<size_t>(rows) * group_state * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(
          grad_weight, 0,
          static_cast<size_t>(weight_values) * sizeof(float), nsos::gpu::current_stream()) != cudaSuccess) {
    return false;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba2_faithful_conv_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      gx_out, gb_out, gc_out, x_in, b_in, c_in, weight, gx_in, gb_in,
      gc_in, grad_weight, seq, inner, group_state, K, convdim, total);
  return true;
}

bool launch_mamba2_faithful_bias_backward(
    const float *gx_pre, const float *gb_pre, const float *gc_pre,
    float *grad_bias, int rows, int inner, int group_state) {
  int convdim = 0;
  int total = 0;
  if (gx_pre == nullptr || gb_pre == nullptr || gc_pre == nullptr ||
      grad_bias == nullptr ||
      !checked_faithful_convdim(inner, group_state, &convdim) ||
      !checked_positive_product_to_int(rows, convdim, &total)) {
    return false;
  }
  mamba2_faithful_bias_backward_kernel<<<convdim, 256, 0, nsos::gpu::current_stream()>>>(
      gx_pre, gb_pre, gc_pre, grad_bias, rows, inner, group_state,
      convdim);
  return true;
}

__global__ void mamba2_pack_projection_grads_kernel(
    const float *__restrict__ gx, const float *__restrict__ gz,
    const float *__restrict__ gb, const float *__restrict__ gc,
    const float *__restrict__ gdt, float *__restrict__ packed, int inner,
    int group_state, int heads, int width, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int row = index / width;
  const int column = index - row * width;
  float value = 0.0f;
  if (column < inner) {
    value = gz[static_cast<size_t>(row) * inner + column];
  } else if (column < 2 * inner) {
    value = gx[static_cast<size_t>(row) * inner + column - inner];
  } else if (column < 2 * inner + group_state) {
    value =
        gb[static_cast<size_t>(row) * group_state + column - 2 * inner];
  } else if (column < 2 * inner + 2 * group_state) {
    value = gc[static_cast<size_t>(row) * group_state +
               column - 2 * inner - group_state];
  } else {
    value = gdt[static_cast<size_t>(row) * heads +
                column - 2 * inner - 2 * group_state];
  }
  packed[index] = value;
}

bool launch_mamba2_pack_projection_grads(
    const float *gx, const float *gz, const float *gb, const float *gc,
    const float *gdt, float *packed, int rows, int inner, int group_state,
    int heads) {
  const int64_t width64 =
      2LL * inner + 2LL * group_state + heads;
  if (gx == nullptr || gz == nullptr || gb == nullptr || gc == nullptr ||
      gdt == nullptr || packed == nullptr || rows <= 0 || inner <= 0 ||
      group_state <= 0 || heads <= 0 || width64 <= 0 || width64 > INT_MAX) {
    return false;
  }
  const int width = static_cast<int>(width64);
  int total = 0;
  if (!checked_positive_product_to_int(rows, width, &total)) {
    return false;
  }
  constexpr int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba2_pack_projection_grads_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      gx, gz, gb, gc, gdt, packed, inner, group_state, heads, width, total);
  return true;
}

__global__ void mamba2_unpack_projection_kernel(
    const float *__restrict__ packed, float *__restrict__ z,
    float *__restrict__ x, float *__restrict__ b,
    float *__restrict__ c, float *__restrict__ dt,
    int inner, int group_state, int heads, int width, int total,
    bool sensitive_only) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int row = index / width;
  int column = index - row * width;
  const float value = packed[index];
  if (!sensitive_only) {
    if (column < inner) {
      z[static_cast<size_t>(row) * inner + column] = value;
      return;
    }
    column -= inner;
    if (column < inner) {
      x[static_cast<size_t>(row) * inner + column] = value;
      return;
    }
    column -= inner;
  }
  if (column < group_state) {
    b[static_cast<size_t>(row) * group_state + column] = value;
  } else if (column < 2 * group_state) {
    c[static_cast<size_t>(row) * group_state +
      column - group_state] = value;
  } else {
    dt[static_cast<size_t>(row) * heads +
       column - 2 * group_state] = value;
  }
}

bool launch_mamba2_unpack_projection(
    const float *packed, float *z, float *x, float *b, float *c, float *dt,
    int rows, int inner, int group_state, int heads, bool sensitive_only) {
  const int64_t width64 =
      (sensitive_only ? 0LL : 2LL * inner) +
      2LL * group_state + heads;
  if (packed == nullptr || b == nullptr || c == nullptr || dt == nullptr ||
      (!sensitive_only && (z == nullptr || x == nullptr)) || rows <= 0 ||
      inner <= 0 || group_state <= 0 || heads <= 0 || width64 <= 0 ||
      width64 > INT_MAX) {
    return false;
  }
  const int width = static_cast<int>(width64);
  int total = 0;
  if (!checked_positive_product_to_int(rows, width, &total)) {
    return false;
  }
  constexpr int threads = 256;
  mamba2_unpack_projection_kernel
      <<<nsos::gpu::ceil_div_positive(total, threads), threads, 0, nsos::gpu::current_stream()>>>(
          packed, z, x, b, c, dt, inner, group_state, heads, width, total,
          sensitive_only);
  return true;
}

__global__ void mamba2_pack_sensitive_projection_grads_kernel(
    const float *__restrict__ gb, const float *__restrict__ gc,
    const float *__restrict__ gdt, float *__restrict__ packed,
    int group_state, int heads, int width, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int row = index / width;
  const int column = index - row * width;
  if (column < group_state) {
    packed[index] =
        gb[static_cast<size_t>(row) * group_state + column];
  } else if (column < 2 * group_state) {
    packed[index] =
        gc[static_cast<size_t>(row) * group_state +
           column - group_state];
  } else {
    packed[index] =
        gdt[static_cast<size_t>(row) * heads +
            column - 2 * group_state];
  }
}

bool launch_mamba2_pack_sensitive_projection_grads(
    const float *gb, const float *gc, const float *gdt, float *packed,
    int rows, int group_state, int heads) {
  const int64_t width64 = 2LL * group_state + heads;
  if (gb == nullptr || gc == nullptr || gdt == nullptr ||
      packed == nullptr || rows <= 0 || group_state <= 0 ||
      heads <= 0 || width64 <= 0 || width64 > INT_MAX) {
    return false;
  }
  const int width = static_cast<int>(width64);
  int total = 0;
  if (!checked_positive_product_to_int(rows, width, &total)) {
    return false;
  }
  constexpr int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba2_pack_sensitive_projection_grads_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      gb, gc, gdt, packed, group_state, heads, width, total);
  return true;
}

// â”€â”€ Full Mamba-2 SSD with N-dimensional state expansion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// One thread per (batch, head, p-channel); each carries an N-vector state in
// registers and walks the sequence.  Layout matches the host path in
// src/mamba2.cpp::forward_proper_nstate:
//   xc,y : [B,Seq,dim]  (dim=H*P, channel = h*P+p)
//   dt   : [B,Seq,H]    A : [H]    B_in,C_in : [B,Seq,H,N]
//   state_history : [B,Seq,dim,N]  (h_t after update; nullptr to skip)
// N must be <= MAX_N (caller checks via mamba_nstate_max_n()).
int mamba_nstate_max_n() { return MAX_N; }

bool mamba_nstate_deterministic_backward_supported(int P, int N) {
  return P > 0 && N > 0 && N <= MAX_N &&
         static_cast<int64_t>(P) * N <=
             MAMBA_NSTATE_DETERMINISTIC_MAX_CARRY_FLOATS;
}

__global__ void mamba_nstate_forward_kernel(
    const float *__restrict__ xc, const float *__restrict__ dt,
    const float *__restrict__ A, const float *__restrict__ B_in,
    const float *__restrict__ C_in, float *__restrict__ y,
    float *__restrict__ state_history, int Batch, int Seq, int H, int P, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Batch * H * P;
  if (tid >= total) return;
  const int p = tid % P;
  const int h = (tid / P) % H;
  const int b = tid / (P * H);
  const int dim = H * P;
  const int chan = h * P + p;
  const size_t HPN = (size_t)dim * N;
  float state[MAX_N];
  for (int n = 0; n < N; ++n) state[n] = 0.0f;
  for (int t = 0; t < Seq; ++t) {
    const int row = b * Seq + t;
    const MambaDecayTermsDevice terms =
        mamba_decay_terms_dev(dt[row * H + h], A[h]);
    const float dt_scale = terms.delta;
    const float decay = terms.decay;
    const float xcv = xc[(size_t)row * dim + chan];
    float y_acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      const size_t bcidx = (size_t)row * (H * N) + h * N + n;
      const float hv =
          decay * state[n] + dt_scale * B_in[bcidx] * xcv;
      state[n] = hv;
      if (state_history != nullptr) {
        state_history[(size_t)row * HPN + (size_t)chan * N + n] = hv;
      }
      y_acc += hv * C_in[bcidx];
    }
    y[(size_t)row * dim + chan] = y_acc;
  }
}

__global__ void mamba_nstate_backward_kernel(
    const float *__restrict__ gy, const float *__restrict__ xc,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ state_history, float *__restrict__ gXc,
    float *__restrict__ gDt, float *__restrict__ gA, float *__restrict__ gB,
    float *__restrict__ gC, int Batch, int Seq, int H, int P, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Batch * H * P;
  if (tid >= total) return;
  const int p = tid % P;
  const int h = (tid / P) % H;
  const int b = tid / (P * H);
  const int dim = H * P;
  const int chan = h * P + p;
  const size_t HPN = (size_t)dim * N;
  float carry[MAX_N];
  for (int n = 0; n < N; ++n) carry[n] = 0.0f;
  // gC/gB/gDt/gA are shared across the P threads of a head -> atomicAdd.
  // gXc is unique per (b,h,p) channel -> direct write.
  for (int t = Seq - 1; t >= 0; --t) {
    const int row = b * Seq + t;
    const float dt_val = dt[row * H + h];
    const MambaDecayTermsDevice terms =
        mamba_decay_terms_dev(dt_val, A[h]);
    const float sp = terms.delta;
    const float decay = terms.decay;
    const float xcv = xc[(size_t)row * dim + chan];
    const float gyv = gy[(size_t)row * dim + chan];
    float ddecay = 0.0f;
    float dinput_scale = 0.0f;
    float gxc_acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      const size_t bcidx = (size_t)row * (H * N) + h * N + n;
      const float h_t = state_history[(size_t)row * HPN + (size_t)chan * N + n];
      const float h_prev =
          (t == 0) ? 0.0f
                   : state_history[(size_t)(row - 1) * HPN + (size_t)chan * N + n];
      const float cval = C_in[bcidx];
      const float bval = B_in[bcidx];
      atomicAdd(&gC[bcidx], gyv * h_t);
      const float grad_h = gyv * cval + carry[n];
      atomicAdd(&gB[bcidx], grad_h * sp * xcv);
      gxc_acc += grad_h * sp * bval;
      dinput_scale += grad_h * bval * xcv;
      ddecay += grad_h * h_prev;
      carry[n] = grad_h * decay;
    }
    gXc[(size_t)row * dim + chan] = gxc_acc;
    atomicAdd(&gDt[row * H + h],
              dinput_scale * terms.delta_grad -
                  ddecay * terms.decay_dt_factor);
    // N1: per-head dA_log += dDecayÂ·decayÂ·(-sp)Â·A_eff (unconditional).
    atomicAdd(&gA[h], -ddecay * terms.decay_alog_factor);
  }
}

// One block owns one head and thread 0 performs the same b/t/p/n traversal as
// the host reference. The per-(p,n) recurrence carry lives in shared memory.
// Every output address has exactly one writer, so this path has no atomics and
// is bit-reproducible across repeated launches on the same backend.
__global__ void mamba_nstate_backward_deterministic_kernel(
    const float *__restrict__ gy, const float *__restrict__ xc,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ state_history, float *__restrict__ gXc,
    float *__restrict__ gDt, float *__restrict__ gA,
    float *__restrict__ gB, float *__restrict__ gC, int Batch, int Seq,
    int H, int P, int N) {
  const int h = blockIdx.x;
  if (h >= H || threadIdx.x != 0) return;

  extern __shared__ float carry[];
  const int dim = H * P;
  const size_t HPN = static_cast<size_t>(dim) * N;
  float grad_a = 0.0f;

  for (int b = 0; b < Batch; ++b) {
    for (int pn = 0; pn < P * N; ++pn) {
      carry[pn] = 0.0f;
    }
    for (int t = 0; t < Seq; ++t) {
      const int row = b * Seq + t;
      gDt[row * H + h] = 0.0f;
      for (int n = 0; n < N; ++n) {
        const size_t bcidx =
            static_cast<size_t>(row) * (H * N) + h * N + n;
        gB[bcidx] = 0.0f;
        gC[bcidx] = 0.0f;
      }
    }

    for (int t = Seq - 1; t >= 0; --t) {
      const int row = b * Seq + t;
      const MambaDecayTermsDevice terms =
          mamba_decay_terms_dev(dt[row * H + h], A[h]);
      float ddecay = 0.0f;
      float dinput_scale = 0.0f;
      for (int p = 0; p < P; ++p) {
        const int chan = h * P + p;
        const float xcv = xc[static_cast<size_t>(row) * dim + chan];
        const float gyv = gy[static_cast<size_t>(row) * dim + chan];
        float gxc = 0.0f;
        for (int n = 0; n < N; ++n) {
          const size_t bcidx =
              static_cast<size_t>(row) * (H * N) + h * N + n;
          const size_t state_index =
              static_cast<size_t>(row) * HPN +
              static_cast<size_t>(chan) * N + n;
          const float h_t = state_history[state_index];
          const float h_prev =
              t == 0 ? 0.0f : state_history[state_index - HPN];
          const float cval = C_in[bcidx];
          const float bval = B_in[bcidx];
          const int carry_index = p * N + n;
          gC[bcidx] += gyv * h_t;
          const float grad_h = gyv * cval + carry[carry_index];
          gB[bcidx] += grad_h * terms.delta * xcv;
          gxc += grad_h * terms.delta * bval;
          dinput_scale += grad_h * bval * xcv;
          ddecay += grad_h * h_prev;
          carry[carry_index] = grad_h * terms.decay;
        }
        gXc[static_cast<size_t>(row) * dim + chan] = gxc;
      }
      gDt[row * H + h] =
          dinput_scale * terms.delta_grad -
          ddecay * terms.decay_dt_factor;
      grad_a -= ddecay * terms.decay_alog_factor;
    }
  }
  gA[h] = grad_a;
}

void launch_mamba_nstate_forward(const float *xc, const float *dt,
                                 const float *A, const float *B_in,
                                 const float *C_in, float *y,
                                 float *state_history, int Batch, int Seq, int H,
                                 int P, int N) {
  int total = 0;
  if (Seq <= 0 || N <= 0 || N > MAX_N ||
      !checked_positive_product_to_int(Batch, H, P, &total)) {
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba_nstate_forward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      xc, dt, A, B_in, C_in, y, state_history, Batch, Seq, H, P, N);
}

void launch_mamba_nstate_backward(const float *gy, const float *xc,
                                  const float *dt, const float *A,
                                  const float *B_in, const float *C_in,
                                  const float *state_history, float *gXc,
                                  float *gDt, float *gA, float *gB, float *gC,
                                  int Batch, int Seq, int H, int P, int N) {
  int total = 0;
  if (Seq <= 0 || N <= 0 || N > MAX_N ||
      !checked_positive_product_to_int(Batch, H, P, &total)) {
    return;
  }
  // gB/gC/gDt/gA accumulate via atomicAdd -> zero first (async, serializes
  // with the kernel on the default stream).  gXc is written fully by the kernel.
  mamba_gpu_runtime_check(
      cudaMemsetAsync(
          gB, 0, static_cast<size_t>(Batch) * Seq * H * N * sizeof(float), nsos::gpu::current_stream()),
      "mamba nstate grad_B clear");
  mamba_gpu_runtime_check(
      cudaMemsetAsync(
          gC, 0, static_cast<size_t>(Batch) * Seq * H * N * sizeof(float), nsos::gpu::current_stream()),
      "mamba nstate grad_C clear");
  mamba_gpu_runtime_check(
      cudaMemsetAsync(
          gDt, 0, static_cast<size_t>(Batch) * Seq * H * sizeof(float), nsos::gpu::current_stream()),
      "mamba nstate grad_dt clear");
  mamba_gpu_runtime_check(
      cudaMemsetAsync(gA, 0, static_cast<size_t>(H) * sizeof(float), nsos::gpu::current_stream()),
      "mamba nstate grad_A clear");
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba_nstate_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      gy, xc, dt, A, B_in, C_in, state_history, gXc, gDt, gA, gB, gC, Batch, Seq,
      H, P, N);
}

bool launch_mamba_nstate_backward_deterministic(
    const float *gy, const float *xc, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *gXc, float *gDt, float *gA, float *gB, float *gC, int Batch,
    int Seq, int H, int P, int N) {
  int rows = 0;
  int output_values = 0;
  int state_values = 0;
  if (gy == nullptr || xc == nullptr || dt == nullptr || A == nullptr ||
      B_in == nullptr || C_in == nullptr || state_history == nullptr ||
      gXc == nullptr || gDt == nullptr || gA == nullptr || gB == nullptr ||
      gC == nullptr || H <= 0 ||
      !mamba_nstate_deterministic_backward_supported(P, N) ||
      !checked_positive_product_to_int(Batch, Seq, &rows) ||
      !checked_positive_product_to_int(rows, H, P, &output_values) ||
      !checked_positive_product_to_int(output_values, N, &state_values)) {
    return false;
  }
  const size_t shared_bytes =
      static_cast<size_t>(P) * N * sizeof(float);
  mamba_nstate_backward_deterministic_kernel<<<H, 1, shared_bytes, nsos::gpu::current_stream()>>>(
      gy, xc, dt, A, B_in, C_in, state_history, gXc, gDt, gA, gB, gC,
      Batch, Seq, H, P, N);
  return true;
}

// â”€â”€ Faithful Mamba-2: grouped B/C + per-head D skip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
__global__ void mamba2_faithful_precompute_decay_kernel(
    const float *__restrict__ dt, const float *__restrict__ A,
    float *__restrict__ packed_terms, int values, int H) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const int h = index % H;
  const MambaDecayTermsDevice terms =
      mamba_decay_terms_dev(dt[index], A[h]);
  float *destination =
      packed_terms +
      static_cast<size_t>(index) * MAMBA_DECAY_TERMS_WIDTH;
  destination[0] = terms.delta;
  destination[1] = terms.delta_grad;
  destination[2] = terms.decay;
  destination[3] = terms.decay_dt_factor;
  destination[4] = terms.decay_alog_factor;
}

bool faithful_boundary_history_enabled() {
  return nsos::training_policy::boundary_history();
}

template <int HISTORY_CHUNK>
__device__ __forceinline__ void store_faithful_history(
    float* history, float before, float after, int b, int t, int Seq,
    int chan, int n, int inner, int N, bool state_major) {
  if (!history) return;
  if constexpr (HISTORY_CHUNK > 0) {
    if (t % HISTORY_CHUNK != 0) return;
    const int chunks = (Seq + HISTORY_CHUNK - 1) / HISTORY_CHUNK;
    history[mamba_faithful_state_index(b * chunks + t / HISTORY_CHUNK,
        chan, n, inner, N, state_major)] = before;
  } else {
    history[mamba_faithful_state_index(b * Seq + t, chan, n,
        inner, N, state_major)] = after;
  }
}

template <int HISTORY_CHUNK>
__global__ void mamba2_faithful_forward_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in,
    const float *__restrict__ C_in, const float *__restrict__ D,
    float *__restrict__ y, float *__restrict__ state_history, int Batch,
    int Seq, int H, int P, int N, int G,
    int waves_per_head, bool head_channel_geometry,
    bool head_wave_geometry, bool state_major_history, int history_chunk_size) {
  int p = 0;
  int h = 0;
  int b = 0;
  if (head_wave_geometry) {
    const int wave_block = blockIdx.x % waves_per_head;
    const int head_block = blockIdx.x / waves_per_head;
    p = wave_block * blockDim.x + threadIdx.x;
    h = head_block % H;
    b = head_block / H;
  } else if (head_channel_geometry) {
    p = threadIdx.x;
    if (p >= P) return;
    h = blockIdx.x % H;
    b = blockIdx.x / H;
  } else {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Batch * H * P;
    if (tid >= total) return;
    p = tid % P;
    h = (tid / P) % H;
    b = tid / (H * P);
  }
  const int group = (h * G) / H;
  const int inner = H * P;
  const int chan = h * P + p;
  const size_t state_row = static_cast<size_t>(inner) * N;
  float state[MAX_N];
  for (int n = 0; n < N; ++n) state[n] = 0.0f;
  for (int t = 0; t < Seq; ++t) {
    const int row = b * Seq + t;
    const size_t decay_index =
        static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float delta = terms.delta;
    const float decay = terms.decay;
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    float out = D[h] * xv;
    for (int n = 0; n < N; ++n) {
      const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
      const float hv = decay * state[n] + delta * B_in[bc] * xv;
      if constexpr (HISTORY_CHUNK > 0)
        store_faithful_history<HISTORY_CHUNK>(state_history, state[n], hv, b, t, Seq,
            chan, n, inner, N, state_major_history);
      state[n] = hv;
      out += hv * C_in[bc];
      if constexpr (HISTORY_CHUNK == 0)
        store_faithful_history<0>(state_history, 0.0f, hv, b, t, Seq,
            chan, n, inner, N, state_major_history);
    }
    y[static_cast<size_t>(row) * inner + chan] = out;
  }
}

// Compile-time state-width specialization of the faithful forward scan.
//
// The generic kernel above declares `float state[MAX_N]` and walks it with a
// runtime trip count.  The AMDGPU backend cannot keep a dynamically indexed
// array in registers, so it lowers `state` to scratch:
// -Rpass-analysis=kernel-resource-usage reports 272 scratch bytes/lane for
// gfx1102, paid twice per (timestep, state) pair across the whole sequence.
// Binding N at compile time lets the array live in VGPRs and the inner loop
// unroll fully.
//
// The emitted operation sequence, its order and every rounding step are
// identical to the generic kernel -- only the storage class of `state`
// changes -- so results stay bit-for-bit equal and the deterministic
// execution contract is preserved.
template <int N_FIXED, int HISTORY_CHUNK = 0>
__global__ void mamba2_faithful_forward_fixed_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in,
    const float *__restrict__ C_in, const float *__restrict__ D,
    float *__restrict__ y, float *__restrict__ state_history, int Batch,
    int Seq, int H, int P, int N, int G,
    int waves_per_head, bool head_channel_geometry,
    bool head_wave_geometry, bool state_major_history, int history_chunk_size) {
  int p = 0;
  int h = 0;
  int b = 0;
  if (head_wave_geometry) {
    const int wave_block = blockIdx.x % waves_per_head;
    const int head_block = blockIdx.x / waves_per_head;
    p = wave_block * blockDim.x + threadIdx.x;
    h = head_block % H;
    b = head_block / H;
  } else if (head_channel_geometry) {
    p = threadIdx.x;
    if (p >= P) return;
    h = blockIdx.x % H;
    b = blockIdx.x / H;
  } else {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Batch * H * P;
    if (tid >= total) return;
    p = tid % P;
    h = (tid / P) % H;
    b = tid / (H * P);
  }
  const int group = (h * G) / H;
  const int inner = H * P;
  const int chan = h * P + p;
  float state[N_FIXED];
#pragma unroll
  for (int n = 0; n < N_FIXED; ++n) state[n] = 0.0f;
  for (int t = 0; t < Seq; ++t) {
    const int row = b * Seq + t;
    const size_t decay_index =
        static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float delta = terms.delta;
    const float decay = terms.decay;
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    float out = D[h] * xv;
#pragma unroll
    for (int n = 0; n < N_FIXED; ++n) {
      const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
      const float hv = decay * state[n] + delta * B_in[bc] * xv;
      if constexpr (HISTORY_CHUNK > 0)
        store_faithful_history<HISTORY_CHUNK>(state_history, state[n], hv, b, t, Seq,
            chan, n, inner, N, state_major_history);
      state[n] = hv;
      out += hv * C_in[bc];
      if constexpr (HISTORY_CHUNK == 0)
        store_faithful_history<0>(state_history, 0.0f, hv, b, t, Seq,
            chan, n, inner, N, state_major_history);
    }
    y[static_cast<size_t>(row) * inner + chan] = out;
  }
}

// Opt-in for the compile-time specialization above.
//
// Measured on gfx1102 (RX 7600, pilot preset: 16 layers, d_model 768,
// seq 512, N=64) the specialization raises raw throughput from 1579.8 to
// 1782.8 tokens/s p50 (+12.85%) by cutting the forward scan's scratch
// traffic from 272 to 124 bytes/lane.
//
// It is NOT bit-identical to the generic kernel.  Unrolling the inner loop
// lets the compiler contract multiply-add pairs into different fused
// operations, which showed up as a 1-2 ULP divergence in the per-step loss
// trace of a 1000-step pilot run.  The learning signal is unaffected
// (held-out loss 4.0912 vs 4.0849), but the project's bitwise reproduction
// contract is, so the generic kernel stays the default and this path must be
// requested explicitly.
bool faithful_fixed_state_forward_enabled() {
  static const bool enabled = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_FAITHFUL_FIXED_STATE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

// ---------------------------------------------------------------------------
// Chunked (time-parallel) faithful forward scan.
//
// The sequential kernels above expose only Batch*H*(P/warp) wavefronts -- 48
// for the pilot shape on gfx1102, against 64 SIMDs.  Every memory stall is
// therefore fully exposed: there is no second wave resident on the SIMD to
// switch to.  The backward pass already solved this with a
// summary -> prefix -> apply decomposition; these kernels give the forward the
// same treatment.
//
// The recurrence h_t = decay_t*h_{t-1} + delta_t*B_t*x_t is affine, so a chunk
// starting from zero can be replayed later from its true entry state:
//
//   summary : per chunk, scan locally from h=0, keep the chunk end state and
//             the product of its decays.  No history is written.
//   prefix  : carry_c = D_c * carry_{c-1} + end_local_c, sequential over the
//             (few) chunks, parallel over every channel/state pair.
//   apply   : replay each chunk from its entry carry.  This is the original
//             recurrence verbatim, so it writes the final history and output.
//
// Sequential depth drops from Seq to 2*chunk_size while wavefront count rises
// by the chunk factor.  Only the entry carry differs from the fully sequential
// value -- it is accumulated as D_c*carry + end_local instead of step by step
// -- so results are mathematically equal but not bit-identical.  This path is
// therefore opt-in and recorded in the runtime execution identity.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int mamba_faithful_channel_state_index(
    int b, int h, int p, int n, int H, int P, int N) {
  return (((b * H + h) * P + p) * N) + n;
}

template <int N_FIXED>
__global__ void mamba2_faithful_forward_chunk_summary_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in,
    float *__restrict__ chunk_end_local,
    float *__restrict__ chunk_total_decay, int Batch, int Seq, int H,
    int P, int N, int G, int waves_per_head, int chunks, int chunk_size,
    int channel_states) {
  const int chunk = blockIdx.x % chunks;
  const int rest = blockIdx.x / chunks;
  const int wave_block = rest % waves_per_head;
  const int head_block = rest / waves_per_head;
  const int p = wave_block * blockDim.x + threadIdx.x;
  const int h = head_block % H;
  const int b = head_block / H;
  const int group = (h * G) / H;
  const int inner = H * P;
  const int chan = h * P + p;
  const int t_begin = chunk * chunk_size;
  int t_end = t_begin + chunk_size;
  if (t_end > Seq) t_end = Seq;

  float state[N_FIXED];
#pragma unroll
  for (int n = 0; n < N_FIXED; ++n) state[n] = 0.0f;
  float decay_product = 1.0f;

  for (int t = t_begin; t < t_end; ++t) {
    const int row = b * Seq + t;
    const size_t decay_index = static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float delta = terms.delta;
    const float decay = terms.decay;
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    decay_product *= decay;
#pragma unroll
    for (int n = 0; n < N_FIXED; ++n) {
      const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
      state[n] = decay * state[n] + delta * B_in[bc] * xv;
    }
  }

  const int base =
      mamba_faithful_channel_state_index(b, h, p, 0, H, P, N);
#pragma unroll
  for (int n = 0; n < N_FIXED; ++n) {
    chunk_end_local[static_cast<size_t>(chunk) * channel_states + base +
                    n] = state[n];
  }
  if (p == 0) {
    chunk_total_decay[static_cast<size_t>(b * H + h) * chunks + chunk] =
        decay_product;
  }
}

__global__ void mamba2_faithful_forward_chunk_prefix_kernel(
    const float *__restrict__ chunk_end_local,
    const float *__restrict__ chunk_total_decay,
    float *__restrict__ carry_entering, int H, int P, int N, int chunks,
    int channel_states) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= channel_states) return;
  const int bh = index / (P * N);
  const int h = bh % H;
  const int b = bh / H;

  float carry = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const size_t slot =
        static_cast<size_t>(chunk) * channel_states + index;
    carry_entering[slot] = carry;
    const float decay_product =
        chunk_total_decay[static_cast<size_t>(b * H + h) * chunks + chunk];
    carry = decay_product * carry + chunk_end_local[slot];
  }
}

template <int N_FIXED, int HISTORY_CHUNK = 0>
__global__ void mamba2_faithful_forward_chunk_apply_kernel(
    const float *__restrict__ x, const float *__restrict__ dt,
    const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ D,
    const float *__restrict__ carry_entering, float *__restrict__ y,
    float *__restrict__ state_history, int Batch, int Seq, int H, int P,
    int N, int G, int waves_per_head, int chunks, int chunk_size,
    int channel_states, bool state_major_history, int history_chunk_size) {
  const int chunk = blockIdx.x % chunks;
  const int rest = blockIdx.x / chunks;
  const int wave_block = rest % waves_per_head;
  const int head_block = rest / waves_per_head;
  const int p = wave_block * blockDim.x + threadIdx.x;
  const int h = head_block % H;
  const int b = head_block / H;
  const int group = (h * G) / H;
  const int inner = H * P;
  const int chan = h * P + p;
  const int t_begin = chunk * chunk_size;
  int t_end = t_begin + chunk_size;
  if (t_end > Seq) t_end = Seq;

  const int base =
      mamba_faithful_channel_state_index(b, h, p, 0, H, P, N);
  float state[N_FIXED];
#pragma unroll
  for (int n = 0; n < N_FIXED; ++n) {
    state[n] = carry_entering[static_cast<size_t>(chunk) * channel_states +
                              base + n];
  }

  // Verbatim replay of the sequential recurrence from the chunk entry state.
  for (int t = t_begin; t < t_end; ++t) {
    const int row = b * Seq + t;
    const size_t decay_index = static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float delta = terms.delta;
    const float decay = terms.decay;
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    float out = D[h] * xv;
#pragma unroll
    for (int n = 0; n < N_FIXED; ++n) {
      const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
      const float hv = decay * state[n] + delta * B_in[bc] * xv;
      if constexpr (HISTORY_CHUNK > 0)
        store_faithful_history<HISTORY_CHUNK>(state_history, state[n], hv, b, t, Seq,
            chan, n, inner, N, state_major_history);
      state[n] = hv;
      out += hv * C_in[bc];
      if constexpr (HISTORY_CHUNK == 0)
        store_faithful_history<0>(state_history, 0.0f, hv, b, t, Seq,
            chan, n, inner, N, state_major_history);
    }
    y[static_cast<size_t>(row) * inner + chan] = out;
  }
}

// Opt-in for the chunked forward scan and its chunk width.
bool faithful_chunked_forward_enabled() {
  static const bool enabled = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

int faithful_forward_chunk_size() {
  static const int size = [] {
    const char *value =
        std::getenv("NSOS_MAMBA_FORWARD_CHUNK_SIZE");
    if (value == nullptr) return 128;
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed < 8 || parsed > 4096) return 128;
    return static_cast<int>(parsed);
  }();
  return size;
}

__global__ void mamba2_faithful_backward_kernel(
    const float *__restrict__ gy, const float *__restrict__ x,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ D, const float *__restrict__ state_history,
    float *__restrict__ gX, float *__restrict__ gDt, float *__restrict__ gA,
    float *__restrict__ gB, float *__restrict__ gC, float *__restrict__ gD,
    int Batch, int Seq, int H, int P, int N, int G,
    bool allow_warp_aggregate, bool head_channel_geometry,
    bool state_major_history) {
  int p = 0;
  int h = 0;
  int b = 0;
  if (head_channel_geometry) {
    p = threadIdx.x;
    if (p >= P) return;
    h = blockIdx.x % H;
    b = blockIdx.x / H;
  } else {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Batch * H * P;
    if (tid >= total) return;
    p = tid % P;
    h = (tid / P) % H;
    b = tid / (H * P);
  }
  const int group = (h * G) / H;
  const int inner = H * P;
  const int chan = h * P + p;
  const size_t state_row = static_cast<size_t>(inner) * N;
  // With P aligned to warpSize, every warp is wholly contained in one
  // (batch,head), so all lanes target the same reduction addresses below.
  // total is then also warp-aligned and every shuffle has a full active warp.
  const bool warp_aggregate =
      allow_warp_aggregate && P % warpSize == 0;
  float carry[MAX_N];
  for (int n = 0; n < N; ++n) carry[n] = 0.0f;
  // A and D are shared across the complete sequence. Accumulate their
  // per-channel contributions in registers and publish once after the reverse
  // scan. Reducing them inside every timestep needlessly multiplied global
  // atomic traffic by Seq.
  float local_gA = 0.0f;
  float local_gD = 0.0f;
  for (int t = Seq - 1; t >= 0; --t) {
    const int row = b * Seq + t;
    const float dt_raw = dt[row * H + h];
    const size_t decay_index =
        static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt_raw, A[h]);
    const float delta = terms.delta;
    const float decay = terms.decay;
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    const float go = gy[static_cast<size_t>(row) * inner + chan];
    float gx = go * D[h];
    float ddecay = 0.0f;
    float dinput_scale = 0.0f;
    local_gD += go * xv;
    for (int n = 0; n < N; ++n) {
      const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
      const size_t state_idx = mamba_faithful_state_index(
          row, chan, n, inner, N, state_major_history);
      const float ht = state_history[state_idx];
      const float hprev =
          t == 0
              ? 0.0f
              : state_history[mamba_faithful_state_index(
                    row - 1, chan, n, inner, N,
                    state_major_history)];
      mamba_channel_reduction_add(
          &gC[bc], go * ht, p, warp_aggregate);
      const float gh = go * C_in[bc] + carry[n];
      mamba_channel_reduction_add(
          &gB[bc], gh * delta * xv, p, warp_aggregate);
      gx += gh * delta * B_in[bc];
      dinput_scale += gh * B_in[bc] * xv;
      ddecay += gh * hprev;
      carry[n] = gh * decay;
    }
    gX[static_cast<size_t>(row) * inner + chan] = gx;
    mamba_channel_reduction_add(
        &gDt[row * H + h],
        dinput_scale * terms.delta_grad -
            ddecay * terms.decay_dt_factor,
        p, warp_aggregate);
    local_gA -= ddecay * terms.decay_alog_factor;
  }
  mamba_channel_reduction_add(
      &gA[h], local_gA, p, warp_aggregate);
  mamba_channel_reduction_add(
      &gD[h], local_gD, p, warp_aggregate);
}

template <bool UseSharedCarry>
__global__ void mamba2_faithful_backward_deterministic_partials_kernel(
    const float *__restrict__ gy, const float *__restrict__ x,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ D, const float *__restrict__ state_history,
    float *__restrict__ gX, float *__restrict__ partial_B,
    float *__restrict__ partial_C, float *__restrict__ partial_dt,
    float *__restrict__ partial_A, float *__restrict__ partial_D,
    int Batch, int Seq, int H, int P, int N, int G,
    int waves_per_head, int partials_per_group,
    bool head_channel_geometry, bool head_wave_geometry,
    bool state_major_history) {
  int p = 0;
  int h = 0;
  int b = 0;
  if (head_wave_geometry) {
    const int wave_block = blockIdx.x % waves_per_head;
    const int head_block = blockIdx.x / waves_per_head;
    p = wave_block * blockDim.x + threadIdx.x;
    h = head_block % H;
    b = head_block / H;
  } else if (head_channel_geometry) {
    p = threadIdx.x;
    if (p >= P) return;
    h = blockIdx.x % H;
    b = blockIdx.x / H;
  } else {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Batch * H * P;
    if (tid >= total) return;
    p = tid % P;
    h = (tid / P) % H;
    b = tid / (H * P);
  }
  const int heads_per_group = H / G;
  const int group = h / heads_per_group;
  const int head_in_group = h - group * heads_per_group;
  const int wave = p / warpSize;
  const int lane = p % warpSize;
  const int partial_slot =
      head_in_group * waves_per_head + wave;
  const int inner = H * P;
  const int chan = h * P + p;
  const size_t state_row = static_cast<size_t>(inner) * N;
  // A runtime-indexed float[MAX_N] becomes a 272-byte private segment per
  // thread on gfx1102. In the one-wave-per-block geometry, shared memory is a
  // strictly private slice for each lane and preserves every load/store and
  // arithmetic dependency while avoiding global scratch traffic.
  extern __shared__ float shared_carry[];
  float local_carry[UseSharedCarry ? 1 : MAX_N];
  float *carry = UseSharedCarry
                     ? shared_carry +
                           static_cast<size_t>(threadIdx.x) * N
                     : local_carry;
  for (int n = 0; n < N; ++n) carry[n] = 0.0f;
  float local_gA = 0.0f;
  float local_gD = 0.0f;
  for (int t = Seq - 1; t >= 0; --t) {
    const int row = b * Seq + t;
    const float dt_raw = dt[row * H + h];
    const size_t decay_index =
        static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt_raw, A[h]);
    const float delta = terms.delta;
    const float decay = terms.decay;
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    const float go = gy[static_cast<size_t>(row) * inner + chan];
    float gx = go * D[h];
    float ddecay = 0.0f;
    float dinput_scale = 0.0f;
    local_gD += go * xv;
    for (int n = 0; n < N; ++n) {
      const size_t bc =
          static_cast<size_t>(row) * G * N + group * N + n;
      const size_t state_idx = mamba_faithful_state_index(
          row, chan, n, inner, N, state_major_history);
      const float ht = state_history[state_idx];
      const float hprev =
          t == 0
              ? 0.0f
              : state_history[mamba_faithful_state_index(
                    row - 1, chan, n, inner, N,
                    state_major_history)];
      const float c_partial = mamba_warp_sum(go * ht);
      const float gh = go * C_in[bc] + carry[n];
      const float b_partial =
          mamba_warp_sum(gh * delta * xv);
      if (lane == 0) {
        const size_t destination =
            bc * partials_per_group + partial_slot;
        partial_C[destination] = c_partial;
        partial_B[destination] = b_partial;
      }
      gx += gh * delta * B_in[bc];
      dinput_scale += gh * B_in[bc] * xv;
      ddecay += gh * hprev;
      carry[n] = gh * decay;
    }
    gX[static_cast<size_t>(row) * inner + chan] = gx;
    const float dt_partial = mamba_warp_sum(
        dinput_scale * terms.delta_grad -
        ddecay * terms.decay_dt_factor);
    if (lane == 0) {
      partial_dt[
          (static_cast<size_t>(row) * H + h) * waves_per_head + wave] =
          dt_partial;
    }
    local_gA -= ddecay * terms.decay_alog_factor;
  }
  const float a_partial = mamba_warp_sum(local_gA);
  const float d_partial = mamba_warp_sum(local_gD);
  if (lane == 0) {
    const size_t destination =
        (static_cast<size_t>(b) * H + h) * waves_per_head + wave;
    partial_A[destination] = a_partial;
    partial_D[destination] = d_partial;
  }
}

// Exact-order state-parallel deterministic backward, pass 1. Each block owns
// one (batch, head, wave, state), so the reverse recurrence has a scalar carry
// and exposes N times more independent work than the fused channel/state
// kernel. Shared B/C reductions retain the original native-wave tree and the
// exact same partial slot. gh_history is the only additional workspace.
__global__ void mamba2_faithful_backward_state_parallel_kernel(
    const float *__restrict__ gy, const float *__restrict__ x,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ state_history,
    float *__restrict__ gh_history, float *__restrict__ partial_B,
    float *__restrict__ partial_C, int Batch, int Seq, int H, int P,
    int N, int G, int waves_per_head, int partials_per_group,
    bool state_major_history) {
  const int n = blockIdx.x % N;
  const int head_wave_block = blockIdx.x / N;
  const int wave = head_wave_block % waves_per_head;
  const int batch_head = head_wave_block / waves_per_head;
  const int h = batch_head % H;
  const int b = batch_head / H;
  const int p = wave * blockDim.x + threadIdx.x;
  if (b >= Batch || p >= P) return;

  const int lane = threadIdx.x;
  const int heads_per_group = H / G;
  const int group = h / heads_per_group;
  const int head_in_group = h - group * heads_per_group;
  const int partial_slot =
      head_in_group * waves_per_head + wave;
  const int inner = H * P;
  const int chan = h * P + p;
  const size_t state_row = static_cast<size_t>(inner) * N;
  float carry = 0.0f;
  for (int t = Seq - 1; t >= 0; --t) {
    const int row = b * Seq + t;
    const size_t decay_index =
        static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    const float go = gy[static_cast<size_t>(row) * inner + chan];
    const size_t bc =
        static_cast<size_t>(row) * G * N + group * N + n;
    const size_t state_idx = mamba_faithful_state_index(
        row, chan, n, inner, N, state_major_history);
    const float ht = state_history[state_idx];
    const float c_partial = mamba_warp_sum(go * ht);
    const float gh = go * C_in[bc] + carry;
    const float b_partial =
        mamba_warp_sum(gh * terms.delta * xv);
    if (lane == 0) {
      const size_t destination =
          bc * partials_per_group + partial_slot;
      partial_C[destination] = c_partial;
      partial_B[destination] = b_partial;
    }
    gh_history[state_idx] = gh;
    carry = gh * terms.decay;
  }
}

// Pass 2 restores the original per-channel arithmetic order exactly: reverse
// timesteps, ascending states, then the same wave reductions and partial slots.
__global__ void mamba2_faithful_backward_state_parallel_finalize_kernel(
    const float *__restrict__ gy, const float *__restrict__ x,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in, const float *__restrict__ D,
    const float *__restrict__ state_history,
    const float *__restrict__ gh_history, float *__restrict__ gX,
    float *__restrict__ partial_dt, float *__restrict__ partial_A,
    float *__restrict__ partial_D, int Batch, int Seq, int H, int P,
    int N, int G, int waves_per_head, bool state_major_history) {
  const int wave = blockIdx.x % waves_per_head;
  const int batch_head = blockIdx.x / waves_per_head;
  const int h = batch_head % H;
  const int b = batch_head / H;
  const int p = wave * blockDim.x + threadIdx.x;
  if (b >= Batch || p >= P) return;

  const int lane = threadIdx.x;
  const int heads_per_group = H / G;
  const int group = h / heads_per_group;
  const int inner = H * P;
  const int chan = h * P + p;
  const size_t state_row = static_cast<size_t>(inner) * N;
  float local_gA = 0.0f;
  float local_gD = 0.0f;
  for (int t = Seq - 1; t >= 0; --t) {
    const int row = b * Seq + t;
    const size_t decay_index =
        static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    const float go = gy[static_cast<size_t>(row) * inner + chan];
    float gx = go * D[h];
    float dinput_scale = 0.0f;
    float ddecay = 0.0f;
    local_gD += go * xv;
    for (int n = 0; n < N; ++n) {
      const size_t bc =
          static_cast<size_t>(row) * G * N + group * N + n;
      const size_t state_idx = mamba_faithful_state_index(
          row, chan, n, inner, N, state_major_history);
      const float hprev =
          t == 0
              ? 0.0f
              : state_history[mamba_faithful_state_index(
                    row - 1, chan, n, inner, N,
                    state_major_history)];
      const float gh = gh_history[state_idx];
      gx += gh * terms.delta * B_in[bc];
      dinput_scale += gh * B_in[bc] * xv;
      ddecay += gh * hprev;
    }
    gX[static_cast<size_t>(row) * inner + chan] = gx;
    const float dt_partial = mamba_warp_sum(
        dinput_scale * terms.delta_grad -
        ddecay * terms.decay_dt_factor);
    if (lane == 0) {
      partial_dt[
          (static_cast<size_t>(row) * H + h) * waves_per_head + wave] =
          dt_partial;
    }
    local_gA -= ddecay * terms.decay_alog_factor;
  }
  const float a_partial = mamba_warp_sum(local_gA);
  const float d_partial = mamba_warp_sum(local_gD);
  if (lane == 0) {
    const size_t destination =
        (static_cast<size_t>(b) * H + h) * waves_per_head + wave;
    partial_A[destination] = a_partial;
    partial_D[destination] = d_partial;
  }
}

__device__ __forceinline__ size_t mamba2_chunk_carry_index(
    int b, int h, int wave, int chunk, int n, int lane, int H,
    int waves_per_head, int chunks, int N, int runtime_warp_size) {
  return (((((static_cast<size_t>(b) * H + h) * waves_per_head + wave) *
             chunks + chunk) * N + n) * runtime_warp_size + lane);
}

__device__ __forceinline__ size_t mamba2_chunk_lane_partial_index(
    int b, int h, int wave, int chunk, int lane, int H,
    int waves_per_head, int chunks, int runtime_warp_size) {
  return ((((static_cast<size_t>(b) * H + h) * waves_per_head + wave) *
            chunks + chunk) * runtime_warp_size + lane);
}

// Pass 1 builds an affine reverse-recurrence summary for every temporal
// chunk. A block owns (batch, head, native wave, chunk), while each lane owns
// one channel and carries all states in the same ascending order as the
// reference kernel. Only the compact chunk boundary is materialized.
template <bool StateMajorLds>
__device__ __forceinline__ size_t mamba2_chunk_lds_index(
    int lane, int n, int N, int runtime_warp_size) {
  return StateMajorLds
             ? static_cast<size_t>(n) * runtime_warp_size + lane
             : static_cast<size_t>(lane) * N + n;
}

template <bool StateMajorLds>
__global__ void mamba2_faithful_backward_chunk_summary_kernel(
    const float *__restrict__ gy, const float *__restrict__ dt,
    const float *__restrict__ A, const float *__restrict__ decay_terms,
    const float *__restrict__ C_in, float *__restrict__ chunk_carry,
    float *__restrict__ chunk_scale, int Batch, int Seq, int H, int P,
    int N, int G, int waves_per_head, int chunks, int chunk_size,
    int runtime_warp_size) {
  extern __shared__ float shared_state[];
  const int chunk = blockIdx.x % chunks;
  const int head_wave_block = blockIdx.x / chunks;
  const int wave = head_wave_block % waves_per_head;
  const int batch_head = head_wave_block / waves_per_head;
  const int h = batch_head % H;
  const int b = batch_head / H;
  const int lane = threadIdx.x;
  const int p = wave * runtime_warp_size + lane;
  if (b >= Batch || p >= P || lane >= runtime_warp_size) return;

  const int inner = H * P;
  const int chan = h * P + p;
  const int heads_per_group = H / G;
  const int group = h / heads_per_group;
  const int begin = chunk * chunk_size;
  const int end = min(begin + chunk_size, Seq);
  // Every address remains lane-private. State-major LDS places adjacent lanes
  // on adjacent banks for each n; the rollback layout preserves [lane,state].
  for (int n = 0; n < N; ++n) {
    shared_state[mamba2_chunk_lds_index<StateMajorLds>(
        lane, n, N, runtime_warp_size)] = 0.0f;
  }
  float scale = 1.0f;
  for (int t = end - 1; t >= begin; --t) {
    const int row = b * Seq + t;
    const size_t decay_index = static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float go = gy[static_cast<size_t>(row) * inner + chan];
    for (int n = 0; n < N; ++n) {
      const size_t bc =
          static_cast<size_t>(row) * G * N + group * N + n;
      const size_t lds_index =
          mamba2_chunk_lds_index<StateMajorLds>(
              lane, n, N, runtime_warp_size);
      shared_state[lds_index] =
          (go * C_in[bc] + shared_state[lds_index]) * terms.decay;
    }
    scale *= terms.decay;
  }
  for (int n = 0; n < N; ++n) {
    chunk_carry[mamba2_chunk_carry_index(
        b, h, wave, chunk, n, lane, H, waves_per_head, chunks, N,
        runtime_warp_size)] =
        shared_state[mamba2_chunk_lds_index<StateMajorLds>(
            lane, n, N, runtime_warp_size)];
  }
  if (wave == 0 && lane == 0) {
    chunk_scale[(static_cast<size_t>(b) * H + h) * chunks + chunk] =
        scale;
  }
}

// Pass 2 connects chunk summaries from the end of the sequence to the start.
// Each thread owns one channel/state and overwrites the summary with the
// incoming carry required by pass 3, so no second large workspace is needed.
__global__ void mamba2_faithful_backward_chunk_prefix_kernel(
    float *__restrict__ chunk_carry,
    const float *__restrict__ chunk_scale, int Batch, int H, int P,
    int N, int waves_per_head, int chunks, int runtime_warp_size,
    int total_channel_states) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_channel_states) return;
  const int n = index % N;
  const int channel = index / N;
  const int p = channel % P;
  const int h = (channel / P) % H;
  const int b = channel / (H * P);
  const int wave = p / runtime_warp_size;
  const int lane = p - wave * runtime_warp_size;
  float future = 0.0f;
  for (int chunk = chunks - 1; chunk >= 0; --chunk) {
    const size_t carry_index = mamba2_chunk_carry_index(
        b, h, wave, chunk, n, lane, H, waves_per_head, chunks, N,
        runtime_warp_size);
    const float bias = chunk_carry[carry_index];
    chunk_carry[carry_index] = future;
    const float scale =
        chunk_scale[(static_cast<size_t>(b) * H + h) * chunks + chunk];
    future = bias + scale * future;
  }
}

// Pass 3 executes the reference arithmetic within each chunk from its exact
// incoming boundary carry and emits the same deterministic wave partial
// layouts used by the production reduction kernels.
template <bool StateMajorLds>
__global__ void mamba2_faithful_backward_chunk_apply_kernel(
    const float *__restrict__ gy, const float *__restrict__ x,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ decay_terms,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ D,
    const float *__restrict__ state_history,
    const float *__restrict__ chunk_carry, float *__restrict__ gX,
    float *__restrict__ partial_B, float *__restrict__ partial_C,
    float *__restrict__ partial_dt, float *__restrict__ chunk_lane_A,
    float *__restrict__ chunk_lane_D, int Batch, int Seq, int H, int P,
    int N, int G, int waves_per_head, int partials_per_group,
    int chunks, int chunk_size, int runtime_warp_size,
    bool state_major_history) {
  extern __shared__ float shared_state[];
  const int chunk = blockIdx.x % chunks;
  const int head_wave_block = blockIdx.x / chunks;
  const int wave = head_wave_block % waves_per_head;
  const int batch_head = head_wave_block / waves_per_head;
  const int h = batch_head % H;
  const int b = batch_head / H;
  const int lane = threadIdx.x;
  const int p = wave * runtime_warp_size + lane;
  if (b >= Batch || p >= P || lane >= runtime_warp_size) return;

  const int inner = H * P;
  const int chan = h * P + p;
  const int heads_per_group = H / G;
  const int group = h / heads_per_group;
  const int head_in_group = h - group * heads_per_group;
  const int partial_slot = head_in_group * waves_per_head + wave;
  const int begin = chunk * chunk_size;
  const int end = min(begin + chunk_size, Seq);
  // Keep the recurrent state lane-private while allowing a bank-coalesced
  // [state,lane] physical layout. The template removes the indexing branch.
  for (int n = 0; n < N; ++n) {
    shared_state[mamba2_chunk_lds_index<StateMajorLds>(
        lane, n, N, runtime_warp_size)] =
        chunk_carry[mamba2_chunk_carry_index(
        b, h, wave, chunk, n, lane, H, waves_per_head, chunks, N,
        runtime_warp_size)];
  }
  float local_gA = 0.0f;
  float local_gD = 0.0f;
  for (int t = end - 1; t >= begin; --t) {
    const int row = b * Seq + t;
    const size_t decay_index = static_cast<size_t>(row) * H + h;
    const MambaDecayTermsDevice terms =
        decay_terms != nullptr
            ? mamba_load_decay_terms_dev(decay_terms, decay_index)
            : mamba_decay_terms_dev(dt[decay_index], A[h]);
    const float xv = x[static_cast<size_t>(row) * inner + chan];
    const float go = gy[static_cast<size_t>(row) * inner + chan];
    float gx = go * D[h];
    float ddecay = 0.0f;
    float dinput_scale = 0.0f;
    local_gD += go * xv;
    for (int n = 0; n < N; ++n) {
      const size_t bc =
          static_cast<size_t>(row) * G * N + group * N + n;
      const size_t state_idx = mamba_faithful_state_index(
          row, chan, n, inner, N, state_major_history);
      const float ht = state_history[state_idx];
      const float hprev =
          t == 0
              ? 0.0f
              : state_history[mamba_faithful_state_index(
                    row - 1, chan, n, inner, N, state_major_history)];
      const float c_partial = mamba_warp_sum(go * ht);
      const size_t lds_index =
          mamba2_chunk_lds_index<StateMajorLds>(
              lane, n, N, runtime_warp_size);
      const float gh = go * C_in[bc] + shared_state[lds_index];
      const float b_partial =
          mamba_warp_sum(gh * terms.delta * xv);
      if (lane == 0) {
        const size_t destination =
            bc * partials_per_group + partial_slot;
        partial_C[destination] = c_partial;
        partial_B[destination] = b_partial;
      }
      gx += gh * terms.delta * B_in[bc];
      dinput_scale += gh * B_in[bc] * xv;
      ddecay += gh * hprev;
      shared_state[lds_index] = gh * terms.decay;
    }
    gX[static_cast<size_t>(row) * inner + chan] = gx;
    const float dt_partial = mamba_warp_sum(
        dinput_scale * terms.delta_grad -
        ddecay * terms.decay_dt_factor);
    if (lane == 0) {
      partial_dt[
          (static_cast<size_t>(row) * H + h) * waves_per_head + wave] =
          dt_partial;
    }
    local_gA -= ddecay * terms.decay_alog_factor;
  }
  const size_t lane_partial_index = mamba2_chunk_lane_partial_index(
      b, h, wave, chunk, lane, H, waves_per_head, chunks,
      runtime_warp_size);
  chunk_lane_A[lane_partial_index] = local_gA;
  chunk_lane_D[lane_partial_index] = local_gD;
}

// Boundary-history backward. Each wave reconstructs one state at a time into
// [time,lane] LDS (33*wave floats), then consumes it in reverse. Register arrays
// accumulate channel/time gradients in ascending state order. No full-sequence
// history or global recomputation scratch is allocated.
__global__ void mamba2_faithful_backward_boundary_apply_kernel(
    const float* gy, const float* x, const float* dt, const float* A,
    const float* decay_terms, const float* B_in, const float* C_in,
    const float* D, const float* boundaries, const float* chunk_carry,
    float* gX, float* partial_B, float* partial_C, float* partial_dt,
    float* chunk_lane_A, float* chunk_lane_D, int Seq, int H, int P,
    int N, int G, int waves_per_head, int partials_per_group, int chunks,
    int wave_size, bool state_major_history) {
  constexpr int T = 32;
  extern __shared__ float history[];
  const int chunk = blockIdx.x % chunks;
  const int hw = blockIdx.x / chunks;
  const int wave = hw % waves_per_head;
  const int h = (hw / waves_per_head) % H;
  const int b = hw / (waves_per_head * H);
  const int lane = threadIdx.x;
  const int chan = h * P + wave * wave_size + lane;
  const int inner = H * P;
  const int group = h / (H / G);
  const int slot = (h - group * (H / G)) * waves_per_head + wave;
  const int begin = chunk * T;
  float gx[T], gd[T], ga[T];
#pragma unroll
  for (int k = 0; k < T; ++k) {
    gx[k] = begin + k < Seq ? gy[(static_cast<size_t>(b) * Seq + begin + k) * inner + chan] * D[h] : 0.0f;
    gd[k] = 0.0f;
    ga[k] = 0.0f;
  }
  for (int n = 0; n < N; ++n) {
    float state = boundaries[mamba_faithful_state_index(
        b * chunks + chunk, chan, n, inner, N, state_major_history)];
    history[lane] = state;
#pragma unroll
    for (int k = 0; k < T; ++k) {
      if (begin + k < Seq) {
        const int row = b * Seq + begin + k;
        const size_t di = static_cast<size_t>(row) * H + h;
        const auto terms = decay_terms ? mamba_load_decay_terms_dev(decay_terms, di)
                                      : mamba_decay_terms_dev(dt[di], A[h]);
        const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
        state = terms.decay * state + terms.delta * B_in[bc] * x[static_cast<size_t>(row) * inner + chan];
        history[(k + 1) * wave_size + lane] = state;
      }
    }
    float future = chunk_carry[mamba2_chunk_carry_index(b, h, wave, chunk,
        n, lane, H, waves_per_head, chunks, N, wave_size)];
#pragma unroll
    for (int k = T - 1; k >= 0; --k) {
      if (begin + k < Seq) {
        const int row = b * Seq + begin + k;
        const size_t di = static_cast<size_t>(row) * H + h;
        const auto terms = decay_terms ? mamba_load_decay_terms_dev(decay_terms, di)
                                      : mamba_decay_terms_dev(dt[di], A[h]);
        const size_t bc = static_cast<size_t>(row) * G * N + group * N + n;
        const float go = gy[static_cast<size_t>(row) * inner + chan];
        const float xv = x[static_cast<size_t>(row) * inner + chan];
        const float gh = go * C_in[bc] + future;
        const float cp = mamba_warp_sum(go * history[(k + 1) * wave_size + lane]);
        const float bp = mamba_warp_sum(gh * terms.delta * xv);
        if (lane == 0) {
          partial_C[bc * partials_per_group + slot] = cp;
          partial_B[bc * partials_per_group + slot] = bp;
        }
        gx[k] += gh * terms.delta * B_in[bc];
        gd[k] += gh * B_in[bc] * xv;
        ga[k] += gh * history[k * wave_size + lane];
        future = gh * terms.decay;
      }
    }
  }
  float gA = 0.0f, gD = 0.0f;
#pragma unroll
  for (int k = T - 1; k >= 0; --k) {
    if (begin + k < Seq) {
      const int row = b * Seq + begin + k;
      const size_t di = static_cast<size_t>(row) * H + h;
      const auto terms = decay_terms ? mamba_load_decay_terms_dev(decay_terms, di)
                                    : mamba_decay_terms_dev(dt[di], A[h]);
      const size_t xi = static_cast<size_t>(row) * inner + chan;
      gX[xi] = gx[k];
      const float dp = mamba_warp_sum(gd[k] * terms.delta_grad - ga[k] * terms.decay_dt_factor);
      if (lane == 0) partial_dt[di * waves_per_head + wave] = dp;
      gA -= ga[k] * terms.decay_alog_factor;
      gD += gy[xi] * x[xi];
    }
  }
  const size_t dest = mamba2_chunk_lane_partial_index(b, h, wave, chunk,
      lane, H, waves_per_head, chunks, wave_size);
  chunk_lane_A[dest] = gA;
  chunk_lane_D[dest] = gD;
}

__global__ void mamba2_faithful_backward_chunk_finalize_ad_kernel(
    const float *__restrict__ chunk_lane_A,
    const float *__restrict__ chunk_lane_D,
    float *__restrict__ partial_A, float *__restrict__ partial_D,
    int Batch, int H, int waves_per_head, int chunks,
    int runtime_warp_size) {
  const int wave = blockIdx.x % waves_per_head;
  const int batch_head = blockIdx.x / waves_per_head;
  const int h = batch_head % H;
  const int b = batch_head / H;
  const int lane = threadIdx.x;
  if (b >= Batch || lane >= runtime_warp_size) return;
  float a_sum = 0.0f;
  float d_sum = 0.0f;
  for (int chunk = chunks - 1; chunk >= 0; --chunk) {
    const size_t index = mamba2_chunk_lane_partial_index(
        b, h, wave, chunk, lane, H, waves_per_head, chunks,
        runtime_warp_size);
    a_sum += chunk_lane_A[index];
    d_sum += chunk_lane_D[index];
  }
  const float a_partial = mamba_warp_sum(a_sum);
  const float d_partial = mamba_warp_sum(d_sum);
  if (lane == 0) {
    const size_t destination =
        (static_cast<size_t>(b) * H + h) * waves_per_head + wave;
    partial_A[destination] = a_partial;
    partial_D[destination] = d_partial;
  }
}

__global__ void mamba2_faithful_reduce_bc_partials_kernel(
    const float *__restrict__ partial_B,
    const float *__restrict__ partial_C, float *__restrict__ gB,
    float *__restrict__ gC, int values, int partials_per_value) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const size_t base =
      static_cast<size_t>(index) * partials_per_value;
  float b_sum = 0.0f;
  float c_sum = 0.0f;
  for (int partial = 0; partial < partials_per_value; ++partial) {
    b_sum += partial_B[base + partial];
    c_sum += partial_C[base + partial];
  }
  gB[index] = b_sum;
  gC[index] = c_sum;
}

__global__ void mamba2_faithful_reduce_dt_partials_kernel(
    const float *__restrict__ partial_dt, float *__restrict__ gDt,
    int values, int waves_per_head) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const size_t base =
      static_cast<size_t>(index) * waves_per_head;
  float sum = 0.0f;
  for (int wave = 0; wave < waves_per_head; ++wave) {
    sum += partial_dt[base + wave];
  }
  gDt[index] = sum;
}

__global__ void mamba2_faithful_reduce_ad_partials_kernel(
    const float *__restrict__ partial_A,
    const float *__restrict__ partial_D, float *__restrict__ gA,
    float *__restrict__ gD, int Batch, int H, int waves_per_head) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float a_sum = 0.0f;
  float d_sum = 0.0f;
  for (int b = 0; b < Batch; ++b) {
    const size_t base =
        (static_cast<size_t>(b) * H + h) * waves_per_head;
    for (int wave = 0; wave < waves_per_head; ++wave) {
      a_sum += partial_A[base + wave];
      d_sum += partial_D[base + wave];
    }
  }
  gA[h] = a_sum;
  gD[h] = d_sum;
}

bool launch_mamba2_faithful_precompute_decay(
    const float *dt, const float *A, float *packed_terms,
    int Batch, int Seq, int H) {
  int rows = 0;
  int values = 0;
  if (dt == nullptr || A == nullptr || packed_terms == nullptr ||
      !checked_positive_product_to_int(Batch, Seq, &rows) ||
      !checked_positive_product_to_int(rows, H, &values)) {
    return false;
  }
  constexpr int threads = 256;
  mamba2_faithful_precompute_decay_kernel
      <<<nsos::gpu::ceil_div_positive(values, threads), threads, 0, nsos::gpu::current_stream()>>>(
          dt, A, packed_terms, values, H);
  return true;
}

bool launch_mamba2_faithful_forward(
    const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in,
    const float *C_in, const float *D, float *y, float *state_history,
    int Batch, int Seq, int H, int P, int N, int G,
    int runtime_warp_size, bool state_major_history, int history_chunk_size) {
  if (history_chunk_size != 0 && history_chunk_size != 32) return false;
  int rows = 0;
  int total = 0;
  int head_blocks = 0;
  int output_values = 0;
  int state_values = 0;
  if (x == nullptr || dt == nullptr || A == nullptr || B_in == nullptr ||
      C_in == nullptr || D == nullptr || y == nullptr || Batch <= 0 ||
      Seq <= 0 || H <= 0 || P <= 0 || N <= 0 || N > MAX_N || G <= 0 ||
      H % G != 0 || runtime_warp_size <= 0 ||
      !checked_positive_product_to_int(Batch, Seq, &rows) ||
      !checked_positive_product_to_int(Batch, H, P, &total) ||
      !checked_positive_product_to_int(Batch, H, &head_blocks) ||
      !checked_positive_product_to_int(rows, H, P, &output_values) ||
      (state_history != nullptr &&
       !checked_positive_product_to_int(output_values, N,
                                        &state_values))) {
    return false;
  }
  const bool head_geometry =
      faithful_head_channel_geometry_enabled(P);
  const bool head_wave_geometry =
      faithful_deterministic_head_wave_geometry_enabled(
          P, runtime_warp_size);
  const int waves_per_head =
      head_wave_geometry ? P / runtime_warp_size : 0;
  int wave_blocks = 0;
  if (head_wave_geometry &&
      !checked_positive_product_to_int(
          head_blocks, waves_per_head, &wave_blocks)) {
    return false;
  }
  const int threads = head_wave_geometry
                          ? runtime_warp_size
                          : (head_geometry ? P : 256);
  const int blocks = head_wave_geometry
                         ? wave_blocks
                         : (head_geometry
                                ? head_blocks
                                : nsos::gpu::ceil_div_positive(
                                      total, threads));
  // Dispatch the compile-time state-width specialization when the configured
  // state fits an instantiated width.  Both kernels emit the same operations
  // in the same order; the specialization only keeps `state` in registers
  // instead of scratch, so the numeric result is bit-identical.
#define NSOS_FAITHFUL_FORWARD_FIXED_LAUNCH(WIDTH, HISTORY)               \
  mamba2_faithful_forward_fixed_kernel<WIDTH, HISTORY><<<blocks, threads, 0, nsos::gpu::current_stream()>>>(     \
      x, dt, A, decay_terms, B_in, C_in, D, y, state_history, Batch,    \
      Seq, H, P, N, G, waves_per_head, head_geometry,                   \
      head_wave_geometry, state_major_history, history_chunk_size)

#define NSOS_FAITHFUL_FORWARD_FIXED(WIDTH) do {                         \
  if (history_chunk_size == 32) { NSOS_FAITHFUL_FORWARD_FIXED_LAUNCH(WIDTH, 32); } \
  else { NSOS_FAITHFUL_FORWARD_FIXED_LAUNCH(WIDTH, 0); }                  \
} while (0)

  if (faithful_fixed_state_forward_enabled()) {
    switch (N) {
      case 64:
        NSOS_FAITHFUL_FORWARD_FIXED(64);
        return true;
      case 32:
        NSOS_FAITHFUL_FORWARD_FIXED(32);
        return true;
      case 16:
        NSOS_FAITHFUL_FORWARD_FIXED(16);
        return true;
      default:
        break;
    }
  }
#undef NSOS_FAITHFUL_FORWARD_FIXED
#undef NSOS_FAITHFUL_FORWARD_FIXED_LAUNCH

#define NSOS_FAITHFUL_FORWARD_GENERIC(HISTORY)                          \
  mamba2_faithful_forward_kernel<HISTORY><<<blocks, threads, 0, nsos::gpu::current_stream()>>>( \
      x, dt, A, decay_terms, B_in, C_in, D, y, state_history, Batch,      \
      Seq, H, P, N, G, waves_per_head, head_geometry,                     \
      head_wave_geometry, state_major_history, history_chunk_size)
  if (history_chunk_size == 32) { NSOS_FAITHFUL_FORWARD_GENERIC(32); }
  else { NSOS_FAITHFUL_FORWARD_GENERIC(0); }
#undef NSOS_FAITHFUL_FORWARD_GENERIC
  return true;
}

bool launch_mamba2_faithful_forward_chunked(
    const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D, float *y, float *state_history,
    float *chunk_end_local, float *chunk_total_decay,
    float *carry_entering, int Batch, int Seq, int H, int P, int N,
    int G, int runtime_warp_size, bool state_major_history,
    int *out_chunks, int history_chunk_size) {
  if (history_chunk_size != 0 &&
      (history_chunk_size != 32 || faithful_forward_chunk_size() % 32 != 0))
    return false;
  int channel_values = 0;
  int channel_states = 0;
  if (x == nullptr || dt == nullptr || A == nullptr || B_in == nullptr ||
      C_in == nullptr || D == nullptr || y == nullptr ||
      chunk_end_local == nullptr || chunk_total_decay == nullptr ||
      carry_entering == nullptr || Batch <= 0 || Seq <= 0 || H <= 0 ||
      P <= 0 || N <= 0 || N > MAX_N || G <= 0 || H % G != 0 ||
      runtime_warp_size <= 0 || P % runtime_warp_size != 0 ||
      !checked_positive_product_to_int(Batch, H, P, &channel_values) ||
      !checked_positive_product_to_int(channel_values, N,
                                       &channel_states)) {
    return false;
  }
  // Only the compile-time state widths are served here: the whole point of
  // the decomposition is to keep the per-chunk state in registers.
  if (N != 64 && N != 32 && N != 16) return false;

  const int chunk_size = faithful_forward_chunk_size();
  const int chunks = nsos::gpu::ceil_div_positive(Seq, chunk_size);
  if (chunks <= 1) return false;  // nothing to parallelize over
  const int waves_per_head = P / runtime_warp_size;
  int head_blocks = 0;
  int wave_blocks = 0;
  int chunk_blocks = 0;
  if (waves_per_head <= 0 ||
      !checked_positive_product_to_int(Batch, H, &head_blocks) ||
      !checked_positive_product_to_int(head_blocks, waves_per_head,
                                       &wave_blocks) ||
      !checked_positive_product_to_int(wave_blocks, chunks,
                                       &chunk_blocks)) {
    return false;
  }
  if (out_chunks != nullptr) *out_chunks = chunks;

#define NSOS_FAITHFUL_FORWARD_CHUNK_APPLY(WIDTH, HISTORY)                 \
    mamba2_faithful_forward_chunk_apply_kernel<WIDTH, HISTORY>            \
        <<<chunk_blocks, runtime_warp_size, 0, nsos::gpu::current_stream()>>>( \
            x, dt, A, decay_terms, B_in, C_in, D, carry_entering, y,       \
            state_history, Batch, Seq, H, P, N, G, waves_per_head,         \
            chunks, chunk_size, channel_states, state_major_history, history_chunk_size)

#define NSOS_FAITHFUL_FORWARD_CHUNKED(WIDTH)                              \
  do {                                                                    \
    mamba2_faithful_forward_chunk_summary_kernel<WIDTH>                   \
        <<<chunk_blocks, runtime_warp_size, 0, nsos::gpu::current_stream()>>>(                            \
            x, dt, A, decay_terms, B_in, chunk_end_local,                 \
            chunk_total_decay, Batch, Seq, H, P, N, G, waves_per_head,    \
            chunks, chunk_size, channel_states);                          \
    if (cudaGetLastError() != cudaSuccess) return false;                  \
    constexpr int kPrefixThreads = 256;                                   \
    mamba2_faithful_forward_chunk_prefix_kernel<<<                        \
        nsos::gpu::ceil_div_positive(channel_states, kPrefixThreads),     \
        kPrefixThreads, 0, nsos::gpu::current_stream()>>>(chunk_end_local, chunk_total_decay,             \
                          carry_entering, H, P, N, chunks,                \
                          channel_states);                                \
    if (cudaGetLastError() != cudaSuccess) return false;                  \
    if (history_chunk_size == 32) { NSOS_FAITHFUL_FORWARD_CHUNK_APPLY(WIDTH, 32); } \
    else { NSOS_FAITHFUL_FORWARD_CHUNK_APPLY(WIDTH, 0); }                  \
    return cudaGetLastError() == cudaSuccess;                             \
  } while (0)

  switch (N) {
    case 64:
      NSOS_FAITHFUL_FORWARD_CHUNKED(64);
    case 32:
      NSOS_FAITHFUL_FORWARD_CHUNKED(32);
    case 16:
      NSOS_FAITHFUL_FORWARD_CHUNKED(16);
    default:
      break;
  }
#undef NSOS_FAITHFUL_FORWARD_CHUNKED
#undef NSOS_FAITHFUL_FORWARD_CHUNK_APPLY
  return false;
}

bool launch_mamba2_faithful_backward(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D,
    const float *state_history, float *gX, float *gDt, float *gA,
    float *gB, float *gC, float *gD, int Batch, int Seq, int H, int P,
    int N, int G, bool state_major_history) {
  int rows = 0;
  int total = 0;
  int head_blocks = 0;
  int bc_values = 0;
  int dt_values = 0;
  int output_values = 0;
  int state_values = 0;
  if (gy == nullptr || x == nullptr || dt == nullptr || A == nullptr ||
      B_in == nullptr || C_in == nullptr || D == nullptr ||
      state_history == nullptr || gX == nullptr || gDt == nullptr ||
      gA == nullptr || gB == nullptr || gC == nullptr || gD == nullptr ||
      Batch <= 0 || Seq <= 0 || H <= 0 || P <= 0 || N <= 0 ||
      N > MAX_N || G <= 0 || H % G != 0 ||
      !checked_positive_product_to_int(Batch, Seq, &rows) ||
      !checked_positive_product_to_int(Batch, H, P, &total) ||
      !checked_positive_product_to_int(Batch, H, &head_blocks) ||
      !checked_positive_product_to_int(rows, G, N, &bc_values) ||
      !checked_positive_product_to_int(rows, H, &dt_values) ||
      !checked_positive_product_to_int(rows, H, P, &output_values) ||
      !checked_positive_product_to_int(output_values, N,
                                       &state_values)) {
    return false;
  }
  if (cudaMemsetAsync(gDt, 0,
                      static_cast<size_t>(dt_values) * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(gA, 0, static_cast<size_t>(H) * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(gB, 0,
                      static_cast<size_t>(bc_values) * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(gC, 0,
                      static_cast<size_t>(bc_values) * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess ||
      cudaMemsetAsync(gD, 0, static_cast<size_t>(H) * sizeof(float), nsos::gpu::current_stream()) !=
          cudaSuccess) {
    return false;
  }
  const bool head_geometry =
      faithful_head_channel_geometry_enabled(P);
  const int threads = head_geometry ? P : 256;
  const int blocks = head_geometry
                         ? head_blocks
                         : nsos::gpu::ceil_div_positive(
                               total, threads);
  mamba2_faithful_backward_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      gy, x, dt, A, decay_terms, B_in, C_in, D, state_history, gX,
      gDt, gA, gB, gC, gD, Batch, Seq, H, P, N, G,
      faithful_warp_aggregation_enabled(), head_geometry,
      state_major_history);
  return true;
}

bool launch_mamba2_faithful_backward_deterministic(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D,
    const float *state_history, float *gX, float *gDt, float *gA,
    float *gB, float *gC, float *gD, float *partial_B,
    float *partial_C, float *partial_dt, float *partial_A,
    float *partial_D, float *gh_history, int Batch, int Seq, int H,
    int P, int N, int G, int runtime_warp_size,
    bool state_major_history) {
  int rows = 0;
  int total = 0;
  int bc_values = 0;
  int dt_values = 0;
  int output_values = 0;
  int state_values = 0;
  if (gy == nullptr || x == nullptr || dt == nullptr || A == nullptr ||
      B_in == nullptr || C_in == nullptr || D == nullptr ||
      state_history == nullptr || gX == nullptr || gDt == nullptr ||
      gA == nullptr || gB == nullptr || gC == nullptr || gD == nullptr ||
      partial_B == nullptr || partial_C == nullptr ||
      partial_dt == nullptr || partial_A == nullptr ||
      partial_D == nullptr || Batch <= 0 || Seq <= 0 || H <= 0 ||
      P <= 0 || N <= 0 || N > MAX_N || G <= 0 || H % G != 0 ||
      runtime_warp_size <= 0 || P % runtime_warp_size != 0 ||
      !checked_positive_product_to_int(Batch, Seq, &rows) ||
      !checked_positive_product_to_int(Batch, H, P, &total) ||
      !checked_positive_product_to_int(rows, G, N, &bc_values) ||
      !checked_positive_product_to_int(rows, H, &dt_values) ||
      !checked_positive_product_to_int(rows, H, P, &output_values) ||
      !checked_positive_product_to_int(output_values, N,
                                       &state_values)) {
    return false;
  }
  const int waves_per_head = P / runtime_warp_size;
  int partials_per_group = 0;
  if (waves_per_head <= 0 ||
      !checked_positive_product_to_int(H / G, waves_per_head,
                                       &partials_per_group)) {
    return false;
  }
  constexpr int threads = 256;
  const bool head_geometry =
      faithful_head_channel_geometry_enabled(P);
  const bool head_wave_geometry =
      faithful_deterministic_head_wave_geometry_enabled(
          P, runtime_warp_size);
  const bool shared_carry =
      head_wave_geometry && faithful_deterministic_shared_carry_enabled();
  const bool state_parallel =
      head_wave_geometry && faithful_state_parallel_backward_enabled();
  if (state_parallel && gh_history == nullptr) {
    return false;
  }
  const int partial_threads =
      head_wave_geometry ? runtime_warp_size
                         : (head_geometry ? P : threads);
  int partial_blocks = 0;
  if (head_wave_geometry) {
    if (!checked_positive_product_to_int(
            Batch, H, waves_per_head, &partial_blocks)) {
      return false;
    }
  } else {
    partial_blocks = head_geometry
                         ? Batch * H
                         : nsos::gpu::ceil_div_positive(
                               total, partial_threads);
  }
  const size_t shared_carry_bytes =
      shared_carry
          ? static_cast<size_t>(partial_threads) * N * sizeof(float)
          : 0;
  if (state_parallel) {
    int state_parallel_blocks = 0;
    if (!checked_positive_product_to_int(
            partial_blocks, N, &state_parallel_blocks)) {
      return false;
    }
    mamba2_faithful_backward_state_parallel_kernel
        <<<state_parallel_blocks, runtime_warp_size, 0, nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, C_in, state_history,
            gh_history, partial_B, partial_C, Batch, Seq, H, P, N, G,
            waves_per_head, partials_per_group, state_major_history);
    mamba2_faithful_backward_state_parallel_finalize_kernel
        <<<partial_blocks, runtime_warp_size, 0, nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, D, state_history,
            gh_history, gX, partial_dt, partial_A, partial_D, Batch, Seq,
            H, P, N, G, waves_per_head, state_major_history);
  } else if (shared_carry) {
    mamba2_faithful_backward_deterministic_partials_kernel<true>
        <<<partial_blocks, partial_threads, shared_carry_bytes, nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, C_in, D, state_history, gX,
            partial_B, partial_C, partial_dt, partial_A, partial_D,
            Batch, Seq, H, P, N, G, waves_per_head,
            partials_per_group, head_geometry, head_wave_geometry,
            state_major_history);
  } else {
    mamba2_faithful_backward_deterministic_partials_kernel<false>
        <<<partial_blocks, partial_threads, 0, nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, C_in, D, state_history, gX,
            partial_B, partial_C, partial_dt, partial_A, partial_D,
            Batch, Seq, H, P, N, G, waves_per_head,
            partials_per_group, head_geometry, head_wave_geometry,
            state_major_history);
  }
  mamba2_faithful_reduce_bc_partials_kernel
      <<<nsos::gpu::ceil_div_positive(bc_values, threads), threads, 0, nsos::gpu::current_stream()>>>(
          partial_B, partial_C, gB, gC, bc_values,
          partials_per_group);
  mamba2_faithful_reduce_dt_partials_kernel
      <<<nsos::gpu::ceil_div_positive(dt_values, threads), threads, 0, nsos::gpu::current_stream()>>>(
          partial_dt, gDt, dt_values, waves_per_head);
  mamba2_faithful_reduce_ad_partials_kernel
      <<<nsos::gpu::ceil_div_positive(H, threads), threads, 0, nsos::gpu::current_stream()>>>(
          partial_A, partial_D, gA, gD, Batch, H, waves_per_head);
  return true;
}

bool launch_mamba2_faithful_backward_deterministic_chunked(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D, const float *state_history, float *gX,
    float *gDt, float *gA, float *gB, float *gC, float *gD,
    float *partial_B, float *partial_C, float *partial_dt,
    float *partial_A, float *partial_D, float *chunk_carry,
    float *chunk_scale, float *chunk_lane_A, float *chunk_lane_D,
    int Batch, int Seq, int H, int P, int N, int G,
    int runtime_warp_size, bool state_major_history, int history_chunk_size) {
  if (history_chunk_size != 0 &&
      (history_chunk_size != 32 || faithful_backward_chunk_size() != 32))
    return false;
  int rows = 0;
  int channel_values = 0;
  int channel_states = 0;
  int bc_values = 0;
  int dt_values = 0;
  if (gy == nullptr || x == nullptr || dt == nullptr || A == nullptr ||
      B_in == nullptr || C_in == nullptr || D == nullptr ||
      state_history == nullptr || gX == nullptr || gDt == nullptr ||
      gA == nullptr || gB == nullptr || gC == nullptr || gD == nullptr ||
      partial_B == nullptr || partial_C == nullptr ||
      partial_dt == nullptr || partial_A == nullptr ||
      partial_D == nullptr || chunk_carry == nullptr ||
      chunk_scale == nullptr || chunk_lane_A == nullptr ||
      chunk_lane_D == nullptr || Batch <= 0 || Seq <= 0 || H <= 0 ||
      P <= 0 || N <= 0 || N > MAX_N || G <= 0 || H % G != 0 ||
      runtime_warp_size <= 0 || P % runtime_warp_size != 0 ||
      !checked_positive_product_to_int(Batch, Seq, &rows) ||
      !checked_positive_product_to_int(Batch, H, P, &channel_values) ||
      !checked_positive_product_to_int(channel_values, N,
                                       &channel_states) ||
      !checked_positive_product_to_int(rows, G, N, &bc_values) ||
      !checked_positive_product_to_int(rows, H, &dt_values)) {
    return false;
  }
  const int waves_per_head = P / runtime_warp_size;
  int partials_per_group = 0;
  int partial_blocks = 0;
  if (waves_per_head <= 0 ||
      !checked_positive_product_to_int(H / G, waves_per_head,
                                       &partials_per_group) ||
      !checked_positive_product_to_int(Batch, H, waves_per_head,
                                       &partial_blocks)) {
    return false;
  }
  const int chunk_size = faithful_backward_chunk_size();
  const int chunks = nsos::gpu::ceil_div_positive(Seq, chunk_size);
  int chunk_blocks = 0;
  if (chunk_size <= 0 || chunks <= 0 ||
      !checked_positive_product_to_int(partial_blocks, chunks,
                                       &chunk_blocks)) {
    return false;
  }
  const size_t shared_state_bytes =
      static_cast<size_t>(runtime_warp_size) * N * sizeof(float);
  const bool state_major_lds =
      faithful_chunk_lds_state_major_enabled();

  if (state_major_lds) {
    mamba2_faithful_backward_chunk_summary_kernel<true>
        <<<chunk_blocks, runtime_warp_size, shared_state_bytes, nsos::gpu::current_stream()>>>(
            gy, dt, A, decay_terms, C_in, chunk_carry, chunk_scale,
            Batch, Seq, H, P, N, G, waves_per_head, chunks, chunk_size,
            runtime_warp_size);
  } else {
    mamba2_faithful_backward_chunk_summary_kernel<false>
        <<<chunk_blocks, runtime_warp_size, shared_state_bytes, nsos::gpu::current_stream()>>>(
            gy, dt, A, decay_terms, C_in, chunk_carry, chunk_scale,
            Batch, Seq, H, P, N, G, waves_per_head, chunks, chunk_size,
            runtime_warp_size);
  }
  if (cudaGetLastError() != cudaSuccess) return false;

  constexpr int threads = 256;
  mamba2_faithful_backward_chunk_prefix_kernel
      <<<nsos::gpu::ceil_div_positive(channel_states, threads), threads, 0, nsos::gpu::current_stream()>>>(
          chunk_carry, chunk_scale, Batch, H, P, N, waves_per_head,
          chunks, runtime_warp_size, channel_states);
  if (cudaGetLastError() != cudaSuccess) return false;

  if (history_chunk_size == 32) {
    mamba2_faithful_backward_boundary_apply_kernel
        <<<chunk_blocks, runtime_warp_size, 33 * runtime_warp_size * sizeof(float), nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, C_in, D, state_history,
            chunk_carry, gX, partial_B, partial_C, partial_dt,
            chunk_lane_A, chunk_lane_D, Seq, H, P, N, G, waves_per_head,
            partials_per_group, chunks, runtime_warp_size, state_major_history);
  } else if (state_major_lds) {
    mamba2_faithful_backward_chunk_apply_kernel<true>
        <<<chunk_blocks, runtime_warp_size, shared_state_bytes, nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, C_in, D, state_history,
            chunk_carry, gX, partial_B, partial_C, partial_dt,
            chunk_lane_A, chunk_lane_D, Batch, Seq, H, P, N, G,
            waves_per_head, partials_per_group, chunks, chunk_size,
            runtime_warp_size, state_major_history);
  } else {
    mamba2_faithful_backward_chunk_apply_kernel<false>
        <<<chunk_blocks, runtime_warp_size, shared_state_bytes, nsos::gpu::current_stream()>>>(
            gy, x, dt, A, decay_terms, B_in, C_in, D, state_history,
            chunk_carry, gX, partial_B, partial_C, partial_dt,
            chunk_lane_A, chunk_lane_D, Batch, Seq, H, P, N, G,
            waves_per_head, partials_per_group, chunks, chunk_size,
            runtime_warp_size, state_major_history);
  }
  if (cudaGetLastError() != cudaSuccess) return false;

  mamba2_faithful_backward_chunk_finalize_ad_kernel
      <<<partial_blocks, runtime_warp_size, 0, nsos::gpu::current_stream()>>>(
          chunk_lane_A, chunk_lane_D, partial_A, partial_D, Batch, H,
          waves_per_head, chunks, runtime_warp_size);
  if (cudaGetLastError() != cudaSuccess) return false;

  mamba2_faithful_reduce_bc_partials_kernel
      <<<nsos::gpu::ceil_div_positive(bc_values, threads), threads, 0, nsos::gpu::current_stream()>>>(
          partial_B, partial_C, gB, gC, bc_values,
          partials_per_group);
  if (cudaGetLastError() != cudaSuccess) return false;
  mamba2_faithful_reduce_dt_partials_kernel
      <<<nsos::gpu::ceil_div_positive(dt_values, threads), threads, 0, nsos::gpu::current_stream()>>>(
          partial_dt, gDt, dt_values, waves_per_head);
  if (cudaGetLastError() != cudaSuccess) return false;
  mamba2_faithful_reduce_ad_partials_kernel
      <<<nsos::gpu::ceil_div_positive(H, threads), threads, 0, nsos::gpu::current_stream()>>>(
          partial_A, partial_D, gA, gD, Batch, H, waves_per_head);
  return cudaGetLastError() == cudaSuccess;
}

__global__ void mamba2_faithful_conv_step_kernel(
    const float *__restrict__ xv, const float *__restrict__ Bv,
    const float *__restrict__ Cv, const float *__restrict__ weight,
    const float *__restrict__ bias, float *__restrict__ ring,
    float *__restrict__ xBC, int inner, int group_state, int convdim, int K) {
  const size_t row = blockIdx.y;
  xv += row * inner;
  Bv += row * group_state;
  Cv += row * group_state;
  ring += row * max((K - 1) * convdim, 1);
  xBC += row * convdim;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= convdim) return;
  const float current =
      c < inner ? xv[c] : (c < inner + group_state
                               ? Bv[c - inner]
                               : Cv[c - inner - group_state]);
  float acc = bias[c];
  for (int j = 0; j < K; ++j) {
    const float src = j < K - 1 ? ring[j * convdim + c] : current;
    acc += weight[c * K + j] * src;
  }
  xBC[c] = acc * (1.0f / (1.0f + expf(-acc)));
  for (int s = 0; s + 1 < K - 1; ++s) {
    ring[s * convdim + c] = ring[(s + 1) * convdim + c];
  }
  if (K > 1) ring[(K - 2) * convdim + c] = current;
}

void launch_mamba2_faithful_conv_step(
    const float *xv, const float *Bv, const float *Cv,
    const float *conv_weight, const float *conv_bias, float *ring,
    float *xBC, int inner, int group_state, int K, int rows) {
  int convdim = 0;
  if (K <= 0 || !checked_faithful_convdim(inner, group_state, &convdim)) {
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(convdim, threads);
  mamba2_faithful_conv_step_kernel<<<dim3(blocks, rows), threads, 0, nsos::gpu::current_stream()>>>(
      xv, Bv, Cv, conv_weight, conv_bias, ring, xBC, inner, group_state,
      convdim, K);
}

__global__ void mamba2_faithful_step_kernel(
    const float *__restrict__ xBC, const float *__restrict__ dt,
    const float *__restrict__ A, const float *__restrict__ D,
    float *__restrict__ state, float *__restrict__ y, int H, int P, int N,
    int G) {
  const int chan = blockIdx.x * blockDim.x + threadIdx.x;
  const int inner = H * P;
  if (chan >= inner) return;
  const int h = chan / P;
  const int group = (h * G) / H;
  const int group_state = G * N;
  xBC += size_t(blockIdx.y) * (inner + 2 * group_state);
  dt += size_t(blockIdx.y) * H;
  state += size_t(blockIdx.y) * inner * N;
  y += size_t(blockIdx.y) * inner;
  const float xv = xBC[chan];
  const float *B = xBC + inner + group * N;
  const float *C = xBC + inner + group_state + group * N;
  const MambaDecayTermsDevice terms = mamba_decay_terms_dev(dt[h], A[h]);
  const float delta = terms.delta;
  const float decay = terms.decay;
  float *state_row = state + static_cast<size_t>(chan) * N;
  float out = D[h] * xv;
  for (int n = 0; n < N; ++n) {
    const float hv = decay * state_row[n] + delta * B[n] * xv;
    state_row[n] = hv;
    out += hv * C[n];
  }
  y[chan] = out;
}

void launch_mamba2_faithful_step(
    const float *xBC, const float *dt, const float *A, const float *D,
    float *state, float *y, int H, int P, int N, int G, int rows) {
  int inner = 0;
  if (N <= 0 || N > MAX_N || G <= 0 || H <= 0 || H % G != 0 ||
      !checked_positive_product_to_int(H, P, &inner)) {
    return;
  }
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(inner, threads);
  mamba2_faithful_step_kernel<<<dim3(blocks, rows), threads, 0, nsos::gpu::current_stream()>>>(
      xBC, dt, A, D, state, y, H, P, N, G);
}

__global__ __launch_bounds__(256) void mamba_gated_rmsnorm_kernel(
    const float* y, const float* z, const float* weight, float* output, int dim, float epsilon) {
  const size_t base = size_t(blockIdx.x) * dim;
  float sum = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float gate = z[base + d] * (1.0f / (1.0f + expf(-z[base + d])));
    const float value = y[base + d] * gate;
    output[base + d] = value;
    sum += value * value;
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = rsqrtf(reduction[0] / dim + epsilon);
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    output[base + d] = (output[base + d] * inverse) * weight[d];
}
void launch_mamba_gated_rmsnorm(const float* y, const float* z,
    const float* weight, float* output, int rows, int dim, float epsilon) {
  if (rows < 1 || dim < 1 || !(epsilon > 0)) throw std::invalid_argument("Invalid Mamba RMS geometry");
  mamba_gated_rmsnorm_kernel<<<rows, 256, 0, nsos::gpu::current_stream()>>>(y, z, weight, output, dim, epsilon);
}

__global__ void mamba_prime_stream_carry_kernel(
    const float *__restrict__ history, int state_width,
    const float *__restrict__ x_values, int x_width,
    const float *__restrict__ b_values, int b_width,
    const float *__restrict__ c_values, int c_width,
    float *__restrict__ state, float *__restrict__ ring,
    int batch, int seq, int taps, int ring_width, int ring_elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < state_width) {
    const int last_row = batch * seq - 1;
    state[index] =
        history[static_cast<size_t>(last_row) * state_width + index];
  }
  if (index >= ring_elements) return;
  if (taps <= 0) {
    if (index == 0) ring[0] = 0.0f;
    return;
  }

  const int slot = index / ring_width;
  const int channel = index - slot * ring_width;
  const int source_time = seq - taps + slot;
  if (source_time < 0) {
    ring[index] = 0.0f;
    return;
  }
  const int source_row = (batch - 1) * seq + source_time;
  if (channel < x_width) {
    ring[index] =
        x_values[static_cast<size_t>(source_row) * x_width + channel];
    return;
  }
  const int after_x = channel - x_width;
  if (after_x < b_width) {
    ring[index] =
        b_values[static_cast<size_t>(source_row) * b_width + after_x];
    return;
  }
  const int after_b = after_x - b_width;
  ring[index] =
      c_values[static_cast<size_t>(source_row) * c_width + after_b];
}

bool launch_mamba_prime_stream_carry(
    const float *history, int state_width,
    const float *x_values, int x_width,
    const float *b_values, int b_width,
    const float *c_values, int c_width,
    float *state, float *ring,
    int batch, int seq, int taps) {
  if (history == nullptr || x_values == nullptr ||
      state == nullptr || ring == nullptr ||
      state_width <= 0 || x_width <= 0 ||
      b_width < 0 || c_width < 0 ||
      batch <= 0 || seq <= 0 || taps < 0 ||
      (b_width > 0 && b_values == nullptr) ||
      (c_width > 0 && c_values == nullptr)) {
    return false;
  }
  const int64_t ring_width_64 =
      static_cast<int64_t>(x_width) + b_width + c_width;
  if (ring_width_64 <= 0 || ring_width_64 > INT_MAX) {
    return false;
  }
  const int ring_width = static_cast<int>(ring_width_64);
  int ring_elements = 1;
  int rows = 0;
  int history_values = 0;
  if ((taps > 0 &&
       !checked_positive_product_to_int(taps, ring_width,
                                        &ring_elements)) ||
      !checked_positive_product_to_int(batch, seq, &rows) ||
      !checked_positive_product_to_int(rows, state_width,
                                       &history_values)) {
    return false;
  }
  const int total =
      state_width > ring_elements ? state_width : ring_elements;
  const int threads = 256;
  const int blocks = nsos::gpu::ceil_div_positive(total, threads);
  mamba_prime_stream_carry_kernel<<<blocks, threads, 0, nsos::gpu::current_stream()>>>(
      history, state_width, x_values, x_width, b_values, b_width,
      c_values, c_width, state, ring, batch, seq, taps,
      ring_width, ring_elements);
  return true;
}

// =====================================================================
// Fused single-token incremental decode step (proper diagonal path).
// One thread per channel.  Mirrors forward_proper_step's host math exactly:
//   conv_pre = sum_j convw[c,j] * window[j]   (window = [ring taps..., xv])
//   xc       = silu(conv_pre)
//   h        = decay*h + softplus(dt)*B*xc
//              decay = exp(-softplus(dt)*exp(A_log))  (N1)
//   y        = h*C
//   gated    = y * silu(z)
// then advances the conv ring in place (shift left, append xv).  Keeps the SSD
// state h and ring resident on the device across tokens -> no per-token host
// round-trip, and capturable by a CUDA graph.
// =====================================================================
__global__ void mamba_proper_step_kernel(
    const float *__restrict__ xv, const float *__restrict__ z,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ convw, float *__restrict__ ring,
    float *__restrict__ h, float *__restrict__ gated, int dim, int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;

  float conv_pre = 0.0f;
  for (int j = 0; j < K; ++j) {
    const float src = (j < K - 1) ? ring[j * dim + c] : xv[c];
    conv_pre += convw[c * K + j] * src;
  }
  const float xc = conv_pre * (1.0f / (1.0f + expf(-conv_pre)));  // silu

  const MambaDecayTermsDevice terms = mamba_decay_terms_dev(dt[c], A[c]);
  const float dt_scale = terms.delta;
  const float decay = terms.decay;
  const float st = decay * h[c] + dt_scale * B_in[c] * xc;
  h[c] = st;
  const float y = st * C_in[c];
  const float gate = z[c] * (1.0f / (1.0f + expf(-z[c])));  // silu(z)
  gated[c] = y * gate;

  // Advance the conv window: shift taps left, append current xv.  Each thread
  // owns channel c, so this is race-free across channels and the reads above
  // used the pre-shift ring.
  for (int s = 0; s + 1 < K - 1; ++s) {
    ring[s * dim + c] = ring[(s + 1) * dim + c];
  }
  if (K - 1 > 0) {
    ring[(K - 2) * dim + c] = xv[c];
  }
}

void launch_mamba_proper_step(const float *xv, const float *z,
                              const float *B_in, const float *C_in,
                              const float *dt, const float *A,
                              const float *convw, float *ring, float *h,
                              float *gated, int dim, int K) {
  const int block = 256;
  const int grid = nsos::gpu::ceil_div_positive(dim, block);
  mamba_proper_step_kernel<<<grid, block, 0, nsos::gpu::current_stream()>>>(xv, z, B_in, C_in, dt, A, convw,
                                            ring, h, gated, dim, K);
}

// N-state single-token decode step (full Mamba-2).  One thread per channel
// c = head*P + p; the per-channel N-vector state lives contiguously at
// h[c*N .. c*N+N-1] (same layout as the host step's (h*P+p)*N + n indexing).
__global__ void mamba_nstate_step_kernel(
    const float *__restrict__ xv, const float *__restrict__ z,
    const float *__restrict__ B_in, const float *__restrict__ C_in,
    const float *__restrict__ dt, const float *__restrict__ A,
    const float *__restrict__ convw, float *__restrict__ ring,
    float *__restrict__ h, float *__restrict__ gated, int dim, int K, int P,
    int N) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;

  float conv_pre = 0.0f;
  for (int j = 0; j < K; ++j) {
    const float src = (j < K - 1) ? ring[j * dim + c] : xv[c];
    conv_pre += convw[c * K + j] * src;
  }
  const float xc = conv_pre * (1.0f / (1.0f + expf(-conv_pre)));  // silu

  const int head = c / max(P, 1);
  const MambaDecayTermsDevice terms =
      mamba_decay_terms_dev(dt[head], A[head]);
  const float dt_scale = terms.delta;
  const float decay = terms.decay;
  const float *b_row = B_in + head * N;
  const float *c_row = C_in + head * N;
  float *h_row = h + static_cast<size_t>(c) * N;
  float y_acc = 0.0f;
  for (int n = 0; n < N; ++n) {
    const float hv = decay * h_row[n] + dt_scale * b_row[n] * xc;
    h_row[n] = hv;
    y_acc += hv * c_row[n];
  }
  const float gate = z[c] * (1.0f / (1.0f + expf(-z[c])));  // silu(z)
  gated[c] = y_acc * gate;

  // Advance the conv window (per-channel, race-free; reads above used the
  // pre-shift ring).
  for (int s = 0; s + 1 < K - 1; ++s) {
    ring[s * dim + c] = ring[(s + 1) * dim + c];
  }
  if (K - 1 > 0) {
    ring[(K - 2) * dim + c] = xv[c];
  }
}

void launch_mamba_nstate_step(const float *xv, const float *z,
                              const float *B_in, const float *C_in,
                              const float *dt, const float *A,
                              const float *convw, float *ring, float *h,
                              float *gated, int dim, int K, int P, int N) {
  const int block = 256;
  const int grid = nsos::gpu::ceil_div_positive(dim, block);
  mamba_nstate_step_kernel<<<grid, block, 0, nsos::gpu::current_stream()>>>(xv, z, B_in, C_in, dt, A, convw,
                                            ring, h, gated, dim, K, P, N);
}

}  // namespace cuda
}  // namespace nsos
