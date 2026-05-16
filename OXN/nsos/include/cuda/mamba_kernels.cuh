#ifndef MAMBA_KERNELS_CUH
#define MAMBA_KERNELS_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstddef>

namespace nsos {
namespace cuda {

#ifdef USE_CUDA
// =====================================================================
// Mamba2 SSD chunked forward kernel (heads × heads_dim × state_dim).
//
// Tensors (all device pointers):
//   x:        [B, Seq, H, P]
//   dt:       [B, Seq, H]
//   A:        [H]
//   B_param:  [B, Seq, H, N]
//   C_param:  [B, Seq, H, N]
//   y:        [B, Seq, H, P]   (output)
//   final_state: [B, H, P, N]  (output / inter-chunk workspace)
//
// IMPORTANT: This launcher synchronizes the device internally because
// the multi-chunk path issues sequential kernels that must serialize on
// the inter-chunk state.  Callers do NOT need to call
// cudaDeviceSynchronize() afterwards, but doing so is harmless.
// =====================================================================
void launch_mamba_ssd_forward(
    const float *x, const float *dt, const float *A, const float *B_param,
    const float *C_param, float *y, float *final_state, int Batch, int Seq,
    int n_heads, int d_head, int d_state);

// =====================================================================
// Selective scan with B/C gating, matched 1:1 against the CPU
// implementation in src/mamba2.cpp::Mamba2SSD::ssd_forward.
//
// Recurrence (per (batch, dim) channel, sequential over time):
//   decay     = exp(-softplus(dt[t]) * max(A[d], 1e-3))
//   state[d]  = state[d] * decay + B_in[t,d] * x[t,d]
//   y[t,d]    = tanh(state[d]) * C_in[t,d]
//
// Tensors (all device pointers):
//   x, dt, B_in, C_in, y :  [Batch, Seq, D]
//   A                    :  [D]
//   state_history (opt)  :  [Batch, Seq, D]   nullptr to skip recording
//
// state_history records the SSM state AFTER the update at each timestep
// so that the matching backward kernel can recompute gradients without
// re-running the forward pass.  Pass nullptr for inference paths that
// do not need backward.
//
// This launcher does NOT call cudaDeviceSynchronize().  Callers that
// must serialize with the host should sync explicitly (e.g. before
// reading the output on the CPU).
// =====================================================================
void launch_mamba_selective_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D);

// =====================================================================
// Selective scan backward, paired with the forward kernel above.
//
// Inputs (all device pointers):
//   grad_y         : [Batch, Seq, D]   upstream gradient
//   x, dt, B_in, C_in : [Batch, Seq, D]   forward inputs
//   A              : [D]                  state-decay weights
//   state_history  : [Batch, Seq, D]      from forward(state_history != null)
//
// Outputs (all device pointers, must be pre-allocated and pre-zeroed
// where appropriate — grad_A is accumulated with atomicAdd so it MUST
// be cudaMemset to 0 before this launcher is called):
//   grad_x   : [Batch, Seq, D]
//   grad_dt  : [Batch, Seq, D]
//   grad_A   : [D]
//   grad_B   : [Batch, Seq, D]
//   grad_C   : [Batch, Seq, D]
//
// Same no-sync contract as the forward launcher.
// =====================================================================
void launch_mamba_selective_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D);

// =====================================================================
// Simplified scan used by some legacy tests.  No B/C gating — kept for
// API stability but new code should prefer launch_mamba_selective_scan_*.
// =====================================================================
void launch_mamba_simple_scan_forward(const float *x, const float *dt,
                                      const float *A, float *y, int Batch,
                                      int Seq, int D);

void launch_mamba_simple_scan_backward(const float *grad_y, const float *y,
                                       const float *dt, const float *A,
                                       float *grad_x, float *grad_dt,
                                       float *grad_A, int Batch, int Seq,
                                       int D);

void launch_mamba_single_token_update(const float *x, const float *dt,
                                      const float *A, float *state, float *y,
                                      int Batch, int D);
#endif

}  // namespace cuda
}  // namespace nsos

#endif  // MAMBA_KERNELS_CUH
