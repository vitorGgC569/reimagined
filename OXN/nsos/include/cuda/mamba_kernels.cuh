#ifndef MAMBA_KERNELS_CUH
#define MAMBA_KERNELS_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstddef>

namespace nsos {
namespace cuda {

#ifdef USE_CUDA
// (launch_mamba_ssd_forward removido — kernel chunked morto, zero callers.)

// =====================================================================
// Selective scan with B/C gating, matched 1:1 against the CPU
// implementation in src/mamba2.cpp::Mamba2SSD::ssd_forward.
//
// Recurrence (per (batch, dim) channel, sequential over time):
//   decay     = exp(-softplus(dt[t]) * exp(A_log[d]))            (N1)
//   state[d]  = state[d] * decay
//               + softplus(dt[t,d]) * B_in[t,d] * x[t,d]
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

// Runtime toggle for the OPT-IN parallel-prefix (associative) selective scan
// (default from NSOS_MAMBA_PARALLEL_SCAN env).  OFF -> the validated channel-
// parallel sequential kernel.  Exposed so the parity test / Python A/B can
// switch paths in a single process.  Must clear the 1e-4 parity gate before
// being promoted to default.
void set_mamba_parallel_scan(bool enabled);
bool mamba_parallel_scan_enabled();

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

// (launch_mamba_simple_scan_forward/backward removidos — mortos.)

void launch_mamba_single_token_update(const float *x, const float *dt,
                                      const float *A, float *state, float *y,
                                      int Batch, int D);

// =====================================================================
// Proper diagonal SSM (linear readout y = h*C, no tanh) — GPU-resident
// path for Mamba2SSD::forward_proper/backward_proper.  Same affine
// recurrence + state_history contract as the legacy selective scan.
// Tensors: x, dt, B_in, C_in, y : [Batch, Seq, D]; A : [D];
// state_history : [Batch, Seq, D] (h_t after each update; nullptr to skip).
// No internal cudaDeviceSynchronize.
// =====================================================================
void launch_mamba_proper_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D);

void launch_mamba_proper_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D);

// Fused single-token incremental decode step for the proper diagonal path.
// xv/z/B_in/C_in/dt/A are [dim]; convw is [dim*K]; ring is [(K-1)*dim] (in/out,
// the carried conv window); h is [dim] (in/out, the carried SSD state); gated is
// [dim] (out).  Device-resident carry -> no per-token host round-trip.
void launch_mamba_proper_step(const float *xv, const float *z,
                              const float *B_in, const float *C_in,
                              const float *dt, const float *A,
                              const float *convw, float *ring, float *h,
                              float *gated, int dim, int K);

// Fused single-token incremental decode step for the N-STATE path (full
// Mamba-2).  xv/z are [dim]; B_in/C_in are [H*N] (per-head N-dim); dt/A are
// [H] (per-head, A in log-domain); convw is [dim*K]; ring is [(K-1)*dim]
// (in/out); h is [dim*N] == [H*P*N] (in/out, the carried N-state); gated is
// [dim] (out).  Mirrors Mamba2SSD::forward_proper_nstate_step's host loop
// 1:1; device-resident carry -> no per-token host round-trip; CUDA-graph
// capturable.
void launch_mamba_nstate_step(const float *xv, const float *z,
                              const float *B_in, const float *C_in,
                              const float *dt, const float *A,
                              const float *convw, float *ring, float *h,
                              float *gated, int dim, int K, int P, int N);

// =====================================================================
// Causal depthwise conv1d (proper-path local mixing).
//   in/out : [Batch, Seq, D]   weight : [D, K]
//   out[b,t,c] = sum_{j} weight[c,j] * in[b, t-(K-1)+j, c]   (in[<0]=0)
// backward: grad_in and grad_weight accumulate via atomicAdd; the
// launcher zeroes both before the kernel.  No internal sync.
// =====================================================================
void launch_conv1d_causal_forward(const float *in, const float *weight,
                                  float *out, int batch, int seq, int dim,
                                  int K);
void launch_conv1d_causal_backward(const float *grad_out, const float *in,
                                   const float *weight, float *grad_in,
                                   float *grad_weight, int batch, int seq,
                                   int dim, int K);

// =====================================================================
// Full Mamba-2 SSD with N-dimensional state expansion (GPU-resident).
// One thread per (batch, head, p-channel), N-vector state in registers.
//   xc,y : [B,Seq,dim] (dim=H*P, channel=h*P+p)
//   dt   : [B,Seq,H]   A : [H]   B_in,C_in : [B,Seq,H,N]
//   state_history : [B,Seq,dim,N]  (h_t after update; nullptr to skip)
// N must be <= mamba_nstate_max_n(); the launcher no-ops otherwise so the
// caller falls back to the host scan.  Backward: gB/gC/gDt/gA accumulate via
// atomicAdd (the launcher zeroes them); gXc is written fully.  No internal sync.
// =====================================================================
int mamba_nstate_max_n();
void launch_mamba_nstate_forward(const float *xc, const float *dt,
                                 const float *A, const float *B_in,
                                 const float *C_in, float *y,
                                 float *state_history, int Batch, int Seq, int H,
                                 int P, int N);
void launch_mamba_nstate_backward(const float *gy, const float *xc,
                                  const float *dt, const float *A,
                                  const float *B_in, const float *C_in,
                                  const float *state_history, float *gXc,
                                  float *gDt, float *gA, float *gB, float *gC,
                                  int Batch, int Seq, int H, int P, int N);

// Faithful Mamba-2 SSD.  B/C are [B,Seq,G,N] and shared by the heads in
// each group; D is [H] and contributes D[h]*x inside the normalized SSM
// branch.  This matches state-spaces/mamba Mamba2 with D_has_hdim=false.
void launch_mamba2_faithful_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, const float *D, float *y, float *state_history,
    int Batch, int Seq, int H, int P, int N, int G);
void launch_mamba2_faithful_backward(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *D,
    const float *state_history, float *gX, float *gDt, float *gA,
    float *gB, float *gC, float *gD, int Batch, int Seq, int H, int P,
    int N, int G);

// Incremental faithful path split in two kernels so all x/B/C convolution
// channels are materialized before the SSD consumes shared B/C values.
void launch_mamba2_faithful_conv_step(
    const float *xv, const float *Bv, const float *Cv,
    const float *conv_weight, const float *conv_bias, float *ring,
    float *xBC, int inner, int group_state, int K);
void launch_mamba2_faithful_step(
    const float *xBC, const float *dt, const float *A, const float *D,
    float *state, float *y, int H, int P, int N, int G);
#endif

}  // namespace cuda
}  // namespace nsos

#endif  // MAMBA_KERNELS_CUH
