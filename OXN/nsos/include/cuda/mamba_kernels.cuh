#ifndef MAMBA_KERNELS_CUH
#define MAMBA_KERNELS_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <vector>

namespace nsos {
namespace cuda {

#ifdef USE_CUDA
// Mamba2 SSD Forward Kernel Launcher
// x: [B, L, H, P]
// dt: [B, L, H]
// A: [H]
// B_param: [B, L, H, N]
// C_param: [B, L, H, N]
// y: [B, L, H, P] (Output)
// initial_state: [B, H, P, N] (Optional, can be null)
void launch_mamba_ssd_forward(
    const float *x, const float *dt, const float *A, const float *B_param,
    const float *C_param, float *y,
    float *final_state, // [B, H, P, N] Output state for cache
    int Batch, int Seq,
    int n_heads, // H
    int d_head,  // P
    int d_state  // N
);

// Simplified NSOS scan used by the current Mamba2SSD implementation.
// x/dt/y are contiguous [B, Seq, D] buffers (or [1, Seq, D] for rank-2 input).
void launch_mamba_simple_scan_forward(const float *x, const float *dt,
                                      const float *A, float *y, int Batch,
                                      int Seq, int D);

void launch_mamba_simple_scan_backward(const float *grad_y, const float *y,
                                       const float *dt, const float *A,
                                       float *grad_x, float *grad_dt,
                                       float *grad_A, int Batch, int Seq,
                                       int D);

void launch_mamba_single_token_update(const float *x,
                                      const float *dt,
                                      const float *A,
                                      float *state,
                                      float *y,
                                      int Batch,
                                      int D);
#endif

} // namespace cuda
} // namespace nsos

#endif // MAMBA_KERNELS_CUH
