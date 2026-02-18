#ifndef MAMBA_KERNELS_CUH
#define MAMBA_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector>

namespace nsos {
namespace cuda {

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

} // namespace cuda
} // namespace nsos

#endif // MAMBA_KERNELS_CUH
