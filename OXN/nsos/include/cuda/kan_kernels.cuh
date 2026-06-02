#ifndef KAN_KERNELS_CUH
#define KAN_KERNELS_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {
namespace cuda {

#ifdef USE_CUDA
// =====================================================================
// KAN (Kolmogorov-Arnold) radial-basis evaluation — GPU equivalents of
// the host loops in src/kan.cpp.  These match the CPU math 1:1 so the
// parity test (tests/gpu/test_gpu_parity_kan.cpp) holds.
//
// Grid layout: centers/widths are [grid] device arrays.  The kernel
// applies the same safe_width(w) = max(w, 1e-3) clamp the CPU path uses.
// All pointers are device pointers; launchers do NOT sync.
// =====================================================================

// Forward: basis[row, feature*grid + g] = exp(-0.5 * ((x - center_g)/width_g)^2)
//   x     : [rows, input_dim]
//   basis : [rows, input_dim*grid]   (output)
void launch_kan_rbf_basis_forward(const float* x, const float* centers,
                                  const float* widths, float* basis, int rows,
                                  int input_dim, int grid);

// Backward: grad_input[row,feature] += sum_g grad_basis[...]*d(basis_g)/d(x).
// grad_input is ACCUMULATED into (it already holds the base-weight term), so it
// must be a valid device buffer the caller has filled.
//   x          : [rows, input_dim]
//   grad_basis : [rows, input_dim*grid]
//   grad_input : [rows, input_dim]    (in/out, accumulated)
void launch_kan_rbf_basis_backward(const float* x, const float* grad_basis,
                                   const float* centers, const float* widths,
                                   float* grad_input, int rows, int input_dim,
                                   int grid);
#endif  // USE_CUDA

}  // namespace cuda
}  // namespace nsos

#endif  // KAN_KERNELS_CUH
