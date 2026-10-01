#ifndef KAN_KERNELS_CUH
#define KAN_KERNELS_CUH

#ifdef USE_CUDA
#include "../gpu_backend.h"
#endif

namespace nsos {
namespace cuda {

#ifdef USE_CUDA
// Ordered QAT: block-tree partials, fixed-order scale and quantization.
// Caller owns partials[min(ceil(elements/256),4096)] and scale[1].
bool launch_kan_prepare_ternary(const float* weight, float* effective,
    float* scale, float* partials, int elements);
// Implicit RBF GEMM. No global [rows,input_dim*grid] basis/gradient basis.
// WMMA opt-in selects RDNA3 wave32 only for lowp rows>=16 and both dims>=32.
// FP32/small shapes retain scalar; selected WMMA failures never fall back.
bool launch_kan_rbf_projection(const float* input, const float* weight,
    const float* centers, const float* widths, const float* base,
    const float* bias, float* output, int rows, int inputs, int outputs,
    int grid, int mode, bool wmma = false);
bool launch_kan_rbf_weight_backward(const float* input, const float* upstream,
    const float* centers, const float* widths, float* weight_gradient,
    int rows, int inputs, int outputs, int grid, int mode, bool wmma = false);
bool launch_kan_rbf_input_backward(const float* input, const float* upstream,
    const float* weight, const float* centers, const float* widths,
    float* input_gradient, int rows, int inputs, int outputs, int grid, int mode, bool wmma = false);
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
