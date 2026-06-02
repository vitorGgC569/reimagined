#include "../../include/cuda/kan_kernels.cuh"

#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace nsos {
namespace cuda {

namespace {

constexpr int kKanThreads = 256;

// One thread per (row, feature); each loops over the (small) grid.  This
// matches the CPU layout in src/kan.cpp::compute_basis exactly.
__global__ void kan_rbf_basis_forward_kernel(const float* __restrict__ x,
                                             const float* __restrict__ centers,
                                             const float* __restrict__ widths,
                                             float* __restrict__ basis, int rows,
                                             int input_dim, int grid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * input_dim;
    if (idx >= total) return;

    const int feature = idx % input_dim;
    const int row = idx / input_dim;
    const float value = x[row * input_dim + feature];
    const int base = (row * input_dim + feature) * grid;

    for (int g = 0; g < grid; ++g) {
        const float w = fmaxf(widths[g], 1e-3f);  // safe_width
        const float diff = (value - centers[g]) / w;
        basis[base + g] = expf(-0.5f * diff * diff);
    }
}

// d(basis_g)/d(value) = basis_g * (center_g - value) / width_g^2.
// Accumulates the RBF contribution into grad_input (which already holds the
// base-weight gradient term), matching src/kan.cpp::backward.
__global__ void kan_rbf_basis_backward_kernel(const float* __restrict__ x,
                                              const float* __restrict__ grad_basis,
                                              const float* __restrict__ centers,
                                              const float* __restrict__ widths,
                                              float* __restrict__ grad_input,
                                              int rows, int input_dim, int grid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * input_dim;
    if (idx >= total) return;

    const int feature = idx % input_dim;
    const int row = idx / input_dim;
    const float value = x[row * input_dim + feature];
    const int base = (row * input_dim + feature) * grid;

    float rbf_dx = 0.0f;
    for (int g = 0; g < grid; ++g) {
        const float w = fmaxf(widths[g], 1e-3f);
        const float diff = (value - centers[g]) / w;
        const float basis = expf(-0.5f * diff * diff);
        const float deriv = basis * (centers[g] - value) / (w * w);
        rbf_dx += grad_basis[base + g] * deriv;
    }
    grad_input[row * input_dim + feature] += rbf_dx;
}

}  // namespace

void launch_kan_rbf_basis_forward(const float* x, const float* centers,
                                  const float* widths, float* basis, int rows,
                                  int input_dim, int grid) {
    const int total = rows * input_dim;
    if (total <= 0) return;
    const int blocks = (total + kKanThreads - 1) / kKanThreads;
    kan_rbf_basis_forward_kernel<<<blocks, kKanThreads>>>(
        x, centers, widths, basis, rows, input_dim, grid);
}

void launch_kan_rbf_basis_backward(const float* x, const float* grad_basis,
                                   const float* centers, const float* widths,
                                   float* grad_input, int rows, int input_dim,
                                   int grid) {
    const int total = rows * input_dim;
    if (total <= 0) return;
    const int blocks = (total + kKanThreads - 1) / kKanThreads;
    kan_rbf_basis_backward_kernel<<<blocks, kKanThreads>>>(
        x, grad_basis, centers, widths, grad_input, rows, input_dim, grid);
}

}  // namespace cuda
}  // namespace nsos

#endif  // USE_CUDA
