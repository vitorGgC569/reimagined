#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace nsos {

constexpr int WARP_SIZE = 32;
constexpr int TILE_M = 64;
constexpr int TILE_OUT = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

/**
 * Kernel CUDA para backward do RBF - Two-Stage Reduction SOTA
 */
__global__ void
kan_rbf_backward_kernel(const float *grad_output, const float *x_expanded,
                        const float *rbf_weight, const float *x_input,
                        const float *rbf_centers, const float *rbf_sigma,
                        float *grad_input, float *grad_rbf_weight,
                        float *grad_centers, float *grad_sigma, int M,
                        int in_features, int out_features, int grid_size) {

  int tile_m = blockIdx.x * TILE_M + threadIdx.y;
  int tile_out = blockIdx.y * TILE_OUT + threadIdx.x;

  __shared__ float shared_centers[20][32]; // Max grid_size = 20
  __shared__ float shared_sigma[32];

  if (threadIdx.x < grid_size && threadIdx.y < 32) {
    shared_centers[threadIdx.x][threadIdx.y] = 0.0f;
  }
  if (threadIdx.x == 0) {
    shared_sigma[threadIdx.y] = 0.0f;
  }
  __syncthreads();

  float sigma = rbf_sigma[0];
  float inv_sigma2 = 1.0f / (sigma * sigma + 1e-8f);
  float inv_sigma3 = inv_sigma2 / (sigma + 1e-8f);

  if (tile_m < M && tile_out < out_features) {
    float dy_val = grad_output[tile_m * out_features + tile_out];

    for (int j = 0; j < in_features; ++j) {
      float x_val = x_input[tile_m * in_features + j];

      for (int k = 0; k < grid_size; ++k) {
        int rbf_idx = tile_m * (in_features * grid_size) + (j * grid_size + k);
        int w_idx = tile_out * (in_features * grid_size) + (j * grid_size + k);

        float rbf_val = x_expanded[rbf_idx];
        float w = rbf_weight[w_idx];
        float c = rbf_centers[k];
        float diff = x_val - c;

        if (rbf_val > 1e-7f) {
          float d_rbf = rbf_val * (-diff * inv_sigma2);
          atomicAdd(&grad_input[tile_m * in_features + j], dy_val * w * d_rbf);
          atomicAdd(&grad_rbf_weight[w_idx], dy_val * rbf_val);

          float center_contrib = dy_val * w * rbf_val * diff * inv_sigma2;
          atomicAdd(&shared_centers[k][threadIdx.y], center_contrib);

          float sigma_contrib =
              dy_val * w * rbf_val * (diff * diff) * inv_sigma3;
          atomicAdd(&shared_sigma[threadIdx.y], sigma_contrib);
        }
      }
    }
  }

  __syncthreads();

  // Final reduction to global
  if (threadIdx.y == 0 && threadIdx.x < grid_size) {
    float sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += shared_centers[threadIdx.x][i];
    atomicAdd(&grad_centers[threadIdx.x], sum);
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += shared_sigma[i];
    atomicAdd(grad_sigma, sum);
  }
}

extern "C" void
launch_kan_rbf_backward(const float *grad_output, const float *x_expanded,
                        const float *rbf_weight, const float *x_input,
                        const float *rbf_centers, const float *rbf_sigma,
                        float *grad_input, float *grad_rbf_weight,
                        float *grad_centers, float *grad_sigma, int M,
                        int in_features, int out_features, int grid_size) {

  dim3 block(TILE_OUT, 32);
  dim3 grid((M + TILE_M - 1) / TILE_M,
            (out_features + TILE_OUT - 1) / TILE_OUT);

  kan_rbf_backward_kernel<<<grid, block>>>(
      grad_output, x_expanded, rbf_weight, x_input, rbf_centers, rbf_sigma,
      grad_input, grad_rbf_weight, grad_centers, grad_sigma, M, in_features,
      out_features, grid_size);
}

} // namespace nsos
