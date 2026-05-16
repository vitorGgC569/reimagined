// =====================================================================
// BitNet 1.58-bit GPU dispatch implementation.  See
// include/bitnet_gpu_dispatch.h for the contract.
//
// Composes three CUDA primitives:
//   1. launch_quantize_activations_bitnet_kernel  (float → int8 + per-row scales)
//   2. launch_bitnet_gemm                         (int8 × packed → fp32 * weight_scale)
//   3. launch_bitnet_apply_act_scales_kernel      (in-place per-row activation scaling)
//
// CPU parity reference:
//   BitLinear::quantize_activations_bitnet → BitNetAdapter::gemm_158bit_i8
// (src/bitlinear.cpp, src/bitnet_adapter*.cpp).
// =====================================================================

#include "bitnet_gpu_dispatch.h"

#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include "cuda/kernels.cuh"
#include <cuda_runtime.h>
#endif

namespace nsos {

#ifdef USE_CUDA

namespace {

// Throws if `t` is not on the GPU; named arg supplies the offending
// parameter name for the error message.
void require_gpu(const Tensor& t, const char* name) {
  if (t.get_device() != Device::GPU) {
    throw std::runtime_error(std::string("bitnet_gemm_158bit_gpu: ") + name +
                             " must be on Device::GPU");
  }
}

void check_cuda_or_throw(const char* phase) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error after ") + phase + ": " +
                             cudaGetErrorString(err));
  }
}

}  // namespace

Tensor bitnet_gemm_158bit_gpu(const Tensor& x_gpu,
                              const Tensor& packed_weights_gpu,
                              float weight_scale, int M, int K, int N,
                              int precision_bits) {
  require_gpu(x_gpu, "x_gpu");
  require_gpu(packed_weights_gpu, "packed_weights_gpu");

  if (M <= 0 || K <= 0 || N <= 0) {
    throw std::runtime_error(
        "bitnet_gemm_158bit_gpu: M, K, N must all be > 0 (got M=" +
        std::to_string(M) + " K=" + std::to_string(K) +
        " N=" + std::to_string(N) + ")");
  }
  if (x_gpu.size != M * K) {
    throw std::runtime_error(
        "bitnet_gemm_158bit_gpu: x_gpu.size (" + std::to_string(x_gpu.size) +
        ") != M*K (" + std::to_string(M * K) + ")");
  }

  // Allocate intermediate device buffers.  We use Tensor for x_q despite
  // the underlying type being int8 — Tensor stores raw bytes and only
  // .data() is used here, so the float-typed shape is purely a sizing
  // hint.  We size by ceil(M*K / 4) floats so the byte count covers the
  // int8 buffer.
  const int int8_words =
      (M * K + static_cast<int>(sizeof(float)) - 1) /
      static_cast<int>(sizeof(float));
  Tensor x_q_storage({int8_words}, Device::GPU);
  Tensor act_scales({M}, Device::GPU);

  // raw_data() everywhere below: these tensors are passed directly to
  // CUDA kernels on the default stream.  No host-side access happens
  // between launches, so the auto-sync that data() would otherwise pay
  // is pure overhead.  Each kernel queues behind the previous one via
  // default-stream serialization, which is exactly the semantics we
  // need.
  int8_t* x_q_ptr = reinterpret_cast<int8_t*>(x_q_storage.raw_data());

  launch_quantize_activations_bitnet_kernel(
      x_gpu.raw_data(), x_q_ptr, act_scales.raw_data(), M, K, precision_bits);
  check_cuda_or_throw("quantize_activations_bitnet");

  Tensor y({M, N}, Device::GPU);

  // Pack as uint32_t pointer for the dp4a kernel.  packed_weights_gpu is
  // a float-typed Tensor whose underlying bytes were copied from a
  // std::vector<uint32_t> via from_blob/copy — see test for the
  // upload pattern.
  const uint32_t* w_ptr =
      reinterpret_cast<const uint32_t*>(packed_weights_gpu.raw_data());

  // The legacy launch_bitnet_gemm signature requires grid_x/grid_y/
  // block_dim from the caller.  For the dp4a kernel a 16x16 block is
  // a natural fit (matches TILE_DIM in cuda/kernels.cu).
  constexpr int kTile = 16;
  const int grid_x = (N + kTile - 1) / kTile;
  const int grid_y = (M + kTile - 1) / kTile;

  launch_bitnet_gemm(x_q_ptr, w_ptr, y.raw_data(), M, K, N, weight_scale,
                     grid_x, grid_y, kTile);
  check_cuda_or_throw("bitnet_gemm");

  // Fold per-row activation scales into the output.
  launch_bitnet_apply_act_scales_kernel(y.raw_data(), act_scales.raw_data(),
                                         M, N);
  check_cuda_or_throw("apply_act_scales");

  return y;
}

#else  // USE_CUDA

Tensor bitnet_gemm_158bit_gpu(const Tensor& /*x_gpu*/,
                              const Tensor& /*packed_weights_gpu*/,
                              float /*weight_scale*/, int /*M*/, int /*K*/,
                              int /*N*/, int /*precision_bits*/) {
  throw std::runtime_error(
      "bitnet_gemm_158bit_gpu: NSOS was built without CUDA support");
}

#endif  // USE_CUDA

}  // namespace nsos
