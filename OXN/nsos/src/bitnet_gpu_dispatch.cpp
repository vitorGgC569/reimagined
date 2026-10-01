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
#include "gpu_execution.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include "cuda/kernels.cuh"
#include "../include/gpu_backend.h"
#endif

namespace nsos {

namespace {
void require_precision_bits(int precision_bits, const char* operation) {
  if (precision_bits < 2 || precision_bits > 8) {
    throw std::invalid_argument(std::string(operation) +
        ": precision_bits must be in the closed interval [2, 8]");
  }
}
}  // namespace

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
    throw std::runtime_error(
        std::string(NSOS_GPU_BACKEND_NAME) + " error after " + phase + ": " +
        cudaGetErrorString(err));
  }
}

}  // namespace

Tensor bitnet_gemm_158bit_gpu(const Tensor& x_gpu,
                              const Tensor& packed_weights_gpu,
                              float weight_scale, int M, int K, int N,
                              int precision_bits, const Tensor* magnitude,
                              const Tensor* bias) {
  require_gpu(x_gpu, "x_gpu");
  require_gpu(packed_weights_gpu, "packed_weights_gpu");

  if (M <= 0 || K <= 0 || N <= 0) {
    throw std::runtime_error(
        "bitnet_gemm_158bit_gpu: M, K, N must all be > 0 (got M=" +
        std::to_string(M) + " K=" + std::to_string(K) +
        " N=" + std::to_string(N) + ")");
  }
  require_precision_bits(precision_bits, "bitnet_gemm_158bit_gpu");
  if (K > INT_MAX / 254)
    throw std::overflow_error("BitNet integer accumulation would overflow");
  for (const Tensor* affine : {magnitude, bias}) {
    if (affine && (affine->get_device() != Device::GPU || affine->size != N)) {
      throw std::invalid_argument("bitnet GPU affine must be a GPU vector of N elements");
    }
  }
  if (!std::isfinite(weight_scale) || weight_scale <= 0.0f) {
    throw std::invalid_argument(
        "bitnet_gemm_158bit_gpu: weight_scale must be finite and positive");
  }
  if (K % 16 != 0) {
    throw std::runtime_error(
        "bitnet_gemm_158bit_gpu: K must be divisible by 16 for the "
        "row-aligned uint32 packed layout");
  }
  const int64_t input_elements =
      static_cast<int64_t>(M) * static_cast<int64_t>(K);
  if (x_gpu.size != input_elements) {
    throw std::runtime_error(
        "bitnet_gemm_158bit_gpu: x_gpu.size (" + std::to_string(x_gpu.size) +
        ") != M*K (" + std::to_string(input_elements) + ")");
  }
  const int64_t packed_words =
      static_cast<int64_t>(N) * (K / 16);
  if (packed_weights_gpu.size != packed_words) {
    throw std::runtime_error(
        "bitnet_gemm_158bit_gpu: packed_weights_gpu has the wrong size (has " +
        std::to_string(packed_weights_gpu.size) + " words, needs " +
        std::to_string(packed_words) + ")");
  }
  const int64_t output_elements =
      static_cast<int64_t>(M) * static_cast<int64_t>(N);
  if (output_elements > INT_MAX) {
    throw std::overflow_error(
        "bitnet_gemm_158bit_gpu: output element count exceeds INT_MAX");
  }

  auto& workspace = gpu::current_execution_context();
  auto* x_q_ptr = static_cast<int8_t*>(workspace.reserve(
      gpu::WorkspaceSlot::BitnetActivations, gpu::StorageType::Int8,
      static_cast<size_t>(input_elements)));
  auto* act_scales = static_cast<float*>(workspace.reserve(
      gpu::WorkspaceSlot::BitnetScales, gpu::StorageType::Float32, M));

  // raw_data() everywhere below: these tensors are passed directly to
  // CUDA kernels on the default stream.  No host-side access happens
  // between launches, so the auto-sync that data() would otherwise pay
  // is pure overhead.  Each kernel queues behind the previous one via
  // default-stream serialization, which is exactly the semantics we
  // need.
  launch_quantize_activations_bitnet_kernel(
      x_gpu.raw_data(), x_q_ptr, act_scales, M, K, precision_bits);
  check_cuda_or_throw("quantize_activations_bitnet");

  Tensor y = Tensor::uninitialized({M, N}, Device::GPU);

  // Pack as uint32_t pointer for the dp4a kernel.  packed_weights_gpu is
  // a float-typed Tensor whose underlying bytes were copied from a
  // std::vector<uint32_t> via from_blob/copy — see test for the
  // upload pattern.
  const uint32_t* w_ptr =
      reinterpret_cast<const uint32_t*>(packed_weights_gpu.raw_data());

  if (M == 1) {
    gpu::record_dispatch(gpu::DispatchPath::BitnetGemv);
    launch_bitnet_gemv_scaled(x_q_ptr, w_ptr, y.raw_data(), K, N, weight_scale,
                             act_scales,
                             magnitude ? magnitude->raw_data() : nullptr,
                             bias ? bias->raw_data() : nullptr);
    check_cuda_or_throw("bitnet_gemv_scaled");
    return y;
  }

  // The legacy launch_bitnet_gemm signature requires grid_x/grid_y/
  // block_dim from the caller.  For the dp4a kernel a 16x16 block is
  // a natural fit (matches TILE_DIM in cuda/kernels.cu).
  constexpr int kTile = 16;
  const int grid_x = nsos::gpu::ceil_div_positive(N, kTile);
  const int grid_y = nsos::gpu::ceil_div_positive(M, kTile);

  launch_bitnet_gemm(x_q_ptr, w_ptr, y.raw_data(), M, K, N, weight_scale,
                     grid_x, grid_y, kTile);
  check_cuda_or_throw("bitnet_gemm");
  gpu::record_dispatch(gpu::DispatchPath::BitnetGemm);

  // Fold per-row activation scales into the output.
  launch_bitnet_apply_act_scales_kernel(y.raw_data(), act_scales,
                                         M, N);
  check_cuda_or_throw("apply_act_scales");
  if (magnitude) y = y.mul(*magnitude);
  if (bias) y = y.add(*bias);

  return y;
}

#else  // USE_CUDA

Tensor bitnet_gemm_158bit_gpu(const Tensor& /*x_gpu*/,
                              const Tensor& /*packed_weights_gpu*/,
                              float /*weight_scale*/, int /*M*/, int /*K*/,
                              int /*N*/, int /*precision_bits*/,
                              const Tensor* /*magnitude*/, const Tensor* /*bias*/) {
  throw std::runtime_error(
      "bitnet_gemm_158bit_gpu: NSOS was built without GPU support");
}

#endif  // USE_CUDA

// =====================================================================
// K3: device-aware QAT fake-quant helpers (GPU kernel + CPU fallback).
// =====================================================================

Tensor qat_fake_quant_ternary(const Tensor& w, float scale) {
  if (!std::isfinite(scale) || scale <= 0.0f) {
    throw std::invalid_argument(
        "qat_fake_quant_ternary requires a finite positive scale");
  }
  Tensor out = Tensor::uninitialized(w.shape.dims, w.get_device());
  const int n = static_cast<int>(w.size);
  if (n <= 0) return out;
#ifdef USE_CUDA
  if (w.get_device() == Device::GPU) {
    launch_fake_quant_ternary_kernel(out.raw_data(), w.raw_data(), scale, n);
    check_cuda_or_throw("qat_fake_quant_ternary");
    return out;
  }
#endif
  const float* wp = w.data();
  float* op = out.data();
  const float inv = 1.0f / (scale + 1e-8f);
  for (int i = 0; i < n; ++i) {
    const float v = wp[i] * inv;
    const float t = (v > 0.5f) ? 1.0f : ((v < -0.5f) ? -1.0f : 0.0f);
    op[i] = t * scale;
  }
  return out;
}

Tensor qat_fake_quant_ternary_absmean(const Tensor& w, Tensor* scale_out) {
  Tensor out = Tensor::uninitialized(w.shape.dims, w.get_device());
  const int n = static_cast<int>(w.size);
  if (n <= 0) return out;
#ifdef USE_CUDA
  if (w.get_device() == Device::GPU) {
    Tensor abs_sum({1}, Device::GPU);
    Tensor scale = Tensor::uninitialized({1}, Device::GPU);
    launch_abs_sum_kernel(abs_sum.raw_data(), w.raw_data(), n);
    check_cuda_or_throw("qat_fake_quant_ternary_absmean_abs_sum");
    launch_fake_quant_ternary_absmean_kernel(out.raw_data(), scale.raw_data(),
                                             w.raw_data(), abs_sum.raw_data(),
                                             n);
    check_cuda_or_throw("qat_fake_quant_ternary_absmean");
    if (scale_out) {
      *scale_out = scale;
    }
    return out;
  }
#endif
  const float scale = tensor_abs_mean(w) + 1e-8f;
  if (scale_out) {
    Tensor scale_tensor({1}, w.get_device());
    scale_tensor.data()[0] = scale;
    *scale_out = scale_tensor;
  }
  return qat_fake_quant_ternary(w, scale);
}

Tensor qat_fake_quant_activations(const Tensor& x, int precision_bits) {
  require_precision_bits(precision_bits, "qat_fake_quant_activations");
  Tensor out = Tensor::uninitialized(x.shape.dims, x.get_device());
  const int K = x.shape.size() > 0 ? x.shape.back() : 0;
  const int M = (K > 0) ? static_cast<int>(x.size / K) : 0;
  if (M <= 0 || K <= 0) return out;
#ifdef USE_CUDA
  if (x.get_device() == Device::GPU) {
    launch_fake_quant_activations_kernel(out.raw_data(), x.raw_data(), M, K,
                                         precision_bits);
    check_cuda_or_throw("qat_fake_quant_activations");
    return out;
  }
#endif
  const float* xp = x.data();
  float* op = out.data();
  const float q_max = (precision_bits <= 2)
                          ? 1.0f
                          : (std::pow(2.0f, static_cast<float>(precision_bits - 1)) - 1.0f);
  for (int r = 0; r < M; ++r) {
    const float* row = xp + static_cast<std::size_t>(r) * K;
    float* orow = op + static_cast<std::size_t>(r) * K;
    float mx = 0.0f;
    for (int j = 0; j < K; ++j) mx = std::max(mx, std::fabs(row[j]));
    const float denom = mx + 1e-8f;
    const float scale = q_max / denom;
    const float inv_scale = denom / q_max;
    for (int j = 0; j < K; ++j) {
      float v = row[j] * scale;
      v = std::min(std::max(v, -q_max), q_max);
      orow[j] = std::round(v) * inv_scale;
    }
  }
  return out;
}

void qat_ste_clip_weight_grad(Tensor& dW, const Tensor& w, float scale) {
  const int n = static_cast<int>(dW.size);
  if (!std::isfinite(scale) || scale <= 0.0f) {
    throw std::invalid_argument(
        "qat_ste_clip_weight_grad requires a finite positive scale");
  }
  if (w.shape != dW.shape || w.get_device() != dW.get_device()) {
    throw std::invalid_argument(
        "qat_ste_clip_weight_grad requires identical shapes and devices");
  }
  if (n <= 0) return;
#ifdef USE_CUDA
  if (dW.get_device() == Device::GPU && w.get_device() == Device::GPU) {
    launch_ste_clip_weight_grad_kernel(dW.raw_data(), w.raw_data(), scale, n);
    check_cuda_or_throw("qat_ste_clip_weight_grad");
    return;
  }
#endif
  float* g = dW.data();
  const float* wp = w.data();
  const float inv = 1.0f / (scale + 1e-8f);
  for (int i = 0; i < n; ++i) {
    if (std::fabs(wp[i] * inv) > 1.0f) g[i] = 0.0f;
  }
}

void qat_ste_clip_weight_grad_device_scale(Tensor& dW, const Tensor& w,
                                           const Tensor& scale) {
  const int n = static_cast<int>(dW.size);
  if (n <= 0 || w.size != dW.size || scale.size <= 0) return;
#ifdef USE_CUDA
  if (dW.get_device() == Device::GPU && w.get_device() == Device::GPU &&
      scale.get_device() == Device::GPU) {
    launch_ste_clip_weight_grad_device_scale_kernel(
        dW.raw_data(), w.raw_data(), scale.raw_data(), n);
    check_cuda_or_throw("qat_ste_clip_weight_grad_device_scale");
    return;
  }
#endif
  const float* sp = scale.data();
  qat_ste_clip_weight_grad(dW, w, sp[0]);
}

}  // namespace nsos
