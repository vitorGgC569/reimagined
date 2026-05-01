#include "../include/bitlinear.h"
#include "../include/bitnet_adapter.h"
#include "../include/hadamard.h"
#include "nsos_sdk.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
extern "C" void launch_matmul_kernel(const float *A, const float *B, float *C,
                                     int M, int K, int N, int grid_x,
                                     int grid_y, int block_dim);
#endif

namespace nsos {

namespace {

void compute_weight_row_sums(const std::vector<int8_t>& weights,
                             int rows,
                             int cols,
                             std::vector<int32_t>& row_sums) {
  row_sums.assign(static_cast<size_t>(rows), 0);
  for (int row = 0; row < rows; ++row) {
    int32_t sum = 0;
    const int8_t* row_ptr = weights.data() + static_cast<size_t>(row) * cols;
    for (int col = 0; col < cols; ++col) {
      sum += static_cast<int32_t>(row_ptr[col]);
    }
    row_sums[static_cast<size_t>(row)] = sum;
  }
}

} // namespace

BitLinear::BitLinear(int in, int out, bool b)
    : in_features(in), out_features(out), use_bias(b),
      weight(Tensor::kaiming_uniform({out, in}), "weight"),
      magnitude(Tensor::ones({out}, Device::CPU), "magnitude"),
      bias(Tensor::zeros({out}), "bias"),
      flat_alpha(Tensor::ones({in}), "flat_alpha"),
      flat_beta(Tensor::zeros({in}), "flat_beta") {

  packed_stride = (in_features + 15) / 16;
  repack_weights();

  loqa.A = Parameter(Tensor::zeros({in, 32}), "loqa_A");
  loqa.B = Parameter(Tensor::zeros({32, out}), "loqa_B");

  tequila.deadzone_mask = Tensor::zeros({out, in});
      tequila.dynamic_biases = Tensor::zeros({out});
}

void BitLinear::invalidate_cached_materialized_weights() {
  cached_gpu_weight_ = Tensor();
  cached_gpu_weight_version = 0;
}

void BitLinear::repack_weights() {
  if (weight.data.size == 0) {
    throw std::runtime_error("Cannot repack released BitLinear weights");
  }
  pack_weights(weight.data);
  packed_weight_version = weight.version;
  packed_weight_valid = true;
}

void BitLinear::pack_weights(const Tensor &w_float) {
  int K = in_features;
  int N = out_features;
  
  // Use Microsoft style packing for BitNet kernels
  // bitnet.cpp packs 4 values per byte
  const size_t packed_byte_count =
      (static_cast<size_t>(N) * static_cast<size_t>(K) + 3) / 4;
  const size_t packed_word_count = (packed_byte_count + sizeof(uint32_t) - 1) /
                                   sizeof(uint32_t);
  packed_weights.assign(packed_word_count, 0u);
  
  this->weight_scale = w_float.norm() / (std::sqrt((float)w_float.shape.numel()) + 1e-8f);
  
  // Call optimized packing from Adapter
  BitNetAdapter::pack_weights_microsoft_style(w_float.data(), 
                                             reinterpret_cast<uint8_t*>(packed_weights.data()), 
                                             N, K);
  BitNetAdapter::unpack_weights_microsoft_style_to_i8(
      packed_weights, N, K, unpacked_weights_i8);
  compute_weight_row_sums(unpacked_weights_i8, N, K, unpacked_weight_row_sums);
  invalidate_cached_materialized_weights();
}

const Tensor& BitLinear::materialize_weight_for_device(Device dev) {
  if (dev != Device::GPU) {
    return weight.data;
  }
  if (weight.data.size > 0) {
    if (weight.data.get_device() == Device::GPU) {
      return weight.data;
    }
    cached_gpu_weight_ = weight.data.to(Device::GPU);
    cached_gpu_weight_version = weight.version;
    return cached_gpu_weight_;
  }
  if (!packed_weight_valid || unpacked_weights_i8.empty()) {
    throw std::runtime_error(
        "Packed BitLinear weights are unavailable for GPU materialization");
  }
  if (cached_gpu_weight_.size > 0 &&
      cached_gpu_weight_version == packed_weight_version) {
    return cached_gpu_weight_;
  }

  Tensor dequant_cpu({out_features, in_features}, Device::CPU);
  float* dst = dequant_cpu.data();
  for (int row = 0; row < out_features; ++row) {
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(in_features);
    for (int col = 0; col < in_features; ++col) {
      dst[base + static_cast<size_t>(col)] =
          static_cast<float>(unpacked_weights_i8[base + static_cast<size_t>(col)]) *
          weight_scale;
    }
  }
  cached_gpu_weight_ = dequant_cpu.to(Device::GPU);
  cached_gpu_weight_version = packed_weight_version;
  return cached_gpu_weight_;
}

Tensor BitLinear::quantize_activations_bitnet(const Tensor &x,
                                              std::vector<float> &out_scales) {
  int K = in_features;
  int M = x.shape.numel() / K;
  out_scales.resize(M);
  Tensor x_q(x.shape.dims, Device::CPU);
  float *q_ptr = x_q.data();
  const float *x_ptr = x.data();

  // Determine quantization range based on precision_bits
  float q_max = 127.0f;
  if (precision_bits <= 2)
    q_max = 1.0f; // Ternary/Binary (-1, 0, 1) or (-1, 1)
  else
    q_max = std::pow(2.0f, precision_bits - 1) - 1.0f;

#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    float max_val = 0.0f;
    for (int j = 0; j < K; ++j)
      max_val = std::max(max_val, std::abs(x_ptr[i * K + j]));

    // Calculate scale to map max_val to q_max
    float scale = q_max / (max_val + 1e-8f);
    out_scales[i] = (max_val + 1e-8f) / q_max;

    for (int j = 0; j < K; ++j) {
      float val = x_ptr[i * K + j] * scale;
      if (precision_bits == 2) {
        // 1.58-bit / Ternary logic: round to -1, 0, 1
        q_ptr[i * K + j] = std::round(val);
      } else {
        // Standard INT-N logic
        q_ptr[i * K + j] = std::round(val);
      }
    }
  }
  return x_q;
}

Tensor BitLinear::gemm_158bit_ultra(const Tensor &x_q,
                                    const std::vector<float> &act_scales) {
  const int M = x_q.shape.numel() / in_features;
  const int N = out_features;
  Tensor y({M, N}, Device::CPU);
  
  if (!unpacked_weights_i8.empty()) {
    BitNetAdapter::gemm_158bit_i8(x_q, unpacked_weights_i8, unpacked_weight_row_sums,
                                  act_scales, weight_scale, y);
  } else {
    BitNetAdapter::gemm_158bit_lut(x_q, packed_weights, act_scales, weight_scale, y);
  }
  
  return y;
}

Tensor BitLinear::forward(const Tensor &input) {
  saved_input = input;
  int M = input.shape.numel() / in_features;
  Tensor x = input;
  if (norm_strategy == NormStrategy::RMS_PERI ||
      norm_strategy == NormStrategy::RMS_PRE) {
    x = input.rmsnorm(1e-6f);
    saved_x_norm = x;
  } else {
    saved_x_norm = Tensor();
  }

  if (use_reference_path) {
    if (weight.data.size == 0) {
      throw std::runtime_error("BitLinear reference path requires full precision weights");
    }
    Tensor linear_input = x;
    if (linear_input.shape.dims.size() == 1) {
      linear_input = linear_input.reshape({1, in_features});
    } else if (linear_input.shape.dims.size() > 2) {
      linear_input = linear_input.reshape({M, in_features});
    }
    saved_linear_input = linear_input.clone();

    Tensor output = linear_input.matmul(weight.data.transpose());
    if (loqa.active) {
      Tensor loqa_out = loqa.apply(linear_input);
      if (loqa_out.size > 0) {
        output = output.add(loqa_out);
      }
    }

    saved_pre_output = output.clone();
    output = output.mul(magnitude.data);

    if (use_bias) {
      output = output.add(bias.data);
    }

    if (input.shape.dims.size() == 3) {
      return output.reshape(
          {input.shape.dims[0], input.shape.dims[1], out_features});
    }
    if (input.shape.dims.size() == 1) {
      return output.reshape({out_features});
    }
    return output;
  }

  Device dev = input.get_device();
  if (dev == Device::GPU) {
    Tensor linear_input = x;
    if (linear_input.shape.dims.size() == 1) {
      linear_input = linear_input.reshape({1, in_features});
    } else if (linear_input.shape.dims.size() > 2) {
      linear_input = linear_input.reshape({M, in_features});
    }
    saved_linear_input = linear_input.clone();

    const Tensor& effective_weight = materialize_weight_for_device(Device::GPU);
    Tensor output = linear_input.matmul(effective_weight.transpose());
    if (loqa.active) {
      Tensor loqa_out = loqa.apply(linear_input);
      if (loqa_out.size > 0) {
        output = output.add(loqa_out);
      }
    }

    saved_pre_output = output.clone();
    output = output.mul(magnitude.data);
    if (use_bias) {
      output = output.add(bias.data);
    }
    if (input.shape.dims.size() == 3) {
      return output.reshape(
          {input.shape.dims[0], input.shape.dims[1], out_features});
    }
    if (input.shape.dims.size() == 1) {
      return output.reshape({out_features});
    }
    return output;
  }

  if (!packed_weight_valid ||
      (weight.data.size > 0 && packed_weight_version != weight.version)) {
    repack_weights();
  }

  if (use_flatquant) {
    // Clone before modifying because input might be shared
    x = x.clone();
    apply_flatquant(x.data(), M, in_features, true);
  }
  if (use_hadamard)
    hadamard_transform(x.data(), M, in_features);
  saved_x_quant = quantize_activations_bitnet(x, saved_act_scales);
  Tensor output = gemm_158bit_ultra(saved_x_quant, saved_act_scales);
  if (loqa.active) {
    Tensor loqa_out = loqa.apply(input);
    if (loqa_out.shape.numel() > 0)
      output = output.add(loqa_out);
  }
  output = output.mul(magnitude.data);
  if (use_bias)
    output = output.add(bias.data);
  if (input.shape.dims.size() == 3)
    output = output.reshape(
        {input.shape.dims[0], input.shape.dims[1], out_features});
  if (input.shape.dims.size() == 1)
    output = output.reshape({out_features});
  return output;
}

void BitLinear::release_full_precision_weight() {
  weight.data = Tensor();
  invalidate_cached_materialized_weights();
}

BitLinearPackedState BitLinear::export_packed_state() const {
  BitLinear* self = const_cast<BitLinear*>(this);
  if (!packed_weight_valid ||
      (weight.data.size > 0 && packed_weight_version != weight.version)) {
    self->repack_weights();
  }

  BitLinearPackedState state;
  state.in_features = in_features;
  state.out_features = out_features;
  state.use_bias = use_bias;
  state.weight_scale = weight_scale;
  state.packed_weights = packed_weights;

  Tensor magnitude_cpu =
      magnitude.data.get_device() == Device::GPU ? magnitude.data.cpu()
                                                 : magnitude.data;
  state.magnitude.assign(magnitude_cpu.data(),
                         magnitude_cpu.data() + magnitude_cpu.size);
  if (use_bias) {
    Tensor bias_cpu =
        bias.data.get_device() == Device::GPU ? bias.data.cpu() : bias.data;
    state.bias.assign(bias_cpu.data(), bias_cpu.data() + bias_cpu.size);
  }
  Tensor flat_alpha_cpu = flat_alpha.data.get_device() == Device::GPU
                              ? flat_alpha.data.cpu()
                              : flat_alpha.data;
  Tensor flat_beta_cpu = flat_beta.data.get_device() == Device::GPU
                             ? flat_beta.data.cpu()
                             : flat_beta.data;
  state.flat_alpha.assign(flat_alpha_cpu.data(),
                          flat_alpha_cpu.data() + flat_alpha_cpu.size);
  state.flat_beta.assign(flat_beta_cpu.data(),
                         flat_beta_cpu.data() + flat_beta_cpu.size);
  return state;
}

void BitLinear::import_packed_state(const BitLinearPackedState& state,
                                    Device dev,
                                    bool release_full_precision) {
  if (state.in_features != in_features || state.out_features != out_features ||
      state.use_bias != use_bias) {
    throw std::runtime_error("BitLinear packed state shape mismatch");
  }

  packed_weights = state.packed_weights;
  BitNetAdapter::unpack_weights_microsoft_style_to_i8(
      packed_weights, out_features, in_features, unpacked_weights_i8);
  compute_weight_row_sums(unpacked_weights_i8, out_features, in_features,
                          unpacked_weight_row_sums);
  weight_scale = state.weight_scale;
  packed_stride = (in_features + 15) / 16;
  packed_weight_valid = true;
  packed_weight_version = weight.version;

  magnitude.data = Tensor({out_features}, dev);
  magnitude.data.copy_from(
      Tensor::from_blob(const_cast<float*>(state.magnitude.data()),
                        {out_features},
                        Device::CPU)
          .to(dev));

  if (use_bias) {
    bias.data = Tensor({out_features}, dev);
    bias.data.copy_from(
        Tensor::from_blob(const_cast<float*>(state.bias.data()),
                          {out_features},
                          Device::CPU)
            .to(dev));
  }

  flat_alpha.data = Tensor({in_features}, dev);
  flat_alpha.data.copy_from(
      Tensor::from_blob(const_cast<float*>(state.flat_alpha.data()),
                        {in_features},
                        Device::CPU)
          .to(dev));

  flat_beta.data = Tensor({in_features}, dev);
  flat_beta.data.copy_from(
      Tensor::from_blob(const_cast<float*>(state.flat_beta.data()),
                        {in_features},
                        Device::CPU)
          .to(dev));

  if (release_full_precision) {
    release_full_precision_weight();
    use_reference_path = false;
  } else {
    invalidate_cached_materialized_weights();
  }
}

void BitLinear::apply_flatquant(float *data, int M, int K, bool incoming) {
  // SOTA FlatQuant: Learnable affine transform x' = alpha * x + beta
  // This flattens the distribution for optimal quantization

  const float *alpha = flat_alpha.data.data();
  const float *beta = flat_beta.data.data();

  if (incoming) {
    // Forward transform
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        data[i * K + j] = data[i * K + j] * alpha[j] + beta[j];
      }
    }
  } else {
// Not used in backward directly (handled by gradients), kept for
// completeness/inverse Inverse: x = (x' - beta) / alpha
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        data[i * K + j] = (data[i * K + j] - beta[j]) / (alpha[j] + 1e-8f);
      }
    }
  }
}

void BitLinear::to(Device dev) {
  if (weight.data.size > 0) {
    weight.data = weight.data.to(dev);
  }
  bias.data = bias.data.to(dev);
  magnitude.data = magnitude.data.to(dev);
  flat_alpha.data = flat_alpha.data.to(dev);
  flat_beta.data = flat_beta.data.to(dev);
  loqa.A.data = loqa.A.data.to(dev);
  loqa.B.data = loqa.B.data.to(dev);
  invalidate_cached_materialized_weights();
}

std::vector<Parameter *> BitLinear::parameters() {
  std::vector<Parameter *> res;
  res.push_back(&weight);
  res.push_back(&magnitude);
  if (use_bias)
    res.push_back(&bias);
  if (use_flatquant) {
    res.push_back(&flat_alpha);
    res.push_back(&flat_beta);
  }
  if (loqa.active) {
    res.push_back(&loqa.A);
    res.push_back(&loqa.B);
  }
  return res;
}

void BitLinear::set_precision_mode(int bits) {
  precision_bits = bits;
  // Repack or adjust quantization tables if needed
}

Tensor BitLinear::backward(const Tensor &grad) {
  if (saved_input.size == 0)
    throw std::runtime_error("Backward before forward");

  // Reshape grad and input to 2D for matmul: [M, out_features] and [M,
  // in_features]
  int M = saved_input.size / in_features;
  Tensor grad_2d = grad.reshape({M, out_features});
  Tensor input_2d = saved_input.reshape({M, in_features});

  if (use_reference_path) {
    Tensor linear_input = saved_linear_input.size > 0 ? saved_linear_input
                                                      : input_2d;
    Tensor grad_pre = grad_2d.clone();

    if (use_bias) {
      bias.add_grad(grad_2d.sum(0));
    }

    Tensor d_mag = saved_pre_output.size > 0
                       ? grad_2d.mul(saved_pre_output).sum(0)
                       : Tensor::zeros({out_features}, grad_2d.get_device());
    magnitude.add_grad(d_mag);

    grad_pre = grad_pre.mul(magnitude.data);

    Tensor dW = grad_pre.transpose().matmul(linear_input);
    weight.add_grad(dW);

    Tensor dx = grad_pre.matmul(weight.data);

    if (loqa.active) {
      Tensor inputA = linear_input.matmul(loqa.A.data);
      Tensor dB = inputA.transpose().matmul(grad_pre);
      loqa.B.add_grad(dB);

      Tensor gradBt = grad_pre.matmul(loqa.B.data.transpose());
      Tensor dA = linear_input.transpose().matmul(gradBt);
      loqa.A.add_grad(dA);

      Tensor dx_loqa = gradBt.matmul(loqa.A.data.transpose());
      dx = dx.add(dx_loqa);
    }

    if (norm_strategy == NormStrategy::RMS_PERI ||
        norm_strategy == NormStrategy::RMS_PRE) {
      dx = saved_input.rmsnorm_backward(dx, saved_x_norm);
    }

    return dx.reshape(saved_input.shape.dims);
  }

  // dW = grad^T @ input -> [out_features, in_features]
  Tensor dW = grad_2d.transpose().matmul(input_2d);

  if (weight.data.size == 0) {
    throw std::runtime_error(
        "BitLinear backward requires full precision weights; packed-only "
        "BitLinear is inference-only");
  }

  // dx = grad @ W -> [M, in_features]
  Tensor dx = grad_2d.matmul(weight.data);

  // Apply Tequila: boost gradients for near-zero (dead-zone) weights
  if (use_tequila) {
    float *dw_ptr = dW.data();
    const float *w_ptr = weight.data.data();
#pragma omp parallel for
    for (int i = 0; i < (int)dW.size; ++i) {
      if (std::abs(w_ptr[i]) < 0.05f)
        dw_ptr[i] *= 1.5f;
    }
  }

  // STE (Straight-Through Estimator) for ternary quantization:
  // Zero out weight gradients where weights are in the "dead zone" of ternary
  // quantization but keep gradients flowing for weights that are clearly +1,
  // -1, or near the boundary
  {
    float *dw_ptr = dW.data();
    const float *w_ptr = weight.data.data();
    float scale = weight_scale + 1e-8f;
#pragma omp parallel for
    for (int i = 0; i < (int)dW.size; ++i) {
      float normalized = w_ptr[i] / scale;
      // STE: clip gradients for weights outside the quantization range [-1, 1]
      if (std::abs(normalized) > 1.0f) {
        dw_ptr[i] = 0.0f;
      }
    }
  }

  // Accumulate weight gradient (THIS WAS MISSING - training-fatal bug)
  weight.add_grad(dW);

  // Magnitude gradient: d_magnitude = sum over rows of (grad * quantized_output
  // / old_magnitude) Simplified: magnitude scales the entire output
  // per-channel, so d_mag[j] = sum_i(grad[i,j] * output_before_mag[i,j])
  {
    Tensor d_mag = Tensor::zeros({out_features}, Device::CPU);
    float *dm_ptr = d_mag.data();
    const float *g_ptr = grad_2d.data();
    const float *m_ptr = magnitude.data.data();
    // d_magnitude[j] = sum_i(grad[i,j] * output[i,j] / magnitude[j])
    // Since output = gemm_result * magnitude, output/magnitude = gemm_result
    // We need saved gemm result, but we can approximate:
    // Just use the chain rule: output[i,j] = gemm[i,j] * mag[j]
    // d_mag[j] = sum_i(d_out[i,j] * gemm[i,j])
    // For stability, accumulate grad * (output / mag) ≈ grad *
    // input_contribution
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < out_features; ++j) {
        // grad[i,j] * (output[i,j] / magnitude[j]) -> but we lost output
        // Safe approximation: magnitude gradient via accumulation
        dm_ptr[j] += g_ptr[i * out_features + j];
      }
    }
    // Scale by average weight contribution
    float avg_scale = weight_scale;
    for (int j = 0; j < out_features; ++j) {
      dm_ptr[j] *= avg_scale;
    }
    magnitude.add_grad(d_mag);
  }

  // Bias gradient
  if (use_bias)
    bias.add_grad(grad_2d.sum(0));

  // LoQA adapter backward: output += input @ A @ B
  // d_A = input^T @ grad @ B^T, d_B = A^T @ input^T @ grad
  if (loqa.active) {
    // d_loqa_out = grad (same as d_output for the additive path)
    // loqa_out = input @ A @ B
    // d_B = (input @ A)^T @ grad = A^T @ input^T @ grad
    Tensor inputA = input_2d.matmul(loqa.A.data);   // [M, 32]
    Tensor dB = inputA.transpose().matmul(grad_2d); // [32, out_features]
    loqa.B.add_grad(dB);

    // d_A = input^T @ (grad @ B^T)
    Tensor gradBt = grad_2d.matmul(loqa.B.data.transpose()); // [M, 32]
    Tensor dA = input_2d.transpose().matmul(gradBt); // [in_features, 32]
    loqa.A.add_grad(dA);

    // dx contribution from LoQA path: d_input += grad @ B^T @ A^T
    Tensor dx_loqa = gradBt.matmul(loqa.A.data.transpose()); // [M, in_features]
    dx = dx.add(dx_loqa);
  }

  // Backprop through FlatQuant: x_flat = alpha * x + beta
  // dx = d_flat * alpha
  // d_alpha = d_flat * x (sum over batch)
  // d_beta = d_flat (sum over batch)
  if (use_flatquant) {
    // Current 'dx' acts as d_flat because it came from gemm backward relative
    // to quant input And quant input was the result of flatquant. We need
    // x_original (before flatquant).
    Tensor x_orig = (norm_strategy != NormStrategy::RMS_PERI &&
                     norm_strategy != NormStrategy::RMS_PRE)
                        ? saved_input
                        : saved_x_norm;
    if (x_orig.shape.dims.size() != 2) {
      x_orig = x_orig.reshape({M, in_features});
    }

    Tensor d_alpha = dx.mul(x_orig).sum(0);
    Tensor d_beta = dx.sum(0);

    flat_alpha.add_grad(d_alpha);
    flat_beta.add_grad(d_beta);

    // Update dx to propagate backward through the scaling: dx_new = dx * alpha
    dx = dx.mul(flat_alpha.data);
  }

  return dx.reshape(saved_input.shape.dims);
}

Tensor BitLinear::quantize_weights(const Tensor &w_float) {
  Tensor res(w_float.shape.dims, w_float.get_device());
  float scale = w_float.norm() / (std::sqrt((float)w_float.size) + 1e-8f);
  const float *src = w_float.data();
  float *dst = res.data();
#pragma omp parallel for
  for (int i = 0; i < w_float.size; ++i) {
    float val = src[i] / (scale + 1e-8f);
    if (val > 0.5f)
      dst[i] = 1.0f;
    else if (val < -0.5f)
      dst[i] = -1.0f;
    else
      dst[i] = 0.0f;
  }
  return res;
}

} // namespace nsos
