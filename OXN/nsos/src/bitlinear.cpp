#include <cstdio>
#include <cstdlib>
#include "../include/bitlinear.h"
#include "../include/bitnet_adapter.h"
#include "../include/bitnet_gpu_dispatch.h"
#include "../include/hadamard.h"
#include "../include/lut_tmac.h"
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

#if defined(NSOS_ENABLE_AVX2_KERNELS)
// Forward decl implemented in src/bitlinear_quantize_avx2.cpp.
int32_t bitlinear_row_sum_i8_avx2(const int8_t* row_ptr, int cols);
#endif

namespace {

void compute_weight_row_sums(const std::vector<int8_t>& weights,
                             int rows,
                             int cols,
                             std::vector<int32_t>& row_sums) {
  row_sums.assign(static_cast<size_t>(rows), 0);
  // SIMD Gap #4 fix (2026-05-17): per-row int8 sum via AVX2 when
  // available.  Called at repack/load time, not on the per-token
  // forward path — so the wall-time impact is small but the
  // implementation is identical math (no quantization error) and
  // the kernel pattern is reused by the inference path for activation
  // statistics.  Scalar fallback below for non-x86 builds.
  for (int row = 0; row < rows; ++row) {
    const int8_t* row_ptr = weights.data() + static_cast<size_t>(row) * cols;
#if defined(NSOS_ENABLE_AVX2_KERNELS)
    row_sums[static_cast<size_t>(row)] = bitlinear_row_sum_i8_avx2(row_ptr, cols);
#else
    int32_t sum = 0;
    for (int col = 0; col < cols; ++col) {
      sum += static_cast<int32_t>(row_ptr[col]);
    }
    row_sums[static_cast<size_t>(row)] = sum;
#endif
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

// Seeded ctor — identical to the un-seeded version except for the
// kaiming_uniform call.  When seed=0, falls through to the existing
// un-seeded path for byte-exact backwards compatibility.
BitLinear::BitLinear(int in, int out, bool b, uint64_t seed)
    : in_features(in), out_features(out), use_bias(b),
      weight((seed == 0)
                ? Tensor::kaiming_uniform({out, in})
                : Tensor::kaiming_uniform({out, in}, Device::CPU, seed),
             "weight"),
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

  // Canonical NSOS ternary rule — single source of truth.  Quantize with
  // quantize_weights() (t = clamp(round(W / scale), -1, +1), scale =
  // ||W|| / sqrt(numel)) and pack the resulting {-1,0,+1} codes.  This makes
  // the packed weights identical to quantize_weights() and to the QAT
  // regularizer target, so quantized training (STE) and packed inference
  // share exactly one quantization rule.
  Tensor ternary_codes = quantize_weights(w_float);

  // Pack the {-1,0,+1} codes.  The adapter's +-0.25 raw threshold maps the
  // exact integer codes to their 2-bit values losslessly.
  BitNetAdapter::pack_weights_microsoft_style(ternary_codes.data(),
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

#if defined(NSOS_ENABLE_AVX2_KERNELS)
// Forward decls implemented in src/bitlinear_quantize_avx2.cpp.  Both
// require the file they live in to be compiled with /arch:AVX2 (MSVC)
// or -mavx2 (GCC/Clang); the CMake build registers that property.
float bitlinear_row_max_abs_avx2(const float* row_ptr, int K);
void  bitlinear_row_scale_round_avx2(float* dst, const float* src,
                                      int K, float scale);
#endif

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
    const float* row_ptr = x_ptr + static_cast<size_t>(i) * K;
    float* row_q_ptr = q_ptr + static_cast<size_t>(i) * K;

    // SIMD Gap #1 fix (2026-05-17): dispatch to AVX2 row kernels
    // when the build enabled them.  The math is bit-exact identical
    // to the scalar code: branch-free abs via sign-mask AND, max
    // reduction over 8 lanes, _mm256_round_ps with
    // _MM_FROUND_TO_NEAREST_INT matches std::round under IEEE 754
    // (round-half-to-even).  Scalar tails handle K not divisible
    // by 8.  Scalar fallback below stays for non-x86 builds and
    // for builds that explicitly disabled AVX2 kernels.
#if defined(NSOS_ENABLE_AVX2_KERNELS)
    const float max_val = bitlinear_row_max_abs_avx2(row_ptr, K);
    const float scale = q_max / (max_val + 1e-8f);
    out_scales[i] = (max_val + 1e-8f) / q_max;
    bitlinear_row_scale_round_avx2(row_q_ptr, row_ptr, K, scale);
#else
    float max_val = 0.0f;
    for (int j = 0; j < K; ++j)
      max_val = std::max(max_val, std::abs(row_ptr[j]));
    float scale = q_max / (max_val + 1e-8f);
    out_scales[i] = (max_val + 1e-8f) / q_max;
    for (int j = 0; j < K; ++j) {
      float val = row_ptr[j] * scale;
      if (precision_bits == 2) {
        row_q_ptr[j] = std::round(val);
      } else {
        row_q_ptr[j] = std::round(val);
      }
    }
#endif
  }
  return x_q;
}

Tensor BitLinear::gemm_158bit_ultra(const Tensor &x_q,
                                    const std::vector<float> &act_scales,
                                    bool fuse_output_affine) {
  const int M = x_q.shape.numel() / in_features;
  const int N = out_features;
  Tensor y({M, N}, Device::CPU);
  const float* magnitude_ptr = fuse_output_affine ? magnitude.data.data() : nullptr;
  const float* bias_ptr = fuse_output_affine && use_bias ? bias.data.data() : nullptr;

  // T-MAC opt-in path (LUT-style ternary GEMM with block-sparse skip).
  // See include/lut_tmac.h + docs/LUT_TMAC_DESIGN.md.  When the env var
  // NSOS_TMAC_LUT_GEMM=1 is set AND the weight shape is compatible
  // (in_features % 4 == 0), we route here.  Falls back to the existing
  // gemm_158bit_i8 / gemm_158bit_lut path otherwise.  Numerical
  // equivalence is verified by tests/test_lut_tmac.cpp.
  if (lut_tmac::env_opt_in() && lut_tmac::format_supported(in_features)) {
    if (cached_heat_map_.empty()) {
      cached_heat_map_ = lut_tmac::compute_heat_map(packed_weights, N, in_features);
    }
    lut_tmac::gemm_158bit_lut_tmac(
        x_q, packed_weights, cached_heat_map_, act_scales, weight_scale, y,
        magnitude_ptr, bias_ptr, fuse_output_affine && use_bias);
    return y;
  }

  if (!unpacked_weights_i8.empty()) {
    BitNetAdapter::gemm_158bit_i8(x_q, unpacked_weights_i8, unpacked_weight_row_sums,
                                  act_scales, weight_scale, y, magnitude_ptr, bias_ptr,
                                  fuse_output_affine && use_bias);
  } else {
    BitNetAdapter::gemm_158bit_lut(x_q, packed_weights, act_scales, weight_scale, y,
                                   magnitude_ptr, bias_ptr,
                                   fuse_output_affine && use_bias);
  }

  return y;
}

Tensor BitLinear::forward(const Tensor &input) {
  // (auditoria #10) NSOS_EDGE_DIAG=1: imprime UMA vez qual caminho o forward
  // de inferência CPU tomou (packed ternário vs float reference) — responde
  // em 1 linha o "por que 0.7 tok/s" sem profiler.
  static const bool edge_diag = [] {
    const char* e = std::getenv("NSOS_EDGE_DIAG");
    return e != nullptr && e[0] == '1';
  }();

  // Clone so the activation saved for backward (rmsnorm_backward, STE) can
  // never be corrupted by an in-place mutation of the caller's input buffer.
  saved_input = input.clone();
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
    if (edge_diag) {
      static bool printed_ref = false;
      if (!printed_ref) {
        printed_ref = true;
        std::fprintf(stderr, "[edge] BitLinear::forward caminho = REFERENCE (float matmul)\n");
      }
    }
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

#ifdef USE_CUDA
    // Phase 5b GPU __dp4a fast path.  Engaged only when:
    //   * `set_gpu_packed_inference(true)` was called (inference engines
    //     opt in after the model has been loaded and packed)
    //   * packed weights have been computed and are still current
    //   * LoQA adapter is not active (would require a separate
    //     dequantize-then-add path; falls back to float matmul for
    //     correctness when active)
    //   * input dimensions exercise the kernel within its supported
    //     range (M and N positive; K must be a multiple of 16 because
    //     of the 2-bit-per-weight packing scheme)
    //
    // Backward path is intentionally NOT modified: the float weight is
    // still available via `materialize_weight_for_device` for gradient
    // computation, so training gradients are byte-identical to the
    // pre-Phase-5b implementation.
    const bool dp4a_eligible =
        gpu_packed_inference_enabled_ && packed_weight_valid &&
        !loqa.active && M > 0 && in_features > 0 && out_features > 0 &&
        (in_features % 16 == 0);
    if (dp4a_eligible) {
      // Refresh GPU packed-weights cache when the underlying weights
      // have changed since the last upload.  The buffer is sized in
      // float-words because Tensor today only knows the float type;
      // bitnet_gemm_158bit_gpu reinterprets it as uint32_t* internally.
      const int packed_float_words =
          static_cast<int>(packed_weights.size());
      const bool cache_stale =
          cached_gpu_packed_weights_.size != packed_float_words ||
          cached_gpu_packed_version_ != packed_weight_version;
      if (cache_stale && packed_float_words > 0) {
        Tensor cpu_view({packed_float_words}, Device::CPU);
        std::memcpy(cpu_view.data(), packed_weights.data(),
                    static_cast<size_t>(packed_float_words) * sizeof(uint32_t));
        cached_gpu_packed_weights_ = cpu_view.to(Device::GPU);
        cached_gpu_packed_version_ = packed_weight_version;
      }

      Tensor y = bitnet_gemm_158bit_gpu(
          linear_input, cached_gpu_packed_weights_, weight_scale, M,
          in_features, out_features, precision_bits);

      saved_pre_output = y.clone();
      y = y.mul(magnitude.data);
      if (use_bias) {
        y = y.add(bias.data);
      }
      if (input.shape.dims.size() == 3) {
        return y.reshape(
            {input.shape.dims[0], input.shape.dims[1], out_features});
      }
      if (input.shape.dims.size() == 1) {
        return y.reshape({out_features});
      }
      return y;
    }
#endif

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

  if (edge_diag) {
    static bool printed_packed = false;
    if (!printed_packed) {
      printed_packed = true;
      std::fprintf(stderr, "[edge] BitLinear::forward caminho = PACKED ternario (1.58-bit)\n");
    }
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
  const bool fuse_output_affine = !loqa.active;
  Tensor output = gemm_158bit_ultra(saved_x_quant, saved_act_scales, fuse_output_affine);
  if (loqa.active) {
    Tensor loqa_out = loqa.apply(input);
    if (loqa_out.shape.numel() > 0)
      output = output.add(loqa_out);
  }
  if (!fuse_output_affine) {
    output = output.mul(magnitude.data);
    if (use_bias)
      output = output.add(bias.data);
  }
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

  // ── NON-REFERENCE PATH: straight-through estimator (STE) backward for the
  // packed ternary forward.  The real integer kernel ran in the forward; this
  // back-propagates onto the FP32 latent parameters.  The forward computed:
  //   x_norm = rmsnorm(input)                                   [saved_x_norm]
  //   xt     = flatquant(x_norm)             (if use_flatquant)
  //   xt     = hadamard(xt)                  (if use_hadamard)
  //   x_q    = round(xt / act_scale)         [saved_x_quant], act_scale [saved_act_scales]
  //   w_t    = ternary(W) in {-1,0,+1}       [unpacked_weights_i8], scale = weight_scale
  //   pre    = (x_q * act_scale) @ (w_t * weight_scale)^T   (+ loqa(input) if active)
  //   out    = pre * magnitude + bias
  // STE treats both quantizers as identity for gradient flow; gradients
  // accumulate into the latent FP32 weight / magnitude / bias / flat params.
  if (weight.data.size == 0) {
    throw std::runtime_error(
        "BitLinear STE backward requires latent full-precision weights; "
        "packed-only BitLinear is inference-only");
  }
  if (saved_x_quant.size == 0) {
    throw std::runtime_error(
        "BitLinear STE backward: missing saved quantized activations "
        "(forward did not run the packed path)");
  }

  // Reconstruct the exact dequantized activation the forward multiplied:
  //   act_dequant[i,k] = x_q[i,k] * act_scale[i]
  Tensor act_dequant({M, in_features}, Device::CPU);
  {
    float *a = act_dequant.data();
    const float *xq = saved_x_quant.data();
    for (int i = 0; i < M; ++i) {
      const float s = i < static_cast<int>(saved_act_scales.size())
                          ? saved_act_scales[static_cast<size_t>(i)]
                          : 1.0f;
      const size_t base = static_cast<size_t>(i) * static_cast<size_t>(in_features);
      for (int k = 0; k < in_features; ++k) {
        a[base + static_cast<size_t>(k)] =
            xq[base + static_cast<size_t>(k)] * s;
      }
    }
  }
  // Reconstruct the effective ternary weight the forward multiplied:
  //   w_eff[j,k] = ternary[j,k] * weight_scale
  Tensor w_eff({out_features, in_features}, Device::CPU);
  {
    float *we = w_eff.data();
    const size_t total =
        static_cast<size_t>(out_features) * static_cast<size_t>(in_features);
    if (unpacked_weights_i8.size() == total) {
      for (size_t i = 0; i < total; ++i) {
        we[i] = static_cast<float>(unpacked_weights_i8[i]) * weight_scale;
      }
    } else {
      // Fallback: derive ternary from the latent weights with the canonical
      // NSOS rule (identical to quantize_weights / pack_weights).
      const float *w = weight.data.data();
      const float scale = weight_scale + 1e-8f;
      for (size_t i = 0; i < total; ++i) {
        const float v = w[i] / scale;
        we[i] = (v > 0.5f ? 1.0f : (v < -0.5f ? -1.0f : 0.0f)) * weight_scale;
      }
    }
  }

  // LoQA contributes to the pre-magnitude output (it adds loqa(input)).
  Tensor loqa_out;
  if (loqa.active) {
    loqa_out = input_2d.matmul(loqa.A.data).matmul(loqa.B.data); // [M, out]
  }

  // Recover `pre` (gemm output before the magnitude/bias affine) for the exact
  // magnitude gradient.  The fused forward folds the affine into the gemm, so
  // recompute the un-fused gemm from the saved quantized activation.
  Tensor pre_2d =
      gemm_158bit_ultra(saved_x_quant, saved_act_scales, false)
          .reshape({M, out_features});
  Tensor pre_full =
      (loqa.active && loqa_out.size > 0) ? pre_2d.add(loqa_out) : pre_2d;

  // d_magnitude[j] = Sum_i grad[i,j] * pre_full[i,j]   (exact)
  magnitude.add_grad(grad_2d.mul(pre_full).sum(0));

  // d_bias = Sum_i grad[i,j]
  if (use_bias) {
    bias.add_grad(grad_2d.sum(0));
  }

  // Gradient into the pre-magnitude output.
  Tensor grad_pre = grad_2d.mul(magnitude.data);

  // Weight gradient (STE through ternary): dW = grad_pre^T @ act_dequant.
  Tensor dW = grad_pre.transpose().matmul(act_dequant);

  // Tequila dead-zone gradient boost (opt-in): help near-zero latent weights
  // escape the ternary dead band.
  if (use_tequila) {
    float *dw_ptr = dW.data();
    const float *w_ptr = weight.data.data();
#pragma omp parallel for
    for (int i = 0; i < (int)dW.size; ++i) {
      if (std::abs(w_ptr[i]) < 0.05f)
        dw_ptr[i] *= 1.5f;
    }
  }
  // STE clip: stop pushing latent weights already saturated past the ternary
  // band (|W / scale| > 1) so they do not drift unboundedly.
  {
    float *dw_ptr = dW.data();
    const float *w_ptr = weight.data.data();
    const float scale = weight_scale + 1e-8f;
#pragma omp parallel for
    for (int i = 0; i < (int)dW.size; ++i) {
      if (std::abs(w_ptr[i] / scale) > 1.0f) {
        dw_ptr[i] = 0.0f;
      }
    }
  }
  weight.add_grad(dW);

  // LoQA adapter backward (its input is the raw layer input; its output is
  // scaled by magnitude in the forward, so it uses grad_pre).
  Tensor dx_loqa;
  if (loqa.active) {
    Tensor inputA = input_2d.matmul(loqa.A.data);             // [M, r]
    Tensor dB = inputA.transpose().matmul(grad_pre);          // [r, out]
    loqa.B.add_grad(dB);
    Tensor gradBt = grad_pre.matmul(loqa.B.data.transpose()); // [M, r]
    Tensor dA = input_2d.transpose().matmul(gradBt);          // [in, r]
    loqa.A.add_grad(dA);
    dx_loqa = gradBt.matmul(loqa.A.data.transpose());         // [M, in] (wrt raw input)
  }

  // Gradient into the (flatquant/hadamard-transformed) activation, STE through
  // the int8 activation quantizer: uses the effective ternary weight.
  Tensor d_act = grad_pre.matmul(w_eff); // [M, in]

  // Reverse the Hadamard transform (orthonormal -> self-inverse).  Off by
  // default (use_hadamard=false); applied symmetrically when enabled.
  if (use_hadamard) {
    d_act = d_act.clone();
    hadamard_transform(d_act.data(), M, in_features);
  }

  // Reverse FlatQuant: xt = alpha * x_norm + beta, on the pre-flatquant
  // activation x_norm (= rmsnorm(input), or input itself if no norm).
  Tensor dx;
  if (use_flatquant) {
    Tensor x_norm = (norm_strategy == NormStrategy::RMS_PERI ||
                     norm_strategy == NormStrategy::RMS_PRE)
                        ? saved_x_norm
                        : saved_input;
    if (x_norm.shape.dims.size() != 2) {
      x_norm = x_norm.reshape({M, in_features});
    }
    flat_alpha.add_grad(d_act.mul(x_norm).sum(0));
    flat_beta.add_grad(d_act.sum(0));
    dx = d_act.mul(flat_alpha.data);
  } else {
    dx = d_act;
  }

  // Add the LoQA input-gradient (it bypasses flatquant; its input is raw).
  if (loqa.active && dx_loqa.size > 0) {
    dx = dx.add(dx_loqa);
  }

  // Reverse RMSNorm back to the layer input.
  if (norm_strategy == NormStrategy::RMS_PERI ||
      norm_strategy == NormStrategy::RMS_PRE) {
    dx = saved_input.rmsnorm_backward(dx, saved_x_norm);
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
