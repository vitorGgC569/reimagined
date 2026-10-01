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
#include <limits>

#ifdef USE_CUDA
#include "../include/gpu_backend.h"
#endif

namespace nsos {

#if defined(NSOS_ENABLE_AVX2_KERNELS)
// Forward decl implemented in src/bitlinear_quantize_avx2.cpp.
int32_t bitlinear_row_sum_i8_avx2(const int8_t* row_ptr, int cols);
#endif

namespace {

int require_positive_feature_count(int value, const char* label) {
  if (value <= 0) {
    throw std::invalid_argument(
        std::string("BitLinear ") + label + " must be positive");
  }
  return value;
}

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
  //
  // The AVX2 kernel is selected by avx2_runtime_supported(), not by the build
  // switch alone: NSOS_ENABLE_AVX2_KERNELS defaults to ON for every x86_64
  // build, so dispatching on it would execute an illegal instruction on a
  // pre-AVX2 CPU. This matches how gemm_158bit_i8 already dispatches.
#if defined(NSOS_ENABLE_AVX2_KERNELS)
  const bool use_avx2 = avx2_runtime_supported();
#else
  constexpr bool use_avx2 = false;
#endif
  for (int row = 0; row < rows; ++row) {
    const int8_t* row_ptr = weights.data() + static_cast<size_t>(row) * cols;
#if defined(NSOS_ENABLE_AVX2_KERNELS)
    if (use_avx2) {
      row_sums[static_cast<size_t>(row)] =
          bitlinear_row_sum_i8_avx2(row_ptr, cols);
      continue;
    }
#endif
    (void)use_avx2;
    int32_t sum = 0;
    for (int col = 0; col < cols; ++col) {
      sum += static_cast<int32_t>(row_ptr[col]);
    }
    row_sums[static_cast<size_t>(row)] = sum;
  }
}

} // namespace

BitLinear::BitLinear(int in, int out, bool b)
    : in_features(require_positive_feature_count(in, "input features")),
      out_features(require_positive_feature_count(out, "output features")),
      use_bias(b),
      weight(Tensor::kaiming_uniform({out_features, in_features}), "weight"),
      magnitude(Tensor::ones({out_features}, Device::CPU), "magnitude"),
      bias(Tensor::zeros({out_features}), "bias"),
      flat_alpha(Tensor::ones({in_features}), "flat_alpha", false),
      flat_beta(Tensor::zeros({in_features}), "flat_beta", false) {

  packed_stride =
      static_cast<size_t>(in_features / 16 + (in_features % 16 != 0));
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
    : in_features(require_positive_feature_count(in, "input features")),
      out_features(require_positive_feature_count(out, "output features")),
      use_bias(b),
      weight((seed == 0)
                ? Tensor::kaiming_uniform({out_features, in_features})
                : Tensor::kaiming_uniform(
                      {out_features, in_features}, Device::CPU, seed),
             "weight"),
      magnitude(Tensor::ones({out_features}, Device::CPU), "magnitude"),
      bias(Tensor::zeros({out_features}), "bias"),
      flat_alpha(Tensor::ones({in_features}), "flat_alpha", false),
      flat_beta(Tensor::zeros({in_features}), "flat_beta", false) {

  packed_stride =
      static_cast<size_t>(in_features / 16 + (in_features % 16 != 0));
  repack_weights();

  loqa.A = Parameter(Tensor::zeros({in, 32}), "loqa_A");
  loqa.B = Parameter(Tensor::zeros({32, out}), "loqa_B");

  tequila.deadzone_mask = Tensor::zeros({out, in});
      tequila.dynamic_biases = Tensor::zeros({out});
}

void BitLinear::invalidate_cached_materialized_weights() {
  cached_gpu_weight_ = Tensor();
  cached_gpu_weight_version = 0;
  cached_gpu_packed_weights_ = Tensor();
  cached_gpu_packed_version_ = 0;
  cached_heat_map_.clear();
  // The QAT inference ternary cache derives from the same weights; every
  // point that invalidates the materialized-weight cache (load, device move,
  // repack, release) invalidates it too.
  qat_inference_cache_valid_ = false;
  qat_inference_w_eff_ = Tensor();
  qat_inference_weight_version_ = 0;
  saved_qat_w_eff_ = Tensor();
  saved_qat_scale_ = Tensor();
  saved_qat_weight_version_ = 0;
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
  // Packing is host-side preparation. Native device storage crosses an
  // explicit D2H boundary; host code never dereferences a cudaMalloc pointer.
  const Tensor w_pack_source =
      w_float.get_device() == Device::GPU ? w_float.cpu() : w_float;
  
  // Use Microsoft style packing for BitNet kernels
  // bitnet.cpp packs 4 values per byte
  const size_t packed_byte_count =
      (static_cast<size_t>(N) * static_cast<size_t>(K) + 3) / 4;
  const size_t packed_word_count = (packed_byte_count + sizeof(uint32_t) - 1) /
                                   sizeof(uint32_t);
  packed_weights.assign(packed_word_count, 0u);
  
  // Canonical ternary scale = ABSMEAN (BitNet b1.58 reference recipe:
  // scale = mean(|W|), W_q = clip(round(W/scale), ±1) — arXiv:2402.17764).
  // Was RMS (||W||/√numel), an undocumented divergence: RMS ≥ absmean always,
  // so the ternary came out sparser than the recipe proven at 2B/4T scale —
  // and inconsistent with the rest of this codebase (Slender embedding eq.8
  // and the OXTA-CRIT branch-gain rule both already use absmean).
  this->weight_scale = tensor_abs_mean(w_pack_source) + 1e-8f;

  // Canonical NSOS ternary rule — single source of truth.  Quantize with
  // quantize_weights() (t = clamp(round(W / scale), -1, +1), scale =
  // mean(|W|)) and pack the resulting {-1,0,+1} codes.  This makes
  // the packed weights identical to quantize_weights() and to the QAT
  // regularizer target, so quantized training (STE) and packed inference
  // share exactly one quantization rule.
  Tensor ternary_codes = quantize_weights(w_pack_source);

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
    // Gated on avx2_runtime_supported(), not on the build switch: see the
    // note in compute_weight_row_sums above.
#if defined(NSOS_ENABLE_AVX2_KERNELS)
    if (avx2_runtime_supported()) {
      const float max_val = bitlinear_row_max_abs_avx2(row_ptr, K);
      const float scale = q_max / (max_val + 1e-8f);
      out_scales[i] = (max_val + 1e-8f) / q_max;
      bitlinear_row_scale_round_avx2(row_q_ptr, row_ptr, K, scale);
      continue;
    }
#endif
    float max_val = 0.0f;
    for (int j = 0; j < K; ++j)
      max_val = std::max(max_val, std::abs(row_ptr[j]));
    float scale = q_max / (max_val + 1e-8f);
    out_scales[i] = (max_val + 1e-8f) / q_max;
    for (int j = 0; j < K; ++j) {
      float val = row_ptr[j] * scale;
      row_q_ptr[j] = std::round(val);
    }
  }
  return x_q;
}

Tensor BitLinear::gemm_158bit_ultra(const Tensor &x_q,
                                    const std::vector<float> &act_scales,
                                    bool fuse_output_affine) {
  const int M = x_q.shape.numel() / in_features;
  const int N = out_features;
  // The packed adapters consume [rows, features]. A batched input preserves
  // its [batch, time, features] shape through quantization, so flatten it here
  // just as the reference and GPU projections do.
  const Tensor x_flat = x_q.reshape({M, in_features});
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
        x_flat, packed_weights, cached_heat_map_, act_scales, weight_scale, y,
        magnitude_ptr, bias_ptr, fuse_output_affine && use_bias);
    return y;
  }

  if (!unpacked_weights_i8.empty()) {
    BitNetAdapter::gemm_158bit_i8(x_flat, unpacked_weights_i8, unpacked_weight_row_sums,
                                  act_scales, weight_scale, y, magnitude_ptr, bias_ptr,
                                  fuse_output_affine && use_bias);
  } else {
    BitNetAdapter::gemm_158bit_lut(x_flat, packed_weights, act_scales, weight_scale, y,
                                   magnitude_ptr, bias_ptr,
                                   fuse_output_affine && use_bias);
  }

  return y;
}

bool BitLinear::prepare_gpu_decode_view(GpuLinearView& view) {
  view = {};
#ifdef USE_CUDA
  if (training_mode_ || loqa.active || matmul_precision_mode() != 0 ||
      in_features <= 0 || out_features <= 0 || precision_bits < 2 || precision_bits > 8 ||
      magnitude.data.get_device() != Device::GPU ||
      (norm_strategy != NormStrategy::NONE &&
       norm_strategy != NormStrategy::RMS_PRE &&
       norm_strategy != NormStrategy::RMS_PERI)) return false;
  view.inputs = in_features;
  view.outputs = out_features;
  view.rms_input = norm_strategy != NormStrategy::NONE;
  view.magnitude = exact_linear_mode_ ? nullptr : magnitude.data.raw_data();
  view.bias = use_bias ? bias.data.raw_data() : nullptr;
  if (use_reference_path) {
    if (weight.data.size == 0 || weight.data.get_device() != Device::GPU) return false;
    view.weight = weight.data.raw_data();
    return true;
  }
  if (gpu_packed_inference_enabled_ && packed_weight_valid && in_features % 16 == 0) {
    const int words = static_cast<int>(packed_weights.size());
    if (cached_gpu_packed_weights_.size != words ||
        cached_gpu_packed_version_ != packed_weight_version) {
      Tensor host({words}, Device::CPU);
      std::memcpy(host.data(), packed_weights.data(), packed_weights.size() * sizeof(uint32_t));
      cached_gpu_packed_weights_ = host.to(Device::GPU);
      cached_gpu_packed_version_ = packed_weight_version;
    }
    view.packed = reinterpret_cast<const uint32_t*>(cached_gpu_packed_weights_.raw_data());
    view.activation_bits = precision_bits;
    view.weight_scale = weight_scale;
    return true;
  }
  // Keep the fake-quant path identical to forward(), including its opt-in
  // packed policy and version-based invalidation.
  if (!gpu_packed_inference_enabled_ && !quantization_sensitive_ && weight.data.size > 0) {
    if (!qat_inference_cache_valid_ ||
        qat_inference_w_eff_.size != weight.data.size ||
        qat_inference_weight_version_ != weight.version ||
        qat_inference_w_eff_.get_device() != Device::GPU) {
      qat_inference_scale_ = tensor_abs_mean(weight.data) + 1e-8f;
      qat_inference_w_eff_ = qat_fake_quant_ternary(weight.data, qat_inference_scale_);
      qat_inference_weight_version_ = weight.version;
      qat_inference_cache_valid_ = true;
    }
    view.weight = qat_inference_w_eff_.raw_data();
    view.activation_bits = precision_bits;
    return true;
  }
  view.weight = materialize_weight_for_device(Device::GPU).raw_data();
  return true;
#else
  return false;
#endif
}

bool BitLinear::supports_gpu_grouped_training() const {
#ifdef USE_CUDA
  return training_mode_ && !loqa.active &&
      (use_reference_path || !quantization_sensitive_) &&
      (norm_strategy == NormStrategy::NONE || norm_strategy == NormStrategy::RMS_PRE ||
       norm_strategy == NormStrategy::RMS_PERI) &&
      weight.data.get_device() == Device::GPU &&
      weight.data.shape.dims == std::vector<int>{out_features, in_features} &&
      (exact_linear_mode_ || (magnitude.data.get_device() == Device::GPU && magnitude.data.size == out_features)) &&
      (!use_bias || (bias.data.get_device() == Device::GPU && bias.data.size == out_features)) &&
      precision_bits >= 1 && precision_bits <= 16;
#else
  return false;
#endif
}

void BitLinear::prepare_gpu_grouped_training_view(GpuMoeTrainingLinearView& view,
    Tensor& effective_weight, Tensor& qat_scale, bool defer_qat) {
  if (!supports_gpu_grouped_training())
    throw std::invalid_argument("Grouped MoE training requires GPU float/QAT BitLinear without LoQA");
  discard_backward_state();
  view = {};
  view.inputs = in_features; view.outputs = out_features;
  view.rms_input = norm_strategy != NormStrategy::NONE;
  view.magnitude = exact_linear_mode_ ? nullptr : magnitude.data.raw_data();
  view.bias = use_bias ? bias.data.raw_data() : nullptr;
  view.latent_weight = weight.data.raw_data();
  if (use_reference_path) {
    effective_weight = weight.data;
    qat_scale = Tensor();
  } else {
    if (defer_qat) {
      // The grouped owner schedules preparation after device routing. No
      // inactive expert's latent weights need to be read/quantized.
      if (effective_weight.shape != weight.data.shape || effective_weight.get_device() != Device::GPU ||
          effective_weight.raw_data() == weight.data.raw_data())
        effective_weight = Tensor::uninitialized(weight.data.shape.dims, Device::GPU);
      if (qat_scale.size != 1 || qat_scale.get_device() != Device::GPU)
        qat_scale = Tensor::uninitialized({1}, Device::GPU);
    } else {
      effective_weight = qat_fake_quant_ternary_absmean(weight.data, &qat_scale);
    }
    view.activation_bits = precision_bits;
    view.qat_scale = qat_scale.raw_data();
  }
  view.weight = effective_weight.raw_data();
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
  // Inference (training_mode_ == false) never calls backward, so skip the
  // clone-heavy saves entirely — this is pure per-token, per-layer overhead
  // on the decode hot path.
  qat_gpu_active_ = false;  // K3: only the GPU QAT branch below sets this true
  saved_qat_w_eff_ = Tensor();
  saved_qat_scale_ = Tensor();
  saved_qat_weight_version_ = 0;
  if (training_mode_) saved_input = input.clone();
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
    if (training_mode_) saved_linear_input = linear_input.clone();

    Tensor output = matmul_nt_cached_weight(
        linear_input, weight.data, weight.version);
    if (loqa.active) {
      Tensor loqa_out = loqa.apply(linear_input);
      if (loqa_out.size > 0) {
        output = output.add(loqa_out);
      }
    }

    if (training_mode_ && !exact_linear_mode_) {
      saved_pre_output = output.clone();
    }
    if (!exact_linear_mode_) {
      output = output.mul(magnitude.data);
    }

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
    if (training_mode_) saved_linear_input = linear_input.clone();

    // ── K3: GPU 1.58-bit ternary path (fake-quant), TRAIN + INFERENCE ─────
    // Reaching the GPU branch means use_reference_path == false, i.e. the QAT
    // scheduler put this (non-sensitive) layer into the QUANTIZED phase.  Run the
    // forward through ternary-weight + int8-activation FAKE quantization
    // (dequantized, so it stays differentiable): during training this is the
    // QAT STE forward (we also cache x_dq/pre for the matching STE backward);
    // during INFERENCE it is the real ternary deployment forward on the GPU.
    // Previously this was gated on training_mode_, so GPU EVAL of a QAT model
    // silently fell back to a float matmul (materialize_weight_for_device) —
    // wrong: a quantized layer must infer in ternary on GPU too, not only on the
    // CPU packed path.  Sensitive layers (Mamba dt/B/C, FP router) keep the float
    // reference path (reference_path stays true, they return earlier).  When the
    // caller opted into the faster __dp4a packed kernel
    // (gpu_packed_inference_enabled_) we defer to it below instead of this
    // float-of-dequant path.  LoQA active falls back to float matmul.
    const bool qat_ternary_path =
        (training_mode_ || !gpu_packed_inference_enabled_) &&
        !quantization_sensitive_ && !loqa.active && weight.data.size > 0 &&
        M > 0 && in_features > 0 && out_features > 0;
    if (qat_ternary_path) {
      // Same scale rule as pack_weights()/quantize_weights() so the QAT codes
      // match what packed inference will use.
      Tensor w_eff;
      Tensor qat_scale;
      if (training_mode_) {
        // Weights change every optimizer step: recompute per forward, but keep
        // the absmean scale on device for GPU QAT to avoid a D2H sync per
        // quantized layer.
        w_eff = qat_fake_quant_ternary_absmean(weight.data, &qat_scale);  // [out,in]
        if (qat_scale.size > 0 && qat_scale.get_device() == Device::CPU) {
          weight_scale = qat_scale.data()[0];
        }
      } else {
        // Inference: the weights are frozen, so the absmean scale (a device
        // reduction + sync D2H on GPU) and the ternary w_eff materialization
        // are computed ONCE and reused for every token.  Besides the obvious
        // decode win, the sync D2H would also abort CUDA-graph capture.
        if (!qat_inference_cache_valid_ ||
            qat_inference_w_eff_.size != weight.data.size ||
            qat_inference_w_eff_.get_device() != weight.data.get_device() ||
            qat_inference_weight_version_ != weight.version) {
          qat_inference_scale_ = tensor_abs_mean(weight.data) + 1e-8f;
          qat_inference_w_eff_ =
              qat_fake_quant_ternary(weight.data, qat_inference_scale_);
          qat_inference_weight_version_ = weight.version;
          qat_inference_cache_valid_ = true;
        }
        weight_scale = qat_inference_scale_;
        w_eff = qat_inference_w_eff_;
      }
      Tensor x_dq = qat_fake_quant_activations(linear_input, precision_bits);  // [M,in]
      Tensor pre = training_mode_
                       ? matmul_nt(x_dq, w_eff)
                       : matmul_nt_cached_weight(
                             x_dq, w_eff, weight.version);                     // [M,out]
      if (training_mode_) {
        // Backward state — only needed for the STE backward during training.
        qat_gpu_active_ = true;
        saved_qat_x_dq_ = x_dq;
        saved_qat_pre_ = pre;
        saved_qat_w_eff_ = w_eff;
        saved_qat_scale_ = qat_scale;
        saved_qat_weight_version_ = weight.version;
      }
      Tensor out_q =
          exact_linear_mode_ ? pre : pre.mul(magnitude.data);
      if (use_bias) {
        out_q = out_q.add(bias.data);
      }
      if (input.shape.dims.size() == 3) {
        return out_q.reshape(
            {input.shape.dims[0], input.shape.dims[1], out_features});
      }
      if (input.shape.dims.size() == 1) {
        return out_q.reshape({out_features});
      }
      return out_q;
    }

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
        !training_mode_ && gpu_packed_inference_enabled_ && packed_weight_valid &&
        !loqa.active && M > 0 && in_features > 0 && out_features > 0 &&
        (in_features % 16 == 0);
    if (dp4a_eligible) {
      gpu_packed_dispatch_count_.fetch_add(1, std::memory_order_relaxed);
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
          in_features, out_features, precision_bits,
          exact_linear_mode_ ? nullptr : &magnitude.data,
          use_bias ? &bias.data : nullptr);
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
    Tensor output = matmul_nt_cached_weight(
        linear_input, effective_weight, weight.version);
    if (loqa.active) {
      Tensor loqa_out = loqa.apply(linear_input);
      if (loqa_out.size > 0) {
        output = output.add(loqa_out);
      }
    }

    if (training_mode_ && !exact_linear_mode_) {
      saved_pre_output = output.clone();
    }
    if (!exact_linear_mode_) {
      output = output.mul(magnitude.data);
    }
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

  // LoQA operates on the SAME (rmsnorm'd) activations as the reference/GPU
  // paths — the reference path feeds it linear_input = x ({M, in_features}).
  // The old code fed loqa the RAW pre-norm `input` here, so the adapter's
  // contribution diverged from training and from the GPU path.  Capture the
  // normalized activations before the packed-only transforms below.
  const Tensor loqa_input = x.reshape({M, in_features});
  // FlatQuant is an activation-space transform, but the reference (float) path
  // used for TRAINING/QAT applies it NOT, pack_weights does not fold it into
  // the packed weights, and the GPU packed path also skips it.  Applying it
  // only here made CPU packed inference diverge from both the trained numerics
  // and the GPU path.  Keep packed inference consistent: do not apply FlatQuant.
  // (Re-enabling FlatQuant requires wiring it — with its gradient — into the
  // reference path and every inference path together.)
  if (use_hadamard) {
    x = x.clone();  // clone before the in-place transform (input may be shared)
    hadamard_transform(x.data(), M, in_features);
  }
  saved_x_quant = quantize_activations_bitnet(x, saved_act_scales);
  const bool fuse_output_affine =
      !loqa.active && !exact_linear_mode_;
  Tensor output = gemm_158bit_ultra(saved_x_quant, saved_act_scales, fuse_output_affine);
  if (loqa.active) {
    Tensor loqa_out = loqa.apply(loqa_input);
    if (loqa_out.shape.numel() > 0)
      output = output.add(loqa_out);
  }
  if (!fuse_output_affine) {
    if (!exact_linear_mode_) {
      output = output.mul(magnitude.data);
    }
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

void BitLinear::discard_backward_state() {
  saved_input = Tensor();
  saved_linear_input = Tensor();
  saved_pre_output = Tensor();
  saved_x_norm = Tensor();
  saved_x_quant = Tensor();
  saved_act_scales.clear();
  saved_qat_x_dq_ = Tensor();
  saved_qat_pre_ = Tensor();
  saved_qat_w_eff_ = Tensor();
  saved_qat_scale_ = Tensor();
  saved_qat_weight_version_ = 0;
  qat_gpu_active_ = false;
}

void BitLinear::release_full_precision_weight() {
  weight.data = Tensor();
  weight.grad = Tensor();
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

size_t BitLinear::auxiliary_memory_usage_bytes() const {
  size_t bytes = 0;
  const auto add = [&](size_t amount) {
    if (amount > (std::numeric_limits<size_t>::max)() - bytes) {
      bytes = (std::numeric_limits<size_t>::max)();
    } else {
      bytes += amount;
    }
  };
  const auto vector_bytes = [&](size_t capacity, size_t element_size) {
    if (capacity > (std::numeric_limits<size_t>::max)() / element_size) {
      add((std::numeric_limits<size_t>::max)());
    } else {
      add(capacity * element_size);
    }
  };
  const auto tensor_bytes = [&](const Tensor& tensor) {
    if (tensor.size > 0) {
      const auto elements = static_cast<size_t>(tensor.size);
      vector_bytes(elements, sizeof(float));
    }
  };

  vector_bytes(packed_weights.capacity(), sizeof(uint32_t));
  vector_bytes(unpacked_weights_i8.capacity(), sizeof(int8_t));
  vector_bytes(unpacked_weight_row_sums.capacity(), sizeof(int32_t));
  vector_bytes(cached_heat_map_.capacity(), sizeof(uint8_t));
  tensor_bytes(cached_gpu_weight_);
  tensor_bytes(cached_gpu_packed_weights_);
  tensor_bytes(qat_inference_w_eff_);
  return bytes;
}

BitLinear::PreparedPackedState BitLinear::prepare_packed_state(
    const BitLinearPackedState& state,
    Device dev,
    bool release_full_precision,
    const Tensor* exact_full_precision_weight) const {
  if (state.in_features != in_features || state.out_features != out_features ||
      state.use_bias != use_bias) {
    throw std::runtime_error("BitLinear packed state shape mismatch");
  }

  const size_t total_weights =
      static_cast<size_t>(out_features) *
      static_cast<size_t>(in_features);
  const size_t expected_words = (total_weights + 15U) / 16U;
  if (state.packed_weights.size() != expected_words ||
      state.magnitude.size() != static_cast<size_t>(out_features) ||
      state.bias.size() != (use_bias
                                ? static_cast<size_t>(out_features)
                                : 0U) ||
      state.flat_alpha.size() != static_cast<size_t>(in_features) ||
      state.flat_beta.size() != static_cast<size_t>(in_features)) {
    throw std::runtime_error(
        "BitLinear packed state buffer length mismatch");
  }
  if (!std::isfinite(state.weight_scale) ||
      state.weight_scale <= 0.0f) {
    throw std::runtime_error(
        "BitLinear packed state has an invalid weight scale");
  }
  const auto require_finite = [](const std::vector<float>& values,
                                 const char* label) {
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error(
          std::string("BitLinear packed state contains a non-finite ") +
          label);
    }
  };
  require_finite(state.magnitude, "magnitude");
  require_finite(state.bias, "bias");
  require_finite(state.flat_alpha, "flat-alpha buffer");
  require_finite(state.flat_beta, "flat-beta buffer");
  for (size_t index = 0; index < total_weights; ++index) {
    const uint32_t code =
        (state.packed_weights[index / 16U] >>
         (2U * static_cast<unsigned int>(index % 16U))) &
        0x3U;
    if (code == 0x3U) {
      throw std::runtime_error(
          "BitLinear packed state contains a reserved ternary code");
    }
  }

  if (release_full_precision && exact_full_precision_weight != nullptr) {
    throw std::invalid_argument(
        "Cannot release and replace a BitLinear full-precision weight in the "
        "same packed-state transaction");
  }

  // Prepare every allocation and derived buffer first. If any allocation or
  // transfer fails, the live layer remains byte-for-byte unchanged.
  PreparedPackedState prepared;
  prepared.packed_weights = state.packed_weights;
  BitNetAdapter::unpack_weights_microsoft_style_to_i8(
      prepared.packed_weights, out_features, in_features,
      prepared.unpacked_weights);
  compute_weight_row_sums(prepared.unpacked_weights, out_features, in_features,
                          prepared.row_sums);
  const auto stage_tensor = [dev](const std::vector<float>& values,
                                  int count) {
    Tensor host =
        Tensor::from_blob(const_cast<float*>(values.data()), {count},
                          Device::CPU)
            .clone();
    return dev == Device::CPU ? host : host.to(dev);
  };
  prepared.magnitude = stage_tensor(state.magnitude, out_features);
  prepared.bias =
      use_bias ? stage_tensor(state.bias, out_features) : Tensor();
  prepared.flat_alpha = stage_tensor(state.flat_alpha, in_features);
  prepared.flat_beta = stage_tensor(state.flat_beta, in_features);
  prepared.weight_scale = state.weight_scale;

  if (!release_full_precision) {
    const Tensor* candidate =
        exact_full_precision_weight != nullptr
            ? exact_full_precision_weight
            : &weight.data;
    if (candidate->size == 0 ||
        candidate->shape.dims !=
            std::vector<int>({out_features, in_features}) ||
        candidate->get_device() != dev) {
      throw std::invalid_argument(
          "Retaining BitLinear FP32 weights requires a matching base weight "
          "with the layer shape and target device");
    }
    const Tensor candidate_cpu =
        candidate->get_device() == Device::GPU
            ? candidate->cpu()
            : *candidate;
    const float* candidate_values = candidate_cpu.data();
    if (!std::all_of(
            candidate_values,
            candidate_values + candidate_cpu.size,
            [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error(
          "Retained BitLinear FP32 weight contains a non-finite value");
    }
    const float candidate_scale =
        tensor_abs_mean(candidate_cpu) + 1e-8f;
    const float scale_tolerance =
        std::max(1e-7f,
                 std::abs(state.weight_scale) * 1e-6f);
    if (std::abs(candidate_scale - state.weight_scale) >
        scale_tolerance) {
      throw std::runtime_error(
          "Retained BitLinear FP32 weight does not match the edge-pack "
          "quantization scale (fp32=" +
          std::to_string(candidate_scale) +
          ", packed=" + std::to_string(state.weight_scale) +
          ", in=" + std::to_string(in_features) +
          ", out=" + std::to_string(out_features) + ")");
    }
    for (size_t index = 0; index < total_weights; ++index) {
      const float normalized =
          candidate_values[index] /
          (candidate_scale + 1e-8f);
      const uint32_t expected_code =
          normalized > 0.5f
              ? 0x2u
              : (normalized < -0.5f ? 0x0u : 0x1u);
      const uint32_t packed_code =
          (state.packed_weights[index / 16U] >>
           (2U * static_cast<unsigned int>(index % 16U))) &
          0x3U;
      if (packed_code != expected_code) {
        throw std::runtime_error(
            "Retained BitLinear FP32 weight does not match the edge-pack "
            "ternary codes");
      }
    }
    // A live layer may intentionally share its weight storage with another
    // parameter (the tied token embedding/value head is the canonical case).
    // Replacing that Tensor during the noexcept commit would silently break
    // the alias even though the candidate was already the exact validated
    // destination storage. Preserve it in place; external staged weights are
    // still cloned here so commit remains allocation-free and transactional.
    prepared.preserve_existing_full_precision_storage =
        candidate->raw_data() == weight.data.raw_data();
    if (!prepared.preserve_existing_full_precision_storage) {
      prepared.full_precision_weight = candidate->clone();
    }
    prepared.keep_full_precision_weight = true;
  }
  return prepared;
}

void BitLinear::commit_prepared_packed_state(
    PreparedPackedState&& prepared) noexcept {
  packed_weights = std::move(prepared.packed_weights);
  unpacked_weights_i8 = std::move(prepared.unpacked_weights);
  unpacked_weight_row_sums = std::move(prepared.row_sums);
  weight_scale = prepared.weight_scale;
  packed_stride =
      static_cast<size_t>(in_features / 16 + (in_features % 16 != 0));
  packed_weight_valid = true;
  magnitude.data = std::move(prepared.magnitude);
  magnitude.grad = Tensor();
  magnitude.mark_updated();
  if (use_bias) {
    bias.data = std::move(prepared.bias);
    bias.grad = Tensor();
    bias.mark_updated();
  }
  flat_alpha.data = std::move(prepared.flat_alpha);
  flat_alpha.grad = Tensor();
  flat_alpha.mark_updated();
  flat_beta.data = std::move(prepared.flat_beta);
  flat_beta.grad = Tensor();
  flat_beta.mark_updated();

  if (prepared.keep_full_precision_weight) {
    if (!prepared.preserve_existing_full_precision_storage) {
      weight.data = std::move(prepared.full_precision_weight);
    }
    weight.grad = Tensor();
    weight.mark_updated();
  } else {
    weight.data = Tensor();
    weight.grad = Tensor();
    use_reference_path = false;
  }
  packed_weight_version = weight.version;
  invalidate_cached_materialized_weights();
}

void BitLinear::import_packed_state(const BitLinearPackedState& state,
                                    Device dev,
                                    bool release_full_precision) {
  PreparedPackedState prepared =
      prepare_packed_state(state, dev, release_full_precision);
  commit_prepared_packed_state(std::move(prepared));
}

void BitLinear::to(Device dev) {
  auto move_parameter = [dev](Parameter& parameter) {
    if (parameter.data.size > 0 &&
        parameter.data.get_device() != dev) {
      parameter.data = parameter.data.to(dev);
    }
    if (parameter.grad.size > 0 &&
        parameter.grad.get_device() != dev) {
      parameter.grad = parameter.grad.to(dev);
    }
  };
  move_parameter(weight);
  move_parameter(bias);
  move_parameter(magnitude);
  move_parameter(flat_alpha);
  move_parameter(flat_beta);
  move_parameter(loqa.A);
  move_parameter(loqa.B);
  invalidate_cached_materialized_weights();
}

std::vector<Parameter *> BitLinear::parameters() {
  std::vector<Parameter *> res;
  res.push_back(&weight);
  if (!exact_linear_mode_)
    res.push_back(&magnitude);
  if (use_bias)
    res.push_back(&bias);
  // Kept in the model parameter registry for checkpoint compatibility only.
  // Trainer explicitly filters these fixed legacy identity buffers out of
  // gradient, optimizer, and Trainer-sidecar state.
  res.push_back(&flat_alpha);
  res.push_back(&flat_beta);
  if (loqa.active) {
    res.push_back(&loqa.A);
    res.push_back(&loqa.B);
  }
  return res;
}

void BitLinear::set_precision_mode(int bits) {
  if (bits < 2 || bits > 8) {
    throw std::invalid_argument(
        "BitLinear precision mode must be in the closed interval [2, 8]");
  }
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

  // ── K3: GPU QAT backward (fake-quant STE) ─────────────────────────────────
  // Checked first: the QAT forward saved its own state (x_dq, pre) and used
  // ternary/int8 fake-quant.  Mirrors the CPU packed STE backward but on
  // device-agnostic Tensor ops (so it runs on the GPU end-to-end).
  if (qat_gpu_active_) {
    if (saved_qat_x_dq_.size == 0 || saved_qat_pre_.size == 0 ||
        weight.data.size == 0) {
      throw std::runtime_error("BitLinear QAT backward: missing saved QAT state");
    }
    // out = pre * magnitude + bias. Exact-linear mode bypasses the
    // compatibility-only magnitude buffer entirely.
    if (!exact_linear_mode_) {
      magnitude.add_grad(grad_2d.mul(saved_qat_pre_).sum(0));
    }
    if (use_bias) {
      bias.add_grad(grad_2d.sum(0));
    }
    Tensor grad_pre =
        exact_linear_mode_ ? grad_2d : grad_2d.mul(magnitude.data);
    // Weight grad (STE through ternary): dW = grad_pre^T @ x_dq (the dequantized
    // activations actually multiplied).
    Tensor dW = matmul_tn(grad_pre, saved_qat_x_dq_);
    // STE clip: zero grad for latent weights already saturated past |W/scale|>1.
    if (saved_qat_scale_.size == 1 &&
        saved_qat_scale_.get_device() == weight.data.get_device() &&
        saved_qat_weight_version_ == weight.version) {
      qat_ste_clip_weight_grad_device_scale(dW, weight.data, saved_qat_scale_);
    } else {
      weight_scale = tensor_abs_mean(weight.data) + 1e-8f;
      qat_ste_clip_weight_grad(dW, weight.data, weight_scale);
    }
    weight.add_grad(dW);
    // Input grad (STE through the int8 activation quantizer): dx = grad_pre @
    // w_eff, using the effective (fake-quantized) weight that the forward used.
    Tensor w_eff =
        (saved_qat_w_eff_.size == weight.data.size &&
         saved_qat_w_eff_.get_device() == weight.data.get_device() &&
         saved_qat_weight_version_ == weight.version)
            ? saved_qat_w_eff_
            : qat_fake_quant_ternary(weight.data, weight_scale);
    Tensor dx = grad_pre.matmul(w_eff);
    dx = dx.reshape(saved_input.shape.dims);
    if (norm_strategy == NormStrategy::RMS_PERI ||
        norm_strategy == NormStrategy::RMS_PRE) {
      dx = saved_input.rmsnorm_backward(dx, saved_x_norm);
    }
    return dx;
  }

  if (use_reference_path) {
    Tensor linear_input = saved_linear_input.size > 0 ? saved_linear_input
                                                      : input_2d;
    // All consumers below are read-only; magnitude.mul() allocates its result.
    // Share the upstream buffer instead of copying the whole [rows, out] tensor.
    Tensor grad_pre = grad_2d;

    if (use_bias) {
      bias.add_grad(grad_2d.sum(0));
    }

    if (!exact_linear_mode_) {
      if (saved_pre_output.size == 0) {
        throw std::logic_error(
            "BitLinear backward is missing the pre-magnitude output");
      }
      magnitude.add_grad(
          grad_2d.mul(saved_pre_output).sum(0));
      grad_pre = grad_pre.mul(magnitude.data);
    }

    Tensor dW = matmul_tn(grad_pre, linear_input);
    weight.add_grad(dW);

    Tensor dx = grad_pre.matmul(weight.data);

    if (loqa.active) {
      Tensor inputA = linear_input.matmul(loqa.A.data);
      Tensor dB = matmul_tn(inputA, grad_pre);
      loqa.B.add_grad(dB);

      Tensor gradBt = matmul_nt(grad_pre, loqa.B.data);
      Tensor dA = matmul_tn(linear_input, gradBt);
      loqa.A.add_grad(dA);

      Tensor dx_loqa = matmul_nt(gradBt, loqa.A.data);
      dx = dx.add(dx_loqa);
    }

    dx = dx.reshape(saved_input.shape.dims);
    if (norm_strategy == NormStrategy::RMS_PERI ||
        norm_strategy == NormStrategy::RMS_PRE) {
      dx = saved_input.rmsnorm_backward(dx, saved_x_norm);
    }

    return dx;
  }

  // ── NON-REFERENCE PATH: straight-through estimator (STE) backward for the
  // packed ternary forward.  The real integer kernel ran in the forward; this
  // back-propagates onto the FP32 latent parameters.  The forward computed:
  //   x_norm = rmsnorm(input)                                   [saved_x_norm]
  //   xt     = hadamard(x_norm)               (if use_hadamard)
  //   x_q    = round(xt / act_scale)         [saved_x_quant], act_scale [saved_act_scales]
  //   w_t    = ternary(W) in {-1,0,+1}       [unpacked_weights_i8], scale = weight_scale
  //   pre    = (x_q * act_scale) @ (w_t * weight_scale)^T
  //            + loqa(x_norm) if active
  //   out    = pre * magnitude + bias
  // STE treats both quantizers as identity for gradient flow; gradients
  // accumulate into the latent FP32 weight / magnitude / bias parameters.
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

  // LoQA bypasses the packed transforms, but consumes the same normalized
  // activation as the reference/GPU paths.
  Tensor loqa_input =
      saved_x_norm.size > 0 ? saved_x_norm.reshape({M, in_features}) : input_2d;
  Tensor loqa_out;
  if (loqa.active) {
    loqa_out = loqa_input.matmul(loqa.A.data).matmul(loqa.B.data); // [M, out]
  }

  if (!exact_linear_mode_) {
    // Recover `pre` (gemm output before the magnitude/bias affine) for the
    // exact magnitude gradient. The fused forward folds the affine into the
    // gemm, so recompute the un-fused result from the saved activation.
    Tensor pre_2d =
        gemm_158bit_ultra(saved_x_quant, saved_act_scales, false)
            .reshape({M, out_features});
    Tensor pre_full =
        (loqa.active && loqa_out.size > 0) ? pre_2d.add(loqa_out) : pre_2d;
    magnitude.add_grad(grad_2d.mul(pre_full).sum(0));
  }

  // d_bias = Sum_i grad[i,j]
  if (use_bias) {
    bias.add_grad(grad_2d.sum(0));
  }

  // Gradient into the pre-magnitude output.
  Tensor grad_pre =
      exact_linear_mode_ ? grad_2d : grad_2d.mul(magnitude.data);

  // Weight gradient (STE through ternary): dW = grad_pre^T @ act_dequant.
  Tensor dW = matmul_tn(grad_pre, act_dequant);

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

  // LoQA adapter backward.  Its input is x_norm, and its output is scaled by
  // magnitude in the forward, so it uses grad_pre.
  Tensor dx_loqa;
  if (loqa.active) {
    Tensor inputA = loqa_input.matmul(loqa.A.data);           // [M, r]
    Tensor dB = matmul_tn(inputA, grad_pre);                  // [r, out]
    loqa.B.add_grad(dB);
    Tensor gradBt = matmul_nt(grad_pre, loqa.B.data);         // [M, r]
    Tensor dA = matmul_tn(loqa_input, gradBt);                // [in, r]
    loqa.A.add_grad(dA);
    dx_loqa = matmul_nt(gradBt, loqa.A.data);                 // [M, in] (wrt x_norm)
  }

  // Gradient into the Hadamard-transformed activation, STE through
  // the int8 activation quantizer: uses the effective ternary weight.
  Tensor d_act = grad_pre.matmul(w_eff); // [M, in]

  // Reverse the Hadamard transform (orthonormal -> self-inverse).  Off by
  // default (use_hadamard=false); applied symmetrically when enabled.
  if (use_hadamard) {
    d_act = d_act.clone();
    hadamard_transform(d_act.data(), M, in_features);
  }

  // FlatQuant is deliberately dormant in every forward path (see forward):
  // do not manufacture gradients for a transform that did not run.
  Tensor dx = d_act;

  // Add the LoQA gradient in x_norm space before reversing RMSNorm.
  if (loqa.active && dx_loqa.size > 0) {
    dx = dx.add(dx_loqa);
  }

  // Reverse RMSNorm back to the layer input.
  dx = dx.reshape(saved_input.shape.dims);
  if (norm_strategy == NormStrategy::RMS_PERI ||
      norm_strategy == NormStrategy::RMS_PRE) {
    dx = saved_input.rmsnorm_backward(dx, saved_x_norm);
  }

  return dx;
}

Tensor BitLinear::quantize_weights(const Tensor &w_float) {
  const Device original_device = w_float.get_device();
  const Tensor source =
      original_device == Device::GPU ? w_float.cpu() : w_float;
  Tensor res(source.shape.dims, Device::CPU);
  // Absmean scale (BitNet b1.58) — must match pack_weights / the QAT forward.
  float scale = tensor_abs_mean(source) + 1e-8f;
  const float *src = source.data();
  float *dst = res.data();
#pragma omp parallel for
  for (int i = 0; i < source.size; ++i) {
    float val = src[i] / (scale + 1e-8f);
    if (val > 0.5f)
      dst[i] = 1.0f;
    else if (val < -0.5f)
      dst[i] = -1.0f;
    else
      dst[i] = 0.0f;
  }
  return original_device == Device::GPU ? res.to(Device::GPU) : res;
}

Tensor BitLinear::add_qat_regularization_grad(float regularization) {
  if (regularization <= 0.0f || quantization_sensitive_ ||
      weight.data.size == 0) {
    return Tensor();
  }

  Tensor ternary_target;
  if (saved_qat_w_eff_.size == weight.data.size &&
      saved_qat_w_eff_.get_device() == weight.data.get_device() &&
      saved_qat_weight_version_ == weight.version) {
    // Training GPU QAT already materialized the scaled ternary weight
    // (scale * code) in forward.  Reuse it for the regularizer instead of
    // calling quantize_weights(), which is host-loop based and forces GPU
    // managed-memory migration.
    ternary_target = saved_qat_w_eff_;
  } else {
    // Warmup and topology-dormant linears have no saved QAT forward tensor.
    // Compute both absmean and fake quantization on device instead of reading
    // one scale scalar per layer back to the host.
    ternary_target =
        qat_fake_quant_ternary_absmean(weight.data, nullptr);
  }

  Tensor penalty_grad = weight.data.sub(ternary_target).mul(regularization);
  weight.add_grad(penalty_grad);
  return penalty_grad;
}

} // namespace nsos
