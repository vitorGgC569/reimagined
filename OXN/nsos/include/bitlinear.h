#ifndef BITLINEAR_H
#define BITLINEAR_H

#include "autograd.h"
#include "tensor.h"
#include <cstdint>
#include <vector>

namespace nsos {

// ============================================================================
// BitLinear Ultra SOTA (2025-2026)
// ============================================================================

enum class NormStrategy { RMS_PRE, RMS_PERI, LN_PRE };

struct TequilaState {
  Tensor deadzone_mask;
  Tensor dynamic_biases;
  float trust_threshold = 0.85f;
};

struct LoQAAdapter {
  Parameter A;
  Parameter B;
  bool active = false;
  Tensor apply(const Tensor &input) {
    if (!active)
      return Tensor();
    return input.matmul(A.data).matmul(B.data);
  }
};

struct BitLinearPackedState {
  int in_features = 0;
  int out_features = 0;
  bool use_bias = true;
  float weight_scale = 1.0f;
  std::vector<uint32_t> packed_weights;
  std::vector<float> magnitude;
  std::vector<float> bias;
  std::vector<float> flat_alpha;
  std::vector<float> flat_beta;
};

class BitLinear {
public:
  BitLinear(int in, int out, bool b = true);

  void set_precision_mode(int bits);
  void set_mixed_precision(bool enabled) {
    (void)enabled;
  } // Ultra-SOTA compatibility
  void set_use_hadamard(bool use) { use_hadamard = use; }
  void set_use_tequila(bool use) { use_tequila = use; }
  void set_use_loqa(bool use) { loqa.active = use; }
  void set_reference_path(bool use) { use_reference_path = use; }
  bool reference_path_enabled() const { return use_reference_path; }

  Tensor forward(const Tensor &input);
  Tensor backward(const Tensor &grad_output);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void repack_weights();
  void release_full_precision_weight();
  bool has_full_precision_weight() const { return weight.data.size > 0; }
  BitLinearPackedState export_packed_state() const;
  void import_packed_state(const BitLinearPackedState& state,
                           Device dev = Device::CPU,
                           bool release_full_precision = false);
  Tensor quantize_weights(const Tensor &w_float); // SOTA for benchmarks/tests

private:
  Tensor quantize_activations_bitnet(const Tensor &x,
                                     std::vector<float> &out_scales);
  Tensor gemm_158bit_ultra(const Tensor &x_q,
                           const std::vector<float> &act_scales,
                           bool fuse_output_affine = false);
  const Tensor& materialize_weight_for_device(Device dev);
  void invalidate_cached_materialized_weights();

  void apply_flatquant(float *data, int M, int K, bool incoming);

  int in_features;
  int out_features;
  bool use_bias;
  bool use_hadamard = false;
  bool use_tequila = false;
  bool use_flatquant = true;
  bool use_reference_path = true;

  NormStrategy norm_strategy = NormStrategy::RMS_PERI;
  float weight_scale = 1.0f;

public:
  Parameter weight;
  Parameter magnitude;
  Parameter bias;

  // Learnable affine parameters for FlatQuant
  Parameter flat_alpha;
  Parameter flat_beta;

  LoQAAdapter loqa;
  TequilaState tequila;

  int precision_bits = 8;

private:
  void pack_weights(const Tensor &w_float);

  Tensor saved_input;
  Tensor saved_x_norm;
  Tensor saved_linear_input;
  Tensor saved_pre_output;
  Tensor saved_x_quant;
  std::vector<float> saved_act_scales;

  std::vector<uint32_t> packed_weights;
  std::vector<int8_t> unpacked_weights_i8;
  std::vector<int32_t> unpacked_weight_row_sums;
  size_t packed_stride;
  uint64_t packed_weight_version = 0;
  bool packed_weight_valid = false;
  Tensor cached_gpu_weight_;
  uint64_t cached_gpu_weight_version = 0;
};

} // namespace nsos

#endif // BITLINEAR_H
