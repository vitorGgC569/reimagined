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

  Tensor forward(const Tensor &input);
  Tensor backward(const Tensor &grad_output);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void repack_weights();
  Tensor quantize_weights(const Tensor &w_float); // SOTA for benchmarks/tests

private:
  Tensor quantize_activations_bitnet(const Tensor &x,
                                     std::vector<float> &out_scales);
  Tensor gemm_158bit_ultra(const Tensor &x_q,
                           const std::vector<float> &act_scales);

  // SOTA Helpers
  void apply_hadamard(float *data, int M, int K);
  void apply_flatquant(float *data, int M, int K, bool incoming);
  void compute_trust_weights(const Tensor &x_norm, const Tensor &x_q);

  int in_features;
  int out_features;
  bool use_bias;
  bool use_hadamard = false;
  bool use_tequila = true;
  bool use_flatquant = true;

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
  Tensor saved_x_quant;
  std::vector<float> saved_act_scales;

  std::vector<uint32_t> packed_weights;
  size_t packed_stride;
};

} // namespace nsos

#endif // BITLINEAR_H
