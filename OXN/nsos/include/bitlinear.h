#ifndef BITLINEAR_H
#define BITLINEAR_H

#include "autograd.h"
#include "tensor.h"
#include "gpu_linear_view.h"
#include "tiled_cross_entropy.h"
#include <atomic>
#include <cstdint>
#include <vector>

namespace nsos {

// ============================================================================
// BitLinear Ultra SOTA (2025-2026)
// ============================================================================

enum class NormStrategy { NONE, RMS_PRE, RMS_PERI, LN_PRE };

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
  // Fully allocated, validated import transaction. Preparation may throw
  // without touching a live layer; commit only moves already-owned buffers.
  class PreparedPackedState {
  public:
    PreparedPackedState(PreparedPackedState&&) noexcept = default;
    PreparedPackedState& operator=(PreparedPackedState&&) noexcept = default;
    PreparedPackedState(const PreparedPackedState&) = delete;
    PreparedPackedState& operator=(const PreparedPackedState&) = delete;

  private:
    friend class BitLinear;
    PreparedPackedState() = default;

    std::vector<uint32_t> packed_weights;
    std::vector<int8_t> unpacked_weights;
    std::vector<int32_t> row_sums;
    float weight_scale = 1.0f;
    Tensor magnitude;
    Tensor bias;
    Tensor flat_alpha;
    Tensor flat_beta;
    Tensor full_precision_weight;
    bool keep_full_precision_weight = false;
    bool preserve_existing_full_precision_storage = false;
  };

  BitLinear(int in, int out, bool b = true);
  // Seeded variant — guarantees reproducible weight init for the same seed.
  // Required by TTTLayer (and any other consumer that needs determinism
  // between instances).  Seed=0 falls back to the un-seeded path.
  BitLinear(int in, int out, bool b, uint64_t seed);

  void set_precision_mode(int bits);
  // Enable BF16 Tensor-Core compute for matmuls (master weights stay FP32).
  // Mixed precision is a global GEMM mode; this toggles it (BF16 when enabled).
  void set_mixed_precision(bool enabled) {
    set_matmul_precision_mode(enabled ? 1 : 0);
  }
  void set_use_hadamard(bool use) { use_hadamard = use; }
  void set_use_tequila(bool use) { use_tequila = use; }
  void set_use_loqa(bool use) { loqa.active = use; }
  void set_reference_path(bool use) { use_reference_path = use; }
  bool reference_path_enabled() const { return use_reference_path; }
  // Make the float path numerically equivalent to a plain nn.Linear:
  // no implicit input RMSNorm and no trainable per-output magnitude.  The
  // magnitude buffer remains serialized for compatibility but is bypassed by
  // forward/backward, so even a legacy non-unit value cannot change exact
  // linear math. Weight ternarization remains available.
  void set_exact_linear_mode(bool enabled) {
    if (exact_linear_mode_ != enabled) {
      // A compatibility-mode magnitude gradient must never survive a switch
      // into exact-linear mode and later leak into an optimizer state.
      magnitude.grad = Tensor();
    }
    exact_linear_mode_ = enabled;
    magnitude.trainable = !enabled;
    norm_strategy = enabled ? NormStrategy::NONE : NormStrategy::RMS_PERI;
  }
  bool exact_linear_mode() const { return exact_linear_mode_; }
  // Register every possible trainable expert parameter, including parameters
  // temporarily excluded by exact-linear mode or a disabled adapter.
  void track_gradient_contributions() noexcept {
    weight.track_gradient_contributions();
    magnitude.track_gradient_contributions();
    bias.track_gradient_contributions();
    flat_alpha.track_gradient_contributions();
    flat_beta.track_gradient_contributions();
    loqa.A.track_gradient_contributions();
    loqa.B.track_gradient_contributions();
  }
  NormStrategy input_norm_strategy() const { return norm_strategy; }

  // Quantization-sensitive layers (e.g. Mamba's dt/B/C "sensitive" input
  // projection) are kept on the float reference path during quantization-aware
  // training, preserving the mixed-precision design: ternary weights for the
  // robust projections, higher precision for the sensitive SSM scan
  // parameters.  The QAT scheduler in trainer.cpp honors this flag.
  void set_quantization_sensitive(bool sensitive) {
    quantization_sensitive_ = sensitive;
  }
  bool quantization_sensitive() const { return quantization_sensitive_; }

  // Opts the GPU forward path into the __dp4a-accelerated 1.58-bit
  // dispatch (see src/bitnet_gpu_dispatch.cpp).  This is INFERENCE-ONLY:
  // backward and gradient computation continue to use the float matmul
  // path so training behavior is unchanged.  Default is OFF — callers
  // such as the inference engine flip it on after weight pack is
  // finalized.  No-op when CUDA is disabled.
  void set_gpu_packed_inference(bool enabled) {
    gpu_packed_inference_enabled_ = enabled;
  }
  bool gpu_packed_inference_enabled() const {
    return gpu_packed_inference_enabled_;
  }
  uint64_t gpu_packed_dispatch_count() const {
    return gpu_packed_dispatch_count_.load(std::memory_order_relaxed);
  }
  void reset_gpu_packed_dispatch_count() {
    gpu_packed_dispatch_count_.store(0, std::memory_order_relaxed);
  }

  // Inference fast path.  When false, forward() skips the clone-heavy
  // backward-state saves (saved_input / saved_linear_input / saved_pre_output),
  // which are only ever read by backward().  Default true preserves training
  // behavior exactly; JambaModel::set_training_mode propagates this so serving
  // never pays for backward bookkeeping it will not use.  The forward OUTPUT is
  // byte-identical either way; only the saved state is elided.
  void set_training_mode(bool enabled) {
    // Any mode transition invalidates the QAT inference ternary cache: on
    // entering training the weights are about to change; on leaving it the
    // cache (if any) was computed from pre-training weights.
    if (enabled != training_mode_) {
      qat_inference_cache_valid_ = false;
      qat_inference_w_eff_ = Tensor();
      qat_inference_weight_version_ = 0;
      saved_qat_w_eff_ = Tensor();
      saved_qat_scale_ = Tensor();
      saved_qat_weight_version_ = 0;
    }
    training_mode_ = enabled;
  }
  bool training_mode() const { return training_mode_; }

  // Release only activation state retained for backward.  This is used by
  // layer-level gradient checkpointing after a forward result has been
  // produced: parameters, packed weights and inference caches remain intact,
  // while the next backward explicitly recomputes these activations.
  void discard_backward_state();

  Tensor forward(const Tensor &input);
  // Inference-only borrowed view for device-selected grouped experts. Returns
  // false for unsupported adapters/precision rather than changing their math.
  bool prepare_gpu_decode_view(GpuLinearView& view);
  bool supports_gpu_grouped_training() const;
  void prepare_gpu_grouped_training_view(GpuMoeTrainingLinearView& view,
      Tensor& effective_weight, Tensor& qat_scale, bool defer_qat = false);
  Tensor backward(const Tensor &grad_output);
  // Combined training operation. Recomputes bounded vocabulary/row tiles;
  // publishes head gradients once all tiles have completed. Never saves logits.
  TiledCrossEntropyResult cross_entropy_tiled(
      const Tensor& input, const TiledCrossEntropyOptions& options);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void repack_weights();
  void release_full_precision_weight();
  bool has_full_precision_weight() const { return weight.data.size > 0; }
  int input_features() const { return in_features; }
  int output_features() const { return out_features; }
  bool uses_bias() const { return use_bias; }
  BitLinearPackedState export_packed_state() const;
  PreparedPackedState prepare_packed_state(
      const BitLinearPackedState& state,
      Device dev = Device::CPU,
      bool release_full_precision = false,
      const Tensor* exact_full_precision_weight = nullptr) const;
  void commit_prepared_packed_state(PreparedPackedState&& prepared) noexcept;
  // Bytes owned by packed/inference caches, excluding public Parameters
  // (which callers account separately).
  size_t auxiliary_memory_usage_bytes() const;
  void import_packed_state(const BitLinearPackedState& state,
                           Device dev = Device::CPU,
                           bool release_full_precision = false);
  Tensor quantize_weights(const Tensor &w_float); // SOTA for benchmarks/tests
  // Adds regularization * (W - stopgrad(Q(W))) to weight.grad and returns that
  // exact penalty-gradient tensor so Trainer can reduce the matching scalar
  // objective with one device synchronization for the whole model.
  Tensor add_qat_regularization_grad(float regularization);

private:
  Tensor quantize_activations_bitnet(const Tensor &x,
                                     std::vector<float> &out_scales);
  Tensor gemm_158bit_ultra(const Tensor &x_q,
                           const std::vector<float> &act_scales,
                           bool fuse_output_affine = false);
  const Tensor& materialize_weight_for_device(Device dev);
  void invalidate_cached_materialized_weights();

  int in_features;
  int out_features;
  bool use_bias;
  bool use_hadamard = false;
  bool use_tequila = false;
  bool use_reference_path = true;
  bool quantization_sensitive_ = false;
  bool exact_linear_mode_ = false;

  NormStrategy norm_strategy = NormStrategy::RMS_PERI;
  float weight_scale = 1.0f;
  // Inference-time QAT ternary cache: at inference the weights are frozen, so
  // the absmean scale (a device reduction + sync D2H when the weights live on
  // the GPU) and the ternary w_eff materialization are computed once, not per
  // forward.  Invalidated on training-mode transitions and everywhere
  // invalidate_cached_materialized_weights() fires (load/move/repack).
  bool qat_inference_cache_valid_ = false;
  float qat_inference_scale_ = 0.0f;
  Tensor qat_inference_w_eff_;
  uint64_t qat_inference_weight_version_ = 0;

public:
  Parameter weight;
  Parameter magnitude;
  Parameter bias;

  // Legacy identity buffers retained only for edge-pack compatibility.
  // FlatQuant is not part of any forward path. They remain in the canonical
  // registry/checkpoint inventory as non-trainable state, while Trainer
  // excludes them from gradients and optimizer sidecars.
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

  // K3: GPU quantization-aware-training (fake-quant STE) state.  Set by the GPU
  // forward when QAT is active (training, non-reference, non-sensitive); the
  // matching backward (checked first) consumes them.  qat_gpu_active_ is reset
  // at the top of every forward so a later inference/reference forward never
  // routes into the QAT backward.
  bool qat_gpu_active_ = false;
  Tensor saved_qat_x_dq_;   // dequantized activations actually multiplied
  Tensor saved_qat_pre_;    // pre-(magnitude/bias) output, for the magnitude grad
  Tensor saved_qat_w_eff_;  // scaled ternary weights used by the QAT forward
  Tensor saved_qat_scale_;  // device scalar scale used by saved_qat_w_eff_
  uint64_t saved_qat_weight_version_ = 0;

  std::vector<uint32_t> packed_weights;
  std::vector<int8_t> unpacked_weights_i8;
  std::vector<int32_t> unpacked_weight_row_sums;
  size_t packed_stride;
  uint64_t packed_weight_version = 0;
  bool packed_weight_valid = false;
  Tensor cached_gpu_weight_;
  uint64_t cached_gpu_weight_version = 0;

  // GPU-resident packed weight buffer for the __dp4a inference fast
  // path (Phase 5b).  Lazy-allocated and refreshed when
  // packed_weight_version changes.  The Tensor is allocated as a flat
  // float buffer whose underlying bytes are reinterpret_cast as
  // uint32_t* by bitnet_gemm_158bit_gpu — float is used because
  // Tensor only knows the float type today (changing that is a much
  // larger refactor not in scope here).
  Tensor cached_gpu_packed_weights_;
  uint64_t cached_gpu_packed_version_ = 0;
  bool gpu_packed_inference_enabled_ = false;
  std::atomic<uint64_t> gpu_packed_dispatch_count_{0};
  bool training_mode_ = true;

  // T-MAC block-sparse heat map, lazily computed on first opt-in use.
  // Cleared on any operation that invalidates packed_weights.  See
  // include/lut_tmac.h.
  std::vector<uint8_t> cached_heat_map_;
};

} // namespace nsos

#endif // BITLINEAR_H
