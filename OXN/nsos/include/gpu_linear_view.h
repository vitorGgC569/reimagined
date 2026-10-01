#pragma once
#include <cstdint>
#include <type_traits>

namespace nsos {
// Borrowed immutable device metadata, not a checkpoint format. Owner layers
// retain the storage and must outlive every enqueued use of the descriptor.
struct GpuLinearView {
  const float* weight = nullptr;
  const uint32_t* packed = nullptr;
  const float* magnitude = nullptr;
  const float* bias = nullptr;
  int inputs = 0;
  int outputs = 0;
  int activation_bits = 0;  // zero means unquantized activations
  int rms_input = 0;
  float weight_scale = 1.0f;
  bool operator==(const GpuLinearView& other) const noexcept {
    return weight == other.weight && packed == other.packed &&
        magnitude == other.magnitude && bias == other.bias &&
        inputs == other.inputs && outputs == other.outputs &&
        activation_bits == other.activation_bits && rms_input == other.rms_input &&
        weight_scale == other.weight_scale;
  }
};
static_assert(std::is_trivially_copyable_v<GpuLinearView>);
// Training descriptors retain separate latent and QAT effective weights.
struct GpuMoeTrainingLinearView {
  const float* weight = nullptr;
  const float* latent_weight = nullptr;
  const float* qat_scale = nullptr;
  const float* magnitude = nullptr;
  const float* bias = nullptr;
  int inputs = 0;
  int outputs = 0;
  int rms_input = 0;
  int activation_bits = 0;
  bool operator==(const GpuMoeTrainingLinearView& other) const noexcept {
    return weight == other.weight && latent_weight == other.latent_weight &&
        qat_scale == other.qat_scale && magnitude == other.magnitude && bias == other.bias &&
        inputs == other.inputs && outputs == other.outputs &&
        rms_input == other.rms_input && activation_bits == other.activation_bits;
  }
};
static_assert(std::is_trivially_copyable_v<GpuMoeTrainingLinearView>);
// Borrowed gradient destinations. Bits 0/1/2 select addition (versus first
// contribution overwrite) for weight/bias/magnitude independently. Never
// aliases grouped scratch; Parameters own every destination across steps.
struct GpuMoeGradientView {
  float* weight = nullptr;
  float* bias = nullptr;
  float* magnitude = nullptr;
  unsigned add_mask = 0;
  bool operator==(const GpuMoeGradientView& other) const noexcept {
    return weight == other.weight && bias == other.bias &&
        magnitude == other.magnitude && add_mask == other.add_mask;
  }
};
static_assert(std::is_trivially_copyable_v<GpuMoeGradientView>);
}  // namespace nsos
