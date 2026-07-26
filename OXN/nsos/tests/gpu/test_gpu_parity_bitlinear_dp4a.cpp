// BitLinear GPU __dp4a inference fast-path parity (Phase 5b).
//
// Validates that `BitLinear::forward` with `set_gpu_packed_inference(true)`
// produces an output close to the CPU packed path on the same weights
// and inputs.  This exercises the integration in src/bitlinear.cpp that
// composes:
//   1. quantize_activations_bitnet_kernel  (GPU)
//   2. launch_bitnet_gemm  (GPU __dp4a)
//   3. bitnet_apply_act_scales_kernel  (GPU)
//   4. mul magnitude + add bias  (GPU element-wise)
//
// We compare against a CPU BitLinear running its packed fast path
// (set_reference_path(false)), since both sides use the same packed
// 1.58-bit weights and ternary activation quant.  Tolerance accounts
// for fp32 reduction-order and 1/2 LSB rounding differences across
// devices.

#include "bitlinear.h"
#include "gpu_parity_common.h"
#include "tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using nsos::BitLinear;
using nsos::Parameter;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

void mirror_parameters(const std::vector<Parameter*>& src,
                       const std::vector<Parameter*>& dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error(
        "mirror_parameters: parameter count mismatch (cpu=" +
        std::to_string(src.size()) + " gpu=" + std::to_string(dst.size()) +
        ")");
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (!src[i] || !dst[i]) continue;
    if (src[i]->data.size != dst[i]->data.size) {
      throw std::runtime_error(
          "mirror_parameters: parameter size mismatch at index " +
          std::to_string(i));
    }
    Tensor cpu_src = src[i]->data.get_device() == Device::GPU
                         ? src[i]->data.cpu()
                         : src[i]->data;
    if (dst[i]->data.get_device() == Device::GPU) {
      dst[i]->data.copy_from(cpu_src.to(Device::GPU));
    } else {
      dst[i]->data.copy_from(cpu_src);
    }
  }
}

}  // namespace

int main() {
  return run_parity("bitlinear_dp4a", [] {
    constexpr int in_features = 64;   // multiple of 16 (kernel constraint)
    constexpr int out_features = 32;
    constexpr int batch = 4;

    BitLinear cpu_layer(in_features, out_features, /*bias=*/true);
    BitLinear gpu_layer(in_features, out_features, /*bias=*/true);

    // Both layers must share identical weights to make parity meaningful.
    auto cpu_params = cpu_layer.parameters();
    auto gpu_params = gpu_layer.parameters();
    mirror_parameters(cpu_params, gpu_params);

    // Move GPU layer to GPU and switch precision to 1.58-bit / ternary.
    gpu_layer.to(Device::GPU);
    cpu_layer.set_precision_mode(2);
    gpu_layer.set_precision_mode(2);

    // CPU side: use the packed fast path (not the float reference path)
    // so the comparison is "packed CPU vs packed GPU __dp4a" — same
    // numerical contract on both sides.
    cpu_layer.set_reference_path(false);
    gpu_layer.set_reference_path(false);
    cpu_layer.set_training_mode(false);
    gpu_layer.set_training_mode(false);

    // Force packed-weight materialization on both sides so the GPU
    // dispatch sees a valid packed buffer.
    cpu_layer.repack_weights();
    gpu_layer.repack_weights();

    // Engage the Phase 5b __dp4a path on the GPU layer.  Without this
    // flag, gpu_layer.forward would still use float matmul.
    gpu_layer.set_gpu_packed_inference(true);

    // Deterministic input pattern.
    Tensor x_cpu({batch, in_features}, Device::CPU);
    for (int i = 0; i < x_cpu.size; ++i) {
      const float t = static_cast<float>(i) * 0.011f;
      x_cpu.data()[i] = std::sin(t) * 0.6f - std::cos(t * 0.7f) * 0.4f;
    }

    Tensor y_cpu = cpu_layer.forward(x_cpu);
    Tensor y_gpu_dev = gpu_layer.forward(x_cpu.to(Device::GPU));
    cuda_sync_or_throw("bitlinear_dp4a/forward");
    Tensor y_gpu = y_gpu_dev.cpu();

    // Tolerance budget:
    //   * Activation quant introduces ~scale/2 drift per element
    //   * Reduction order over 64 inputs differs (CPU sequential vs
    //     GPU parallel)
    //   * Weight magnitude scaling adds one fp32 multiply
    // 1e-2 absolute is generous but still catches systemic mismatches
    // (wrong scale, wrong sign, transposed weights).
    const float tol = 1e-2f;
    int worst_idx = -1;
    float worst_diff = 0.0f;
    for (int i = 0; i < y_cpu.size; ++i) {
      const float diff = std::abs(y_cpu.data()[i] - y_gpu.data()[i]);
      if (diff > worst_diff) {
        worst_diff = diff;
        worst_idx = i;
      }
    }
    if (worst_diff > tol) {
      throw std::runtime_error(
          "bitlinear_dp4a parity failed: worst |delta|=" +
          std::to_string(worst_diff) + " at idx=" +
          std::to_string(worst_idx) + " cpu=" +
          std::to_string(y_cpu.data()[worst_idx]) + " gpu=" +
          std::to_string(y_gpu.data()[worst_idx]) + " tol=" +
          std::to_string(tol));
    }
  });
}
