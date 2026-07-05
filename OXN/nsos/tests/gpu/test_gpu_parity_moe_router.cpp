// MoERouter top-k mask + load accumulation CPU↔GPU parity.
//
// Exercises the GPU fast-path added in jamba.cpp::MoERouter::forward
// (Phase 4 of the GPU optimization plan).  The fast-path replaced a
// per-token GPU→CPU→GPU round-trip with two on-device kernels:
//   * launch_moe_topk_mask_kernel    (in-place mask + renormalize)
//   * launch_moe_load_accumulate_kernel  (per-expert load sum)
//
// We construct two routers with identical bit-for-bit gate weights, run
// the same input on CPU and GPU, and assert:
//   * post-mask weights agree per row
//   * each row sums to ~1 after renormalization
//   * exactly top_k entries per row are non-zero
//   * expert_loads agree across devices

#include "gpu_parity_common.h"
#include "jamba.h"
#include "tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using nsos::MoERouter;
using nsos::Parameter;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

// Mirror parameters CPU → GPU.  Kept local to this TU to avoid a
// dependency on an internal helper that may move during refactors.
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

// Counts non-zero entries per row.  The CPU and GPU paths must produce
// exactly `top_k` non-zero entries per row after masking.
int count_nonzero_in_row(const float* row, int num_experts) {
  int n = 0;
  for (int e = 0; e < num_experts; ++e) {
    if (row[e] != 0.0f) ++n;
  }
  return n;
}

}  // namespace

int main() {
  return run_parity("moe_router", [] {
    constexpr int d_model = 16;
    constexpr int num_experts = 8;
    constexpr int top_k = 2;
    constexpr int rows = 5;

    MoERouter cpu_router(d_model, num_experts, top_k);
    MoERouter gpu_router(d_model, num_experts, top_k);
    gpu_router.to(Device::GPU);

    auto cpu_params = cpu_router.parameters();
    auto gpu_params = gpu_router.parameters();
    mirror_parameters(cpu_params, gpu_params);

    Tensor input = Tensor::random({rows, d_model}, Device::CPU);

    auto [_cpu_logits, cpu_weights] = cpu_router.forward(input);
    auto [_gpu_logits, gpu_weights_dev] =
        gpu_router.forward(input.to(Device::GPU));
    cuda_sync_or_throw("moe_router/forward");

    // Pull GPU weights to CPU for byte-level comparison.
    Tensor gpu_weights = gpu_weights_dev.cpu();

    // Tolerance:
    //   * The mask is identical across devices when no ties exist.
    //   * Renormalization adds one fp32 division per row.
    //   * 1e-4 is comfortably above the worst-case fp32 drift here.
    assert_close(cpu_weights, gpu_weights, 1e-4f, "moe_router_weights");

    // Structural check: exactly top_k non-zero entries per row, and
    // each row sums to 1.
    const float* cpu_w = cpu_weights.data();
    const float* gpu_w = gpu_weights.data();
    for (int r = 0; r < rows; ++r) {
      const int cpu_nnz = count_nonzero_in_row(cpu_w + r * num_experts,
                                               num_experts);
      const int gpu_nnz = count_nonzero_in_row(gpu_w + r * num_experts,
                                               num_experts);
      if (cpu_nnz != top_k || gpu_nnz != top_k) {
        throw std::runtime_error(
            "row " + std::to_string(r) + " has cpu_nnz=" +
            std::to_string(cpu_nnz) + " gpu_nnz=" + std::to_string(gpu_nnz) +
            " expected=" + std::to_string(top_k));
      }
      float cpu_sum = 0.0f, gpu_sum = 0.0f;
      for (int e = 0; e < num_experts; ++e) {
        cpu_sum += cpu_w[r * num_experts + e];
        gpu_sum += gpu_w[r * num_experts + e];
      }
      if (std::abs(cpu_sum - 1.0f) > 1e-4f) {
        throw std::runtime_error("cpu row " + std::to_string(r) +
                                 " sum=" + std::to_string(cpu_sum));
      }
      if (std::abs(gpu_sum - 1.0f) > 1e-4f) {
        throw std::runtime_error("gpu row " + std::to_string(r) +
                                 " sum=" + std::to_string(gpu_sum));
      }
    }

    // expert_loads parity.  Each device computes loads from its own
    // weights; both views must agree because the weights agreed above.
    const auto& cpu_loads = cpu_router.expert_loads;
    const auto& gpu_loads = gpu_router.expert_loads;
    if (cpu_loads.size() != gpu_loads.size()) {
      throw std::runtime_error("expert_loads size mismatch");
    }
    for (size_t e = 0; e < cpu_loads.size(); ++e) {
      const float diff = std::abs(cpu_loads[e] - gpu_loads[e]);
      if (diff > 1e-4f) {
        throw std::runtime_error(
            "expert_loads mismatch at e=" + std::to_string(e) +
            " cpu=" + std::to_string(cpu_loads[e]) +
            " gpu=" + std::to_string(gpu_loads[e]) +
            " |delta|=" + std::to_string(diff));
      }
    }

    // Regression for the former fixed local arrays[64] in the CUDA top-k
    // kernel.  65 experts used to let eff_k index beyond those arrays.
    constexpr int wide_experts = 65;
    constexpr int wide_top_k = 3;
    MoERouter wide_cpu(d_model, wide_experts, wide_top_k);
    MoERouter wide_gpu(d_model, wide_experts, wide_top_k);
    wide_gpu.to(Device::GPU);
    mirror_parameters(wide_cpu.parameters(), wide_gpu.parameters());
    Tensor wide_input = Tensor::random({2, d_model}, Device::CPU);
    auto [_wide_cpu_logits, wide_cpu_weights] = wide_cpu.forward(wide_input);
    auto [_wide_gpu_logits, wide_gpu_weights_device] =
        wide_gpu.forward(wide_input.to(Device::GPU));
    cuda_sync_or_throw("moe_router/forward_65_experts");
    Tensor wide_gpu_weights = wide_gpu_weights_device.cpu();
    assert_close(wide_cpu_weights, wide_gpu_weights, 1e-4f,
                 "moe_router_weights_65_experts");
    for (int row = 0; row < 2; ++row) {
      if (count_nonzero_in_row(
              wide_gpu_weights.data() + row * wide_experts,
              wide_experts) != wide_top_k) {
        throw std::runtime_error(
            "65-expert CUDA router did not preserve top-k cardinality");
      }
    }
  });
}
