// KAN (Kolmogorov-Arnold) layer CPU↔GPU parity.
//
// Guards the CUDA RBF-basis kernels added in Phase 2 (src/cuda/kan_kernels.cu):
// historically BitFastKANLayer::compute_basis and the backward RBF derivative
// ran host loops over `.data()`, so the layer could not run on a GPU tensor at
// all.  This test constructs CPU and GPU layers, weight-synchronises them, and
// checks forward output, grad_input, and every parameter gradient match.

#include "gpu_parity_common.h"
#include "kan.h"
#include "tensor.h"

#include <vector>

using nsos::BitFastKANLayer;
using nsos::Device;
using nsos::Parameter;
using nsos::Tensor;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

// Byte-for-byte copy of every Parameter from `src` to `dst` (same ordering,
// since both layers are built with identical constructor arguments).
void mirror_parameters(const std::vector<Parameter*>& src,
                       const std::vector<Parameter*>& dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error("mirror_parameters: parameter count mismatch");
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (!src[i] || !dst[i]) continue;
    if (src[i]->data.size != dst[i]->data.size) {
      throw std::runtime_error("mirror_parameters: size mismatch at index " +
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
  return run_parity("kan", [] {
    constexpr int in_features = 24;
    constexpr int out_features = 32;
    constexpr int grid = 5;
    constexpr int batch = 2;
    constexpr int seq = 4;

    BitFastKANLayer cpu_layer(in_features, out_features, grid);
    BitFastKANLayer gpu_layer(in_features, out_features, grid);
    gpu_layer.to(Device::GPU);
    mirror_parameters(cpu_layer.parameters(), gpu_layer.parameters());

    Tensor input = Tensor::random({batch, seq, in_features}, Device::CPU);

    // Forward parity.
    const Tensor cpu_out = cpu_layer.forward(input);
    const Tensor gpu_out = gpu_layer.forward(input.to(Device::GPU)).cpu();
    cuda_sync_or_throw("kan/forward");
    assert_close(cpu_out, gpu_out, 2e-3f, "kan_forward");

    // Backward parity: grad_input plus every parameter gradient.
    Tensor grad = Tensor::random({batch, seq, out_features}, Device::CPU);
    const Tensor cpu_gi = cpu_layer.backward(grad);
    const Tensor gpu_gi = gpu_layer.backward(grad.to(Device::GPU)).cpu();
    cuda_sync_or_throw("kan/backward");
    assert_close(cpu_gi, gpu_gi, 2e-3f, "kan_grad_input");

    const auto cpu_params = cpu_layer.parameters();
    const auto gpu_params = gpu_layer.parameters();
    for (size_t i = 0; i < cpu_params.size(); ++i) {
      assert_close(cpu_params[i]->grad, gpu_params[i]->grad, 2e-3f,
                   "kan_param_grad");
    }
  });
}
