// Mamba2 SSD full-sequence (non-streaming) CPU↔GPU parity.
//
// This complements `test_gpu_parity_mamba_stream.cpp` by exercising
// `Mamba2SSD::forward` with rank-3 input (batch, seq, d_model).  In this
// path the selective scan is dispatched as a single launch covering the
// whole sequence — historically `ssd_forward` always copied tensors to
// the CPU, so this test guards the GPU dispatch added in PR-Phase3 of
// the GPU optimization plan.
//
// Both layers are constructed independently and then weight-synchronized
// from the CPU layer, so the only delta between outputs is the kernel
// implementation, not random init.

#include "gpu_parity_common.h"
#include "mamba2.h"
#include "tensor.h"

#include <vector>

using nsos::Mamba2SSD;
using nsos::Parameter;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

// Copies every Parameter byte-for-byte from `src` to `dst`.  Both
// modules must have an identical parameter ordering (same constructor
// arguments).  We avoid `Module::load`/`save` here because we want to
// keep the test independent of pack serialization.
void mirror_parameters(const std::vector<Parameter*>& src,
                       const std::vector<Parameter*>& dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error("mirror_parameters: parameter count mismatch");
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (!src[i] || !dst[i]) continue;
    if (src[i]->data.size != dst[i]->data.size) {
      throw std::runtime_error("mirror_parameters: parameter size mismatch at index " +
                               std::to_string(i));
    }
    // Move to CPU to copy contents, then restore device on the destination.
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

} // namespace

int main() {
  return run_parity("mamba_scan", [] {
    constexpr int d_model = 32;
    constexpr int d_state = 16;
    constexpr int n_heads = 2;
    constexpr int batch = 1;
    constexpr int seq = 6;

    Mamba2SSD cpu_layer(d_model, d_state, n_heads);
    Mamba2SSD gpu_layer(d_model, d_state, n_heads);
    gpu_layer.to(Device::GPU);

    auto cpu_params = cpu_layer.parameters();
    auto gpu_params = gpu_layer.parameters();
    mirror_parameters(cpu_params, gpu_params);

    Tensor input = Tensor::random({batch, seq, d_model}, Device::CPU);

    const Tensor cpu_out = cpu_layer.forward(input, /*ctx=*/nullptr);
    const Tensor gpu_out =
        gpu_layer.forward(input.to(Device::GPU), /*ctx=*/nullptr).cpu();
    cuda_sync_or_throw("mamba_scan/forward");

    // Tolerance: SSM accumulates over `seq` timesteps, so per-element drift
    // can grow with seq_len.  3e-3 matches the streaming variant which
    // uses 8 steps; scaling proportionally we allow 3e-3 here too.
    assert_close(cpu_out, gpu_out, 3e-3f, "mamba_selective_scan_forward");
  });
}
