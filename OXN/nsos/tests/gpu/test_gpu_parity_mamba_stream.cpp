// Mamba2 SSD streaming CPU↔GPU parity.
//
// This is the test that historically appeared to segfault on CUDA 12.9
// inside the monolithic test_gpu_parity binary.  After decomposition we
// observed that the failure mode in current CUDA 12.9 is actually a
// numerical mismatch (not a segfault), and that mismatch was caused by
// a long-standing test-side bug: the original test created two
// independently random-initialized Mamba2SSD instances and compared
// their outputs.  With independent random weights the layers are
// strictly different functions, so the parity assertion was checking
// nothing meaningful — it only "passed" when the two random rolls
// happened to land within tolerance.
//
// Fix: mirror parameters from the CPU layer to the GPU layer so the
// only delta between outputs is the kernel implementation.  This
// matches the pattern used by test_gpu_parity_mamba_scan and
// test_gpu_parity_moe_router.

#include "gpu_parity_common.h"
#include "mamba2.h"
#include "tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using nsos::Mamba2SSD;
using nsos::Parameter;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

// Copy weight bytes CPU → GPU (or vice versa) parameter-by-parameter.
// Both modules must have identical parameter ordering, which is
// guaranteed when they are constructed with the same arguments.
void mirror_parameters(const std::vector<Parameter*>& src,
                       const std::vector<Parameter*>& dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error(
        "mirror_parameters: parameter count mismatch (src=" +
        std::to_string(src.size()) + " dst=" + std::to_string(dst.size()) +
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
  return run_parity("mamba_stream", [] {
    Mamba2SSD cpu_layer(/*d_model=*/32, /*d_state=*/16, /*n_heads=*/2);
    Mamba2SSD gpu_layer(/*d_model=*/32, /*d_state=*/16, /*n_heads=*/2);
    gpu_layer.to(Device::GPU);

    // Mirror weights so the two layers represent the same function.
    auto cpu_params = cpu_layer.parameters();
    auto gpu_params = gpu_layer.parameters();
    mirror_parameters(cpu_params, gpu_params);

    cpu_layer.set_streaming_mode(true);
    gpu_layer.set_streaming_mode(true);

    Tensor final_cpu;
    Tensor final_gpu;
    for (int step = 0; step < 8; ++step) {
      Tensor token = Tensor::random({1, 32}, Device::CPU);
      final_cpu = cpu_layer.forward(token, /*ctx=*/nullptr);
      final_gpu = gpu_layer.forward(token.to(Device::GPU), /*ctx=*/nullptr).cpu();
      cuda_sync_or_throw("mamba_stream/step");
    }
    // Tolerance reflects accumulated fp32 drift over 8 sequential
    // updates plus the extra rounding from the GPU shared-mem path.
    // With weights mirrored, anything looser than ~5e-3 would mask a
    // real kernel divergence; anything tighter triggers false positives
    // on accumulated rounding.
    assert_close(final_cpu, final_gpu, 5e-3f, "mamba_streaming");
  });
}
