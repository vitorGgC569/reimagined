// BitLinear (1.58-bit ternary GEMM) CPU↔GPU parity.
//
// On the GPU side this exercises `bitnet_gemm_kernel` in
// `src/cuda/kernels.cu`, which uses `__dp4a` for 4-element INT8 dot
// products on Pascal (sm_61) and newer.  Tolerance is wider than the
// fp32 elementwise tests because BitLinear quantizes activations and
// packs weights into 2-bit codes — small per-element drift is expected
// and only systemic mismatch should fail this test.

#include "bitlinear.h"
#include "gpu_parity_common.h"
#include "tensor.h"

using nsos::BitLinear;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

int main() {
  return run_parity("bitlinear", [] {
    BitLinear cpu_layer(32, 24, /*bias=*/true);
    BitLinear gpu_layer(32, 24, /*bias=*/true);
    gpu_layer.to(Device::GPU);

    // Force the GPU layer to share weights with the CPU layer so the
    // only delta is kernel implementation, not random init.
    gpu_layer.weight.data.copy_from(cpu_layer.weight.data.to(Device::GPU));

    Tensor x = Tensor::random({4, 32}, Device::CPU);
    const Tensor cpu_out = cpu_layer.forward(x);
    const Tensor gpu_out = gpu_layer.forward(x.to(Device::GPU)).cpu();
    cuda_sync_or_throw("bitlinear/forward");
    assert_close(cpu_out, gpu_out, 2e-3f, "bitlinear");
  });
}
