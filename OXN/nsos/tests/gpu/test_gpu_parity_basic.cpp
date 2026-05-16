// Tensor-add / matmul / RMSNorm CPU↔GPU parity.  These three are the
// smallest end-to-end checks of the CUDA elementwise, tiled-matmul and
// reduction kernels respectively.  Kept together because they share
// the same trivial fixture (a 2×3 / 3×2 pair of constant tensors) and
// each runs in milliseconds; splitting further would obscure the
// "basic kernels still work" signal.

#include "gpu_parity_common.h"
#include "tensor.h"

using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

int main() {
  return run_parity("basic", [] {
    // Deterministic operands.  i+1 in [1..6] for `a`, i-2 in [-2..3] for `b`.
    Tensor a({2, 3}, Device::CPU);
    Tensor b({3, 2}, Device::CPU);
    for (int i = 0; i < a.size; ++i) a.data()[i] = static_cast<float>(i + 1);
    for (int i = 0; i < b.size; ++i) b.data()[i] = static_cast<float>(i - 2);

    // tensor_add: A + A on both devices.
    {
      const Tensor cpu_add = a.add(a);
      const Tensor gpu_add = a.to(Device::GPU).add(a.to(Device::GPU)).cpu();
      cuda_sync_or_throw("basic/tensor_add");
      assert_close(cpu_add, gpu_add, 1e-4f, "tensor_add");
    }

    // matmul: 2x3 · 3x2 → 2x2 on both devices.
    {
      const Tensor cpu_mm = a.matmul(b);
      const Tensor gpu_mm = a.to(Device::GPU).matmul(b.to(Device::GPU)).cpu();
      cuda_sync_or_throw("basic/matmul");
      assert_close(cpu_mm, gpu_mm, 1e-4f, "matmul");
    }

    // rmsnorm: row-wise normalization on a 2x3 tensor.  RMSNorm allocates
    // shared memory and uses block-wide reduction on the GPU; the CPU
    // path is a straight scalar loop, so any divergence here points at
    // the warp/block reduction kernel.
    {
      const Tensor cpu_norm = a.rmsnorm();
      const Tensor gpu_norm = a.to(Device::GPU).rmsnorm().cpu();
      cuda_sync_or_throw("basic/rmsnorm");
      assert_close(cpu_norm, gpu_norm, 1e-4f, "rmsnorm");
    }
  });
}
