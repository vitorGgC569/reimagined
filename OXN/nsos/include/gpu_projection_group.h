#pragma once
#include "tensor.h"
#include "gpu_linear_view.h"
#ifdef USE_CUDA
#include "cuda/device_buffer.h"
#endif
#include <vector>
namespace nsos {
class BitLinear;
// Inference-only, row-major decode projections. Empty result requests the
// established implementation; no fallback is allowed after a launch failure.
class GpuProjectionGroup {
 public:
  std::vector<Tensor> forward(const Tensor& input, const std::vector<BitLinear*>& layers);
 private:
  std::vector<GpuLinearView> host_views_;
#ifdef USE_CUDA
  cuda_detail::DeviceBuffer<GpuLinearView> device_views_;
#endif
};
}
