#include "gpu_projection_group.h"
#include "bitlinear.h"
#include "gpu_execution.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "cuda/kernels.cuh"
#endif
namespace nsos {
std::vector<Tensor> GpuProjectionGroup::forward(const Tensor& input, const std::vector<BitLinear*>& layers) {
#ifdef USE_CUDA
  const char* flag = std::getenv("NSOS_GPU_GROUPED_PROJECTIONS");
  if ((flag && flag[0] == '0') || input.get_device() != Device::GPU || layers.empty()) return {};
  std::vector<GpuLinearView> views(layers.size());
  int outputs = 0;
  for (size_t i = 0; i < layers.size(); ++i) {
    if (!layers[i] || !layers[i]->prepare_gpu_decode_view(views[i]) ||
        input.size != views[i].inputs) return {};
    outputs = (std::max)(outputs, views[i].outputs);
  }
  if (views != host_views_) {
    auto* ptr = device_views_.ensure(views.size());
    if (!ptr) throw std::bad_alloc();
    auto status = cudaMemcpy(ptr, views.data(), views.size() * sizeof(GpuLinearView), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) throw std::runtime_error(std::string("Grouped descriptor upload: ") + cudaGetErrorString(status));
    host_views_ = views;
  }
  if (views.size() > size_t(std::numeric_limits<int>::max() / outputs))
    throw std::overflow_error("Grouped projection output size overflow");
  auto& workspace = gpu::current_execution_context();
  auto* prepared = static_cast<float*>(workspace.reserve(gpu::WorkspaceSlot::GroupedPrepared,
      gpu::StorageType::Float32, views.size() * size_t(input.size)));
  auto* scales = static_cast<float*>(workspace.reserve(gpu::WorkspaceSlot::GroupedScales,
      gpu::StorageType::Float32, views.size()));
  Tensor storage = Tensor::uninitialized({int(views.size()), outputs}, Device::GPU);
  launch_grouped_decode_projections(device_views_.get(), input.raw_data(), prepared,
                                    scales, storage.raw_data(), int(views.size()), outputs);
  gpu::record_dispatch(gpu::DispatchPath::GroupedProjection);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(std::string("Grouped projection: ") + cudaGetErrorString(status));
  std::vector<Tensor> result;
  result.reserve(views.size());
  for (size_t i = 0; i < views.size(); ++i)
    result.push_back(storage.storage_view(i * size_t(outputs), {1, views[i].outputs}));
  return result;
#else
  (void)input; (void)layers;
  return {};
#endif
}
}
