#include "gpu_kv_cache.h"
#include "gpu_backend.h"
#ifdef USE_CUDA
#include "cuda/device_buffer.h"
#include "cuda/kernels.cuh"
#endif
#include <limits>
#include <stdexcept>
namespace nsos {
namespace {
size_t checked_elements(int batch, int capacity, int width) {
  if (batch <= 0 || capacity <= 0 || width <= 0 ||
      size_t(batch) > std::numeric_limits<size_t>::max() / size_t(capacity) / size_t(width) / 4)
    throw std::invalid_argument("Invalid compact KV geometry");
  return size_t(batch) * size_t(capacity) * size_t(width);
}
}
struct GpuKvCache::Storage {
#ifdef USE_CUDA
  cuda_detail::DeviceBuffer<uint16_t> keys, values;
#endif
};
GpuKvCache::GpuKvCache(int batch, int capacity, int width)
    : batch_(batch), capacity_(capacity), width_(width), storage_(std::make_unique<Storage>()) {
  const auto count = checked_elements(batch, capacity, width);
#ifdef USE_CUDA
  if (!storage_->keys.ensure(count) || !storage_->values.ensure(count)) throw std::bad_alloc();
#else
  (void)count;
  throw std::runtime_error("Compact GPU KV requires a GPU build");
#endif
}
GpuKvCache::~GpuKvCache() = default;
void* GpuKvCache::keys() const noexcept {
#ifdef USE_CUDA
  return storage_->keys.get();
#else
  return nullptr;
#endif
}
void* GpuKvCache::values() const noexcept {
#ifdef USE_CUDA
  return storage_->values.get();
#else
  return nullptr;
#endif
}
size_t GpuKvCache::bytes() const noexcept { return size_t(batch_) * capacity_ * width_ * 4; }
std::shared_ptr<GpuKvCache> GpuKvCache::from_float(const Tensor& keys, const Tensor& values,
    int batch, int capacity, int width, int live) {
  const auto count = checked_elements(batch, capacity, width);
  if (live < 0 || live > capacity || keys.get_device() != Device::GPU ||
      values.get_device() != Device::GPU ||
      size_t(keys.size) < count || size_t(values.size) < count)
    throw std::invalid_argument("Invalid FP32 source for compact KV");
  auto result = std::make_shared<GpuKvCache>(batch, capacity, width);
#ifdef USE_CUDA
  for (int b = 0; b < batch; ++b) {
    const auto offset = size_t(b) * capacity * width;
    launch_kv_half_convert(keys.raw_data() + offset, static_cast<uint16_t*>(result->keys()) + offset, size_t(live) * width, 1);
    launch_kv_half_convert(values.raw_data() + offset, static_cast<uint16_t*>(result->values()) + offset, size_t(live) * width, 1);
  }
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(std::string("Compact KV conversion: ") + cudaGetErrorString(status));
#endif
  return result;
}
void GpuKvCache::copy_row_from(int destination, const GpuKvCache& source, int row, int live) {
  if (destination < 0 || destination >= batch_ || row < 0 || row >= source.batch_ ||
      width_ != source.width_ || live < 0 || live > capacity_ || live > source.capacity_)
    throw std::invalid_argument("Invalid compact KV row copy");
#ifdef USE_CUDA
  const auto dst = size_t(destination) * capacity_ * width_;
  const auto src = size_t(row) * source.capacity_ * width_;
  for (const auto pair : {std::pair{keys(), source.keys()}, std::pair{values(), source.values()}}) {
    auto status = cudaMemcpyAsync(static_cast<uint16_t*>(pair.first) + dst,
        static_cast<const uint16_t*>(pair.second) + src, size_t(live) * width_ * 2,
        cudaMemcpyDeviceToDevice, gpu::current_stream());
    if (status != cudaSuccess) throw std::runtime_error(std::string("Compact KV copy: ") + cudaGetErrorString(status));
  }
#endif
}
std::shared_ptr<GpuKvCache> GpuKvCache::resize(int capacity, int live) const {
  auto result = std::make_shared<GpuKvCache>(batch_, capacity, width_);
  for (int b = 0; b < batch_; ++b) result->copy_row_from(b, *this, b, live);
  return result;
}
std::shared_ptr<GpuKvCache> GpuKvCache::row(int index, int live) const {
  auto result = std::make_shared<GpuKvCache>(1, capacity_, width_);
  result->copy_row_from(0, *this, index, live);
  return result;
}
std::pair<Tensor, Tensor> GpuKvCache::materialize(int live) const {
  if (live < 0 || live > capacity_) throw std::invalid_argument("Invalid compact KV length");
  // Zero padding is intentional: snapshots expose their full capacity.
  Tensor k({batch_, capacity_, width_}, Device::GPU), v({batch_, capacity_, width_}, Device::GPU);
#ifdef USE_CUDA
  for (int b = 0; b < batch_; ++b) {
    const auto offset = size_t(b) * capacity_ * width_;
    launch_kv_half_convert(static_cast<const uint16_t*>(keys()) + offset, k.raw_data() + offset, size_t(live) * width_, 0);
    launch_kv_half_convert(static_cast<const uint16_t*>(values()) + offset, v.raw_data() + offset, size_t(live) * width_, 0);
  }
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(std::string("Compact KV materialization: ") + cudaGetErrorString(status));
#endif
  return {std::move(k), std::move(v)};
}
}
