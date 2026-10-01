#pragma once
#include "tensor.h"
#include <memory>
#include <utility>
namespace nsos {
// FP16 row-major [batch, capacity, kv_width]. This is an inference storage
// format, never a Tensor pretending its float elements contain half values.
class GpuKvCache {
 public:
  GpuKvCache(int batch, int capacity, int width);
  ~GpuKvCache();
  GpuKvCache(const GpuKvCache&) = delete;
  GpuKvCache& operator=(const GpuKvCache&) = delete;
  static std::shared_ptr<GpuKvCache> from_float(const Tensor& keys, const Tensor& values,
      int batch, int capacity, int width, int live);
  std::shared_ptr<GpuKvCache> resize(int capacity, int live) const;
  std::shared_ptr<GpuKvCache> row(int index, int live) const;
  void copy_row_from(int destination, const GpuKvCache& source, int row, int live);
  std::pair<Tensor, Tensor> materialize(int live) const;
  void* keys() const noexcept;
  void* values() const noexcept;
  size_t bytes() const noexcept;
  int batch() const noexcept { return batch_; }
  int capacity() const noexcept { return capacity_; }
  int width() const noexcept { return width_; }
 private:
  int batch_, capacity_, width_;
  struct Storage;
  std::unique_ptr<Storage> storage_;
};
}
