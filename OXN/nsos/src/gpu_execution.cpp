#include "gpu_execution.h"
#include "gpu_backend.h"
#ifdef USE_CUDA
#include "cuda/device_buffer.h"
#endif
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace nsos::gpu {
namespace {
thread_local ExecutionContext* bound_context = nullptr;
std::array<std::atomic<std::uint64_t>, static_cast<unsigned>(DispatchPath::Count)> counters{};
#ifdef USE_CUDA
thread_local cudaStream_t bound_stream = nullptr;
thread_local bool has_bound_stream = false;
thread_local const int* bound_decode_position = nullptr;
void checked(cudaError_t status, const char* phase) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(phase) + ": " + cudaGetErrorString(status));
}
cudaStream_t default_stream() noexcept {
#ifdef NSOS_CUDA_PTDS
  return cudaStreamPerThread;
#else
  return nullptr;
#endif
}
#endif
std::size_t width(StorageType type) {
  switch (type) {
    case StorageType::Int8: case StorageType::Bytes: return 1;
    case StorageType::Int16: case StorageType::Float16: return 2;
    case StorageType::Float32: case StorageType::UInt32: return 4;
  }
  throw std::invalid_argument("Unknown GPU storage type");
}
}

struct ExecutionContext::Impl {
  std::recursive_mutex mutex;
  bool frozen = false;
#ifdef USE_CUDA
  struct Entry {
    cuda_detail::DeviceBuffer<unsigned char> buffer;
    StorageType type = StorageType::Bytes;
    std::size_t bytes = 0;
  };
  std::array<Entry, static_cast<unsigned>(WorkspaceSlot::Count)> entries;
  cudaStream_t stream = nullptr;
  cudaEvent_t enter_event = nullptr, leave_event = nullptr;
  int device = -1;
  void initialize() {
    if (stream) return;
    std::string error;
    if (!select_preferred_device(&device, &error))
      throw std::runtime_error("Execution context: " + error);
    try {
      checked(cudaStreamCreate(&stream), "create execution stream");
      checked(cudaEventCreate(&enter_event), "create entry fence");
      checked(cudaEventCreate(&leave_event), "create exit fence");
    } catch (...) {
      if (enter_event) (void)cudaEventDestroy(enter_event);
      if (leave_event) (void)cudaEventDestroy(leave_event);
      if (stream) (void)cudaStreamDestroy(stream);
      enter_event = leave_event = nullptr;
      stream = nullptr;
      throw;
    }
  }
  ~Impl() {
    if (device >= 0) {
      int previous = -1;
      (void)cudaGetDevice(&previous);
      if (cudaSetDevice(device) != cudaSuccess) return;
      if (stream) report_cleanup_status(cudaStreamSynchronize(stream), "execution drain");
      // Release on the owning device, before destroying its stream.
      for (auto& entry : entries) entry.buffer.release();
      if (enter_event) report_cleanup_status(cudaEventDestroy(enter_event), "entry fence");
      if (leave_event) report_cleanup_status(cudaEventDestroy(leave_event), "exit fence");
      if (stream) report_cleanup_status(cudaStreamDestroy(stream), "execution stream");
      if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
    }
  }
#endif
};

ExecutionContext::ExecutionContext() : impl_(std::make_unique<Impl>()) {}
void record_dispatch(DispatchPath path) noexcept {
  const auto index = static_cast<unsigned>(path);
  if (index < counters.size()) counters[index].fetch_add(1, std::memory_order_relaxed);
}
std::array<std::uint64_t, static_cast<unsigned>(DispatchPath::Count)> dispatch_counters() noexcept {
  std::array<std::uint64_t, static_cast<unsigned>(DispatchPath::Count)> result{};
  for (unsigned i = 0; i < result.size(); ++i) result[i] = counters[i].load(std::memory_order_relaxed);
  return result;
}
ExecutionContext::~ExecutionContext() = default;

void* ExecutionContext::reserve(WorkspaceSlot slot, StorageType type, std::size_t elements) {
  std::lock_guard lock(impl_->mutex);
  const auto index = static_cast<unsigned>(slot);
  if (index >= static_cast<unsigned>(WorkspaceSlot::Count))
    throw std::invalid_argument("Invalid GPU workspace slot");
  const auto element_bytes = width(type);
  if (elements > std::numeric_limits<std::size_t>::max() / element_bytes)
    throw std::overflow_error("GPU workspace size overflow");
#ifdef USE_CUDA
  auto& entry = impl_->entries[index];
  const auto bytes = elements * element_bytes;
  if (entry.bytes && entry.type != type)
    throw std::logic_error("GPU workspace dtype cannot change");
  if (impl_->frozen && bytes > entry.bytes)
    throw std::logic_error("GPU workspace growth during capture/replay");
  auto* result = entry.buffer.ensure(bytes);
  if (bytes && !result) throw std::bad_alloc();
  entry.type = type;
  entry.bytes = (std::max)(entry.bytes, bytes);
  return result;
#else
  (void)elements;
  throw std::runtime_error("GPU workspace requested in a CPU build");
#endif
}
std::size_t ExecutionContext::reserved_bytes() const noexcept {
  std::lock_guard lock(impl_->mutex);
  std::size_t bytes = 0;
#ifdef USE_CUDA
  for (const auto& entry : impl_->entries) bytes += entry.bytes;
#endif
  return bytes;
}
void ExecutionContext::freeze(bool enabled) noexcept {
  std::lock_guard lock(impl_->mutex);
  impl_->frozen = enabled;
}

struct ExecutionContext::Scope::State {
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext* previous;
#ifdef USE_CUDA
  cudaStream_t previous_stream;
  bool previous_bound;
  Impl* impl;
#endif
  explicit State(ExecutionContext& context)
      : lock(context.impl_->mutex), previous(bound_context)
#ifdef USE_CUDA
      , previous_stream(current_stream()), previous_bound(has_bound_stream), impl(context.impl_.get())
#endif
  {
#ifdef USE_CUDA
    impl->initialize();
    checked(cudaSetDevice(impl->device), "bind execution device");
    if (previous != &context) {
      checked(cudaEventRecord(impl->enter_event, previous_stream), "record producer fence");
      checked(cudaStreamWaitEvent(impl->stream, impl->enter_event, 0), "wait producer fence");
    }
    bound_stream = impl->stream;
    has_bound_stream = true;
#endif
    bound_context = &context;
  }
  ~State() {
#ifdef USE_CUDA
    if (previous != bound_context) {
      auto status = cudaEventRecord(impl->leave_event, impl->stream);
      if (status == cudaSuccess) status = cudaStreamWaitEvent(previous_stream, impl->leave_event, 0);
      if (status != cudaSuccess) {
        report_cleanup_status(status, "execution exit fence");
        report_cleanup_status(cudaDeviceSynchronize(), "execution emergency drain");
      }
    }
    bound_stream = previous_stream;
    has_bound_stream = previous_bound;
#endif
    bound_context = previous;
  }
};
ExecutionContext::Scope::Scope(ExecutionContext& context, bool enabled) {
  if (enabled) state_ = std::make_unique<State>(context);
}
ExecutionContext::Scope::~Scope() = default;
ExecutionContext& current_execution_context() {
  if (bound_context) return *bound_context;
  thread_local ExecutionContext fallback;
  return fallback;
}
#ifdef USE_CUDA
cudaStream_t current_stream() noexcept {
  return has_bound_stream ? bound_stream : default_stream();
}
const int* decode_position() noexcept { return bound_decode_position; }
void set_decode_position(const int* position) noexcept { bound_decode_position = position; }
#endif
}  // namespace nsos::gpu
