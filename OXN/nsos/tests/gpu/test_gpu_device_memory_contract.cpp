// Native-device-memory safety contract.
//
// Strict GPU execution allocates Tensor storage with cudaMalloc.  Such storage
// must never be exposed as a host-dereferenceable pointer through data(); CUDA
// APIs use raw_data(), and explicit host inspection goes through cpu(). The
// same rule is provenance-based for externally adopted CUDA storage.

#include "gpu_parity_common.h"
#include "tensor.h"

#include <memory>
#include <stdexcept>
#include <vector>

using nsos::Device;
using nsos::Tensor;
using nsos::set_strict_gpu_execution;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::run_parity;

int main() {
  return run_parity("device_memory_contract", [] {
    set_strict_gpu_execution(true);
    Tensor device_tensor({16}, Device::GPU, 2.0f);
    if (device_tensor.raw_data() == nullptr) {
      throw std::runtime_error("cudaMalloc-backed Tensor has a null raw pointer");
    }

    bool host_access_rejected = false;
    try {
      (void)device_tensor.data();
    } catch (const std::runtime_error& error) {
      host_access_rejected =
          std::string(error.what()).find("cudaMalloc-backed") !=
          std::string::npos;
    }
    if (!host_access_rejected) {
      throw std::runtime_error(
          "data() exposed cudaMalloc-backed storage to host code");
    }
    if (device_tensor.is_host_accessible()) {
      throw std::runtime_error(
          "cudaMalloc-backed Tensor reported host-accessible provenance");
    }

    const nsos::PoolStats before_external_ownership =
        nsos::pool_stats();
    auto cuda_free = [](float* pointer) {
      if (pointer != nullptr) {
        (void)cudaFree(pointer);
      }
    };
    float* external_device = nullptr;
    if (cudaMalloc(&external_device, 4 * sizeof(float)) != cudaSuccess) {
      throw std::runtime_error("external cudaMalloc failed");
    }
    std::unique_ptr<float, decltype(cuda_free)> external_device_guard(
        external_device, cuda_free);
    {
      Tensor adopted_device =
          Tensor::from_blob(external_device_guard.get(), {4}, Device::GPU,
                            true);
      (void)external_device_guard.release();
      if (adopted_device.is_host_accessible()) {
        throw std::runtime_error(
            "adopted cudaMalloc storage lost device-only provenance");
      }
      bool adopted_host_access_rejected = false;
      try {
        (void)adopted_device.data();
      } catch (const std::runtime_error& error) {
        adopted_host_access_rejected =
            std::string(error.what()).find("cudaMalloc-backed") !=
            std::string::npos;
      }
      if (!adopted_host_access_rejected) {
        throw std::runtime_error(
            "data() exposed adopted cudaMalloc storage to host code");
      }
    }

    float* external_managed = nullptr;
    if (cudaMallocManaged(&external_managed, 4 * sizeof(float)) != cudaSuccess) {
      throw std::runtime_error("external cudaMallocManaged failed");
    }
    std::unique_ptr<float, decltype(cuda_free)> external_managed_guard(
        external_managed, cuda_free);
    {
      Tensor adopted_managed =
          Tensor::from_blob(external_managed_guard.get(), {4}, Device::GPU,
                            true);
      (void)external_managed_guard.release();
      if (!adopted_managed.is_host_accessible()) {
        throw std::runtime_error(
            "cudaMallocManaged storage lost host-accessible provenance");
      }
      adopted_managed.data()[0] = 7.0f;
      if (adopted_managed.cpu().data()[0] != 7.0f) {
        throw std::runtime_error(
            "managed external storage failed explicit CPU inspection");
      }
    }
    const nsos::PoolStats after_external_ownership =
        nsos::pool_stats();
    if (
        after_external_ownership.unknown_deallocation_attempts !=
        before_external_ownership.unknown_deallocation_attempts) {
      throw std::runtime_error(
          "externally owned GPU storage was misrouted through the pool");
    }

    float host_storage[4] = {};
    bool mislabeled_host_rejected = false;
    try {
      (void)Tensor::from_blob(host_storage, {4}, Device::GPU, false);
    } catch (const std::invalid_argument&) {
      mislabeled_host_rejected = true;
    }
    if (!mislabeled_host_rejected) {
      throw std::runtime_error(
          "from_blob accepted host storage mislabeled as Device::GPU");
    }

    Tensor doubled_gpu = device_tensor.add(device_tensor);
    Tensor expected({16}, Device::CPU, 4.0f);
    assert_close(doubled_gpu.cpu(), expected, 1e-6f,
                 "device_memory_explicit_cpu_copy");

#ifdef NSOS_ENABLE_TEST_HOOKS
    if (!nsos::pool_stats().pool_enabled) {
      throw std::runtime_error(
          "release-failure recovery gate requires the GPU pool");
    }
    nsos::release_cached_memory();
    const nsos::PoolStats before_failure = nsos::pool_stats();
    {
      Tensor cached_then_faulted({64}, Device::GPU, 3.0f);
    }
    nsos::set_gpu_pool_release_failure_countdown(0);
    nsos::release_cached_memory();
    nsos::set_gpu_pool_release_failure_countdown(-1);
    const nsos::PoolStats retained = nsos::pool_stats();
    if (
        retained.release_failures != before_failure.release_failures + 1 ||
        retained.retained_release_blocks !=
            before_failure.retained_release_blocks + 1 ||
        retained.retained_release_bytes <=
            before_failure.retained_release_bytes) {
      throw std::runtime_error(
          "pool release failure was not retained and reported");
    }
    nsos::release_cached_memory();
    const nsos::PoolStats recovered = nsos::pool_stats();
    if (
        recovered.retained_release_blocks !=
            before_failure.retained_release_blocks ||
        recovered.retained_release_bytes !=
            before_failure.retained_release_bytes) {
      throw std::runtime_error(
          "pool could not recover a retained driver release");
    }

    // Exercise the ownership registry's fail-closed duplicate-deallocation
    // path. The first explicit deleter returns the block to the cache; resetting
    // the real shared owner must be rejected without inserting the address a
    // second time.
    nsos::release_cached_memory();
    const nsos::PoolStats before_duplicate = nsos::pool_stats();
    Tensor duplicate_probe({257}, Device::GPU, 1.0f);
    float* duplicate_pointer = duplicate_probe.raw_data();
    nsos::TensorDeleter(Device::GPU, true)(duplicate_pointer);
    duplicate_probe.data_ptr.reset();
    const nsos::PoolStats after_duplicate = nsos::pool_stats();
    if (after_duplicate.unknown_deallocation_attempts !=
            before_duplicate.unknown_deallocation_attempts + 1 ||
        after_duplicate.cached_blocks != before_duplicate.cached_blocks + 1) {
      throw std::runtime_error(
          "pool duplicate deallocation guard corrupted cache accounting");
    }

    // Fixed allocation shapes must reach a stable reserved footprint. Syncing
    // before each scope ends makes this a pool-accounting test, not a scheduler
    // timing test for still-pending stream events.
    nsos::release_cached_memory();
    auto allocation_cycle = [] {
      std::vector<Tensor> tensors;
      tensors.reserve(12);
      for (int index = 1; index <= 12; ++index) {
        tensors.emplace_back(
            std::vector<int>{index * 257}, Device::GPU, 0.25f);
      }
      nsos::gpu_parity_test::cuda_sync_or_throw(
          "device_memory_contract/pool_growth_cycle_before_release");
      tensors.clear();
      nsos::gpu_parity_test::cuda_sync_or_throw(
          "device_memory_contract/pool_growth_cycle_after_release");
    };
    allocation_cycle();
    const nsos::PoolStats steady_pool = nsos::pool_stats();
    for (int cycle = 0; cycle < 32; ++cycle) {
      allocation_cycle();
    }
    const nsos::PoolStats repeated_pool = nsos::pool_stats();
    if (repeated_pool.reserved_bytes != steady_pool.reserved_bytes ||
        repeated_pool.cached_blocks != steady_pool.cached_blocks ||
        repeated_pool.live_blocks != steady_pool.live_blocks) {
      throw std::runtime_error(
          "GPU pool grew under a fixed repeated allocation workload");
    }
    nsos::release_cached_memory();
#endif
  });
}
