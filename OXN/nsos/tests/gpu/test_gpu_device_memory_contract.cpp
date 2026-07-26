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
  });
}
