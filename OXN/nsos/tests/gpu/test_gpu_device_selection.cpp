#include "gpu_parity_common.h"
#include "cuda/device_buffer.h"
#include "gpu_backend.h"
#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using nsos::Device;
using nsos::Tensor;
using nsos::gpu_parity_test::run_parity;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  return run_parity("device_selection", [] {
    int selected = -1;
    std::string selection_error;
    require(
        nsos::gpu::select_preferred_device(
            &selected, &selection_error),
        "preferred GPU selection failed");
    const auto devices = nsos::gpu::enumerate_devices();
    const auto selected_info =
        std::find_if(
            devices.begin(), devices.end(),
            [selected](const nsos::gpu::DeviceInfo& info) {
              return info.index == selected;
            });
    require(selected_info != devices.end(),
            "selected GPU is absent from enumeration");
    require(selected_info->compiled,
            "selected GPU architecture is absent from the binary");

    int device_count = 0;
    require(
        cudaGetDeviceCount(&device_count) == cudaSuccess,
        "visible GPU count query failed");
    if (device_count > 1) {
      int alternate = selected == 0 ? 1 : 0;
      require(
          cudaSetDevice(alternate) == cudaSuccess,
          "could not simulate an external per-thread device change");
      int rebound = -1;
      require(
          nsos::gpu::select_preferred_device(
              &rebound, &selection_error) &&
              rebound == selected,
          "preferred selection did not repair an external device change");
      int active = -1;
      require(
          cudaGetDevice(&active) == cudaSuccess &&
              active == selected,
          "runtime remained bound to the externally selected device");
    }

    constexpr int kThreads = 8;
    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int worker = 0; worker < kThreads; ++worker) {
      workers.emplace_back([selected, &failures] {
        try {
          int thread_selected = -1;
          if (!nsos::gpu::select_preferred_device(
                  &thread_selected, nullptr) ||
              thread_selected != selected) {
            ++failures;
            return;
          }
          int active = -1;
          if (cudaGetDevice(&active) != cudaSuccess ||
              active != selected) {
            ++failures;
            return;
          }
          nsos::cuda_detail::DeviceBuffer<int> scratch;
          if (scratch.ensure(256) == nullptr ||
              scratch.device_id() != selected) {
            ++failures;
            return;
          }
          Tensor tensor =
              Tensor::ones({256}, Device::GPU);
          const Tensor host = tensor.cpu();
          for (int index = 0; index < host.size; ++index) {
            if (host.data()[index] != 1.0f) {
              ++failures;
              return;
            }
          }
        } catch (...) {
          ++failures;
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    require(failures.load() == 0,
            "GPU device binding failed under host-thread concurrency");
  });
}
