#include "../include/gpu_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace nsos::gpu {

namespace {

bool allow_integrated_gpu() {
  const char* value = std::getenv("NSOS_ALLOW_INTEGRATED_GPU");
  return value != nullptr && std::string(value) == "1";
}

#if defined(NSOS_GPU_BACKEND_HIP)
bool architecture_starts_with(const std::string& architecture,
                              std::string_view prefix) {
  return architecture.size() >= prefix.size() &&
         architecture.compare(0, prefix.size(), prefix) == 0;
}

bool amd_bf16_matrix_supported(const std::string& architecture) {
  // Architectures with validated native BF16 matrix instructions in the
  // ROCm/hipBLAS path. Keep this allow-list explicit: accepting an unknown
  // gfx target and silently emulating BF16 would invalidate benchmark claims.
  return architecture_starts_with(architecture, "gfx90a") ||
         architecture_starts_with(architecture, "gfx94") ||
         architecture_starts_with(architecture, "gfx11") ||
         architecture_starts_with(architecture, "gfx12");
}

std::string amd_architecture_base(const std::string& architecture) {
  const std::size_t feature_separator = architecture.find(':');
  return architecture.substr(0, feature_separator);
}

bool amd_architecture_compiled(const std::string& architecture) {
#ifdef NSOS_HIP_ARCHITECTURES_CSV
  const std::string targets = NSOS_HIP_ARCHITECTURES_CSV;
  const std::string device_target =
      amd_architecture_base(architecture);
  std::size_t begin = 0;
  while (begin <= targets.size()) {
    const std::size_t end = targets.find(',', begin);
    const std::string configured =
        targets.substr(
            begin,
            end == std::string::npos
                ? std::string::npos
                : end - begin);
    if (amd_architecture_base(configured) == device_target) {
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return false;
#else
  (void)architecture;
  return false;
#endif
}
#elif defined(NSOS_GPU_BACKEND_CUDA)
bool cuda_architecture_compiled(int device_architecture) {
#ifdef NSOS_CUDA_ARCHITECTURES_CSV
  const std::string targets = NSOS_CUDA_ARCHITECTURES_CSV;
  std::size_t begin = 0;
  while (begin <= targets.size()) {
    const std::size_t end = targets.find(',', begin);
    const std::string token =
        targets.substr(begin, end == std::string::npos
                                  ? std::string::npos
                                  : end - begin);
    char* suffix = nullptr;
    errno = 0;
    const long target = std::strtol(
        token.c_str(), &suffix, 10);
    if (errno == 0 && suffix != token.c_str() &&
        target > 0 &&
        target <= std::numeric_limits<int>::max()) {
      const bool virtual_code =
          std::string_view(suffix).find("-virtual") == 0;
      if ((virtual_code &&
           device_architecture >= static_cast<int>(target)) ||
          (!virtual_code &&
           device_architecture / 10 ==
               static_cast<int>(target) / 10 &&
           device_architecture >= static_cast<int>(target))) {
        return true;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
#else
  (void)device_architecture;
#endif
  return false;
}
#endif

bool parse_device_override(int* device, std::string* error) {
  const char* value = std::getenv("NSOS_GPU_DEVICE");
  if (value == nullptr || *value == '\0') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    if (error != nullptr) {
      *error = "NSOS_GPU_DEVICE must be a non-negative integer";
    }
    throw std::invalid_argument(
        "NSOS_GPU_DEVICE must be a non-negative integer");
  }
  *device = static_cast<int>(parsed);
  return true;
}

}  // namespace

std::vector<DeviceInfo> enumerate_devices() {
  // Runtime visibility and device properties are immutable after the GPU
  // context is initialized. Runtime identity is validated on every training
  // transaction, so querying every property every step would add avoidable
  // driver calls. A first-query failure remains fail-closed for this process.
  static const std::vector<DeviceInfo> cached_devices = [] {
  std::vector<DeviceInfo> devices;
#ifdef USE_CUDA
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
    (void)cudaGetLastError();
    return devices;
  }
  devices.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, index) != cudaSuccess) {
      (void)cudaGetLastError();
      continue;
    }
    DeviceInfo info;
    info.index = index;
    info.name = properties.name;
    info.total_memory = properties.totalGlobalMem;
    info.warp_size = properties.warpSize;
    info.integrated = properties.integrated != 0;
#if defined(NSOS_GPU_BACKEND_HIP)
    info.architecture = properties.gcnArchName;
    info.compiled = amd_architecture_compiled(info.architecture);
    // gfx9+ covers the ROCm targets supported by this backend's FP16
    // hipBLAS kernels. BF16 is narrower and intentionally fail-closed.
    info.fp16 = properties.major >= 9;
    info.bf16 = amd_bf16_matrix_supported(info.architecture);
#else
    info.architecture =
        "sm_" + std::to_string(properties.major) +
        std::to_string(properties.minor);
    info.compiled = cuda_architecture_compiled(
        properties.major * 10 + properties.minor);
    info.fp16 = properties.major >= 7;
    info.bf16 = properties.major >= 8;
#endif
    devices.push_back(std::move(info));
  }
#endif
  return devices;
  }();
  return cached_devices;
}

bool select_preferred_device(int* selected_device, std::string* error) {
#ifndef USE_CUDA
  if (error != nullptr) {
    *error = "NSOS was built without a GPU backend";
  }
  return false;
#else
  // Selection policy is immutable for the process: environment controls are
  // read once. cudaSetDevice itself is deliberately repeated because the
  // active runtime device is host-thread local on CUDA/HIP.
  static std::once_flag selection_once;
  static int chosen = -1;
  static std::string selection_error;
  std::call_once(selection_once, [] {
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
      selection_error =
          std::string("No ") + backend_name() +
          " device is visible to the runtime";
      return;
    }

    try {
      if (parse_device_override(&chosen, &selection_error)) {
        const auto match =
            std::find_if(devices.begin(), devices.end(),
                         [](const DeviceInfo& info) {
                           return info.index == chosen;
                         });
        if (match == devices.end()) {
          selection_error =
              "NSOS_GPU_DEVICE does not identify a visible GPU";
          chosen = -1;
          return;
        }
        if (!match->compiled) {
          selection_error =
              "NSOS_GPU_DEVICE selects architecture '" +
              match->architecture +
              "', which is absent from this binary";
          chosen = -1;
          return;
        }
        if (match->integrated && !allow_integrated_gpu()) {
          selection_error =
              "NSOS_GPU_DEVICE selects an integrated GPU; set "
              "NSOS_ALLOW_INTEGRATED_GPU=1 only for an explicitly validated "
              "development configuration";
          chosen = -1;
          return;
        }
      }
    } catch (const std::invalid_argument&) {
      chosen = -1;
      return;
    }

    if (chosen < 0) {
      std::size_t best_memory = 0;
      for (const DeviceInfo& info : devices) {
        if (info.integrated || !info.compiled) {
          continue;
        }
        if (chosen < 0 || info.total_memory > best_memory) {
          chosen = info.index;
          best_memory = info.total_memory;
        }
      }
    }

    if (chosen < 0 && allow_integrated_gpu()) {
      const auto integrated =
          std::find_if(devices.begin(), devices.end(),
                       [](const DeviceInfo& info) {
                         return info.integrated && info.compiled;
                       });
      if (integrated != devices.end()) {
        chosen = integrated->index;
      }
    }
    if (chosen < 0) {
      selection_error =
          "No discrete GPU architecture compiled into this binary is "
          "visible; integrated devices are fail-closed unless explicitly "
          "allowed and compiled";
      return;
    }

    const cudaError_t set_status = cudaSetDevice(chosen);
    if (set_status != cudaSuccess) {
      selection_error =
          std::string("Failed to select GPU: ") +
          cudaGetErrorString(set_status);
      chosen = -1;
      return;
    }
    const cudaError_t context_status = cudaFree(nullptr);
    if (context_status != cudaSuccess) {
      selection_error =
          std::string("Failed to initialize GPU context: ") +
          cudaGetErrorString(context_status);
      chosen = -1;
    }
  });

  if (chosen < 0) {
    if (error != nullptr) {
      *error = selection_error;
    }
    return false;
  }
  // Do not trust a private thread-local cache here. CUDA/HIP device state is
  // host-thread-local, but other runtime consumers in the same process may
  // legally call cudaSetDevice between NSOS operations. Querying the runtime
  // makes this boundary self-healing instead of dispatching a kernel against
  // allocations owned by a different device.
  int active_device = -1;
  const cudaError_t get_status = cudaGetDevice(&active_device);
  if (get_status != cudaSuccess || active_device != chosen) {
    if (get_status != cudaSuccess) {
      (void)cudaGetLastError();
    }
    const cudaError_t set_status = cudaSetDevice(chosen);
    if (set_status != cudaSuccess) {
      if (error != nullptr) {
        *error = std::string("Failed to select GPU: ") +
                 cudaGetErrorString(set_status);
      }
      return false;
    }
  }
  if (selected_device != nullptr) {
    *selected_device = chosen;
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
#endif
}

}  // namespace nsos::gpu
