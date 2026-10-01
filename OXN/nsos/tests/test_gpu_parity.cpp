#include "bitlinear.h"
#include "jamba.h"
#include "mamba2.h"
#include "tensor.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include "gpu_backend.h"
#endif

using namespace nsos;

namespace {

void assert_close(const Tensor& lhs, const Tensor& rhs, float tol = 1e-4f) {
  Tensor lhs_cpu = lhs.cpu();
  Tensor rhs_cpu = rhs.cpu();
  if (lhs_cpu.shape.dims != rhs_cpu.shape.dims ||
      lhs_cpu.size != rhs_cpu.size) {
    throw std::runtime_error("GPU parity shape mismatch");
  }
  for (int i = 0; i < lhs_cpu.size; ++i) {
    const float left = lhs_cpu.data()[i];
    const float right = rhs_cpu.data()[i];
    if (!std::isfinite(left) || !std::isfinite(right) ||
        std::abs(left - right) > tol) {
      throw std::runtime_error(
          "GPU parity mismatch at index " + std::to_string(i) +
          ": cpu=" + std::to_string(left) +
          " gpu=" + std::to_string(right) +
          " tolerance=" + std::to_string(tol));
    }
  }
}

void run_case(const char* name, const std::function<void()>& fn) {
  const char* selected = std::getenv("NSOS_GPU_PARITY_CASE");
  if (selected && std::string(selected) != name) {
    std::cout << "[GPUParity] filtered " << name << std::endl;
    return;
  }
  std::cout << "[GPUParity] begin " << name << std::endl;
  fn();
#ifdef USE_CUDA
  const cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA sync failed after ") + name + ": " +
                             cudaGetErrorString(sync_status));
  }
#endif
  std::cout << "[GPUParity] pass " << name << std::endl;
}

void assert_streaming_mamba_parity() {
  Mamba2SSD cpu_layer(32, 16, 2);
  Mamba2SSD gpu_layer(32, 16, 2);
  gpu_layer.to(Device::GPU);

  // Both instances are independently initialized.  Compare kernels, not two
  // unrelated random functions: mirror every parameter before the first
  // streaming step (same contract as the decomposed Mamba streaming gate).
  const auto cpu_parameters = cpu_layer.parameters();
  const auto gpu_parameters = gpu_layer.parameters();
  if (cpu_parameters.size() != gpu_parameters.size()) {
    throw std::runtime_error("Mamba streaming parameter layout mismatch");
  }
  for (size_t index = 0; index < cpu_parameters.size(); ++index) {
    if (!cpu_parameters[index] || !gpu_parameters[index] ||
        cpu_parameters[index]->data.size != gpu_parameters[index]->data.size) {
      throw std::runtime_error(
          "Mamba streaming parameter mismatch at index " +
          std::to_string(index));
    }
    gpu_parameters[index]->data.copy_from(
        cpu_parameters[index]->data.to(Device::GPU));
  }

  cpu_layer.set_streaming_mode(true);
  gpu_layer.set_streaming_mode(true);

  Tensor final_cpu;
  Tensor final_gpu;
  for (int step = 0; step < 8; ++step) {
    Tensor token = Tensor::random({1, 32}, Device::CPU);
    final_cpu = cpu_layer.forward(token, nullptr);
    final_gpu = gpu_layer.forward(token.to(Device::GPU), nullptr).cpu();
  }
  assert_close(final_cpu, final_gpu, 5e-3f);
}

void assert_bitlinear_parity() {
  BitLinear cpu_layer(32, 24, true);
  BitLinear gpu_layer(32, 24, true);
  gpu_layer.to(Device::GPU);
  gpu_layer.weight.data.copy_from(cpu_layer.weight.data.to(Device::GPU));

  Tensor x = Tensor::random({4, 32}, Device::CPU);
  Tensor cpu_out = cpu_layer.forward(x);
  Tensor gpu_out = gpu_layer.forward(x.to(Device::GPU)).cpu();
  assert_close(cpu_out, gpu_out, 2e-3f);
}

void assert_jamba_forward_ids_batch_parity() {
  ModelConfig config;
  config.architecture_schema_version = 2;
  config.num_layers = 2;
  config.d_model = 64;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 2;
  config.attention_slot = 1;
  config.hybrid_composition = HybridComposition::ParallelGated;
  config.force_mamba_last_layer = false;
  config.faithful_attention_linears = true;
  config.mamba2_faithful = true;
  config.mamba_state_expansion = true;
  config.mamba_d_state = 8;
  config.mamba_head_dim = 32;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_exact_attention_training = true;
  config.use_gradient_checkpointing = true;

  JambaModel cpu_model(config, Device::CPU);
  JambaModel gpu_model(config, Device::GPU);
  gpu_model.to(Device::GPU);
  const auto temp_path =
      std::filesystem::temp_directory_path() / "nsos_gpu_parity_model.bin";
  cpu_model.save(temp_path.string());
  gpu_model.load(temp_path.string());

  const std::vector<std::vector<int>> batch_ids = {{1, 2, 3, 4}, {5, 6, 7}};
  Tensor cpu_logits = cpu_model.forward_ids_batch(batch_ids, nullptr).cpu();
  Tensor gpu_logits = gpu_model.forward_ids_batch(batch_ids, nullptr).cpu();
  assert_close(cpu_logits, gpu_logits, 3e-3f);

  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
}

} // namespace

int main() {
#ifndef USE_CUDA
  std::cerr << "GPU parity test failed: CUDA was not enabled at build time."
            << std::endl;
  return 1;
#else
  try {
    int selected_device = -1;
    std::string selection_error;
    if (!gpu::select_preferred_device(
            &selected_device, &selection_error)) {
      throw std::runtime_error(
          "GPU device is required but unavailable: " +
          selection_error);
    }
    cudaDeviceProp props{};
    const cudaError_t props_status =
        cudaGetDeviceProperties(&props, selected_device);
    if (props_status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaGetDeviceProperties failed: ") +
          cudaGetErrorString(props_status));
    }
    std::cout << "[GPUParity] backend=" << gpu::backend_name()
              << " device_index=" << selected_device
              << " device=" << props.name;
#if defined(NSOS_GPU_BACKEND_HIP)
    std::cout << " arch=" << props.gcnArchName;
#else
    std::cout << " arch=sm_" << props.major << props.minor;
#endif
    std::cout << " runtime=" << CUDART_VERSION << std::endl;
    Tensor a({2, 3}, Device::CPU);
    Tensor b({3, 2}, Device::CPU);
    for (int i = 0; i < a.size; ++i) a.data()[i] = static_cast<float>(i + 1);
    for (int i = 0; i < b.size; ++i) b.data()[i] = static_cast<float>(i - 2);

    run_case("tensor_add", [&] {
      const Tensor cpu_add = a.add(a);
      const Tensor gpu_add = a.to(Device::GPU).add(a.to(Device::GPU)).cpu();
      assert_close(cpu_add, gpu_add);
    });

    run_case("matmul", [&] {
      const Tensor cpu_mm = a.matmul(b);
      const Tensor gpu_mm = a.to(Device::GPU).matmul(b.to(Device::GPU)).cpu();
      assert_close(cpu_mm, gpu_mm);
    });

    run_case("rmsnorm", [&] {
      const Tensor cpu_norm = a.rmsnorm();
      const Tensor gpu_norm = a.to(Device::GPU).rmsnorm().cpu();
      assert_close(cpu_norm, gpu_norm);
    });

    run_case("bitlinear", assert_bitlinear_parity);
    run_case("mamba_streaming", assert_streaming_mamba_parity);
    run_case("jamba_batch", assert_jamba_forward_ids_batch_parity);

    std::cout << "GPU parity test passed!" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "GPU parity test failed: " << ex.what() << std::endl;
    return 1;
  }
#endif
}
