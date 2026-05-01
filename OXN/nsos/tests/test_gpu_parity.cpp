#include "bitlinear.h"
#include "jamba.h"
#include "mamba2.h"
#include "tensor.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace nsos;

namespace {

void assert_close(const Tensor& lhs, const Tensor& rhs, float tol = 1e-4f) {
  Tensor lhs_cpu = lhs.cpu();
  Tensor rhs_cpu = rhs.cpu();
  assert(lhs_cpu.size == rhs_cpu.size);
  for (int i = 0; i < lhs_cpu.size; ++i) {
    assert(std::abs(lhs_cpu.data()[i] - rhs_cpu.data()[i]) <= tol);
  }
}

void run_case(const char* name, const std::function<void()>& fn) {
  const char* selected = std::getenv("NSOS_GPU_PARITY_CASE");
  if (selected && std::string(selected) != name) {
    std::cout << "[GPUParity] skip " << name << std::endl;
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

  cpu_layer.set_streaming_mode(true);
  gpu_layer.set_streaming_mode(true);

  Tensor final_cpu;
  Tensor final_gpu;
  for (int step = 0; step < 8; ++step) {
    Tensor token = Tensor::random({1, 32}, Device::CPU);
    final_cpu = cpu_layer.forward(token, nullptr);
    final_gpu = gpu_layer.forward(token.to(Device::GPU), nullptr).cpu();
  }
  assert_close(final_cpu, final_gpu, 3e-3f);
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
  config.num_layers = 4;
  config.d_model = 64;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 64;
  config.attention_slot = 63;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_exact_attention_training = false;

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
  std::cout << "GPU parity test skipped: CUDA not enabled." << std::endl;
  return 0;
#else
  try {
#ifdef USE_CUDA
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
      std::cout << "GPU parity test skipped: no CUDA device available." << std::endl;
      return 0;
    }
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    std::cout << "[GPUParity] device=" << props.name << " cc=" << props.major << "."
              << props.minor << " runtime=" << CUDART_VERSION << std::endl;
#endif
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
