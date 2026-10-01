// BitLinear float-reference CPU↔GPU forward/backward parity and upstream
// gradient ownership, including exact-linear, magnitude and LoQA variants.
// Packed ternary dispatch is covered separately by test_gpu_parity_bitlinear_dp4a.

#include "bitlinear.h"
#include "gpu_parity_common.h"
#include "tensor.h"
#include <cstring>
#include <vector>

using nsos::BitLinear;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

// Removing a reference-backward clone must not mutate a caller-owned gradient,
// including the magnitude and LoQA paths and repeated gradient accumulation.
static void check_reference_backward_input_ownership(bool exact, bool loqa) {
  BitLinear cpu(32, 24, true, 713);
  BitLinear gpu(32, 24, true, 713);
  for (BitLinear* layer : {&cpu, &gpu}) {
    layer->set_reference_path(true);
    layer->set_exact_linear_mode(exact);
    layer->set_use_loqa(loqa);
  }
  auto cpu_params = cpu.parameters();
  auto gpu_params = gpu.parameters();
  for (size_t p = 0; p < cpu_params.size(); ++p) {
    if (cpu_params[p]->trainable) {
      for (int i = 0; i < cpu_params[p]->data.size; ++i) {
        cpu_params[p]->data.data()[i] =
            0.03f * std::sin(0.31f * static_cast<float>(i + 1 + p));
      }
    }
    gpu_params[p]->data.copy_from(cpu_params[p]->data);
  }
  gpu.to(Device::GPU);
  Tensor x({2, 3, 32}, Device::CPU);
  Tensor dy({2, 3, 24}, Device::CPU);
  for (int i = 0; i < x.size; ++i)
    x.data()[i] = std::cos(0.07f * static_cast<float>(i + 3));
  for (int i = 0; i < dy.size; ++i)
    dy.data()[i] = std::sin(0.13f * static_cast<float>(i + 1));
  const Tensor incoming = dy.clone();
  Tensor gpu_dy = dy.to(Device::GPU);
  assert_close(cpu.forward(x), gpu.forward(x.to(Device::GPU)), 2e-4f,
               "bitlinear/reference-forward", 2e-3f);
  std::vector<Tensor> first_cpu_grads, first_gpu_grads;
  for (int pass = 0; pass < 2; ++pass) {
    assert_close(cpu.backward(dy), gpu.backward(gpu_dy), 2e-4f,
                 "bitlinear/reference-dx", 2e-3f);
    Tensor gpu_incoming = gpu_dy.cpu();
    const size_t gradient_bytes = static_cast<size_t>(dy.size) * sizeof(float);
    if (std::memcmp(incoming.data(), dy.data(), gradient_bytes) != 0 ||
        std::memcmp(incoming.data(), gpu_incoming.data(), gradient_bytes) != 0)
      throw std::runtime_error("BitLinear backward modified its incoming gradient");
    for (size_t p = 0; p < cpu_params.size(); ++p) {
      if (!cpu_params[p]->trainable) continue;
      if (cpu_params[p]->grad.size == 0 || gpu_params[p]->grad.size == 0)
        throw std::runtime_error("BitLinear reference parameter gradient missing");
      assert_close(cpu_params[p]->grad, gpu_params[p]->grad, 3e-4f,
                   "bitlinear/reference-parameter-gradient", 3e-3f);
      if (pass == 0) {
        first_cpu_grads.push_back(cpu_params[p]->grad.clone());
        first_gpu_grads.push_back(gpu_params[p]->grad.clone());
      }
    }
    if (pass == 1) {
      size_t active = 0;
      for (size_t p = 0; p < cpu_params.size(); ++p) {
        if (!cpu_params[p]->trainable) continue;
        assert_close(cpu_params[p]->grad, first_cpu_grads[active].mul(2.0f),
                     1e-6f, "bitlinear/cpu-accumulation", 1e-5f);
        assert_close(gpu_params[p]->grad, first_gpu_grads[active].mul(2.0f),
                     1e-6f, "bitlinear/gpu-accumulation", 1e-5f);
        ++active;
      }
    }
  }
}

int main() {
  return run_parity("bitlinear", [] {
    BitLinear cpu_layer(32, 24, /*bias=*/true);
    BitLinear gpu_layer(32, 24, /*bias=*/true);
    gpu_layer.to(Device::GPU);

    // Force the GPU layer to share weights with the CPU layer so the
    // only delta is kernel implementation, not random init.
    gpu_layer.weight.data.copy_from(cpu_layer.weight.data.to(Device::GPU));

    Tensor x = Tensor::random({4, 32}, Device::CPU);
    const Tensor cpu_out = cpu_layer.forward(x);
    const Tensor gpu_out = gpu_layer.forward(x.to(Device::GPU)).cpu();
    cuda_sync_or_throw("bitlinear/forward");
    assert_close(cpu_out, gpu_out, 2e-3f, "bitlinear");
    for (bool exact : {false, true})
      for (bool loqa : {false, true})
        check_reference_backward_input_ownership(exact, loqa);
  });
}
