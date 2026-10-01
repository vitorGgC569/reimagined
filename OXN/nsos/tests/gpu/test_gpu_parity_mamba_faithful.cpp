#include "gpu_parity_common.h"

#include "../../include/mamba2.h"
#include "../../include/nsos/determinism.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

int main() {
  return run_parity("mamba_faithful", [] {
    constexpr int D = 8;
    constexpr int N = 3;
    constexpr int L = 6;
    MambaConfig cfg;
    cfg.faithful_mamba2 = true;
    cfg.expand = 2;
    cfg.head_dim = 4;
    cfg.n_groups = 1;
    cfg.conv_kernel = 3;
    Mamba2SSD layer(D, N, 1, cfg);
    Tensor x({L, D}, Device::CPU);
    for (int i = 0; i < x.size; ++i) {
      x.data()[i] = 0.4f * std::sin(0.27f * static_cast<float>(i + 1));
    }

    Tensor y_cpu = layer.forward(x);
    Context ctx;
    Tensor dx_cpu = layer.backward(y_cpu.clone(), ctx);

    layer.to(Device::GPU);
    Tensor x_gpu = x.to(Device::GPU);
    Tensor y_gpu = layer.forward(x_gpu);
    Tensor dx_gpu = layer.backward(y_gpu.clone(), ctx);
    assert_close(y_cpu, y_gpu, 2e-3f, "faithful forward");
    assert_close(dx_cpu, dx_gpu, 3e-3f, "faithful backward d/input");
    determinism::set_deterministic_reductions(true);
    Tensor y_deterministic = layer.forward(x_gpu);
    Tensor dx_deterministic =
        layer.backward(y_deterministic.clone(), ctx);
    assert_close(y_cpu, y_deterministic, 2e-3f,
                 "faithful deterministic forward");
    assert_close(dx_cpu, dx_deterministic, 3e-3f,
                 "faithful deterministic backward d/input");
    determinism::set_deterministic_reductions(false);

    // Production head width (P=64) exercises the warp/wave-aggregated
    // backward reduction.  Compare every parameter gradient, not only d/input:
    // gB/gC/gDt/gA/gD are precisely the shared reductions optimized by that
    // kernel and an input-only parity assertion cannot observe their failure.
    MambaConfig aggregate_cfg;
    aggregate_cfg.faithful_mamba2 = true;
    aggregate_cfg.expand = 2;
    aggregate_cfg.head_dim = 64;
    aggregate_cfg.n_groups = 1;
    aggregate_cfg.conv_kernel = 3;
    Mamba2SSD aggregate_layer(32, N, 1, aggregate_cfg);
    Tensor aggregate_input({4, 32}, Device::CPU);
    for (int i = 0; i < aggregate_input.size; ++i) {
      aggregate_input.data()[i] =
          0.35f * std::sin(0.19f * static_cast<float>(i + 1));
    }
    Tensor aggregate_grad({4, 32}, Device::CPU);
    for (int i = 0; i < aggregate_grad.size; ++i) {
      aggregate_grad.data()[i] =
          0.2f * std::cos(0.13f * static_cast<float>(i + 3));
    }

    Tensor aggregate_cpu_output = aggregate_layer.forward(aggregate_input);
    Tensor aggregate_cpu_dx =
        aggregate_layer.backward(aggregate_grad, ctx);
    auto aggregate_params = aggregate_layer.parameters();
    std::vector<Tensor> cpu_parameter_grads;
    cpu_parameter_grads.reserve(aggregate_params.size());
    for (Parameter* parameter : aggregate_params) {
      cpu_parameter_grads.push_back(
          parameter->grad.size > 0 ? parameter->grad.clone() : Tensor());
      parameter->zero_grad();
    }

    aggregate_layer.to(Device::GPU);
    Tensor aggregate_gpu_input = aggregate_input.to(Device::GPU);
    Tensor aggregate_gpu_grad = aggregate_grad.to(Device::GPU);
    Tensor aggregate_gpu_output =
        aggregate_layer.forward(aggregate_gpu_input);
    Tensor aggregate_gpu_dx =
        aggregate_layer.backward(aggregate_gpu_grad, ctx);
    assert_close(aggregate_cpu_output, aggregate_gpu_output, 2e-3f,
                 "faithful warp-aggregated forward");
    assert_close(aggregate_cpu_dx, aggregate_gpu_dx, 4e-3f,
                 "faithful warp-aggregated backward d/input");

    aggregate_params = aggregate_layer.parameters();
    if (aggregate_params.size() != cpu_parameter_grads.size()) {
      throw std::runtime_error(
          "faithful warp-aggregated parameter registry changed across device");
    }
    for (size_t index = 0; index < aggregate_params.size(); ++index) {
      Parameter* parameter = aggregate_params[index];
      const Tensor& cpu_grad = cpu_parameter_grads[index];
      if (cpu_grad.size == 0 && parameter->grad.size == 0) {
        continue;
      }
      if (cpu_grad.size == 0 || parameter->grad.size == 0) {
        throw std::runtime_error(
            "faithful warp-aggregated gradient presence mismatch for " +
            parameter->name);
      }
      assert_close(cpu_grad, parameter->grad, 5e-3f,
                   ("faithful warp-aggregated parameter " +
                    parameter->name).c_str(),
                   2e-4f);
    }

    // Production dimensions exercise the actual reduction depth used by the
    // accounting benchmark (P=64, N=64, K=4) with more than one batch and
    // sequence row. Tiny N=3/L=4 parity can hide summation-order drift that
    // compounds over a long training run.
    MambaConfig production_cfg;
    production_cfg.faithful_mamba2 = true;
    production_cfg.expand = 2;
    production_cfg.head_dim = 64;
    production_cfg.n_groups = 1;
    production_cfg.conv_kernel = 4;
    Mamba2SSD production_layer(128, 64, 1, production_cfg);
    // 65 tokens crosses two 32-token chunk boundaries so the chunked
    // deterministic A/B path cannot pass by exercising only its trivial
    // single-chunk case.
    Tensor production_input({2, 65, 128}, Device::CPU);
    Tensor production_grad({2, 65, 128}, Device::CPU);
    for (int i = 0; i < production_input.size; ++i) {
      production_input.data()[i] =
          0.11f * std::sin(0.017f * static_cast<float>(i + 1));
      production_grad.data()[i] =
          0.07f * std::cos(0.013f * static_cast<float>(i + 5));
    }
    Tensor production_cpu_output =
        production_layer.forward(production_input);
    Tensor production_cpu_dx =
        production_layer.backward(production_grad, ctx);
    auto production_params = production_layer.parameters();
    std::vector<Tensor> production_cpu_parameter_grads;
    production_cpu_parameter_grads.reserve(production_params.size());
    for (Parameter* parameter : production_params) {
      production_cpu_parameter_grads.push_back(
          parameter->grad.size > 0 ? parameter->grad.clone() : Tensor());
      parameter->zero_grad();
    }
    production_layer.to(Device::GPU);
    Tensor production_gpu_output =
        production_layer.forward(production_input.to(Device::GPU));
    Tensor production_gpu_dx =
        production_layer.backward(production_grad.to(Device::GPU), ctx);
    assert_close(production_cpu_output, production_gpu_output, 2e-3f,
                 "faithful production-shape forward");
    assert_close(production_cpu_dx, production_gpu_dx, 3e-3f,
                 "faithful production-shape backward d/input");
    production_params = production_layer.parameters();
    float production_worst_parameter_delta = 0.0f;
    std::string production_worst_parameter;
    for (size_t index = 0; index < production_params.size(); ++index) {
      Parameter* parameter = production_params[index];
      const Tensor& cpu_grad = production_cpu_parameter_grads[index];
      if (cpu_grad.size == 0 && parameter->grad.size == 0) {
        continue;
      }
      if (cpu_grad.size == 0 || parameter->grad.size == 0) {
        throw std::runtime_error(
            "faithful production-shape gradient presence mismatch for " +
            parameter->name);
      }
      Tensor gpu_grad_cpu = parameter->grad.cpu();
      for (int element = 0; element < cpu_grad.size; ++element) {
        const float delta = std::abs(
            cpu_grad.data()[element] - gpu_grad_cpu.data()[element]);
        if (delta > production_worst_parameter_delta) {
          production_worst_parameter_delta = delta;
          production_worst_parameter = parameter->name;
        }
      }
      assert_close(cpu_grad, parameter->grad, 2e-3f,
                   ("faithful production-shape parameter " +
                    parameter->name).c_str(),
                   5e-4f);
    }
    std::cout << "[GPUParity:mamba_faithful] production_worst_parameter="
              << production_worst_parameter
              << " max_abs_grad_delta="
              << production_worst_parameter_delta << std::endl;

    for (Parameter* parameter : production_params) {
      parameter->zero_grad();
    }
    determinism::set_deterministic_reductions(true);
    Tensor deterministic_output =
        production_layer.forward(production_input.to(Device::GPU));
    Tensor deterministic_dx =
        production_layer.backward(production_grad.to(Device::GPU), ctx);
    assert_close(production_cpu_output, deterministic_output, 2e-3f,
                 "faithful deterministic production-shape forward");
    assert_close(production_cpu_dx, deterministic_dx, 3e-3f,
                 "faithful deterministic production-shape backward d/input");
    std::vector<Tensor> deterministic_parameter_grads;
    deterministic_parameter_grads.reserve(production_params.size());
    for (size_t index = 0; index < production_params.size(); ++index) {
      Parameter* parameter = production_params[index];
      const Tensor& cpu_grad = production_cpu_parameter_grads[index];
      if (cpu_grad.size > 0) {
        assert_close(cpu_grad, parameter->grad, 2e-3f,
                     ("faithful deterministic production-shape parameter " +
                      parameter->name).c_str(),
                     5e-4f);
        deterministic_parameter_grads.push_back(
            parameter->grad.clone());
      } else {
        deterministic_parameter_grads.emplace_back();
      }
      parameter->zero_grad();
    }
    Tensor deterministic_output_repeat =
        production_layer.forward(production_input.to(Device::GPU));
    Tensor deterministic_dx_repeat =
        production_layer.backward(production_grad.to(Device::GPU), ctx);
    assert_close(deterministic_output, deterministic_output_repeat, 0.0f,
                 "faithful deterministic forward exact repeat");
    assert_close(deterministic_dx, deterministic_dx_repeat, 0.0f,
                 "faithful deterministic backward exact repeat");
    production_params = production_layer.parameters();
    for (size_t index = 0; index < production_params.size(); ++index) {
      if (deterministic_parameter_grads[index].size == 0) {
        continue;
      }
      assert_close(deterministic_parameter_grads[index],
                   production_params[index]->grad, 0.0f,
                   ("faithful deterministic parameter exact repeat " +
                    production_params[index]->name).c_str());
    }
    if (production_layer.faithful_deterministic_backward_calls() != 2) {
      throw std::runtime_error(
          "faithful deterministic production-shape path was not exercised "
          "exactly twice");
    }
    determinism::set_deterministic_reductions(false);

    // Exercise all three selective-SSM GPU decode implementations with R=2,
    // preserving snapshots on device and comparing against isolated rows.
    for (int mode = 0; mode < 3; ++mode) {
      MambaConfig batch_cfg = cfg;
      batch_cfg.faithful_mamba2 = mode == 0;
      batch_cfg.proper_selective_ssm = mode != 0;
      batch_cfg.proper_state_expansion = mode == 2;
      Mamba2SSD batched(D, N, 1, batch_cfg);
      batched.to(Device::GPU);
      batched.set_training_mode(false);
      std::vector<MambaStreamSnapshot> carried;
      for (int row = 0; row < 2; ++row) {
        batched.reset();
        batched.set_streaming_mode(true);
        (void)batched.forward(x.slice(0, row, row + 3).to(Device::GPU));
        carried.push_back(batched.snapshot_streaming_state(true));
        if (carried.back().proper_state_device.get_device() != Device::GPU ||
            carried.back().proper_state_device.size == 0 ||
            !carried.back().proper_state.empty())
          throw std::runtime_error("Mamba snapshot did not retain GPU state");
      }
      batched.restore_streaming_state_batch(carried);
      for (int step = 0; step < 3; ++step) {
        const auto before = batched.snapshot_streaming_state_batch(true);
        Tensor tokens = x.slice(0, step, step + 2).reshape({2, 1, D}).to(Device::GPU);
        Tensor output = batched.forward(tokens).reshape({2, D});
        const auto after = batched.snapshot_streaming_state_batch(true);
        for (int row = 0; row < 2; ++row) {
          batched.restore_streaming_state(before[row]);
          Tensor isolated = batched.forward(tokens.reshape({2, D}).slice(0, row, row + 1));
          assert_close(output.slice(0, row, row + 1), isolated, 2e-3f,
                       "batched Mamba GPU recurrence vs isolated row");
        }
        batched.restore_streaming_state_batch(after);
      }
    }

    // Full scan and device-resident incremental decode must agree.
    layer.reset();
    Tensor full = layer.forward(x_gpu);
    layer.reset();
    layer.set_streaming_mode(true);
    Tensor prefix({3, D}, Device::CPU);
    std::copy_n(x.data(), 3 * D, prefix.data());
    Tensor streamed_prefix = layer.forward(prefix.to(Device::GPU));
    assert_close(full.slice(0, 0, 3), streamed_prefix, 2e-3f,
                 "faithful streaming prefix");
    for (int t = 3; t < L; ++t) {
      Tensor token({1, D}, Device::CPU);
      std::copy_n(x.data() + t * D, D, token.data());
      Tensor streamed = layer.forward(token.to(Device::GPU));
      assert_close(full.slice(0, t, t + 1), streamed, 2e-3f,
                   "faithful streaming token");
    }
  });
}
