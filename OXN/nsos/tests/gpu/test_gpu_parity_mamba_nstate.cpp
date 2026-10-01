// =====================================================================
// GPU parity — full Mamba-2 SSD with N-state expansion
// (NSOS_MAMBA_PROPER_SSM + proper_state_expansion).
//
// The state-expanded recurrence (per-head dt/A, per-head N-dim B/C, state
// h ∈ R^{H×P×N}, linear readout y=Σ_n h·C) must produce the SAME forward output
// and input gradient on the GPU (mamba_nstate_* CUDA kernels) as on the host
// reference loop.  Same strategy as the diagonal proper parity: one model, run
// CPU then to(GPU), compare output + returned input-gradient.
// =====================================================================
#include "gpu_parity_common.h"

#include "../../include/mamba2.h"
#include "../../include/nsos/determinism.h"

#include <cmath>

using namespace nsos;
using namespace nsos::gpu_parity_test;

int main() {
  return run_parity("mamba_nstate", [] {
    const int H = 2, P = 4, N = 3, L = 6;
    const int D = H * P;  // d_model = n_heads * d_head
    MambaConfig cfg;
    cfg.proper_selective_ssm = true;
    cfg.proper_state_expansion = true;
    cfg.conv_kernel = 3;
    Mamba2SSD layer(D, N, H, cfg);  // CPU weights at construction

    Tensor x({L, D}, Device::CPU);
    for (int i = 0; i < x.size; ++i) {
      x.data()[i] = std::sin(0.3f * static_cast<float>(i + 1));
    }

    // ── CPU reference ──
    Tensor y_cpu = layer.forward(x);
    Tensor dy = y_cpu.clone();
    Context ctx;
    Tensor gin_cpu = layer.backward(dy, ctx);
    auto parameters = layer.parameters();
    std::vector<Tensor> cpu_parameter_grads;
    cpu_parameter_grads.reserve(parameters.size());
    for (Parameter* parameter : parameters) {
      cpu_parameter_grads.push_back(
          parameter->grad.size > 0 ? parameter->grad.clone() : Tensor());
      parameter->zero_grad();
    }

    // ── GPU under test (same weights moved to device) ──
    determinism::set_deterministic_reductions(true);
    layer.to(Device::GPU);
    Tensor x_gpu = x.to(Device::GPU);
    Tensor y_gpu = layer.forward(x_gpu);
    Tensor dy_gpu = dy.to(Device::GPU);
    Tensor gin_gpu = layer.backward(dy_gpu, ctx);

    assert_close(y_cpu, y_gpu, 1e-3f, "mamba_nstate forward y");
    assert_close(gin_cpu, gin_gpu, 1e-3f, "mamba_nstate backward d/input");

    parameters = layer.parameters();
    if (parameters.size() != cpu_parameter_grads.size()) {
      throw std::runtime_error(
          "mamba_nstate parameter registry changed across device");
    }
    std::vector<Tensor> deterministic_parameter_grads;
    deterministic_parameter_grads.reserve(parameters.size());
    for (size_t index = 0; index < parameters.size(); ++index) {
      Parameter* parameter = parameters[index];
      const Tensor& cpu_grad = cpu_parameter_grads[index];
      if (cpu_grad.size == 0 && parameter->grad.size == 0) {
        deterministic_parameter_grads.emplace_back();
        continue;
      }
      if (cpu_grad.size == 0 || parameter->grad.size == 0) {
        throw std::runtime_error(
            "mamba_nstate gradient presence mismatch for " +
            parameter->name);
      }
      assert_close(cpu_grad, parameter->grad, 2e-3f,
                   ("mamba_nstate parameter " + parameter->name).c_str(),
                   2e-4f);
      deterministic_parameter_grads.push_back(parameter->grad.clone());
      parameter->zero_grad();
    }

    Tensor y_repeat = layer.forward(x_gpu);
    Tensor gin_repeat = layer.backward(dy_gpu, ctx);
    assert_close(y_gpu, y_repeat, 0.0f,
                 "mamba_nstate deterministic forward exact repeat");
    assert_close(gin_gpu, gin_repeat, 0.0f,
                 "mamba_nstate deterministic backward exact repeat");
    parameters = layer.parameters();
    for (size_t index = 0; index < parameters.size(); ++index) {
      if (deterministic_parameter_grads[index].size == 0) continue;
      assert_close(
          deterministic_parameter_grads[index], parameters[index]->grad, 0.0f,
          ("mamba_nstate deterministic parameter exact repeat " +
           parameters[index]->name).c_str());
    }
    determinism::set_deterministic_reductions(false);
  });
}
