#include "gpu_parity_common.h"
#include "tensor.h"
#include "ttt_layer.h"
#include "training_runtime_policy.h"

#include <vector>
#include <utility>

using nsos::Device;
using nsos::Parameter;
using nsos::Tensor;
using nsos::TTTLayer;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

void mirror_parameters(const std::vector<Parameter*>& src,
                       const std::vector<Parameter*>& dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error("TTT parameter count mismatch");
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (!src[i] || !dst[i] || src[i]->data.size != dst[i]->data.size) {
      throw std::runtime_error("TTT parameter layout mismatch");
    }
    dst[i]->data.copy_from(src[i]->data.to(dst[i]->data.get_device()));
  }
}

}  // namespace

int main() {
  return run_parity("ttt", [] {
    std::vector<std::pair<int, int>> widths{{8, 6}, {64, 33}};
    // Both axes cross the distributed-column dispatch boundary, with tails.
    if (nsos::training_policy::full_ttt_bptt()) widths.emplace_back(257, 257);
    for (bool hamiltonian : {false, true}) for (bool clipping : {false, true})
    for (const auto& [dim, hidden] : widths) {
    const int rows = dim >= 256 ? 33 : dim == 64 ? 17 : 4;

    TTTLayer cpu_layer(dim, hidden, 0.002f, 0x5a17u);
    TTTLayer gpu_layer(dim, hidden, 0.002f, 0x5a17u);
    gpu_layer.to(Device::GPU);
    mirror_parameters(cpu_layer.parameters(), gpu_layer.parameters());

    cpu_layer.set_use_hamiltonian(hamiltonian);
    gpu_layer.set_use_hamiltonian(hamiltonian);
    cpu_layer.set_max_grad_norm(clipping ? 0.05f : 1e6f);
    gpu_layer.set_max_grad_norm(clipping ? 0.05f : 1e6f);
    cpu_layer.set_friction(0.85f);
    gpu_layer.set_friction(0.85f);

    Tensor input({rows, dim}, Device::CPU);
    Tensor grad({rows, dim}, Device::CPU);
    for (int i = 0; i < input.size; ++i) {
      input.data()[i] = -0.4f + 0.025f * static_cast<float>(i % 37);
      grad.data()[i] = 0.2f - 0.01f * static_cast<float>(i % dim);
    }

    const Tensor cpu_out = cpu_layer.forward(input);
    const Tensor gpu_out = gpu_layer.forward(input.to(Device::GPU)).cpu();
    cuda_sync_or_throw("ttt/forward");
    assert_close(cpu_out, gpu_out, 3e-3f, "ttt_forward");
    assert_close(cpu_layer.get_current_adaptation(),
                 gpu_layer.get_current_adaptation(), 3e-3f,
                 "ttt_adaptation");

    const Tensor cpu_grad_input = cpu_layer.backward(grad);
    const Tensor gpu_grad_input =
        gpu_layer.backward(grad.to(Device::GPU)).cpu();
    cuda_sync_or_throw("ttt/backward");
    assert_close(cpu_grad_input, gpu_grad_input, 4e-3f,
                 "ttt_grad_input");

    const auto cpu_params = cpu_layer.parameters();
    const auto gpu_params = gpu_layer.parameters();
    for (size_t i = 0; i < cpu_params.size(); ++i) {
      assert_close(cpu_params[i]->grad, gpu_params[i]->grad, 4e-3f,
                   "ttt_param_grad");
    }
    const auto cpu_snapshot = cpu_layer.snapshot_state();
    const auto gpu_snapshot = gpu_layer.snapshot_state();
    const Tensor continued_cpu = cpu_layer.forward(input);
    const Tensor continued_gpu = gpu_layer.forward(input.to(Device::GPU));
    assert_close(continued_cpu, continued_gpu, 3e-3f, "ttt_session_continuation");
    cpu_layer.restore_state(cpu_snapshot);
    gpu_layer.restore_state(gpu_snapshot);
    assert_close(continued_cpu, cpu_layer.forward(input), 0, "ttt_cpu_session_restore");
    assert_close(continued_gpu, gpu_layer.forward(input.to(Device::GPU)), 0, "ttt_gpu_session_restore");
    cpu_layer.reset(); gpu_layer.reset();
    assert_close(cpu_out, cpu_layer.forward(input), 0, "ttt_cpu_reset");
    assert_close(gpu_out, gpu_layer.forward(input.to(Device::GPU)), 0, "ttt_gpu_reset");
    }
    if (nsos::training_policy::full_ttt_bptt()) {
      for (bool ham : {false, true}) for (bool clip : {false, true}) {
        TTTLayer cpu(64, 33, 0.003f, 91), gpu(64, 33, 0.003f, 91);
        gpu.to(Device::GPU); mirror_parameters(cpu.parameters(), gpu.parameters());
        cpu.set_use_hamiltonian(ham); gpu.set_use_hamiltonian(ham);
        cpu.set_max_grad_norm(clip ? 0.05f : 0); gpu.set_max_grad_norm(clip ? 0.05f : 0);
        cpu.set_batch_valid_lengths({65, 61, 0}); gpu.set_batch_valid_lengths({65, 61, 0});
        Tensor x({3, 65, 64}, Device::CPU), g({3, 65, 64}, Device::CPU);
        for (int i = 0; i < x.size; ++i) {
          x.data()[i] = 0.015f * (i % 43 - 21); g.data()[i] = 0.007f * (i % 19 - 9);
        }
        assert_close(cpu.forward(x), gpu.forward(x.to(Device::GPU)), 3e-3f, "full_ttt_padded_batch_forward");
        const size_t expected = 2 * 3 * 3 * 33 * 64 * sizeof(float);
        if (cpu.saved_state_history_bytes() != expected || gpu.saved_state_history_bytes() != expected)
          throw std::runtime_error("Full TTT did not retain boundary history");
        const Tensor cg = cpu.backward(g), gg = gpu.backward(g.to(Device::GPU)).cpu();
        assert_close(cg, gg, 4e-3f, "full_ttt_padded_batch_vjp");
        if (cpu.saved_state_history_bytes() || gpu.saved_state_history_bytes())
          throw std::runtime_error("Full TTT backward retained consumed history");
        const auto cp = cpu.parameters(), gp = gpu.parameters();
        for (size_t p = 0; p < cp.size(); ++p)
          assert_close(cp[p]->grad, gp[p]->grad, 4e-3f, "full_ttt_batch_param_vjp");
        for (int b = 1; b < 3; ++b) for (int t = b == 1 ? 61 : 0; t < 65; ++t)
          for (int d = 0; d < 64; ++d) if (cg.data()[(b * 65 + t) * 64 + d] != 0 ||
              gg.data()[(b * 65 + t) * 64 + d] != 0)
            throw std::runtime_error("Full TTT padding has nonzero gradient");
      }
    }
  });
}
