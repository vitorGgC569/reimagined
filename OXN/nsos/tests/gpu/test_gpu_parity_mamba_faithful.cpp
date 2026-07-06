#include "gpu_parity_common.h"

#include "../../include/mamba2.h"
#include "../../include/nsos/determinism.h"

#include <algorithm>
#include <cmath>

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
