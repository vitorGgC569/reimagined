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

    // ── GPU under test (same weights moved to device) ──
    layer.to(Device::GPU);
    Tensor x_gpu = x.to(Device::GPU);
    Tensor y_gpu = layer.forward(x_gpu);
    Tensor dy_gpu = dy.to(Device::GPU);
    Tensor gin_gpu = layer.backward(dy_gpu, ctx);

    assert_close(y_cpu, y_gpu, 1e-3f, "mamba_nstate forward y");
    assert_close(gin_cpu, gin_gpu, 1e-3f, "mamba_nstate backward d/input");
  });
}
