// =====================================================================
// GPU parity — Mamba2SSD PROPER selective path (NSOS_MAMBA_PROPER_SSM).
//
// The corrected diagonal SSM (independent δ/B/C/z projections + causal
// conv1d + linear readout + SiLU gate) must compute the SAME forward output
// and the SAME input gradient on the GPU (CUDA kernels: conv1d_causal_* and
// mamba_proper_scan_*) as on the host reference loop.  This is the gate that
// promotes the GPU-resident proper path.
//
// Strategy: one model (deterministic init), run forward+backward on CPU,
// capture the output and the returned input-gradient, then to(GPU) (weights
// move to device) and run forward+backward on GPU; assert closeness.  We
// compare the layer output and the input gradient — both traverse the full
// conv + scan + gate + projection chain end to end.
// =====================================================================
#include "gpu_parity_common.h"

#include "../../include/mamba2.h"
#include "../../include/nsos/determinism.h"

#include <cmath>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

int main() {
  return run_parity("mamba_proper", [] {
    const int D = 8, N = 8, H = 1, L = 6;
    MambaConfig cfg;
    cfg.proper_selective_ssm = true;
    cfg.conv_kernel = 3;
    Mamba2SSD layer(D, N, H, cfg);  // CPU weights at construction

    Tensor x({L, D}, Device::CPU);
    for (int i = 0; i < x.size; ++i) {
      x.data()[i] = std::sin(0.3f * static_cast<float>(i + 1));
    }

    // ── CPU reference ──
    Tensor y_cpu = layer.forward(x);
    Tensor dy = y_cpu.clone();  // arbitrary but reproducible upstream grad
    Context ctx;
    Tensor gin_cpu = layer.backward(dy, ctx);

    // ── GPU under test (same weights moved to device) ──
    layer.to(Device::GPU);
    Tensor x_gpu = x.to(Device::GPU);
    Tensor y_gpu = layer.forward(x_gpu);
    Tensor dy_gpu = dy.to(Device::GPU);
    Tensor gin_gpu = layer.backward(dy_gpu, ctx);

    assert_close(y_cpu, y_gpu, 1e-3f, "mamba_proper forward y");
    assert_close(gin_cpu, gin_gpu, 1e-3f, "mamba_proper backward d/input");

    determinism::set_deterministic_reductions(true);
    Tensor y_det = layer.forward(x_gpu);
    Tensor gin_det = layer.backward(y_det.clone(), ctx);
    assert_close(y_cpu, y_det, 1e-3f, "mamba_proper deterministic forward");
    assert_close(gin_cpu, gin_det, 1e-3f,
                 "mamba_proper deterministic backward d/input");
    determinism::set_deterministic_reductions(false);
  });
}
