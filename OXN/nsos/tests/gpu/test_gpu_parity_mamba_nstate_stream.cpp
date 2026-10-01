// =====================================================================
// GPU parity — N-STATE on-device single-token decode (NSOS_MAMBA_GPU_STEP,
// GPU-first default ON).
//
// forward_proper_nstate_step's fused kernel (mamba_nstate_step_kernel: conv
// from the device-resident ring + per-head N-state recurrence + linear
// readout + SiLU gate, carrying h ∈ R^{H·P·N} and the conv window on the GPU
// across tokens) must reproduce the full-sequence N-state scan
// token-for-token.  Sibling of test_gpu_parity_mamba_proper_stream (the
// diagonal path); this gate promotes the FULL Mamba-2 decode step.
// =====================================================================
#include "gpu_parity_common.h"

#include "../../include/mamba2.h"

#include <cmath>
#include <cstdlib>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

int main() {
  // GPU-first default is ON; set explicitly so the gate is self-contained
  // even if the default ever changes.
#ifdef _WIN32
  _putenv_s("NSOS_MAMBA_GPU_STEP", "1");
#else
  setenv("NSOS_MAMBA_GPU_STEP", "1", 1);
#endif

  return run_parity("mamba_nstate_stream", [] {
    const int D = 8, N = 8, H = 2, L = 6;
    MambaConfig cfg;
    cfg.proper_selective_ssm = true;
    cfg.proper_state_expansion = true;
    cfg.conv_kernel = 3;
    Mamba2SSD layer(D, N, H, cfg);  // CPU weights at construction
    layer.to(Device::GPU);

    Tensor x({L, D}, Device::CPU);
    for (int i = 0; i < x.size; ++i) {
      x.data()[i] = std::sin(0.3f * static_cast<float>(i + 1));
    }
    Tensor x_gpu = x.to(Device::GPU);

    // Reference: full-sequence N-state scan (streaming off).
    Tensor y_full = layer.forward(x_gpu);  // [L, D]
    Tensor y_full_h = y_full.cpu();

    // Under test: prime with token 0, then fused on-device N-state steps.
    layer.reset();
    layer.set_streaming_mode(true);
    Tensor y_stream_h({L, D}, Device::CPU);
    for (int t = 0; t < L; ++t) {
      Tensor xt({1, D}, Device::CPU);
      for (int c = 0; c < D; ++c) {
        xt.data()[c] = x.data()[static_cast<size_t>(t) * D + c];
      }
      Tensor yt = layer.forward(xt.to(Device::GPU)).cpu();  // [1, D]
      for (int c = 0; c < D; ++c) {
        y_stream_h.data()[static_cast<size_t>(t) * D + c] = yt.data()[c];
      }
    }
    if (layer.stream_priming_gpu_calls() != 1 ||
        layer.stream_priming_host_fallbacks() != 0) {
      throw std::runtime_error(
          "N-state Mamba prefill did not prime streaming carry on-device");
    }
    layer.set_streaming_mode(false);

    // Both paths are GPU kernels (full N-state scan vs fused step); tolerance
    // covers the small fp reordering between them.
    assert_close(y_full_h, y_stream_h, 2e-3f,
                 "mamba_nstate GPU step == full scan");
  });
}
