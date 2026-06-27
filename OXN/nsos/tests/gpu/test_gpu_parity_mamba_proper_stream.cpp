// =====================================================================
// GPU parity — proper-path on-device single-token decode (NSOS_MAMBA_GPU_STEP).
//
// forward_proper_step's fused on-device kernel (mamba_proper_step_kernel: conv
// from the device-resident ring + diagonal SSD recurrence + SiLU gate, carrying
// h and the conv window on the GPU across tokens) must reproduce the full-sequence
// scan token-for-token.  The host step is already proven byte-identical to the
// full scan on CPU (test_gradcheck check_proper_streaming_parity); this gate
// promotes the GPU step by comparing prefill+steps to the GPU full scan.
//
// Strategy: one proper-path layer on the GPU.  Run the whole sequence through
// forward() (streaming off) as the reference, then reset, enable streaming, and
// drive the same tokens one at a time with NSOS_MAMBA_GPU_STEP=1 so each step
// takes the device kernel path.  Assert the per-token outputs match the scan.
// =====================================================================
#include "gpu_parity_common.h"

#include "../../include/mamba2.h"

#include <cmath>
#include <cstdlib>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

int main() {
  // Enable the on-device step BEFORE any layer use (the env gate caches on first
  // read).  forward_proper_step then takes the fused device-kernel path.
#ifdef _WIN32
  _putenv_s("NSOS_MAMBA_GPU_STEP", "1");
#else
  setenv("NSOS_MAMBA_GPU_STEP", "1", 1);
#endif

  return run_parity("mamba_proper_stream", [] {
    const int D = 8, N = 8, H = 1, L = 6;
    MambaConfig cfg;
    cfg.proper_selective_ssm = true;
    cfg.conv_kernel = 3;
    Mamba2SSD layer(D, N, H, cfg);  // CPU weights at construction
    layer.to(Device::GPU);

    Tensor x({L, D}, Device::CPU);
    for (int i = 0; i < x.size; ++i) {
      x.data()[i] = std::sin(0.3f * static_cast<float>(i + 1));
    }
    Tensor x_gpu = x.to(Device::GPU);

    // Reference: full-sequence scan on the GPU (streaming off).
    Tensor y_full = layer.forward(x_gpu);  // [L, D] on GPU
    Tensor y_full_h = y_full.cpu();

    // Under test: prefill token 0, then on-device single-token steps.
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
    layer.set_streaming_mode(false);

    // Both paths are GPU kernels (full-scan vs fused step); tolerance covers the
    // small fp reordering between them.
    assert_close(y_full_h, y_stream_h, 2e-3f,
                 "mamba_proper GPU step == full scan");
  });
}
