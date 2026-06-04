// Parity gate (Fase 2): opt-in Mamba2 parallel-prefix scan vs the validated
// channel-parallel sequential selective-scan kernel.
//
// Both compute the same recurrence per (batch,dim) channel:
//   h_t = decay_t * h_{t-1} + B_t * x_t ,  y_t = tanh(h_t) * C_t
// The parallel kernel reassociates the floating-point summation (Hillis-Steele
// scan), so results must agree with the sequential reference within 1e-4.
//
// Skips cleanly (exit 0) when CUDA is disabled or no GPU is present, so it is
// safe in the default CTest run; on a real GPU (Colab T4) it is the one-command
// gate that promotes NSOS_MAMBA_PARALLEL_SCAN to default:  ctest -R parallel_scan
#include "tensor.h"
#include "cuda/mamba_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace nsos;

int main() {
#ifndef USE_CUDA
  std::printf("[mamba_parallel_scan_parity] CUDA disabled at build time — skip\n");
  return 0;
#else
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    std::printf("[mamba_parallel_scan_parity] no CUDA device — skip\n");
    return 0;
  }

  const int Batch = 2, Seq = 192, D = 48;
  const int N = Batch * Seq * D;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);

  auto make = [&]() {
    Tensor t({Batch, Seq, D}, Device::CPU);
    float *p = t.data();
    for (int i = 0; i < N; ++i) p[i] = u(rng);
    return t;
  };
  Tensor x = make(), dt = make(), B = make(), C = make();
  Tensor A({D}, Device::CPU);
  for (int d = 0; d < D; ++d) A.data()[d] = std::fabs(u(rng)) + 0.05f;

  Tensor xg = x.to(Device::GPU), dtg = dt.to(Device::GPU), Bg = B.to(Device::GPU),
         Cg = C.to(Device::GPU), Ag = A.to(Device::GPU);
  Tensor y_seq({Batch, Seq, D}, Device::GPU);
  Tensor y_par({Batch, Seq, D}, Device::GPU);

  // Sequential reference.
  cuda::set_mamba_parallel_scan(false);
  cuda::launch_mamba_selective_scan_forward(
      xg.raw_data(), dtg.raw_data(), Ag.raw_data(), Bg.raw_data(), Cg.raw_data(),
      y_seq.raw_data(), nullptr, Batch, Seq, D);
  cudaDeviceSynchronize();
  Tensor y_seq_h = y_seq.to(Device::CPU);

  // Parallel-prefix path under test.
  cuda::set_mamba_parallel_scan(true);
  cuda::launch_mamba_selective_scan_forward(
      xg.raw_data(), dtg.raw_data(), Ag.raw_data(), Bg.raw_data(), Cg.raw_data(),
      y_par.raw_data(), nullptr, Batch, Seq, D);
  cudaDeviceSynchronize();
  Tensor y_par_h = y_par.to(Device::CPU);
  cuda::set_mamba_parallel_scan(false);  // restore default

  const float *a = y_seq_h.data();
  const float *b = y_par_h.data();
  float max_diff = 0.0f;
  for (int i = 0; i < N; ++i) {
    max_diff = std::fmax(max_diff, std::fabs(a[i] - b[i]));
  }
  const float tol = 1e-4f;
  std::printf("[mamba_parallel_scan_parity] Batch=%d Seq=%d D=%d  max_abs_diff=%.3e  tol=%.1e\n",
              Batch, Seq, D, max_diff, tol);
  if (max_diff <= tol) {
    std::printf("[mamba_parallel_scan_parity] PASS\n");
    return 0;
  }
  std::printf("[mamba_parallel_scan_parity] FAIL (max_abs_diff > tol)\n");
  return 1;
#endif
}
