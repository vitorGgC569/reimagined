// Parity gate (Fase 2): opt-in Mamba2 parallel-prefix scan vs the validated
// channel-parallel sequential selective-scan kernel.
//
// Both compute the same recurrence per (batch,dim) channel:
//   h_t = decay_t * h_{t-1} + softplus(dt_t) * B_t * x_t
//   y_t = tanh(h_t) * C_t
// The parallel kernel reassociates the floating-point summation (Hillis-Steele
// scan), so results must agree with the sequential reference within 1e-4.
//
// This binary is fail-closed: a non-CUDA build or missing device is a failure.
// CPU-only CTest lanes must not register it as a GPU validation gate.
#include "tensor.h"
#include "cuda/mamba_kernels.cuh"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace nsos;

int main() {
#ifndef USE_CUDA
  std::fprintf(
      stderr,
      "[mamba_parallel_scan_parity] CUDA required but disabled\n");
  return 1;
#else
  int dev_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&dev_count);
  if (device_status != cudaSuccess || dev_count == 0) {
    std::fprintf(
        stderr,
        "[mamba_parallel_scan_parity] CUDA device required but unavailable "
        "(cudaGetDeviceCount=%d)\n",
        static_cast<int>(device_status));
    return 1;
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
  cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    std::fprintf(stderr, "[mamba_parallel_scan_parity] sequential sync failed: %s\n",
                 cudaGetErrorString(sync_status));
    return 1;
  }
  Tensor y_seq_h = y_seq.to(Device::CPU);

  // Parallel-prefix path under test.
  cuda::set_mamba_parallel_scan(true);
  cuda::launch_mamba_selective_scan_forward(
      xg.raw_data(), dtg.raw_data(), Ag.raw_data(), Bg.raw_data(), Cg.raw_data(),
      y_par.raw_data(), nullptr, Batch, Seq, D);
  sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    std::fprintf(stderr, "[mamba_parallel_scan_parity] parallel sync failed: %s\n",
                 cudaGetErrorString(sync_status));
    return 1;
  }
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
