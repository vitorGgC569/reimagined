// Micro-bench: blocked SIMD MathOps::gemm vs the old naive matmul triple loop.
// Links against the already-built nsos_core.lib (gemm compiled with AVX).
#include "../include/nsos_math.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace nsos;

static void naive(int M, int N, int K, const float* A, const float* B, float* C) {
#pragma omp parallel for
  for (int r = 0; r < M; ++r) {
    float* o = C + static_cast<size_t>(r) * N;
    std::fill_n(o, N, 0.0f);
    const float* a = A + static_cast<size_t>(r) * K;
    for (int kk = 0; kk < K; ++kk) {
      const float av = a[kk];
      const float* b = B + static_cast<size_t>(kk) * N;
      for (int c = 0; c < N; ++c) o[c] += av * b[c];
    }
  }
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  const int sizes[][3] = {{256, 256, 256}, {512, 512, 512}, {768, 768, 3072}, {1024, 1024, 1024}};

  std::printf("  M    N    K   | naive ms | gemm ms | speedup | gemm GFLOP/s | max|diff|\n");
  for (auto& s : sizes) {
    const int M = s[0], N = s[1], K = s[2];
    std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
    std::vector<float> Cn(static_cast<size_t>(M) * N), Cg(static_cast<size_t>(M) * N);
    for (auto& x : A) x = u(rng);
    for (auto& x : B) x = u(rng);

    const int reps = (static_cast<long long>(M) * N * K > 200000000LL) ? 3 : 8;
    naive(M, N, K, A.data(), B.data(), Cn.data());  // warm
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) naive(M, N, K, A.data(), B.data(), Cn.data());
    const double tn = (now_s() - t0) / reps;

    MathOps::gemm(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, Cg.data(), N);  // warm
    t0 = now_s();
    for (int r = 0; r < reps; ++r)
      MathOps::gemm(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, Cg.data(), N);
    const double tg = (now_s() - t0) / reps;

    float maxd = 0.0f;
    for (size_t i = 0; i < Cn.size(); ++i) maxd = std::max(maxd, std::abs(Cn[i] - Cg[i]));
    const double gflops = 2.0 * M * N * K / tg / 1e9;
    std::printf(" %4d %4d %4d | %8.2f | %7.2f | %6.2fx | %11.1f | %.2e\n",
                M, N, K, tn * 1e3, tg * 1e3, tn / tg, gflops, maxd);
  }
  return 0;
}
