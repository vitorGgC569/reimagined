// Validates the BLIS-style MathOps::gemm (a packed + AVX2 micro-kernel that
// was previously dead code) against the naive, already-trusted Tensor::matmul.
// This de-risks wiring MathOps::gemm into the production matmul path: prove it
// is numerically correct across multiple shapes (including non-multiples of the
// MR=6 / NR=16 micro-kernel tile) BEFORE making it the default.
#include "../include/nsos_math.h"
#include "../include/tensor.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nsos;

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

void check_shape(int M, int N, int K) {
  Tensor A({M, K}, Device::CPU);
  Tensor B({K, N}, Device::CPU);
  for (int i = 0; i < A.size; ++i) A.data()[i] = 0.5f * std::sin(0.3f * i + 1.0f);
  for (int i = 0; i < B.size; ++i) B.data()[i] = 0.4f * std::cos(0.21f * i + 0.5f);

  Tensor ref = A.matmul(B);  // naive, trusted

  std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
  MathOps::gemm(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);

  float max_abs = 0.0f, max_rel = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    const float r = ref.data()[i];
    const float g = C[static_cast<size_t>(i)];
    const float a = std::abs(r - g);
    max_abs = std::max(max_abs, a);
    max_rel = std::max(max_rel, a / (std::abs(r) + 1e-4f));
  }
  std::printf("[gemm] M=%-4d N=%-4d K=%-4d  max|abs|=%.3e  max|rel|=%.3e\n",
              M, N, K, max_abs, max_rel);
  require(max_rel < 1e-3f, "MathOps::gemm disagrees with naive matmul");
}

}  // namespace

int main() {
  try {
    check_shape(6, 16, 8);    // exact micro-kernel tile (MR=6, NR=16)
    check_shape(5, 15, 7);    // non-multiples of MR/NR -> tail handling
    check_shape(1, 1, 1);     // degenerate
    check_shape(7, 3, 5);     // small, ragged
    check_shape(32, 32, 32);  // larger square
    check_shape(13, 40, 64);  // mixed
    std::cout << "MathOps::gemm parity test passed!" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "GEMM parity test failed: " << ex.what() << std::endl;
    return 1;
  }
}
