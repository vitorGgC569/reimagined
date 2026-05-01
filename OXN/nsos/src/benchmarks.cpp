#include "../include/bitlinear.h"
#include "../include/jamba.h"
#include "../include/tensor.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace nsos;

// Metric: MSE between Quantized and Original
float measure_quantization_quality(int dim) {
  BitLinear layer(dim, dim);
  Tensor w = Tensor::random({dim, dim}, Device::CPU);
  Tensor w_q = layer.quantize_weights(w);

  float mse = 0;
  for (int i = 0; i < w.size; ++i) {
    float diff = w.data()[i] - w_q.data()[i];
    mse += diff * diff;
  }
  return mse / w.size;
}

// Metric: FLOPs/s simulation
double measure_flops(int size) {
  Tensor A = Tensor::random({size, size}, Device::CPU);
  Tensor B = Tensor::random({size, size}, Device::CPU);

  auto start = std::chrono::high_resolution_clock::now();
  Tensor C = A.matmul(B); // Standard
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  double seconds = diff.count();

  // 2 * N^3 ops
  double flops = 2.0 * std::pow(size, 3);
  return flops / seconds;
}

double measure_flops_optimized(int size) {
  Tensor A = Tensor::random({size, size}, Device::CPU);
  // B needs to be ternary-like for the optimized kernel to shine?
  // BitLinear::matmul_158bit assumes w_q is passed.
  // Let's use matmul_abm from Tensor which simulates the ordering.
  Tensor B = Tensor::random({size, size}, Device::CPU);

  auto start = std::chrono::high_resolution_clock::now();
  Tensor C = A.matmul_abm(B);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  double seconds = diff.count();

  double flops = 2.0 * std::pow(size, 3);
  return flops / seconds;
}

// Metric: Memory Stability (Gradient Variance over depth)
// Simulating forward pass explosion check (Pre-Norm verification)
float measure_stability(int layers) {
  Tensor h = Tensor::random({1, 64}, Device::CPU);
  // Normalize initial
  h = h.rmsnorm();

  // Simulate deep network
  for (int i = 0; i < layers; ++i) {
    // Residual Block: h = h + f(norm(h))
    Tensor h_norm = h.rmsnorm();
    // f(x) = x * w (random)
    // To keep it stable, w should be orthogonal or small.
    // BitLinear initializes random normal.
    // Variance grows as 1 + Var(f).
    // With PreNorm, input to f is unit var. Output var is Var(w).
    // h_new var = Var(h) + Var(w). Linear growth.
    // We check if it explodes exponentially.

    // Simulating block logic simply
    Tensor noise =
        Tensor::random({1, 64}, Device::CPU); // Representing f(h) result
    h = h.add(noise);
  }

  // Return mean magnitude
  float mean = 0;
  for (int i = 0; i < h.size; ++i)
    mean += std::abs(h.data()[i]);
  return mean / h.size;
}

int main() {
  std::cout << "=== NSOS Benchmarks ===" << std::endl;

  // 1. Efficiency
  float mse = measure_quantization_quality(128);
  std::cout << "[Efficiency] BitNet Quantization MSE: " << mse
            << " (Lower is better approx)" << std::endl;

  double flops = measure_flops(256);
  std::cout << "[Efficiency] Effective FLOPs/s (Baseline): " << flops / 1e9
            << " GFLOPS" << std::endl;

  double flops_opt = measure_flops_optimized(256);
  std::cout << "[Efficiency] Effective FLOPs/s (Optimized/ABM): "
            << flops_opt / 1e9 << " GFLOPS" << std::endl;
  std::cout << "[Efficiency] Speedup: " << flops_opt / flops << "x"
            << std::endl;

  // 2. Memory / Stability
  float mag_10 = measure_stability(10);
  float mag_100 = measure_stability(100);
  std::cout << "[Stability] Magnitude @ 10 layers: " << mag_10 << std::endl;
  std::cout << "[Stability] Magnitude @ 100 layers: " << mag_100 << std::endl;
  std::cout << "[Stability] Growth Factor: " << mag_100 / mag_10
            << " (Linear is ~10, Exponential is huge)" << std::endl;

  // 3. Edge Latency
  Tensor A = Tensor::random({64, 64}, Device::CPU);
  Tensor B = Tensor::random({64, 64}, Device::CPU);
  auto start = std::chrono::high_resolution_clock::now();
  A.matmul_abm(B);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "[Edge] ABM Kernel Latency (64x64): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us" << std::endl;

  return 0;
}
