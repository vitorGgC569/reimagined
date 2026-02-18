#include "bitlinear.h"
#include "jamba.h"
#include "memory_system.h"
#include "nsos_sdk.h"
#include "tensor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace nsos;

// High-Precision Timer
class Timer {
  std::chrono::high_resolution_clock::time_point start_time;

public:
  Timer() { reset(); }
  void reset() { start_time = std::chrono::high_resolution_clock::now(); }
  double elapsed_ms() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time)
        .count();
  }
};

void print_header(const std::string &title) {
  std::cout << "\n========================================" << std::endl;
  std::cout << " BENCHMARK: " << title << std::endl;
  std::cout << "========================================" << std::endl;
}

void bench_kan_layer() {
  print_header("BitFastKANLayer (Core Kernel)");
  int batch = 32;
  int dim = 512;
  int iterations = 100;

  std::cout << "Config: Batch=" << batch << ", Dim=" << dim
            << ", Iters=" << iterations << std::endl;

  Tensor x = Tensor::random({batch, dim}, Device::CPU);
  BitFastKANLayer kan(dim, dim); // In/Out

  // Warmup
  kan.forward(x);

  Timer t;
  for (int i = 0; i < iterations; ++i) {
    Tensor out = kan.forward(x);
    // Force sync if we were on GPU (currently CPU for this bench)
  }
  double total_ms = t.elapsed_ms();
  double avg_ms = total_ms / iterations;

  std::cout << "Total Time: " << total_ms << " ms" << std::endl;
  std::cout << "Avg Latency: " << avg_ms << " ms/iter" << std::endl;
  std::cout << "Throughput: " << (batch * iterations) / (total_ms / 1000.0)
            << " samples/sec" << std::endl;
}

void bench_mamba_ssd() {
  print_header("Mamba2 SSD (Sequence Modeling)");
  int batch = 4;
  int seq_len = 1024;
  int d_model = 256;
  int iterations = 10;

  std::cout << "Config: Batch=" << batch << ", Seq=" << seq_len
            << ", Dim=" << d_model << std::endl;

  Tensor x = Tensor::random({batch, seq_len, d_model}, Device::CPU);
  Mamba2SSD mamba(d_model, 16, 4); // d_model, d_state, n_heads

  // Warmup
  try {
    mamba.forward(x);
  } catch (const std::exception &e) {
    std::cerr << "Initialization failed: " << e.what() << std::endl;
    return;
  }

  Timer t;
  for (int i = 0; i < iterations; ++i) {
    Tensor out = mamba.forward(x);
  }
  double total_ms = t.elapsed_ms();

  double tokens = (double)batch * seq_len * iterations;
  double tps = tokens / (total_ms / 1000.0);

  std::cout << "Total Tokens: " << (size_t)tokens << std::endl;
  std::cout << "Total Time: " << total_ms << " ms" << std::endl;
  std::cout << "Tokens/Sec: " << tps << " tok/s" << std::endl;
}

void bench_jamba_full() {
  print_header("JambaModel (End-to-End Inference)");
  // Small model for CPU benchmarking
  int layers = 2;
  int d_model = 128;
  int vocab = 1000;

  JambaModel model(layers, d_model, vocab, Device::CPU);

  std::string prompt = "Hello world this is a benchmark";
  // Mock tokenization
  std::vector<int> input_ids(20, 1);

  Timer t;
  Tensor out = model.forward_ids(input_ids);
  double ms = t.elapsed_ms();

  std::cout << "Forward Pass (20 tokens): " << ms << " ms" << std::endl;
}

int main() {
  std::cout << "NSOS MICRO-BENCHMARKS" << std::endl;
  std::cout << "Device: CPU (Default)" << std::endl;

  try {
    bench_kan_layer();
    bench_mamba_ssd();
    bench_jamba_full();
  } catch (const std::exception &e) {
    std::cerr << "CRITICAL ERROR DURING BENCHMARK: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
