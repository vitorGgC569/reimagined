// bench_gpu_vs_cpu — direct CPU vs GPU performance benchmark for the
// operations the GPU optimization plan touches.  Uses chrono::steady_clock
// for wall time and forces a CUDA sync after every measured call so the
// numbers reflect kernel runtime, not async dispatch.
//
// Output is a JSON object on stdout.  Optional --report writes the same
// JSON to disk.  No model is loaded — every layer is constructed with
// random init in-process, so this is a pure microbenchmark of the
// kernels added in Phase 3 / 4 / 5.

#include "bitlinear.h"
#include "jamba.h"
#include "mamba2.h"
#include "tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

void cuda_sync() {
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
}

double sec_since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

struct Stat {
  std::string label;
  int repeat = 0;
  double mean_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
};

template <typename Fn>
Stat time_call(const std::string& label, Fn&& fn, int repeat = 20,
               int warmup = 3) {
  for (int i = 0; i < warmup; ++i) {
    fn();
    cuda_sync();
  }
  Stat s;
  s.label = label;
  s.repeat = repeat;
  s.min_s = std::numeric_limits<double>::infinity();
  s.max_s = 0.0;
  double total = 0.0;
  for (int i = 0; i < repeat; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    cuda_sync();
    const double dt = sec_since(t0);
    s.min_s = std::min(s.min_s, dt);
    s.max_s = std::max(s.max_s, dt);
    total += dt;
  }
  s.mean_s = total / std::max(repeat, 1);
  return s;
}

void mirror_parameters(const std::vector<nsos::Parameter*>& src,
                       const std::vector<nsos::Parameter*>& dst) {
  for (size_t i = 0; i < std::min(src.size(), dst.size()); ++i) {
    if (!src[i] || !dst[i]) continue;
    nsos::Tensor cpu_src = src[i]->data.get_device() == nsos::Device::GPU
                               ? src[i]->data.cpu()
                               : src[i]->data;
    if (dst[i]->data.get_device() == nsos::Device::GPU) {
      dst[i]->data.copy_from(cpu_src.to(nsos::Device::GPU));
    } else {
      dst[i]->data.copy_from(cpu_src);
    }
  }
}

struct Bench {
  std::string op;
  std::string shape;
  Stat cpu;
  Stat gpu;
  double speedup_cpu_div_gpu() const {
    return gpu.mean_s > 0 ? cpu.mean_s / gpu.mean_s : 0.0;
  }
};

void emit_json_field(std::ostream& out, const std::string& key,
                      const Stat& s, bool comma_after) {
  out << "      \"" << key << "\": {"
      << "\"label\": \"" << s.label << "\", "
      << "\"repeat\": " << s.repeat << ", "
      << "\"mean_s\": " << std::scientific << std::setprecision(6) << s.mean_s
      << ", \"min_s\": " << s.min_s
      << ", \"max_s\": " << s.max_s << "}";
  if (comma_after) out << ",";
  out << "\n";
}

// ────────────────────────────────────────────────────────────────────
// Individual benches
// ────────────────────────────────────────────────────────────────────

Bench bench_matmul(int M, int K, int N, int repeat) {
  nsos::Tensor a_cpu = nsos::Tensor::random({M, K}, nsos::Device::CPU);
  nsos::Tensor b_cpu = nsos::Tensor::random({K, N}, nsos::Device::CPU);
  nsos::Tensor a_gpu = a_cpu.to(nsos::Device::GPU);
  nsos::Tensor b_gpu = b_cpu.to(nsos::Device::GPU);

  std::ostringstream sh;
  sh << M << "x" << K << "x" << N;
  Bench b;
  b.op = "matmul";
  b.shape = sh.str();
  b.cpu = time_call(
      "matmul[" + sh.str() + "] cpu",
      [&] { (void)a_cpu.matmul(b_cpu); },
      repeat);
  b.gpu = time_call(
      "matmul[" + sh.str() + "] gpu",
      [&] { (void)a_gpu.matmul(b_gpu); },
      repeat);
  return b;
}

Bench bench_bitlinear(int in_features, int out_features, int batch, int repeat,
                       bool dp4a) {
  nsos::BitLinear cpu_layer(in_features, out_features, /*bias=*/true);
  nsos::BitLinear gpu_layer(in_features, out_features, /*bias=*/true);

  auto cpu_params = cpu_layer.parameters();
  auto gpu_params = gpu_layer.parameters();
  mirror_parameters(cpu_params, gpu_params);

  gpu_layer.to(nsos::Device::GPU);
  cpu_layer.set_precision_mode(2);
  gpu_layer.set_precision_mode(2);
  cpu_layer.set_reference_path(false);
  gpu_layer.set_reference_path(false);
  cpu_layer.repack_weights();
  gpu_layer.repack_weights();
  if (dp4a) gpu_layer.set_gpu_packed_inference(true);

  nsos::Tensor x_cpu = nsos::Tensor::random({batch, in_features},
                                              nsos::Device::CPU);
  nsos::Tensor x_gpu = x_cpu.to(nsos::Device::GPU);

  std::ostringstream sh;
  sh << batch << "x" << in_features << "->" << out_features;
  Bench b;
  b.op = dp4a ? "bitlinear_dp4a" : "bitlinear_float";
  b.shape = sh.str();
  b.cpu = time_call(
      "bitlinear[" + sh.str() + "] cpu_packed",
      [&] { (void)cpu_layer.forward(x_cpu); },
      repeat);
  b.gpu = time_call(
      std::string("bitlinear[") + sh.str() + "] " + (dp4a ? "gpu_dp4a" : "gpu_float"),
      [&] { (void)gpu_layer.forward(x_gpu); },
      repeat);
  return b;
}

Bench bench_mamba(int d_model, int d_state, int n_heads, int batch, int seq,
                   int repeat) {
  nsos::Mamba2SSD cpu_layer(d_model, d_state, n_heads);
  nsos::Mamba2SSD gpu_layer(d_model, d_state, n_heads);

  auto cpu_params = cpu_layer.parameters();
  auto gpu_params = gpu_layer.parameters();
  mirror_parameters(cpu_params, gpu_params);

  gpu_layer.to(nsos::Device::GPU);

  nsos::Tensor x_cpu = nsos::Tensor::random({batch, seq, d_model},
                                              nsos::Device::CPU);
  nsos::Tensor x_gpu = x_cpu.to(nsos::Device::GPU);

  std::ostringstream sh;
  sh << batch << "x" << seq << "x" << d_model;
  Bench b;
  b.op = "mamba_selective_scan";
  b.shape = sh.str();
  b.cpu = time_call(
      "mamba[" + sh.str() + "] cpu",
      [&] { (void)cpu_layer.forward(x_cpu, /*ctx=*/nullptr); },
      repeat);
  b.gpu = time_call(
      "mamba[" + sh.str() + "] gpu",
      [&] { (void)gpu_layer.forward(x_gpu, /*ctx=*/nullptr); },
      repeat);
  return b;
}

Bench bench_moe_router(int d_model, int num_experts, int top_k, int rows,
                        int repeat) {
  nsos::MoERouter cpu_router(d_model, num_experts, top_k);
  nsos::MoERouter gpu_router(d_model, num_experts, top_k);

  auto cpu_params = cpu_router.parameters();
  auto gpu_params = gpu_router.parameters();
  mirror_parameters(cpu_params, gpu_params);

  gpu_router.to(nsos::Device::GPU);

  nsos::Tensor x_cpu = nsos::Tensor::random({rows, d_model}, nsos::Device::CPU);
  nsos::Tensor x_gpu = x_cpu.to(nsos::Device::GPU);

  std::ostringstream sh;
  sh << rows << "x" << d_model << "->" << num_experts << "@" << top_k;
  Bench b;
  b.op = "moe_router";
  b.shape = sh.str();
  b.cpu = time_call(
      "moe_router[" + sh.str() + "] cpu",
      [&] { (void)cpu_router.forward(x_cpu); },
      repeat);
  b.gpu = time_call(
      "moe_router[" + sh.str() + "] gpu_fastpath",
      [&] { (void)gpu_router.forward(x_gpu); },
      repeat);
  return b;
}

void print_bench(std::ostream& out, const Bench& b) {
  out << "  " << std::setw(24) << std::left << b.op
      << " " << std::setw(20) << std::left << b.shape
      << "  cpu=" << std::fixed << std::setprecision(3)
      << (b.cpu.mean_s * 1e3) << "ms"
      << "  gpu=" << (b.gpu.mean_s * 1e3) << "ms"
      << "  speedup=" << std::setprecision(2)
      << b.speedup_cpu_div_gpu() << "x"
      << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef USE_CUDA
  std::cerr << "[bench] CUDA not enabled at build time" << std::endl;
  return 2;
#else
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::cerr << "[bench] no CUDA device available" << std::endl;
    return 2;
  }
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, 0);
  std::cout << "[bench] device=" << props.name << " cc=" << props.major
            << "." << props.minor << " runtime=" << CUDART_VERSION
            << std::endl;

  int repeat = 10;
  std::string report_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--repeat" && i + 1 < argc) {
      repeat = std::stoi(argv[++i]);
    } else if (arg == "--report" && i + 1 < argc) {
      report_path = argv[++i];
    }
  }

  std::vector<Bench> all;

  std::cout << "\n=== matmul ===" << std::endl;
  for (auto [M, K, N] : std::vector<std::tuple<int, int, int>>{
           {64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {512, 512, 512}}) {
    Bench b = bench_matmul(M, K, N, repeat);
    print_bench(std::cout, b);
    all.push_back(b);
  }

  std::cout << "\n=== bitlinear ===" << std::endl;
  for (auto [in_f, out_f, batch] : std::vector<std::tuple<int, int, int>>{
           {64, 64, 4}, {128, 128, 8}, {256, 256, 16}, {512, 512, 32}}) {
    Bench b1 = bench_bitlinear(in_f, out_f, batch, repeat, /*dp4a=*/false);
    print_bench(std::cout, b1);
    all.push_back(b1);
    Bench b2 = bench_bitlinear(in_f, out_f, batch, repeat, /*dp4a=*/true);
    print_bench(std::cout, b2);
    all.push_back(b2);
  }

  std::cout << "\n=== mamba ===" << std::endl;
  for (auto [d_model, d_state, n_heads, batch, seq] :
       std::vector<std::tuple<int, int, int, int, int>>{
           {64, 16, 4, 1, 32}, {128, 16, 4, 1, 64}, {256, 16, 4, 1, 128}}) {
    Bench b = bench_mamba(d_model, d_state, n_heads, batch, seq, repeat);
    print_bench(std::cout, b);
    all.push_back(b);
  }

  std::cout << "\n=== moe_router ===" << std::endl;
  for (auto [d_model, num_experts, top_k, rows] :
       std::vector<std::tuple<int, int, int, int>>{
           {64, 8, 2, 32}, {128, 8, 2, 64}, {256, 16, 2, 128}}) {
    Bench b = bench_moe_router(d_model, num_experts, top_k, rows, repeat);
    print_bench(std::cout, b);
    all.push_back(b);
  }

  if (!report_path.empty()) {
    std::ofstream out(report_path);
    out << "{\n  \"device\": \"" << props.name
        << "\", \"cc\": " << props.major << props.minor
        << ", \"runtime\": " << CUDART_VERSION
        << ",\n  \"results\": [\n";
    for (size_t i = 0; i < all.size(); ++i) {
      const auto& b = all[i];
      out << "    {\n"
          << "      \"op\": \"" << b.op << "\",\n"
          << "      \"shape\": \"" << b.shape << "\",\n";
      emit_json_field(out, "cpu", b.cpu, true);
      emit_json_field(out, "gpu", b.gpu, true);
      out << "      \"speedup_cpu_div_gpu\": " << std::scientific
          << std::setprecision(6) << b.speedup_cpu_div_gpu() << "\n"
          << "    }";
      if (i + 1 < all.size()) out << ",";
      out << "\n";
    }
    out << "  ]\n}\n";
    std::cout << "\n[bench] report → " << report_path << std::endl;
  }

  return 0;
#endif
}
