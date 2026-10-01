// Isolated synthetic M=1 experiment. Not model throughput or a quality test.
#include "bitnet_gpu_dispatch.h"
#include "gpu_backend.h"
#include "gpu_execution.h"
#include "nsos/sha256.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nsos;
namespace {
void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Event {
  cudaEvent_t value = nullptr;
  Event() { checked(cudaEventCreate(&value)); }
  ~Event() { if (value) gpu::report_cleanup_status(cudaEventDestroy(value), "benchmark event"); }
};
}

int main(int argc, char** argv) {
  try {
    const int k = argc > 1 ? std::stoi(argv[1]) : 1024;
    const int n = argc > 2 ? std::stoi(argv[2]) : 4096;
    if (argc > 3 || k < 16 || k > 16384 || k % 16 || n < 1 || n > 32768)
      throw std::invalid_argument("Usage: bench_decode_gemv [K multiple of 16, <=16384] [N <=32768]");
    int selected = -1;
    std::string error;
    if (!gpu::select_preferred_device(&selected, &error)) throw std::runtime_error(error);
    const auto devices = gpu::enumerate_devices();
    const char* experiment = std::getenv("NSOS_GPU_EXPERIMENT");
    if (!experiment) experiment = "none";
    Tensor x({1, k}, Device::CPU);
    float maximum = 0;
    for (int i = 0; i < k; ++i) {
      x.data()[i] = std::sin(float(i) * .013f) + .25f * std::cos(float(i) * .037f);
      maximum = std::max(maximum, std::abs(x.data()[i]));
    }
    const float inverse_scale = (maximum + 1e-8f) / 127.f;
    const float scale = 127.f / (maximum + 1e-8f);
    std::vector<int> quantized(k);
    for (int i = 0; i < k; ++i)
      quantized[i] = int(std::nearbyint(std::clamp(x.data()[i] * scale, -127.f, 127.f)));
    std::vector<uint32_t> packed(size_t(n) * k / 16);
    uint32_t random = 20260919;
    for (auto& word : packed) for (int j = 0; j < 16; ++j) {
      random = random * 1664525u + 1013904223u;
      word |= ((random >> 16) % 3) << (2 * j);
    }
    Tensor weights({int(packed.size())}, Device::CPU);
    std::memcpy(weights.data(), packed.data(), packed.size() * sizeof(uint32_t));
    Tensor xg = x.to(Device::GPU), wg = weights.to(Device::GPU);
    Tensor y = bitnet_gemm_158bit_gpu(xg, wg, 1.f, 1, k, n, 8);
    Tensor actual = y.cpu();
    float maximum_error = 0;
    for (int row = 0; row < n; ++row) {
      int sum = 0;
      for (int col = 0; col < k; ++col) {
        const int weight = int((packed[size_t(row) * (k / 16) + col / 16] >> (2 * (col % 16))) & 3u) - 1;
        sum += quantized[col] * weight;
      }
      const float expected = float(sum) * inverse_scale;
      const float difference = std::abs(actual.data()[row] - expected);
      maximum_error = std::max(maximum_error, difference);
      if (!std::isfinite(actual.data()[row]) || difference > 1e-4f + 1e-5f * std::abs(expected))
        throw std::runtime_error("Packed GEMV disagrees with the CPU integer oracle");
    }
    constexpr int warmup = 20, iterations = 100, repeats = 7;
    for (int i = 0; i < warmup; ++i) y = bitnet_gemm_158bit_gpu(xg, wg, 1.f, 1, k, n, 8);
    checked(cudaDeviceSynchronize());
    Event start, end;
    std::vector<double> wall_samples, gpu_samples;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto begin = std::chrono::steady_clock::now();
      checked(cudaEventRecord(start.value, gpu::current_stream()));
      for (int i = 0; i < iterations; ++i) y = bitnet_gemm_158bit_gpu(xg, wg, 1.f, 1, k, n, 8);
      checked(cudaEventRecord(end.value, gpu::current_stream()));
      checked(cudaEventSynchronize(end.value));
      wall_samples.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations);
      float milliseconds = 0;
      checked(cudaEventElapsedTime(&milliseconds, start.value, end.value));
      gpu_samples.push_back(double(milliseconds) * 1000 / iterations);
    }
    const auto counters = gpu::dispatch_counters();
    if (counters[unsigned(gpu::DispatchPath::BitnetGemv)] < repeats * iterations)
      throw std::runtime_error("GEMV dispatch was not active");
    auto ordered = wall_samples;
    std::sort(ordered.begin(), ordered.end());
    std::cout << std::setprecision(10)
      << "{\"kind\":\"synthetic_packed_gemv_only\",\"backend\":" << std::quoted(gpu::backend_name())
      << ",\"device\":" << selected << ",\"experiment\":" << std::quoted(experiment)
      << ",\"binary_sha256\":" << std::quoted(integrity::sha256_file(argv[0]))
      << ",\"weights_sha256\":" << std::quoted(integrity::sha256_hex(packed.data(), packed.size() * 4))
      << ",\"input_sha256\":" << std::quoted(integrity::sha256_hex(x.data(), size_t(k) * 4))
      << ",\"K\":" << k << ",\"N\":" << n << ",\"activation_bits\":8,\"weight_bits\":2"
      << ",\"packed_bytes\":" << packed.size() * 4 << ",\"max_absolute_error\":" << maximum_error
      << ",\"warmup\":" << warmup << ",\"iterations_per_sample\":" << iterations
      << ",\"median_wall_us\":" << ordered[ordered.size()/2] << ",\"wall_us\":[";
    for (size_t i = 0; i < wall_samples.size(); ++i) std::cout << (i ? "," : "") << wall_samples[i];
    std::cout << "],\"stream_elapsed_us_including_submission_gaps\":[";
    for (size_t i = 0; i < gpu_samples.size(); ++i) std::cout << (i ? "," : "") << gpu_samples[i];
    std::cout << "],\"quality_validated\":false}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GEMV experiment failed: " << error.what() << std::endl;
    return 1;
  }
}
