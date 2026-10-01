#include "gpu_parity_common.h"
#include "bitnet_gpu_dispatch.h"
#include "cuda/kernels.cuh"
#include "cuda/mamba_kernels.cuh"
#include "gpu_execution.h"
#include "gpu_kv_cache.h"
#include "gpu_projection_group.h"
#include "jamba.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;
namespace {
struct Environment {
  std::string key, old;
  bool existed;
  Environment(const char* name, const char* value) : key(name), existed(std::getenv(name) != nullptr) {
    if (existed) old = std::getenv(name);
    set(value);
  }
  void set(const char* value) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), value)) throw std::runtime_error("Cannot set test environment");
#else
    if (setenv(key.c_str(), value, 1)) throw std::runtime_error("Cannot set test environment");
#endif
  }
  ~Environment() {
#ifdef _WIN32
    (void)_putenv_s(key.c_str(), existed ? old.c_str() : "");
#else
    if (existed) (void)setenv(key.c_str(), old.c_str(), 1); else (void)unsetenv(key.c_str());
#endif
  }
};
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
Tensor pattern(std::vector<int> shape, float frequency = .013f) {
  Tensor t(shape, Device::CPU);
  for (int64_t i = 0; i < t.size; ++i) t.data()[i] = std::sin(float(i + 1) * frequency) * .4f;
  return t;
}

void gemv_contract() {
  Environment experiment("NSOS_GPU_EXPERIMENT", "none");
  gpu::ExecutionContext context;
  gpu::ExecutionContext::Scope scope(context);
  for (const int K : {16, 48, 512, 1024}) for (const int N : {3, 17, 33}) {
    Tensor input = pattern({1, K}, .071f);
    Tensor packed({N * K / 16}, Device::CPU);
    auto* words = reinterpret_cast<uint32_t*>(packed.data());
    for (int n = 0; n < N; ++n) for (int kb = 0; kb < K / 16; ++kb) {
      uint32_t word = 0;
      for (int bit = 0; bit < 16; ++bit) word |= uint32_t((n + kb * 16 + bit) % 3) << (bit * 2);
      words[n * K / 16 + kb] = word;
    }
    Tensor magnitude = pattern({N}, .12f), bias = pattern({N}, .1f);
    Tensor x = input.to(Device::GPU), w = packed.to(Device::GPU);
    Tensor m = magnitude.to(Device::GPU), b = bias.to(Device::GPU);
    for (int bits : {2, 4, 8}) {
      const float qmax = float((1 << (bits - 1)) - 1);
      float maximum = 0;
      for (int k = 0; k < K; ++k) maximum = std::max(maximum, std::abs(input.data()[k]));
      const float denominator = maximum + 1e-8f;
      Tensor reference({1, N}, Device::CPU);
      for (int n = 0; n < N; ++n) {
        int sum = 0;
        for (int k = 0; k < K; ++k)
          sum += int(std::round(input.data()[k] * (qmax / denominator))) * ((n + k) % 3 - 1);
        reference.data()[n] = ((float(sum) * .3f) * (denominator / qmax)) * magnitude.data()[n] + bias.data()[n];
      }
      for (const char* mode : {"none", "persistent_gemv", "ternary_lut"}) {
        experiment.set(mode);
        Tensor output = bitnet_gemm_158bit_gpu(x, w, .3f, 1, K, N, bits, &m, &b);
        assert_close(output, reference, 2e-5f, mode, 2e-5f);
      }
    }
  }
  auto* first = context.reserve(gpu::WorkspaceSlot::BitnetScales, gpu::StorageType::Float32, 1);
  context.freeze(true);
  require(first == context.reserve(gpu::WorkspaceSlot::BitnetScales, gpu::StorageType::Float32, 1), "workspace pointer changed");
  bool rejected = false;
  try { context.reserve(gpu::WorkspaceSlot::BitnetScales, gpu::StorageType::Float32, 100000); }
  catch (const std::logic_error&) { rejected = true; }
  require(rejected, "capture workspace growth was accepted");
  context.freeze(false);
  rejected = false;
  try { context.reserve(gpu::WorkspaceSlot::BitnetScales, gpu::StorageType::Int8, 1); }
  catch (const std::logic_error&) { rejected = true; }
  require(rejected, "workspace dtype change was accepted");
}

void inactive_experts_contract() {
  constexpr int E = 3, D = 16, H = 32, R = 2;
  Tensor x = pattern({R, D}).to(Device::GPU);
  Tensor up = pattern({E, H, D}, .005f), down = pattern({E, D, H}, .004f);
  // Expert 1 is never selected. Poisoning its weights catches 0*NaN and
  // accidental all-expert arithmetic without relying on timing heuristics.
  std::fill(up.data() + H * D, up.data() + 2 * H * D, std::numeric_limits<float>::quiet_NaN());
  std::fill(down.data() + D * H, down.data() + 2 * D * H, std::numeric_limits<float>::quiet_NaN());
  Tensor ug = up.to(Device::GPU), dg = down.to(Device::GPU);
  Tensor routing({R, E}, Device::CPU);
  routing.data()[0] = .25f; routing.data()[2] = .75f;
  routing.data()[3] = 1.0f;
  Tensor rg = routing.to(Device::GPU);
  std::vector<GpuLinearView> views(2 * E);
  for (int e = 0; e < E; ++e) {
    views[e].weight = ug.raw_data() + e * H * D;
    views[e].inputs = D; views[e].outputs = H;
    views[E + e].weight = dg.raw_data() + e * D * H;
    views[E + e].inputs = H; views[E + e].outputs = D;
  }
  cuda_detail::DeviceBuffer<GpuLinearView> descriptors;
  require(descriptors.ensure(views.size()) != nullptr, "descriptor allocation failed");
  require(cudaMemcpy(descriptors.get(), views.data(), views.size() * sizeof(GpuLinearView), cudaMemcpyHostToDevice) == cudaSuccess, "descriptor copy failed");
  Tensor prepared = Tensor::uninitialized({R * E, H}, Device::GPU);
  Tensor scales = Tensor::uninitialized({R * E}, Device::GPU);
  Tensor hidden = Tensor::uninitialized({R * E, H}, Device::GPU);
  Tensor contributions = Tensor::uninitialized({R * E, D}, Device::GPU);
  Tensor output = Tensor::uninitialized({R, D}, Device::GPU);
  launch_moe_sparse_decode(descriptors.get(), rg.raw_data(), x.raw_data(), prepared.raw_data(),
      scales.raw_data(), hidden.raw_data(), contributions.raw_data(), output.raw_data(), E, D, H, R);
  Tensor input = x.cpu(), reference({R, D}, Device::CPU);
  for (int r = 0; r < R; ++r) for (int e = 0; e < E; ++e) {
    if (routing.data()[r * E + e] == 0) continue;
    std::vector<float> h(H);
    for (int n = 0; n < H; ++n) {
      float sum = 0;
      for (int k = 0; k < D; ++k) sum += input.data()[r * D + k] * up.data()[(e * H + n) * D + k];
      h[n] = std::max(sum, 0.f) * std::max(sum, 0.f);
    }
    for (int n = 0; n < D; ++n) {
      float sum = 0;
      for (int k = 0; k < H; ++k) sum += h[k] * down.data()[(e * D + n) * H + k];
      reference.data()[r * D + n] += sum * routing.data()[r * E + e];
    }
  }
  assert_close(output, reference, 2e-5f, "inactive experts and batched sparse decode", 1e-4f);
}

void attention_contract() {
  bool rejected = false;
  try {
    (void)GpuKvCache::from_float(Tensor(), Tensor(), INT_MAX, INT_MAX, INT_MAX, 0);
  } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "compact KV geometry overflow was not rejected");
  // More than the previous 12K shared-score limit, odd head width and GQA.
  constexpr int T = 16385, H = 4, KH = 2, D = 17, B = 2, W = KH * D;
  Tensor q = pattern({B, H * D}, .11f), kv = pattern({B, 2 * W}, .03f);
  Tensor keys = pattern({B, T, W}, .001f), values = pattern({B, T, W}, .003f);
  Tensor qg = q.to(Device::GPU), kvg = kv.to(Device::GPU);
  Tensor kg = keys.to(Device::GPU), vg = values.to(Device::GPU);
  Tensor output = Tensor::uninitialized({B, H * D}, Device::GPU);
  launch_gqa_append_decode_batch(qg.raw_data(), kvg.raw_data(), kg.raw_data(), vg.raw_data(),
      output.raw_data(), B, T, T - 1, H, KH, D, H / KH, 10000.f, T, 0);
  Tensor result = output.cpu(), actual_keys = kg.cpu(), actual_values = vg.cpu();
  Tensor reference({B, H * D}, Device::CPU);
  for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
    std::vector<double> rotated(D), scores(T);
    for (int d = 0; d < D; ++d) rotated[d] = q.data()[b * H * D + h * D + d];
    for (int d = 0; d < D / 2; ++d) {
      const double angle = double(T - 1) * std::pow(10000., -2. * d / D);
      const double x0 = rotated[d], x1 = rotated[D / 2 + d];
      rotated[d] = x0 * std::cos(angle) - x1 * std::sin(angle);
      rotated[D / 2 + d] = x0 * std::sin(angle) + x1 * std::cos(angle);
    }
    double maximum = -1e100;
    for (int t = 0; t < T; ++t) {
      double score = 0;
      for (int d = 0; d < D; ++d) score += rotated[d] * actual_keys.data()[(b * T + t) * W + (h / (H / KH)) * D + d];
      scores[t] = score / std::sqrt(double(D)); maximum = std::max(maximum, scores[t]);
    }
    double denominator = 0;
    for (double& s : scores) { s = std::exp(s - maximum); denominator += s; }
    for (int d = 0; d < D; ++d) {
      double sum = 0;
      for (int t = 0; t < T; ++t) sum += scores[t] * actual_values.data()[(b * T + t) * W + (h / (H / KH)) * D + d];
      reference.data()[b * H * D + h * D + d] = float(sum / denominator);
    }
  }
  assert_close(result, reference, 3e-4f, "tiled long GQA", 2e-3f);
  auto compact = GpuKvCache::from_float(kg, vg, B, T, W, T);
  require(compact->bytes() == size_t(B) * T * W * 4, "compact KV physical byte count");
  auto first_row = compact->row(0, T);
  auto resized = first_row->resize(T + 64, T);
  auto expanded = resized->materialize(T);
  assert_close(expanded.first.storage_view(0, {T, W}), actual_keys.storage_view(0, {T, W}), 3e-4f, "compact KV copy and growth");
  launch_gqa_append_decode_batch(qg.raw_data(), kvg.raw_data(), compact->keys(), compact->values(),
      output.raw_data(), B, T, T - 1, H, KH, D, H / KH, 10000.f, T, 1);
  assert_close(output, result, 3e-4f, "compact attention", 2e-3f);
}

void fused_mamba_contract() {
  Tensor y = pattern({3, 257}).to(Device::GPU), z = pattern({3, 257}, .03f).to(Device::GPU);
  Tensor w = pattern({257}, .17f).to(Device::GPU);
  Tensor reference = y.mul(z.silu()).rmsnorm(1e-6f).mul(w);
  Tensor output = Tensor::uninitialized({3, 257}, Device::GPU);
  cuda::launch_mamba_gated_rmsnorm(y.raw_data(), z.raw_data(), w.raw_data(), output.raw_data(), 3, 257, 1e-6f);
  assert_close(output, reference, 2e-5f, "fused Mamba epilogue", 2e-5f);
}

void graph_and_session_contract() {
  Environment graph("NSOS_GPU_GRAPH_DECODE", "1"), sync("NSOS_CUDA_SYNC", "0");
  Environment dtype("NSOS_GPU_KV_DTYPE", "fp32"), experiment("NSOS_GPU_EXPERIMENT", "none");
  ModelConfig config;
  config.num_layers = 2; config.d_model = 32; config.vocab_size = 64;
  config.n_heads = 4; config.n_kv_heads = 2;
  config.attention_period = 2; config.attention_slot = 1;
  config.mamba_d_state = 8; config.mamba_head_dim = 8; config.mamba_n_groups = 1;
  config.use_moe = false; config.use_ttt = false; config.use_chrass = false; config.use_kan = false;
  config.dropout = 0; config.max_context_tokens = 64; config.sliding_window = 64;
  JambaModel model(config, Device::GPU);
  model.to(Device::GPU); model.set_training_mode(false); model.set_streaming_inference(true);
  model.reserve_kv_cache(64, Device::GPU);
  (void)model.forward_ids({1, 3, 5});
  auto initial = model.fork_session(true);
  std::vector<Tensor> reference;
  for (int token : {7, 9, 11, 13}) reference.push_back(model.forward_ids({token}).cpu());
  model.restore_session(initial);
  model.reserve_kv_cache(64, Device::GPU);
  int step = 0;
  for (int token : {7, 9, 11, 13}) {
    Tensor logits = model.forward_ids_decode_graph(token);
    require(logits.size > 0, model.decode_graph_status().c_str());
    assert_close(logits, reference[step++], 4e-3f, "graph hybrid state/position parity");
  }
  require(model.decode_graph_active(), model.decode_graph_status().c_str());
  auto graph_snapshot = model.fork_session(true);
  std::vector<Tensor> frozen_keys;
  for (const auto& block : graph_snapshot.blocks)
    frozen_keys.push_back(block.attention.key_cache.size ? block.attention.key_cache.cpu() : Tensor());
  Tensor graph_next = model.forward_ids_decode_graph(15).cpu();
  for (size_t i = 0; i < frozen_keys.size(); ++i)
    if (frozen_keys[i].size)
      assert_close(graph_snapshot.blocks[i].attention.key_cache, frozen_keys[i], 0.f, "graph snapshot storage isolation");
  model.reserve_kv_cache(128, Device::GPU);
  require(!model.decode_graph_active(), "KV reservation retained stale graph pointers");
  model.restore_session(graph_snapshot);
  assert_close(model.forward_ids({15}), graph_next, 4e-3f, "graph snapshot continuation");
  model.restore_session(initial);
  dtype.set("fp16");
  (void)model.forward_ids({7});
  auto snapshot = model.fork_session(true);
  bool compact_found = false;
  for (const auto& block : snapshot.blocks) compact_found |= bool(block.attention.compact);
  require(compact_found, "compact KV session snapshot was not retained");
  Tensor first = model.forward_ids({9}).cpu();
  model.restore_session(snapshot);
  Tensor repeated = model.forward_ids({9}).cpu();
  assert_close(first, repeated, 1e-5f, "compact KV/SSM session copy-on-write");
  model.restore_session_batch({snapshot, snapshot});
  Tensor batched = model.forward_ids_batch({{9}, {9}}).reshape({2, config.vocab_size});
  assert_close(batched.slice(0, 0, 1), repeated, 5e-3f, "compact session batching");
  assert_close(batched.slice(0, 0, 1), batched.slice(0, 1, 2), 1e-5f, "independent identical sessions");
}
}

int main() {
  return run_parity("decode_runtime", [] {
    Environment no_forced_sync("NSOS_CUDA_SYNC", "0");
    set_matmul_precision_mode(0);
    gemv_contract();
    inactive_experts_contract();
    attention_contract();
    fused_mamba_contract();
    graph_and_session_contract();
    cuda_sync_or_throw("decode_runtime/end");
  });
}
