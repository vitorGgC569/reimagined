// Jamba forward_ids_batch CPU↔GPU parity WITH SSA (sparse attention) enabled.
//
// Coverage gate for WS-6b: wiring the SSA GPU kernel into the attention
// fast-path.  Before that wiring the GPU attention fast-path runs DENSE
// attention and returns before ever reaching the sparse branch, so a GPU model
// silently ignores SSA -- this test FAILS (GPU dense != CPU sparse).  After
// wiring it PASSES (GPU sparse == CPU sparse).
//
// attention_period=2 (<= num_layers) with slot=1 puts real attention layers in
// the stack; top_k_blocks=1 over several blocks makes the sparse result
// materially different from dense -- the test asserts that difference (so it
// genuinely exercises SSA) before checking CPU/GPU parity.

#include "gpu_parity_common.h"
#include "jamba.h"
#include "tensor.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using nsos::Device;
using nsos::JambaModel;
using nsos::ModelConfig;
using nsos::Tensor;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

float max_abs_diff(const Tensor& a, const Tensor& b) {
  Tensor ac = a.cpu();
  Tensor bc = b.cpu();
  if (ac.size != bc.size) throw std::runtime_error("max_abs_diff: size mismatch");
  float m = 0.0f;
  for (int i = 0; i < ac.size; ++i)
    m = std::max(m, std::abs(ac.data()[i] - bc.data()[i]));
  return m;
}

ModelConfig make_config() {
  ModelConfig config;
  config.num_layers = 4;
  config.d_model = 64;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 2;  // <= num_layers so attention layers exist
  config.attention_slot = 1;    // one-based layers 2 and 4 are attention
  config.use_moe = false;
  config.use_ttt = false;
  config.use_exact_attention_training = false;
  return config;
}

}  // namespace

int main() {
  return run_parity("jamba_sparse", [] {
    const ModelConfig config = make_config();

    JambaModel cpu_dense_model(config, Device::CPU);
    const auto temp_path =
        std::filesystem::temp_directory_path() / "nsos_gpu_parity_jamba_sparse.bin";
    cpu_dense_model.save(temp_path.string());

    JambaModel cpu_model(config, Device::CPU);
    cpu_model.load(temp_path.string());
    JambaModel gpu_model(config, Device::GPU);
    gpu_model.to(Device::GPU);
    gpu_model.load(temp_path.string());

    std::vector<int> a, b;
    for (int i = 0; i < 18; ++i) a.push_back((i * 7 + 3) % config.vocab_size);
    for (int i = 0; i < 13; ++i) b.push_back((i * 5 + 11) % config.vocab_size);
    const std::vector<std::vector<int>> batch_ids = {a, b};

    // Dense baseline (SSA off) on CPU.
    const Tensor cpu_dense = cpu_dense_model.forward_ids_batch(batch_ids, nullptr).cpu();

    // SSA on. With block_size=2 the longer row has nine causal blocks and
    // sink + local + top-1 retain at most three, making this a stable wiring
    // discriminator instead of the near-dense 3-of-5 selection produced by
    // block_size=4.
    cpu_model.set_sparse_attention(true, /*block_size=*/2, /*top_k_blocks=*/1,
                                   /*local_blocks=*/1, /*sink_blocks=*/1);
    gpu_model.set_sparse_attention(true, 2, 1, 1, 1);

    const Tensor cpu_sparse = cpu_model.forward_ids_batch(batch_ids, nullptr).cpu();
    const Tensor gpu_sparse = gpu_model.forward_ids_batch(batch_ids, nullptr).cpu();
    cuda_sync_or_throw("jamba_sparse/forward");

    // Sanity: SSA must actually change the output, else this test proves
    // nothing about SSA being applied on the GPU.
    const float sparse_effect = max_abs_diff(cpu_dense, cpu_sparse);
    std::cout << "    sparse-vs-dense effect (CPU)=" << sparse_effect;
    if (!(sparse_effect > 1e-2f)) {
      throw std::runtime_error(
          "SSA had no measurable effect on CPU (effect=" +
          std::to_string(sparse_effect) + "); test config is not discriminating");
    }

    // The real gate: GPU-with-SSA must equal CPU-with-SSA.
    assert_close(cpu_sparse, gpu_sparse, 3e-3f, "jamba_forward_ids_batch_sparse");

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);
  });
}
