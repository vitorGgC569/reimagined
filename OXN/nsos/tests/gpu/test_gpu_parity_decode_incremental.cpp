// Real incremental-decode parity for a hybrid Mamba + attention model.
//
// The former test at this slot returned success without exercising behavior.
// This gate performs a prompt prefill followed by one-token streaming steps on
// CUDA.  Every step is checked against both CPU streaming and a fresh
// full-prefix CPU forward, so stale SSM state, KV-cache corruption and
// token-position drift all fail the test.

#include "gpu_parity_common.h"
#include "jamba.h"

#include <filesystem>
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

Tensor last_token_logits(const Tensor& logits) {
  if (logits.shape.dims.size() != 2 || logits.shape.dims[0] <= 0) {
    throw std::runtime_error(
        "incremental decode expected rank-2 non-empty logits");
  }
  return logits.slice(0, logits.shape.dims[0] - 1, logits.shape.dims[0]);
}

} // namespace

int main() {
  return run_parity("decode_incremental", [] {
    ModelConfig config;
    config.num_layers = 2;
    config.d_model = 32;
    config.vocab_size = 64;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.attention_period = 2;
    config.attention_slot = 1;
    config.mamba_d_state = 8;
    config.mamba_head_dim = 8;
    config.mamba_n_groups = 1;
    config.max_context_tokens = 64;
    config.sliding_window = 64;
    config.use_moe = false;
    config.use_ttt = false;
    config.use_chrass = false;
    config.use_kan = false;
    config.use_exact_attention_training = false;
    config.dropout = 0.0f;

    JambaModel cpu_reference(config, Device::CPU);
    JambaModel cpu_stream(config, Device::CPU);
    JambaModel gpu_stream(config, Device::GPU);
    gpu_stream.to(Device::GPU);

    const auto model_path = std::filesystem::temp_directory_path() /
                            "nsos_gpu_decode_incremental.bin";
    cpu_reference.save(model_path.string());
    cpu_stream.load(model_path.string());
    gpu_stream.load(model_path.string());

    cpu_reference.set_training_mode(false);
    cpu_stream.set_training_mode(false);
    gpu_stream.set_training_mode(false);
    cpu_stream.set_streaming_inference(true);
    gpu_stream.set_streaming_inference(true);

    const std::vector<int> prompt = {1, 5, 7, 3};
    const std::vector<int> decode_tokens = {9, 11, 2, 13};
    gpu_stream.reserve_kv_cache(
        static_cast<int>(prompt.size() + decode_tokens.size()),
        Device::GPU, 1);

    Tensor cpu_prefill = cpu_stream.forward_ids(prompt);
    Tensor gpu_prefill = gpu_stream.forward_ids(prompt);
    cuda_sync_or_throw("decode_incremental/prefill");
    assert_close(last_token_logits(cpu_prefill),
                 last_token_logits(gpu_prefill).cpu(), 4e-3f,
                 "decode_incremental_prefill_cpu_gpu");

    std::vector<int> full_prefix = prompt;
    for (size_t step = 0; step < decode_tokens.size(); ++step) {
      const int token = decode_tokens[step];
      full_prefix.push_back(token);

      cpu_reference.reset_session();
      cpu_reference.set_streaming_inference(false);
      Tensor full_logits = cpu_reference.forward_ids(full_prefix);
      Tensor cpu_step = cpu_stream.forward_ids({token});
      Tensor gpu_step = gpu_stream.forward_ids({token});
      cuda_sync_or_throw("decode_incremental/step");

      const Tensor expected = last_token_logits(full_logits);
      const Tensor cpu_actual = last_token_logits(cpu_step);
      const Tensor gpu_actual = last_token_logits(gpu_step).cpu();
      assert_close(cpu_actual, expected, 5e-4f,
                   "decode_incremental_cpu_stream_vs_full");
      assert_close(gpu_actual, cpu_actual, 4e-3f,
                   "decode_incremental_gpu_vs_cpu_stream");
      assert_close(gpu_actual, expected, 4e-3f,
                   "decode_incremental_gpu_vs_full");
      std::cout << "[GPUParity:decode_incremental] step=" << step
                << " token=" << token << " compared=cpu_stream+cpu_full"
                << std::endl;
    }

    // Merge GPU-resident prefill snapshots once, then advance independent rows
    // for several steps without round-tripping the Mamba carry through the CPU.
    std::vector<std::vector<int>> prefixes{{1, 5, 7, 3}, {2, 4, 6, 8}};
    std::vector<nsos::JambaSessionSnapshot> snapshots;
    for (const auto& row : prefixes) {
      gpu_stream.reset_session();
      gpu_stream.set_streaming_inference(true);
      Tensor last = gpu_stream.forward_ids_last(row);
      if (last.shape.dims != std::vector<int>({1, config.vocab_size}))
        throw std::runtime_error("GPU last-token projection retained prompt rows");
      snapshots.push_back(gpu_stream.fork_session(true));
    }
    gpu_stream.restore_session_batch(snapshots);
    gpu_stream.reserve_kv_cache(12, Device::GPU, 2);
    for (int step = 0; step < 3; ++step) {
      std::vector<std::vector<int>> next{{9 + step}, {13 + step}};
      Tensor actual = gpu_stream.forward_ids_batch(next).reshape({2, config.vocab_size}).cpu();
      for (int row = 0; row < 2; ++row) {
        prefixes[row].push_back(next[row][0]);
        cpu_reference.reset_session();
        Tensor expected = cpu_reference.forward_ids_last(prefixes[row]);
        assert_close(actual.slice(0, row, row + 1), expected, 5e-3f,
                     "device-resident batched decode vs full CPU prefix");
      }
    }
    const auto saved_batch = gpu_stream.fork_session_batch(true);
    Tensor before_restore = gpu_stream.forward_ids_batch({{3}, {4}});
    gpu_stream.restore_session_batch(saved_batch);
    Tensor after_restore = gpu_stream.forward_ids_batch({{3}, {4}});
    assert_close(before_restore, after_restore, 1e-5f, "GPU batch snapshot repeat");

    std::error_code ec;
    std::filesystem::remove(model_path, ec);
  });
}
