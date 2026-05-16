// Jamba forward_ids_batch CPU↔GPU parity.
//
// Exercises the full model stack — embedding → mamba blocks → attention
// → output projection — on a tiny configuration.  The test saves the
// CPU model to a temp file and reloads it on the GPU side so both
// devices use byte-identical weights; otherwise random initialization
// would dominate any kernel-level mismatch.
//
// MoE and TTT are explicitly disabled because they are exercised by
// dedicated tests and would couple this signal to unrelated kernels.

#include "gpu_parity_common.h"
#include "jamba.h"
#include "tensor.h"

#include <filesystem>
#include <system_error>
#include <vector>

using nsos::JambaModel;
using nsos::ModelConfig;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

int main() {
  return run_parity("jamba_batch", [] {
    ModelConfig config;
    config.num_layers = 4;
    config.d_model = 64;
    config.vocab_size = 96;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.attention_period = 64;
    config.attention_slot = 63;
    config.use_moe = false;
    config.use_ttt = false;
    config.use_exact_attention_training = false;

    JambaModel cpu_model(config, Device::CPU);
    JambaModel gpu_model(config, Device::GPU);
    gpu_model.to(Device::GPU);

    const auto temp_path =
        std::filesystem::temp_directory_path() / "nsos_gpu_parity_jamba.bin";
    cpu_model.save(temp_path.string());
    gpu_model.load(temp_path.string());

    const std::vector<std::vector<int>> batch_ids = {{1, 2, 3, 4}, {5, 6, 7}};
    Tensor cpu_logits = cpu_model.forward_ids_batch(batch_ids, nullptr).cpu();
    Tensor gpu_logits = gpu_model.forward_ids_batch(batch_ids, nullptr).cpu();
    cuda_sync_or_throw("jamba_batch/forward");
    assert_close(cpu_logits, gpu_logits, 3e-3f, "jamba_forward_ids_batch");

    // Best-effort cleanup; failure to remove a temp file is not a parity
    // signal so we swallow ec.
    std::error_code ec;
    std::filesystem::remove(temp_path, ec);
  });
}
