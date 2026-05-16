// MoE batched forward CPU↔GPU parity (Phase 4-extended).
//
// Validates that JambaBlock::forward_moe_gpu_batched produces output
// equivalent to the CPU forward_moe path within numerical tolerance.
// We exercise this end-to-end through JambaModel::forward_ids_batch
// because the MoE block is private to JambaBlock.
//
// Both models are constructed with identical configuration and then
// weight-synchronized via save/load so the only delta between outputs
// is the kernel implementation (CPU per-expert std::memcpy loops vs
// the device-resident count→scan→gather→scatter pipeline).

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
  return run_parity("moe_batched", [] {
    // Small but non-trivial config: 2 layers (with at least one MoE
    // block), d_model=64, num_experts=8, top_k=2.  Sequence lengths
    // chosen to exercise both single-token and multi-token batching
    // through the same forward pass.
    ModelConfig config;
    config.num_layers = 2;
    config.d_model = 64;
    config.vocab_size = 96;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.attention_period = 64;
    config.attention_slot = 63;
    config.use_moe = true;            // engage MoE blocks
    config.num_experts = 8;
    config.num_experts_per_token = 2;
    config.use_ttt = false;
    config.use_exact_attention_training = false;

    JambaModel cpu_model(config, Device::CPU);
    JambaModel gpu_model(config, Device::GPU);
    gpu_model.to(Device::GPU);

    // Sync weights via the production save/load pack so both models
    // represent the same function exactly.
    const auto temp_path = std::filesystem::temp_directory_path() /
                           "nsos_gpu_parity_moe_batched_model.bin";
    cpu_model.save(temp_path.string());
    gpu_model.load(temp_path.string());

    // Two input batches with different sequence lengths to make sure
    // padding-aware code paths agree across devices.
    const std::vector<std::vector<int>> batch_ids = {
        {1, 2, 3, 4, 5},
        {6, 7, 8},
    };
    Tensor cpu_logits =
        cpu_model.forward_ids_batch(batch_ids, /*ctx=*/nullptr).cpu();
    Tensor gpu_logits =
        gpu_model.forward_ids_batch(batch_ids, /*ctx=*/nullptr).cpu();
    cuda_sync_or_throw("moe_batched/forward");

    // Tolerance: MoE introduces top-k softmax + rescaling per token,
    // and the scatter-add kernel uses atomicAdd which is order-
    // dependent within a row when top_k > 1.  3e-3 is comfortably
    // wider than the per-element rounding budget (~1e-5) but tight
    // enough to catch wrong-expert routing or scale mismatches.
    assert_close(cpu_logits, gpu_logits, 3e-3f,
                 "jamba_forward_ids_batch_with_moe");

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);
  });
}
