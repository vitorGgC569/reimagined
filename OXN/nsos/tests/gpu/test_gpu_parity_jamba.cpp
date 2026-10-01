// Jamba forward_ids_batch CPU↔GPU parity.
//
// Exercises the full model stack — embedding → mamba blocks → attention
// → output projection — on a tiny configuration.  The test saves the
// CPU model to a temp file and reloads it on the GPU side so both
// devices use byte-identical weights; otherwise random initialization
// would dominate any kernel-level mismatch.
//
// The hybrid fixture disables MoE and TTT. A separate fixture below exercises
// deterministic MoE training under strict GPU memory/device contracts.

#include "gpu_parity_common.h"
#include "jamba.h"
#include "nsos/determinism.h"
#include "tensor.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using nsos::JambaModel;
using nsos::ModelConfig;
using nsos::HybridComposition;
using nsos::Tensor;
using nsos::Device;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {
void check_deterministic_moe_training() {
    nsos::determinism::set_deterministic_reductions(true);
    nsos::set_strict_gpu_execution(true);
    ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.d_model = 64;
    cfg.vocab_size = 96;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.attention_period = 99;
    cfg.attention_slot = 98;
    cfg.use_ttt = false;
    cfg.use_kan = false;
    cfg.use_chrass = false;
    cfg.use_moe = true;
    cfg.moe_period = 2;
    cfg.moe_slot = 0;
    cfg.num_experts = 4;
    cfg.num_experts_per_token = 2;
    cfg.moe_expert_hidden_dim = 64;
    cfg.mamba2_faithful = true;
    cfg.mamba_d_state = 8;
    cfg.mamba_head_dim = 32;
    cfg.mamba_n_groups = 1;
    cfg.dropout = 0.0f;
    JambaModel cpu(cfg, Device::CPU);
    JambaModel gpu(cfg, Device::GPU);
    gpu.to(Device::GPU);
    cpu.set_reference_path(true);
    gpu.set_reference_path(true);
    cpu.set_training_mode(true);
    gpu.set_training_mode(true);
    const auto cp = cpu.parameters();
    const auto gp = gpu.parameters();
    if (cp.size() != gp.size() || !gpu.layers[0]->uses_moe()) {
        throw std::runtime_error("deterministic MoE fixture is not active");
    }
    for (size_t i = 0; i < cp.size(); ++i) {
        if (cp[i]->name != gp[i]->name) {
            throw std::runtime_error("deterministic MoE registry mismatch");
        }
        gp[i]->copy_data_from(cp[i]->data.to(Device::GPU));
    }
    const std::vector<std::vector<int>> ids = {{1, 3, 2, 3}, {5, 7, 5}};
    nsos::Context cc, gc;
    Tensor expected = cpu.forward_ids_batch(ids, &cc);
    Tensor actual = gpu.forward_ids_batch(ids, &gc);
    assert_close(expected, actual, 3e-3f, "deterministic_moe_forward", 3e-2f);
    Tensor grad = Tensor::ones(expected.shape.dims, Device::CPU).mul(1e-3f);
    cpu.backward(grad, cc);
    gpu.backward(grad.to(Device::GPU), gc);
    cuda_sync_or_throw("deterministic_moe_backward");
    bool expert_gradient = false;
    bool router_gradient = false;
    for (size_t i = 0; i < cp.size(); ++i) {
        if (cp[i]->grad.size == 0 && gp[i]->grad.size == 0) continue;
        if (cp[i]->grad.size == 0 || gp[i]->grad.size == 0) {
            throw std::runtime_error("deterministic MoE missing gradient: " + cp[i]->name);
        }
        assert_close(cp[i]->grad, gp[i]->grad, 3e-3f, cp[i]->name.c_str(), 3e-2f);
        if (cp[i]->grad.norm() > 0.0f) {
            expert_gradient |= cp[i]->name.find("experts.") != std::string::npos;
            router_gradient |= cp[i]->name.find("router.") != std::string::npos;
        }
    }
    if (!expert_gradient || !router_gradient) {
        throw std::runtime_error("deterministic MoE backward did not reach experts and router");
    }
}
}  // namespace

int main() {
  return run_parity("jamba_hybrid_batch", [] {
    nsos::determinism::set_deterministic_reductions(false);
    ModelConfig config;
    config.architecture_schema_version = 2;
    config.num_layers = 2;
    config.d_model = 64;
    config.vocab_size = 96;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.attention_period = 2;
    config.attention_slot = 1;
    config.hybrid_composition = HybridComposition::ParallelGated;
    config.force_mamba_last_layer = false;
    config.faithful_attention_linears = true;
    config.hybrid_mamba_gate_init = 1.0f;
    config.hybrid_attention_gate_init = 0.01f;
    config.hybrid_ffn_gate_init = 0.01f;
    config.mamba2_faithful = true;
    config.mamba_state_expansion = true;
    config.mamba_d_state = 8;
    config.mamba_head_dim = 32;
    config.mamba_n_groups = 1;
    config.use_moe = false;
    config.use_ttt = false;
    config.use_kan = false;
    config.use_chrass = false;
    config.use_exact_attention_training = true;
    config.use_gradient_checkpointing = true;
    config.dropout = 0.0f;

    JambaModel cpu_model(config, Device::CPU);
    JambaModel gpu_model(config, Device::GPU);
    gpu_model.to(Device::GPU);
    if (!cpu_model.layers[1]->attn_layer ||
        !cpu_model.layers[1]->mamba_layer ||
        cpu_model.layers[1]->audit_block_type() !=
            "mamba2+attention+ffn") {
      throw std::runtime_error(
          "GPU parity fixture does not contain the product hybrid block");
    }

    const auto temp_path =
        std::filesystem::temp_directory_path() / "nsos_gpu_parity_jamba.bin";
    cpu_model.save(temp_path.string());
    gpu_model.load(temp_path.string());

    const std::vector<std::vector<int>> batch_ids = {{1, 2, 3, 4}, {5, 6, 7}};
    nsos::Context cpu_context;
    nsos::Context gpu_context;
    Tensor cpu_logits =
        cpu_model.forward_ids_batch(batch_ids, &cpu_context);
    Tensor gpu_logits =
        gpu_model.forward_ids_batch(batch_ids, &gpu_context);
    cuda_sync_or_throw("jamba_hybrid_batch/forward");
    assert_close(cpu_logits, gpu_logits, 3e-3f, "jamba_forward_ids_batch");

    Tensor cpu_gradient =
        Tensor::ones(cpu_logits.shape.dims, Device::CPU).mul(1e-3f);
    cpu_model.backward(cpu_gradient, cpu_context);
    gpu_model.backward(cpu_gradient.to(Device::GPU), gpu_context);
    cuda_sync_or_throw("jamba_hybrid_batch/backward");

    const auto cpu_parameters = cpu_model.parameters();
    const auto gpu_parameters = gpu_model.parameters();
    if (cpu_parameters.size() != gpu_parameters.size()) {
      throw std::runtime_error(
          "hybrid model parameter registry differs across devices");
    }
    bool saw_mamba_gradient = false;
    bool saw_attention_gradient = false;
    for (size_t index = 0; index < cpu_parameters.size(); ++index) {
      const nsos::Parameter* cpu_parameter = cpu_parameters[index];
      const nsos::Parameter* gpu_parameter = gpu_parameters[index];
      if (!cpu_parameter || !gpu_parameter ||
          cpu_parameter->name != gpu_parameter->name) {
        throw std::runtime_error(
            "hybrid parameter identity mismatch at index " +
            std::to_string(index));
      }
      if (cpu_parameter->grad.size == 0 &&
          gpu_parameter->grad.size == 0) {
        continue;
      }
      if (cpu_parameter->grad.size == 0 ||
          gpu_parameter->grad.size == 0) {
        throw std::runtime_error(
            "hybrid gradient presence mismatch for " +
            cpu_parameter->name);
      }
      assert_close(cpu_parameter->grad, gpu_parameter->grad,
                   2e-3f, cpu_parameter->name.c_str(), 3e-2f);
      if (cpu_parameter->name.find("layers.1.mamba.") == 0 &&
          cpu_parameter->grad.norm() > 0.0f) {
        saw_mamba_gradient = true;
      }
      if (cpu_parameter->name.find("layers.1.attn.") == 0 &&
          cpu_parameter->grad.norm() > 0.0f) {
        saw_attention_gradient = true;
      }
    }
    if (!saw_mamba_gradient || !saw_attention_gradient) {
      throw std::runtime_error(
          "hybrid backward did not reach both Mamba and Attention branches");
    }

    std::vector<Tensor> gradients_before_duplicate;
    gradients_before_duplicate.reserve(gpu_parameters.size());
    for (const nsos::Parameter* parameter : gpu_parameters) {
      gradients_before_duplicate.push_back(
          parameter && parameter->grad.size > 0
              ? parameter->grad.cpu()
              : Tensor());
    }

    const auto telemetry = gpu_model.runtime_telemetry();
    if (telemetry.faithful_recompute_forwards != 2 ||
        telemetry.faithful_selective_history_recomputes != 2 ||
        telemetry.faithful_full_block_recompute_forwards != 0 ||
        telemetry.faithful_warp_aggregated_backward_calls != 2 ||
        telemetry.faithful_scalar_atomic_backward_calls != 0 ||
        telemetry.faithful_reduced_conv_backward_calls != 2 ||
        telemetry.faithful_generic_atomic_conv_backward_calls != 0 ||
        telemetry.faithful_forward_gpu_calls != 4 ||
        telemetry.faithful_grouped_projection_forward_calls != 2) {
      throw std::runtime_error(
          "selective SSD checkpoint recomputed projections or missed a "
          "history rebuild");
    }
    bool rejected_consumed_backward = false;
    try {
      gpu_model.backward(
          cpu_gradient.to(Device::GPU), gpu_context);
    } catch (const std::runtime_error&) {
      rejected_consumed_backward = true;
    }
    if (!rejected_consumed_backward) {
      throw std::runtime_error(
          "faithful forward state accepted a duplicate backward");
    }
    for (size_t index = 0; index < gpu_parameters.size(); ++index) {
      const nsos::Parameter* parameter = gpu_parameters[index];
      if (!parameter || parameter->grad.size == 0) {
        if (gradients_before_duplicate[index].size != 0) {
          throw std::runtime_error(
              "duplicate backward changed gradient presence");
        }
        continue;
      }
      assert_close(parameter->grad, gradients_before_duplicate[index],
                   0.0f, parameter->name.c_str());
    }

    // Best-effort cleanup; failure to remove a temp file is not a parity
    // signal so we swallow ec.
    std::error_code ec;
    std::filesystem::remove(temp_path, ec);
    check_deterministic_moe_training();
  });
}
