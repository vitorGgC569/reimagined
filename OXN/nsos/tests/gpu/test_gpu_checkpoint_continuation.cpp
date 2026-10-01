// Current training-state checkpoint continuation on CUDA or HIP.
//
// The model, gradients, Adam moments and optimizer update all remain on GPU.
// After one step we save a host-owned model + v10 sidecar snapshot, restore it
// into a new GPU model,
// and require the next uninterrupted/resumed updates to agree.

#include "gpu_parity_common.h"
#include "training_runtime_policy.h"
#include "gpu_execution.h"
#include "jamba.h"
#include "nsos/determinism.h"
#include "optimizer_runtime_policy.h"
#include "trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using nsos::Device;
using nsos::HybridComposition;
using nsos::JambaModel;
using nsos::ModelConfig;
using nsos::Parameter;
using nsos::Tensor;
using nsos::TensorRandomState;
using nsos::Trainer;
using nsos::TrainPhaseScheduler;
using nsos::capture_tensor_random_state;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void set_full_ttt_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_TTT_FULL_BPTT", enabled ? "1" : "0") == 0,
          "cannot set checkpoint TTT policy");
#else
  require(setenv("NSOS_TTT_FULL_BPTT", enabled ? "1" : "0", 1) == 0,
          "cannot set checkpoint TTT policy");
#endif
}
void set_kan_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_KAN_RECOMPUTE_TRAINING",enabled?"1":"0")==0,"cannot set checkpoint KAN policy");
#else
  require(setenv("NSOS_KAN_RECOMPUTE_TRAINING",enabled?"1":"0",1)==0,"cannot set checkpoint KAN policy");
#endif
}
void set_kan_wmma_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_KAN_WMMA_TRAINING",enabled?"1":"0")==0,"cannot set checkpoint KAN WMMA policy");
#else
  require(setenv("NSOS_KAN_WMMA_TRAINING",enabled?"1":"0",1)==0,"cannot set checkpoint KAN WMMA policy");
#endif
}
void set_grouped_moe_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_MOE_GROUPED_TRAINING", enabled ? "1" : "0") == 0,
          "cannot set checkpoint MoE policy");
#else
  require(setenv("NSOS_MOE_GROUPED_TRAINING", enabled ? "1" : "0", 1) == 0,
          "cannot set checkpoint MoE policy");
#endif
}
void set_wmma_moe_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_MOE_WMMA_TRAINING", enabled ? "1" : "0") == 0, "cannot set WMMA policy");
#else
  require(setenv("NSOS_MOE_WMMA_TRAINING", enabled ? "1" : "0", 1) == 0, "cannot set WMMA policy");
#endif
}

float max_parameter_difference(JambaModel& lhs, JambaModel& rhs) {
  const auto left = lhs.parameters();
  const auto right = rhs.parameters();
  require(left.size() == right.size(), "checkpoint parameter count mismatch");
  float maximum = 0.0f;
  for (size_t parameter_index = 0; parameter_index < left.size();
       ++parameter_index) {
    const Parameter* a_parameter = left[parameter_index];
    const Parameter* b_parameter = right[parameter_index];
    require(a_parameter && b_parameter, "checkpoint contains null parameter");
    require(a_parameter->data.shape == b_parameter->data.shape,
            "checkpoint parameter shape mismatch");
    const Tensor a = a_parameter->data.cpu();
    const Tensor b = b_parameter->data.cpu();
    for (int element = 0; element < a.size; ++element) {
      maximum = std::max(
          maximum, std::abs(a.data()[element] - b.data()[element]));
    }
  }
  return maximum;
}

std::vector<Tensor> snapshot_parameters(JambaModel& model) {
  std::vector<Tensor> snapshot;
  for (const Parameter* parameter : model.parameters()) {
    require(parameter != nullptr, "snapshot contains null parameter");
    snapshot.push_back(parameter->data.cpu().clone());
  }
  return snapshot;
}

float max_parameter_difference(
    JambaModel& model, const std::vector<Tensor>& snapshot) {
  const auto parameters = model.parameters();
  require(parameters.size() == snapshot.size(),
          "snapshot parameter count mismatch");
  float maximum = 0.0f;
  for (size_t parameter_index = 0; parameter_index < parameters.size();
       ++parameter_index) {
    const Parameter* parameter = parameters[parameter_index];
    require(parameter != nullptr, "snapshot comparison has null parameter");
    const Tensor current = parameter->data.cpu();
    require(current.shape == snapshot[parameter_index].shape,
            "snapshot parameter shape mismatch");
    for (int element = 0; element < current.size; ++element) {
      maximum = std::max(
          maximum,
          std::abs(current.data()[element] -
                   snapshot[parameter_index].data()[element]));
    }
  }
  return maximum;
}

using OptimizerStateSnapshot =
    std::unordered_map<Parameter*, Tensor>;

OptimizerStateSnapshot snapshot_optimizer_state(
    const std::unordered_map<Parameter*, Tensor>& state) {
  OptimizerStateSnapshot snapshot;
  for (const auto& [parameter, tensor] : state) {
    require(parameter != nullptr,
            "optimizer-state snapshot contains a null parameter");
    snapshot.emplace(parameter, tensor.cpu().clone());
  }
  return snapshot;
}

void require_optimizer_state_equal(
    const std::unordered_map<Parameter*, Tensor>& state,
    const OptimizerStateSnapshot& snapshot,
    const char* message) {
  require(state.size() == snapshot.size(), message);
  for (const auto& [parameter, expected] : snapshot) {
    const auto current_entry = state.find(parameter);
    require(current_entry != state.end(), message);
    const Tensor current = current_entry->second.cpu();
    require(current.shape == expected.shape, message);
    require(
        std::memcmp(
            current.data(), expected.data(),
            static_cast<size_t>(current.size) * sizeof(float)) == 0,
        message);
  }
}

void require_cuda_parameters(JambaModel& model) {
  const auto parameters = model.parameters();
  require(!parameters.empty(), "GPU checkpoint fixture has no parameters");
  for (const Parameter* parameter : parameters) {
    if (parameter && parameter->data.size > 0) {
      require(parameter->data.get_device() == Device::GPU,
              "checkpoint fixture contains a host-resident model parameter");
    }
  }
}

void require_optimizer_states_equal(
    Trainer& lhs, JambaModel& lhs_model,
    Trainer& rhs, JambaModel& rhs_model) {
  const auto lhs_parameters = lhs_model.parameters();
  const auto rhs_parameters = rhs_model.parameters();
  require(lhs_parameters.size() == rhs_parameters.size(),
          "deterministic optimizer parameter registry differs");
  const auto compare_state = [&](const auto& lhs_state,
                                 const auto& rhs_state,
                                 const char* message) {
    require(lhs_state.size() == rhs_state.size(), message);
    for (size_t index = 0; index < lhs_parameters.size(); ++index) {
      Parameter* lhs_parameter = lhs_parameters[index];
      Parameter* rhs_parameter = rhs_parameters[index];
      const auto lhs_entry = lhs_state.find(lhs_parameter);
      const auto rhs_entry = rhs_state.find(rhs_parameter);
      if (lhs_entry == lhs_state.end() && rhs_entry == rhs_state.end()) {
        continue;
      }
      require(lhs_entry != lhs_state.end() &&
                  rhs_entry != rhs_state.end(),
              message);
      const Tensor lhs_tensor = lhs_entry->second.cpu();
      const Tensor rhs_tensor = rhs_entry->second.cpu();
      require(lhs_tensor.shape == rhs_tensor.shape, message);
      require(
          std::memcmp(
              lhs_tensor.data(), rhs_tensor.data(),
              static_cast<size_t>(lhs_tensor.size) * sizeof(float)) == 0,
          message);
    }
  };
  compare_state(lhs.m_state, rhs.m_state,
                "deterministic first moments differ");
  compare_state(lhs.v_state, rhs.v_state,
                "deterministic second moments differ");
}

} // namespace

int main() {
  return run_parity("checkpoint_continuation", [] {
    nsos::set_strict_gpu_execution(true);
    nsos::determinism::set_deterministic_reductions(true);

    ModelConfig config;
    config.num_layers = 1;
    config.d_model = 16;
    config.vocab_size = 40;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.attention_period = 64;
    config.attention_slot = 63;
    config.use_moe = false;
    config.use_ttt = false;
    config.use_chrass = false;
    config.use_kan = false;
    config.mamba2_faithful = false;
    config.tie_word_embeddings = false;
    config.use_exact_attention_training = false;
    config.dropout = 0.0f;

    JambaModel uninterrupted(config, Device::GPU);
    Trainer first(&uninterrupted, 1.5e-3f);
    TrainPhaseScheduler schedule;
    schedule.progressive_qat_enabled = false;
    first.configure_progressive_qat(schedule);
    first.weight_decay = 0.0f;
    first.warmup_steps = 1;
    first.total_training_steps = 8;
    first.max_grad_norm = 10.0f;
    first.loss_scale = 4096.0f;
    first.loss_scale_growth_tracker = 17;
    uninterrupted.set_training_rng_sequence(1234);
    require_cuda_parameters(uninterrupted);

    const std::vector<int> tokens = {1, 3, 5, 7, 9, 11};
    const std::vector<int> targets = {3, 5, 7, 9, 11, 13};
    const auto before_backward = snapshot_parameters(uninterrupted);
    const float backward_loss =
        first.accumulate_gradients(tokens, targets);
    require(std::isfinite(backward_loss),
            "GPU forward/backward loss is non-finite");
    require(first.global_step_count == 0,
            "gradient-only GPU pass advanced the optimizer");
    require(first.m_state.empty() && first.v_state.empty(),
            "gradient-only GPU pass allocated optimizer moments");
    require(max_parameter_difference(uninterrupted, before_backward) == 0.0f,
            "gradient-only GPU pass mutated model weights");
    bool saw_nonzero_gpu_gradient = false;
    for (Parameter* parameter : uninterrupted.parameters()) {
      if (!parameter || parameter->grad.size == 0) {
        continue;
      }
      require(parameter->grad.get_device() == Device::GPU,
              "backward produced a host-resident parameter gradient");
      if (parameter->grad.norm() > 0.0f) {
        saw_nonzero_gpu_gradient = true;
      }
      parameter->zero_grad();
    }
    require(saw_nonzero_gpu_gradient,
            "GPU backward produced no nonzero parameter gradient");
    cuda_sync_or_throw("checkpoint/backward_only");

    const float first_loss = first.train_step(tokens, targets);
    require(std::isfinite(first_loss), "first GPU training loss is non-finite");
    cuda_sync_or_throw("checkpoint/first_step");
    const float first_update =
        max_parameter_difference(uninterrupted, before_backward);
    require(first_update > 0.0f,
            "first GPU optimizer step changed no model parameter");
    require(first.global_step_count == 1,
            "first GPU optimizer commit did not advance exactly one step");
    require(!first.m_state.empty() &&
                first.m_state.size() == first.v_state.size(),
            "first GPU optimizer commit did not create aligned Adam moments");
    for (const auto& [parameter, moment] : first.m_state) {
      require(parameter != nullptr && moment.get_device() == Device::GPU,
              "first Adam moment is not resident on GPU");
    }
    for (const auto& [parameter, moment] : first.v_state) {
      require(parameter != nullptr && moment.get_device() == Device::GPU,
              "second Adam moment is not resident on GPU");
    }
    std::cout << "[GPUParity:checkpoint_continuation] backward_loss="
              << backward_loss << " first_loss=" << first_loss
              << " first_update_max=" << first_update << std::endl;

#ifdef NSOS_ENABLE_TEST_HOOKS
    // The common FP32 path defers the pre-update finite result to the fused
    // optimizer's single device-to-host status read. Prove that a rejected
    // device-side gate performs no partial weight/moment update and leaves the
    // trainer recoverable.
    const auto before_rejected_step =
        snapshot_parameters(uninterrupted);
    const auto m_before_rejected_step =
        snapshot_optimizer_state(first.m_state);
    const auto v_before_rejected_step =
        snapshot_optimizer_state(first.v_state);
    const int global_step_before_rejection =
        first.global_step_count;
    nsos::testing::inject_training_nan_before_optimizer();
    bool rejected_nonfinite = false;
    try {
      (void)first.train_step(tokens, targets);
    } catch (const std::runtime_error&) {
      rejected_nonfinite = true;
    }
    cuda_sync_or_throw("checkpoint/deferred_finite_gate");
    require(rejected_nonfinite,
            "deferred GPU finite gate accepted an injected NaN");
    require(first.last_optimizer_step_skipped,
            "deferred GPU finite gate did not report a skipped step");
    require(!first.optimizer_state_poisoned(),
            "clean deferred GPU rejection poisoned the optimizer");
    require(first.global_step_count == global_step_before_rejection,
            "deferred GPU rejection advanced the optimizer step");
    require(
        max_parameter_difference(
            uninterrupted, before_rejected_step) == 0.0f,
        "deferred GPU rejection changed a model parameter");
    require_optimizer_state_equal(
        first.m_state, m_before_rejected_step,
        "deferred GPU rejection changed first moments");
    require_optimizer_state_equal(
        first.v_state, v_before_rejected_step,
        "deferred GPU rejection changed second moments");
    std::cout
        << "[GPUParity:checkpoint_continuation] deferred_finite_gate="
        << "rejected_without_commit" << std::endl;
#endif

    const auto root = std::filesystem::temp_directory_path();
    const auto model_path = root / "nsos_gpu_checkpoint_model.bin";
    const auto state_path = root / "nsos_gpu_checkpoint_state.bin";
    const auto pool_before_snapshot = nsos::pool_stats();
    const TensorRandomState random_state_before_snapshot =
        capture_tensor_random_state();
    auto checkpoint_snapshot = first.capture_checkpoint_snapshot();
    const TensorRandomState random_state_after_snapshot =
        capture_tensor_random_state();
    const auto pool_after_snapshot = nsos::pool_stats();
    require(checkpoint_snapshot &&
                checkpoint_snapshot->global_step() == first.global_step_count,
            "host checkpoint snapshot captured the wrong safe point");
    require(random_state_before_snapshot == random_state_after_snapshot,
            "host checkpoint snapshot perturbed the tensor RNG stream");
    require(pool_after_snapshot.allocated_bytes ==
                pool_before_snapshot.allocated_bytes &&
                pool_after_snapshot.reserved_bytes ==
                    pool_before_snapshot.reserved_bytes,
            "host checkpoint snapshot duplicated model state in VRAM");
    checkpoint_snapshot->write(model_path.string(), state_path.string());
    bool rejected_second_write = false;
    try {
      checkpoint_snapshot->write(model_path.string(), state_path.string());
    } catch (const std::logic_error&) {
      rejected_second_write = true;
    }
    require(rejected_second_write,
            "checkpoint snapshot was not one-shot");

    {
      std::ifstream state(state_path, std::ios::binary);
      require(static_cast<bool>(state), "cannot read checkpoint sidecar");
      uint32_t magic = 0;
      uint32_t version = 0;
      state.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      state.read(reinterpret_cast<char*>(&version), sizeof(version));
      require(static_cast<bool>(state), "checkpoint sidecar header is truncated");
      require(magic == 0x4E535452u,
              "training-state sidecar magic is not NSTR");
      require(version == 10u, "training-state sidecar is not checkpoint v10");
    }

    JambaModel resumed(config, Device::GPU);
    resumed.load(model_path.string(), true);
    Trainer resumed_trainer(&resumed, 9.0f);
    resumed_trainer.load_training_state(state_path.string(),
                                        model_path.string());
    require_cuda_parameters(resumed);
    require(resumed_trainer.global_step_count == first.global_step_count,
            "GPU checkpoint did not restore global step");
    require(resumed_trainer.loss_scale == first.loss_scale &&
                resumed_trainer.loss_scale_growth_tracker ==
                    first.loss_scale_growth_tracker,
            "GPU checkpoint did not restore loss scaler");
    require(resumed.training_rng_sequence() ==
                uninterrupted.training_rng_sequence(),
            "GPU checkpoint did not restore training RNG sequence");
    require(resumed_trainer.m_state.size() == first.m_state.size() &&
                resumed_trainer.v_state.size() == first.v_state.size(),
            "GPU checkpoint did not restore Adam state");
    require_optimizer_states_equal(
        first, uninterrupted, resumed_trainer, resumed);
    const float round_trip_difference =
        max_parameter_difference(uninterrupted, resumed);
    require(round_trip_difference == 0.0f,
            "GPU checkpoint model round-trip changed parameters");

    const float uninterrupted_loss = first.train_step(tokens, targets);
    const float resumed_loss = resumed_trainer.train_step(tokens, targets);
    cuda_sync_or_throw("checkpoint/continuation");
    require(std::isfinite(uninterrupted_loss) && std::isfinite(resumed_loss),
            "GPU continuation produced a non-finite loss");
    require(
        std::memcmp(
            &uninterrupted_loss, &resumed_loss, sizeof(float)) == 0,
        "GPU resumed loss is not bitwise identical to uninterrupted loss");
    const float continuation_difference =
        max_parameter_difference(uninterrupted, resumed);
    require(continuation_difference == 0.0f,
            "GPU resumed update is not bitwise identical");
    require_optimizer_states_equal(
        first, uninterrupted, resumed_trainer, resumed);
    std::cout << "[GPUParity:checkpoint_continuation] version=10"
              << " round_trip_param_diff=" << round_trip_difference
              << " continuation_loss_diff="
              << std::abs(uninterrupted_loss - resumed_loss)
              << " continuation_param_diff=" << continuation_difference
              << std::endl;

    std::filesystem::remove(model_path);
    std::filesystem::remove(state_path);

    // Production determinism gate: two fresh GPU models for every supported
    // architecture arm must produce exactly the same reported loss, weights
    // and Adam moments. Besides the ordered loss reduction, embedding scatter
    // and optimizer, this directly covers faithful Mamba, exact Attention and
    // the real parallel-gated Mamba+Attention composition.
    ModelConfig deterministic_base = config;
    deterministic_base.architecture_schema_version = 2;
    deterministic_base.d_model = 64;
    deterministic_base.mamba2_faithful = true;
    deterministic_base.mamba_expand = 2;
    deterministic_base.mamba_state_expansion = true;
    deterministic_base.mamba_d_state = 8;
    deterministic_base.mamba_head_dim = 32;
    deterministic_base.mamba_n_groups = 1;
    deterministic_base.faithful_attention_linears = true;
    deterministic_base.force_mamba_last_layer = false;
    deterministic_base.use_gradient_checkpointing = false;

    const auto run_deterministic_arm =
        [&](const char* arm, ModelConfig arm_config,
            bool require_mamba_update, bool require_attention_update,
            uint64_t seed) {
      std::vector<int> arm_tokens=tokens, arm_targets=targets;
      if ((arm_config.use_moe && nsos::training_policy::moe_wmma_training()) ||
          (arm_config.use_kan && nsos::training_policy::kan_wmma_training())) {
        arm_tokens.resize(33); arm_targets.resize(33);
        for (int i=0;i<33;++i) { arm_tokens[i]=1+i%31; arm_targets[i]=1+(i+1)%31; }
      }
      nsos::determinism::DeterminismManager::instance().set_global_seed(seed);
      JambaModel model_a(arm_config, Device::GPU);
      nsos::determinism::DeterminismManager::instance().set_global_seed(seed);
      JambaModel model_b(arm_config, Device::GPU);
      require(max_parameter_difference(model_a, model_b) == 0.0f,
              "same-seed deterministic models initialized differently");
      const auto initial_parameters = snapshot_parameters(model_a);

      Trainer trainer_a(&model_a, 1.5e-3f);
      Trainer trainer_b(&model_b, 1.5e-3f);
      for (Trainer* trainer : {&trainer_a, &trainer_b}) {
        TrainPhaseScheduler deterministic_schedule;
        deterministic_schedule.progressive_qat_enabled = false;
        trainer->configure_progressive_qat(deterministic_schedule);
        trainer->weight_decay = 0.01f;
        trainer->warmup_steps = 1;
        trainer->total_training_steps = 4;
        trainer->max_grad_norm = 1.0f;
      }

      const std::string checkpoint_stem =
          "nsos_gpu_deterministic_" + std::string(arm) + "_" +
          std::to_string(seed);
      const auto deterministic_model_path =
          root / (checkpoint_stem + "_model.bin");
      const auto deterministic_state_path =
          root / (checkpoint_stem + "_state.bin");
      std::unique_ptr<JambaModel> resumed_model;
      std::unique_ptr<Trainer> resumed_arm_trainer;

      for (int step = 0; step < 4; ++step) {
        nsos::reset_gpu_transfer_stats();
        const float loss_a = trainer_a.train_step(arm_tokens, arm_targets);
        const float loss_b = trainer_b.train_step(arm_tokens, arm_targets);
        const nsos::GpuTransferStats training_transfers =
            nsos::gpu_transfer_stats();
        const bool combined_finite_status =
            nsos::optimizer_policy::
                deterministic_finite_gate_deferred_enabled();
        const uint64_t expected_calls_per_model =
            (combined_finite_status ? 3u : 4u) + (arm_config.use_moe ? 2u : 0u);
        const uint64_t expected_bytes_per_model =
            (combined_finite_status ? 16u : 20u) + (arm_config.use_moe ?
                (arm_config.num_experts + 2u)*sizeof(int) + sizeof(float) : 0u);
        // Per model and step, production transfers loss f32 + finite i32 +
        // deterministic norm f64 + commit i32 (20 bytes). The experimental
        // combined lane transfers loss f32 + norm/finite f64 + commit i32
        // (16 bytes). Both contracts forbid per-parameter D2H payloads.
        // The one-layer grouped MoE arm adds only late registry offsets and
        // one aggregate Switch-loss scalar. Load telemetry is device-resident.
        require(training_transfers.d2h_calls ==
                    2u * expected_calls_per_model,
                "deterministic training emitted unexpected D2H calls: observed=" +
                    std::to_string(training_transfers.d2h_calls) + " expected=" +
                    std::to_string(2u * expected_calls_per_model));
        require(training_transfers.d2h_bytes ==
                    2u * expected_bytes_per_model,
                "deterministic training copied a non-control GPU payload");
        cuda_sync_or_throw("checkpoint/deterministic_training");
        require(std::memcmp(&loss_a, &loss_b, sizeof(float)) == 0,
                "same-seed deterministic training loss differs bitwise");
        require(max_parameter_difference(model_a, model_b) == 0.0f,
                "same-seed deterministic weights differ bitwise");
        require_optimizer_states_equal(
            trainer_a, model_a, trainer_b, model_b);

        if (resumed_model && resumed_arm_trainer) {
          nsos::reset_gpu_transfer_stats();
          const float resumed_arm_loss =
              resumed_arm_trainer->train_step(arm_tokens, arm_targets);
          const nsos::GpuTransferStats resumed_transfers =
              nsos::gpu_transfer_stats();
          require(resumed_transfers.d2h_calls ==
                      expected_calls_per_model,
                  "resumed deterministic training emitted unexpected D2H calls");
          require(resumed_transfers.d2h_bytes ==
                      expected_bytes_per_model,
                  "resumed deterministic training copied a non-control payload");
          cuda_sync_or_throw("checkpoint/deterministic_resume");
          require(
              std::memcmp(
                  &loss_a, &resumed_arm_loss, sizeof(float)) == 0,
              "resumed deterministic arm loss differs bitwise");
          require(max_parameter_difference(
                      model_a, *resumed_model) == 0.0f,
                  "resumed deterministic arm weights differ bitwise");
          require_optimizer_states_equal(
              trainer_a, model_a, *resumed_arm_trainer, *resumed_model);
        }

        if (step == 1) {
          model_a.save(deterministic_model_path.string());
          trainer_a.save_training_state(
              deterministic_state_path.string(),
              deterministic_model_path.string());
          resumed_model =
              std::make_unique<JambaModel>(arm_config, Device::GPU);
          resumed_model->load(deterministic_model_path.string(), true);
          resumed_arm_trainer =
              std::make_unique<Trainer>(resumed_model.get(), 9.0f);
          if (arm_config.use_moe && nsos::training_policy::grouped_moe_training()) {
            const auto weights_before = snapshot_parameters(*resumed_model);
            const auto moments_before = snapshot_optimizer_state(resumed_arm_trainer->m_state);
            const auto variance_before = snapshot_optimizer_state(resumed_arm_trainer->v_state);
            const auto rng_before = resumed_model->training_rng_sequence();
            const auto step_before = resumed_arm_trainer->global_step_count;
            const auto lr_before = resumed_arm_trainer->learning_rate;
            std::string mismatch;
            set_grouped_moe_policy(false);
            try {
              resumed_arm_trainer->load_training_state(
                  deterministic_state_path.string(), deterministic_model_path.string());
            } catch (const std::exception& error) { mismatch = error.what(); }
            catch (...) { set_grouped_moe_policy(true); throw; }
            set_grouped_moe_policy(true);
            require(mismatch.find("moe.training_compute_policy") != std::string::npos,
                    "grouped MoE checkpoint accepted a different compute policy");
            require(max_parameter_difference(*resumed_model, weights_before) == 0,
                    "MoE policy rejection mutated model weights");
            require_optimizer_state_equal(resumed_arm_trainer->m_state, moments_before,
                                          "MoE policy rejection mutated first moments");
            require_optimizer_state_equal(resumed_arm_trainer->v_state, variance_before,
                                          "MoE policy rejection mutated second moments");
            require(resumed_arm_trainer->global_step_count == step_before &&
                    resumed_arm_trainer->learning_rate == lr_before &&
                    resumed_model->training_rng_sequence() == rng_before,
                    "MoE policy rejection committed metadata");
            if (nsos::training_policy::moe_wmma_training()) {
              mismatch.clear();
              set_wmma_moe_policy(false);
              try {
                resumed_arm_trainer->load_training_state(deterministic_state_path.string(), deterministic_model_path.string());
              } catch (const std::exception& error) { mismatch=error.what(); }
              catch (...) { set_wmma_moe_policy(true); throw; }
              set_wmma_moe_policy(true);
              require(mismatch.find("moe.training_wmma_policy")!=std::string::npos,"WMMA checkpoint accepted scalar policy");
              require(max_parameter_difference(*resumed_model,weights_before)==0,"WMMA rejection mutated weights");
              require_optimizer_state_equal(resumed_arm_trainer->m_state,moments_before,"WMMA rejection mutated first moments");
              require_optimizer_state_equal(resumed_arm_trainer->v_state,variance_before,"WMMA rejection mutated second moments");
              require(resumed_arm_trainer->global_step_count==step_before && resumed_arm_trainer->learning_rate==lr_before &&
                      resumed_model->training_rng_sequence()==rng_before,"WMMA rejection committed metadata");
            }
          }
          if (arm_config.use_ttt && nsos::training_policy::full_ttt_bptt()) {
            const auto weights_before = snapshot_parameters(*resumed_model);
            const auto moments_before = snapshot_optimizer_state(resumed_arm_trainer->m_state);
            const auto variance_before = snapshot_optimizer_state(resumed_arm_trainer->v_state);
            const auto rng_before = resumed_model->training_rng_sequence();
            const auto step_before = resumed_arm_trainer->global_step_count;
            const float lr_before = resumed_arm_trainer->learning_rate;
            std::string mismatch;
            set_full_ttt_policy(false);
            try {
              resumed_arm_trainer->load_training_state(
                  deterministic_state_path.string(), deterministic_model_path.string());
            } catch (const std::exception& error) {
              mismatch = error.what();
            } catch (...) {
              set_full_ttt_policy(true);
              throw;
            }
            set_full_ttt_policy(true);
            require(mismatch.find("ttt.training_policy") != std::string::npos,
                    "full TTT checkpoint accepted a truncated derivative policy");
            require(max_parameter_difference(*resumed_model, weights_before) == 0.0f,
                    "TTT policy rejection mutated model weights");
            require_optimizer_state_equal(resumed_arm_trainer->m_state, moments_before,
                                          "TTT policy rejection mutated first moments");
            require_optimizer_state_equal(resumed_arm_trainer->v_state, variance_before,
                                          "TTT policy rejection mutated second moments");
            require(resumed_arm_trainer->global_step_count == step_before &&
                        resumed_arm_trainer->learning_rate == lr_before &&
                        resumed_model->training_rng_sequence() == rng_before,
                    "TTT policy rejection committed training metadata");
            std::cout << "[GPUParity:checkpoint_continuation] full_ttt_policy_mismatch=rejected_without_commit\n";
          }
          if (arm_config.use_kan && nsos::training_policy::kan_recompute_training()) {
            const auto before_weights=snapshot_parameters(*resumed_model);
            const auto before_m=snapshot_optimizer_state(resumed_arm_trainer->m_state);
            const auto before_v=snapshot_optimizer_state(resumed_arm_trainer->v_state);
            const auto before_step=resumed_arm_trainer->global_step_count;
            const bool had_wmma=nsos::training_policy::kan_wmma_training();
            std::string mismatch;
            if (had_wmma) {
              set_kan_wmma_policy(false);
              try { resumed_arm_trainer->load_training_state(deterministic_state_path.string(),deterministic_model_path.string()); }
              catch (const std::exception& error) { mismatch=error.what(); }
              catch (...) { set_kan_wmma_policy(true); throw; }
              set_kan_wmma_policy(true);
              require(mismatch.find("kan.wmma_policy")!=std::string::npos,"KAN checkpoint accepted a scalar RBF provider");
              require(max_parameter_difference(*resumed_model,before_weights)==0,"KAN WMMA rejection changed weights");
              require_optimizer_state_equal(resumed_arm_trainer->m_state,before_m,"KAN WMMA rejection changed first moments");
              require_optimizer_state_equal(resumed_arm_trainer->v_state,before_v,"KAN WMMA rejection changed second moments");
              require(resumed_arm_trainer->global_step_count==before_step,"KAN WMMA rejection advanced step");
            }
            mismatch.clear();
            set_kan_wmma_policy(false);
            set_kan_policy(false);
            try { resumed_arm_trainer->load_training_state(deterministic_state_path.string(),deterministic_model_path.string()); }
            catch (const std::exception& error) { mismatch=error.what(); }
            catch (...) { set_kan_policy(true); set_kan_wmma_policy(had_wmma); throw; }
            set_kan_policy(true);
            set_kan_wmma_policy(had_wmma);
            require(mismatch.find("kan.training_policy")!=std::string::npos,"KAN checkpoint accepted a legacy compute policy");
            require(max_parameter_difference(*resumed_model,before_weights)==0,"KAN policy rejection changed weights");
            require_optimizer_state_equal(resumed_arm_trainer->m_state,before_m,"KAN policy rejection changed first moments");
            require_optimizer_state_equal(resumed_arm_trainer->v_state,before_v,"KAN policy rejection changed second moments");
            require(resumed_arm_trainer->global_step_count==before_step,"KAN policy rejection advanced step");
          }
          resumed_arm_trainer->load_training_state(
              deterministic_state_path.string(),
              deterministic_model_path.string());
          require_cuda_parameters(*resumed_model);
          require(resumed_arm_trainer->global_step_count ==
                      trainer_a.global_step_count,
                  "deterministic resume restored the wrong global step");
          require(resumed_model->training_rng_sequence() ==
                      model_a.training_rng_sequence(),
                  "deterministic resume restored the wrong RNG sequence");
          require(max_parameter_difference(
                      model_a, *resumed_model) == 0.0f,
                  "deterministic checkpoint round trip changed weights");
          require_optimizer_states_equal(
              trainer_a, model_a, *resumed_arm_trainer, *resumed_model);
        }
      }

      bool saw_mamba_update = false;
      bool saw_attention_update = false;
      bool saw_ttt_update = false;
      bool saw_ttt_task_gradient = false;
      bool saw_kan_task_gradient = false;
      bool saw_expert_gradient = false, saw_router_gradient = false;
      const auto trained_parameters = model_a.parameters();
      require(trained_parameters.size() == initial_parameters.size(),
              "deterministic parameter registry changed during training");
      for (size_t index = 0; index < trained_parameters.size(); ++index) {
        const Parameter* parameter = trained_parameters[index];
        require(parameter != nullptr,
                "deterministic training contains a null parameter");
        const Tensor current = parameter->data.cpu();
        const Tensor& initial = initial_parameters[index];
        const bool changed =
            std::memcmp(current.data(), initial.data(),
                        static_cast<size_t>(current.size) * sizeof(float)) != 0;
        if (!changed) {
          continue;
        }
        saw_mamba_update =
            saw_mamba_update ||
            parameter->name.find(".mamba.") != std::string::npos;
        saw_attention_update =
            saw_attention_update ||
            parameter->name.find(".attn.") != std::string::npos;
        saw_ttt_update = saw_ttt_update || parameter->name.find(".ttt.") != std::string::npos;
        if (parameter->grad.size > 0) {
          const float norm = parameter->grad.norm();
          const bool active = std::isfinite(norm) && norm > 0;
          saw_expert_gradient |= active && parameter->name.find(".experts.") != std::string::npos;
          saw_router_gradient |= active && parameter->name.find(".router.") != std::string::npos;
          saw_kan_task_gradient |= active && parameter->name.find(".kan.") != std::string::npos;
        }
        if (parameter->name.find(".ttt.") != std::string::npos && parameter->grad.size > 0) {
          const float norm = parameter->grad.norm();
          saw_ttt_task_gradient = saw_ttt_task_gradient || (std::isfinite(norm) && norm > 0);
        }
      }
      require(!require_mamba_update || saw_mamba_update,
              "deterministic arm did not update its Mamba branch");
      require(!require_attention_update || saw_attention_update,
              "deterministic arm did not update its Attention branch");
      require(!arm_config.use_ttt || saw_ttt_update, "deterministic arm did not update its TTT branch");
      require(!arm_config.use_ttt || saw_ttt_task_gradient, "TTT changed only by decay, not task gradient");
      require(!arm_config.use_moe || (saw_expert_gradient && saw_router_gradient),
              "MoE did not train both experts and router");
      require(!arm_config.use_kan || saw_kan_task_gradient,
              "KAN changed only by decay, not task gradient");
      require(resumed_model && resumed_arm_trainer,
              "deterministic checkpoint continuation was not exercised");
      std::filesystem::remove(deterministic_model_path);
      std::filesystem::remove(deterministic_state_path);
      std::cout
          << "[GPUParity:checkpoint_continuation] deterministic_" << arm
          << "=loss_weights_moments_resume_exact steps=4"
          << " mamba_update=" << saw_mamba_update
          << " attention_update=" << saw_attention_update << std::endl;
    };

    ModelConfig deterministic_mamba = deterministic_base;
    deterministic_mamba.num_layers = 1;
    deterministic_mamba.hybrid_composition =
        HybridComposition::ParallelGated;
    deterministic_mamba.attention_period = 64;
    deterministic_mamba.attention_slot = 63;
    deterministic_mamba.use_exact_attention_training = false;
    run_deterministic_arm(
        "mamba", deterministic_mamba, true, false, 20260729ULL);

    ModelConfig deterministic_attention = deterministic_base;
    deterministic_attention.num_layers = 1;
    deterministic_attention.hybrid_composition =
        HybridComposition::LegacyReplacement;
    deterministic_attention.attention_period = 1;
    deterministic_attention.attention_slot = 0;
    deterministic_attention.use_exact_attention_training = true;
    run_deterministic_arm(
        "attention", deterministic_attention, false, true, 20260730ULL);

    ModelConfig deterministic_hybrid = deterministic_base;
    deterministic_hybrid.num_layers = 2;
    deterministic_hybrid.hybrid_composition =
        HybridComposition::ParallelGated;
    deterministic_hybrid.attention_period = 2;
    deterministic_hybrid.attention_slot = 1;
    deterministic_hybrid.use_exact_attention_training = true;
    run_deterministic_arm(
        "hybrid", deterministic_hybrid, true, true, 20260731ULL);
    if (nsos::training_policy::full_ttt_bptt()) {
      ModelConfig deterministic_ttt = deterministic_base;
      deterministic_ttt.num_layers = 1;
      deterministic_ttt.use_ttt = true;
      deterministic_ttt.ttt_period = 1;
      deterministic_ttt.ttt_slot = 0;
      deterministic_ttt.attention_period = 64;
      deterministic_ttt.attention_slot = 63;
      run_deterministic_arm("ttt_full", deterministic_ttt, false, false, 20260930ULL);
    }
    if (nsos::training_policy::kan_recompute_training()) {
      ModelConfig deterministic_kan=deterministic_mamba;
      deterministic_kan.use_kan=true;
      const auto before=nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanRecompute)];
      const auto before_wmma=nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanWmmaGemm)];
      run_deterministic_arm("kan_recompute",deterministic_kan,true,false,20260930ULL);
      require(nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanRecompute)]>before,
              "KAN checkpoint arm never dispatched recompute");
      if (nsos::training_policy::kan_wmma_training())
        require(nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanWmmaGemm)]>before_wmma,
                "KAN checkpoint arm never dispatched WMMA");
    }
    if (nsos::training_policy::grouped_moe_training()) {
      ModelConfig deterministic_moe = deterministic_mamba;
      deterministic_moe.use_moe = true;
      deterministic_moe.moe_period = 1;
      deterministic_moe.moe_slot = 0;
      deterministic_moe.num_experts = 4;
      deterministic_moe.num_experts_per_token = 2;
      deterministic_moe.moe_expert_hidden_dim = 64;
      const auto wmma_before=nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::GroupedMoeWmmaGemm)];
      run_deterministic_arm("moe_grouped", deterministic_moe, true, false, 20260930ULL);
      if (nsos::training_policy::moe_wmma_training())
        require(nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::GroupedMoeWmmaGemm)]>wmma_before,
                "WMMA checkpoint arm never dispatched WMMA");
    }

    nsos::determinism::set_deterministic_reductions(false);
    nsos::set_strict_gpu_execution(false);
  });
}
