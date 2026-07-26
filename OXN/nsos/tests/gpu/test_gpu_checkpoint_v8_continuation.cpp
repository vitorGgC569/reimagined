// Training-state checkpoint v8 continuation on CUDA.
//
// The model, gradients, Adam moments and optimizer update all remain on GPU.
// After one step we save the model + v8 sidecar, restore into a new GPU model,
// and require the next uninterrupted/resumed updates to agree.

#include "gpu_parity_common.h"
#include "jamba.h"
#include "trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

using nsos::Device;
using nsos::JambaModel;
using nsos::ModelConfig;
using nsos::Parameter;
using nsos::Tensor;
using nsos::Trainer;
using nsos::TrainPhaseScheduler;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
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

} // namespace

int main() {
  return run_parity("checkpoint_v8_continuation", [] {
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
    const float first_loss = first.train_step(tokens, targets);
    require(std::isfinite(first_loss), "first GPU training loss is non-finite");
    cuda_sync_or_throw("checkpoint_v8/first_step");

    const auto root = std::filesystem::temp_directory_path();
    const auto model_path = root / "nsos_gpu_checkpoint_v8_model.bin";
    const auto state_path = root / "nsos_gpu_checkpoint_v8_state.bin";
    uninterrupted.save(model_path.string());
    first.save_training_state(state_path.string(), model_path.string());

    {
      std::ifstream state(state_path, std::ios::binary);
      require(static_cast<bool>(state), "cannot read checkpoint v8 sidecar");
      uint32_t magic = 0;
      uint32_t version = 0;
      state.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      state.read(reinterpret_cast<char*>(&version), sizeof(version));
      require(static_cast<bool>(state), "checkpoint sidecar header is truncated");
      require(magic == 0x4E535452u,
              "training-state sidecar magic is not NSTR");
      require(version == 8u, "training-state sidecar is not checkpoint v8");
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
    const float round_trip_difference =
        max_parameter_difference(uninterrupted, resumed);
    require(round_trip_difference == 0.0f,
            "GPU checkpoint model round-trip changed parameters");

    const float uninterrupted_loss = first.train_step(tokens, targets);
    const float resumed_loss = resumed_trainer.train_step(tokens, targets);
    cuda_sync_or_throw("checkpoint_v8/continuation");
    require(std::isfinite(uninterrupted_loss) && std::isfinite(resumed_loss),
            "GPU continuation produced a non-finite loss");
    require(std::abs(uninterrupted_loss - resumed_loss) <= 1e-6f,
            "GPU resumed loss diverged from uninterrupted loss");
    const float continuation_difference =
        max_parameter_difference(uninterrupted, resumed);
    require(continuation_difference <= 1e-6f,
            "GPU resumed update diverged from uninterrupted update");
    std::cout << "[GPUParity:checkpoint_v8_continuation] version=8"
              << " round_trip_param_diff=" << round_trip_difference
              << " continuation_loss_diff="
              << std::abs(uninterrupted_loss - resumed_loss)
              << " continuation_param_diff=" << continuation_difference
              << std::endl;

    std::filesystem::remove(model_path);
    std::filesystem::remove(state_path);
  });
}
