#include "gpu_parity_common.h"

#include "jamba.h"
#include "nsos/determinism.h"
#include "trainer.h"
#include "optimizer_runtime_policy.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void set_optimizer_policy(const char* value) {
#ifdef _WIN32
  if (_putenv_s("NSOS_DETERMINISTIC_MULTI_TENSOR_OPT", value) != 0) {
    throw std::runtime_error("cannot set deterministic optimizer test policy");
  }
#else
  if (setenv("NSOS_DETERMINISTIC_MULTI_TENSOR_OPT", value, 1) != 0) {
    throw std::runtime_error("cannot set deterministic optimizer test policy");
  }
#endif
}

void set_clip_policy(const char* value) {
#ifdef _WIN32
  require(_putenv_s("NSOS_DEVICE_GRAD_CLIP", value) == 0, "cannot set clip policy");
#else
  require(setenv("NSOS_DEVICE_GRAD_CLIP", value, 1) == 0, "cannot set clip policy");
#endif
}

void set_chunk_policy(const char* enabled, const char* elements) {
#ifdef _WIN32
  if (_putenv_s("NSOS_DETERMINISTIC_ADAMW_CHUNKED", enabled) != 0 ||
      _putenv_s("NSOS_DETERMINISTIC_ADAMW_CHUNK_ELEMENTS", elements) != 0) {
    throw std::runtime_error("cannot set deterministic AdamW chunk policy");
  }
#else
  if (setenv("NSOS_DETERMINISTIC_ADAMW_CHUNKED", enabled, 1) != 0 ||
      setenv("NSOS_DETERMINISTIC_ADAMW_CHUNK_ELEMENTS", elements, 1) != 0) {
    throw std::runtime_error("cannot set deterministic AdamW chunk policy");
  }
#endif
}

void set_finite_scan_policy(const char* enabled, const char* elements) {
#ifdef _WIN32
  if (_putenv_s("NSOS_OPTIMIZER_FINITE_CHUNKED", enabled) != 0 ||
      _putenv_s("NSOS_OPTIMIZER_FINITE_CHUNK_ELEMENTS", elements) != 0) {
    throw std::runtime_error("cannot set optimizer finite-scan policy");
  }
#else
  if (setenv("NSOS_OPTIMIZER_FINITE_CHUNKED", enabled, 1) != 0 ||
      setenv("NSOS_OPTIMIZER_FINITE_CHUNK_ELEMENTS", elements, 1) != 0) {
    throw std::runtime_error("cannot set optimizer finite-scan policy");
  }
#endif
}

struct MilestoneSnapshot {
  int step = 0;
  float loss = 0.0f;
  std::vector<Tensor> weights;
  std::vector<std::optional<Tensor>> first_moments;
  std::vector<std::optional<Tensor>> second_moments;
  std::vector<uint64_t> versions;
};

MilestoneSnapshot capture(Trainer& trainer, JambaModel& model, float loss) {
  MilestoneSnapshot result;
  result.step = trainer.global_step_count;
  result.loss = loss;
  for (Parameter* parameter : model.parameters()) {
    require(parameter != nullptr, "AdamW snapshot has a null parameter");
    result.weights.push_back(parameter->data.cpu().clone());
    result.versions.push_back(parameter->version);
    const auto m = trainer.m_state.find(parameter);
    const auto v = trainer.v_state.find(parameter);
    require((m == trainer.m_state.end()) == (v == trainer.v_state.end()),
            "AdamW snapshot has only one moment tensor");
    if (m == trainer.m_state.end()) {
      result.first_moments.emplace_back(std::nullopt);
      result.second_moments.emplace_back(std::nullopt);
    } else {
      result.first_moments.emplace_back(m->second.cpu().clone());
      result.second_moments.emplace_back(v->second.cpu().clone());
    }
  }
  return result;
}

void require_tensor_bits(const Tensor& expected, const Tensor& actual,
                         const char* message) {
  require(expected.shape == actual.shape, message);
  require(std::memcmp(expected.data(), actual.data(),
                      static_cast<size_t>(expected.size) * sizeof(float)) == 0,
          message);
}

void compare(const MilestoneSnapshot& expected, Trainer& trainer,
             JambaModel& model, float loss) {
  require(expected.step == trainer.global_step_count,
          "AdamW milestone step mismatch");
  require(std::memcmp(&expected.loss, &loss, sizeof(float)) == 0,
          "AdamW milestone loss differs bitwise");
  const auto parameters = model.parameters();
  require(parameters.size() == expected.weights.size(),
          "AdamW parameter registry changed");
  for (size_t index = 0; index < parameters.size(); ++index) {
    Parameter* parameter = parameters[index];
    require(parameter != nullptr, "AdamW comparison has a null parameter");
    const Tensor weight = parameter->data.cpu();
    require_tensor_bits(expected.weights[index], weight,
                        "AdamW weights differ bitwise");
    require(parameter->version == expected.versions[index],
            "AdamW content versions differ");
    const auto m = trainer.m_state.find(parameter);
    const auto v = trainer.v_state.find(parameter);
    require((m == trainer.m_state.end()) == (v == trainer.v_state.end()),
            "AdamW comparison has only one moment tensor");
    const bool expected_state = expected.first_moments[index].has_value();
    require(expected_state == expected.second_moments[index].has_value(),
            "AdamW reference has only one moment tensor");
    require(expected_state == (m != trainer.m_state.end()),
            "AdamW optimizer-state presence differs");
    if (expected_state) {
      const Tensor first = m->second.cpu();
      const Tensor second = v->second.cpu();
      require_tensor_bits(*expected.first_moments[index], first,
                          "AdamW first moments differ bitwise");
      require_tensor_bits(*expected.second_moments[index], second,
                          "AdamW second moments differ bitwise");
    }
  }
}

ModelConfig fixture_config() {
  ModelConfig config;
  config.num_layers = 1;
  config.d_model = 8;
  config.vocab_size = 16;
  config.n_heads = 2;
  config.n_kv_heads = 1;
  config.attention_period = 64;
  config.attention_slot = 63;
  config.mamba2_faithful = false;
  config.tie_word_embeddings = true;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_chrass = false;
  config.use_kan = false;
  config.dropout = 0.0f;
  config.use_exact_attention_training = false;
  return config;
}

void configure(Trainer& trainer) {
  TrainPhaseScheduler schedule;
  schedule.progressive_qat_enabled = false;
  trainer.configure_progressive_qat(schedule);
  trainer.weight_decay = 0.01f;
  trainer.max_grad_norm = 1.0f;
  trainer.warmup_steps = 1;
  trainer.total_training_steps = 1'000;
}

}  // namespace

int main() {
  return run_parity("deterministic_adamw_1000", [] {
    set_strict_gpu_execution(true);
    determinism::set_deterministic_reductions(true);
    const std::vector<int> tokens = {1, 2, 3, 4};
    const std::vector<int> targets = {2, 3, 4, 5};
    constexpr std::array<int, 4> milestones = {1, 10, 100, 1'000};
    std::vector<MilestoneSnapshot> reference;
    const bool compare_device_clip = optimizer_policy::device_gradient_clip_enabled();
    if (compare_device_clip) set_clip_policy("0");

    set_optimizer_policy("0");
    set_finite_scan_policy("0", "32");
    determinism::DeterminismManager::instance().set_global_seed(0xA17D5EEDu);
    {
      JambaModel model(fixture_config(), Device::GPU);
      Trainer trainer(&model, 7.5e-4f);
      configure(trainer);
      size_t milestone_index = 0;
      for (int step = 1; step <= milestones.back(); ++step) {
        const float loss = trainer.train_step(tokens, targets);
        require(std::isfinite(loss), "legacy deterministic AdamW loss is non-finite");
        if (step == milestones[milestone_index]) {
          cuda_sync_or_throw("deterministic_adamw/reference_milestone");
          reference.push_back(capture(trainer, model, loss));
          ++milestone_index;
          if (milestone_index == milestones.size()) break;
        }
      }
    }

    if (compare_device_clip) set_clip_policy("1");
    auto compare_policy = [&](const char* chunked, const char* elements,
                              const char* finite_chunked,
                              const char* finite_elements) {
      set_optimizer_policy("1");
      set_chunk_policy(chunked, elements);
      set_finite_scan_policy(finite_chunked, finite_elements);
      determinism::DeterminismManager::instance().set_global_seed(0xA17D5EEDu);
      JambaModel model(fixture_config(), Device::GPU);
      Trainer trainer(&model, 7.5e-4f);
      configure(trainer);
      size_t milestone_index = 0;
      for (int step = 1; step <= milestones.back(); ++step) {
        const float loss = trainer.train_step(tokens, targets);
        require(std::isfinite(loss), "multi-tensor AdamW loss is non-finite");
        if (step == milestones[milestone_index]) {
          cuda_sync_or_throw("deterministic_adamw/multi_tensor_milestone");
          compare(reference[milestone_index], trainer, model, loss);
          ++milestone_index;
          if (milestone_index == milestones.size()) break;
        }
      }
    };

    // Prove both rollback and production against the per-tensor reference.
    // Thirty-two elements force several fixture tensors across chunk bounds.
    compare_policy("0", "32", "0", "32");
    compare_policy("1", "32", "0", "32");
    compare_policy("1", "32", "1", "32");
    set_optimizer_policy("1");
    set_chunk_policy("1", "8192");
    set_finite_scan_policy("1", "8192");
    determinism::set_deterministic_reductions(false);
    set_strict_gpu_execution(false);
  });
}
