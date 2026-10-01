#include "../include/jamba.h"
#include "../include/trainer.h"
#include "../include/nsos/determinism.h"
#include "../include/nsos/sha256.h"
#include "sparse_gradient_contract.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nsos;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void set_chunk_size(const char* value) {
#ifdef _WIN32
  _putenv_s("NSOS_TRAIN_CHUNK_SIZE", value);
#else
  setenv("NSOS_TRAIN_CHUNK_SIZE", value, 1);
#endif
}

void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

struct ScopedTemporaryDirectory {
  std::filesystem::path path;

  explicit ScopedTemporaryDirectory(const std::string& prefix) {
    const auto timestamp =
        std::chrono::steady_clock::now()
            .time_since_epoch()
            .count();
    path = std::filesystem::temp_directory_path() /
           (prefix + "_" + std::to_string(timestamp));
    std::filesystem::create_directories(path);
  }

  ~ScopedTemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

std::vector<unsigned char> read_binary_file(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.is_open(), "could not open binary fixture");
  return std::vector<unsigned char>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

void write_binary_file(
    const std::filesystem::path& path,
    const std::vector<unsigned char>& bytes) {
  std::ofstream output(
      path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "could not write binary fixture");
  if (!bytes.empty()) {
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
  }
  output.close();
  require(output.good(), "binary fixture write failed");
}

void require_tensor_bitwise_equal(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& context) {
  require(actual.shape == expected.shape,
          context + ": tensor-shape mismatch");
  require(actual.get_device() == expected.get_device(),
          context + ": tensor-device mismatch");
  if (actual.size == 0) return;
  const Tensor actual_cpu = actual.cpu();
  const Tensor expected_cpu = expected.cpu();
  require(
      std::memcmp(
          actual_cpu.data(), expected_cpu.data(),
          static_cast<size_t>(actual_cpu.size) *
              sizeof(float)) == 0,
      context + ": tensor bytes changed");
}

#ifdef NSOS_ENABLE_TEST_HOOKS
struct TrainingStateFaultReset {
  ~TrainingStateFaultReset() {
    testing::clear_training_state_stage_failure();
  }
};
#endif

float max_parameter_difference(JambaModel& lhs, JambaModel& rhs) {
  auto left = lhs.parameters();
  auto right = rhs.parameters();
  require(left.size() == right.size(), "parameter-count mismatch");
  float maximum = 0.0f;
  for (size_t p = 0; p < left.size(); ++p) {
    require(left[p]->data.shape == right[p]->data.shape,
            "parameter-shape mismatch");
    Tensor a = left[p]->data.cpu();
    Tensor b = right[p]->data.cpu();
    for (int i = 0; i < a.size; ++i) {
      maximum = std::max(maximum, std::abs(a.data()[i] - b.data()[i]));
    }
  }
  return maximum;
}

void require_model_parameters_bitwise_equal(
    JambaModel& lhs, JambaModel& rhs,
    const std::string& context) {
  const auto left = lhs.parameters();
  const auto right = rhs.parameters();
  require(left.size() == right.size(),
          context + ": parameter-count mismatch");
  for (size_t index = 0; index < left.size(); ++index) {
    require(left[index] != nullptr && right[index] != nullptr,
            context + ": null parameter");
    require(left[index]->name == right[index]->name,
            context + ": parameter registry order differs");
    require_tensor_bitwise_equal(
        left[index]->data, right[index]->data,
        context + ": " + left[index]->name);
  }
}

void require_optimizer_states_by_registry_bitwise_equal(
    Trainer& lhs_trainer, JambaModel& lhs_model,
    Trainer& rhs_trainer, JambaModel& rhs_model,
    const std::string& context) {
  const auto left = lhs_model.parameters();
  const auto right = rhs_model.parameters();
  require(left.size() == right.size(),
          context + ": optimizer registry size differs");
  const auto compare =
      [&](const auto& lhs_state, const auto& rhs_state,
          const std::string& state_name) {
        require(lhs_state.size() == rhs_state.size(),
                context + ": " + state_name +
                    " cardinality differs");
        for (size_t index = 0; index < left.size(); ++index) {
          const auto lhs_entry = lhs_state.find(left[index]);
          const auto rhs_entry = rhs_state.find(right[index]);
          if (lhs_entry == lhs_state.end() &&
              rhs_entry == rhs_state.end()) {
            continue;
          }
          require(
              lhs_entry != lhs_state.end() &&
                  rhs_entry != rhs_state.end(),
              context + ": " + state_name +
                  " registry membership differs");
          require_tensor_bitwise_equal(
              lhs_entry->second, rhs_entry->second,
              context + ": " + state_name + ": " +
                  left[index]->name);
        }
      };
  compare(lhs_trainer.m_state, rhs_trainer.m_state, "first moment");
  compare(lhs_trainer.v_state, rhs_trainer.v_state, "second moment");
}

std::vector<Tensor> clone_parameter_gradients(JambaModel& model) {
  std::vector<Tensor> snapshots;
  const auto parameters = model.parameters();
  snapshots.reserve(parameters.size());
  for (const Parameter* parameter : parameters) {
    snapshots.push_back(
        parameter && parameter->grad.size > 0
            ? parameter->grad.clone()
            : Tensor());
  }
  return snapshots;
}

std::vector<Tensor> clone_parameter_values(JambaModel& model) {
  std::vector<Tensor> snapshots;
  const auto parameters = model.parameters();
  snapshots.reserve(parameters.size());
  for (const Parameter* parameter : parameters) {
    require(parameter != nullptr,
            "cannot snapshot a null model parameter");
    snapshots.push_back(parameter->data.clone());
  }
  return snapshots;
}

void require_parameter_values_bitwise_equal(
    JambaModel& model,
    const std::vector<Tensor>& expected,
    const std::string& context) {
  const auto parameters = model.parameters();
  require(parameters.size() == expected.size(),
          context + ": parameter-count mismatch");
  for (size_t index = 0; index < parameters.size(); ++index) {
    require(parameters[index] != nullptr,
            context + ": null parameter");
    require_tensor_bitwise_equal(
        parameters[index]->data, expected[index],
        context + ": " + parameters[index]->name);
  }
}

std::unordered_map<Parameter*, Tensor> clone_tensor_state_map(
    const std::unordered_map<Parameter*, Tensor>& source) {
  std::unordered_map<Parameter*, Tensor> snapshot;
  snapshot.reserve(source.size());
  for (const auto& [parameter, tensor] : source) {
    snapshot.emplace(parameter, tensor.clone());
  }
  return snapshot;
}

void require_tensor_state_map_bitwise_equal(
    const std::unordered_map<Parameter*, Tensor>& actual,
    const std::unordered_map<Parameter*, Tensor>& expected,
    const std::string& context) {
  require(actual.size() == expected.size(),
          context + ": state-map cardinality changed");
  for (const auto& [parameter, tensor] : expected) {
    const auto found = actual.find(parameter);
    require(found != actual.end(),
            context + ": state-map parameter disappeared");
    require_tensor_bitwise_equal(
        found->second, tensor, context + ": state tensor");
  }
}

void require_parameter_gradients_equal(
    JambaModel& model,
    const std::vector<Tensor>& expected,
    const std::string& context) {
  const auto parameters = model.parameters();
  require(parameters.size() == expected.size(),
          context + ": parameter-count mismatch");
  for (size_t parameter_index = 0;
       parameter_index < parameters.size();
       ++parameter_index) {
    const Parameter* parameter = parameters[parameter_index];
    require(parameter != nullptr, context + ": null parameter");
    require(parameter->grad.size == expected[parameter_index].size,
            context + ": gradient-presence mismatch for " +
                parameter->name);
    if (parameter->grad.size == 0) continue;
    const Tensor actual = parameter->grad.cpu();
    const Tensor reference = expected[parameter_index].cpu();
    require(actual.size == reference.size,
            context + ": gradient-size mismatch for " +
                parameter->name);
    for (int element = 0; element < actual.size; ++element) {
      require(actual.data()[element] == reference.data()[element],
              context + ": gradient changed for " + parameter->name);
    }
  }
}

void test_model_backward_one_shot_contract() {
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 8;
  cfg.vocab_size = 32;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 1;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  JambaModel model(cfg, Device::CPU);
  model.set_training_mode(true);
  const std::vector<int> ids = {1, 2, 3};
  Context context;
  Tensor logits = model.forward_ids(ids, &context);
  const Tensor gradient = Tensor::ones(logits.shape.dims, Device::CPU)
                              .mul(1e-3f);
  model.backward(gradient, context);

  const auto after_first_backward = clone_parameter_gradients(model);
  bool duplicate_rejected = false;
  try {
    model.backward(gradient, context);
  } catch (const std::runtime_error&) {
    duplicate_rejected = true;
  }
  require(duplicate_rejected,
          "model accepted a duplicate backward");
  require_parameter_gradients_equal(
      model, after_first_backward,
      "duplicate backward rejection");

  (void)model.forward_ids(ids, &context);
  bool invalid_forward_rejected = false;
  try {
    (void)model.forward_ids({}, &context);
  } catch (const std::invalid_argument&) {
    invalid_forward_rejected = true;
  }
  require(invalid_forward_rejected,
          "invalid forward fixture was not rejected");
  bool stale_after_failure_rejected = false;
  try {
    model.backward(gradient, context);
  } catch (const std::runtime_error&) {
    stale_after_failure_rejected = true;
  }
  require(stale_after_failure_rejected,
          "failed forward preserved a stale backward ticket");

  (void)model.forward_ids(ids, &context);
  (void)model.forward_trunk(ids, &context);
  bool trunk_stale_rejected = false;
  try {
    model.backward(gradient, context);
  } catch (const std::runtime_error&) {
    trunk_stale_rejected = true;
  }
  require(trunk_stale_rejected,
          "trunk-only forward preserved a full-model backward ticket");
  std::cout << "[invariant] model backward one-shot contract enforced"
            << std::endl;
}

void test_train_loop_chunk_invariance() {
  ModelConfig cfg;
  cfg.num_layers = 2;
  cfg.d_model = 16;
  cfg.vocab_size = 48;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  determinism::DeterminismManager::instance().set_global_seed(20260718);
  JambaModel source(cfg, Device::CPU);
  const auto checkpoint =
      std::filesystem::temp_directory_path() / "nsos_chunk_invariance.bin";
  source.save(checkpoint.string());

  JambaModel full_batch(cfg, Device::CPU);
  JambaModel micro_chunked(cfg, Device::CPU);
  full_batch.load(checkpoint.string(), true);
  micro_chunked.load(checkpoint.string(), true);

  std::vector<int> tokens(40);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<int>((i * 7 + 3) % cfg.vocab_size);
  }

  Trainer full_trainer(&full_batch, 2e-3f);
  Trainer micro_trainer(&micro_chunked, 2e-3f);
  for (Trainer* trainer : {&full_trainer, &micro_trainer}) {
    trainer->warmup_steps = 1;
    trainer->total_training_steps = 10;
    trainer->weight_decay = 0.0f;
    trainer->max_grad_norm = 1000.0f;
  }

  set_chunk_size("0");
  full_trainer.train_loop(tokens, 1, 4, 4, nullptr, 1);
  set_chunk_size("1");
  micro_trainer.train_loop(tokens, 1, 4, 4, nullptr, 1);
  set_chunk_size("0");

  const float max_diff = max_parameter_difference(full_batch, micro_chunked);
  std::filesystem::remove(checkpoint);
  std::cout << "[invariant] train_loop full-vs-micro max param diff="
            << max_diff << std::endl;
  require(max_diff < 5e-5f,
          "train_loop update depends on NSOS_TRAIN_CHUNK_SIZE");
}

void copy_router_parameters(MoERouter& destination, MoERouter& source) {
  auto dst = destination.parameters();
  auto src = source.parameters();
  require(dst.size() == src.size(), "router parameter-count mismatch");
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i]->data.copy_from(src[i]->data);
    dst[i]->zero_grad();
    src[i]->zero_grad();
  }
}

void test_router_input_vjp() {
  MoERouter router(4, 3, 2);
  Tensor x({2, 4}, Device::CPU);
  Tensor upstream({2, 3}, Device::CPU);
  for (int i = 0; i < x.size; ++i) x.data()[i] = -0.3f + 0.11f * i;
  for (int i = 0; i < upstream.size; ++i) {
    upstream.data()[i] = 0.2f - 0.07f * i;
  }

  auto objective = [&]() {
    auto routed = router.forward(x);
    Tensor weights = routed.second.cpu();
    double loss = 0.0;
    for (int i = 0; i < weights.size; ++i) {
      loss += static_cast<double>(weights.data()[i]) * upstream.data()[i];
    }
    return static_cast<float>(loss);
  };

  const float eps = 1e-3f;
  std::vector<float> numerical(static_cast<size_t>(x.size), 0.0f);
  for (int i = 0; i < x.size; ++i) {
    const float original = x.data()[i];
    x.data()[i] = original + eps;
    const float plus = objective();
    x.data()[i] = original - eps;
    const float minus = objective();
    x.data()[i] = original;
    numerical[static_cast<size_t>(i)] = (plus - minus) / (2.0f * eps);
  }

  (void)router.forward(x);
  Tensor analytic = router.accumulate_task_router_grad(upstream).cpu();
  float max_abs = 0.0f;
  for (int i = 0; i < analytic.size; ++i) {
    max_abs = std::max(
        max_abs, std::abs(analytic.data()[i] - numerical[static_cast<size_t>(i)]));
  }
  std::cout << "[invariant] router input VJP max abs err=" << max_abs
            << std::endl;
  require(max_abs < 4e-3f, "router input VJP is incomplete");
}

void test_switch_aux_chunk_invariance() {
  MoERouter split(4, 3, 2);
  MoERouter combined(4, 3, 2);
  copy_router_parameters(combined, split);

  Tensor x1({2, 4}, Device::CPU);
  Tensor x2({2, 4}, Device::CPU);
  Tensor both({4, 4}, Device::CPU);
  for (int i = 0; i < x1.size; ++i) {
    x1.data()[i] = -0.4f + 0.06f * i;
    x2.data()[i] = 0.3f - 0.05f * i;
    both.data()[i] = x1.data()[i];
    both.data()[x1.size + i] = x2.data()[i];
  }

  split.begin_aux_accumulation();
  (void)split.forward(x1);
  (void)split.forward(x2);
  const float split_loss = split.accumulate_switch_aux_grad(0.07f);

  combined.begin_aux_accumulation();
  (void)combined.forward(both);
  const float combined_loss = combined.accumulate_switch_aux_grad(0.07f);

  require(std::abs(split_loss - combined_loss) < 1e-6f,
          "Switch auxiliary loss depends on chunking");
  auto split_params = split.parameters();
  auto combined_params = combined.parameters();
  float max_grad_diff = 0.0f;
  for (size_t p = 0; p < split_params.size(); ++p) {
    Tensor a = split_params[p]->grad.cpu();
    Tensor b = combined_params[p]->grad.cpu();
    require(a.shape == b.shape, "Switch auxiliary gradient-shape mismatch");
    for (int i = 0; i < a.size; ++i) {
      max_grad_diff =
          std::max(max_grad_diff, std::abs(a.data()[i] - b.data()[i]));
    }
  }
  std::cout << "[invariant] Switch aux full-vs-split max grad diff="
            << max_grad_diff << std::endl;
  require(max_grad_diff < 5e-6f,
          "Switch auxiliary gradient depends on chunking");
}

void test_switch_aux_padding_invariance() {
  MoERouter compact(4, 3, 2);
  MoERouter padded(4, 3, 2);
  copy_router_parameters(padded, compact);

  Tensor real({3, 4}, Device::CPU);
  Tensor with_padding({5, 4}, Device::CPU);
  for (int i = 0; i < real.size; ++i) {
    const float value = -0.35f + 0.047f * static_cast<float>(i);
    real.data()[i] = value;
    with_padding.data()[i] = value;
  }
  // Deliberately non-zero padding proves that correctness comes from the
  // explicit validity contract, not from assuming a bias-free zero row.
  for (int i = real.size; i < with_padding.size; ++i) {
    with_padding.data()[i] = 1.5f - 0.09f * static_cast<float>(i);
  }

  compact.begin_aux_accumulation();
  (void)compact.forward(real);
  const std::vector<float> compact_loads = compact.expert_loads;
  const float compact_loss = compact.accumulate_switch_aux_grad(0.07f);

  padded.begin_aux_accumulation();
  (void)padded.forward(with_padding, {1U, 1U, 1U, 0U, 0U});
  const std::vector<float> padded_loads = padded.expert_loads;
  const float padded_loss = padded.accumulate_switch_aux_grad(0.07f);

  require(std::abs(compact_loss - padded_loss) < 1e-6f,
          "Switch auxiliary loss depends on padding rows");
  require(compact_loads.size() == padded_loads.size(),
          "masked router-load size mismatch");
  for (size_t expert = 0; expert < compact_loads.size(); ++expert) {
    require(std::abs(compact_loads[expert] - padded_loads[expert]) < 1e-6f,
            "expert load includes padding rows");
  }
  const auto compact_params = compact.parameters();
  const auto padded_params = padded.parameters();
  require(compact_params.size() == padded_params.size(),
          "padding test router parameter-count mismatch");
  for (size_t parameter = 0; parameter < compact_params.size(); ++parameter) {
    Tensor left = compact_params[parameter]->grad.cpu();
    Tensor right = padded_params[parameter]->grad.cpu();
    require(left.shape == right.shape, "padding test gradient-shape mismatch");
    for (int index = 0; index < left.size; ++index) {
      require(std::abs(left.data()[index] - right.data()[index]) < 5e-6f,
              "Switch auxiliary parameter gradient depends on padding");
    }
  }
  std::cout << "[invariant] Switch aux padding mask loss=" << padded_loss
            << std::endl;
}

void test_weighted_cross_entropy_contract() {
  Tensor logits({2, 3}, Device::CPU);
  const float values[] = {0.8f, -0.2f, 0.3f, -0.4f, 0.6f, 0.1f};
  std::copy_n(values, 6, logits.data());
  const std::vector<int> targets{1, 2};
  const std::vector<float> weights{2.0f, 0.25f};
  auto [loss, analytic] = logits.cross_entropy_weighted(targets, weights);
  const float epsilon = 1e-3f;
  for (int index = 0; index < logits.size; ++index) {
    const float original = logits.data()[index];
    logits.data()[index] = original + epsilon;
    const float plus = logits.cross_entropy_weighted(targets, weights).first;
    logits.data()[index] = original - epsilon;
    const float minus = logits.cross_entropy_weighted(targets, weights).first;
    logits.data()[index] = original;
    const float numerical = (plus - minus) / (2.0f * epsilon);
    require(std::abs(numerical - analytic.data()[index]) < 2e-4f,
            "weighted cross-entropy scalar/gradient contract mismatch");
  }

  Tensor extreme({1, 3}, Device::CPU);
  extreme.data()[0] = 1000.0f;
  extreme.data()[1] = -1000.0f;
  extreme.data()[2] = 0.0f;
  auto [extreme_loss, extreme_grad] =
      extreme.cross_entropy_weighted({1}, {1.0f});
  require(std::isfinite(extreme_loss) && extreme_loss > 1900.0f,
          "extreme cross-entropy loss was capped or became non-finite");
  for (int index = 0; index < extreme_grad.size; ++index) {
    require(std::isfinite(extreme_grad.data()[index]),
            "extreme cross-entropy gradient became non-finite");
  }
  std::cout << "[invariant] weighted CE loss=" << loss
            << " extreme=" << extreme_loss << std::endl;
}

void test_chrass_layerscale_vjp() {
  JambaBlock block(
      8, true, false, false, 0, 1, 2, 1, 1, 1, true, 0.0f, false,
      16, true, 0.35f, 1234u, false, true, true, 64, 2, true, 2, 4, 1);
  Tensor x({1, 2, 8}, Device::CPU);
  Tensor dy({1, 2, 8}, Device::CPU);
  for (int i = 0; i < x.size; ++i) {
    x.data()[i] = -0.2f + 0.025f * i;
    dy.data()[i] = 0.15f - 0.013f * i;
  }

  auto objective = [&]() {
    block.reset();
    Context ctx;
    Tensor y = block.forward(x, &ctx);
    double loss = 0.0;
    for (int i = 0; i < y.size; ++i) loss += y.data()[i] * dy.data()[i];
    return static_cast<float>(loss);
  };

  const float eps = 1e-3f;
  std::vector<float> numerical(static_cast<size_t>(x.size), 0.0f);
  for (int i = 0; i < x.size; ++i) {
    const float original = x.data()[i];
    x.data()[i] = original + eps;
    const float plus = objective();
    x.data()[i] = original - eps;
    const float minus = objective();
    x.data()[i] = original;
    numerical[static_cast<size_t>(i)] = (plus - minus) / (2.0f * eps);
  }

  block.reset();
  Context ctx;
  (void)block.forward(x, &ctx);
  Tensor analytic = block.backward(dy, &ctx).cpu();
  float max_abs = 0.0f;
  for (int i = 0; i < analytic.size; ++i) {
    max_abs = std::max(
        max_abs, std::abs(analytic.data()[i] - numerical[static_cast<size_t>(i)]));
  }
  std::cout << "[invariant] CHRASS+LayerScale VJP max abs err=" << max_abs
            << std::endl;
  require(max_abs < 4e-2f, "CHRASS backward violates LayerScale chain rule");
}

void test_selector_uses_trainer_gradient() {
  Attention attention(8, 2, 32, 1);
  attention.set_sparse_attention(true, 2, 1, 1, 1);
  attention.set_training_mode(true);
  attention.set_batch_valid_lengths({4, 3});
  Tensor x({2, 4, 8}, Device::CPU);
  for (int i = 0; i < x.size; ++i) x.data()[i] = 0.01f * (i - 16);

  Tensor before = attention.sparse_selector_param()->data.clone();
  Context ctx;
  (void)attention.forward(x, &ctx);
  const float loss = attention.accumulate_selector_distill_grad(1.0f);
  Parameter* selector = attention.sparse_selector_param();
  require(std::isfinite(loss), "selector distillation loss is not finite");
  require(selector->grad.size == selector->data.size,
          "selector did not accumulate an optimizer gradient");
  float grad_norm = selector->grad.norm();
  float data_diff = 0.0f;
  for (int i = 0; i < before.size; ++i) {
    data_diff = std::max(
        data_diff, std::abs(before.data()[i] - selector->data.data()[i]));
  }
  require(grad_norm > 0.0f, "selector gradient is zero");
  require(data_diff == 0.0f,
          "selector performed a hidden optimizer update outside Trainer");
  const auto params = attention.parameters();
  require(std::find(params.begin(), params.end(), selector) != params.end(),
          "selector is absent from Attention::parameters");
  std::cout << "[invariant] SSA selector grad norm=" << grad_norm << std::endl;
}

void test_dropout_sequence_contract() {
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 8;
  cfg.vocab_size = 32;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 1;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.5f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;
  determinism::DeterminismManager::instance().set_global_seed(919191);
  JambaModel model(cfg, Device::CPU);
  model.set_training_mode(true);
  Tensor x({4, cfg.d_model}, Device::CPU);
  for (int i = 0; i < x.size; ++i) x.data()[i] = 0.02f * (i - 8);

  model.set_training_rng_sequence(0);
  Tensor first = model.forward(x).cpu();
  Tensor second = model.forward(x).cpu();
  float across_steps = 0.0f;
  for (int i = 0; i < first.size; ++i) {
    across_steps = std::max(across_steps,
                            std::abs(first.data()[i] - second.data()[i]));
  }
  model.set_training_rng_sequence(0);
  Tensor replay = model.forward(x).cpu();
  float replay_error = 0.0f;
  for (int i = 0; i < first.size; ++i) {
    replay_error = std::max(replay_error,
                            std::abs(first.data()[i] - replay.data()[i]));
  }
  require(across_steps > 1e-7f,
          "dropout reused the same mask on consecutive forwards");
  require(replay_error == 0.0f,
          "dropout sequence cannot be replayed deterministically");
  std::cout << "[invariant] dropout step delta=" << across_steps
            << " replay=" << replay_error << std::endl;
}

void test_model_config_is_authoritative() {
  ModelConfig invalid;
  invalid.d_model = 32;
  invalid.n_heads = 3;
  bool rejected = false;
  try {
    JambaModel ignored(invalid, Device::CPU);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "invalid attention heads were silently sanitized");

  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 32;
  cfg.vocab_size = 48;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 1;
  cfg.attention_slot = 0;
  cfg.use_moe = false;
  cfg.mamba_head_dim = 64;
  JambaModel model(cfg, Device::CPU);
  require(model.layers.size() == 1 && model.layers[0]->uses_attention(),
          "faithful schedule silently replaced the configured final attention layer");
  std::cout << "[invariant] strict config preserves final attention layer"
            << std::endl;
}

void test_sliding_window_contract() {
  Attention attention(8, 2, 16, 1, 10000.0f, 1);
  Tensor a({3, 8}, Device::CPU);
  Tensor b({3, 8}, Device::CPU);
  for (int i = 0; i < a.size; ++i) {
    a.data()[i] = 0.01f * static_cast<float>(i - 7);
    b.data()[i] = a.data()[i];
  }
  for (int i = 0; i < 16; ++i) {
    b.data()[i] += 3.0f + 0.1f * static_cast<float>(i);
  }

  Context ctx;
  attention.set_training_mode(true);
  Tensor full_a = attention.forward(a, &ctx).cpu();
  Tensor full_b = attention.forward(b, &ctx).cpu();
  float last_delta = 0.0f;
  for (int d = 0; d < 8; ++d) {
    last_delta = std::max(
        last_delta,
        std::abs(full_a.data()[16 + d] - full_b.data()[16 + d]));
  }
  require(last_delta < 1e-7f,
          "prefill attention reads tokens outside sliding_window");

  (void)attention.forward(a, &ctx);
  Tensor dy({3, 8}, Device::CPU);
  for (int d = 0; d < 8; ++d) dy.data()[16 + d] = 1.0f;
  Tensor dx = attention.backward(dy, &ctx).cpu();
  float stale_grad = 0.0f;
  for (int i = 0; i < 16; ++i) {
    stale_grad = std::max(stale_grad, std::abs(dx.data()[i]));
  }
  require(stale_grad < 1e-7f,
          "attention backward crosses the configured sliding_window");

  attention.set_training_mode(false);
  attention.set_streaming_mode(true);
  auto decode_last = [&](const Tensor& sequence) {
    attention.reset();
    Tensor output;
    for (int token = 0; token < 3; ++token) {
      Tensor row({8}, Device::CPU);
      std::memcpy(row.data(), sequence.data() + token * 8,
                  8 * sizeof(float));
      output = attention.forward(row, &ctx).cpu();
    }
    return output;
  };
  Tensor stream_a = decode_last(a);
  Tensor stream_b = decode_last(b);
  float stream_delta = 0.0f;
  for (int d = 0; d < 8; ++d) {
    stream_delta = std::max(
        stream_delta,
        std::abs(stream_a.data()[d] - stream_b.data()[d]));
  }
  require(stream_delta < 1e-7f,
          "KV-cache decode reads tokens outside sliding_window");
  std::cout << "[invariant] sliding-window prefill=" << last_delta
            << " backward=" << stale_grad
            << " decode=" << stream_delta << std::endl;
}

void test_training_checkpoint_continuation() {
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 16;
  cfg.vocab_size = 40;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.25f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  determinism::DeterminismManager::instance().set_global_seed(20260719);
  JambaModel source(cfg, Device::CPU);
  Trainer source_trainer(&source, 1.5e-3f);
  source_trainer.weight_decay = 0.0f;
  source_trainer.warmup_steps = 2;
  source_trainer.total_training_steps = 20;
  source_trainer.max_grad_norm = 10.0f;
  const std::vector<int> tokens = {1, 3, 5, 7, 9, 11};
  const std::vector<int> targets = {3, 5, 7, 9, 11, 13};
  (void)source_trainer.train_step(tokens, targets);
  auto source_parameters = source.parameters();
  require(!source_parameters.empty(), "resume fixture has no parameters");
  source_trainer.set_lr_scale_by_name(source_parameters.front()->name, 1.25f);
  source_trainer.criticality_lr_scale[source_parameters.front()] = 0.8f;
  source_trainer.crit_g0_state[source_parameters.front()] = 1.125f;

  ScopedTemporaryDirectory directory("nsos_training_resume");
  const auto model_path = directory.path / "model.bin";
  const auto state_path = directory.path / "model.trainer.bin";
  const auto wrong_path = directory.path / "wrong.bin";
  const auto corrupt_path = directory.path / "corrupt.trainer.bin";
  const auto truncated_path =
      directory.path / "truncated.trainer.bin";
  const auto trailing_path =
      directory.path / "trailing.trainer.bin";
  const auto semantic_nan_path =
      directory.path / "semantic_nan.trainer.bin";
  source.save(model_path.string());
  source_trainer.save_training_state(state_path.string(), model_path.string());
  const std::vector<unsigned char> valid_state =
      read_binary_file(state_path);
  constexpr size_t kSha256TrailerBytes =
      sizeof(uint32_t) + sizeof(uint64_t) + 64u;
  require(valid_state.size() > kSha256TrailerBytes + 100u,
          "training-state fixture is unexpectedly small");

  std::vector<unsigned char> corrupt_state = valid_state;
  corrupt_state[100] ^= 0x5au;
  write_binary_file(corrupt_path, corrupt_state);

  std::vector<unsigned char> truncated_state = valid_state;
  truncated_state.resize(truncated_state.size() - 13u);
  write_binary_file(truncated_path, truncated_state);

  std::vector<unsigned char> trailing_state = valid_state;
  trailing_state.push_back(0xa5u);
  write_binary_file(trailing_path, trailing_state);

  // Version 9 begins trainer floats after magic, version, architecture
  // fingerprint, model byte count, model SHA-256 and RNG sequence.
  constexpr size_t kLearningRateOffset =
      sizeof(uint32_t) * 3u + sizeof(uint64_t) + 64u +
      sizeof(uint64_t);
  std::vector<unsigned char> semantic_nan_state = valid_state;
  const uint32_t quiet_nan = 0x7fc00000u;
  std::memcpy(semantic_nan_state.data() + kLearningRateOffset,
              &quiet_nan, sizeof(quiet_nan));
  const size_t semantic_payload_bytes =
      semantic_nan_state.size() - kSha256TrailerBytes;
  const std::string semantic_digest = integrity::sha256_hex(
      semantic_nan_state.data(), semantic_payload_bytes);
  require(semantic_digest.size() == 64u,
          "SHA-256 fixture digest has invalid size");
  std::copy(semantic_digest.begin(), semantic_digest.end(),
            semantic_nan_state.end() - 64);
  write_binary_file(semantic_nan_path, semantic_nan_state);

  JambaModel wrong_model(cfg, Device::CPU);
  wrong_model.save(wrong_path.string());
  Trainer transaction_probe(&wrong_model, 9.0f);
  transaction_probe.global_step_count = 777;
  bool wrong_pair_rejected = false;
  try {
    transaction_probe.load_training_state(state_path.string(),
                                           wrong_path.string());
  } catch (const std::runtime_error&) {
    wrong_pair_rejected = true;
  }
  require(wrong_pair_rejected,
          "training state accepted a different model checkpoint");
  require(transaction_probe.global_step_count == 777,
          "failed training-state load partially mutated Trainer");

  JambaModel failure_model(cfg, Device::CPU);
  failure_model.load(model_path.string(), true);
  Trainer failure_probe(&failure_model, 9.0f);
  failure_probe.global_step_count = 777;
  failure_probe.warmup_steps = 123;
  failure_probe.total_training_steps = 987;
  failure_probe.last_optimizer_step_skipped = true;
  failure_probe.phase_scheduler.semantic_warmup_steps = 41;
  failure_probe.phase_scheduler.qat_start_step = 654;
  failure_model.set_training_rng_sequence(999);
  auto failure_parameters = failure_model.parameters();
  require(!failure_parameters.empty(),
          "failure-transaction fixture has no parameters");
  Parameter* failure_parameter = failure_parameters.front();
  Tensor live_m = Tensor::ones(
      failure_parameter->data.shape.dims, Device::CPU);
  Tensor live_v = Tensor::ones(
      failure_parameter->data.shape.dims, Device::CPU)
                      .mul(2.0f);
  failure_probe.m_state.emplace(failure_parameter, live_m.clone());
  failure_probe.v_state.emplace(failure_parameter, live_v.clone());
  failure_probe.crit_g0_state.emplace(failure_parameter, 4.25f);
  failure_probe.external_lr_scale.emplace(failure_parameter, 1.75f);
  failure_probe.criticality_lr_scale.emplace(failure_parameter, 0.625f);

  const auto require_failure_probe_unchanged =
      [&](const std::string& context) {
        require(failure_probe.learning_rate == 9.0f,
                context + ": learning rate changed");
        require(failure_probe.global_step_count == 777 &&
                    failure_probe.warmup_steps == 123 &&
                    failure_probe.total_training_steps == 987,
                context + ": scalar metadata changed");
        require(failure_probe.last_optimizer_step_skipped,
                context + ": optimizer status changed");
        require(
            failure_probe.phase_scheduler.semantic_warmup_steps == 41 &&
                failure_probe.phase_scheduler.qat_start_step == 654,
            context + ": scheduler changed");
        require(failure_model.training_rng_sequence() == 999,
                context + ": model RNG changed");
        require(failure_probe.m_state.size() == 1u &&
                    failure_probe.v_state.size() == 1u &&
                    failure_probe.quant_state.empty(),
                context + ": optimizer-state cardinality changed");
        require_tensor_bitwise_equal(
            failure_probe.m_state.at(failure_parameter), live_m,
            context + ": first moment");
        require_tensor_bitwise_equal(
            failure_probe.v_state.at(failure_parameter), live_v,
            context + ": second moment");
        require(failure_probe.crit_g0_state.size() == 1u &&
                    failure_probe.crit_g0_state.at(failure_parameter) ==
                        4.25f &&
                    failure_probe.external_lr_scale.size() == 1u &&
                    failure_probe.external_lr_scale.at(failure_parameter) ==
                        1.75f &&
                    failure_probe.criticality_lr_scale.size() == 1u &&
                    failure_probe.criticality_lr_scale.at(
                        failure_parameter) == 0.625f,
                context + ": per-parameter controller state changed");
      };

  const auto expect_transactional_rejection =
      [&](const std::filesystem::path& path,
          const std::string& context) {
        bool rejected = false;
        try {
          failure_probe.load_training_state(
              path.string(), model_path.string());
        } catch (const std::exception&) {
          rejected = true;
        }
        require(rejected, context + ": invalid state was accepted");
        require_failure_probe_unchanged(context);
      };
  expect_transactional_rejection(
      corrupt_path, "checksum corruption");
  expect_transactional_rejection(
      truncated_path, "truncated state");
  expect_transactional_rejection(
      trailing_path, "trailing state");
  expect_transactional_rejection(
      semantic_nan_path, "semantic NaN");

#ifdef NSOS_ENABLE_TEST_HOOKS
  {
    TrainingStateFaultReset reset_faults;
    testing::set_training_state_stage_failure_countdown(2);
    bool oom_rejected = false;
    try {
      failure_probe.load_training_state(
          state_path.string(), model_path.string());
    } catch (const std::bad_alloc&) {
      oom_rejected = true;
    }
    require(oom_rejected,
            "injected mid-stage OOM did not abort training-state load");
    require_failure_probe_unchanged("mid-stage OOM");
  }
  {
    TrainingStateFaultReset reset_faults;
    const int saved_total_steps =
        source_trainer.total_training_steps;
    source_trainer.total_training_steps =
        saved_total_steps + 1;
    testing::set_training_state_save_failure_before_replace(true);
    bool interruption_rejected = false;
    try {
      source_trainer.save_training_state(
          state_path.string(), model_path.string());
    } catch (const std::runtime_error&) {
      interruption_rejected = true;
    }
    source_trainer.total_training_steps =
        saved_total_steps;
    require(interruption_rejected,
            "injected save interruption did not abort");
    require(read_binary_file(state_path) == valid_state,
            "interrupted save replaced the last valid sidecar");
    const std::string temporary_prefix =
        state_path.filename().string() + ".tmp.";
    for (const auto& entry :
         std::filesystem::directory_iterator(directory.path)) {
      require(
          entry.path().filename().string().rfind(
              temporary_prefix, 0) != 0,
          "interrupted save leaked a temporary sidecar");
    }
  }
#endif

  JambaModel resumed(cfg, Device::CPU);
  resumed.load(model_path.string(), true);
  Trainer resumed_trainer(&resumed, 9.0f);
  resumed_trainer.load_training_state(state_path.string(), model_path.string());
  require(resumed_trainer.global_step_count ==
              source_trainer.global_step_count,
          "global step was not restored");
  require(resumed.training_rng_sequence() == source.training_rng_sequence(),
          "dropout RNG sequence was not restored");
  require(resumed_trainer.m_state.size() == source_trainer.m_state.size() &&
              resumed_trainer.v_state.size() == source_trainer.v_state.size(),
          "Adam state cardinality was not restored");
  require_model_parameters_bitwise_equal(
      source, resumed, "CPU checkpoint round trip");
  require_optimizer_states_by_registry_bitwise_equal(
      source_trainer, source, resumed_trainer, resumed,
      "CPU checkpoint round trip");
  auto resumed_parameters = resumed.parameters();
  require(!resumed_parameters.empty(), "resumed fixture has no parameters");
  require(resumed_trainer.lr_scale_for(resumed_parameters.front()) ==
              source_trainer.lr_scale_for(source_parameters.front()),
          "composed per-parameter LR scales were not restored");
  require(resumed_trainer.crit_g0_state.size() ==
              source_trainer.crit_g0_state.size(),
          "criticality baselines were not restored");

  bool invalid_scale_rejected = false;
  try {
    resumed_trainer.set_lr_scale_by_name(resumed_parameters.front()->name,
                                         -1.0f);
  } catch (const std::invalid_argument&) {
    invalid_scale_rejected = true;
  }
  require(invalid_scale_rejected,
          "non-positive per-parameter LR scale was accepted");

  const float source_loss = source_trainer.train_step(tokens, targets);
  const float resumed_loss = resumed_trainer.train_step(tokens, targets);
  const float continuation_diff = max_parameter_difference(source, resumed);
  require(std::memcmp(&source_loss, &resumed_loss, sizeof(float)) == 0,
          "resumed objective is not bitwise identical");
  require_model_parameters_bitwise_equal(
      source, resumed, "CPU checkpoint continuation");
  require_optimizer_states_by_registry_bitwise_equal(
      source_trainer, source, resumed_trainer, resumed,
      "CPU checkpoint continuation");
  std::cout << "[invariant] training resume loss="
            << std::abs(source_loss - resumed_loss)
            << " params=" << continuation_diff << std::endl;
}

void test_nonfinite_optimizer_rejection_and_recovery() {
#ifdef NSOS_ENABLE_TEST_HOOKS
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 16;
  cfg.vocab_size = 40;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  determinism::DeterminismManager::instance().set_global_seed(20260728);
  JambaModel model(cfg, Device::CPU);
  Trainer trainer(&model, 1.0e-3f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 10.0f;
  const std::vector<int> tokens = {1, 3, 5, 7, 9, 11};
  const std::vector<int> targets = {3, 5, 7, 9, 11, 13};

  const float warmup_loss = trainer.train_step(tokens, targets);
  require(std::isfinite(warmup_loss),
          "NaN recovery fixture warmup was non-finite");
  const int step_before_fault = trainer.global_step_count;
  const auto weights_before_fault =
      clone_parameter_values(model);
  const auto m_before_fault =
      clone_tensor_state_map(trainer.m_state);
  const auto v_before_fault =
      clone_tensor_state_map(trainer.v_state);

  TrainingStateFaultReset reset_faults;
  testing::inject_training_nan_before_optimizer();
  bool rejected = false;
  try {
    (void)trainer.train_step(tokens, targets);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected,
          "injected optimizer NaN did not reject the training step");
  require(trainer.last_optimizer_step_skipped,
          "NaN rejection did not report a skipped optimizer step");
  require(trainer.global_step_count == step_before_fault,
          "NaN rejection advanced the optimizer step");
  require_parameter_values_bitwise_equal(
      model, weights_before_fault, "NaN rejection");
  require_tensor_state_map_bitwise_equal(
      trainer.m_state, m_before_fault, "NaN rejection m");
  require_tensor_state_map_bitwise_equal(
      trainer.v_state, v_before_fault, "NaN rejection v");
  for (const Parameter* parameter : model.parameters()) {
    if (!parameter || !parameter->trainable ||
        parameter->grad.size == 0) {
      continue;
    }
    const Tensor gradient = parameter->grad.cpu();
    for (int index = 0; index < gradient.size; ++index) {
      require(gradient.data()[index] == 0.0f,
              "NaN rejection retained a poisoned gradient");
    }
  }

  const float recovered_loss = trainer.train_step(tokens, targets);
  require(std::isfinite(recovered_loss),
          "training did not recover after a rejected NaN step");
  require(!trainer.last_optimizer_step_skipped,
          "successful recovery step remained marked skipped");
  require(trainer.global_step_count == step_before_fault + 1,
          "recovery did not apply exactly one optimizer step");

  // Adam's second moment is a non-negative quantity. A finite but corrupt
  // negative value must be rejected by the same pre-commit gate; discovering
  // it only after updating a prefix would unnecessarily poison the trainer.
  Parameter* negative_v_parameter = nullptr;
  for (auto& entry : trainer.v_state) {
    if (entry.first && entry.second.size > 0 &&
        entry.second.get_device() == Device::CPU) {
      negative_v_parameter = entry.first;
      break;
    }
  }
  require(negative_v_parameter != nullptr,
          "negative-second-moment fixture has no CPU Adam state");
  Tensor& negative_v = trainer.v_state.at(negative_v_parameter);
  const float valid_v0 = negative_v.data()[0];
  negative_v.data()[0] = -1.0f;
  const auto weights_before_negative_v =
      clone_parameter_values(model);
  const auto m_before_negative_v =
      clone_tensor_state_map(trainer.m_state);
  const auto v_before_negative_v =
      clone_tensor_state_map(trainer.v_state);
  const int step_before_negative_v = trainer.global_step_count;
  bool negative_v_rejected = false;
  try {
    (void)trainer.train_step(tokens, targets);
  } catch (const std::runtime_error&) {
    negative_v_rejected = true;
  }
  require(negative_v_rejected,
          "negative Adam second moment passed the pre-commit gate");
  require(!trainer.optimizer_state_poisoned(),
          "pre-commit second-moment rejection poisoned the trainer");
  require(trainer.global_step_count == step_before_negative_v,
          "negative second moment advanced the optimizer step");
  require_parameter_values_bitwise_equal(
      model, weights_before_negative_v,
      "negative second-moment rejection");
  require_tensor_state_map_bitwise_equal(
      trainer.m_state, m_before_negative_v,
      "negative second-moment rejection m");
  require_tensor_state_map_bitwise_equal(
      trainer.v_state, v_before_negative_v,
      "negative second-moment rejection v");
  negative_v.data()[0] = valid_v0;

  std::cout << "[invariant] NaN rejected and recovered at step="
            << trainer.global_step_count << std::endl;
#endif
}

void test_optimizer_fail_stop_and_checkpoint_recovery() {
#ifdef NSOS_ENABLE_TEST_HOOKS
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 16;
  cfg.vocab_size = 40;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  determinism::DeterminismManager::instance().set_global_seed(
      20260729);
  JambaModel model(cfg, Device::CPU);
  Trainer trainer(&model, 1.0e-3f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 10.0f;
  const std::vector<int> tokens = {1, 3, 5, 7, 9, 11};
  const std::vector<int> targets = {3, 5, 7, 9, 11, 13};
  (void)trainer.train_step(tokens, targets);

  ScopedTemporaryDirectory directory(
      "nsos_optimizer_fail_stop");
  const auto model_path = directory.path / "model.bin";
  const auto state_path =
      directory.path / "model.trainer.bin";
  model.save(model_path.string());
  trainer.save_training_state(
      state_path.string(), model_path.string());
  const auto durable_sidecar =
      read_binary_file(state_path);
  const int durable_step = trainer.global_step_count;

  TrainingStateFaultReset reset_faults;
  testing::set_optimizer_commit_failure_countdown(0);
  bool interruption_rejected = false;
  try {
    (void)trainer.train_step(tokens, targets);
  } catch (const std::runtime_error&) {
    interruption_rejected = true;
  }
  require(interruption_rejected,
          "injected mid-commit optimizer failure was accepted");
  require(trainer.optimizer_state_poisoned(),
          "ambiguous optimizer failure did not poison Trainer");
  require(trainer.global_step_count == durable_step,
          "partial optimizer cohort published a completed step");

  const auto ambiguous_weights =
      clone_parameter_values(model);
  bool continuation_rejected = false;
  try {
    (void)trainer.train_step(tokens, targets);
  } catch (const std::runtime_error&) {
    continuation_rejected = true;
  }
  require(continuation_rejected,
          "poisoned Trainer allowed another training attempt");
  require_parameter_values_bitwise_equal(
      model, ambiguous_weights,
      "poisoned continuation rejection");

  bool sidecar_export_rejected = false;
  try {
    trainer.save_training_state(
        state_path.string(), model_path.string());
  } catch (const std::runtime_error&) {
    sidecar_export_rejected = true;
  }
  require(sidecar_export_rejected,
          "poisoned Trainer exported an ambiguous sidecar");
  require(read_binary_file(state_path) == durable_sidecar,
          "failed poisoned export replaced the durable sidecar");

  bool sidecar_only_recovery_rejected = false;
  try {
    trainer.load_training_state(
        state_path.string(), model_path.string());
  } catch (const std::runtime_error&) {
    sidecar_only_recovery_rejected = true;
  }
  require(sidecar_only_recovery_rejected,
          "sidecar load cleared poison without restoring model weights");
  require(trainer.optimizer_state_poisoned(),
          "failed recovery cleared optimizer poison");

  model.load(model_path.string(), true);
  trainer.load_training_state(
      state_path.string(), model_path.string());
  require(!trainer.optimizer_state_poisoned(),
          "validated model+sidecar recovery retained poison");
  require(trainer.global_step_count == durable_step,
          "validated recovery restored the wrong step");
  const float recovered =
      trainer.train_step(tokens, targets);
  require(std::isfinite(recovered),
          "training did not resume after validated fail-stop recovery");
  require(trainer.global_step_count == durable_step + 1,
          "recovery did not commit exactly one optimizer step");
  std::cout << "[invariant] optimizer fail-stop recovered at step="
            << trainer.global_step_count << std::endl;
#endif
}

void test_training_cancellation_contract() {
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 16;
  cfg.vocab_size = 40;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  determinism::DeterminismManager::instance().set_global_seed(
      20260729);
  JambaModel model(cfg, Device::CPU);
  Trainer trainer(&model, 1.0e-3f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 10.0f;
  const std::vector<int> step_tokens =
      {1, 3, 5, 7, 9, 11};
  const std::vector<int> step_targets =
      {3, 5, 7, 9, 11, 13};

  const auto before_cancel =
      clone_parameter_values(model);
  trainer.request_cancellation();
  bool entry_cancelled = false;
  try {
    (void)trainer.train_step(step_tokens, step_targets);
  } catch (const AbortException&) {
    entry_cancelled = true;
  }
  require(entry_cancelled,
          "pre-requested training cancellation was ignored");
  require(trainer.cancellation_requested(),
          "cancellation request was cleared implicitly");
  require(trainer.global_step_count == 0,
          "cancelled entry advanced the optimizer");
  require(trainer.m_state.empty() && trainer.v_state.empty(),
          "cancelled entry allocated optimizer state");
  require_parameter_values_bitwise_equal(
      model, before_cancel, "pre-requested cancellation");

  trainer.clear_cancellation();
  int completed_callbacks = 0;
  const std::vector<int> loop_tokens =
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  bool boundary_cancelled = false;
  try {
    trainer.train_loop(
        loop_tokens, 2, 1, 3,
        [&](int, float) {
          ++completed_callbacks;
          trainer.request_cancellation();
        },
        3, 0);
  } catch (const AbortException&) {
    boundary_cancelled = true;
  }
  require(boundary_cancelled,
          "train_loop did not stop after a callback cancellation");
  require(completed_callbacks == 1,
          "train_loop crossed more than one committed safe point");
  require(trainer.global_step_count == 1,
          "safe-point cancellation committed an unexpected step count");

  trainer.clear_cancellation();
  const float recovered =
      trainer.train_step(step_tokens, step_targets);
  require(std::isfinite(recovered),
          "training did not recover after cancellation");
  require(trainer.global_step_count == 2,
          "post-cancellation recovery did not commit exactly one step");
  std::cout << "[invariant] cancellation stopped at safe point and "
               "recovered"
            << std::endl;
}

void test_criticality_regularizer_contract() {
  set_test_env("NSOS_CRIT_REG", "1");
  set_test_env("NSOS_CRIT_REG_EVERY", "1");
  set_test_env("NSOS_CRIT_REG_ETA", "0.2");

  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 16;
  cfg.vocab_size = 32;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  JambaModel model(cfg, Device::CPU);
  Trainer trainer(&model, 1e-3f);
  trainer.phase_scheduler.semantic_warmup_steps = 0;
  trainer.phase_scheduler.qat_start_step = 0;
  trainer.phase_scheduler.ternary_regularization = 0.0f;
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 1000.0f;
  const std::vector<int> tokens = {1, 2, 3, 4, 5};
  const std::vector<int> targets = {2, 3, 4, 5, 6};

  // The authoritative schedule guarantees at least one semantic warmup step.
  (void)trainer.train_step(tokens, targets);
  (void)trainer.train_step(tokens, targets);
  require(trainer.last_objective_stats.criticality_regularization == 0.0f,
          "criticality baseline capture added a spurious objective");

  std::vector<BitLinear*> ternary_layers;
  for (BitLinear* layer : model.collect_bitlinear_layers()) {
    if (layer && !layer->quantization_sensitive() &&
        !layer->reference_path_enabled() && layer->has_full_precision_weight() &&
        layer->weight.grad.size == layer->weight.data.size) {
      ternary_layers.push_back(layer);
    }
  }
  require(!ternary_layers.empty(), "criticality fixture has no ternary layers");
  require(trainer.crit_g0_state.size() == ternary_layers.size(),
          "criticality included non-BitLinear or sensitive parameters");
  Parameter& weight = ternary_layers.front()->weight;
  for (int i = 0; i < weight.data.size; ++i) weight.data.data()[i] *= 1.5f;
  weight.mark_updated();

  const float total = trainer.train_step(tokens, targets);
  const float regularizer =
      trainer.last_objective_stats.criticality_regularization;
  require(std::isfinite(regularizer) && regularizer > 0.0f,
          "criticality radial objective did not detect gain drift");
  require(std::abs(total - trainer.last_objective_stats.total) < 1e-7f,
          "criticality objective was not included in reported total");
  require(trainer.m_state.find(&weight) != trainer.m_state.end(),
          "criticality weight was not updated through Adam state");
  std::cout << "[invariant] criticality radial loss=" << regularizer
            << " eligible=" << ternary_layers.size() << std::endl;

  set_test_env("NSOS_CRIT_REG", "0");
}

void test_post_commit_callback_failure_poisoning() {
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 16;
  cfg.vocab_size = 40;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 64;
  cfg.use_moe = false;
  cfg.use_ttt = false;
  cfg.use_chrass = false;
  cfg.dropout = 0.0f;
  cfg.mamba2_faithful = false;
  cfg.tie_word_embeddings = false;

  JambaModel model(cfg, Device::CPU);
  Trainer trainer(&model, 1.0e-3f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 10.0f;
  const std::vector<int> tokens =
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  bool callback_failure_observed = false;
  try {
    trainer.train_loop(
        tokens, 1, 1, 3,
        [](int, float) {
          throw std::runtime_error("injected post-commit callback failure");
        },
        1, 0);
  } catch (const std::runtime_error&) {
    callback_failure_observed = true;
  }
  require(callback_failure_observed,
          "post-commit callback failure did not escape train_loop");
  require(trainer.global_step_count == 1,
          "post-commit callback fixture did not publish exactly one step");
  require(trainer.optimizer_state_poisoned(),
          "post-commit callback failure did not poison retry semantics");

  bool retry_rejected = false;
  try {
    (void)trainer.train_step({1, 2, 3, 4}, {2, 3, 4, 5});
  } catch (const OptimizerStatePoisonedException&) {
    retry_rejected = true;
  }
  require(retry_rejected,
          "post-commit callback failure allowed an ambiguous retry");
  std::cout << "[invariant] post-commit callback failure poisoned retry"
            << std::endl;
}

}  // namespace

int main() {
  try {
    sparse_gradient_test::parameter_contract(Device::CPU);
    sparse_gradient_test::routed_optimizer_contract(Device::CPU);
    sparse_gradient_test::routed_optimizer_contract(Device::CPU, 4);
    test_model_backward_one_shot_contract();
    test_train_loop_chunk_invariance();
    test_router_input_vjp();
    test_switch_aux_chunk_invariance();
    test_switch_aux_padding_invariance();
    test_weighted_cross_entropy_contract();
    test_chrass_layerscale_vjp();
    test_selector_uses_trainer_gradient();
    test_dropout_sequence_contract();
    test_model_config_is_authoritative();
    test_sliding_window_contract();
    test_training_checkpoint_continuation();
    test_nonfinite_optimizer_rejection_and_recovery();
    test_optimizer_fail_stop_and_checkpoint_recovery();
    test_training_cancellation_contract();
    test_post_commit_callback_failure_poisoning();
    test_criticality_regularizer_contract();
    std::cout << "All training invariants passed!" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Training invariant failed: " << error.what() << std::endl;
    return 1;
  }
}
