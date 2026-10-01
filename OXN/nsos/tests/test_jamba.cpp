#include "../include/jamba.h"
#include "../include/layer_audit.h"
#include "../include/trainer.h"
#include "../include/nsos/determinism.h"
#include "tensor.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#undef assert
[[noreturn]] static void nsos_test_check_failed(const char* expression,
                                                const char* file, int line) {
  throw std::runtime_error(std::string("check failed: ") + expression +
                           " at " + file + ":" + std::to_string(line));
}
#define assert(expression) \
  ((expression) ? static_cast<void>(0) \
                : nsos_test_check_failed(#expression, __FILE__, __LINE__))

using namespace nsos;

namespace {

void assert_tensor_close(const Tensor& lhs, const Tensor& rhs, float tolerance = 1e-3f) {
  Tensor lhs_cpu = lhs.cpu();
  Tensor rhs_cpu = rhs.cpu();
  assert(lhs_cpu.size == rhs_cpu.size);
  for (int i = 0; i < lhs_cpu.size; ++i) {
    const float diff = std::abs(lhs_cpu.data()[i] - rhs_cpu.data()[i]);
    assert(diff < tolerance);
  }
}

Tensor concat_rows(const Tensor& a, const Tensor& b) {
  Tensor a_cpu = a.cpu();
  Tensor b_cpu = b.cpu();
  const int cols = a.shape[1];
  assert(b.shape[1] == cols);
  Tensor out({a.shape[0] + b.shape[0], cols}, Device::CPU);
  std::memcpy(out.data(), a_cpu.data(), static_cast<size_t>(a.size) * sizeof(float));
  std::memcpy(out.data() + a.size, b_cpu.data(), static_cast<size_t>(b.size) * sizeof(float));
  return out;
}

Tensor make_padded_batch(const std::vector<Tensor>& sequences) {
  assert(!sequences.empty());
  const int cols = sequences.front().shape[1];
  int max_rows = 0;
  for (const auto& sequence : sequences) {
    assert(sequence.shape.size() == 2);
    assert(sequence.shape[1] == cols);
    max_rows = std::max(max_rows, sequence.shape[0]);
  }

  Tensor out({static_cast<int>(sequences.size()), max_rows, cols}, Device::CPU);
  std::fill_n(out.data(), out.size, 0.0f);
  for (int batch = 0; batch < static_cast<int>(sequences.size()); ++batch) {
    Tensor seq_cpu = sequences[static_cast<size_t>(batch)].cpu();
    for (int row = 0; row < seq_cpu.shape[0]; ++row) {
      const size_t dst_offset =
          ((static_cast<size_t>(batch) * max_rows) + row) * static_cast<size_t>(cols);
      const size_t src_offset = static_cast<size_t>(row) * static_cast<size_t>(cols);
      std::memcpy(out.data() + dst_offset,
                  seq_cpu.data() + src_offset,
                  static_cast<size_t>(cols) * sizeof(float));
    }
  }
  return out;
}

Parameter* parameter_named(JambaModel& model,
                           const std::string& name) {
  for (Parameter* parameter : model.parameters()) {
    if (parameter && parameter->name == name) {
      return parameter;
    }
  }
  return nullptr;
}

void fill_parameter(Parameter* parameter, float value) {
  assert(parameter != nullptr);
  assert(parameter->data.get_device() == Device::CPU);
  std::fill_n(parameter->data.data(), parameter->data.size,
              value);
  parameter->mark_updated();
}

ModelConfig edge_pack_test_config() {
  ModelConfig config;
  config.num_layers = 1;
  config.d_model = 12;  // Deliberately not divisible by the 16-code word size.
  config.vocab_size = 31;
  config.n_heads = 3;
  config.n_kv_heads = 1;
  config.attention_period = 64;
  config.attention_slot = 63;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_chrass = false;
  config.use_kan = false;
  config.mamba2_faithful = false;
  config.tie_word_embeddings = false;
  config.dropout = 0.0f;
  return config;
}

struct ScopedTestDirectory {
  std::filesystem::path path;

  ScopedTestDirectory() {
    const auto ordinal =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("nsos_edge_pack_test_" + std::to_string(ordinal));
    std::filesystem::create_directories(path);
  }

  ~ScopedTestDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

std::vector<unsigned char> read_binary_file(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  assert(input.is_open());
  return std::vector<unsigned char>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

void write_binary_file(const std::filesystem::path& path,
                       const std::vector<unsigned char>& bytes) {
  std::ofstream output(
      path, std::ios::binary | std::ios::trunc);
  assert(output.is_open());
  if (!bytes.empty()) {
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
  }
  output.close();
  assert(output.good());
}

void assert_packed_state_equal(const BitLinearPackedState& lhs,
                               const BitLinearPackedState& rhs) {
  assert(lhs.in_features == rhs.in_features);
  assert(lhs.out_features == rhs.out_features);
  assert(lhs.use_bias == rhs.use_bias);
  assert(lhs.weight_scale == rhs.weight_scale);
  assert(lhs.packed_weights == rhs.packed_weights);
  assert(lhs.magnitude == rhs.magnitude);
  assert(lhs.bias == rhs.bias);
  assert(lhs.flat_alpha == rhs.flat_alpha);
  assert(lhs.flat_beta == rhs.flat_beta);
}

void assert_tensor_bitwise_equal(const Tensor& lhs,
                                 const Tensor& rhs) {
  assert(lhs.shape == rhs.shape);
  assert(lhs.get_device() == rhs.get_device());
  if (lhs.size == 0) {
    return;
  }
  const Tensor lhs_cpu = lhs.cpu();
  const Tensor rhs_cpu = rhs.cpu();
  assert(std::memcmp(
             lhs_cpu.data(), rhs_cpu.data(),
             static_cast<size_t>(lhs_cpu.size) * sizeof(float)) == 0);
}

struct EdgeLayerSnapshot {
  BitLinearPackedState packed;
  Tensor weight;
  Tensor weight_grad;
  uint64_t weight_version = 0;
  bool reference_path = false;
};

std::vector<EdgeLayerSnapshot> snapshot_edge_layers(
    JambaModel& model) {
  std::vector<EdgeLayerSnapshot> snapshots;
  for (BitLinear* layer : model.collect_bitlinear_layers()) {
    assert(layer != nullptr);
    snapshots.push_back({
        layer->export_packed_state(),
        layer->weight.data.clone(),
        layer->weight.grad.clone(),
        layer->weight.version,
        layer->reference_path_enabled()});
  }
  return snapshots;
}

void assert_edge_layers_unchanged(
    JambaModel& model,
    const std::vector<EdgeLayerSnapshot>& expected) {
  const auto layers = model.collect_bitlinear_layers();
  assert(layers.size() == expected.size());
  for (size_t index = 0; index < layers.size(); ++index) {
    assert_packed_state_equal(
        layers[index]->export_packed_state(),
        expected[index].packed);
    assert_tensor_bitwise_equal(
        layers[index]->weight.data, expected[index].weight);
    assert_tensor_bitwise_equal(
        layers[index]->weight.grad, expected[index].weight_grad);
    assert(layers[index]->weight.version ==
           expected[index].weight_version);
    assert(layers[index]->reference_path_enabled() ==
           expected[index].reference_path);
  }
}

void expect_edge_load_rejected_without_mutation(
    JambaModel& model,
    const std::filesystem::path& path) {
  const auto before = snapshot_edge_layers(model);
  bool rejected = false;
  try {
    model.load_edge_linear_pack(path.string(), true);
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
  assert_edge_layers_unchanged(model, before);
}

} // namespace

void test_jamba_structure() {
  int layers = 16;
  int d_model = 8;
  int vocab_size = 32;

  ModelConfig structure_config;
  structure_config.num_layers = layers;
  structure_config.d_model = d_model;
  structure_config.vocab_size = vocab_size;
  structure_config.n_heads = 2;
  structure_config.n_kv_heads = 1;
  structure_config.use_moe = true;
  structure_config.mamba_expand = 2;
  structure_config.mamba_head_dim = 4;
  structure_config.mamba_n_groups = 1;
  JambaModel model(structure_config, Device::CPU);

  // Stable schedule: attention only appears on every 8th layer.
  assert(model.layers[7]->uses_attention() == true);
  assert(model.layers[0]->uses_attention() == false);

  // Deep blocks should expose MoE as an optional capability.
  assert(model.layers[5]->uses_moe() == true);

  Tensor x = Tensor::random({4, d_model}); // Sequence length 4
  Tensor y = model.forward(x);

  assert(y.shape[0] == 4);
  assert(y.shape[1] == vocab_size);
  std::cout << "Jamba Architecture test passed!" << std::endl;
}

void test_faithful_last_attention_guard() {
  ModelConfig config;
  config.num_layers = 4;
  config.d_model = 32;
  config.vocab_size = 64;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 2;
  config.attention_slot = 1;
  config.use_moe = false;
  config.mamba2_faithful = true;
  config.mamba_expand = 2;
  config.mamba_head_dim = 16;
  config.mamba_n_groups = 1;
  config.force_mamba_last_layer = true;

  JambaModel guarded(config, Device::CPU);
  assert(guarded.layers.size() == 4);
  assert(guarded.layers[1]->uses_attention());
  assert(!guarded.layers[3]->uses_attention());

  config.force_mamba_last_layer = false;
  JambaModel override_disabled(config, Device::CPU);
  assert(override_disabled.layers[1]->uses_attention());
  assert(override_disabled.layers[3]->uses_attention());

  config.force_mamba_last_layer = true;
  config.mamba2_faithful = false;
  JambaModel non_faithful(config, Device::CPU);
  assert(non_faithful.layers[1]->uses_attention());
  assert(non_faithful.layers[3]->uses_attention());

  std::cout << "Faithful last-layer Mamba guard test passed!" << std::endl;
}

void test_parallel_hybrid_registry_and_zero_gate_reduction() {
  ModelConfig hybrid_config;
  hybrid_config.num_layers = 1;
  hybrid_config.d_model = 16;
  hybrid_config.vocab_size = 48;
  hybrid_config.n_heads = 2;
  hybrid_config.n_kv_heads = 1;
  hybrid_config.attention_period = 1;
  hybrid_config.attention_slot = 0;
  hybrid_config.force_mamba_last_layer = false;
  hybrid_config.hybrid_composition =
      HybridComposition::ParallelGated;
  hybrid_config.faithful_attention_linears = true;
  hybrid_config.use_moe = false;
  hybrid_config.use_ttt = false;
  hybrid_config.use_chrass = false;
  hybrid_config.use_kan = false;
  hybrid_config.dropout = 0.0f;
  hybrid_config.mamba_expand = 2;
  hybrid_config.mamba_head_dim = 16;
  hybrid_config.mamba_n_groups = 1;

  ModelConfig mamba_config = hybrid_config;
  mamba_config.attention_period = 64;
  mamba_config.attention_slot = 63;

  JambaModel hybrid(hybrid_config, Device::CPU);
  JambaModel mamba(mamba_config, Device::CPU);
  assert(hybrid.layers[0]->audit_block_type() ==
         "mamba2+attention+ffn");

  std::unordered_set<std::string> names;
  for (Parameter* parameter : hybrid.parameters()) {
    assert(parameter != nullptr);
    assert(!parameter->name.empty());
    assert(names.insert(parameter->name).second);
  }
  assert(parameter_named(
             hybrid,
             "layers.0.mamba.pre_norm.weight") != nullptr);
  assert(parameter_named(
             hybrid,
             "layers.0.mamba.norm.weight") != nullptr);
  assert(parameter_named(
             hybrid,
             "layers.0.attn.pre_norm.weight") != nullptr);
  assert(parameter_named(
             hybrid,
             "layers.0.ffn.pre_norm.weight") != nullptr);
  assert(parameter_named(hybrid,
                         "layers.0.mamba.gate") != nullptr);
  assert(parameter_named(hybrid,
                         "layers.0.attn.gate") != nullptr);
  assert(parameter_named(hybrid,
                         "layers.0.ffn.gate") != nullptr);
  assert(parameter_named(
             hybrid,
             "layers.0.mamba.x_proj.weight") != nullptr);
  assert(parameter_named(
             hybrid,
             "layers.0.attn.q_down_proj.weight") != nullptr);

  std::unordered_map<std::string, Parameter*> hybrid_by_name;
  for (Parameter* parameter : hybrid.parameters()) {
    hybrid_by_name.emplace(parameter->name, parameter);
  }
  for (Parameter* source : mamba.parameters()) {
    auto destination = hybrid_by_name.find(source->name);
    if (destination != hybrid_by_name.end() &&
        destination->second->data.shape == source->data.shape) {
      destination->second->data.copy_from(source->data);
      destination->second->mark_updated();
    }
  }
  Parameter* pure_norm =
      parameter_named(mamba, "layers.0.norm.weight");
  Parameter* hybrid_mamba_norm = parameter_named(
      hybrid, "layers.0.mamba.pre_norm.weight");
  assert(pure_norm != nullptr && hybrid_mamba_norm != nullptr);
  hybrid_mamba_norm->data.copy_from(pure_norm->data);
  hybrid_mamba_norm->mark_updated();
  fill_parameter(
      parameter_named(hybrid, "layers.0.mamba.gate"),
      1.0f);
  fill_parameter(
      parameter_named(hybrid, "layers.0.attn.gate"),
      0.0f);
  fill_parameter(
      parameter_named(hybrid, "layers.0.ffn.gate"),
      0.0f);

  Tensor input = Tensor::random({5, hybrid_config.d_model},
                                Device::CPU);
  Context hybrid_context;
  Context mamba_context;
  Tensor hybrid_logits = hybrid.forward(input, &hybrid_context);
  Tensor mamba_logits = mamba.forward(input, &mamba_context);
  assert_tensor_close(hybrid_logits, mamba_logits, 1e-5f);
  std::cout
      << "Parallel hybrid registry and zero-gate reduction test passed!"
      << std::endl;
}

void test_parallel_hybrid_gradients_and_interaction_audit() {
  ModelConfig config;
  config.num_layers = 1;
  config.d_model = 16;
  config.vocab_size = 48;
  config.n_heads = 2;
  config.n_kv_heads = 1;
  config.attention_period = 1;
  config.attention_slot = 0;
  config.force_mamba_last_layer = false;
  config.hybrid_composition =
      HybridComposition::ParallelGated;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_chrass = false;
  config.use_kan = false;
  config.dropout = 0.0f;
  config.mamba_expand = 2;
  config.mamba_head_dim = 16;

  JambaModel model(config, Device::CPU);
  LayerAuditCollector audit;
  audit.begin_run("parallel_hybrid_gradients");
  audit.set_phase("train");
  audit.set_enabled(true);
  model.set_audit_collector(&audit);

  Tensor input =
      Tensor::random({4, config.d_model}, Device::CPU);
  Context context;
  Tensor logits = model.forward(input, &context);
  Tensor grad(logits.shape.dims, Device::CPU);
  for (int index = 0; index < grad.size; ++index) {
    grad.data()[index] =
        0.001f * static_cast<float>((index % 17) - 8);
  }
  model.backward(grad, context);

  for (const std::string& name :
       {"layers.0.mamba.gate",
        "layers.0.attn.gate",
        "layers.0.ffn.gate",
        "layers.0.mamba.pre_norm.weight",
        "layers.0.attn.pre_norm.weight",
        "layers.0.ffn.pre_norm.weight"}) {
    Parameter* parameter = parameter_named(model, name);
    assert(parameter != nullptr);
    assert(parameter->grad.size == parameter->data.size);
    assert(std::isfinite(parameter->grad.norm()));
    assert(parameter->grad.norm() > 0.0f);
  }
  Parameter* mamba_weight = parameter_named(
      model, "layers.0.mamba.x_proj.weight");
  Parameter* attention_weight = parameter_named(
      model, "layers.0.attn.q_down_proj.weight");
  assert(mamba_weight != nullptr &&
         mamba_weight->grad.size == mamba_weight->data.size);
  assert(attention_weight != nullptr &&
         attention_weight->grad.size ==
             attention_weight->data.size);
  assert(mamba_weight->grad.norm() > 0.0f);
  assert(attention_weight->grad.norm() > 0.0f);

  const auto interactions =
      audit.hybrid_interaction_records();
  assert(interactions.size() == 2);
  assert(interactions[0].pass == "forward");
  assert(interactions[1].pass == "backward");
  for (const auto& record : interactions) {
    assert(record.layer_index == 0);
    assert(std::isfinite(record.signal_cosine));
    assert(std::isfinite(record.contribution_cosine));
    assert(std::isfinite(record.mamba_ffn_signal_cosine));
    assert(std::isfinite(record.attention_ffn_signal_cosine));
    assert(std::isfinite(
        record.mamba_ffn_contribution_cosine));
    assert(std::isfinite(
        record.attention_ffn_contribution_cosine));
    assert(record.ffn_signal.elements == input.size);
    assert(record.ffn_contribution.elements == input.size);
    assert(std::isfinite(record.cancellation_fraction));
    assert(record.cancellation_fraction >= 0.0);
    assert(record.cancellation_fraction <= 1.0);
  }
  const LayerAuditSummary summary =
      audit.summarize_phase("train");
  assert(summary.hybrid_interaction_records == 2);
  assert(summary.healthy());
  std::cout
      << "Parallel hybrid gradient and interaction audit test passed!"
      << std::endl;
}

void test_edge_pack_integrity_transaction_and_legacy_compatibility() {
  ScopedTestDirectory directory;
  const auto current_path = directory.path / "model.edge";
  const auto corrupted_path = directory.path / "model.corrupt.edge";
  const auto truncated_path = directory.path / "model.truncated.edge";
  const auto legacy_path = directory.path / "model.v2.edge";
  const auto legacy_truncated_path =
      directory.path / "model.v2.truncated.edge";
  const auto legacy_trailing_path =
      directory.path / "model.v2.trailing.edge";

  const ModelConfig config = edge_pack_test_config();
  JambaModel source(config, Device::CPU);
  const auto source_layers = source.collect_bitlinear_layers();
  assert(!source_layers.empty());
  bool exercised_non_aligned_weight_count = false;
  for (BitLinear* layer : source_layers) {
    const uint64_t count =
        static_cast<uint64_t>(layer->input_features()) *
        static_cast<uint64_t>(layer->output_features());
    exercised_non_aligned_weight_count |= (count % 16u) != 0u;
  }
  assert(exercised_non_aligned_weight_count);
  source.save_edge_linear_pack(current_path.string());

  const std::vector<unsigned char> current_bytes =
      read_binary_file(current_path);
  constexpr size_t integrity_trailer_bytes = 4u + 8u + 64u;
  assert(current_bytes.size() >
         12u + integrity_trailer_bytes);

  // Current v3 round-trip: packed robust projections are released, while
  // quantization-sensitive projections retain their exact FP32 weights.
  JambaModel released_target(config, Device::CPU);
  released_target.load_edge_linear_pack(
      current_path.string(), true);
  const auto released_layers =
      released_target.collect_bitlinear_layers();
  assert(released_layers.size() == source_layers.size());
  for (size_t index = 0; index < source_layers.size(); ++index) {
    assert_packed_state_equal(
        source_layers[index]->export_packed_state(),
        released_layers[index]->export_packed_state());
    if (source_layers[index]->quantization_sensitive()) {
      assert(released_layers[index]->has_full_precision_weight());
      assert(released_layers[index]->reference_path_enabled());
      assert_tensor_bitwise_equal(
          source_layers[index]->weight.data,
          released_layers[index]->weight.data);
    } else {
      assert(!released_layers[index]->has_full_precision_weight());
      assert(!released_layers[index]->reference_path_enabled());
    }
  }

  // Retaining the reference path is only valid on top of the matching FP32
  // base checkpoint. A random target must be rejected without mutation.
  JambaModel retained_target(config, Device::CPU);
  const auto random_target_before =
      snapshot_edge_layers(retained_target);
  bool mismatched_base_rejected = false;
  try {
    retained_target.load_edge_linear_pack(
        current_path.string(), false);
  } catch (const std::exception&) {
    mismatched_base_rejected = true;
  }
  assert(mismatched_base_rejected);
  assert_edge_layers_unchanged(
      retained_target, random_target_before);

  const auto source_parameters = source.parameters();
  const auto retained_parameters = retained_target.parameters();
  assert(source_parameters.size() ==
         retained_parameters.size());
  for (size_t index = 0;
       index < source_parameters.size(); ++index) {
    assert(source_parameters[index] != nullptr);
    assert(retained_parameters[index] != nullptr);
    assert(source_parameters[index]->name ==
           retained_parameters[index]->name);
    retained_parameters[index]->data.copy_from(
        source_parameters[index]->data);
    retained_parameters[index]->version =
        source_parameters[index]->version;
  }
  retained_target.load_edge_linear_pack(
      current_path.string(), false);
  const auto retained_layers =
      retained_target.collect_bitlinear_layers();
  for (size_t layer_index = 0;
       layer_index < retained_layers.size(); ++layer_index) {
    BitLinear* layer = retained_layers[layer_index];
    assert(layer->has_full_precision_weight());
    assert(layer->reference_path_enabled());
    assert_tensor_bitwise_equal(
        source_layers[layer_index]->weight.data,
        layer->weight.data);
  }

  // A flipped payload byte and a truncated integrity trailer are rejected
  // before any live-layer mutation.
  std::vector<unsigned char> corrupted = current_bytes;
  corrupted[16] ^= 0x01u;
  write_binary_file(corrupted_path, corrupted);
  JambaModel rejection_target(config, Device::CPU);
  expect_edge_load_rejected_without_mutation(
      rejection_target, corrupted_path);

  std::vector<unsigned char> truncated = current_bytes;
  truncated.resize(truncated.size() - 13u);
  write_binary_file(truncated_path, truncated);
  expect_edge_load_rejected_without_mutation(
      rejection_target, truncated_path);

  // v2 had the same payload but no integrity trailer. Preserve read
  // compatibility while enforcing exact EOF and parse-before-commit behavior.
  std::vector<unsigned char> legacy(
      current_bytes.begin(),
      current_bytes.end() -
          static_cast<std::ptrdiff_t>(integrity_trailer_bytes));
  const uint32_t legacy_version = 2u;
  std::memcpy(
      legacy.data() + sizeof(uint32_t),
      &legacy_version, sizeof(legacy_version));
  write_binary_file(legacy_path, legacy);
  JambaModel legacy_target(config, Device::CPU);
  legacy_target.load_edge_linear_pack(
      legacy_path.string(), true);
  const auto legacy_layers =
      legacy_target.collect_bitlinear_layers();
  for (size_t index = 0; index < source_layers.size(); ++index) {
    assert_packed_state_equal(
        source_layers[index]->export_packed_state(),
        legacy_layers[index]->export_packed_state());
  }

  std::vector<unsigned char> legacy_truncated = legacy;
  legacy_truncated.pop_back();
  write_binary_file(
      legacy_truncated_path, legacy_truncated);
  expect_edge_load_rejected_without_mutation(
      rejection_target, legacy_truncated_path);

  std::vector<unsigned char> legacy_trailing = legacy;
  legacy_trailing.push_back(0xA5u);
  write_binary_file(
      legacy_trailing_path, legacy_trailing);
  expect_edge_load_rejected_without_mutation(
      rejection_target, legacy_trailing_path);

  std::cout
      << "Edge-pack integrity, transaction and v2 compatibility test passed!"
      << std::endl;
}

void test_attention_gqa_streaming() {
  const int d_model = 32;
  Attention attention(d_model, 4, 512, 2);
  assert(attention.num_query_heads() == 4);
  assert(attention.num_kv_heads() == 2);

  Tensor x = Tensor::random({130, d_model});
  Tensor full = attention.forward(x, nullptr);
  Tensor expected_last = full.slice(0, 129, 130);

  attention.reset();
  attention.set_streaming_mode(true);
  Tensor last;
  for (int i = 0; i < 130; ++i) {
    last = attention.forward(x.slice(0, i, i + 1), nullptr);
  }
  attention.set_streaming_mode(false);

  assert_tensor_close(expected_last, last);

  std::cout << "Jamba GQA streaming test passed!" << std::endl;
}

void test_attention_snapshot_branching() {
  const int d_model = 32;
  Attention attention(d_model, 4, 512, 2);
  Tensor prefix = Tensor::random({96, d_model});
  Tensor branch_a = Tensor::random({2, d_model});
  Tensor branch_b = Tensor::random({2, d_model});

  attention.set_streaming_mode(true);
  for (int i = 0; i < prefix.shape[0]; ++i) {
    (void)attention.forward(prefix.slice(0, i, i + 1), nullptr);
  }
  AttentionCacheSnapshot snapshot = attention.snapshot_cache();

  Tensor last_a = attention.forward(branch_a.slice(0, 0, 1), nullptr);
  last_a = attention.forward(branch_a.slice(0, 1, 2), nullptr);

  attention.restore_cache(snapshot);
  Tensor last_b = attention.forward(branch_b.slice(0, 0, 1), nullptr);
  last_b = attention.forward(branch_b.slice(0, 1, 2), nullptr);
  attention.set_streaming_mode(false);

  attention.reset();
  Tensor expected_a =
      attention.forward(concat_rows(prefix, branch_a), nullptr).slice(0, prefix.shape[0] + 1,
                                                                      prefix.shape[0] + 2);
  attention.reset();
  Tensor expected_b =
      attention.forward(concat_rows(prefix, branch_b), nullptr).slice(0, prefix.shape[0] + 1,
                                                                      prefix.shape[0] + 2);

  assert_tensor_close(last_a, expected_a);
  assert_tensor_close(last_b, expected_b);
  std::cout << "Attention cache snapshot branching test passed!" << std::endl;
}

void test_attention_gpu_prefill_parity() {
#ifdef USE_CUDA
  const int d_model = 32;
  Attention attention(d_model, 4, 512, 2);
  Tensor x = Tensor::random({48, d_model});

  Tensor cpu_out = attention.forward(x, nullptr);
  attention.to(Device::GPU);
  Tensor gpu_out = attention.forward(x.to(Device::GPU), nullptr).cpu();

  assert_tensor_close(cpu_out, gpu_out, 2e-3f);
  std::cout << "Attention GPU prefill parity test passed!" << std::endl;
#else
  std::cout << "Attention GPU prefill parity test skipped." << std::endl;
#endif
}

void test_mamba_session_fork_restore() {
  JambaModel model(4, 16, 64);
  assert(model.supports_streaming_inference());
  model.set_streaming_inference(true);

  const std::vector<int> prefix = {1, 2, 3, 4, 5};
  (void)model.forward_ids(prefix, nullptr);
  JambaSessionSnapshot snapshot = model.fork_session();
  assert(snapshot.input_ids == prefix);

  Tensor branch_a = model.forward_ids({6}, nullptr);
  model.restore_session(snapshot);
  Tensor branch_b = model.forward_ids({7}, nullptr);

  model.reset_session();
  model.set_streaming_inference(true);
  (void)model.forward_ids(prefix, nullptr);
  Tensor expected_a = model.forward_ids({6}, nullptr);

  model.reset_session();
  model.set_streaming_inference(true);
  (void)model.forward_ids(prefix, nullptr);
  Tensor expected_b = model.forward_ids({7}, nullptr);

  assert_tensor_close(branch_a, expected_a, 1e-4f);
  assert_tensor_close(branch_b, expected_b, 1e-4f);
  std::cout << "Jamba session fork/restore test passed!" << std::endl;
}

void test_mamba_streaming_prefill_matches_incremental() {
  JambaModel model(4, 16, 64);
  assert(model.supports_streaming_inference());

  const std::vector<int> prefix = {1, 2, 3, 4, 5, 6};
  const std::vector<int> next = {7};

  model.reset_session();
  model.set_streaming_inference(true);
  (void)model.forward_ids(prefix, nullptr);
  Tensor full_prefill_next = model.forward_ids(next, nullptr).cpu();

  model.reset_session();
  model.set_streaming_inference(true);
  for (int token : prefix) {
    (void)model.forward_ids({token}, nullptr);
  }
  Tensor incremental_next = model.forward_ids(next, nullptr).cpu();

  assert_tensor_close(full_prefill_next, incremental_next, 1e-4f);
  std::cout << "Mamba streaming prefill parity test passed!" << std::endl;
}

void test_mamba_batched_streaming_decode() {
  JambaModel model(4, 16, 64);
  assert(model.supports_streaming_inference());
  assert(model.supports_batched_streaming_inference());

  const std::vector<std::vector<int>> prompts = {{1, 2, 3, 4}, {5, 6, 7}};
  const std::vector<int> next_tokens = {8, 9};
  std::vector<JambaSessionSnapshot> snapshots;
  std::vector<Tensor> expected_logits;

  model.set_streaming_inference(true);
  for (size_t i = 0; i < prompts.size(); ++i) {
    model.reset_session();
    model.set_streaming_inference(true);
    (void)model.forward_ids(prompts[i], nullptr);
    JambaSessionSnapshot snapshot = model.fork_session();
    snapshots.push_back(snapshot);

    model.restore_session(snapshot);
    Tensor logits = model.forward_ids({next_tokens[i]}, nullptr).cpu();
    if (logits.shape.size() == 3) {
      logits = logits.reshape({1, logits.shape[2]});
    }
    expected_logits.push_back(std::move(logits));
  }

  model.restore_session_batch(snapshots);
  Tensor batch_logits =
      model.forward_ids_batch({{next_tokens[0]}, {next_tokens[1]}}, nullptr).cpu();
  assert(batch_logits.shape.size() == 3);
  assert(batch_logits.shape[0] == 2);
  assert(batch_logits.shape[1] == 1);

  const int vocab = batch_logits.shape[2];
  for (int row = 0; row < 2; ++row) {
    Tensor row_logits({1, vocab}, Device::CPU);
    std::memcpy(row_logits.data(),
                batch_logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab),
                static_cast<size_t>(vocab) * sizeof(float));
    assert_tensor_close(row_logits, expected_logits[static_cast<size_t>(row)], 1e-4f);
  }

  auto roundtrip = model.fork_session_batch();
  assert(roundtrip.size() == prompts.size());
  std::cout << "Mamba batched streaming decode test passed!" << std::endl;
}

void test_attention_batched_streaming_decode() {
  ModelConfig config;
  config.num_layers = 4;
  config.d_model = 32;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 1;
  config.attention_slot = 0;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_exact_attention_training = false;

  JambaModel model(config, Device::CPU);
  assert(model.supports_streaming_inference());
  assert(model.supports_batched_streaming_inference());

  const std::vector<std::vector<int>> prompts = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const std::vector<int> next_tokens = {9, 10};
  std::vector<JambaSessionSnapshot> snapshots;
  std::vector<Tensor> expected_logits;

  model.set_streaming_inference(true);
  for (size_t i = 0; i < prompts.size(); ++i) {
    model.reset_session();
    model.set_streaming_inference(true);
    for (int token : prompts[i]) {
      (void)model.forward_ids({token}, nullptr);
    }
    JambaSessionSnapshot snapshot = model.fork_session();
    snapshots.push_back(snapshot);

    model.restore_session(snapshot);
    Tensor logits = model.forward_ids({next_tokens[i]}, nullptr).cpu();
    if (logits.shape.size() == 3) {
      logits = logits.reshape({1, logits.shape[2]});
    }
    expected_logits.push_back(std::move(logits));
  }

  model.restore_session_batch(snapshots);
  Tensor batch_logits =
      model.forward_ids_batch({{next_tokens[0]}, {next_tokens[1]}}, nullptr).cpu();
  assert(batch_logits.shape.size() == 3);
  assert(batch_logits.shape[0] == 2);
  assert(batch_logits.shape[1] == 1);

  const int vocab = batch_logits.shape[2];
  for (int row = 0; row < 2; ++row) {
    Tensor row_logits({1, vocab}, Device::CPU);
    std::memcpy(row_logits.data(),
                batch_logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab),
                static_cast<size_t>(vocab) * sizeof(float));
    assert_tensor_close(row_logits, expected_logits[static_cast<size_t>(row)], 1e-4f);
  }

  auto roundtrip = model.fork_session_batch();
  assert(roundtrip.size() == prompts.size());
  std::cout << "Attention batched streaming decode test passed!" << std::endl;
}

void test_attention_batched_cache_growth() {
  ModelConfig config;
  config.num_layers = 1;
  config.d_model = 16;
  config.vocab_size = 80;
  config.mamba_head_dim = 16;
  config.n_heads = 2;
  config.n_kv_heads = 1;
  config.attention_period = 1;
  config.attention_slot = 0;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_exact_attention_training = false;

  JambaModel model(config, Device::CPU);
  model.set_streaming_inference(true);

  std::vector<std::vector<int>> prompts(2);
  for (int row = 0; row < 2; ++row) {
    for (int i = 0; i < 63; ++i) {
      prompts[static_cast<size_t>(row)].push_back(1 + ((i + row * 13) % 70));
    }
  }
  const std::vector<std::vector<int>> next = {{71, 72, 73}, {74, 75, 76}};
  std::vector<JambaSessionSnapshot> snapshots;
  for (const auto& prompt : prompts) {
    model.reset_session();
    model.set_streaming_inference(true);
    (void)model.forward_ids(prompt, nullptr);
    snapshots.push_back(model.fork_session());
  }

  std::vector<std::vector<Tensor>> expected(3, std::vector<Tensor>(2));
  for (int row = 0; row < 2; ++row) {
    model.restore_session(snapshots[static_cast<size_t>(row)]);
    for (int step = 0; step < 3; ++step) {
      Tensor logits =
          model.forward_ids({next[static_cast<size_t>(row)][static_cast<size_t>(step)]},
                            nullptr)
              .cpu();
      expected[static_cast<size_t>(step)][static_cast<size_t>(row)] =
          logits.reshape({1, logits.shape.back()});
    }
  }

  model.restore_session_batch(snapshots);
  for (int step = 0; step < 3; ++step) {
    Tensor logits =
        model
            .forward_ids_batch(
                {{next[0][static_cast<size_t>(step)]},
                 {next[1][static_cast<size_t>(step)]}},
                nullptr)
            .cpu();
    const int vocab = logits.shape.back();
    for (int row = 0; row < 2; ++row) {
      Tensor actual_row({1, vocab}, Device::CPU);
      std::memcpy(actual_row.data(),
                  logits.data() + static_cast<size_t>(row) * vocab,
                  static_cast<size_t>(vocab) * sizeof(float));
      assert_tensor_close(
          actual_row,
          expected[static_cast<size_t>(step)][static_cast<size_t>(row)],
          1e-4f);
    }
  }

  std::cout << "Attention batched cache-growth test passed!" << std::endl;
}

void test_dropout_backward_gradcheck() {
  // Keep the finite-difference oracle reproducible; do not sample ill-conditioned
  // random weights differently on every release-gate run.
  nsos::determinism::DeterminismManager::instance().set_global_seed(0xD09ULL);
  JambaBlock block(
      8,      // d_model
      false,  // mamba core
      false,  // dense FFN
      false,  // no TTT
      0, 1,   // layer index/count
      2, 1,   // query/KV heads
      1, 1,   // experts/top-k (unused)
      false,  // exact attention training
      0.25f,  // dropout
      false,  // gradient checkpointing
      16,     // compact FFN hidden size
      false, 0.1f, 0u, false,
      true, true, 4,
      4, false); // non-faithful compact Mamba for this local VJP test

  Tensor x({1, 2, 8}, Device::CPU);
  Tensor dy({1, 2, 8}, Device::CPU);
  for (int i = 0; i < x.size; ++i) {
    x.data()[i] = -0.35f + 0.07f * static_cast<float>(i);
    dy.data()[i] = 0.22f - 0.03f * static_cast<float>(i);
  }

  auto objective = [&]() {
    block.reset();
    Context ctx;
    Tensor y = block.forward(x, &ctx);
    double value = 0.0;
    for (int i = 0; i < y.size; ++i) {
      value += static_cast<double>(y.data()[i]) * dy.data()[i];
    }
    return value;
  };

  const float eps = 1e-3f;
  std::vector<float> numerical(static_cast<size_t>(x.size), 0.0f);
  for (int i = 0; i < x.size; ++i) {
    const float original = x.data()[i];
    x.data()[i] = original + eps;
    const double plus = objective();
    x.data()[i] = original - eps;
    const double minus = objective();
    x.data()[i] = original;
    numerical[static_cast<size_t>(i)] = (plus - minus) / (2.0f * eps);
  }

  block.reset();
  Context ctx;
  (void)block.forward(x, &ctx);
  Tensor analytic = block.backward(dy, &ctx);
  for (int i = 0; i < analytic.size; ++i) {
    if (!std::isfinite(analytic.data()[i]) ||
        !std::isfinite(numerical[static_cast<size_t>(i)]) ||
        std::abs(analytic.data()[i] - numerical[static_cast<size_t>(i)]) >= 2e-2f) {
      throw std::runtime_error("dropout VJP mismatch at input " + std::to_string(i) +
          ": analytic=" + std::to_string(analytic.data()[i]) +
          " numerical=" + std::to_string(numerical[static_cast<size_t>(i)]));
    }
  }

  std::cout << "Dropout backward gradcheck passed!" << std::endl;
}

void test_ttt_batched_streaming_decode() {
  ModelConfig config;
  config.num_layers = 4;
  config.d_model = 32;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 64;
  config.attention_slot = 63;
  config.use_moe = false;
  config.use_ttt = true;
  config.ttt_period = 1;
  config.ttt_slot = 0;
  config.use_exact_attention_training = false;

  JambaModel model(config, Device::CPU);
  assert(model.supports_streaming_inference());
  assert(model.supports_batched_streaming_inference());

  const std::vector<std::vector<int>> prompts = {{2, 3, 4}, {7, 8, 9}};
  const std::vector<int> next_tokens = {5, 10};
  std::vector<JambaSessionSnapshot> snapshots;
  std::vector<Tensor> expected_logits;

  model.set_streaming_inference(true);
  for (size_t i = 0; i < prompts.size(); ++i) {
    model.reset_session();
    model.set_streaming_inference(true);
    for (int token : prompts[i]) {
      (void)model.forward_ids({token}, nullptr);
    }
    JambaSessionSnapshot snapshot = model.fork_session();
    snapshots.push_back(snapshot);

    model.restore_session(snapshot);
    Tensor logits = model.forward_ids({next_tokens[i]}, nullptr).cpu();
    if (logits.shape.size() == 3) {
      logits = logits.reshape({1, logits.shape[2]});
    }
    expected_logits.push_back(std::move(logits));
  }

  model.restore_session_batch(snapshots);
  Tensor batch_logits =
      model.forward_ids_batch({{next_tokens[0]}, {next_tokens[1]}}, nullptr).cpu();
  assert(batch_logits.shape.size() == 3);
  assert(batch_logits.shape[0] == 2);
  assert(batch_logits.shape[1] == 1);

  const int vocab = batch_logits.shape[2];
  for (int row = 0; row < 2; ++row) {
    Tensor row_logits({1, vocab}, Device::CPU);
    std::memcpy(row_logits.data(),
                batch_logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab),
                static_cast<size_t>(vocab) * sizeof(float));
    assert_tensor_close(row_logits, expected_logits[static_cast<size_t>(row)], 1e-4f);
  }

  auto roundtrip = model.fork_session_batch();
  assert(roundtrip.size() == prompts.size());
  std::cout << "TTT batched streaming decode test passed!" << std::endl;
}

void test_jamba_rank3_batch_forward_backward() {
  JambaModel model(4, 16, 64);
  const std::vector<std::vector<int>> batch_ids = {{1, 2, 3, 4}, {5, 6}};

  Tensor batched = model.forward_ids_batch(batch_ids, nullptr);
  assert(batched.shape.size() == 3);
  assert(batched.shape[0] == 2);
  assert(batched.shape[1] == 4);
  assert(batched.shape[2] == 64);

  for (int batch = 0; batch < static_cast<int>(batch_ids.size()); ++batch) {
    model.reset_session();
    Tensor single = model.forward_ids(batch_ids[static_cast<size_t>(batch)], nullptr);
    Tensor batched_slice = batched.slice(0, batch, batch + 1).reshape({4, 64});
    Tensor valid_slice = batched_slice.slice(0, 0, static_cast<int>(batch_ids[static_cast<size_t>(batch)].size()));
    assert_tensor_close(valid_slice, single, 1e-4f);
    for (int row = static_cast<int>(batch_ids[static_cast<size_t>(batch)].size());
         row < batched_slice.shape[0];
         ++row) {
      for (int col = 0; col < batched_slice.shape[1]; ++col) {
        assert(std::abs(batched_slice.data()[row * batched_slice.shape[1] + col]) < 1e-6f);
      }
    }
  }

  model.reset_session();
  Context ctx;
  batched = model.forward_ids_batch(batch_ids, &ctx);
  Tensor grad = Tensor::zeros(batched.shape.dims, batched.get_device());
  Tensor grad_cpu = grad.cpu();
  for (int batch = 0; batch < static_cast<int>(batch_ids.size()); ++batch) {
    const auto& sequence = batch_ids[static_cast<size_t>(batch)];
    for (int row = 0; row < static_cast<int>(sequence.size()); ++row) {
      const int target = (sequence[static_cast<size_t>(row)] + 1) % 64;
      grad_cpu.data()[((batch * batched.shape[1] + row) * batched.shape[2]) + target] =
          1.0f / static_cast<float>(sequence.size());
    }
  }
  grad.copy_from(grad_cpu);
  model.backward_external(grad, ctx);
  assert(model.embedding->weight.grad.size == model.embedding->weight.data.size);
  assert(model.embedding->weight.grad.norm() > 0.0f);
  std::cout << "Jamba rank-3 batch forward/backward test passed!" << std::endl;
}

void test_attention_rank3_heterogeneous_backward() {
  Attention attention(32, 4, 512, 2);
  Tensor seq_a = Tensor::random({5, 32});
  Tensor seq_b = Tensor::random({3, 32});
  Tensor batch = make_padded_batch({seq_a, seq_b});

  attention.set_training_mode(true);
  attention.set_batch_valid_lengths({5, 3});
  Tensor batch_out = attention.forward(batch, nullptr);
  (void)batch_out;

  Tensor grad_batch({2, 5, 32}, Device::CPU);
  std::fill_n(grad_batch.data(), grad_batch.size, 0.0f);
  for (int row = 0; row < 5; ++row) {
    grad_batch.data()[(row * 32) + (row % 32)] = 0.25f;
  }
  for (int row = 0; row < 3; ++row) {
    grad_batch.data()[((5 + row) * 32) + ((row + 7) % 32)] = 0.5f;
  }

  Tensor dx_batch = attention.backward(grad_batch, nullptr).cpu();
  Tensor dx_batch_a = dx_batch.slice(0, 0, 1).reshape({5, 32});
  Tensor dx_batch_b_full = dx_batch.slice(0, 1, 2).reshape({5, 32});
  Tensor dx_batch_b = dx_batch_b_full.slice(0, 0, 3);

  attention.set_batch_valid_lengths({});
  Tensor grad_a = grad_batch.slice(0, 0, 1).reshape({5, 32});
  (void)attention.forward(seq_a, nullptr);
  Tensor dx_a = attention.backward(grad_a, nullptr).cpu();

  Tensor grad_b = grad_batch.slice(0, 1, 2).reshape({5, 32}).slice(0, 0, 3);
  (void)attention.forward(seq_b, nullptr);
  Tensor dx_b = attention.backward(grad_b, nullptr).cpu();

  assert_tensor_close(dx_batch_a, dx_a, 1e-3f);
  assert_tensor_close(dx_batch_b, dx_b, 1e-3f);
  for (int row = 3; row < 5; ++row) {
    for (int col = 0; col < 32; ++col) {
      assert(std::abs(dx_batch_b_full.data()[row * 32 + col]) < 1e-5f);
    }
  }
  std::cout << "Attention heterogeneous rank-3 backward test passed!" << std::endl;
}

void test_sparse_router_topk() {
  MoERouter router(32, 8, 2);
  Tensor x = Tensor::random({4, 32});
  auto [logits, weights] = router.forward(x);
  (void)logits;

  Tensor weights_cpu = weights.cpu();
  for (int row = 0; row < weights_cpu.shape[0]; ++row) {
    int non_zero = 0;
    float sum = 0.0f;
    for (int expert = 0; expert < weights_cpu.shape[1]; ++expert) {
      const float value = weights_cpu.data()[row * weights_cpu.shape[1] + expert];
      if (value > 1e-6f) {
        ++non_zero;
        sum += value;
      }
    }
    assert(non_zero <= 2);
    assert(std::abs(sum - 1.0f) < 1e-4f);
  }
  std::cout << "Sparse router top-k test passed!" << std::endl;

#ifdef USE_CUDA
  router.to(Device::GPU);
  auto [gpu_logits, gpu_weights] = router.forward(x.to(Device::GPU));
  (void)gpu_logits;
  Tensor gpu_weights_cpu = gpu_weights.cpu();
  for (int row = 0; row < gpu_weights_cpu.shape[0]; ++row) {
    int non_zero = 0;
    float sum = 0.0f;
    for (int expert = 0; expert < gpu_weights_cpu.shape[1]; ++expert) {
      const float value = gpu_weights_cpu.data()[row * gpu_weights_cpu.shape[1] + expert];
      if (value > 1e-6f) {
        ++non_zero;
        sum += value;
      }
    }
    assert(non_zero <= 2);
    assert(std::abs(sum - 1.0f) < 1e-4f);
  }
  std::cout << "Sparse router GPU top-k test passed!" << std::endl;
#endif
}

void test_embedding_heterogeneous_batch_device_safe() {
  Embedding embedding(32, 16);
  const std::vector<std::vector<int>> ids = {{1, 2, 3, 4}, {5, 6}};

  Tensor batch = embedding.forward_batch(ids).cpu();
  assert(batch.shape.size() == 3);
  assert(batch.shape[0] == 2);
  assert(batch.shape[1] == 4);
  assert(batch.shape[2] == 16);

  Tensor single_a = embedding.forward(ids[0]).cpu();
  Tensor single_b = embedding.forward(ids[1]).cpu();
  assert_tensor_close(batch.slice(0, 0, 1).reshape({4, 16}), single_a);
  assert_tensor_close(batch.slice(0, 1, 2).reshape({4, 16}).slice(0, 0, 2), single_b);
  for (int col = 0; col < 16; ++col) {
    assert(std::abs(batch.data()[(1 * 4 + 2) * 16 + col]) < 1e-6f);
    assert(std::abs(batch.data()[(1 * 4 + 3) * 16 + col]) < 1e-6f);
  }

  Tensor grad({2, 4, 16}, Device::CPU);
  std::fill_n(grad.data(), grad.size, 0.0f);
  for (int i = 0; i < grad.size; ++i) {
    grad.data()[i] = 0.1f + static_cast<float>(i % 7) * 0.01f;
  }
  embedding.backward_batch(grad, ids);
  assert(embedding.weight.grad.size > 0);
  assert(embedding.weight.grad.norm() > 0.0f);

#ifdef USE_CUDA
  Embedding gpu_embedding(32, 16);
  gpu_embedding.to(Device::GPU);
  gpu_embedding.weight.data.copy_from(embedding.weight.data.to(Device::GPU));
  Tensor gpu_batch = gpu_embedding.forward_batch(ids).cpu();
  assert_tensor_close(batch, gpu_batch, 1e-4f);

  Tensor grad_gpu = grad.to(Device::GPU);
  gpu_embedding.backward_batch(grad_gpu, ids);
  assert(gpu_embedding.weight.grad.size > 0);
  assert(gpu_embedding.weight.grad.cpu().norm() > 0.0f);

  // Deterministic present-ID reduction must preserve the legacy increasing
  // position order for repeats, valid token 0, EOS-like final IDs, and the
  // implicit -1 padding positions of a heterogeneous batch.
  Embedding ordered_cpu(32, 16);
  Embedding ordered_gpu(32, 16);
  ordered_gpu.to(Device::GPU);
  ordered_gpu.weight.data.copy_from(
      ordered_cpu.weight.data.to(Device::GPU));
  const std::vector<std::vector<int>> ordered_ids = {
      {0, 7, 7, 31}, {7, 0}};
  Tensor ordered_grad({2, 4, 16}, Device::CPU);
  for (int i = 0; i < ordered_grad.size; ++i) {
    ordered_grad.data()[i] =
        0.03125f * static_cast<float>((i % 13) - 6);
  }
  nsos::determinism::set_deterministic_reductions(true);
  ordered_cpu.backward_batch(ordered_grad, ordered_ids);
  ordered_gpu.backward_batch(
      ordered_grad.to(Device::GPU), ordered_ids);
  Tensor first_ordered_gpu_grad = ordered_gpu.weight.grad.cpu();
  Tensor ordered_cpu_grad = ordered_cpu.weight.grad.cpu();
  assert(std::memcmp(
             first_ordered_gpu_grad.data(), ordered_cpu_grad.data(),
             static_cast<size_t>(ordered_cpu_grad.size) * sizeof(float)) == 0);
  ordered_gpu.weight.zero_grad();
  ordered_gpu.backward_batch(
      ordered_grad.to(Device::GPU), ordered_ids);
  Tensor repeated_ordered_gpu_grad = ordered_gpu.weight.grad.cpu();
  assert(std::memcmp(
             first_ordered_gpu_grad.data(),
             repeated_ordered_gpu_grad.data(),
             static_cast<size_t>(first_ordered_gpu_grad.size) *
                 sizeof(float)) == 0);

  const Tensor invalid_forward =
      ordered_gpu.forward({-1, 32}).cpu();
  for (int i = 0; i < invalid_forward.size; ++i) {
    assert(invalid_forward.data()[i] == 0.0f);
  }
  nsos::determinism::set_deterministic_reductions(false);
#endif

  std::cout << "Embedding heterogeneous device-safe batch test passed!" << std::endl;
}

// Explicit executable oracle for integration plumbing, not language quality.
#include "verified_reasoning_fixture.h"

void test_fullstack_aux_training_smoke() {
  try {
  ModelConfig config;
  config.num_layers = 6;
  config.d_model = 32;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 3;
  config.attention_slot = 1;
  config.use_moe = true;
  config.num_experts = 4;
  config.num_experts_per_token = 2;
  config.moe_period = 3;
  config.moe_slot = 2;
  config.use_ttt = true;
  config.ttt_period = 3;
  config.ttt_slot = 0;
  config.use_exact_attention_training = true;

  JambaModel model(config, Device::CPU);
  assert(model.layers[0]->uses_ttt());
  assert(model.layers[1]->uses_attention());
  assert(model.layers[2]->uses_moe());
  model.set_reasoning_policy(arithmetic_reasoning_fixture());

  Trainer trainer(&model, 5e-4f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 2.0f;
  trainer.phase_scheduler.auxiliary_stack_enabled = true;
  trainer.phase_scheduler.auxiliary_session_adapt_enabled = true;
  trainer.phase_scheduler.auxiliary_reasoning_enabled = true;
  trainer.phase_scheduler.auxiliary_memory_enabled = true;
  trainer.phase_scheduler.auxiliary_reasoning_iterations = 2;
  trainer.phase_scheduler.auxiliary_reasoning_simulations = 12;
  trainer.phase_scheduler.auxiliary_memory_blend = 0.4f;

  const std::vector<std::vector<int>> prompt_batch = {
      {1, 2, 3, 4, 5},
      {6, 7, 8, 9},
  };
  const std::vector<std::vector<int>> answer_batch = {
      {10, 11},
      {12, 13, 14},
  };

  const float loss = trainer.train_supervised_batch(prompt_batch, answer_batch);
  assert(std::isfinite(loss));
  assert(loss > 0.0f);
  assert(model.embedding->weight.grad.size == model.embedding->weight.data.size);
  std::cout << "Full-stack auxiliary training smoke test passed!" << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "Full-stack auxiliary training smoke failed: " << ex.what() << std::endl;
    throw std::runtime_error(ex.what());
  }
}

int main() {
  try {
    test_jamba_structure();
    test_faithful_last_attention_guard();
    test_parallel_hybrid_registry_and_zero_gate_reduction();
    test_parallel_hybrid_gradients_and_interaction_audit();
    test_edge_pack_integrity_transaction_and_legacy_compatibility();
    test_attention_gqa_streaming();
    test_attention_snapshot_branching();
    test_attention_gpu_prefill_parity();
    test_mamba_session_fork_restore();
    test_mamba_streaming_prefill_matches_incremental();
    test_mamba_batched_streaming_decode();
    test_attention_batched_streaming_decode();
    test_attention_batched_cache_growth();
    test_dropout_backward_gradcheck();
    test_ttt_batched_streaming_decode();
    test_jamba_rank3_batch_forward_backward();
    test_attention_rank3_heterogeneous_backward();
    test_sparse_router_topk();
    test_embedding_heterogeneous_batch_device_safe();
    test_fullstack_aux_training_smoke();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_jamba failed: " << error.what() << '\n';
    return 1;
  }
}
