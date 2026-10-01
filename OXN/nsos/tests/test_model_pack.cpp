#include "nsos_sdk.h"
#include "nsos/sha256.h"
#include "nsos_serializer.h"
#include "checkpoint_io.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

using namespace nsos;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<unsigned char> read_binary(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.is_open(), "failed to open binary file for reading");
  return std::vector<unsigned char>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

void write_binary(const std::filesystem::path& path,
                  const std::vector<unsigned char>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "failed to open binary file for writing");
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  require(static_cast<bool>(output), "failed to write binary file");
}

template <typename T>
T read_scalar(const std::vector<unsigned char>& bytes, size_t offset) {
  require(offset <= bytes.size() &&
              sizeof(T) <= bytes.size() - offset,
          "binary scalar is out of bounds");
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

void write_scalar(std::vector<unsigned char>* bytes, size_t offset,
                  float value) {
  require(bytes != nullptr && offset <= bytes->size() &&
              sizeof(value) <= bytes->size() - offset,
          "binary float is out of bounds");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::string manifest_value(const std::string& manifest,
                           const std::string& key) {
  const std::string prefix = key + "=";
  const size_t position = manifest.find(prefix);
  require(position != std::string::npos &&
              (position == 0 ||
               manifest[position - 1] == '\n'),
          "manifest is missing key: " + key);
  const size_t value_begin = position + prefix.size();
  const size_t value_end = manifest.find('\n', value_begin);
  const std::string value = manifest.substr(
      value_begin,
      value_end == std::string::npos
          ? std::string::npos
          : value_end - value_begin);
  require(!value.empty(), "manifest value is empty: " + key);
  return value;
}

std::vector<Tensor> snapshot_parameters(JambaModel& model) {
  std::vector<Tensor> snapshot;
  for (Parameter* parameter : model.parameters()) {
    require(parameter != nullptr, "null model parameter");
    snapshot.push_back(parameter->data.cpu().clone());
  }
  return snapshot;
}

void require_parameters_unchanged(
    JambaModel& model, const std::vector<Tensor>& snapshot) {
  const auto parameters = model.parameters();
  require(parameters.size() == snapshot.size(),
          "parameter count changed after rejected load");
  for (size_t parameter_index = 0;
       parameter_index < parameters.size(); ++parameter_index) {
    const Tensor current = parameters[parameter_index]->data.cpu();
    require(current.size == snapshot[parameter_index].size,
            "parameter size changed after rejected load");
    const size_t bytes =
        static_cast<size_t>(current.size) * sizeof(float);
    require(bytes == 0 ||
                std::memcmp(current.data(),
                            snapshot[parameter_index].data(),
                            bytes) == 0,
            "parameter changed after rejected load");
  }
}

#ifdef NSOS_ENABLE_TEST_HOOKS
struct ModelCheckpointFaultReset {
  ~ModelCheckpointFaultReset() {
    testing::clear_model_checkpoint_stage_failure();
  }
};
#endif

} // namespace

int main() {
  try {
    const std::filesystem::path long_destination =
        std::filesystem::path("checkpoint-temp-contract") /
        (std::string(180, 'x') + ".bin");
    const std::filesystem::path first_temporary =
        checkpoint_io::unique_temporary_path(long_destination);
    const std::filesystem::path second_temporary =
        checkpoint_io::unique_temporary_path(long_destination);
    require(first_temporary.parent_path() == long_destination.parent_path(),
            "checkpoint temporary escaped destination directory");
    require(first_temporary.filename().string().size() <= 64,
            "checkpoint temporary basename is not path-budget bounded");
    require(first_temporary != second_temporary,
            "checkpoint temporary names are not unique");
    std::cout << "[ModelPackTest] create engine" << std::endl;
    ModelConfig config;
    config.num_layers = 1;
    config.d_model = 32;
    config.vocab_size = 128;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.max_context_tokens = 64;

    InferenceEngine engine;
    require(engine.load_model("", config), "initial load_model failed");

    std::cout << "[ModelPackTest] train once" << std::endl;
    const float loss = engine.train_step("123456123456");
    require(std::isfinite(loss), "non-finite training loss");

    const std::filesystem::path pack_dir =
        std::filesystem::temp_directory_path() / "nsos_model_pack_test";
    std::filesystem::remove_all(pack_dir);

    std::cout << "[ModelPackTest] save pack to " << pack_dir.string() << std::endl;
    require(engine.save_model_pack(pack_dir.string()), "save_model_pack returned false");
    require(std::filesystem::exists(pack_dir / "manifest.nsos"), "missing manifest");
    const std::vector<unsigned char> manifest_bytes =
        read_binary(pack_dir / "manifest.nsos");
    const std::string manifest_text = [&]() {
      std::ifstream input(pack_dir / "manifest.nsos");
      return std::string(std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>());
    }();
    const std::filesystem::path weights_relative =
        manifest_value(manifest_text, "weights");
    const std::filesystem::path tokenizer_relative =
        manifest_value(manifest_text, "tokenizer");
    const std::filesystem::path config_relative =
        manifest_value(manifest_text, "config");
    const std::filesystem::path edge_relative =
        manifest_value(manifest_text, "edge_linear");
    require(std::filesystem::exists(pack_dir / weights_relative),
            "missing manifest-selected weights");
    require(std::filesystem::exists(pack_dir / tokenizer_relative),
            "missing manifest-selected tokenizer");
    require(std::filesystem::exists(pack_dir / config_relative),
            "missing manifest-selected config");
    require(std::filesystem::exists(pack_dir / edge_relative),
            "missing manifest-selected edge pack");
    require(manifest_text.find("sha256_weights=") != std::string::npos,
            "manifest missing sha256_weights");
    require(manifest_text.find("sha256_config=") != std::string::npos,
            "manifest missing sha256_config");
    require(manifest_text.find("format=nsos-pack-v2") != std::string::npos,
            "manifest missing format");
    require(manifest_text.find("version=2") != std::string::npos,
            "manifest missing version");
    require(manifest_value(manifest_text, "quantization_ready") == "0",
            "pre-QAT checkpoint was advertised as deployment-quantized");

    const std::filesystem::path qat_pack_dir =
        std::filesystem::temp_directory_path() /
        "nsos_model_pack_test_qat";
    std::filesystem::remove_all(qat_pack_dir);
    auto qat_export = engine.clone_for_training();
    TrainPhaseScheduler immediate_qat =
        qat_export->trainer->phase_scheduler;
    immediate_qat.semantic_warmup_steps = 0;
    immediate_qat.qat_start_step = 0;
    qat_export->trainer->configure_progressive_qat(immediate_qat);
    require(qat_export->save_model_pack(qat_pack_dir.string()),
            "active-QAT pack export failed");
    const std::string qat_manifest_text = [&]() {
      std::ifstream input(qat_pack_dir / "manifest.nsos");
      return std::string(std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>());
    }();
    require(manifest_value(qat_manifest_text, "quantization_ready") == "1",
            "active-QAT checkpoint was not advertised as quantization-ready");

    std::cout << "[ModelPackTest] failed replacement preserves generation"
              << std::endl;
    auto failed_export = engine.clone_for_training();
    require(failed_export != nullptr &&
                failed_export->model != nullptr,
            "failed-export clone is unavailable");
    auto failed_export_parameters =
        failed_export->model->parameters();
    require(!failed_export_parameters.empty(),
            "failed-export clone has no parameters");
    failed_export_parameters.front()->data.data()[0] =
        std::numeric_limits<float>::quiet_NaN();
    require(!failed_export->save_model_pack(pack_dir.string()),
            "non-finite pack export should fail");
    require(read_binary(pack_dir / "manifest.nsos") ==
                manifest_bytes,
            "failed pack export replaced the valid manifest");

    std::cout << "[ModelPackTest] reload from pack" << std::endl;
    InferenceEngine reloaded;
    require(reloaded.load_model(pack_dir.string(), ModelConfig{}), "reload load_model failed");
    require(reloaded.config.num_layers == engine.config.num_layers, "num_layers mismatch");
    require(reloaded.config.d_model == engine.config.d_model, "d_model mismatch");
    require(reloaded.config.vocab_size == engine.config.vocab_size, "vocab mismatch");
    require(reloaded.config.max_context_tokens == engine.config.max_context_tokens,
            "context mismatch");

    auto lhs_params = engine.model->parameters();
    auto rhs_params = reloaded.model->parameters();
    require(!lhs_params.empty(), "no parameters on source model");
    require(lhs_params.size() == rhs_params.size(), "parameter count mismatch");

    auto* lhs = lhs_params.front();
    auto* rhs = rhs_params.front();
    require(lhs->data.size == rhs->data.size, "parameter size mismatch");
    for (int i = 0; i < std::min<int64_t>(lhs->data.size, 32); ++i) {
      if (std::abs(lhs->data.data()[i] - rhs->data.data()[i]) >= 1e-5f) {
        throw std::runtime_error("reloaded weight mismatch at index " + std::to_string(i));
      }
    }

    std::cout << "[ModelPackTest] generate from reloaded pack" << std::endl;
    GenerationOptions options;
    options.max_tokens = 4;
    const std::string generated = reloaded.generate("123", options);
    (void)generated;
    const auto& metrics = reloaded.last_generation_metrics();
    require(metrics.loaded_from_pack, "metrics did not mark pack load");

    std::cout << "[ModelPackTest] tamper detection" << std::endl;
    const std::filesystem::path tampered_dir =
        std::filesystem::temp_directory_path() / "nsos_model_pack_test_tampered";
    std::filesystem::remove_all(tampered_dir);
    std::filesystem::create_directories(tampered_dir);
    for (const auto& entry : std::filesystem::directory_iterator(pack_dir)) {
      std::filesystem::copy(entry.path(), tampered_dir / entry.path().filename(),
                            std::filesystem::copy_options::recursive |
                                std::filesystem::copy_options::overwrite_existing);
    }
    {
      std::fstream tampered_weights(
                                    tampered_dir / weights_relative,
                                    std::ios::in | std::ios::out | std::ios::binary);
      require(tampered_weights.is_open(), "failed to open tampered weights");
      char byte = 0;
      tampered_weights.read(&byte, 1);
      require(tampered_weights.good(), "failed to read tampered weights byte");
      byte ^= static_cast<char>(0x5Au);
      tampered_weights.seekp(0);
      tampered_weights.write(&byte, 1);
      require(tampered_weights.good(), "failed to write tampered weights byte");
    }
    InferenceEngine tampered;
    require(!tampered.load_model(tampered_dir.string(), ModelConfig{}),
            "tampered pack should fail integrity verification");

    std::cout << "[ModelPackTest] manifest path traversal rejection" << std::endl;
    const std::filesystem::path traversal_dir =
        std::filesystem::temp_directory_path() / "nsos_model_pack_test_traversal";
    std::filesystem::remove_all(traversal_dir);
    std::filesystem::create_directories(traversal_dir);
    for (const auto& entry : std::filesystem::directory_iterator(pack_dir)) {
      std::filesystem::copy(entry.path(), traversal_dir / entry.path().filename(),
                            std::filesystem::copy_options::recursive |
                                std::filesystem::copy_options::overwrite_existing);
    }
    {
      std::string traversal_manifest = manifest_text;
      const std::string needle =
          "weights=" + weights_relative.generic_string();
      const auto pos = traversal_manifest.find(needle);
      require(pos != std::string::npos, "manifest missing weights entry");
      traversal_manifest.replace(pos, needle.size(), "weights=..\\evil.bin");
      std::ofstream manifest(traversal_dir / "manifest.nsos", std::ios::trunc);
      require(manifest.is_open(), "failed to open traversal manifest");
      manifest << traversal_manifest;
    }
    InferenceEngine traversal;
    require(!traversal.load_model(traversal_dir.string(), ModelConfig{}),
            "manifest path traversal should fail");

    std::cout << "[ModelPackTest] mandatory SHA-256 rejection"
              << std::endl;
    const std::filesystem::path missing_digest_dir =
        std::filesystem::temp_directory_path() /
        "nsos_model_pack_test_missing_digest";
    std::filesystem::remove_all(missing_digest_dir);
    std::filesystem::create_directories(missing_digest_dir);
    for (const auto& entry :
         std::filesystem::directory_iterator(pack_dir)) {
      std::filesystem::copy(
          entry.path(),
          missing_digest_dir / entry.path().filename(),
          std::filesystem::copy_options::recursive |
              std::filesystem::copy_options::overwrite_existing);
    }
    {
      std::string missing_digest_manifest = manifest_text;
      const std::string prefix = "sha256_weights=";
      const size_t begin = missing_digest_manifest.find(prefix);
      require(begin != std::string::npos,
              "manifest missing weights SHA-256");
      size_t end = missing_digest_manifest.find('\n', begin);
      if (end == std::string::npos) {
        end = missing_digest_manifest.size();
      } else {
        ++end;
      }
      missing_digest_manifest.erase(begin, end - begin);
      std::ofstream manifest(
          missing_digest_dir / "manifest.nsos",
          std::ios::trunc);
      require(manifest.is_open(),
              "failed to open missing-digest manifest");
      manifest << missing_digest_manifest;
    }
    InferenceEngine missing_digest;
    require(!missing_digest.load_model(
                missing_digest_dir.string(), ModelConfig{}),
            "pack without mandatory SHA-256 should fail");

    std::cout << "[ModelPackTest] partial load across architecture change" << std::endl;
    const std::filesystem::path partial_path =
        std::filesystem::temp_directory_path() / "nsos_partial_resume_test.bin";
    JambaModel source_model(2, 32, 128, Device::CPU);
    source_model.save(partial_path.string());

    ModelConfig partial_config;
    partial_config.num_layers = 1;
    partial_config.d_model = 32;
    partial_config.vocab_size = 128;
    partial_config.n_heads = 4;
    partial_config.n_kv_heads = 2;
    partial_config.use_ttt = false;
    partial_config.use_moe = false;
    JambaModel partial_target(partial_config, Device::CPU);
    partial_target.load(partial_path.string(), false);
    auto partial_params = partial_target.parameters();
    require(!partial_params.empty(), "partial target has no parameters");
    require(partial_params.front()->data.size > 0, "partial target first parameter empty");

    std::cout << "[ModelPackTest] semantic corruption is transactional"
              << std::endl;
    const std::filesystem::path semantic_path =
        std::filesystem::temp_directory_path() /
        "nsos_semantic_corruption_test.bin";
    source_model.save(semantic_path.string());
    std::vector<unsigned char> semantic_bytes =
        read_binary(semantic_path);
    constexpr size_t integrity_trailer_bytes = 4u + 8u + 64u;
    require(semantic_bytes.size() >
                80u + integrity_trailer_bytes,
            "checkpoint is too small");
    const uint32_t version =
        read_scalar<uint32_t>(semantic_bytes, 4u);
    require(version == NSOS_MODEL_VERSION,
            "unexpected checkpoint version");
    const uint32_t parameter_count =
        read_scalar<uint32_t>(semantic_bytes, 76u);
    require(parameter_count > 0u,
            "checkpoint has no parameters");
    size_t cursor = 80u;
    const uint32_t first_name_length =
        read_scalar<uint32_t>(semantic_bytes, cursor);
    cursor += sizeof(uint32_t);
    require(first_name_length <= semantic_bytes.size() - cursor,
            "checkpoint first name is truncated");
    cursor += first_name_length;
    const uint32_t first_rank =
        read_scalar<uint32_t>(semantic_bytes, cursor);
    cursor += sizeof(uint32_t);
    require(first_rank <= 8u,
            "checkpoint first rank is invalid");
    require(static_cast<size_t>(first_rank) <=
                (semantic_bytes.size() - cursor) / sizeof(int32_t),
            "checkpoint first shape is truncated");
    cursor += static_cast<size_t>(first_rank) * sizeof(int32_t);
    const uint32_t first_payload_bytes =
        read_scalar<uint32_t>(semantic_bytes, cursor);
    cursor += sizeof(uint32_t);
    require(first_payload_bytes >= sizeof(float) &&
                first_payload_bytes <=
                    semantic_bytes.size() - cursor,
            "checkpoint first payload is invalid");
    write_scalar(&semantic_bytes, cursor,
                 std::numeric_limits<float>::quiet_NaN());

    const size_t trailer_offset =
        semantic_bytes.size() - integrity_trailer_bytes;
    const uint64_t recorded_payload_bytes =
        read_scalar<uint64_t>(semantic_bytes,
                              trailer_offset + sizeof(uint32_t));
    require(recorded_payload_bytes == trailer_offset,
            "checkpoint integrity payload length mismatch");
    const std::string semantic_hash =
        integrity::sha256_hex(semantic_bytes.data(), trailer_offset);
    require(semantic_hash.size() == 64u,
            "semantic SHA-256 length mismatch");
    std::memcpy(
        semantic_bytes.data() + trailer_offset +
            sizeof(uint32_t) + sizeof(uint64_t),
        semantic_hash.data(), semantic_hash.size());
    write_binary(semantic_path, semantic_bytes);

    JambaModel semantic_target(2, 32, 128, Device::CPU);
    const auto semantic_before =
        snapshot_parameters(semantic_target);
    bool semantic_rejected = false;
    try {
      semantic_target.load(semantic_path.string(), true);
    } catch (const std::exception&) {
      semantic_rejected = true;
    }
    require(semantic_rejected,
            "checkpoint containing NaN should be rejected");
    require_parameters_unchanged(
        semantic_target, semantic_before);

#ifdef NSOS_ENABLE_TEST_HOOKS
    std::cout << "[ModelPackTest] staging failure is transactional"
              << std::endl;
    JambaModel staging_failure_target(
        2, 32, 128, Device::CPU);
    const auto staging_failure_before =
        snapshot_parameters(staging_failure_target);
    bool staging_failure_rejected = false;
    {
      ModelCheckpointFaultReset fault_reset;
      testing::set_model_checkpoint_stage_failure_countdown(1);
      try {
        staging_failure_target.load(
            partial_path.string(), true);
      } catch (const std::bad_alloc&) {
        staging_failure_rejected = true;
      }
    }
    require(staging_failure_rejected,
            "injected checkpoint staging failure did not fire");
    require_parameters_unchanged(
        staging_failure_target, staging_failure_before);
#endif

    std::cout << "[ModelPackTest] non-finite save preserves destination"
              << std::endl;
    const std::filesystem::path nonfinite_save_path =
        std::filesystem::temp_directory_path() /
        "nsos_nonfinite_save_test.bin";
    semantic_target.save(nonfinite_save_path.string());
    const auto valid_destination =
        read_binary(nonfinite_save_path);
    auto semantic_parameters = semantic_target.parameters();
    require(!semantic_parameters.empty(),
            "semantic target has no parameters");
    semantic_parameters.front()->data.data()[0] =
        std::numeric_limits<float>::infinity();
    bool nonfinite_save_rejected = false;
    try {
      semantic_target.save(nonfinite_save_path.string());
    } catch (const std::exception&) {
      nonfinite_save_rejected = true;
    }
    require(nonfinite_save_rejected,
            "non-finite model save should be rejected");
    require(read_binary(nonfinite_save_path) == valid_destination,
            "rejected model save replaced the valid destination");

    std::cout << "[ModelPackTest] in-memory inference clone" << std::endl;
    const std::string source_generated = engine.generate("321", options);
    (void)source_generated;
    auto replica = engine.clone_for_inference();
    require(replica != nullptr, "clone_for_inference returned null");
    require(replica->config.vocab_size == engine.config.vocab_size, "replica vocab mismatch");
    require(replica->tokenizer.vocab_size == engine.tokenizer.vocab_size,
            "replica tokenizer mismatch");
    auto replica_params = replica->model->parameters();
    require(replica_params.size() == lhs_params.size(), "replica parameter count mismatch");
    require(std::abs(replica->model->parameters().front()->data.data()[0] -
                     engine.model->parameters().front()->data.data()[0]) < 1e-5f,
            "replica first parameter mismatch");
    const std::string replica_generated = replica->generate("321", options);
    (void)replica_generated;

    std::cout << "[ModelPackTest] transactional training clone" << std::endl;
    engine.trainer->dynamic_loss_scaling_enabled = true;
    engine.trainer->loss_scale = 2048.0f;
    engine.trainer->loss_scale_growth_tracker = 17;
    engine.trainer->last_optimizer_step_skipped = true;
    engine.trainer->last_auxiliary_stats.memory_count = 3;
    engine.model->set_training_rng_sequence(123456u);
    MemorySystem& source_memory =
        engine.trainer->auxiliary_memory_store(4, 7);
    source_memory.add_cluster(
        Tensor::ones({4}, Device::CPU), "clone-fixture");
    auto training_clone = engine.clone_for_training();
    require(training_clone != nullptr && training_clone->trainer != nullptr,
            "clone_for_training did not preserve trainer");
    require(training_clone->trainer->global_step_count == engine.trainer->global_step_count,
            "training clone scheduler step mismatch");
    require(training_clone->trainer->m_state.size() == engine.trainer->m_state.size() &&
                training_clone->trainer->v_state.size() == engine.trainer->v_state.size(),
            "training clone optimizer state mismatch");
    require(training_clone->trainer->loss_scale == 2048.0f &&
                training_clone->trainer->loss_scale_growth_tracker == 17 &&
                training_clone->trainer->last_optimizer_step_skipped,
            "training clone lost dynamic loss-scale state");
    require(training_clone->trainer->last_auxiliary_stats.memory_count == 3,
            "training clone lost auxiliary metrics");
    require(training_clone->model->training_rng_sequence() == 123456u,
            "training clone lost model RNG sequence");
    const auto cloned_memory =
        training_clone->trainer
            ->auxiliary_memory_store(4, 7)
            .snapshot_runtime_clusters();
    require(cloned_memory.size() == 1u,
            "training clone lost auxiliary-memory clusters");
    source_memory.add_cluster(
        Tensor::ones({4}, Device::CPU).mul(2.0f),
        "source-only");
    require(
        training_clone->trainer
                ->auxiliary_memory_store(4, 7)
                .snapshot_runtime_clusters()
                .size() == 1u,
        "training clone shares auxiliary-memory storage with source");
    const float original_first_weight = engine.model->parameters().front()->data.data()[0];
    training_clone->model->parameters().front()->data.data()[0] += 1.0f;
    require(engine.model->parameters().front()->data.data()[0] == original_first_weight,
            "training clone shares mutable parameter storage with source");
    engine.trainer->phase_scheduler.auxiliary_oxtamem_enabled = true;
    engine.trainer->phase_scheduler.auxiliary_oxtamem_store_path =
        "durable-clone-fixture";
    bool durable_clone_rejected = false;
    try {
      (void)engine.clone_for_training();
    } catch (const std::runtime_error&) {
      durable_clone_rejected = true;
    }
    engine.trainer->phase_scheduler.auxiliary_oxtamem_enabled = false;
    engine.trainer->phase_scheduler.auxiliary_oxtamem_store_path.clear();
    require(durable_clone_rejected,
            "transactional clone shared a durable OxtaMem arena");

    std::cout << "Model pack test passed!" << std::endl;
    std::filesystem::remove_all(pack_dir);
    std::filesystem::remove_all(qat_pack_dir);
    std::filesystem::remove_all(tampered_dir);
    std::filesystem::remove_all(traversal_dir);
    std::filesystem::remove_all(missing_digest_dir);
    std::filesystem::remove(partial_path);
    std::filesystem::remove(semantic_path);
    std::filesystem::remove(nonfinite_save_path);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Model pack test failed: " << ex.what() << std::endl;
    return 1;
  }
}
