#include "nsos_sdk.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace nsos;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
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
    require(std::filesystem::exists(pack_dir / "model.nsos.bin"), "missing weights");
    require(std::filesystem::exists(pack_dir / "tokenizer.nsos"), "missing tokenizer");
    require(std::filesystem::exists(pack_dir / "config.nsos"), "missing config");
    const std::string manifest_text = [&]() {
      std::ifstream input(pack_dir / "manifest.nsos");
      return std::string(std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>());
    }();
    require(manifest_text.find("sha256_weights=") != std::string::npos,
            "manifest missing sha256_weights");
    require(manifest_text.find("sha256_config=") != std::string::npos,
            "manifest missing sha256_config");
    require(manifest_text.find("format=nsos-pack-v2") != std::string::npos,
            "manifest missing format");
    require(manifest_text.find("version=2") != std::string::npos,
            "manifest missing version");

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
      std::fstream tampered_weights(tampered_dir / "model.nsos.bin",
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
      const std::string needle = "weights=model.nsos.bin";
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
    auto training_clone = engine.clone_for_training();
    require(training_clone != nullptr && training_clone->trainer != nullptr,
            "clone_for_training did not preserve trainer");
    require(training_clone->trainer->global_step_count == engine.trainer->global_step_count,
            "training clone scheduler step mismatch");
    require(training_clone->trainer->m_state.size() == engine.trainer->m_state.size() &&
                training_clone->trainer->v_state.size() == engine.trainer->v_state.size(),
            "training clone optimizer state mismatch");
    const float original_first_weight = engine.model->parameters().front()->data.data()[0];
    training_clone->model->parameters().front()->data.data()[0] += 1.0f;
    require(engine.model->parameters().front()->data.data()[0] == original_first_weight,
            "training clone shares mutable parameter storage with source");

    std::cout << "Model pack test passed!" << std::endl;
    std::filesystem::remove_all(pack_dir);
    std::filesystem::remove_all(tampered_dir);
    std::filesystem::remove_all(traversal_dir);
    std::filesystem::remove(partial_path);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Model pack test failed: " << ex.what() << std::endl;
    return 1;
  }
}
