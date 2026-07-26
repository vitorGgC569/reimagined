// Model-pack lifecycle contract for automatic CUDA DP4A inference.
//
// A QAT-ready pack is authored on CPU, loaded with the GPU runtime override,
// cloned for serving, and executed through the packed kernel on both instances.
// This catches the historical gap where packed state survived load/clone but
// gpu_packed_inference_enabled did not.

#include "gpu_parity_common.h"
#include "nsos_sdk.h"

#include <filesystem>
#include <stdexcept>
#include <system_error>

using nsos::BitLinear;
using nsos::Device;
using nsos::InferenceEngine;
using nsos::ModelConfig;
using nsos::Tensor;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

size_t find_dp4a_layer(const std::vector<BitLinear*>& layers) {
  for (size_t index = 0; index < layers.size(); ++index) {
    const BitLinear* layer = layers[index];
    if (layer && !layer->quantization_sensitive() &&
        !layer->has_full_precision_weight() &&
        layer->input_features() % 16 == 0) {
      return index;
    }
  }
  throw std::runtime_error(
      "loaded pack has no released DP4A-eligible BitLinear layer");
}

Tensor execute_dp4a(BitLinear* layer, int seed_offset) {
  require(layer != nullptr, "null DP4A layer");
  require(layer->gpu_packed_inference_enabled(),
          "automatic DP4A flag is disabled");
  layer->reset_gpu_packed_dispatch_count();

  Tensor input_cpu({2, layer->input_features()}, Device::CPU);
  for (int index = 0; index < input_cpu.size; ++index) {
    input_cpu.data()[index] =
        static_cast<float>(((index + seed_offset) % 23) - 11) / 13.0f;
  }
  Tensor output = layer->forward(input_cpu.to(Device::GPU));
  cuda_sync_or_throw("model_pack_dp4a/forward");
  require(layer->gpu_packed_dispatch_count() == 1,
          "forward did not dispatch exactly once through packed DP4A");
  return output.cpu();
}

} // namespace

int main() {
  return run_parity("model_pack_dp4a_lifecycle", [] {
    ModelConfig authoring_config;
    authoring_config.num_layers = 1;
    authoring_config.d_model = 32;
    authoring_config.vocab_size = 64;
    authoring_config.n_heads = 4;
    authoring_config.n_kv_heads = 2;
    authoring_config.attention_period = 64;
    authoring_config.attention_slot = 63;
    authoring_config.mamba_d_state = 8;
    authoring_config.mamba_head_dim = 8;
    authoring_config.use_moe = false;
    authoring_config.use_ttt = false;
    authoring_config.use_chrass = false;
    authoring_config.use_kan = false;
    authoring_config.use_exact_attention_training = false;
    authoring_config.tie_word_embeddings = false;
    authoring_config.dropout = 0.0f;

    InferenceEngine author;
    require(author.load_model("", authoring_config),
            "failed to create authoring model");
    require(author.trainer != nullptr, "authoring trainer is missing");
    author.trainer->phase_scheduler.progressive_qat_enabled = true;
    author.trainer->global_step_count = 1;
    author.model->set_reference_path(false);
    for (BitLinear* layer : author.model->collect_bitlinear_layers()) {
      if (layer && !layer->quantization_sensitive()) {
        layer->repack_weights();
      }
    }

    const auto pack_dir = std::filesystem::temp_directory_path() /
                          "nsos_gpu_dp4a_lifecycle_pack";
    std::error_code cleanup_error;
    std::filesystem::remove_all(pack_dir, cleanup_error);
    require(author.save_model_pack(pack_dir.string()),
            "failed to save QAT-ready model pack");

    ModelConfig gpu_override;
    gpu_override.use_cuda = true;
    InferenceEngine loaded;
    require(loaded.load_model(pack_dir.string(), gpu_override),
            "failed to load model pack on GPU");

    const auto loaded_layers = loaded.model->collect_bitlinear_layers();
    require(!loaded_layers.empty(), "loaded model has no BitLinear layers");
    for (const BitLinear* layer : loaded_layers) {
      if (layer && !layer->has_full_precision_weight() &&
          !layer->quantization_sensitive()) {
        require(layer->gpu_packed_inference_enabled(),
                "GPU pack load did not propagate automatic DP4A");
      }
    }

    auto clone = loaded.clone_for_inference();
    require(clone != nullptr && clone->model != nullptr,
            "inference clone was not created");
    const auto clone_layers = clone->model->collect_bitlinear_layers();
    require(clone_layers.size() == loaded_layers.size(),
            "clone BitLinear layout differs from source");
    for (size_t index = 0; index < loaded_layers.size(); ++index) {
      require(clone_layers[index]->gpu_packed_inference_enabled() ==
                  loaded_layers[index]->gpu_packed_inference_enabled(),
              "clone lost the DP4A enablement state");
    }

    const size_t dp4a_index = find_dp4a_layer(loaded_layers);
    Tensor loaded_output = execute_dp4a(loaded_layers[dp4a_index], 0);
    Tensor clone_output = execute_dp4a(clone_layers[dp4a_index], 0);
    assert_close(loaded_output, clone_output, 1e-6f,
                 "model_pack_dp4a_loaded_vs_clone");
    std::cout << "[GPUParity:model_pack_dp4a_lifecycle] layer="
              << dp4a_index << " load_dispatches="
              << loaded_layers[dp4a_index]->gpu_packed_dispatch_count()
              << " clone_dispatches="
              << clone_layers[dp4a_index]->gpu_packed_dispatch_count()
              << std::endl;

    std::filesystem::remove_all(pack_dir, cleanup_error);
  });
}
