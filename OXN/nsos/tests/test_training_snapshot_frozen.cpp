#include "jamba.h"
#include "trainer.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nsos;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
ModelConfig config() {
    ModelConfig c;
    c.num_layers = 1; c.d_model = 16; c.vocab_size = 32;
    c.n_heads = 2; c.n_kv_heads = 1;
    c.attention_period = 64; c.force_mamba_last_layer = true;
    c.mamba2_faithful = false; c.mamba_head_dim = 8;
    c.use_moe = true; c.moe_period = 1; c.moe_slot = 0;
    c.num_experts = 4; c.num_experts_per_token = 1; c.moe_expert_hidden_dim = 32;
    c.use_ttt = false; c.use_kan = false; c.use_chrass = false;
    c.dropout = 0; c.tie_word_embeddings = false;
    return c;
}
void policy(JambaModel& model, bool freeze_router) {
    model.set_reference_path(true);
    auto& gate = *model.layers.at(0)->router->gate;
    gate.set_exact_linear_mode(true); // also freezes the unused magnitude
    gate.weight.trainable = !freeze_router;
}
void setup(Trainer& trainer) {
    trainer.phase_scheduler.progressive_qat_enabled = false;
    trainer.phase_scheduler.ternary_regularization = 0;
    trainer.dynamic_loss_scaling_enabled = false;
    trainer.optimizer_state_bits = 32;
    trainer.moe_aux_loss_scale = 0;
}
void same_registry(JambaModel& left, JambaModel& right) {
    const auto a = left.parameters(), b = right.parameters();
    require(a.size() == b.size(), "parameter registry count differs");
    for (size_t i = 0; i < a.size(); ++i) {
        require(a[i]->name == b[i]->name, "parameter registry name differs");
        require(a[i]->trainable == b[i]->trainable,
                "clone did not preserve trainable: " + a[i]->name);
        const auto x = a[i]->data.cpu(), y = b[i]->data.cpu();
        require(x.shape == y.shape && std::memcmp(x.data(), y.data(),
                static_cast<size_t>(x.size) * sizeof(float)) == 0,
                "cloned checkpoint weight differs: " + a[i]->name);
    }
}
void snapshot_roundtrip(JambaModel& source, Trainer& trainer) {
    const auto stem = std::filesystem::temp_directory_path() /
        ("nsos-snapshot-frozen-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto model_file = stem.string() + ".model";
    const auto state_file = stem.string() + ".state";
    struct Files {
        std::string model, state;
        ~Files() { std::error_code ec; std::filesystem::remove(model, ec); std::filesystem::remove(state, ec); }
    } files{model_file, state_file};
    auto snapshot = trainer.capture_checkpoint_snapshot();
    // The captured freeze policy must be independent of later live metadata.
    source.layers[0]->router->gate->weight.trainable = true;
    snapshot->write(model_file, state_file);
    source.layers[0]->router->gate->weight.trainable = false;
    JambaModel resumed(config(), Device::CPU);
    policy(resumed, true);
    resumed.load(model_file);
    require(!resumed.layers[0]->router->gate->weight.trainable,
            "model load changed the caller's router freeze policy");
    Trainer restored(&resumed, .002f); setup(restored);
    restored.load_training_state(state_file, model_file);
    same_registry(source, resumed);
    require(trainer.global_step_count == restored.global_step_count &&
            trainer.tokens_committed == restored.tokens_committed &&
            trainer.m_state.empty() && restored.m_state.empty() &&
            trainer.v_state.empty() && restored.v_state.empty(),
            "frozen cold snapshot changed step/token/lazy moment state");
    std::cout << "PASS portable snapshot frozen cohort and sidecar load\n";
}
} // namespace
int main() {
    try {
        JambaModel source(config(), Device::CPU);
        policy(source, true);
        Trainer trainer(&source, .002f); setup(trainer);
        snapshot_roundtrip(source, trainer);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL frozen snapshot regression: " << e.what() << '\n'; return 1;
    }
}
