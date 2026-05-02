#include "layer_audit.h"
#include "nsos_sdk.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nsos;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_tensor_close(const Tensor& lhs, const Tensor& rhs, float tolerance) {
    Tensor lhs_cpu = lhs.cpu();
    Tensor rhs_cpu = rhs.cpu();
    require(lhs_cpu.shape == rhs_cpu.shape, "tensor shape mismatch");
    for (int i = 0; i < lhs_cpu.size; ++i) {
        const float diff = std::abs(lhs_cpu.data()[i] - rhs_cpu.data()[i]);
        if (diff > tolerance) {
            throw std::runtime_error("tensor mismatch at index " + std::to_string(i));
        }
    }
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.is_open(), "failed to open file: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::filesystem::path unique_temp_dir(const std::string& name) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (name + "_" + std::to_string(stamp));
}

} // namespace

int main() {
    try {
        ModelConfig config;
        config.num_layers = 3;
        config.d_model = 32;
        config.vocab_size = 96;
        config.n_heads = 4;
        config.n_kv_heads = 2;
        config.max_context_tokens = 64;
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

        InferenceEngine engine;
        require(engine.load_model("", config), "initial load_model failed");

        LayerAuditCollector audit;
        audit.begin_run("layer_audit_e2e");
        audit.set_enabled(true);
        engine.model->set_audit_collector(&audit);

        audit.set_phase("train");
        engine.trainer->weight_decay = 0.0f;
        engine.trainer->max_grad_norm = 2.0f;
        const float loss = engine.trainer->train_supervised_batch(
            {{1, 2, 3, 4}, {5, 6, 7}},
            {{8, 9}, {10, 11, 12}});
        require(std::isfinite(loss), "non-finite training loss");

        LayerAuditSummary train_summary = audit.summarize_phase("train");
        require(train_summary.forward_records >= static_cast<size_t>(config.num_layers),
                "missing training forward layer records");
        require(train_summary.backward_records >= static_cast<size_t>(config.num_layers),
                "missing training backward layer records");
        require(train_summary.router_records > 0, "missing MoE router audit records");
        require(train_summary.training_steps == 1, "missing training step audit record");
        require(train_summary.healthy(), "training audit contains NaN/Inf");

        const std::vector<int> probe = {1, 2, 3, 4, 5};
        audit.set_phase("pre_reload_probe");
        engine.model->set_training_mode(false);
        engine.model->reset_session();
        Tensor before = engine.model->forward_ids(probe, nullptr).cpu();
        LayerAuditSummary pre_summary = audit.summarize_phase("pre_reload_probe");
        require(pre_summary.forward_records >= static_cast<size_t>(config.num_layers),
                "missing pre-reload forward records");
        require(pre_summary.token_contexts > 0, "missing pre-reload token context");
        require(pre_summary.healthy(), "pre-reload audit contains NaN/Inf");

        const std::filesystem::path pack_dir = unique_temp_dir("nsos_layer_audit_pack");
        const std::filesystem::path audit_json = unique_temp_dir("nsos_layer_audit") /
                                                "audit.json";
        require(engine.save_model_pack(pack_dir.string()), "save_model_pack failed");

        InferenceEngine reloaded;
        require(reloaded.load_model(pack_dir.string(), ModelConfig{}),
                "reload model pack failed");
        reloaded.model->set_audit_collector(&audit);

        audit.set_phase("post_reload_probe");
        reloaded.model->set_training_mode(false);
        reloaded.model->reset_session();
        Tensor after = reloaded.model->forward_ids(probe, nullptr).cpu();
        require_tensor_close(before, after, 1e-5f);

        std::string compare_reason;
        require(audit.compare_phase_health("pre_reload_probe",
                                           "post_reload_probe",
                                           &compare_reason),
                compare_reason.empty() ? "phase health comparison failed" : compare_reason);

        audit.write_json(audit_json.string());
        require(std::filesystem::exists(audit_json), "audit JSON was not written");
        const std::string json = read_file(audit_json);
        require(json.find("\"records\"") != std::string::npos, "audit JSON missing records");
        require(json.find("\"token_contexts\"") != std::string::npos,
                "audit JSON missing token contexts");
        require(json.find("\"training_steps\"") != std::string::npos,
                "audit JSON missing training steps");
        require(json.find("\"router\"") != std::string::npos,
                "audit JSON missing router details");

        std::filesystem::remove_all(pack_dir);
        std::filesystem::remove_all(audit_json.parent_path());

        std::cout << "Layer audit E2E test passed!" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Layer audit E2E test failed: " << ex.what() << std::endl;
        return 1;
    }
}
