#include "jamba.h"
#include "trainer.h"
#include "verified_reasoning_fixture.h"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace nsos;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid reasoning/auxiliary contract accepted");
}

void verified_search(Device device) {
    Tensor root = Tensor::zeros({256}, device);
    auto policy = arithmetic_reasoning_fixture();
    MCTSConfig config;
    config.num_simulations = 8;
    config.max_nodes = 8;
    auto result = run_verified_reasoning(root, policy, config);
    require(result.report.best_score == 1 && result.report.baseline_score == 0,
            "prior/confidence won over exact correctness");
    require(result.report.evaluated_states == 3, "candidate reverified or budget inflated");
    Tensor host = result.state.get_device() == Device::GPU ? result.state.cpu() : result.state;
    require(host.data()[0] == 4, "wrong structured answer selected");
    config.num_simulations = 1;
    result = run_verified_reasoning(root, policy, config);
    require(result.report.evaluated_states == 1 && result.report.proposal_calls == 0,
            "root-only budget exceeded");
    config.num_simulations = 8;
    policy.verify = [](const std::vector<Tensor>&) { return std::vector<ReasoningVerification>{}; };
    rejects([&] { run_verified_reasoning(root, policy, config); });
    policy = arithmetic_reasoning_fixture();
    policy.verify = [](const std::vector<Tensor>& states) {
        return std::vector<ReasoningVerification>(states.size(), {0.5f, ""});
    };
    rejects([&] { run_verified_reasoning(root, policy, config); });
    policy = arithmetic_reasoning_fixture();
    policy.propose = [](const Tensor& state, int, int) {
        return std::vector<ReasoningProposal>{{state.clone(), -1.0f}};
    };
    rejects([&] { run_verified_reasoning(root, policy, config); });
    for (float score : {-1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN()}) {
        policy.verify = [score](const std::vector<Tensor>& states) {
            return std::vector<ReasoningVerification>(states.size(), {score, "test"});
        };
        rejects([&] { run_verified_reasoning(root, policy, config); });
    }
    policy = arithmetic_reasoning_fixture();
    policy.propose = [](const Tensor& state, int, int) {
        return std::vector<ReasoningProposal>{{state.clone(), 1.0f}};
    };
    result = run_verified_reasoning(root, policy, config);
    require(result.report.evaluated_states == 1, "cycle was evaluated again");
    // Tail-only differences must survive dedup (old code hashed only 128 floats).
    policy.propose = [](const Tensor& state, int, int) {
        Tensor host = state.get_device() == Device::GPU ? state.cpu() : state.clone();
        host.data()[255] = 1;
        return std::vector<ReasoningProposal>{{state.get_device() == Device::GPU ? host.to(Device::GPU) : host, 1}};
    };
    result = run_verified_reasoning(root, policy, config);
    require(result.report.evaluated_states == 2, "tail-only candidate collapsed into root");
    policy.propose = [](const Tensor& state, int, int limit) {
        return std::vector<ReasoningProposal>(limit + 1, {state, 1});
    };
    rejects([&] { run_verified_reasoning(root, policy, config); });
    policy.propose = [](const Tensor& state, int, int) {
        return std::vector<ReasoningProposal>{{Tensor::zeros({2}, state.get_device()), 1}};
    };
    rejects([&] { run_verified_reasoning(root, policy, config); });
}

void model_and_training_contracts(Device device) {
    ModelConfig config;
    config.num_layers = 1; config.d_model = 16; config.vocab_size = 32;
    config.n_heads = 2; config.n_kv_heads = 1;
    config.use_ttt = true; config.ttt_period = 1; config.ttt_slot = 0;
    config.mamba_head_dim = 16;
    JambaModel model(config, device);
    Tensor state = Tensor::zeros({16}, device);
    rejects([&] { model.reason(state, 8); });
    model.set_reasoning_policy(arithmetic_reasoning_fixture());
    auto result = model.reason(state, 8);
    require(model.last_reasoning_report().best_score == 1, "model lost verifier report");
    require(model.run_reasoning_loop(state, 2).shape == state.shape,
            "reasoning loop projected hidden state into vocabulary");
    require(model.last_reasoning_report().best_score == 1, "loop returned an unverified state");
    rejects([&] { model.reason(Tensor::zeros({32}, device), 8); });
    Trainer trainer(&model, 1e-4f);
    trainer.phase_scheduler.progressive_qat_enabled = false;
    auto& schedule = trainer.phase_scheduler;
    schedule.auxiliary_stack_enabled = true;
    schedule.auxiliary_reasoning_enabled = true;
    const std::vector<std::vector<int>> prompts{{1, 2, 3}};
    const std::vector<std::vector<int>> answers{{4, 5}};
    rejects([&] { trainer.train_supervised_batch(prompts, answers); });
    require(trainer.global_step_count == 0, "rejected stack mutated optimizer step");
    schedule.auxiliary_session_adapt_enabled = true;
    model.clear_reasoning_policy();
    rejects([&] { trainer.train_supervised_batch(prompts, answers); });
    model.set_reasoning_policy(arithmetic_reasoning_fixture());
    const auto loss = trainer.train_supervised_batch(prompts, answers);
    require(std::isfinite(loss), "consumed auxiliary stack returned invalid loss");
    require(trainer.last_auxiliary_stats.session_adapt_count == 1,
            "auxiliary result not consumed");
    require(trainer.last_auxiliary_stats.answer_tokens == 0,
            "unused answer diagnostic forward still active");
    schedule.auxiliary_memory_enabled = true;
    schedule.auxiliary_memory_blend = 0;
    rejects([&] { trainer.train_supervised_batch(prompts, answers); });
    config.use_ttt = false;
    JambaModel no_ttt(config, device);
    Trainer orphan(&no_ttt, 1e-4f);
    orphan.phase_scheduler.auxiliary_stack_enabled = true;
    orphan.phase_scheduler.auxiliary_session_adapt_enabled = true;
    rejects([&] { orphan.train_supervised_batch(prompts, answers); });
    require(orphan.global_step_count == 0, "absent TTT consumer mutated optimizer");
}

int main(int argc, char**) {
    try {
        const Device device = argc > 1 ? Device::GPU : Device::CPU;
        verified_search(device);
        model_and_training_contracts(device);
        std::cout << "Verified reasoning and auxiliary-consumer contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
