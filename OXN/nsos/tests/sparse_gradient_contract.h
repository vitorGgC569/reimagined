#pragma once

#include "jamba.h"
#include "trainer.h"
#include "nsos/determinism.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

// Shared CPU/HIP/CUDA contract fixture. Routing is real, not a fabricated
// gradient registry: a fixed positive embedding and zero Mamba branch allow
// the gate's selected row to choose a different expert on every step.
namespace nsos::sparse_gradient_test {
inline void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
inline void same(const Tensor& a, const Tensor& b, const char* message) {
    require(a.shape == b.shape, message);
    const Tensor ah = a.cpu(), bh = b.cpu();
    require(ah.size == 0 || std::memcmp(ah.data(), bh.data(),
        static_cast<size_t>(ah.size) * sizeof(float)) == 0, message);
}
inline bool same_quant(const Quant4OptState& a, const Quant4OptState& b) {
    return a.n == b.n && a.quantized == b.quantized &&
        a.v_rank1 == b.v_rank1 && a.v_rows == b.v_rows && a.v_cols == b.v_cols &&
        a.m_codes == b.m_codes && a.m_absmax == b.m_absmax &&
        a.v_codes == b.v_codes && a.v_row == b.v_row && a.v_col == b.v_col &&
        a.v_absmax == b.v_absmax && a.m_fp32 == b.m_fp32 && a.v_fp32 == b.v_fp32;
}

inline void parameter_contract(Device device) {
    Parameter late(Tensor::ones({3, 5}, device), "late.sparse.registration");
    late.add_grad(Tensor::ones({3, 5}, device));
    late.zero_grad();
    Parameter late_copy(late.data.clone(), "late.sparse.clone");
    late_copy.grad = late.grad.clone();
    late_copy.copy_gradient_activity_from(late);
    late_copy.track_gradient_contributions();
    require(!late_copy.has_gradient(), "late sparse registration revived a cleared dense clone");
    late.track_gradient_contributions();
    require(!late.has_gradient() && late.grad.size == 15,
            "late sparse registration revived a cleared legacy buffer");
    late.add_grad(Tensor::zeros({3, 5}, device));
    require(late.has_gradient(), "late registration lost a new zero contribution");
    Parameter p(Tensor::ones({3, 5}, device), "sparse.test");
    p.track_gradient_contributions();
    require(!p.has_gradient(), "new sparse parameter is active");
    p.add_grad(Tensor::ones({3, 5}, device));
    const float* address = p.grad.raw_data();
    require(p.has_gradient(), "first contribution was not published");
    p.zero_grad();
    require(!p.has_gradient() && p.grad.size == 15 && p.grad.raw_data() == address,
            "zeroing conflates inactive status and stable storage");
    p.add_grad(Tensor::zeros({3, 5}, device));
    require(p.has_gradient(), "a contributed zero gradient became absent");
    p.add_grad(Tensor::ones({3, 5}, device));
    same(p.grad, Tensor::ones({3, 5}, device), "second contribution was lost");
    p.zero_grad();
    p.add_grad(Tensor::ones({3, 5}, device).mul(2));
    require(p.grad.raw_data() == address, "reactivation replaced gradient storage");
    same(p.grad, Tensor::ones({3, 5}, device).mul(2), "old group leaked into reactivation");
    Parameter copy(p.data.clone(), "sparse.copy");
    copy.grad = p.grad.clone();
    copy.copy_gradient_activity_from(p);
    require(copy.has_gradient() && copy.tracks_gradient_contributions(),
            "runtime clone lost active contribution status");
    p.zero_grad();
    copy.grad = p.grad.clone();
    copy.copy_gradient_activity_from(p);
    require(!copy.has_gradient() && copy.grad.size == 15,
            "runtime clone revived inactive storage");
    p.grad.copy_from(Tensor::ones({3, 5}, device));
    require(!p.has_gradient(), "unpublished direct sparse write became active");
    p.mark_gradient_contribution();
    require(p.has_gradient(), "explicit external contribution was ignored");
    const Tensor before = p.grad.clone();
    bool rejected = false;
    try { p.add_grad(Tensor::ones({2, 5}, device)); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && p.has_gradient(), "invalid contribution altered activity");
    same(p.grad, before, "invalid contribution changed existing gradient");
}

inline void select(JambaModel& model, int expert) {
    auto& gate = *model.layers.front()->router->gate;
    gate.set_exact_linear_mode(true);
    Tensor weights = Tensor::zeros(gate.weight.data.shape.dims, Device::CPU);
    const int D = model.model_config().d_model;
    for (int d = 0; d < D; ++d) weights.data()[expert * D + d] = 1.0f;
    gate.weight.copy_data_from(weights.to(gate.weight.data.get_device()));
}
inline std::vector<Parameter*> expert_parameters(JambaModel& model, int e) {
    auto result = model.layers.front()->expert_gate_up[e]->parameters();
    const auto down = model.layers.front()->expert_down[e]->parameters();
    result.insert(result.end(), down.begin(), down.end());
    std::vector<Parameter*> active;
    for (auto* p : result) if (p->trainable) active.push_back(p);
    return active;
}
struct Snapshot {
    Parameter* p;
    Tensor weight, m, v;
    Quant4OptState quant;
    uint64_t version;
    const float* gradient_address;
};
inline std::vector<Snapshot> snapshot(Trainer& trainer, JambaModel& model, int expert) {
    std::vector<Snapshot> result;
    for (auto* p : expert_parameters(model, expert)) {
        require(p->has_gradient(), "selected expert did not contribute");
        Snapshot s{p, p->data.clone(), {}, {}, {}, p->version, p->grad.raw_data()};
        if (trainer.optimizer_state_bits == 4 && p->data.get_device() == Device::CPU) {
            s.quant = trainer.quant_state.at(p);
        } else {
            s.m = trainer.m_state.at(p).clone(); s.v = trainer.v_state.at(p).clone();
        }
        result.push_back(std::move(s));
    }
    return result;
}
inline void unchanged(Trainer& trainer, const std::vector<Snapshot>& snapshots) {
    for (const auto& s : snapshots) {
        require(!s.p->has_gradient() && s.p->grad.size > 0,
                "previously selected inactive expert retained gradient activity");
        require(s.p->grad.raw_data() == s.gradient_address, "inactive expert lost reusable storage");
        require(s.p->version == s.version, "inactive expert version advanced");
        same(s.p->data, s.weight, "inactive expert weight decayed or Adam-updated");
        if (trainer.optimizer_state_bits == 4 && s.p->data.get_device() == Device::CPU) {
            require(same_quant(trainer.quant_state.at(s.p), s.quant), "inactive quantized moments changed");
        } else {
            same(trainer.m_state.at(s.p), s.m, "inactive first moment decayed");
            same(trainer.v_state.at(s.p), s.v, "inactive second moment decayed");
        }
    }
}

inline void routed_optimizer_contract(Device device, int state_bits = 32) {
    ModelConfig cfg;
    cfg.num_layers = 1; cfg.d_model = 64; cfg.vocab_size = 67;
    cfg.n_heads = 4; cfg.n_kv_heads = 2; cfg.attention_period = 64;
    cfg.use_moe = true; cfg.moe_period = 1; cfg.moe_slot = 0;
    cfg.num_experts = 4; cfg.num_experts_per_token = 1; cfg.moe_expert_hidden_dim = 128;
    cfg.use_ttt = false; cfg.use_chrass = false; cfg.dropout = 0;
    cfg.mamba_d_state = 8; cfg.mamba_head_dim = 32; cfg.tie_word_embeddings = false;
    JambaModel model(cfg, device);
    require(model.layers.front()->router != nullptr, "sparse fixture has no MoE");
    model.embedding->weight.copy_data_from(Tensor::ones({cfg.vocab_size, cfg.d_model}, device));
    for (auto* p : model.layers.front()->mamba_layer->parameters())
        p->copy_data_from(Tensor::zeros(p->data.shape.dims, device));
    Trainer trainer(&model, 2e-4f);
    trainer.optimizer_state_bits = state_bits;
    trainer.weight_decay = 0.03f; trainer.max_grad_norm = 1e9f;
    trainer.warmup_steps = 0; trainer.total_training_steps = 100;
    trainer.phase_scheduler.progressive_qat_enabled = false;
    trainer.moe_aux_loss_scale = 0;
    std::vector<int> tokens(33, 1), targets(33, 2);
    select(model, 0);
    require(std::isfinite(trainer.train_step(tokens, targets)), "first sparse step is nonfinite");
    const auto first = snapshot(trainer, model, 0);
    for (auto* p : expert_parameters(model, 2)) {
        require(!p->has_gradient() && trainer.m_state.count(p) == 0 &&
                trainer.v_state.count(p) == 0 && trainer.quant_state.count(p) == 0,
                "never-selected expert allocated Adam state");
    }
    select(model, 1);
    require(std::isfinite(trainer.train_step(tokens, targets)), "second sparse step is nonfinite");
    unchanged(trainer, first);
    const auto second = snapshot(trainer, model, 1);

    // Exercise the public portable-snapshot boundary, which invokes the
    // private transactional clone without exposing it as a production API.
    const auto checkpoint = trainer.capture_checkpoint_snapshot();
    require(checkpoint && checkpoint->global_step() == trainer.global_step_count,
            "sparse checkpoint snapshot captured the wrong step");
    unchanged(trainer, first);
    for (const auto& s : second)
        require(s.p->has_gradient(), "checkpoint snapshot consumed active gradients");

    select(model, 0);
    require(std::isfinite(trainer.train_step(tokens, targets)), "reactivated step is nonfinite");
    unchanged(trainer, second);
    for (const auto& s : first) {
        require(s.p->has_gradient() && s.p->version == s.version + 1 &&
                s.p->grad.raw_data() == s.gradient_address, "reactivation broke version/storage contract");
        if (state_bits == 32) {
            const Tensor old_m = s.m.cpu(), old_v = s.v.cpu(), g = s.p->grad.cpu();
            const Tensor m = trainer.m_state.at(s.p).cpu(), v = trainer.v_state.at(s.p).cpu();
            for (int i = 0; i < g.size; ++i) {
                const float expected_m = trainer.beta1 * old_m.data()[i] + (1 - trainer.beta1) * g.data()[i];
                const float expected_v = trainer.beta2 * old_v.data()[i] + (1 - trainer.beta2) * g.data()[i] * g.data()[i];
                require(std::abs(m.data()[i] - expected_m) <= 2e-7f * (1 + std::abs(expected_m)),
                        "reactivation did not resume preserved first moment");
                require(std::abs(v.data()[i] - expected_v) <= 2e-7f * (1 + std::abs(expected_v)),
                        "reactivation did not resume preserved second moment");
            }
        }
    }
    // A real zero contribution is not an absent gradient: it still decays the
    // existing moments and applies Adam/weight decay. This independently tests
    // the activity predicate used by every optimizer cohort.
    const auto before_zero = snapshot(trainer, model, 0);
    select(model, 1);
    trainer.accumulate_microbatch(tokens, targets);
    for (const auto& s : before_zero)
        s.p->add_grad(Tensor::zeros(s.p->data.shape.dims, device));
    trainer.commit_optimizer_step(1);
    for (const auto& s : before_zero) {
        require(s.p->has_gradient() && s.p->version == s.version + 1,
                "contributed zero expert was skipped by optimizer");
        if (state_bits == 32) {
            const Tensor old_m = s.m.cpu(), old_v = s.v.cpu();
            const Tensor m = trainer.m_state.at(s.p).cpu(), v = trainer.v_state.at(s.p).cpu();
            for (int i = 0; i < m.size; ++i) {
                require(m.data()[i] == trainer.beta1 * old_m.data()[i],
                        "contributed zero did not decay first moment");
                require(v.data()[i] == trainer.beta2 * old_v.data()[i],
                        "contributed zero did not decay second moment");
            }
        }
    }
    // Union over two real routed microbatches, then abort: no buffer/activity
    // reset is permitted between them, and neither abort nor routing publishes
    // weights, versions or optimizer moments.
    const auto step_before_abort = trainer.global_step_count;
    select(model, 0);
    trainer.accumulate_microbatch(tokens, targets);
    select(model, 1);
    trainer.accumulate_microbatch(tokens, targets);
    for (int e : {0, 1}) for (auto* p : expert_parameters(model, e))
        require(p->has_gradient(), "microbatch union lost an expert contribution");
    trainer.abort_gradient_accumulation();
    require(trainer.global_step_count == step_before_abort, "abort advanced optimizer step");
    for (int e : {0, 1, 2, 3}) for (auto* p : expert_parameters(model, e))
        require(!p->has_gradient(), "abort left sparse activity published");
    std::cout << "[sparse-gradient] routed active/inactive/reactivation/union/abort device="
              << static_cast<int>(device) << " state_bits=" << state_bits << std::endl;
}
} // namespace nsos::sparse_gradient_test
