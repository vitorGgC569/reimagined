// Prepared fixture for root's FINAL combined binary, after Silex Trainer release.
// No successful execution is certified by the existence of this source file.
#include "gpu_parity_common.h"
#include "gpu_attention_training.h"
#include "gpu_sparse_adam.h"
#include "gpu_moe_training.h"
#include "gpu_execution.h"
#include "jamba.h"
#include "trainer.h"
#include "training_runtime_policy.h"
#include "nsos/determinism.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

using namespace nsos;
namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
struct Policy {
    std::string key, old;
    bool present;
    Policy(const char* name, const char* value) : key(name) {
#ifdef _WIN32
        char* prior = nullptr; size_t length = 0;
        require(_dupenv_s(&prior, &length, name) == 0, "cannot read " + key);
#else
        const char* prior = std::getenv(name);
#endif
        present = prior != nullptr;
        if (prior) old = prior;
#ifdef _WIN32
        std::free(prior);
        require(_putenv_s(name, value) == 0, "cannot set " + key);
#else
        require(setenv(name, value, 1) == 0, "cannot set " + key);
#endif
    }
    ~Policy() {
#ifdef _WIN32
        (void)_putenv_s(key.c_str(), present ? old.c_str() : "");
#else
        if (present) (void)setenv(key.c_str(), old.c_str(), 1);
        else (void)unsetenv(key.c_str());
#endif
    }
};
ModelConfig config() {
    ModelConfig c;
    c.architecture_schema_version = 3;
    c.num_layers = 1; c.d_model = 16; c.vocab_size = 257;
    c.n_heads = 2; c.n_kv_heads = 1;
    c.attention_period = 1; c.attention_slot = 0;
    c.force_mamba_last_layer = false;
    c.hybrid_composition = HybridComposition::ParallelGated;
    c.hybrid_mamba_gate_init = 1; c.hybrid_attention_gate_init = .1f;
    c.hybrid_ffn_gate_init = 1;
    c.faithful_attention_linears = true;
    c.mamba3_enabled = true; c.mamba3_schema_version = 1;
    c.mamba3_state_dim = 128; c.mamba3_mimo = true;
    c.mamba3_mimo_rank = 2; c.mamba3_outproj_norm = true;
    c.mamba_expand = 1; c.mamba_head_dim = 8; c.mamba_n_groups = 1;
    c.use_moe = true; c.moe_period = 1; c.moe_slot = 0;
    c.num_experts = 4; c.num_experts_per_token = 1;
    c.moe_expert_hidden_dim = 32;
    c.use_ttt = false; c.use_kan = false; c.use_chrass = false;
    c.dropout = 0; c.use_gradient_checkpointing = false;
    c.tie_word_embeddings = false; c.max_context_tokens = 64;
    return c;
}
std::vector<int> tokens(int count, int a, int b) {
    std::vector<int> result(count);
    for (int i = 0; i < count; ++i) result[i] = i % 2 ? b : a;
    return result;
}
const auto input0 = tokens(33, 1, 5), input1 = tokens(17, 2, 6);
const std::vector<int> targets0(33, 3), targets1(17, 4);
void prepare(JambaModel& model) {
    model.set_reference_path(true); model.set_training_mode(true);
    auto& block = *model.layers.at(0);
    require(block.attn_layer && block.mamba3_layer && block.router &&
            block.expert_gate_up.size() == 4, "hybrid fixture topology missing");
    require(!block.mamba_layer, "fixture unexpectedly selected Mamba2");
    // Stable, sign-separated routing without rewriting weights mid-group.
    // Both branches execute; the residual preserves the token's signed feature.
    Tensor embedding = Tensor::ones({257, 16}, Device::CPU);
    for (int token = 0; token < 257; ++token) for (int j = 1; j < 16; ++j)
        embedding.data()[token * 16 + j] += .3f * std::sin(.11f * (token+1) * (j+1));
    for (int token : {1,5}) embedding.data()[token*16] = 4;
    for (int token : {2,6}) embedding.data()[token*16] = -4;
    model.embedding->weight.copy_data_from(embedding.to(Device::GPU));
    auto& gate = *block.router->gate;
    gate.set_exact_linear_mode(true);
    Tensor routing = Tensor::zeros(gate.weight.data.shape.dims, Device::CPU);
    routing.data()[0] = 1; routing.data()[16] = -1;
    gate.weight.copy_data_from(routing.to(Device::GPU));
    gate.weight.trainable = false; // freezes only the diagnostic routing fixture
}
void setup(Trainer& t) {
    t.phase_scheduler.progressive_qat_enabled = false;
    t.phase_scheduler.ternary_regularization = 0;
    t.moe_aux_loss_scale = 0; t.logit_l2_beta = 0; t.pantheon_vib_beta = 0;
    t.repetition_unlikelihood_scale = 0;
    t.dynamic_loss_scaling_enabled = false; t.loss_scale = 1;
    t.optimizer_state_bits = 32; t.gradient_accumulation_steps = 2;
    t.weight_decay = .25f; t.max_grad_norm = .05f;
    t.warmup_steps = 0; t.total_training_steps = 100;
    t.min_learning_rate_scale = 1; // constant absolute LR for independent decay oracle
}
struct Weight {
    std::string name;
    Tensor data;
    std::uint64_t version;
};
std::vector<Weight> weights(JambaModel& m) {
    std::vector<Weight> result;
    for (auto* p : m.parameters()) result.push_back({p->name, p->data.cpu().clone(), p->version});
    return result;
}
bool equal(const Tensor& a, const Tensor& b) {
    auto x = a.cpu(), y = b.cpu();
    return x.shape == y.shape &&
        std::memcmp(x.data(), y.data(), static_cast<size_t>(x.size) * sizeof(float)) == 0;
}
void same(JambaModel& m, const std::vector<Weight>& saved) {
    const auto p = m.parameters(); require(p.size() == saved.size(), "registry count changed");
    for (size_t i = 0; i < p.size(); ++i)
        require(p[i]->name == saved[i].name && p[i]->version == saved[i].version &&
                equal(p[i]->data, saved[i].data), "weight/version changed: " + p[i]->name);
}
void finite(const Tensor& tensor, const std::string& label) {
    auto host = tensor.cpu();
    for (int i = 0; i < host.size; ++i) require(std::isfinite(host.data()[i]), label + " is nonfinite");
}
std::vector<Parameter*> expert(JambaModel& m, int e) {
    auto p = m.layers[0]->expert_gate_up[e]->parameters();
    const auto down = m.layers[0]->expert_down[e]->parameters();
    p.insert(p.end(), down.begin(), down.end()); return p;
}
Parameter* named(JambaModel& m, const std::string& suffix) {
    Parameter* found = nullptr;
    for (auto* p : m.parameters()) if (p->name.ends_with(suffix)) {
        require(!found, "ambiguous parameter: " + suffix); found = p;
    }
    require(found != nullptr, "missing parameter: " + suffix); return found;
}
std::vector<GpuSparseAdamState> sparse_states(Trainer& t) {
    require(t.device_sparse_adam != nullptr, "GPU sparse Adam did not execute");
    gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());
    auto states = t.device_sparse_adam->snapshot(); // explicit audit, not an optimizer cohort
    // Owner snapshot moments are borrowed. Freeze evidence across later attempts.
    for (auto& state : states) if (state.initialized) {
        state.m = state.m.clone(); state.v = state.v.clone();
    }
    return states;
}
std::map<std::string, GpuSparseAdamState> by_name(const std::vector<GpuSparseAdamState>& states) {
    std::map<std::string, GpuSparseAdamState> result;
    for (const auto& state : states) require(result.emplace(state.name, state).second, "duplicate sparse state");
    return result;
}
void same_states(const std::vector<GpuSparseAdamState>& a,
                 const std::vector<GpuSparseAdamState>& b, float tolerance = 0,
                 bool compare_versions = true) {
    const auto left = by_name(a), right = by_name(b);
    require(left.size() == right.size(), "sparse checkpoint cohort mismatch");
    for (const auto& [name, state] : left) {
        auto i = right.find(name); require(i != right.end(), "sparse state absent: " + name);
        const auto& other = i->second;
        require((!compare_versions || state.version == other.version) && state.initialized == other.initialized,
                "sparse version/lazy presence changed: " + name);
        if (state.initialized) {
            gpu_parity_test::assert_close(state.m, other.m, tolerance, (name + " m").c_str());
            gpu_parity_test::assert_close(state.v, other.v, tolerance, (name + " v").c_str());
        } else require(!state.m.size && !state.v.size && !other.m.size && !other.v.size,
                       "inactive sparse state exposed moments");
    }
}
#ifdef USE_CUDA
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
void activity(Trainer& t, const std::array<unsigned char, 4>& union_expected,
              const std::array<unsigned char, 4>& current_expected,
              const std::array<unsigned char, 4>& first_expected) {
    require(t.device_sparse_group_open && t.device_moe_groups.size() == 1,
            "real Trainer device accumulation group missing");
    gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());
    auto owner = t.device_moe_groups[0]->device_activity();
    require(owner && owner->producer(), "grouped MoE producer absent");
    const auto view = owner->view();
    std::array<unsigned char, 4> accumulated{}, current{}, first{}; int issue = -1;
    check(cudaStreamSynchronize(gpu::current_stream()));
    check(cudaMemcpy(accumulated.data(), view.accumulated, 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(current.data(), view.current, 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(first.data(), view.first_write, 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&issue, view.abort_issue, sizeof(issue), cudaMemcpyDeviceToHost));
    require(!issue && accumulated == union_expected && current == current_expected &&
            first == first_expected, "device MoE union/current/first-write evidence differs");
}
int attention_issue(Trainer& t, JambaModel& m) {
    gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());
    cuda_detail::DeviceBuffer<int> issue;
    require(issue.ensure(1), "attention audit issue allocation");
    check(cudaMemsetAsync(issue.get(), 0, sizeof(int), gpu::current_stream()));
    attention_training::merge_status(m.parameters(), issue.get());
    int host = 0;
    check(cudaMemcpyAsync(&host, issue.get(), sizeof(host), cudaMemcpyDeviceToHost, gpu::current_stream()));
    check(cudaStreamSynchronize(gpu::current_stream())); return host;
}
#endif
void cce_probe(Trainer& t, JambaModel& m) {
    const auto before = weights(m);
    gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());
    // Device MoE requires a Trainer-owned activity domain even for this
    // standalone CCE probe. Abort it before measuring the real A2 trajectory.
    t.begin_device_sparse_group();
    Context ctx;
    auto hidden = m.forward_ids_training_hidden(input0, &ctx);
    require(hidden.get_device() == Device::GPU && hidden.shape.back() == 16,
            "CCE probe must receive GPU hidden features, not vocabulary logits");
    TiledCrossEntropyOptions options;
    options.targets = targets0; options.weights.assign(input0.size(), 1.f / input0.size());
    options.sequence_length = static_cast<int>(input0.size());
    options.row_tile = 5; options.vocabulary_tile = 17;
    auto result = m.tiled_head_loss(options); // actual CCE call in the complete hybrid graph
    finite(result.losses, "CCE loss"); finite(result.input_gradient, "CCE hidden adjoint");
    require(result.input_gradient.shape == hidden.shape && result.maximum_logit_elements > 0 &&
            result.maximum_logit_elements <= 5 * 17, "CCE tiled workspace contract was not executed");
    require(m.value_head->weight.has_gradient() && m.value_head->weight.grad.norm() > 0,
            "CCE did not publish a real head gradient");
    m.backward_training_hidden(result.input_gradient, ctx);
    t.abort_gradient_accumulation(); // finishes domains, zeros gradients, resets session/status
    require(!t.device_sparse_group_open && t.device_moe_groups.empty() &&
            t.pending_accumulation_microbatches == 0 && t.pending_accumulated_tokens == 0 &&
            t.global_step_count == 0 && t.tokens_committed == 0 &&
            !t.device_sparse_adam && !t.optimizer_state_poisoned(),
            "CCE probe contaminated the Trainer accumulation boundary");
    same(m, before);
    std::cout << "CCE probe actual maximum_logit_elements=" << result.maximum_logit_elements << '\n';
}
void identity(Trainer& t) {
    const auto fields = t.execution_identity_fields();
    auto is = [&](const std::string& key, const std::string& value) {
        return std::find(fields.begin(), fields.end(), std::make_pair(key, value)) != fields.end();
    };
    require(is("mamba3.implementation", "mamba3_dense_fp32_siso_mimo_n128_v1"), "Mamba3 runtime identity absent");
    require(is("attention.training_provider", "rdna_bf16_v1"), "RDNA Attention identity absent");
    require(is("optimizer.sparse_gradient_policy", "explicit_group_contribution_device_v1"), "device sparse identity absent");
    require(is("optimizer.explicit_no_decay", "mamba3_dt_bias_D_v1"), "explicit Mamba3 no-decay identity absent");
    require(is("head.loss_policy", training_policy::head_cce_identity), "CCE checkpoint identity absent");
    require(training_policy::head_cce(), "CCE policy not active");
}
float group(Trainer& t, bool audit_versions = false) {
    // Versions are local cache generations, not portable checkpoint values.
    // Audit the commit delta independently in each replica, before comparing
    // their portable weights/moments/lazy presence.
    const auto parameters = audit_versions ? t.model->parameters() : std::vector<Parameter*>{};
    std::vector<std::uint64_t> versions;
    for (auto* p : parameters) versions.push_back(p->version);
    (void)t.accumulate_microbatch(input0, targets0);
#ifdef USE_CUDA
    activity(t, {1,0,0,0}, {1,0,0,0}, {1,0,0,0});
#endif
    (void)t.accumulate_microbatch(input1, targets1);
#ifdef USE_CUDA
    activity(t, {1,1,0,0}, {0,1,0,0}, {0,1,0,0});
#endif
    std::vector<unsigned char> contributed;
    if (audit_versions) {
        gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());
        DeviceGradientAuditScope audit; // observation only; never drives the optimizer cohort
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto* p = parameters[i];
            require(p->version == versions[i], "continuation BPTT changed a version: " + p->name);
            contributed.push_back(p->trainable && p->has_gradient());
        }
    }
    const float loss = t.commit_optimizer_step(2);
    if (audit_versions) {
        size_t active = 0, inactive = 0;
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto expected = versions[i] + (contributed[i] ? 1 : 0);
            require(parameters[i]->version == expected,
                    "continuation local version delta differs: " + parameters[i]->name);
            if (contributed[i]) ++active; else ++inactive;
        }
        // The activity assertions above require experts 0/1 active and 2/3
        // inactive. Thus both +1 and +0 are exercised, not just permitted.
        require(active > 0 && inactive > 0, "continuation version audit lost active/inactive coverage");
        const auto states = sparse_states(t);
        for (const auto& state : states) {
            auto p = std::find_if(parameters.begin(), parameters.end(),
                [&](Parameter* candidate) { return candidate->name == state.name; });
            require(p != parameters.end() && state.version == (*p)->version,
                    "continuation sparse/Parameter local version mismatch: " + state.name);
        }
        std::cout << "continuation local version deltas active(+1)=" << active
                  << " inactive/frozen(+0)=" << inactive << '\n';
    }
    return loss;
}
void invalid_attention(Trainer& t, JambaModel& m) {
#ifdef USE_CUDA
    const auto before = weights(m);
    const auto moments = sparse_states(t);
    const auto attention_before = attention_training::dispatch_counters();
    const auto dispatch_before = gpu::dispatch_counters();
    const auto mamba_before = m.layers[0]->mamba3_layer->telemetry();
    const auto step = t.global_step_count; const auto tokens = t.tokens_committed;
    auto* bias = named(m, ".attn.q_down_proj.bias");
    const auto clean = bias->data.clone();
    // Finite but outside RDNA |Q|<=64: exercises status, not a NaN finite scan.
    // Test-only fault injection into an exact-linear FP32 leaf bias. Preserve
    // storage/version: the initialized sparse owner rejects external version
    // mutations across groups. No derived weight/QAT cache uses this bias, and
    // the real provider's sticky numeric status proves the injection took effect.
    // Never mutate between a forward and its backward.
    bias->data.copy_from(Tensor::ones(bias->data.shape.dims, Device::GPU).mul(1e4f));
    (void)t.accumulate_microbatch(input0, targets0);
    require(attention_issue(t, m) != 0, "invalid Attention status did not enter sticky ledger");
    bias->data.copy_from(clean);
    (void)t.accumulate_microbatch(input1, targets1);
    require(attention_issue(t, m) != 0, "valid microbatch erased invalid Attention ledger");
    for (auto* p : m.parameters()) {
        if (p->has_device_gradient_activity()) {
            const int e = p->device_gradient_binding().expert;
            if (e == 0 || e == 1) finite(p->grad, p->name + " active gradient");
        } else if (p->has_gradient()) finite(p->grad, p->name + " dense gradient");
    }
    const auto precommit = weights(m);
    require(precommit.size() == before.size(), "failed group registry mismatch");
    for (size_t i = 0; i < before.size(); ++i) {
        require(equal(precommit[i].data, before[i].data), "failed group changed master weight");
        require(precommit[i].version == before[i].version,
                "backward unexpectedly changed a parameter version");
    }
    const auto attention_ready = attention_training::dispatch_counters();
    const auto dispatch_ready = gpu::dispatch_counters();
    const auto mamba_ready = m.layers[0]->mamba3_layer->telemetry();
    require(attention_ready[0] >= attention_before[0] + 2 && attention_ready[1] >= attention_before[1] + 2 &&
            mamba_ready.gpu_backward >= mamba_before.gpu_backward + 2 &&
            dispatch_ready[static_cast<unsigned>(gpu::DispatchPath::GroupedMoeGradientCommit)] >
                dispatch_before[static_cast<unsigned>(gpu::DispatchPath::GroupedMoeGradientCommit)],
            "invalid Attention fixture did not finish real Attention/Mamba3/MoE BPTT");
    bool rejected = false;
    try { (void)t.commit_optimizer_step(2); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected || t.last_optimizer_step_skipped, "invalid Attention permitted optimizer commit");
    require(t.last_optimizer_step_skipped && t.global_step_count == step && t.tokens_committed == tokens &&
            t.pending_accumulation_microbatches == 0 && !t.device_sparse_group_open,
            "rejected hybrid group advanced Adam trajectory or retained group ownership");
    same(m, precommit);
    same_states(moments, sparse_states(t));
    require(!t.optimizer_state_poisoned(), "numeric status rejection poisoned recoverable Trainer");
    require(std::isfinite(group(t)) && t.global_step_count == step + 1,
            "clean group did not recover after Attention rejection");
    std::cout << "sticky Attention invalid -> finite gradients -> no weights/moments/versions/token commit PASS\n";
#else
    (void)t; (void)m; throw std::runtime_error("hybrid status test requires GPU backend");
#endif
}
void integral(bool deterministic) {
    determinism::set_deterministic_reductions(deterministic);
    JambaModel m(config(), Device::GPU); prepare(m);
    Trainer t(&m, .002f); setup(t);
    cce_probe(t, m);
    const auto initial = weights(m);
    const auto dispatch_before = gpu::dispatch_counters();
    const auto attention_before = attention_training::dispatch_counters();
    const auto mamba_before = m.layers[0]->mamba3_layer->telemetry();
    (void)t.accumulate_microbatch(input0, targets0);
    same(m, initial);
#ifdef USE_CUDA
    activity(t, {1,0,0,0}, {1,0,0,0}, {1,0,0,0});
#endif
    require(t.global_step_count == 0 && t.pending_accumulation_microbatches == 1 && t.tokens_committed == 0,
            "first microbatch committed an update");
    (void)t.accumulate_microbatch(input1, targets1); same(m, initial);
#ifdef USE_CUDA
    activity(t, {1,1,0,0}, {0,1,0,0}, {0,1,0,0});
#endif
    require(t.pending_accumulation_microbatches == 2, "accumulation2 was not real");
    identity(t);
    for (const auto& suffix : {".attn.q_down_proj.weight", ".attn.kv_down_proj.weight", ".attn.out_proj.weight"}) {
        auto* p = named(m, suffix);
        require(p->has_gradient() && p->grad.norm() > 0, "actual Attention projection gradient missing: " + p->name);
    }
    const auto excluded = m.no_weight_decay_parameters();
    require(excluded.size() == 2, "dt_bias/D optimizer exclusions missing");
    std::vector<Weight> no_decay;
    for (auto* p : excluded) {
        require(p->name.ends_with(".mamba3.dt_bias") || p->name.ends_with(".mamba3.D"), "wrong no-decay parameter");
        require(p->has_gradient() && p->grad.norm() > 0, "no-decay parameter did not receive real hybrid VJP");
        no_decay.push_back({p->name, p->data.cpu().clone(), p->version});
        // Clear numerical values ONLY: retain the legitimate contribution predicate.
        p->grad.copy_from(Tensor::zeros(p->grad.shape.dims, Device::GPU));
    }
    auto* control = named(m, ".mamba3.in_proj.weight");
    require(control->has_gradient() && control->grad.norm() > 0, "Mamba3 dense projection adjoint missing");
    const auto control_before = control->data.cpu().clone();
    control->grad.copy_from(Tensor::zeros(control->grad.shape.dims, Device::GPU));
    require(std::isfinite(t.commit_optimizer_step(2)), "valid hybrid optimizer group rejected");
    require(t.device_sparse_adam && !t.device_sparse_group_open && t.global_step_count == 1 &&
            t.tokens_committed == 50 && t.last_accumulation_steps == 2,
            "device sparse Adam/accumulation/commit evidence missing");
    require(std::isfinite(t.last_grad_norm_pre_clip) && t.last_grad_norm_pre_clip > 0 &&
            t.last_grad_norm_post_clip <= t.max_grad_norm + 1e-6f, "device clipping telemetry invalid");
    for (size_t i = 0; i < excluded.size(); ++i)
        require(equal(excluded[i]->data, no_decay[i].data) && excluded[i]->version == no_decay[i].version + 1,
                "dt_bias/D decayed or lost zero-valued contribution: " + excluded[i]->name);
    auto actual = control->data.cpu(); bool decay_observed = false;
    for (int i = 0; i < actual.size; ++i) {
        const float w = control_before.data()[i], expected = w - t.learning_rate * t.weight_decay * w;
        require(std::abs(actual.data()[i] - expected) <= 2e-7f, "positive decay control violates AdamW oracle");
        decay_observed |= std::abs(actual.data()[i] - w) > 1e-6f;
    }
    require(decay_observed, "no-decay test never executed weight decay");
    const auto states = by_name(sparse_states(t));
    bool expert_update = false, attention_update = false, head_update = false;
    const auto current = m.parameters();
    for (size_t i = 0; i < current.size(); ++i) {
        if (current[i]->name.find(".attn.q_down_proj.weight") != std::string::npos)
            attention_update |= !equal(current[i]->data, initial[i].data);
        if (current[i]->name == "value_head.weight") head_update |= !equal(current[i]->data, initial[i].data);
    }
    for (int e : {0,1}) for (auto* p : expert(m,e)) if (p->trainable) {
        require(states.at(p->name).initialized, "microbatch union lost active expert moments");
        auto before = std::find_if(initial.begin(), initial.end(), [&](const Weight& w) { return w.name == p->name; });
        require(before != initial.end() && p->version == before->version + 1, "active expert version did not publish once");
        expert_update |= !equal(p->data, before->data);
    }
    for (int e : {2,3}) for (auto* p : expert(m,e)) if (p->trainable) {
        require(!states.at(p->name).initialized, "inactive expert allocated logical Adam moments");
        auto before = std::find_if(initial.begin(), initial.end(), [&](const Weight& w) { return w.name == p->name; });
        require(before != initial.end() && p->version == before->version && equal(p->data, before->data),
                "inactive expert updated/decayed/published a version");
    }
    require(expert_update && attention_update && head_update, "hybrid Attention/MoE/CCE head had no real parameter update");
    const auto dispatch_after = gpu::dispatch_counters();
    const auto attention_after = attention_training::dispatch_counters();
    const auto mamba_after = m.layers[0]->mamba3_layer->telemetry();
    const auto grouped = static_cast<unsigned>(gpu::DispatchPath::GroupedMoeTraining);
    const auto commits = static_cast<unsigned>(gpu::DispatchPath::GroupedMoeGradientCommit);
    require(dispatch_after[grouped] >= dispatch_before[grouped] + 2 && dispatch_after[commits] > dispatch_before[commits],
            "grouped MoE forward/gradient publication counters did not execute");
    require(attention_after[0] >= attention_before[0] + 2 && attention_after[1] >= attention_before[1] + 2 &&
            attention_after[2] > attention_before[2], "RDNA Attention forward/backward/status merge not executed");
    require(mamba_after.gpu_forward >= mamba_before.gpu_forward + 2 &&
            mamba_after.gpu_backward >= mamba_before.gpu_backward + 2 &&
            mamba_after.cpu_forward == mamba_before.cpu_forward && mamba_after.cpu_backward == mamba_before.cpu_backward,
            "Mamba3 N128 GPU forward/backward missing or fell back to CPU");
    const auto stem = std::filesystem::temp_directory_path() /
        ("nsos-hybrid-integral-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto model_file = stem.string() + ".model", state_file = stem.string() + ".state";
    struct Files {
        std::string model, state;
        ~Files() { std::error_code ec; std::filesystem::remove(model,ec); std::filesystem::remove(state,ec); }
    } files{model_file, state_file};
    t.capture_checkpoint_snapshot()->write(model_file, state_file);
    JambaModel resumed(config(), Device::GPU); prepare(resumed); resumed.load(model_file);
    Trainer rt(&resumed, .002f); setup(rt); rt.load_training_state(state_file, model_file);
    require(t.execution_identity_digest() == rt.execution_identity_digest(), "hybrid resume fingerprint mismatch");
    const float a = group(t, true), b = group(rt, true);
    const float tolerance = deterministic ? 0.f : 5e-6f;
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tolerance,
            "hybrid resumed accumulated loss differs");
    const auto left = m.parameters(), right = resumed.parameters();
    require(left.size() == right.size(), "hybrid resume parameter count differs");
    for (size_t i = 0; i < left.size(); ++i) {
        require(left[i]->name == right[i]->name, "resume parameter name differs");
        gpu_parity_test::assert_close(left[i]->data, right[i]->data, tolerance, left[i]->name.c_str());
    }
    // Each replica's version deltas and sparse-bank mirror were audited by
    // group(..., true). Absolute cache generations need not survive serialization.
    same_states(sparse_states(t), sparse_states(rt), tolerance, false);
    require(t.global_step_count == rt.global_step_count && t.tokens_committed == rt.tokens_committed &&
            t.global_step_count == 2 && t.tokens_committed == 100, "resume scheduler/token trajectory differs");
    // The next real update must also reach Mamba3 after the initial decay probe.
    require(!equal(control->data, actual), "Mamba3 projection did not update after checkpoint continuation");
    for (size_t i = 0; i < excluded.size(); ++i)
        require(!equal(excluded[i]->data, no_decay[i].data), "dt_bias/D lost normal Adam updates after decay probe");
    invalid_attention(t, m);
    std::cout << "HYBRID lane=" << (deterministic ? "deterministic" : "ordinary")
              << " Mamba3 GPU fwd/bwd=" << mamba_after.gpu_forward-mamba_before.gpu_forward << '/'
              << mamba_after.gpu_backward-mamba_before.gpu_backward
              << " Attention fwd/bwd/merge=" << attention_after[0]-attention_before[0] << '/'
              << attention_after[1]-attention_before[1] << '/' << attention_after[2]-attention_before[2]
              << " grouped MoE/gradcommit=" << dispatch_after[grouped]-dispatch_before[grouped] << '/'
              << dispatch_after[commits]-dispatch_before[commits]
              << " accumulation=2 no_decay=dt_bias,D checkpoint_resume=PASS\n";
}
} // namespace
int main() {
    try {
        Policy memory("NSOS_GPU_MEMORY","device"), advice("NSOS_NO_MEMADVISE","1"), sync("NSOS_CUDA_SYNC","0");
        Policy head("NSOS_HEAD_CCE","1"), attention("NSOS_ATTN_TRAINING_PROVIDER","rdna_bf16_v1");
        Policy tiled("NSOS_ATTN_TILED_TRAINING","0"), host("NSOS_ATTN_BWD_HOST","0");
        Policy device("NSOS_MOE_DEVICE_ADAM","1"), grouped("NSOS_MOE_GROUPED_TRAINING","1");
        Policy ordered("NSOS_MOE_ORDERED_DEVICE","1"), wmma("NSOS_MOE_WMMA_TRAINING","0");
        Policy crit("NSOS_CRIT_REG","0"), crit_lr("NSOS_CRIT_LR","0");
        set_matmul_precision_mode(0); set_strict_gpu_execution(true);
        gpu_parity_test::require_cuda_device("hybrid_integral");
        require(attention_rdna::supported({1,33,2,1,8,64,.35355339f,attention_rdna::Precision::BF16}),
                "final hybrid binary lacks RDNA3 Attention coverage/device");
        integral(true); integral(false);
        gpu_parity_test::cuda_sync_or_throw("hybrid integral final");
        std::cout << "PASS GPU hybrid integral actual combined binary\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL hybrid integral: " << e.what() << '\n'; return 1; }
}
