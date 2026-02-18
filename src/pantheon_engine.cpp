#include "pantheon_engine.hpp"
#include <iostream>
#include <stdexcept>

#include "host/uhk_runtime.h"

namespace pantheon {

    PantheonEngine::PantheonEngine() {
        std::cout << "[Pantheon] Creating Engine instance..." << std::endl;
        d_A = new float[1024];
        d_B = new float[1024];
        d_C = new float[1024];
    }

    PantheonEngine::~PantheonEngine() {
        if (initialized_) {
            shutdown();
        }
        delete[] d_A;
        delete[] d_B;
        delete[] d_C;
    }

    void PantheonEngine::initialize() {
        if (initialized_) {
            std::cout << "[Pantheon] Already initialized." << std::endl;
            return;
        }

        std::cout << "[Pantheon] Initializing KernelOpen Runtime..." << std::endl;
        try {
            kernel_runtime_ = std::make_unique<uhk::runtime::UniversalKernelRuntime>();
            initialized_ = true;
            std::cout << "[Pantheon] KernelOpen initialized successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Pantheon] Failed to initialize KernelOpen: " << e.what() << std::endl;
            throw;
        }
    }

    void PantheonEngine::shutdown() {
        if (!initialized_) return;

        std::cout << "[Pantheon] Shutting down..." << std::endl;
        if (kernel_runtime_) {
            kernel_runtime_->shutdown();
            kernel_runtime_.reset();
        }
        initialized_ = false;
    }

    void PantheonEngine::submit_dummy_task() {
        if (!initialized_ || !kernel_runtime_) return;
        kernel_runtime_->submit_bitnet_gemm(32, 32, 32, d_A, d_B, d_C);
    }

    float PantheonEngine::get_throughput() const {
        if (!initialized_ || !kernel_runtime_) return 0.0f;
        return kernel_runtime_->get_throughput();
    }

    std::string PantheonEngine::get_status() const {
        if (!initialized_) return "Uninitialized";
        return "Running (Connected to KernelOpen)";
    }

    // Part 1 Structure
    float PantheonEngine::compute_contrastive_loss(const std::vector<float>& s, const std::vector<float>& t, int b, int d) {
        return pantheon::structure::ContrastiveDistillation::compute_loss(s, t, b, d);
    }
    float PantheonEngine::compute_relational_loss(const std::vector<float>& s, const std::vector<float>& t, int b, int d) {
        return pantheon::structure::RelationalDistillation::compute_distance_loss(s, t, b, d);
    }
    float PantheonEngine::compute_msdcrd_loss(const std::vector<float>& s, const std::vector<float>& t, int b, int c, int h, int w, int ph, int pw) {
        return pantheon::structure::MultiScaleDistillation::compute_msdcrd_loss(s, t, b, c, h, w, ph, pw);
    }
    float PantheonEngine::compute_cam_loss(const std::vector<float>& sf, const std::vector<float>& sw, const std::vector<float>& tf, const std::vector<float>& tw, int b, int c, int h, int w, int cl, int tg) {
        return pantheon::structure::AttentionTransfer::compute_cam_loss(sf, sw, tf, tw, b, c, h, w, cl, tg);
    }

    // Part 1 Response
    float PantheonEngine::compute_ntce_loss(const std::vector<float>& s, const std::vector<float>& t, int tg, float b) {
        return pantheon::response::LogitDistillation::compute_ntce_loss(s, t, tg, 1.0f, b);
    }
    float PantheonEngine::compute_optimal_transport_loss(const std::vector<float>& s, const std::vector<float>& t) {
        return pantheon::response::LogitDistillation::compute_optimal_transport_loss(s, t);
    }

    // Part 2 Cognition
    float PantheonEngine::compute_chunk_wise_loss(const std::vector<float>& s, const std::vector<float>& t, int c) {
        return pantheon::cognition::ChainOfThought::compute_chunk_wise_loss(s, t, c);
    }
    std::vector<int> PantheonEngine::adjust_granularity(const std::vector<int>& s, float c) {
        return pantheon::cognition::ChainOfThought::adjust_granularity(s, c);
    }
    float PantheonEngine::compute_symbolic_loss(const std::vector<float>& o, float e) {
        return pantheon::cognition::SymbolicEngine::compute_symbolic_loss(o, [e](const std::vector<float>& v){ return pantheon::cognition::SymbolicEngine::verify_program_spec(v, e); });
    }

    // Part 3 Physics
    float PantheonEngine::compute_gradient_loss(const std::vector<float>& t, const std::vector<float>& s) {
        return pantheon::physics::GradientMatcher::compute_loss(t, s);
    }
    float PantheonEngine::compute_ib_loss(const std::vector<float>& m, const std::vector<float>& l) {
        return pantheon::physics::InformationBottleneck::compute_ib_compression_loss(m, l);
    }
    float PantheonEngine::compute_topology_loss(const std::vector<float>& s, const std::vector<float>& t, int n) {
        return pantheon::physics::TopologyDistillation::compute_topology_loss(s, t, n);
    }
    std::vector<float> PantheonEngine::compute_ode_adjoint(const std::vector<float>& z, const std::vector<float>& g, float dt) {
        return pantheon::physics::NeuralODE::compute_adjoint_step(z, g, dt);
    }

    // Part 4 Frontier
    float PantheonEngine::update_meta_policy(float t, float l) {
        return pantheon::frontier::MetaDistiller::update_temperature(t, l);
    }
    float PantheonEngine::compute_causal_loss(const std::vector<float>& o, const std::vector<float>& i) {
        return pantheon::frontier::CausalInvariance::compute_invariance_loss(o, i);
    }
    float PantheonEngine::compute_quantum_loss(const std::vector<float>& s, const std::vector<float>& t) {
        return pantheon::frontier::QuantumKernel::compute_fidelity_loss(s, t);
    }
    float PantheonEngine::compute_spike_loss(const std::vector<int>& s, const std::vector<int>& t) {
        return pantheon::frontier::SpikingDistance::compute_spike_loss(s, t);
    }

    // Part 5 Physical
    float PantheonEngine::compute_photonic_loss(const std::vector<float>& w, int width, int height) {
        return pantheon::frontier::PhysicalDistiller::compute_photonic_loss(w, width, height);
    }
    std::vector<float> PantheonEngine::inject_memristive_noise(const std::vector<float>& w, float r) {
        return pantheon::frontier::PhysicalDistiller::apply_memristive_noise(w, r);
    }

    // Part 6 Abstract
    float PantheonEngine::compute_functorial_loss(const std::vector<float>& mt, const std::vector<float>& ms) {
        return pantheon::structure::CategoryTheory::compute_functorial_loss(mt, ms);
    }
    float PantheonEngine::compute_tom_loss(const std::vector<float>& tb, const std::vector<float>& pb) {
        return pantheon::cognition::TheoryOfMind::compute_tom_loss(tb, pb);
    }

    // Part 7 Social
    float PantheonEngine::compute_nash_regret(const std::vector<float>& n, const std::vector<float>& s) {
        return pantheon::social::GameTheory::compute_nash_regret(n, s);
    }
    std::vector<float> PantheonEngine::update_swarm(const std::vector<float>& c, const std::vector<float>& b, const std::vector<float>& v) {
        return pantheon::social::SwarmDistiller::update_swarm_weights(c, b, v);
    }

    // Part 8 Integrity
    std::vector<float> PantheonEngine::compute_pate_aggregation(const std::vector<int>& v, float e) {
        return pantheon::frontier::PrivacyEngine::aggregate_pate(v, e);
    }

}
