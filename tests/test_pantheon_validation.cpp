// ============================================================================
//  test_pantheon_validation.cpp -- Pantheon distillation library validation
// ============================================================================
//
//  Pantheon é uma biblioteca header-only de losses/kernels para
//  distillation:
//    response/logit_distillation:    NTCE-KD + Sinkhorn OT
//    structure/contrastive:          InfoNCE-style CRD
//    structure/msdcrd:               Multi-Scale Distillation CRD
//    structure/relational:           Relational distillation
//    physics/information_bottleneck: VIB compression
//    physics/gradient_matching:      Gradient distillation
//    physics/neural_ode:             ODE adjoint
//    frontier/quantum:               Quantum fidelity
//    cognition/chain_of_thought:     CoT trace loss
//  (15 outros headers existem mas não foram inspecionados nesta auditoria.)
//
//  Os .cpp são stubs vazios — toda a lógica está nos .hpp como static
//  inline functions.  Não há build system pra eles em produção.  Este
//  teste compila standalone com apenas STL.
// ============================================================================

#include "pantheon/response/logit_distillation.hpp"
#include "pantheon/structure/contrastive.hpp"
#include "pantheon/physics/information_bottleneck.hpp"
#include "pantheon/physics/neural_ode.hpp"
#include "pantheon/frontier/quantum.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_passed = 0;
int g_failed = 0;
std::vector<std::string> g_failures;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream _o;                                             \
            _o << __func__ << ":" << __LINE__ << " " << msg;                   \
            g_failures.push_back(_o.str());                                    \
            std::cerr << "    FAIL: " << _o.str() << std::endl;                \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RUN(fn)                                                                \
    do {                                                                       \
        std::cout << "[" << #fn << "]" << std::flush;                          \
        bool _ok = fn();                                                       \
        if (_ok) { std::cout << "  PASS\n"; ++g_passed; }                      \
        else     { std::cout << "  FAIL\n"; ++g_failed; }                      \
    } while (0)

}  // namespace

// ============================================================================
// LogitDistillation — NTCE-KD
// ============================================================================

bool test_ntce_perfect_match_yields_low_loss() {
    // Se student == teacher, loss deveria ser baixa
    std::vector<float> logits = {2.0f, 0.5f, -1.0f, 3.5f, 0.0f};
    int target = 3;
    float loss = pantheon::response::LogitDistillation::compute_ntce_loss(
        logits, logits, target, 1.0f, 2.0f);
    std::cout << "    NTCE loss (s=t): " << loss;
    CHECK(loss >= 0.0f, "NTCE loss negative");
    CHECK(loss < 10.0f, "NTCE loss too high for perfect match: " << loss);
    return true;
}

bool test_ntce_amplifies_non_target() {
    // beta=1 (no amplification) vs beta=10 (strong amplification)
    std::vector<float> s = {1.0f, 0.5f, -0.5f, 2.0f};
    std::vector<float> t = {1.5f, 0.2f, -0.8f, 1.8f};
    int target = 3;

    float loss_1 = pantheon::response::LogitDistillation::compute_ntce_loss(
        s, t, target, 1.0f, 1.0f);
    float loss_10 = pantheon::response::LogitDistillation::compute_ntce_loss(
        s, t, target, 1.0f, 10.0f);
    CHECK(loss_10 > loss_1, "amplification beta=10 should increase loss vs beta=1: "
          << loss_1 << " vs " << loss_10);
    return true;
}

bool test_ntce_size_mismatch_throws() {
    std::vector<float> s = {1.0f, 2.0f};
    std::vector<float> t = {1.0f, 2.0f, 3.0f};
    bool threw = false;
    try {
        pantheon::response::LogitDistillation::compute_ntce_loss(s, t, 0);
    } catch (...) { threw = true; }
    CHECK(threw, "size mismatch should throw");
    return true;
}

// ============================================================================
// LogitDistillation — Sinkhorn Optimal Transport
// ============================================================================

bool test_sinkhorn_identical_distributions() {
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 3.0f, 1.5f};
    float dist = pantheon::response::LogitDistillation::compute_optimal_transport_loss(
        logits, logits, 1.0f, 50, 0.1f);
    std::cout << "    OT distance (identical): " << dist;
    CHECK(dist >= 0.0f, "OT distance negative");
    // Pra distribuições idênticas, OT distance deve ser ~0 (Sinkhorn não converge a zero exatamente)
    CHECK(dist < 1.5f, "OT distance too high for identical: " << dist);
    return true;
}

bool test_sinkhorn_different_distributions_positive() {
    std::vector<float> s = {3.0f, 0.1f, 0.1f, 0.1f};  // peaks at 0
    std::vector<float> t = {0.1f, 0.1f, 0.1f, 3.0f};  // peaks at 3
    float dist = pantheon::response::LogitDistillation::compute_optimal_transport_loss(
        s, t, 1.0f, 100, 0.1f);
    std::cout << "    OT distance (peaks far apart): " << dist;
    CHECK(dist > 0.1f, "OT distance too low for distinct distributions: " << dist);
    return true;
}

// ============================================================================
// Contrastive (InfoNCE / CRD)
// ============================================================================

bool test_crd_aligned_features_low_loss() {
    int batch = 4, dim = 8;
    std::vector<float> z_s(batch * dim, 0.0f);
    std::vector<float> z_t(batch * dim, 0.0f);
    // Make z_s == z_t (perfect alignment) - distinct per row
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < dim; ++j) {
            z_s[i * dim + j] = (float)((i + j) % 4 + 1);
            z_t[i * dim + j] = z_s[i * dim + j];
        }
    }
    float loss = pantheon::structure::ContrastiveDistillation::compute_loss(
        z_s, z_t, batch, dim, 0.07f);
    std::cout << "    CRD loss (aligned): " << loss;
    CHECK(loss >= 0.0f, "CRD loss negative");
    // InfoNCE com positivos perfeitos e negativos diferentes deve ser baixa
    // (≤ log(batch) que é o random guess level)
    CHECK(loss < std::log((float)batch) + 0.5f,
          "CRD loss higher than chance for aligned: " << loss);
    return true;
}

bool test_crd_size_mismatch_throws() {
    std::vector<float> z_s(20, 1.0f);  // 5x4
    std::vector<float> z_t(30, 1.0f);  // mismatch
    bool threw = false;
    try {
        pantheon::structure::ContrastiveDistillation::compute_loss(z_s, z_t, 5, 4);
    } catch (...) { threw = true; }
    CHECK(threw, "size mismatch should throw");
    return true;
}

// ============================================================================
// InformationBottleneck (VIB)
// ============================================================================

bool test_vib_zero_mean_unit_var_minimal() {
    // Quando mean=0 e log_var=0 (var=1), KL com N(0,1) é zero.
    std::vector<float> means(16, 0.0f);
    std::vector<float> log_vars(16, 0.0f);
    float loss = pantheon::physics::InformationBottleneck::compute_ib_compression_loss(
        means, log_vars, 1.0f);
    std::cout << "    VIB loss (N(0,1) prior match): " << loss;
    CHECK(std::fabs(loss) < 1e-5f, "VIB loss should be ~0 for prior match: " << loss);
    return true;
}

bool test_vib_drift_from_prior_positive() {
    // mean=2.0, var=1 -> KL > 0
    std::vector<float> means(16, 2.0f);
    std::vector<float> log_vars(16, 0.0f);
    float loss = pantheon::physics::InformationBottleneck::compute_ib_compression_loss(
        means, log_vars, 1.0f);
    std::cout << "    VIB loss (mean=2 drift): " << loss;
    CHECK(loss > 0.5f, "VIB should penalize drift: " << loss);
    return true;
}

bool test_vib_beta_scales_loss() {
    std::vector<float> means(8, 1.0f);
    std::vector<float> log_vars(8, 0.0f);
    float l1 = pantheon::physics::InformationBottleneck::compute_ib_compression_loss(
        means, log_vars, 1.0f);
    float l5 = pantheon::physics::InformationBottleneck::compute_ib_compression_loss(
        means, log_vars, 5.0f);
    CHECK(std::fabs(l5 - 5.0f * l1) < 1e-3f,
          "beta should scale linearly: l1=" << l1 << " l5=" << l5);
    return true;
}

// ============================================================================
// Quantum Fidelity
// ============================================================================

bool test_quantum_identical_states_zero_loss() {
    // |psi> = (1, 0) - encoded as [1, 0, 0, 0] (re0, im0, re1, im1)
    std::vector<float> state = {1.0f, 0.0f, 0.0f, 0.0f};
    float loss = pantheon::frontier::QuantumKernel::compute_fidelity_loss(state, state);
    std::cout << "    Quantum loss (identical): " << loss;
    CHECK(loss < 0.01f, "fidelity loss too high for identical: " << loss);
    return true;
}

bool test_quantum_orthogonal_states_max_loss() {
    // |psi_1> = (1, 0) vs |psi_2> = (0, 1) -- orthogonal, fidelity = 0
    std::vector<float> s = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> t = {0.0f, 0.0f, 1.0f, 0.0f};
    float loss = pantheon::frontier::QuantumKernel::compute_fidelity_loss(s, t);
    std::cout << "    Quantum loss (orthogonal): " << loss;
    CHECK(loss > 0.99f, "fidelity loss too low for orthogonal: " << loss);
    return true;
}

bool test_quantum_odd_size_returns_error() {
    std::vector<float> s = {1.0f, 0.0f, 0.0f};  // 3 elements (odd)
    std::vector<float> t = {1.0f, 0.0f, 0.0f};
    float loss = pantheon::frontier::QuantumKernel::compute_fidelity_loss(s, t);
    CHECK(loss >= 1.0f, "odd size should return error indicator");
    return true;
}

// ============================================================================
// Neural ODE adjoint
// ============================================================================

bool test_neural_ode_adjoint_runs() {
    std::vector<float> z = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> grad = {0.1f, 0.1f, 0.1f, 0.1f};
    auto out = pantheon::physics::NeuralODE::compute_adjoint_step(z, grad, 0.01f);
    CHECK(out.size() == z.size(), "adjoint output size wrong");
    for (auto v : out) {
        CHECK(!std::isnan(v), "adjoint produced NaN");
    }
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "==========================================================\n";
    std::cout << " Pantheon distillation library validation\n";
    std::cout << "==========================================================\n";

    // LogitDistillation
    RUN(test_ntce_perfect_match_yields_low_loss);
    RUN(test_ntce_amplifies_non_target);
    RUN(test_ntce_size_mismatch_throws);
    RUN(test_sinkhorn_identical_distributions);
    RUN(test_sinkhorn_different_distributions_positive);

    // Contrastive
    RUN(test_crd_aligned_features_low_loss);
    RUN(test_crd_size_mismatch_throws);

    // InformationBottleneck
    RUN(test_vib_zero_mean_unit_var_minimal);
    RUN(test_vib_drift_from_prior_positive);
    RUN(test_vib_beta_scales_loss);

    // Quantum
    RUN(test_quantum_identical_states_zero_loss);
    RUN(test_quantum_orthogonal_states_max_loss);
    RUN(test_quantum_odd_size_returns_error);

    // Neural ODE
    RUN(test_neural_ode_adjoint_runs);

    std::cout << "==========================================================\n";
    std::cout << " RESULT: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "==========================================================\n";
    if (g_failed > 0) {
        for (auto& f : g_failures) std::cerr << "  - " << f << "\n";
    }
    return g_failed == 0 ? 0 : 1;
}
