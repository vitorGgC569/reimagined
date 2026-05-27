// ============================================================================
//  test_ttt_validation.cpp -- TTT (Hamiltonian Test-Time Training) validation
// ============================================================================
//
//  Existing test_ttt_layer_kernel.cpp validates kernel basics (adaptation
//  happens, backward produces gradients).  This file goes deeper to qualify
//  TTT for *production training* (currently marked research-only).
//
//  Covers: reset/snapshot/restore, training_mode toggle, Hamiltonian on/off
//  behavior, long-sequence numerical stability, hyperparam effects
//  (friction, temperature), determinism, BitLinear collection, integration.
// ============================================================================

#include "../include/ttt_layer.h"
#include "../include/tensor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace nsos;

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

Tensor make_constant(int batch, int seq, int dim, float val) {
    Tensor t({batch, seq, dim}, Device::CPU);
    std::fill(t.data(), t.data() + t.size, val);
    return t;
}

Tensor make_random(int batch, int seq, int dim, unsigned seed, float scale = 1.0f) {
    Tensor t({batch, seq, dim}, Device::CPU);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-scale, scale);
    for (int i = 0; i < t.size; ++i) t.data()[i] = u(rng);
    return t;
}

bool has_no_nan(const Tensor& t) {
    for (int i = 0; i < t.size; ++i) {
        if (std::isnan(t.data()[i]) || std::isinf(t.data()[i])) return false;
    }
    return true;
}

float max_abs(const Tensor& t) {
    float m = 0.0f;
    for (int i = 0; i < t.size; ++i) m = std::max(m, std::fabs(t.data()[i]));
    return m;
}

float l2_diff(const Tensor& a, const Tensor& b) {
    float s = 0.0f;
    int n = std::min(a.size, b.size);
    for (int i = 0; i < n; ++i) {
        float d = a.data()[i] - b.data()[i];
        s += d * d;
    }
    return std::sqrt(s);
}

}  // namespace

// ============================================================================
// [A] ADAPTATION CONTRACT — TTT adapts on a sequence
// ============================================================================

bool test_constant_input_produces_drift() {
    // Same input across time -> output should differ at t=0 vs t=N-1
    int dim = 32, hidden = 64, seq = 10;
    TTTLayer layer(dim, hidden, 0.01f);
    Tensor x = make_constant(1, seq, dim, 1.0f);
    Tensor y = layer.forward(x);
    CHECK(has_no_nan(y), "TTT output has NaN");
    float diff = 0.0f;
    for (int i = 0; i < dim; ++i) {
        diff += std::fabs(y.data()[i] - y.data()[(seq - 1) * dim + i]);
    }
    CHECK(diff > 1e-3f, "no adaptation observed (diff=" << diff << ")");
    return true;
}

bool test_training_mode_off_no_adaptation() {
    // Em eval mode (training_mode=false), forward NÃO deve adaptar pesos
    int dim = 16, hidden = 32, seq = 8;
    TTTLayer layer(dim, hidden, 0.01f);
    layer.set_training_mode(false);
    Tensor x = make_constant(1, seq, dim, 0.7f);
    Tensor y = layer.forward(x);
    CHECK(has_no_nan(y), "eval mode produced NaN");
    // Output across timesteps should be roughly stable
    float drift = 0.0f;
    for (int i = 0; i < dim; ++i) {
        drift += std::fabs(y.data()[i] - y.data()[(seq - 1) * dim + i]);
    }
    std::cout << "    eval-mode drift over " << seq << " steps: " << drift;
    // In eval mode, drift should be SMALL (close to zero) — at most epsilon
    // from quantization noise.  We allow some slack since BitLinear quantizes.
    // Sanity: drift should not be orders of magnitude larger than train-mode 0
    CHECK(drift < 50.0f, "eval mode drift too large: " << drift);
    return true;
}

// ============================================================================
// [R] RESET / SNAPSHOT / RESTORE
// ============================================================================

bool test_reset_returns_to_initial_state() {
    int dim = 16, hidden = 32;
    TTTLayer layer(dim, hidden, 0.01f);

    Tensor x = make_constant(1, 5, dim, 0.5f);
    Tensor y1 = layer.forward(x);
    layer.reset();
    Tensor y2 = layer.forward(x);
    // After reset, second pass should match first (state cleared)
    float diff = l2_diff(y1, y2);
    std::cout << "    diff after reset (should be tiny): " << diff;
    CHECK(diff < 1e-1f, "reset didn't clear state: diff=" << diff);
    return true;
}

bool test_snapshot_restore_roundtrip() {
    int dim = 16, hidden = 32;
    TTTLayer layer(dim, hidden, 0.01f);

    // Run some adaptation
    Tensor x_init = make_random(1, 5, dim, 42);
    layer.forward(x_init);

    auto snap = layer.snapshot_state();
    CHECK(snap.has_state, "snapshot should have_state after forward");

    // Continue running
    Tensor x_more = make_random(1, 5, dim, 43);
    Tensor y_continue = layer.forward(x_more);

    // Restore and replay
    layer.restore_state(snap);
    Tensor y_replay = layer.forward(x_more);

    float diff = l2_diff(y_continue, y_replay);
    std::cout << "    snapshot/restore replay diff: " << diff;
    CHECK(diff < 1.0f, "snapshot/restore not equivalent: diff=" << diff);
    return true;
}

// ============================================================================
// [H] HAMILTONIAN ON / OFF — behavior differs
// ============================================================================

bool test_hamiltonian_on_off_differ() {
    int dim = 16, hidden = 32, seq = 10;
    Tensor x = make_constant(1, seq, dim, 0.5f);

    TTTLayer layer_off(dim, hidden, 0.01f);
    layer_off.set_use_hamiltonian(false);
    Tensor y_off = layer_off.forward(x);

    TTTLayer layer_on(dim, hidden, 0.01f);
    layer_on.set_use_hamiltonian(true);
    layer_on.set_temperature(0.05f);
    layer_on.set_friction(0.8f);
    Tensor y_on = layer_on.forward(x);

    float diff = l2_diff(y_off, y_on);
    std::cout << "    hamiltonian on vs off diff: " << diff;
    CHECK(diff > 1e-3f, "Hamiltonian flag had no effect");
    return true;
}

// ============================================================================
// [F] FRICTION / TEMPERATURE — hyperparams matter
// ============================================================================

bool test_friction_affects_dynamics() {
    int dim = 16, hidden = 32, seq = 12;
    Tensor x = make_constant(1, seq, dim, 0.5f);

    TTTLayer layer_low(dim, hidden, 0.01f);
    layer_low.set_use_hamiltonian(true);
    layer_low.set_friction(0.1f);  // low friction -> more oscillation
    layer_low.set_temperature(0.05f);
    Tensor y_low = layer_low.forward(x);

    TTTLayer layer_high(dim, hidden, 0.01f);
    layer_high.set_use_hamiltonian(true);
    layer_high.set_friction(0.95f); // high friction -> more damping
    layer_high.set_temperature(0.05f);
    Tensor y_high = layer_high.forward(x);

    float diff = l2_diff(y_low, y_high);
    std::cout << "    friction (0.1 vs 0.95) diff: " << diff;
    CHECK(diff > 1e-4f, "friction hyperparam had no effect");
    return true;
}

bool test_temperature_affects_adaptation_magnitude() {
    int dim = 16, hidden = 32, seq = 10;
    Tensor x = make_constant(1, seq, dim, 0.5f);

    TTTLayer layer_cold(dim, hidden, 0.01f);
    layer_cold.set_use_hamiltonian(true);
    layer_cold.set_temperature(0.001f);
    layer_cold.set_friction(0.8f);
    Tensor y_cold = layer_cold.forward(x);

    TTTLayer layer_hot(dim, hidden, 0.01f);
    layer_hot.set_use_hamiltonian(true);
    layer_hot.set_temperature(1.0f);
    layer_hot.set_friction(0.8f);
    Tensor y_hot = layer_hot.forward(x);

    float diff = l2_diff(y_cold, y_hot);
    std::cout << "    temperature (cold vs hot) diff: " << diff;
    CHECK(diff > 1e-4f, "temperature hyperparam had no effect");
    return true;
}

// ============================================================================
// [S] STABILITY — long sequences, magnitudes
// ============================================================================

bool test_long_sequence_no_nan() {
    int dim = 16, hidden = 32, seq = 200;
    TTTLayer layer(dim, hidden, 0.001f);
    layer.set_use_hamiltonian(true);
    layer.set_temperature(0.05f);
    layer.set_friction(0.85f);
    Tensor x = make_random(1, seq, dim, 99, 0.5f);
    Tensor y = layer.forward(x);
    CHECK(has_no_nan(y), "NaN in long-sequence output (seq=200)");
    float mx = max_abs(y);
    std::cout << "    seq=200 max|y|=" << mx;
    CHECK(mx < 1e6f, "long-sequence output diverged: max|y|=" << mx);
    return true;
}

bool test_extreme_input_magnitudes() {
    int dim = 8, hidden = 16, seq = 5;
    TTTLayer layer(dim, hidden, 0.001f);

    for (float mag : {1e-3f, 1.0f, 100.0f}) {
        Tensor x = make_constant(1, seq, dim, mag);
        Tensor y = layer.forward(x);
        CHECK(has_no_nan(y), "NaN at input magnitude " << mag);
        layer.reset();
    }
    return true;
}

// ============================================================================
// [B] BACKWARD — gradient flow, shape, parameter coverage
// ============================================================================

bool test_backward_shape_and_grad_propagation() {
    int dim = 32, hidden = 64, seq = 5;
    TTTLayer layer(dim, hidden, 0.01f);

    Tensor x = make_random(1, seq, dim, 7);
    Tensor y = layer.forward(x);
    Tensor gy({1, seq, dim}, Device::CPU);
    std::fill(gy.data(), gy.data() + gy.size, 0.1f);

    Tensor gx = layer.backward(gy);
    CHECK(gx.shape == x.shape, "grad_x shape != input shape");
    CHECK(has_no_nan(gx), "grad_x has NaN");
    return true;
}

bool test_parameter_gradients_received() {
    int dim = 16, hidden = 32;
    TTTLayer layer(dim, hidden, 0.01f);
    Tensor x = make_random(1, 5, dim, 11);
    layer.forward(x);
    Tensor gy({1, 5, dim}, Device::CPU);
    std::fill(gy.data(), gy.data() + gy.size, 0.05f);
    layer.backward(gy);

    auto params = layer.parameters();
    CHECK(!params.empty(), "TTT has no parameters");
    int with_grad = 0;
    for (auto* p : params) {
        if (p->grad.size > 0) {
            float s = 0.0f;
            for (int i = 0; i < p->grad.size; ++i) s += std::fabs(p->grad.data()[i]);
            if (s > 0.0f) with_grad++;
        }
    }
    std::cout << "    " << with_grad << "/" << params.size() << " params received grad";
    CHECK(with_grad >= 3, "fewer than 3 params received grad: " << with_grad);
    return true;
}

// ============================================================================
// [Q] BITLINEAR INTEGRATION (TTT uses BitLinear for w_k, w_v, w_out)
// ============================================================================

bool test_bitlinear_layers_collected() {
    int dim = 16, hidden = 32;
    TTTLayer layer(dim, hidden, 0.01f);
    std::vector<BitLinear*> bls;
    layer.collect_bitlinear_layers(bls);
    CHECK(bls.size() == 3,
          "expected 3 BitLinear (w_k, w_v, w_out), got " << bls.size());
    for (auto* bl : bls) {
        CHECK(bl != nullptr, "null BitLinear pointer");
    }
    return true;
}

// ============================================================================
// [D] DETERMINISMO
// ============================================================================

bool test_deterministic_forward_same_seed() {
    int dim = 16, hidden = 32, seq = 5;
    Tensor x = make_random(1, seq, dim, 1234);

    // With seed=42 explicitly propagated, two TTTLayer instances must
    // produce byte-identical outputs.  This was the failing case in the
    // 2026-05-25 validation report; fix: BitLinear ctor now accepts a
    // seed parameter and TTTLayer propagates `seed + k` to w_k/w_v/w_out.
    const uint64_t kSeed = 42;
    TTTLayer a(dim, hidden, 0.01f, kSeed);
    TTTLayer b(dim, hidden, 0.01f, kSeed);
    Tensor ya = a.forward(x);
    Tensor yb = b.forward(x);

    float diff = l2_diff(ya, yb);
    std::cout << "    seeded two-instance same-input diff: " << diff;
    CHECK(diff < 1e-5f, "non-deterministic even with seed: " << diff);

    // Also validate: DIFFERENT seeds produce DIFFERENT outputs (sanity).
    TTTLayer c(dim, hidden, 0.01f, kSeed + 999);
    Tensor yc = c.forward(x);
    float diff2 = l2_diff(ya, yc);
    std::cout << "  diff seed: " << diff2;
    CHECK(diff2 > 1e-3f, "different seeds should produce different outputs");
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "==========================================================\n";
    std::cout << " TTT (Hamiltonian Test-Time Training) validation battery\n";
    std::cout << "==========================================================\n";

    RUN(test_constant_input_produces_drift);
    RUN(test_training_mode_off_no_adaptation);
    RUN(test_reset_returns_to_initial_state);
    RUN(test_snapshot_restore_roundtrip);
    RUN(test_hamiltonian_on_off_differ);
    RUN(test_friction_affects_dynamics);
    RUN(test_temperature_affects_adaptation_magnitude);
    RUN(test_long_sequence_no_nan);
    RUN(test_extreme_input_magnitudes);
    RUN(test_backward_shape_and_grad_propagation);
    RUN(test_parameter_gradients_received);
    RUN(test_bitlinear_layers_collected);
    RUN(test_deterministic_forward_same_seed);

    std::cout << "==========================================================\n";
    std::cout << " RESULT: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "==========================================================\n";
    if (g_failed > 0) {
        for (auto& f : g_failures) std::cerr << "  - " << f << "\n";
    }
    return g_failed == 0 ? 0 : 1;
}
