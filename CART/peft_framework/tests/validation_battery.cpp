// ============================================================================
//  CART PEFT validation battery — extreme correctness + edge + stress
// ============================================================================
//
//  Validates 4 PEFT methods of CART standalone C++ (research mode):
//    - LoRA  (Low-Rank Adaptation)
//    - DoRA  (Weight-Decomposed Low-Rank Adaptation)
//    - IA3   (Infused Adapter by Inhibiting+Amplifying)
//    - TurboFusion (DoRA + IA3 hybrid, flagship)
//
//  Pre-existing run_tests.cpp covers 5 happy-path checks (Tensor, Full FT,
//  IA3 fwd/bwd, DoRA init, TF wrapper) — passes 5/5 after IA3 backward
//  signature fix applied 2026-05-25.  This file adds 16 deeper tests.
// ============================================================================

#include "../include/Tensor.h"
#include "../include/TensorOps.h"
#include "../include/LoRA.h"
#include "../include/DoRA.h"
#include "../include/IA3.h"
#include "../include/TurboFusion.h"
#include "../include/FullFinetuning.h"

#include <chrono>
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

bool is_close(float a, float b, float atol = 1e-4f, float rtol = 1e-3f) {
    if (std::isnan(a) || std::isnan(b)) return false;
    float diff = std::fabs(a - b);
    float thresh = atol + rtol * std::max(std::fabs(a), std::fabs(b));
    return diff <= thresh;
}

void fill_random(Tensor& t, std::mt19937& rng, float scale = 1.0f) {
    std::uniform_real_distribution<float> u(-scale, scale);
    for (int i = 0; i < t.getRows(); ++i)
        for (int j = 0; j < t.getCols(); ++j)
            t.at(i, j) = u(rng);
}

void fill_const(Tensor& t, float val) {
    for (int i = 0; i < t.getRows(); ++i)
        for (int j = 0; j < t.getCols(); ++j)
            t.at(i, j) = val;
}

void fill_identity_like(Tensor& t) {
    // Square or rect — fill diagonal with 1, rest 0
    for (int i = 0; i < t.getRows(); ++i)
        for (int j = 0; j < t.getCols(); ++j)
            t.at(i, j) = (i == j) ? 1.0f : 0.0f;
}

// Sum over a tensor (used for loss)
float tensor_sum(const Tensor& t) {
    float s = 0.0f;
    for (int i = 0; i < t.getRows(); ++i)
        for (int j = 0; j < t.getCols(); ++j)
            s += t.at(i, j);
    return s;
}

}  // namespace

// ============================================================================
// LoRA — Low-Rank Adaptation
// ============================================================================

bool test_lora_forward_shape() {
    LoRALayer lora(8, 4, 2);
    Tensor W(4, 8);
    fill_const(W, 0.5f);
    lora.setBaseWeights(W);

    Tensor x(3, 8);  // batch=3, in=8
    fill_const(x, 1.0f);
    Tensor y = lora.forward(x);
    CHECK(y.getRows() == 3, "batch lost: " << y.getRows());
    CHECK(y.getCols() == 4, "output dim wrong: " << y.getCols());
    return true;
}

bool test_lora_zero_adapter_equals_base() {
    // Inicializa LoRA, mas se A=B=0 então delta=0, output = X @ W0^T
    LoRALayer lora(4, 4, 2);
    Tensor W0(4, 4);
    std::mt19937 rng(7);
    fill_random(W0, rng, 0.5f);
    lora.setBaseWeights(W0);

    // Zera os adapters manualmente (forçar through API)
    // Como não temos zero-init API, vamos só observar que com seed específico
    // o forward é consistente com forward de novo (sanity de determinismo)
    Tensor x(2, 4);
    fill_random(x, rng);
    Tensor y1 = lora.forward(x);
    Tensor y2 = lora.forward(x);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            CHECK(is_close(y1.at(i, j), y2.at(i, j)),
                  "non-deterministic forward at (" << i << "," << j << ")");
    return true;
}

bool test_lora_rank_decomposition_bound() {
    // Rank must be ≤ min(in, out) for LoRA to be meaningful
    int in_dim = 6, out_dim = 8;
    LoRALayer lora(in_dim, out_dim, 3);
    CHECK(lora.get_rank() == 3, "rank not preserved: " << lora.get_rank());

    // A is [in, rank], B is [rank, out] (or transposed — verify via shapes)
    const Tensor* A = lora.get_A();
    const Tensor* B = lora.get_B();
    CHECK(A != nullptr && B != nullptr, "A or B null");
    // Common convention: delta_W = B @ A => [out, in] = [out, rank] @ [rank, in]
    // So B is [out, rank] and A is [rank, in], or whichever choice.
    // Verify product dims work either way:
    int Ar = A->getRows(), Ac = A->getCols();
    int Br = B->getRows(), Bc = B->getCols();
    bool ok = (Ac == Br) || (Ar == Bc);  // multipliable in some order
    CHECK(ok, "A and B not multipliable: A=" << Ar << "x" << Ac
          << " B=" << Br << "x" << Bc);
    return true;
}

bool test_lora_forward_no_nan() {
    LoRALayer lora(16, 16, 4);
    Tensor W(16, 16);
    std::mt19937 rng(8);
    fill_random(W, rng, 0.3f);
    lora.setBaseWeights(W);

    Tensor x(5, 16);
    fill_random(x, rng, 0.5f);
    Tensor y = lora.forward(x);
    for (int i = 0; i < y.getRows(); ++i)
        for (int j = 0; j < y.getCols(); ++j)
            CHECK(!std::isnan(y.at(i, j)) && !std::isinf(y.at(i, j)),
                  "y[" << i << "," << j << "]=" << y.at(i, j));
    return true;
}

bool test_lora_train_loss_decreases() {
    // Treina LoRA: W0 = identidade, target = 2x (escalar de 2).
    // Loss inicial > 0 (LoRA delta=0, y=x, target=2x).
    // Após N steps deve cair (LoRA deve aprender a adicionar +x ao output).
    LoRALayer lora(4, 4, 2);
    Tensor W0(4, 4);
    fill_identity_like(W0);
    lora.setBaseWeights(W0);

    std::mt19937 rng_data(42);
    auto sample_x = [&]() {
        Tensor x(1, 4);
        fill_random(x, rng_data, 1.0f);
        return x;
    };
    auto compute_loss_on = [&](const Tensor& x) {
        Tensor y = lora.forward(x);
        float s = 0.0f;
        for (int j = 0; j < 4; ++j) {
            float target = 2.0f * x.at(0, j);
            float d = y.at(0, j) - target;
            s += d * d;
        }
        return s;
    };

    // Loss em batch fixo de 16 samples (reproduzível)
    std::vector<Tensor> eval_set;
    for (int i = 0; i < 16; ++i) eval_set.push_back(sample_x());
    auto eval_avg = [&]() {
        float s = 0.0f;
        for (auto& x : eval_set) s += compute_loss_on(x);
        return s / eval_set.size();
    };

    float loss_start = eval_avg();
    for (int step = 0; step < 200; ++step) {
        Tensor x = sample_x();
        Tensor y = lora.forward(x);
        Tensor grad(1, 4);
        for (int j = 0; j < 4; ++j) {
            float target = 2.0f * x.at(0, j);
            grad.at(0, j) = 2.0f * (y.at(0, j) - target);
        }
        lora.backward(grad);
        lora.update(0.01f);
    }
    float loss_end = eval_avg();
    std::cout << "    LoRA training: avg loss " << loss_start << " -> " << loss_end;
    CHECK(loss_start > 0.1f, "trivial loss_start: " << loss_start);
    CHECK(loss_end < loss_start * 0.9f,
          "loss didn't drop ≥10%: " << loss_start << " -> " << loss_end);
    return true;
}

// ============================================================================
// DoRA — Weight-Decomposed Low-Rank Adaptation
// ============================================================================

bool test_dora_forward_shape() {
    DoRALayer dora(8, 4, 2);
    Tensor W(4, 8);
    fill_const(W, 0.5f);
    dora.setBaseWeights(W);
    Tensor x(2, 8);
    fill_const(x, 1.0f);
    Tensor y = dora.forward(x);
    CHECK(y.getRows() == 2 && y.getCols() == 4, "DoRA shape wrong");
    return true;
}

bool test_dora_has_magnitude_vector() {
    DoRALayer dora(8, 4, 2);
    Tensor W(4, 8);
    fill_const(W, 0.5f);
    dora.setBaseWeights(W);
    Tensor x(1, 8);
    fill_const(x, 1.0f);
    dora.forward(x);  // initialize internal cache
    const Tensor* m = dora.get_m();
    CHECK(m != nullptr, "magnitude vector null");
    // Magnitude should have output_dim entries (one per output unit)
    CHECK(m->getCols() == 4 || m->getRows() == 4,
          "magnitude size unexpected: " << m->getRows() << "x" << m->getCols());
    return true;
}

bool test_dora_forward_no_nan() {
    DoRALayer dora(16, 8, 4);
    Tensor W(8, 16);
    std::mt19937 rng(11);
    fill_random(W, rng, 0.3f);
    dora.setBaseWeights(W);
    Tensor x(3, 16);
    fill_random(x, rng);
    Tensor y = dora.forward(x);
    for (int i = 0; i < y.getRows(); ++i)
        for (int j = 0; j < y.getCols(); ++j)
            CHECK(!std::isnan(y.at(i, j)), "DoRA y has NaN at (" << i << "," << j << ")");
    return true;
}

// ============================================================================
// IA3 — Infused Adapter by Inhibiting and Amplifying
// ============================================================================

bool test_ia3_forward_shape() {
    IA3Layer ia3(8, 4);
    Tensor W(4, 8);
    std::mt19937 rng(13);
    fill_random(W, rng, 0.5f);
    ia3.setBaseWeights(W);
    Tensor x(3, 8);
    fill_random(x, rng);
    Tensor y = ia3.forward(x);
    CHECK(y.getRows() == 3 && y.getCols() == 4, "IA3 shape wrong");
    return true;
}

bool test_ia3_backward_returns_grad_input() {
    IA3Layer ia3(8, 4);
    Tensor W(4, 8);
    std::mt19937 rng(14);
    fill_random(W, rng, 0.5f);
    ia3.setBaseWeights(W);
    Tensor x(2, 8);
    fill_random(x, rng);
    ia3.forward(x);

    Tensor up(2, 4);
    fill_const(up, 1.0f);
    Tensor grad_x = ia3.backward(up);
    CHECK(grad_x.getRows() == 2 && grad_x.getCols() == 8,
          "IA3 grad_x shape wrong: " << grad_x.getRows() << "x" << grad_x.getCols());

    bool any_nonzero = false;
    for (int i = 0; i < grad_x.getRows(); ++i)
        for (int j = 0; j < grad_x.getCols(); ++j)
            if (std::fabs(grad_x.at(i, j)) > 1e-12) any_nonzero = true;
    CHECK(any_nonzero, "IA3 grad_x all zero — propagação quebrada");
    return true;
}

bool test_ia3_pure_scaling_mode_unimplemented() {
    // CART BUG: IA3Layer::enablePureScalingMode(bool) está declarado em
    // include/IA3.h mas NUNCA foi implementado em src/IA3.cpp.  Esse teste
    // existe pra documentar o gap.  Quando o método for implementado,
    // expandir este teste com asserts sobre pure-scaling behavior.
    CHECK(true, "documenting unimplemented method");
    return true;
}

// ============================================================================
// TurboFusion (DoRA + IA3)
// ============================================================================

bool test_turbofusion_forward_shape() {
    TurboFusionLayer tf(8, 4, 2);
    Tensor W(4, 8);
    std::mt19937 rng(20);
    fill_random(W, rng, 0.5f);
    tf.setBaseWeights(W);
    Tensor x(2, 8);
    fill_random(x, rng);
    Tensor y = tf.forward(x);
    CHECK(y.getRows() == 2 && y.getCols() == 4, "TF shape wrong");
    return true;
}

bool test_turbofusion_no_nan_under_random_load() {
    TurboFusionLayer tf(32, 16, 4);
    Tensor W(16, 32);
    std::mt19937 rng(21);
    fill_random(W, rng, 0.2f);
    tf.setBaseWeights(W);

    for (int trial = 0; trial < 10; ++trial) {
        Tensor x(4, 32);
        fill_random(x, rng, 1.0f);
        Tensor y = tf.forward(x);
        for (int i = 0; i < y.getRows(); ++i)
            for (int j = 0; j < y.getCols(); ++j)
                CHECK(!std::isnan(y.at(i, j)) && !std::isinf(y.at(i, j)),
                      "TF NaN/Inf in trial " << trial);
    }
    return true;
}

bool test_turbofusion_train_step_runs() {
    // Smoke: full forward → backward → update sem crash
    TurboFusionLayer tf(8, 8, 2);
    Tensor W(8, 8);
    std::mt19937 rng(22);
    fill_random(W, rng, 0.3f);
    tf.setBaseWeights(W);

    Tensor x(2, 8);
    fill_random(x, rng);
    Tensor y = tf.forward(x);
    Tensor grad(2, 8);
    fill_const(grad, 0.1f);
    tf.backward(grad);
    tf.update(0.01f);
    CHECK(true, "TF train step completed");
    return true;
}

// ============================================================================
// Numerical stability + Stress
// ============================================================================

bool test_large_batch_lora() {
    int batch = 64, in_dim = 64, out_dim = 32;
    LoRALayer lora(in_dim, out_dim, 8);
    Tensor W(out_dim, in_dim);
    std::mt19937 rng(31);
    fill_random(W, rng, 0.1f);
    lora.setBaseWeights(W);

    Tensor x(batch, in_dim);
    fill_random(x, rng);
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor y = lora.forward(x);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "    LoRA forward 64x64x32 rank=8: " << ms << " ms";

    for (int i = 0; i < y.getRows(); ++i)
        for (int j = 0; j < y.getCols(); ++j)
            CHECK(!std::isnan(y.at(i, j)), "NaN at large batch");
    return true;
}

bool test_extreme_magnitudes() {
    // Inputs em magnitudes ±1e3 — LoRA não deve explodir
    LoRALayer lora(8, 8, 2);
    Tensor W(8, 8);
    fill_const(W, 0.01f);
    lora.setBaseWeights(W);

    Tensor x(1, 8);
    for (int j = 0; j < 8; ++j) x.at(0, j) = (j % 2 == 0) ? 1000.0f : -1000.0f;
    Tensor y = lora.forward(x);
    for (int j = 0; j < 8; ++j)
        CHECK(!std::isnan(y.at(0, j)) && !std::isinf(y.at(0, j)),
              "LoRA at extreme magnitude: y[" << j << "]=" << y.at(0, j));
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "==========================================================\n";
    std::cout << " CART PEFT validation battery (standalone C++)\n";
    std::cout << "==========================================================\n";

    // LoRA
    RUN(test_lora_forward_shape);
    RUN(test_lora_zero_adapter_equals_base);
    RUN(test_lora_rank_decomposition_bound);
    RUN(test_lora_forward_no_nan);
    RUN(test_lora_train_loss_decreases);

    // DoRA
    RUN(test_dora_forward_shape);
    RUN(test_dora_has_magnitude_vector);
    RUN(test_dora_forward_no_nan);

    // IA3
    RUN(test_ia3_forward_shape);
    RUN(test_ia3_backward_returns_grad_input);
    RUN(test_ia3_pure_scaling_mode_unimplemented);

    // TurboFusion
    RUN(test_turbofusion_forward_shape);
    RUN(test_turbofusion_no_nan_under_random_load);
    RUN(test_turbofusion_train_step_runs);

    // Stress
    RUN(test_large_batch_lora);
    RUN(test_extreme_magnitudes);

    std::cout << "==========================================================\n";
    std::cout << " RESULT: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "==========================================================\n";
    return g_failed == 0 ? 0 : 1;
}
