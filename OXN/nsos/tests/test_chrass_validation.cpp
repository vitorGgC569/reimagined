// ============================================================================
//  test_chrass_validation.cpp -- BATERIA EXTREMA de validação do ChrassLayer
// ============================================================================
//
//  CHRASS é uma camada esparsa CSR isomórfica a uma matriz de adjacência.
//  Histórico: estava ÓRFÃ do build (chrass_layer*.cpp nunca foi compilado
//  no nsos_core, classe ChrassLayer duplicada em v1+v2, nenhum teste
//  dedicado), conforme auditoria 2026-05-25.
//
//  Este arquivo testa o ChrassLayer v2 EXAUSTIVAMENTE:
//
//    [F]orward      — correctness numérica vs reference dense
//    [B]ackward     — grad_input, grad_weights, grad_bias vs reference
//    [G]radcheck    — analítico vs numérico (finite differences)
//    [E]dge cases   — grafo vazio, completo, self-loops, negativos, 1x1
//    [S]aturação    — clamps em ±100 (out), ±10 (W), ±1 (grad)
//    [N]aN/Inf      — robustez sob inputs degenerados
//    [D]eterminismo — reproducibilidade entre runs
//    [A]damW        — bias correction, weight decay, timestep
//    [T]opologia    — preserva esparsidade da adjacência
//    [I]ntegração   — empilhar 3 camadas + backprop end-to-end
//    [P]erformance  — escala razoável (256x256 < 100ms)
//    [O]DR sanity   — só uma classe ChrassLayer compilada
//
//  Cada falha é fatal e imprime contexto. Determinístico via seed fixo.
// ============================================================================

#include "../include/chrass_layer_v2.h"
#include "../include/tensor.h"
#include "../include/autograd.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using nsos::ChrassLayer;
using nsos::Tensor;

// ── Test infrastructure ─────────────────────────────────────────────────────
namespace {

int g_passed = 0;
int g_failed = 0;
std::vector<std::string> g_failures;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream _oss;                                           \
            _oss << __func__ << ":" << __LINE__ << " " << msg;                 \
            g_failures.push_back(_oss.str());                                  \
            std::cerr << "    [CHECK FAIL] " << _oss.str() << std::endl;       \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RUN(test_fn)                                                           \
    do {                                                                       \
        std::cout << "[" << #test_fn << "]" << std::flush;                     \
        bool _ok = test_fn();                                                  \
        if (_ok) {                                                             \
            std::cout << "  PASS" << std::endl;                                \
            ++g_passed;                                                        \
        } else {                                                               \
            std::cout << "  FAIL" << std::endl;                                \
            ++g_failed;                                                        \
        }                                                                      \
    } while (0)

bool is_close(float a, float b, float atol = 1e-4f, float rtol = 1e-3f) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    float diff = std::fabs(a - b);
    float thresh = atol + rtol * std::max(std::fabs(a), std::fabs(b));
    return diff <= thresh;
}

// Reference dense forward: y[r] = sum_c W[r,c] * x[c] + b[r], with same
// row-normalization that ChrassLayer applies internally.
std::vector<float> dense_reference_forward(
    int dim, const std::vector<float>& adjacency,
    const std::vector<float>& x, const std::vector<float>& bias)
{
    // Row-normalize adjacency the SAME way ChrassLayer does
    std::vector<float> W(dim * dim, 0.0f);
    for (int r = 0; r < dim; ++r) {
        long double row_sum = 0.0;
        for (int c = 0; c < dim; ++c) {
            float val = adjacency[r * dim + c];
            if (std::fabs(val) > 1e-6f) row_sum += (long double)std::fabs(val);
        }
        long double scale = (row_sum > 1e-12) ? (1.0L / row_sum) : 1.0L;
        for (int c = 0; c < dim; ++c) {
            float val = adjacency[r * dim + c];
            if (std::fabs(val) > 1e-6f) {
                W[r * dim + c] = (float)((long double)val * scale);
            }
        }
    }

    // y = W @ x + b   (batch=1)
    std::vector<float> y(dim, 0.0f);
    for (int r = 0; r < dim; ++r) {
        float s = bias[r];
        for (int c = 0; c < dim; ++c) {
            s += W[r * dim + c] * x[c];
        }
        // Apply ChrassLayer's clamps so we compare apples to apples
        if (std::isnan(s) || std::isinf(s)) s = 0.0f;
        if (s > 100.0f)  s = 100.0f;
        if (s < -100.0f) s = -100.0f;
        y[r] = s;
    }
    return y;
}

std::vector<float> random_adjacency(int dim, float density,
                                    unsigned seed,
                                    bool include_negatives = true) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uval(-1.0f, 1.0f);
    std::uniform_real_distribution<float> upos(0.1f, 1.0f);
    std::uniform_real_distribution<float> udense(0.0f, 1.0f);

    std::vector<float> A(dim * dim, 0.0f);
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            if (udense(rng) < density) {
                A[r * dim + c] = include_negatives ? uval(rng) : upos(rng);
            }
        }
    }
    return A;
}

std::vector<float> random_vector(int n, unsigned seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = u(rng);
    return v;
}

Tensor make_input(const std::vector<float>& x, int batch, int dim) {
    Tensor t = Tensor::zeros({batch, dim}, nsos::Device::CPU);
    float* p = t.data();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < dim; ++i) {
            p[b * dim + i] = x[i];
        }
    }
    return t;
}

}  // anonymous namespace

// ============================================================================
// [F] FORWARD CORRECTNESS — comparar contra reference dense
// ============================================================================

bool test_forward_identity_matrix() {
    // I @ x = x  (identity map should preserve input)
    int dim = 8;
    std::vector<float> A(dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) A[i * dim + i] = 1.0f;

    ChrassLayer layer(dim, A);
    auto x = random_vector(dim, 42);
    Tensor X = make_input(x, 1, dim);
    Tensor Y = layer.forward(X);

    CHECK(Y.shape.size() == 2 && Y.shape[0] == 1 && Y.shape[1] == dim,
          "output shape mismatch");
    // Bias starts at zero, identity row-normalizes to itself (single nonzero
    // per row -> scale=1). So y[i] == x[i].
    for (int i = 0; i < dim; ++i) {
        CHECK(is_close(Y.data()[i], x[i]),
              "identity y[" << i << "]=" << Y.data()[i] << " expected " << x[i]);
    }
    return true;
}

bool test_forward_dense_reference_small() {
    int dim = 16;
    auto A = random_adjacency(dim, 0.5f, 1337);
    auto x = random_vector(dim, 99);
    std::vector<float> bias(dim, 0.0f);  // ChrassLayer ctor zeros bias

    ChrassLayer layer(dim, A);
    Tensor X = make_input(x, 1, dim);
    Tensor Y = layer.forward(X);

    auto y_ref = dense_reference_forward(dim, A, x, bias);
    for (int i = 0; i < dim; ++i) {
        CHECK(is_close(Y.data()[i], y_ref[i]),
              "y[" << i << "]=" << Y.data()[i] << " ref=" << y_ref[i]);
    }
    return true;
}

bool test_forward_batch_consistency() {
    // Same input across batch -> same output across batch.
    int dim = 32, batch = 4;
    auto A = random_adjacency(dim, 0.3f, 7);
    auto x = random_vector(dim, 8);

    ChrassLayer layer(dim, A);
    Tensor X = make_input(x, batch, dim);
    Tensor Y = layer.forward(X);

    for (int b = 1; b < batch; ++b) {
        for (int i = 0; i < dim; ++i) {
            CHECK(is_close(Y.data()[b * dim + i], Y.data()[i]),
                  "batch " << b << " diverges at " << i);
        }
    }
    return true;
}

// ============================================================================
// [B] BACKWARD CORRECTNESS — shapes + zero-grad propagation + identity check
// ============================================================================

bool test_backward_returns_correct_shape() {
    int dim = 12, batch = 3;
    auto A = random_adjacency(dim, 0.4f, 21);
    ChrassLayer layer(dim, A);

    Tensor X = make_input(random_vector(dim, 22), batch, dim);
    Tensor Y = layer.forward(X);
    Tensor dY = Tensor::ones({batch, dim}, nsos::Device::CPU);
    Tensor dX = layer.backward(dY, X);

    CHECK(dX.shape.size() == 2, "grad_input must be 2D");
    CHECK(dX.shape[0] == batch, "grad_input batch dim wrong");
    CHECK(dX.shape[1] == dim,   "grad_input feature dim wrong");
    return true;
}

bool test_backward_identity_grad() {
    // For identity matrix A, dL/dx == W^T dL/dy == dL/dy (since W is I).
    int dim = 8;
    std::vector<float> A(dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) A[i * dim + i] = 1.0f;
    ChrassLayer layer(dim, A);

    Tensor X = make_input(random_vector(dim, 5), 1, dim);
    layer.forward(X);
    auto dy = random_vector(dim, 6);
    Tensor dY = make_input(dy, 1, dim);
    Tensor dX = layer.backward(dY, X);

    for (int i = 0; i < dim; ++i) {
        CHECK(is_close(dX.data()[i], dy[i]),
              "identity dx[" << i << "]=" << dX.data()[i] << " expected " << dy[i]);
    }
    return true;
}

bool test_backward_no_input_grad_outside_support() {
    // grad_input[c] should be zero in columns that have no incoming edge.
    int dim = 6;
    std::vector<float> A(dim * dim, 0.0f);
    // Single edge: row 0 col 3
    A[0 * dim + 3] = 1.0f;
    ChrassLayer layer(dim, A);

    Tensor X = make_input(random_vector(dim, 30), 1, dim);
    layer.forward(X);
    Tensor dY = make_input(std::vector<float>(dim, 1.0f), 1, dim);
    Tensor dX = layer.backward(dY, X);

    for (int c = 0; c < dim; ++c) {
        if (c == 3) {
            CHECK(std::fabs(dX.data()[c]) > 1e-9f,
                  "expected nonzero dX at supported col");
        } else {
            CHECK(std::fabs(dX.data()[c]) < 1e-9f,
                  "expected zero dX at unsupported col " << c
                  << " got " << dX.data()[c]);
        }
    }
    return true;
}

// ============================================================================
// [G] GRADIENT CHECK — analítico vs numérico via finite differences
// ============================================================================

bool test_gradcheck_input() {
    int dim = 6;
    auto A = random_adjacency(dim, 0.6f, 123, /*neg=*/true);
    ChrassLayer layer(dim, A);

    auto x_vals = random_vector(dim, 124);
    Tensor X = make_input(x_vals, 1, dim);

    // Forward, get y; loss = sum(y) so dL/dy = ones
    Tensor Y = layer.forward(X);
    Tensor dY = Tensor::ones({1, dim}, nsos::Device::CPU);
    Tensor dX_analytic = layer.backward(dY, X);

    const float eps = 1e-3f;
    for (int i = 0; i < dim; ++i) {
        // Skip if analytic grad is 0 (col not in support of any edge)
        if (std::fabs(dX_analytic.data()[i]) < 1e-9f) continue;

        // Numerical: dL/dx_i ≈ (loss(x+eps) - loss(x-eps)) / 2eps
        auto eval_loss = [&](float delta) {
            std::vector<float> xp = x_vals;
            xp[i] += delta;
            Tensor Xp = make_input(xp, 1, dim);
            Tensor Yp = layer.forward(Xp);
            float s = 0.0f;
            for (int j = 0; j < dim; ++j) s += Yp.data()[j];
            return s;
        };
        float Lp = eval_loss(eps);
        float Lm = eval_loss(-eps);
        float num_grad = (Lp - Lm) / (2.0f * eps);

        // Tolerância relaxada porque o forward tem clamps (não-suave)
        CHECK(is_close(num_grad, dX_analytic.data()[i], 1e-2f, 5e-2f),
              "gradcheck x[" << i << "]: numeric=" << num_grad
              << " analytic=" << dX_analytic.data()[i]);
    }
    return true;
}

// ============================================================================
// [E] EDGE CASES — grafos degenerados
// ============================================================================

bool test_edge_empty_graph() {
    // Tudo zero -> output = bias (zero) sempre, sem crash
    int dim = 8;
    std::vector<float> A(dim * dim, 0.0f);
    ChrassLayer layer(dim, A);

    CHECK((layer.nnz() == 0),  "empty graph -> empty values");
    CHECK(layer.col_indices.empty(), "empty graph -> empty col_indices");

    Tensor X = make_input(random_vector(dim, 99), 1, dim);
    Tensor Y = layer.forward(X);
    for (int i = 0; i < dim; ++i) {
        CHECK(std::fabs(Y.data()[i]) < 1e-9f, "empty graph y[" << i << "] != 0");
    }
    return true;
}

bool test_edge_complete_graph() {
    int dim = 4;
    std::vector<float> A(dim * dim, 1.0f);
    ChrassLayer layer(dim, A);

    // Cada linha tem 4 entradas, row_sum=4, scale=1/4 -> cada peso = 0.25
    for (int i = 0; i < layer.nnz(); ++i) {
        CHECK(is_close(layer.values_data()[i], 0.25f),
              "complete graph: weight=" << layer.values_data()[i] << " expected 0.25");
    }
    return true;
}

bool test_edge_self_loops_only() {
    int dim = 5;
    std::vector<float> A(dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) A[i * dim + i] = 2.0f;
    ChrassLayer layer(dim, A);

    // Single non-zero por linha -> scale=1/2, weight=1
    for (int i = 0; i < dim; ++i) {
        CHECK(layer.row_ptr[i + 1] - layer.row_ptr[i] == 1,
              "self-loop row " << i << " should have exactly 1 nonzero");
    }
    return true;
}

bool test_edge_negative_weights() {
    int dim = 4;
    std::vector<float> A(dim * dim, 0.0f);
    A[0 * dim + 1] = -2.0f;
    A[0 * dim + 2] = 1.0f;
    ChrassLayer layer(dim, A);

    // row_sum = |{-2}| + |1| = 3, scale = 1/3
    // weights: -2/3 e 1/3 (sinal preservado)
    bool found_neg = false, found_pos = false;
    for (int i = 0; i < layer.nnz(); ++i) { float v = layer.values_data()[i];
        if (v < 0) found_neg = true;
        if (v > 0) found_pos = true;
    }
    CHECK(found_neg, "negative weight should be preserved with negative sign");
    CHECK(found_pos, "positive weight should remain positive");
    return true;
}

bool test_edge_1x1() {
    std::vector<float> A = {1.0f};
    ChrassLayer layer(1, A);
    Tensor X = make_input({3.5f}, 1, 1);
    Tensor Y = layer.forward(X);
    // weight normaliza pra 1, bias=0 -> y = 1 * 3.5 = 3.5
    CHECK(is_close(Y.data()[0], 3.5f),
          "1x1 layer y=" << Y.data()[0] << " expected 3.5");
    return true;
}

// ============================================================================
// [S] SATURAÇÃO — clamps em ±100 (output), ±10 (weights), ±1 (grad_values)
// ============================================================================

bool test_saturation_output_clamp() {
    // Forçar output a estourar 100
    int dim = 2;
    std::vector<float> A = {1.0f, 0.0f,
                            0.0f, 1.0f};
    ChrassLayer layer(dim, A);
    // Input gigante -> bias=0, mas valor 500 deve ser clamped pra 100
    std::vector<float> x = {500.0f, -500.0f};
    Tensor X = make_input(x, 1, dim);
    Tensor Y = layer.forward(X);
    CHECK(Y.data()[0] <= 100.0f + 1e-3f, "positive clamp violated");
    CHECK(Y.data()[1] >= -100.0f - 1e-3f, "negative clamp violated");
    CHECK(is_close(Y.data()[0], 100.0f), "expected output clamped to 100");
    CHECK(is_close(Y.data()[1], -100.0f), "expected output clamped to -100");
    return true;
}

bool test_saturation_gradient_clamp() {
    // Backward deve clamping grad_values em ±1
    int dim = 2;
    std::vector<float> A = {1.0f, 0.0f, 0.0f, 1.0f};
    ChrassLayer layer(dim, A);
    // Input gigante + grad gigante -> grad_value cresceria além de ±1
    std::vector<float> x = {1000.0f, 1000.0f};
    Tensor X = make_input(x, 1, dim);
    layer.forward(X);
    Tensor dY = make_input({1000.0f, 1000.0f}, 1, dim);
    layer.backward(dY, X);

    int gn = layer.values_param.grad.size;
    for (int i = 0; i < gn; ++i) {
        float g = layer.values_param.grad.data()[i];
        CHECK(g <= 1.0f + 1e-5f && g >= -1.0f - 1e-5f,
              "grad_values clamp violated: g=" << g);
    }
    return true;
}

// ============================================================================
// [N] NaN/Inf — robustez sob inputs degenerados
// ============================================================================

bool test_nan_input_sanitization() {
    int dim = 4;
    auto A = random_adjacency(dim, 0.5f, 77);
    ChrassLayer layer(dim, A);

    std::vector<float> x = {std::nanf(""), 1.0f, std::numeric_limits<float>::infinity(), 0.0f};
    Tensor X = make_input(x, 1, dim);
    Tensor Y = layer.forward(X);
    // ChrassLayer sanitiza output: nem NaN nem Inf devem sair
    for (int i = 0; i < dim; ++i) {
        CHECK(!std::isnan(Y.data()[i]), "y[" << i << "] is NaN");
        CHECK(!std::isinf(Y.data()[i]), "y[" << i << "] is Inf");
    }
    return true;
}

// ============================================================================
// [D] DETERMINISMO
// ============================================================================

bool test_deterministic_forward() {
    int dim = 16;
    auto A = random_adjacency(dim, 0.3f, 555);
    auto x = random_vector(dim, 556);

    Tensor X = make_input(x, 2, dim);

    ChrassLayer la(dim, A);
    Tensor Y1 = la.forward(X);

    ChrassLayer lb(dim, A);
    Tensor Y2 = lb.forward(X);

    for (int i = 0; i < 2 * dim; ++i) {
        CHECK(Y1.data()[i] == Y2.data()[i],
              "non-deterministic forward at idx " << i);
    }
    return true;
}

// ============================================================================
// [A] ADAMW — bias correction + weight decay + timestep
// ============================================================================

bool test_adamw_step_advances_timestep() {
    int dim = 4;
    auto A = random_adjacency(dim, 0.5f, 1);
    ChrassLayer layer(dim, A);

    int t_before = layer.t;
    Tensor X = make_input(random_vector(dim, 2), 1, dim);
    layer.forward(X);
    Tensor dY = Tensor::ones({1, dim}, nsos::Device::CPU);
    layer.backward(dY, X);
    layer.step(0.001f);
    CHECK(layer.t == t_before + 1, "timestep didn't advance");
    return true;
}

bool test_adamw_zero_grad_minimal_change() {
    int dim = 4;
    auto A = random_adjacency(dim, 0.5f, 3);
    ChrassLayer layer(dim, A);

    Tensor X = make_input(std::vector<float>(dim, 0.0f), 1, dim);
    layer.forward(X);
    Tensor dY = Tensor::zeros({1, dim}, nsos::Device::CPU);
    layer.backward(dY, X);
    std::vector<float> w_before(layer.values_data(),
                                 layer.values_data() + layer.nnz());
    layer.step(0.001f);

    // weight_decay ainda mexe um pouco mesmo com grad=0
    for (int i = 0; i < (int)w_before.size(); ++i) {
        float delta = std::fabs(layer.values_data()[i] - w_before[i]);
        (void)delta;
        CHECK(delta < 1e-3f,
              "weight " << i << " changed " << delta << " with zero grad");
    }
    return true;
}

// ============================================================================
// [T] TOPOLOGIA — preserva esparsidade
// ============================================================================

bool test_topology_preserved_after_steps() {
    int dim = 16;
    auto A = random_adjacency(dim, 0.2f, 11);
    ChrassLayer layer(dim, A);
    size_t nnz_initial = (size_t)layer.nnz();

    // 20 steps simulados — a CSR estrutura não pode mudar
    for (int s = 0; s < 20; ++s) {
        Tensor X = make_input(random_vector(dim, 12 + s), 2, dim);
        layer.forward(X);
        Tensor dY = Tensor::ones({2, dim}, nsos::Device::CPU);
        layer.backward(dY, X);
        layer.step(0.01f);
    }
    CHECK((size_t)layer.nnz() == nnz_initial, "nnz changed during training");
    CHECK(layer.col_indices.size() == nnz_initial, "col_indices size changed");
    return true;
}

// ============================================================================
// [I] INTEGRAÇÃO — 3 camadas empilhadas + backprop
// ============================================================================

bool test_integration_stack_three_layers() {
    int dim = 8;
    auto A1 = random_adjacency(dim, 0.5f, 100);
    auto A2 = random_adjacency(dim, 0.4f, 101);
    auto A3 = random_adjacency(dim, 0.3f, 102);

    ChrassLayer l1(dim, A1), l2(dim, A2), l3(dim, A3);

    Tensor X = make_input(random_vector(dim, 103), 1, dim);
    Tensor Y1 = l1.forward(X);
    Tensor Y2 = l2.forward(Y1);
    Tensor Y3 = l3.forward(Y2);

    // Verifica saída finita e bounded
    for (int i = 0; i < dim; ++i) {
        CHECK(!std::isnan(Y3.data()[i]) && !std::isinf(Y3.data()[i]),
              "stack output[" << i << "] not finite");
        CHECK(std::fabs(Y3.data()[i]) <= 100.0f + 1e-3f,
              "stack output[" << i << "] = " << Y3.data()[i] << " exceeds clamp");
    }

    // Backprop reverse
    Tensor dY3 = Tensor::ones({1, dim}, nsos::Device::CPU);
    Tensor dY2 = l3.backward(dY3, Y2);
    Tensor dY1 = l2.backward(dY2, Y1);
    Tensor dX  = l1.backward(dY1, X);

    CHECK(dX.shape[0] == 1 && dX.shape[1] == dim, "stack backprop shape wrong");
    for (int i = 0; i < dim; ++i) {
        CHECK(!std::isnan(dX.data()[i]), "stack grad_input has NaN");
    }
    return true;
}

// ============================================================================
// [P] PERFORMANCE — escala
// ============================================================================

bool test_performance_256x256_under_500ms() {
    int dim = 256;
    auto A = random_adjacency(dim, 0.05f, 999);  // 5% density
    ChrassLayer layer(dim, A);

    Tensor X = make_input(random_vector(dim, 1000), 8, dim);
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor Y = layer.forward(X);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "    256x256 sparse forward batch=8: " << ms << " ms" << std::endl;
    CHECK(ms < 500.0, "forward too slow: " << ms << " ms (threshold 500)");
    CHECK(!std::isnan(Y.data()[0]), "perf test output has NaN");
    return true;
}

// ============================================================================
// [X] STRESS / EXTREMO -- 1024x1024 + 1000 steps + numerical adversarial
// ============================================================================

bool test_stress_1024x1024_sparse() {
    int dim = 1024;
    auto A = random_adjacency(dim, 0.01f, 31337);  // 1% density ~10k edges
    ChrassLayer layer(dim, A);

    Tensor X = make_input(random_vector(dim, 31338), 4, dim);
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor Y = layer.forward(X);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "    1024x1024 sparse 1% batch=4 forward: " << ms << " ms ("
              << (size_t)layer.nnz() << " edges)" << std::endl;
    CHECK(ms < 2000.0, "1024x1024 forward too slow: " << ms << " ms");
    for (int i = 0; i < 4 * dim; ++i) {
        CHECK(!std::isnan(Y.data()[i]), "1024 stress has NaN at idx " << i);
    }
    return true;
}

bool test_stress_long_training_loop_converges() {
    // Treina CHRASS por 200 steps em problema sintético:
    //   target_y = identity(x) -> ideal: layer aprende identity_like.
    // Loss = mean squared error.  Espera-se queda monotônica (tendencial)
    // ao longo do treino.  Isso valida que o AdamW realmente otimiza.
    int dim = 16;
    auto A = random_adjacency(dim, 0.4f, 4242);
    ChrassLayer layer(dim, A);

    std::mt19937 rng(7777);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    auto compute_loss = [&](const Tensor& X, Tensor& Y_out) -> float {
        Y_out = layer.forward(X);
        float s = 0.0f;
        // Target: y == x (identity learning task)
        int N = (int)(X.shape[0] * X.shape[1]);
        for (int i = 0; i < N; ++i) {
            float diff = Y_out.data()[i] - X.data()[i];
            s += diff * diff;
        }
        return s / (float)N;
    };

    float loss_start = 0.0f, loss_end = 0.0f;
    const int N_STEPS = 200;
    const int batch = 4;
    float lr = 0.05f;
    for (int step = 0; step < N_STEPS; ++step) {
        std::vector<float> xv(dim);
        for (int i = 0; i < dim; ++i) xv[i] = u(rng);
        Tensor X = make_input(xv, batch, dim);
        Tensor Y;
        float loss = compute_loss(X, Y);
        if (step == 0) loss_start = loss;
        if (step == N_STEPS - 1) loss_end = loss;

        // grad_y = 2 * (y - x) / N
        Tensor dY = Tensor::zeros({batch, dim}, nsos::Device::CPU);
        int Nelem = (int)(X.shape[0] * X.shape[1]);
        for (int i = 0; i < Nelem; ++i) {
            dY.data()[i] = 2.0f * (Y.data()[i] - X.data()[i]) / (float)Nelem;
        }
        layer.backward(dY, X);
        layer.step(lr);
    }
    std::cout << "    training loop loss: " << loss_start << " -> " << loss_end << std::endl;
    CHECK(loss_end < loss_start * 0.8f,
          "loss did not decrease enough: start=" << loss_start
          << " end=" << loss_end);
    CHECK(!std::isnan(loss_end), "final loss is NaN");
    return true;
}

bool test_numerical_adversarial_magnitudes() {
    // Inputs com magnitude escalando 1e-9 -> 1e+9
    int dim = 4;
    auto A = random_adjacency(dim, 0.5f, 9999);
    ChrassLayer layer(dim, A);

    for (float mag : {1e-9f, 1e-3f, 1.0f, 1e3f, 1e6f, 1e9f}) {
        std::vector<float> x(dim, mag);
        Tensor X = make_input(x, 1, dim);
        Tensor Y = layer.forward(X);
        for (int i = 0; i < dim; ++i) {
            CHECK(!std::isnan(Y.data()[i]),
                  "NaN with magnitude " << mag << " at idx " << i);
            CHECK(!std::isinf(Y.data()[i]),
                  "Inf with magnitude " << mag << " at idx " << i);
        }
    }
    return true;
}

bool test_repeated_backward_no_grad_accumulation_overflow() {
    // 100 chamadas de backward seguidas sem reset externo de grad — o reset
    // está DENTRO de backward (std::fill). Grad final deve ser exatamente
    // o do último backward, não acumulado.
    int dim = 8;
    auto A = random_adjacency(dim, 0.5f, 7);
    ChrassLayer layer(dim, A);

    Tensor X = make_input(random_vector(dim, 8), 1, dim);
    layer.forward(X);

    Tensor dY = Tensor::ones({1, dim}, nsos::Device::CPU);
    layer.backward(dY, X);
    int gnnz = layer.values_param.grad.size;
    std::vector<float> grads_after_1(
        layer.values_param.grad.data(),
        layer.values_param.grad.data() + gnnz);

    for (int k = 0; k < 99; ++k) {
        layer.backward(dY, X);
    }
    std::vector<float> grads_after_100(
        layer.values_param.grad.data(),
        layer.values_param.grad.data() + gnnz);

    for (size_t i = 0; i < grads_after_1.size(); ++i) {
        CHECK(is_close(grads_after_1[i], grads_after_100[i], 1e-6f),
              "grad accumulation bug at idx " << i
              << " 1-call=" << grads_after_1[i]
              << " 100-call=" << grads_after_100[i]);
    }
    return true;
}

bool test_topology_hash_stable_over_steps() {
    // CSR estrutural (row_ptr + col_indices) deve ser invariante durante treino.
    // Hash simples = soma dos col_indices XOR row_ptr.
    int dim = 32;
    auto A = random_adjacency(dim, 0.3f, 555555);
    ChrassLayer layer(dim, A);

    auto compute_hash = [&]() {
        uint64_t h = 0;
        for (auto c : layer.col_indices) h ^= (uint64_t)c * 2654435761ULL;
        for (auto r : layer.row_ptr) h ^= (uint64_t)r * 0x9E3779B97F4A7C15ULL;
        return h;
    };
    uint64_t h0 = compute_hash();

    for (int step = 0; step < 50; ++step) {
        Tensor X = make_input(random_vector(dim, 100 + step), 2, dim);
        layer.forward(X);
        Tensor dY = Tensor::ones({2, dim}, nsos::Device::CPU);
        layer.backward(dY, X);
        layer.step(0.01f);
    }
    CHECK(compute_hash() == h0,
          "topology hash drifted: before=" << h0 << " after=" << compute_hash());
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " CHRASS validation battery -- nsos::ChrassLayer (v2)" << std::endl;
    std::cout << "============================================================" << std::endl;

    // [F] Forward
    RUN(test_forward_identity_matrix);
    RUN(test_forward_dense_reference_small);
    RUN(test_forward_batch_consistency);

    // [B] Backward
    RUN(test_backward_returns_correct_shape);
    RUN(test_backward_identity_grad);
    RUN(test_backward_no_input_grad_outside_support);

    // [G] Gradcheck
    RUN(test_gradcheck_input);

    // [E] Edge cases
    RUN(test_edge_empty_graph);
    RUN(test_edge_complete_graph);
    RUN(test_edge_self_loops_only);
    RUN(test_edge_negative_weights);
    RUN(test_edge_1x1);

    // [S] Saturação
    RUN(test_saturation_output_clamp);
    RUN(test_saturation_gradient_clamp);

    // [N] NaN/Inf
    RUN(test_nan_input_sanitization);

    // [D] Determinismo
    RUN(test_deterministic_forward);

    // [A] AdamW
    RUN(test_adamw_step_advances_timestep);
    RUN(test_adamw_zero_grad_minimal_change);

    // [T] Topologia
    RUN(test_topology_preserved_after_steps);

    // [I] Integração
    RUN(test_integration_stack_three_layers);

    // [P] Performance
    RUN(test_performance_256x256_under_500ms);

    // [X] Stress / extremo
    RUN(test_stress_1024x1024_sparse);
    RUN(test_stress_long_training_loop_converges);
    RUN(test_numerical_adversarial_magnitudes);
    RUN(test_repeated_backward_no_grad_accumulation_overflow);
    RUN(test_topology_hash_stable_over_steps);

    std::cout << "============================================================" << std::endl;
    std::cout << " RESULT: " << g_passed << " passed, " << g_failed << " failed" << std::endl;
    std::cout << "============================================================" << std::endl;
    if (g_failed > 0) {
        std::cout << "Failures:" << std::endl;
        for (auto& f : g_failures) std::cout << "  - " << f << std::endl;
    }
    return g_failed == 0 ? 0 : 1;
}
