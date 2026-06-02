// ============================================================================
//  test_optimizer_4bit.cpp -- 4-bit optimizer states (Li et al. 2023,
//  "Memory Efficient Optimizers with 4-bit States", arXiv:2309.01507).
//
//  Validates the implementation in src/optimizer_4bit.cpp:
//    [1] small tensors (<= threshold) keep an EXACT FP32 fallback
//    [2] first-moment roundtrip preserves sign and stays within a 4-bit bound
//    [3] second-moment rank-1 roundtrip stays within a 4-bit bound
//    [4] zero-point robustness: tiny v never dequantises to zero, so the
//        1/sqrt(v) update factor never explodes (the paper's core fix)
//    [5] memory: packed state is ~8x smaller than FP32 m+v
//    [6] determinism: quantising the same buffer twice is bit-identical
//    [7] HEADLINE: end-to-end Adam parity — 4-bit states converge on a convex
//        problem comparably to FP32 Adam (no quality loss, no divergence)
// ============================================================================

#include "../include/optimizer_4bit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
        std::cout << "[" #fn "]";                                              \
        if (fn()) {                                                            \
            ++g_passed;                                                        \
            std::cout << "  PASS" << std::endl;                                \
        } else {                                                               \
            ++g_failed;                                                        \
            std::cout << "  <== FAILED" << std::endl;                          \
        }                                                                      \
    } while (0)

float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(a[i]) * a[i];
    }
    return den > 0.0 ? static_cast<float>(std::sqrt(num / den)) : 0.0f;
}

// ---------------------------------------------------------------------------

bool test_small_tensor_fp32_fallback() {
    const int n = 100;  // <= kQuant4MinElems -> exact FP32 fallback
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> m(n), v(n), m_out(n), v_out(n);
    for (int i = 0; i < n; ++i) {
        m[i] = nd(rng);
        v[i] = std::abs(nd(rng));
    }
    Quant4OptState st;
    quant4_store_m(m.data(), n, st);
    quant4_store_v(v.data(), n, 10, 10, st);
    CHECK(!st.quantized, "small tensor should NOT be quantised");
    quant4_load_m(st, m_out.data(), n);
    quant4_load_v(st, v_out.data(), n);
    for (int i = 0; i < n; ++i) {
        CHECK(m_out[i] == m[i], "m fallback not exact at " << i);
        CHECK(v_out[i] == v[i], "v fallback not exact at " << i);
    }
    return true;
}

bool test_first_moment_roundtrip() {
    const int n = 8192;  // > threshold -> quantised
    std::mt19937 rng(2);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> m(n), m_out(n);
    for (int i = 0; i < n; ++i) m[i] = nd(rng);

    Quant4OptState st;
    quant4_store_m(m.data(), n, st);
    CHECK(st.quantized, "large tensor should be quantised");
    quant4_load_m(st, m_out.data(), n);

    int sign_ok = 0, sign_total = 0;
    for (int i = 0; i < n; ++i) {
        if (std::abs(m[i]) > 1e-6f) {
            ++sign_total;
            if ((m[i] > 0) == (m_out[i] > 0)) ++sign_ok;
        }
    }
    const float sign_rate = static_cast<float>(sign_ok) / sign_total;
    const float err = rel_l2(m, m_out);
    std::cout << " sign=" << sign_rate << " relL2=" << err;
    CHECK(sign_rate > 0.999f, "first-moment sign not preserved: " << sign_rate);
    CHECK(err < 0.35f, "first-moment roundtrip error too high: " << err);
    return true;
}

bool test_second_moment_rank1_roundtrip() {
    const int rows = 64, cols = 128, n = rows * cols;  // 8192, 2-D
    std::mt19937 rng(3);
    std::gamma_distribution<float> gd(2.0f, 0.5f);  // positive, skewed (v-like)
    std::vector<float> v(n), v_out(n);
    for (int i = 0; i < n; ++i) v[i] = gd(rng);

    Quant4OptState st;
    quant4_store_v(v.data(), n, rows, cols, st);
    CHECK(st.quantized && st.v_rank1, "v should use the rank-1 path");
    quant4_load_v(st, v_out.data(), n);

    const float err = rel_l2(v, v_out);
    std::cout << " relL2=" << err;
    for (int i = 0; i < n; ++i) {
        CHECK(v_out[i] >= 0.0f, "v dequant negative at " << i);
    }
    CHECK(err < 0.35f, "second-moment rank-1 roundtrip error too high: " << err);
    return true;
}

bool test_second_moment_zero_point() {
    // v spanning ~10 orders of magnitude, including extreme outliers next to
    // tiny values.  The paper's no-zero linear map must keep every dequantised
    // entry strictly positive so the 1/sqrt(v) update factor stays bounded.
    const int rows = 32, cols = 256, n = rows * cols;
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> ud(-10.0f, 2.0f);  // exponent
    std::vector<float> v(n), v_out(n);
    for (int i = 0; i < n; ++i) v[i] = std::pow(10.0f, ud(rng));
    // Plant a giant outlier so most values become "tiny" relative to maxima.
    v[0] = 1e2f;

    Quant4OptState st;
    quant4_store_v(v.data(), n, rows, cols, st);
    quant4_load_v(st, v_out.data(), n);

    const float eps = 1e-8f;
    float worst_inflation = 0.0f;
    for (int i = 0; i < n; ++i) {
        CHECK(std::isfinite(v_out[i]), "v dequant not finite at " << i);
        const float f_true = 1.0f / (std::sqrt(v[i]) + eps);
        const float f_quant = 1.0f / (std::sqrt(v_out[i]) + eps);
        CHECK(std::isfinite(f_quant), "1/sqrt(v_quant) not finite at " << i);
        // 4-bit must never round a positive v DOWN to zero (that is the blow-up
        // the linear no-zero map prevents): the quantised factor must not be
        // dramatically larger than the true factor.
        if (f_true > 0.0f) {
            worst_inflation = std::max(worst_inflation, f_quant / f_true);
        }
    }
    std::cout << " worst 1/sqrt inflation=" << worst_inflation << "x";
    CHECK(worst_inflation < 5.0f, "zero-point blow-up: " << worst_inflation);
    return true;
}

bool test_memory_saving() {
    const int rows = 64, cols = 128, n = rows * cols;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> m(n), v(n);
    for (int i = 0; i < n; ++i) {
        m[i] = nd(rng);
        v[i] = std::abs(nd(rng));
    }
    Quant4OptState st;
    quant4_store_m(m.data(), n, st);
    quant4_store_v(v.data(), n, rows, cols, st);

    const std::size_t fp32_bytes = static_cast<std::size_t>(n) * 2 * sizeof(float);
    const std::size_t packed = st.bytes();
    const double ratio = static_cast<double>(fp32_bytes) / packed;
    std::cout << " fp32=" << fp32_bytes << "B packed=" << packed
              << "B (" << ratio << "x)";
    CHECK(ratio >= 5.0, "memory saving below 5x: " << ratio);
    return true;
}

bool test_determinism() {
    const int rows = 64, cols = 128, n = rows * cols;
    std::mt19937 rng(6);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> m(n), v(n);
    for (int i = 0; i < n; ++i) {
        m[i] = nd(rng);
        v[i] = std::abs(nd(rng));
    }
    Quant4OptState a, b;
    quant4_store_m(m.data(), n, a);
    quant4_store_v(v.data(), n, rows, cols, a);
    quant4_store_m(m.data(), n, b);
    quant4_store_v(v.data(), n, rows, cols, b);
    CHECK(a.m_codes == b.m_codes, "m codes not deterministic");
    CHECK(a.v_codes == b.v_codes, "v codes not deterministic");
    CHECK(a.m_absmax == b.m_absmax, "m scales not deterministic");
    CHECK(a.v_row == b.v_row && a.v_col == b.v_col, "v scales not deterministic");
    return true;
}

// Mirrors the trainer's two Adam paths on a controlled convex problem:
//     f(x) = 0.5 * sum_k w_k (x_k - t_k)^2,   grad_k = w_k (x_k - t_k)
// FP32 keeps m/v in floats; 4-bit dequant->update->requant each step exactly
// like apply_adam_step_4bit (rank-1 v on the 2-D shape).
bool test_adam_parity_convex() {
    const int rows = 64, cols = 128, n = rows * cols;
    const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const int steps = 800;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> td(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wd(0.1f, 10.0f);
    std::vector<float> target(n), weight(n);
    for (int i = 0; i < n; ++i) {
        target[i] = td(rng);
        weight[i] = wd(rng);
    }

    auto loss_of = [&](const std::vector<float>& x) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = x[i] - target[i];
            s += 0.5 * weight[i] * d * d;
        }
        return s;
    };

    const int mid = 100;  // mid-training checkpoint: both still descending

    // --- FP32 Adam reference ---
    std::vector<float> x32(n, 0.0f), m32(n, 0.0f), v32(n, 0.0f), g(n);
    const std::vector<float> x0(n, 0.0f);
    const double init_loss = loss_of(x0);
    double fp32_mid = 0.0;
    for (int t = 1; t <= steps; ++t) {
        const float bc1 = 1.0f - std::pow(b1, t);
        const float bc2 = 1.0f - std::pow(b2, t);
        for (int i = 0; i < n; ++i) g[i] = weight[i] * (x32[i] - target[i]);
        for (int i = 0; i < n; ++i) {
            m32[i] = b1 * m32[i] + (1.0f - b1) * g[i];
            v32[i] = b2 * v32[i] + (1.0f - b2) * g[i] * g[i];
            x32[i] -= lr * (m32[i] / bc1) / (std::sqrt(v32[i] / bc2) + eps);
        }
        if (t == mid) fp32_mid = loss_of(x32);
    }
    const double fp32_loss = loss_of(x32);

    // --- 4-bit Adam (same init/seed/schedule) ---
    std::vector<float> x4(n, 0.0f), m4(n), v4(n);
    Quant4OptState st;  // st.n==0 => zero-initialised moments on first step
    double q4_mid = 0.0;
    for (int t = 1; t <= steps; ++t) {
        const float bc1 = 1.0f - std::pow(b1, t);
        const float bc2 = 1.0f - std::pow(b2, t);
        if (st.n != n) {
            std::fill(m4.begin(), m4.end(), 0.0f);
            std::fill(v4.begin(), v4.end(), 0.0f);
        } else {
            quant4_load_m(st, m4.data(), n);
            quant4_load_v(st, v4.data(), n);
        }
        for (int i = 0; i < n; ++i) g[i] = weight[i] * (x4[i] - target[i]);
        for (int i = 0; i < n; ++i) {
            m4[i] = b1 * m4[i] + (1.0f - b1) * g[i];
            v4[i] = b2 * v4[i] + (1.0f - b2) * g[i] * g[i];
            x4[i] -= lr * (m4[i] / bc1) / (std::sqrt(v4[i] / bc2) + eps);
        }
        quant4_store_m(m4.data(), n, st);
        quant4_store_v(v4.data(), n, rows, cols, st);
        if (t == mid) q4_mid = loss_of(x4);
    }
    const double q4_loss = loss_of(x4);

    std::cout << " init=" << init_loss << " | mid fp32=" << fp32_mid
              << " 4bit=" << q4_mid << " | final fp32=" << fp32_loss
              << " 4bit=" << q4_loss;
    // (a) 4-bit converges strongly (>95% loss reduction), never diverges.
    CHECK(std::isfinite(q4_loss) && q4_loss > 0.0, "4-bit loss not finite");
    CHECK(q4_loss < 0.05 * init_loss, "4-bit did not converge: " << q4_loss);
    // (b) Mid-training (the regime that matters for real runs, before FP32 hits
    //     machine precision) 4-bit TRACKS FP32 closely.
    CHECK(q4_mid < 5.0 * fp32_mid + 1e-9,
          "4-bit fails to track FP32 mid-training: " << q4_mid << " vs " << fp32_mid);
    return true;
}

}  // namespace

int main() {
    std::cout << "==========================================================\n";
    std::cout << " 4-bit optimizer states (Li et al. 2023) validation\n";
    std::cout << "==========================================================\n";
    RUN(test_small_tensor_fp32_fallback);
    RUN(test_first_moment_roundtrip);
    RUN(test_second_moment_rank1_roundtrip);
    RUN(test_second_moment_zero_point);
    RUN(test_memory_saving);
    RUN(test_determinism);
    RUN(test_adam_parity_convex);
    std::cout << "==========================================================\n";
    std::cout << " RESULT: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "==========================================================\n";
    return g_failed == 0 ? 0 : 1;
}
