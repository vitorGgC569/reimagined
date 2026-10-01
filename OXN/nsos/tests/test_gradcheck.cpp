// ============================================================================
// test_gradcheck.cpp — central-difference gradient verification gate
// ----------------------------------------------------------------------------
// WHY THIS EXISTS
//   NSOS has no generic autograd tape: every module derives its backward BY
//   HAND.  Hand-derived gradients are the single largest correctness risk in
//   the codebase (a missing activation derivative in the MoE path was caught
//   only after it had silently corrupted the chain rule — see jamba.h LEARN S1
//   comment).  Per-module unit tests historically only asserted that gradients
//   were NON-ZERO (e.g. test_mamba2.cpp), which does not prove they are RIGHT.
//
//   This test compares each module's analytic backward against a numerical
//   central-difference estimate  dL/dθ ≈ (L(θ+ε) − L(θ−ε)) / 2ε  for a smooth
//   scalar loss L = ½·Σ y².  A relative error below the documented tolerance is
//   the gold-standard proof that the manual chain rule is correct.
//
//   SCOPE: the SMOOTH (differentiable) backward math.  The BitNet ternary STE
//   path is intentionally biased (straight-through) and is NOT finite-difference
//   checkable — its smooth components are covered by test_bitlinear.cpp's
//   reference-path gradcheck.  Here we cover the shared tensor bricks and the
//   modules whose backward is most intricate: Mamba2 SSD and the KAN layer.
//
//   This is a product CTest gate (CMakeLists.txt): a regression in any manual
//   backward fails the build.
// ============================================================================
#include "../include/jamba.h"
#include "../include/kan.h"
#include "../include/mamba2.h"
#include "../include/nsos/determinism.h"
#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace nsos;

namespace {

// Relative-error tolerances.  Central difference on a well-conditioned smooth
// function is O(ε²); with ε=1e-3 the analytic gradient should agree to a few
// parts in 1e3.  We allow 5e-2 for ops with a single nonlinearity and 1e-1 for
// the SSD scan (compounding exp/tanh over time amplifies finite-difference
// truncation error on the slowest channels).
constexpr float kTolStd = 5e-2f;
constexpr float kTolScan = 1e-1f;
constexpr float kEps = 1e-3f;
// Absolute-error floor (numpy.allclose / torch.gradcheck style: pass when
// |num - a| <= atol + rtol*|a|).  Near-zero gradient elements have a central-
// difference estimate dominated by FP rounding — and that rounding differs
// across compilers' FMA contraction (MSVC vs GCC -O3 -march=native), so a pure
// RELATIVE gate spuriously trips on them.  The absolute floor tolerates that
// noise; a genuinely wrong gradient has |num - a| >> atol and still fails.
constexpr float kAtol = 5e-3f;

int g_failures = 0;

// Central-difference check of one analytic gradient buffer against a scalar
// loss closure.  `data`/`n` is the parameter buffer being perturbed; `analytic`
// is the hand-derived gradient for the same buffer.  Pass/fail uses the
// numpy.allclose criterion (atol + rtol); the reported max-rel is computed only
// over meaningful (|a| > atol) elements so near-zero noise doesn't distort it.
float gradcheck_buffer(float *data, int n, const float *analytic,
                       const std::function<double()> &loss_fn, const char *name,
                       float tol) {
  float max_rel = 0.0f;
  bool any_fail = false;
  for (int i = 0; i < n; ++i) {
    const float orig = data[i];
    data[i] = orig + kEps;
    const double lp = loss_fn();
    data[i] = orig - kEps;
    const double lm = loss_fn();
    data[i] = orig;
    const float numeric = static_cast<float>((lp - lm) / (2.0 * kEps));
    const float a = analytic[i];
    const float abs_err = std::fabs(numeric - a);
    // allclose pass/fail: catches real errors (abs_err >> atol) but tolerates
    // FD rounding on near-zero gradients.
    if (abs_err > kAtol + tol * std::fabs(a)) {
      any_fail = true;
    }
    // Display metric: relative error over meaningful elements only.
    if (std::fabs(a) > kAtol) {
      max_rel = std::max(
          max_rel, abs_err / std::max(std::fabs(a), std::fabs(numeric)));
    }
  }
  const bool ok = !any_fail;
  std::printf("[gradcheck] %-28s max rel err = %.3e  (rtol %.0e atol %.0e)  %s\n",
              name, max_rel, tol, kAtol, ok ? "OK" : "FAIL");
  if (!ok)
    ++g_failures;
  return max_rel;
}

double sum_sq(const Tensor &y) {
  double s = 0.0;
  const float *p = y.data();
  for (int i = 0; i < y.size; ++i)
    s += 0.5 * static_cast<double>(p[i]) * static_cast<double>(p[i]);
  return s;
}

void fill_smooth(Tensor &t, float scale, float phase) {
  float *p = t.data();
  for (int i = 0; i < t.size; ++i)
    p[i] = scale * std::sin(phase * static_cast<float>(i + 1));
}

// ── Shared tensor brick: RMSNorm ─────────────────────────────────────────────
// NOTE: the loss MUST NOT be scale-invariant.  RMSNorm removes the scale of x,
// so L = ½·Σy² ≈ N is (near) constant in x and its gradient is ~0 — gradchecking
// that compares noise to noise.  We use a FIXED linear readout L = Σ c·y (with c
// constant), whose dL/dy = c exercises the full RMSNorm Jacobian with non-trivial
// gradients.
void check_rmsnorm() {
  const int rows = 3, dim = 6;
  Tensor x({rows, dim});
  fill_smooth(x, 0.9f, 0.31f);
  Tensor c({rows, dim});
  fill_smooth(c, 1.0f, 0.53f);  // fixed dL/dy
  const float *cp = c.data();
  auto loss_fn = [&]() {
    Tensor y = x.rmsnorm(1e-6f);
    const float *yp = y.data();
    double s = 0.0;
    for (int i = 0; i < y.size; ++i) s += static_cast<double>(cp[i]) * yp[i];
    return s;
  };
  Tensor xn = x.rmsnorm(1e-6f);
  Tensor dx = x.rmsnorm_backward(c, xn);  // dL/dy = c
  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn, "rmsnorm d/dx", kTolStd);
}

// ── Shared tensor brick: squared ReLU (BitNet 2B4T FFN activation) ───────────
void check_squared_relu() {
  const int n = 24;
  Tensor pre({n});
  fill_smooth(pre, 1.3f, 0.7f); // straddles 0 so the kink is exercised
  auto loss_fn = [&]() { return sum_sq(pre.squared_relu()); };
  Tensor y = pre.squared_relu();
  Tensor dy = y.clone();
  Tensor dx = Tensor::squared_relu_backward(dy, pre);
  gradcheck_buffer(pre.data(), pre.size, dx.data(), loss_fn, "squared_relu d/dx",
                   kTolStd);
}

// ── Shared tensor brick: cross-entropy (returns dL/dlogits directly) ─────────
void check_cross_entropy() {
  const int rows = 4, vocab = 7;
  Tensor logits({rows, vocab});
  fill_smooth(logits, 1.1f, 0.23f);
  std::vector<int> target = {0, 3, 6, 2};
  auto loss_fn = [&]() { return logits.cross_entropy(target).first; };
  auto [loss, grad] = logits.cross_entropy(target);
  (void)loss;
  gradcheck_buffer(logits.data(), logits.size, grad.data(), loss_fn,
                   "cross_entropy d/dlogits", kTolStd);
}

// ── Shared tensor brick: matmul backward identities ──────────────────────────
// Every manual backward reuses dA = dC·Bᵀ and dB = Aᵀ·dC.  Validate both.
void check_matmul() {
  const int m = 3, k = 4, nn = 2;
  Tensor A({m, k});
  Tensor B({k, nn});
  fill_smooth(A, 0.8f, 0.19f);
  fill_smooth(B, 0.7f, 0.29f);
  auto loss_fn = [&]() { return sum_sq(A.matmul(B)); };
  Tensor C = A.matmul(B);
  Tensor dC = C.clone();
  Tensor dA = dC.matmul(B.transpose());
  Tensor dB = A.transpose().matmul(dC);
  gradcheck_buffer(A.data(), A.size, dA.data(), loss_fn, "matmul d/dA", kTolStd);
  gradcheck_buffer(B.data(), B.size, dB.data(), loss_fn, "matmul d/dB", kTolStd);
}

// Find a parameter by exact name within a module's parameter list.
Parameter *find_param(const std::vector<Parameter *> &params,
                      const std::string &name) {
  for (auto *p : params)
    if (p && p->name == name)
      return p;
  return nullptr;
}

void zero_all_grads(const std::vector<Parameter *> &params) {
  for (auto *p : params)
    if (p)
      p->zero_grad();
}

// ── Module: Mamba2 SSD (end-to-end forward/backward) ─────────────────────────
// The recurrent core the project most distrusts.  Check both the gradient w.r.t.
// the layer input and w.r.t. the decay parameter A (the time-axis spectrum).
void check_mamba2() {
  const int D = 4, N = 4, H = 1, L = 5;
  Mamba2SSD layer(D, N, H);
  Tensor x({L, D});
  fill_smooth(x, 0.6f, 0.37f);

  auto loss_fn = [&]() { return sum_sq(layer.forward(x)); };

  // Analytic pass (capture grads BEFORE running finite-difference forwards,
  // which overwrite the layer's saved forward state).
  zero_all_grads(layer.parameters());
  Tensor y = layer.forward(x);
  Tensor dy = y.clone();
  Context ctx;
  Tensor dx = layer.backward(dy, ctx);

  Parameter *A = find_param(layer.parameters(), "A");
  assert(A != nullptr && "Mamba2SSD must expose parameter A");
  std::vector<float> gA(static_cast<size_t>(A->grad.size));
  for (int i = 0; i < A->grad.size; ++i)
    gA[static_cast<size_t>(i)] = A->grad.data()[i];

  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn, "mamba2 d/dinput",
                   kTolScan);
  gradcheck_buffer(A->data.data(), A->data.size, gA.data(), loss_fn,
                   "mamba2 d/dA", kTolScan);
}

// ── Module: Mamba2 SSD — PROPER selective path (opt-in) ──────────────────────
// Validates the corrected recurrence: independent delta/B/C/z projections, a
// causal depthwise conv1d, C applied once and a separate SiLU gate.  Checks the
// gradient w.r.t. the input, the decay parameter A, and the conv1d kernel.
void check_mamba2_proper() {
  const int D = 4, N = 4, H = 1, L = 5;
  MambaConfig cfg;
  cfg.proper_selective_ssm = true;
  cfg.conv_kernel = 3;
  Mamba2SSD layer(D, N, H, cfg);
  auto params = layer.parameters();
  Parameter *dt_bias = find_param(params, "dt_proj.bias");
  assert(dt_bias != nullptr && "proper Mamba2SSD must expose dt_proj.bias");
  std::fill_n(dt_bias->data.data(), dt_bias->data.size, -1.5f);
  Tensor x({L, D});
  fill_smooth(x, 0.6f, 0.41f);

  auto loss_fn = [&]() { return sum_sq(layer.forward(x)); };

  zero_all_grads(params);
  Tensor y = layer.forward(x);
  Tensor dy = y.clone();
  Context ctx;
  Tensor dx = layer.backward(dy, ctx);

  Parameter *A = find_param(params, "A");
  Parameter *conv = find_param(params, "conv1d_weight");
  assert(A != nullptr && conv != nullptr &&
         "proper Mamba2SSD must expose A and conv1d_weight");
  std::vector<float> gA(static_cast<size_t>(A->grad.size));
  std::vector<float> gW(static_cast<size_t>(conv->grad.size));
  std::vector<float> gDt(static_cast<size_t>(dt_bias->grad.size));
  for (int i = 0; i < A->grad.size; ++i)
    gA[static_cast<size_t>(i)] = A->grad.data()[i];
  for (int i = 0; i < conv->grad.size; ++i)
    gW[static_cast<size_t>(i)] = conv->grad.data()[i];
  for (int i = 0; i < dt_bias->grad.size; ++i)
    gDt[static_cast<size_t>(i)] = dt_bias->grad.data()[i];

  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn,
                   "mamba2-proper d/dinput", kTolScan);
  gradcheck_buffer(A->data.data(), A->data.size, gA.data(), loss_fn,
                   "mamba2-proper d/dA", kTolScan);
  gradcheck_buffer(conv->data.data(), conv->data.size, gW.data(), loss_fn,
                   "mamba2-proper d/dconv1d", kTolScan);
  gradcheck_buffer(dt_bias->data.data(), dt_bias->data.size, gDt.data(),
                   loss_fn, "mamba2-proper d/ddt_bias", kTolScan);
}

// ── Module: full Mamba-2 SSD with N-state expansion (proper_state_expansion) ──
void check_mamba2_nstate() {
  const int H = 2, P = 4, N = 3, L = 5;
  const int D = H * P;  // d_model must equal n_heads * d_head
  MambaConfig cfg;
  cfg.proper_selective_ssm = true;
  cfg.proper_state_expansion = true;
  cfg.conv_kernel = 3;
  Mamba2SSD layer(D, N, H, cfg);
  auto params = layer.parameters();
  Parameter *dt_bias = find_param(params, "dt_proj.bias");
  assert(dt_bias != nullptr && "nstate Mamba2SSD must expose dt_proj.bias");
  std::fill_n(dt_bias->data.data(), dt_bias->data.size, -1.5f);
  Tensor x({L, D});
  fill_smooth(x, 0.6f, 0.39f);

  auto loss_fn = [&]() { return sum_sq(layer.forward(x)); };

  zero_all_grads(params);
  Tensor y = layer.forward(x);
  Tensor dy = y.clone();
  Context ctx;
  Tensor dx = layer.backward(dy, ctx);

  Parameter *A = find_param(params, "A");
  Parameter *conv = find_param(params, "conv1d_weight");
  assert(A != nullptr && conv != nullptr &&
         "nstate Mamba2SSD must expose A and conv1d_weight");
  std::vector<float> gA(static_cast<size_t>(A->grad.size));
  std::vector<float> gW(static_cast<size_t>(conv->grad.size));
  std::vector<float> gDt(static_cast<size_t>(dt_bias->grad.size));
  for (int i = 0; i < A->grad.size; ++i)
    gA[static_cast<size_t>(i)] = A->grad.data()[i];
  for (int i = 0; i < conv->grad.size; ++i)
    gW[static_cast<size_t>(i)] = conv->grad.data()[i];
  for (int i = 0; i < dt_bias->grad.size; ++i)
    gDt[static_cast<size_t>(i)] = dt_bias->grad.data()[i];

  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn,
                   "mamba2-nstate d/dinput", kTolScan);
  gradcheck_buffer(A->data.data(), A->data.size, gA.data(), loss_fn,
                   "mamba2-nstate d/dA", kTolScan);
  gradcheck_buffer(conv->data.data(), conv->data.size, gW.data(), loss_fn,
                   "mamba2-nstate d/dconv1d", kTolScan);
  gradcheck_buffer(dt_bias->data.data(), dt_bias->data.size, gDt.data(),
                   loss_fn, "mamba2-nstate d/ddt_bias", kTolScan);
}

// ── Module: proper-path incremental streaming decode ─────────────────────────
// The single-token decode (prefill + forward_proper_step) must reproduce the
// full-sequence scan token-for-token.  Both run the same host conv1d + recurrence
// so the carried state + conv window are byte-faithful — this catches any drift
// in the streaming bookkeeping (window shift, prime, state carry) on CPU, before
// it ever burns T4 quota.
void check_mamba2_faithful() {
  const int D = 4, N = 3, L = 4;
  MambaConfig cfg;
  cfg.faithful_mamba2 = true;
  cfg.expand = 2;
  cfg.head_dim = 4;
  cfg.n_groups = 1;
  cfg.conv_kernel = 3;
  Mamba2SSD layer(D, N, /*ignored in faithful mode=*/1, cfg);
  auto params = layer.parameters();
  Parameter *A = find_param(params, "A");
  Parameter *conv = find_param(params, "conv1d_weight");
  Parameter *norm = find_param(params, "norm.weight");
  Parameter *dt_bias = find_param(params, "dt_proj.bias");
  assert(A && conv && norm && dt_bias);
  fill_smooth(A->data, 0.2f, 0.17f);
  fill_smooth(conv->data, 0.12f, 0.23f);
  std::fill_n(norm->data.data(), norm->data.size, 1.0f);
  std::fill_n(dt_bias->data.data(), dt_bias->data.size, -2.0f);
  Tensor x({L, D});
  fill_smooth(x, 0.4f, 0.31f);
  auto loss_fn = [&]() { return sum_sq(layer.forward(x)); };

  zero_all_grads(params);
  Tensor y = layer.forward(x);
  Context ctx;
  Tensor dx = layer.backward(y.clone(), ctx);
  auto copy_grad = [](Parameter *p) {
    return std::vector<float>(p->grad.data(), p->grad.data() + p->grad.size);
  };
  const auto gA = copy_grad(A);
  const auto gConv = copy_grad(conv);
  const auto gNorm = copy_grad(norm);
  const auto gDt = copy_grad(dt_bias);
  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn,
                   "mamba2-faithful d/input", kTolScan);
  gradcheck_buffer(A->data.data(), A->data.size, gA.data(), loss_fn,
                   "mamba2-faithful d/A_log", kTolScan);
  gradcheck_buffer(conv->data.data(), conv->data.size, gConv.data(), loss_fn,
                   "mamba2-faithful d/conv", kTolScan);
  gradcheck_buffer(norm->data.data(), norm->data.size, gNorm.data(), loss_fn,
                   "mamba2-faithful d/norm", kTolScan);
  gradcheck_buffer(dt_bias->data.data(), dt_bias->data.size, gDt.data(),
                   loss_fn, "mamba2-faithful d/dt_bias", kTolScan);
}

void check_mamba2_extreme_decay_finiteness() {
  const int D = 4, N = 3, L = 4;
  MambaConfig cfg;
  cfg.faithful_mamba2 = true;
  cfg.expand = 2;
  cfg.head_dim = 4;
  cfg.n_groups = 1;
  cfg.conv_kernel = 3;
  Mamba2SSD layer(D, N, 1, cfg);
  auto params = layer.parameters();
  Parameter *A = find_param(params, "A");
  Parameter *dt_bias = find_param(params, "dt_proj.bias");
  assert(A && dt_bias);
  std::fill_n(A->data.data(), A->data.size, 1000.0f);
  std::fill_n(dt_bias->data.data(), dt_bias->data.size, -2.0f);
  Tensor x({L, D});
  fill_smooth(x, 0.4f, 0.31f);
  zero_all_grads(params);
  Tensor y = layer.forward(x);
  Context ctx;
  Tensor dx = layer.backward(Tensor::ones(y.shape.dims), ctx);
  bool finite = true;
  for (int i = 0; i < y.size; ++i) finite = finite && std::isfinite(y.data()[i]);
  for (int i = 0; i < dx.size; ++i) finite = finite && std::isfinite(dx.data()[i]);
  for (Parameter *p : params) {
    if (!p || p->grad.size == 0) continue;
    for (int i = 0; i < p->grad.size; ++i) {
      finite = finite && std::isfinite(p->grad.data()[i]);
    }
  }
  std::printf("[gradcheck] %-32s %s\n", "mamba2 extreme decay finite",
              finite ? "OK" : "FAILED");
  if (!finite) ++g_failures;
}

// ── Faithful model wrapper: learned residual/final RMSNorm scales ───────────
// Mamba-2's mixer parity is insufficient if the residual wrapper silently uses
// an unparameterized RMSNorm. Check the two gamma gradients through the complete
// one-layer stack and LM head.
void check_mamba2_faithful_wrapper_norms() {
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.d_model = 8;
  cfg.vocab_size = 7;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 1;
  cfg.mamba_head_dim = 4;
  cfg.attention_period = 99;
  cfg.use_moe = false;
  cfg.use_kan = false;
  cfg.use_chrass = false;
  cfg.mamba2_faithful = true;
  cfg.tie_word_embeddings = false;
  cfg.dropout = 0.0f;
  JambaModel model(cfg, Device::CPU);
  auto params = model.parameters();
  Parameter *layer_norm = find_param(params, "layers.0.norm.weight");
  Parameter *final_norm = find_param(params, "norm_f.weight");
  assert(layer_norm && final_norm);

  Tensor x({3, cfg.d_model});
  fill_smooth(x, 0.35f, 0.29f);
  auto loss_fn = [&]() { return sum_sq(model.forward(x)); };

  zero_all_grads(params);
  Context ctx;
  Tensor y = model.forward(x, &ctx);
  model.backward(y.clone(), ctx);
  const std::vector<float> g_layer(
      layer_norm->grad.data(), layer_norm->grad.data() + layer_norm->grad.size);
  const std::vector<float> g_final(
      final_norm->grad.data(), final_norm->grad.data() + final_norm->grad.size);

  gradcheck_buffer(layer_norm->data.data(), layer_norm->data.size,
                   g_layer.data(), loss_fn, "faithful wrapper d/layernorm",
                   kTolScan);
  gradcheck_buffer(final_norm->data.data(), final_norm->data.size,
                   g_final.data(), loss_fn, "faithful wrapper d/finalnorm",
                   kTolScan);
}

static void check_proper_streaming_parity(bool nstate) {
  const int H = nstate ? 2 : 1;
  const int P = 4;
  const int N = nstate ? 3 : 4;
  const int D = nstate ? H * P : 4;
  const int L = 6;
  MambaConfig cfg;
  cfg.proper_selective_ssm = true;
  cfg.proper_state_expansion = nstate;
  cfg.conv_kernel = 3;
  Mamba2SSD layer(D, N, H, cfg);
  Tensor x({L, D});
  fill_smooth(x, 0.6f, nstate ? 0.39f : 0.41f);

  Tensor full = layer.forward(x);  // full scan, streaming off -> [L, D]

  const char *name =
      nstate ? "mamba2-nstate streaming==scan" : "mamba2-proper streaming==scan";
  const float tol = 1e-4f;
  // Two scenarios: prefix=1 (prefill a single token, then step) and prefix=3 (the
  // production path: the whole prompt primes in ONE multi-token forward, then
  // single-token steps).  Both must reproduce the full scan token-for-token.
  for (int prefix : {1, 3}) {
    layer.reset();
    layer.set_streaming_mode(true);
    float max_abs = 0.0f;
    // Multi-token prefill: forward the first `prefix` rows in one call.
    Tensor xp({prefix, D});
    for (int i = 0; i < prefix * D; ++i) xp.data()[i] = x.data()[i];
    Tensor yp = layer.forward(xp);  // [prefix, D] -> primes the stream
    for (int i = 0; i < prefix * D; ++i) {
      const float diff = std::fabs(yp.data()[i] - full.data()[i]);
      if (diff > max_abs) max_abs = diff;
    }
    // Incremental steps for the remaining tokens.
    for (int t = prefix; t < L; ++t) {
      Tensor xt({1, D});
      for (int c = 0; c < D; ++c)
        xt.data()[c] = x.data()[static_cast<size_t>(t) * D + c];
      Tensor yt = layer.forward(xt);
      for (int c = 0; c < D; ++c) {
        const float diff = std::fabs(
            yt.data()[c] - full.data()[static_cast<size_t>(t) * D + c]);
        if (diff > max_abs) max_abs = diff;
      }
    }
    layer.set_streaming_mode(false);
    if (max_abs <= tol) {
      std::printf("[gradcheck] %-28s OK   (prefix=%d max|delta|=%.2e)\n", name,
                  prefix, max_abs);
    } else {
      std::printf("[gradcheck] %-28s FAIL (prefix=%d max|delta|=%.2e > %.1e)\n",
                  name, prefix, max_abs, tol);
      ++g_failures;
    }
  }
}

// ── Module: BitFastKAN layer (end-to-end forward/backward) ───────────────────
void check_kan() {
  const int in = 5, out = 3, rows = 4, grid = 5;
  BitFastKANLayer layer(in, out, grid);
  fill_smooth(layer.base_weight.data, 0.3f, 0.21f);
  fill_smooth(layer.rbf_weight.data, 0.2f, 0.13f);
  fill_smooth(layer.bias.data, 0.1f, 0.41f);

  Tensor x({rows, in});
  fill_smooth(x, 0.5f, 0.33f);
  auto loss_fn = [&]() { return sum_sq(layer.forward(x)); };

  zero_all_grads(layer.parameters());
  Tensor y = layer.forward(x);
  Tensor dy = y.clone();
  Tensor dx = layer.backward(dy);

  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn, "kan d/dinput", kTolStd);
  gradcheck_buffer(layer.base_weight.data.data(), layer.base_weight.data.size,
                   layer.base_weight.grad.data(), loss_fn, "kan d/dbase_weight",
                   kTolStd);
  gradcheck_buffer(layer.rbf_weight.data.data(), layer.rbf_weight.data.size,
                   layer.rbf_weight.grad.data(), loss_fn, "kan d/drbf_weight",
                   kTolStd);
  gradcheck_buffer(layer.bias.data.data(), layer.bias.data.size,
                   layer.bias.grad.data(), loss_fn, "kan d/dbias", kTolStd);
}

// ── MoE: differentiable Switch aux-loss gradient w.r.t. router logits ────────
// Validates MoERouter::switch_aux_grad_logits against the finite-difference of
// the aux loss seen as a function of the logits.  Logits are constructed with a
// dominant per-expert ramp so the hard top-k set is stable under ±eps (the hard
// dispatch fraction f_e is a stop-gradient constant, as in Switch).
void check_moe_switch_aux() {
  const int T = 4, N = 5, k = 2;
  const float coef = 0.7f;
  Tensor logits({T, N});
  float *lp = logits.data();
  for (int i = 0; i < T; ++i)
    for (int e = 0; e < N; ++e)
      lp[i * N + e] = 0.7f * static_cast<float>(e) +
                      0.15f * std::sin(2.1f * static_cast<float>(i + 1) +
                                       static_cast<float>(e));
  auto loss_fn = [&]() {
    Tensor probs = logits.softmax(-1);
    float L = 0.0f;
    (void)MoERouter::switch_aux_grad_logits(probs, k, coef, &L);
    return static_cast<double>(L);
  };
  Tensor probs = logits.softmax(-1);
  Tensor gz = MoERouter::switch_aux_grad_logits(probs, k, coef, nullptr);
  gradcheck_buffer(logits.data(), logits.size, gz.data(), loss_fn,
                   "moe switch-aux d/dlogits", kTolStd);
}

// ── MoE: TASK gradient to the router w.r.t. logits ───────────────────────────
// Validates MoERouter::router_grad_logits against the finite-difference of
// L(logits) = sum_r sum_{e in topk} w[r,e]*c[r,e], with w = renorm(top-k(softmax))
// and c a fixed coefficient matrix standing in for g_w = sum_dim(dy*expert_out).
// Stable top-k via a dominant per-expert ramp.
void check_moe_router_grad() {
  const int T = 4, N = 5, k = 2;
  Tensor logits({T, N});
  float *lp = logits.data();
  for (int i = 0; i < T; ++i)
    for (int e = 0; e < N; ++e)
      lp[i * N + e] = 0.7f * static_cast<float>(e) +
                      0.15f * std::sin(1.7f * static_cast<float>(i + 1) +
                                       static_cast<float>(e));
  Tensor c({T, N});
  fill_smooth(c, 0.6f, 0.27f);  // stands in for g_w = dL/dw

  auto loss_fn = [&]() {
    Tensor probs = logits.softmax(-1);
    const float *p = probs.data();
    const float *cc = c.data();
    double L = 0.0;
    std::vector<int> ranked(static_cast<size_t>(N));
    for (int i = 0; i < T; ++i) {
      const int base = i * N;
      std::iota(ranked.begin(), ranked.end(), 0);
      std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                        [&](int a, int b) { return p[base + a] > p[base + b]; });
      double S = 0.0;
      for (int r = 0; r < k; ++r) S += p[base + ranked[static_cast<size_t>(r)]];
      if (S <= 0.0) continue;
      for (int r = 0; r < k; ++r) {
        const int e = ranked[static_cast<size_t>(r)];
        L += (p[base + e] / S) * cc[base + e];
      }
    }
    return L;
  };

  Tensor probs = logits.softmax(-1);
  Tensor gz = MoERouter::router_grad_logits(probs, c, k);
  gradcheck_buffer(logits.data(), logits.size, gz.data(), loss_fn,
                   "moe router-grad d/dlogits", kTolStd);
}

// ── Module: ChrassLayer (sparse topological injection) ───────────────────────
// Validates the CSR sparse-linear backward: dL/dx (via Wᵀ), dL/dW (the nnz
// values) and dL/dbias.  Inputs/weights are kept small so the forward stays in
// the linear band (|out| < 100) and the saturation mask is all-ones — i.e. we
// gradcheck the differentiable regime the mask preserves.
void check_chrass() {
  const int dim = 8, batch = 3;
  std::vector<float> adj = ChrassLayer::random_adjacency(dim, 0.5f, 1234u);
  ChrassLayer layer(dim, adj);
  fill_smooth(layer.values_param.data, 0.3f, 0.23f);
  fill_smooth(layer.bias.data, 0.1f, 0.37f);

  Tensor x({batch, dim});
  fill_smooth(x, 0.5f, 0.31f);

  auto loss_fn = [&]() { return sum_sq(layer.forward(x)); };

  zero_all_grads(layer.parameters());
  Tensor y = layer.forward(x);
  Tensor dy = y.clone();
  Tensor dx = layer.backward(dy, x);

  // Snapshot param grads before the finite-difference forwards (backward resets
  // values_param.grad each call, so the snapshot must precede any re-run).
  std::vector<float> gW(static_cast<size_t>(layer.values_param.grad.size));
  for (int i = 0; i < layer.values_param.grad.size; ++i)
    gW[static_cast<size_t>(i)] = layer.values_param.grad.data()[i];
  std::vector<float> gB(static_cast<size_t>(layer.bias.grad.size));
  for (int i = 0; i < layer.bias.grad.size; ++i)
    gB[static_cast<size_t>(i)] = layer.bias.grad.data()[i];

  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn, "chrass d/dinput",
                   kTolStd);
  gradcheck_buffer(layer.values_param.data.data(), layer.values_param.data.size,
                   gW.data(), loss_fn, "chrass d/dvalues", kTolStd);
  gradcheck_buffer(layer.bias.data.data(), layer.bias.data.size, gB.data(),
                   loss_fn, "chrass d/dbias", kTolStd);
}

// ── Module: exact GQA causal Attention (proj + RoPE + causal softmax) ────────
// The most intricate hand-derived backward in the codebase (RoPE rotation,
// causal softmax Jacobian, GQA head folding, q/kv/out projections).  On a CPU
// build the exact-training path is pure host math.  We gradcheck the INPUT
// gradient end-to-end — the chain the attention math is uniquely responsible
// for (the projection weight grads go through BitLinear's reference path, which
// test_bitlinear already gradchecks).  Tolerance is the scan band: the softmax
// over the causal window compounds finite-difference truncation.
void check_attention() {
  const int d_model = 8, n_heads = 2, n_kv_heads = 2, seq = 4;
  Attention attn(d_model, n_heads, /*n_latents=*/d_model, n_kv_heads);
  attn.set_training_mode(true);
  attn.set_exact_training_path(true);

  Tensor x({seq, d_model});
  fill_smooth(x, 0.5f, 0.29f);

  auto loss_fn = [&]() {
    Context c;
    return sum_sq(attn.forward(x, &c));
  };

  Context ctx;
  Tensor y = attn.forward(x, &ctx);
  Tensor dy = y.clone();
  Tensor dx = attn.backward(dy, &ctx);

  gradcheck_buffer(x.data(), x.size, dx.data(), loss_fn, "attention d/dinput",
                   kTolScan);
}

// ── Product hybrid: independent Mamba / Attention / FFN gates ───────────────
// These LayerScale-style vectors are the control surface that lets the hybrid
// preserve Mamba while adding the other branches. A sign/chain-rule error here
// can make a branch appear harmful even when its own backward is correct.
void check_parallel_hybrid_gates() {
  ModelConfig config;
  config.num_layers = 1;
  config.d_model = 8;
  config.vocab_size = 11;
  config.n_heads = 2;
  config.n_kv_heads = 1;
  config.attention_period = 1;
  config.attention_slot = 0;
  config.force_mamba_last_layer = false;
  config.hybrid_composition =
      HybridComposition::ParallelGated;
  config.faithful_attention_linears = true;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_chrass = false;
  config.use_kan = false;
  config.dropout = 0.0f;
  config.mamba2_faithful = true;
  config.mamba_expand = 1;
  config.mamba_head_dim = 4;
  config.mamba_n_groups = 1;
  config.mamba_d_state = 4;
  config.mamba_conv_kernel = 3;
  config.tie_word_embeddings = false;

  JambaModel model(config, Device::CPU);
  Tensor input({3, config.d_model}, Device::CPU);
  fill_smooth(input, 0.35f, 0.23f);
  Tensor readout({3, config.vocab_size}, Device::CPU);
  fill_smooth(readout, 0.27f, 0.31f);

  const auto objective = [&](Parameter* perturbed) {
    if (perturbed != nullptr) {
      // Faithful Mamba/BitLinear grouped caches are versioned. Every finite
      // difference perturbation must invalidate them explicitly.
      perturbed->mark_updated();
    }
    Context context;
    const Tensor output = model.forward(input, &context);
    double value = 0.0;
    for (int index = 0; index < output.size; ++index) {
      value += static_cast<double>(output.data()[index]) *
               static_cast<double>(readout.data()[index]);
    }
    return value;
  };

  auto parameters = model.parameters();
  zero_all_grads(parameters);
  Context context;
  (void)model.forward(input, &context);
  model.backward(readout, context);

  for (const char* name :
       {"layers.0.mamba.gate",
        "layers.0.attn.gate",
        "layers.0.ffn.gate"}) {
    Parameter* parameter = find_param(parameters, name);
    assert(parameter != nullptr);
    assert(parameter->grad.size == parameter->data.size);
    std::vector<float> analytic(
        parameter->grad.data(),
        parameter->grad.data() + parameter->grad.size);
    const std::string label =
        std::string("hybrid ") + name;
    gradcheck_buffer(
        parameter->data.data(), parameter->data.size,
        analytic.data(),
        [&]() { return objective(parameter); },
        label.c_str(), kTolScan);
  }
}

} // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  // Deterministic init so the gate is reproducible run-to-run (unseeded
  // tensor_rng falls back to std::random_device, which made module weights —
  // and thus the finite-difference margins — vary between runs).
  nsos::determinism::DeterminismManager::instance().set_global_seed(0xC0FFEEULL);
  std::cout << "=== NSOS finite-difference gradcheck gate ===" << std::endl;
  check_rmsnorm();
  check_squared_relu();
  check_cross_entropy();
  check_matmul();
  check_mamba2();
  check_mamba2_proper();
  check_mamba2_nstate();
  check_mamba2_faithful();
  check_mamba2_extreme_decay_finiteness();
  check_mamba2_faithful_wrapper_norms();
  check_proper_streaming_parity(false);
  check_proper_streaming_parity(true);
  check_moe_switch_aux();
  check_moe_router_grad();
  check_kan();
  check_chrass();
  check_attention();
  check_parallel_hybrid_gates();
  if (g_failures != 0) {
    std::printf("\n[gradcheck] %d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::cout << "\nAll gradient checks passed!" << std::endl;
  return 0;
}
