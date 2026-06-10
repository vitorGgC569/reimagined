// ============================================================================
// 25-layer forward-only competitive stack — does it COMPOSE hierarchically?
// ============================================================================
// Forward-only/local learning (no backprop) stacked 25 deep.  We test whether
// features COMPOSE with depth on a deliberately NON-LINEAR task (XOR of part-
// groups), and we instrument it so the honest outcome (rises / plateaus /
// degrades) is read straight off the measured numbers — not asserted.
//
// Inter-layer code: graded k-sparse cosine, L2-normalized (not one-hot).
// Anti-collapse (all local): conscience win-frequency bias + dead-unit reinit
// + winner decorrelation.  Probes (NCC + ridge-linear) are MEASUREMENT ONLY —
// closed-form on frozen codes, they never backprop into the stack.
//
// Build:  cl /nologo /O2 /EHsc /std:c++17 stack_demo.cpp /Fe:stack_demo.exe
#include "lateral_inhibition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int D = 64;             // input dim == layer width (square, clean)
constexpr int WIDTH = 64;         // neurons per layer
constexpr int N_LAYERS = 25;      // Oxta's lucky number
constexpr int P = 16;             // parts in the library
constexpr int GROUPS = 4;         // part-groups
constexpr int PARTS_PER_GROUP = 4;
constexpr int N_CLASSES = 4;      // super-classes (XOR of group-presence)

// ---------------------------------------------------------------------------
// Compositional dataset: super-class = XOR of which part-GROUPS are present.
// Provably not linearly separable from the raw part-sum input.
// ---------------------------------------------------------------------------
struct Dataset {
  std::vector<float> X;  // [N * D]
  std::vector<int> y;    // [N]
  int N = 0;
};

void l2norm(float* v, int d) {
  float n = 0.0f;
  for (int i = 0; i < d; ++i) n += v[i] * v[i];
  n = std::sqrt(n) + 1e-8f;
  for (int i = 0; i < d; ++i) v[i] /= n;
}

std::vector<float> make_parts(uint64_t seed) {  // P unit vectors in R^D
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> parts(static_cast<size_t>(P) * D);
  for (int p = 0; p < P; ++p) {
    float* row = &parts[static_cast<size_t>(p) * D];
    for (int i = 0; i < D; ++i) row[i] = nd(rng);
    l2norm(row, D);
  }
  return parts;
}

Dataset gen(int n, const std::vector<float>& parts, uint64_t seed, float sigma) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::bernoulli_distribution coin(0.5);
  std::uniform_int_distribution<int> pick(0, PARTS_PER_GROUP - 1);
  std::normal_distribution<float> noise(0.0f, sigma);
  Dataset ds;
  ds.N = n;
  ds.X.resize(static_cast<size_t>(n) * D);
  ds.y.resize(static_cast<size_t>(n));
  for (int s = 0; s < n;) {
    int b[GROUPS];
    int present = 0;
    float* x = &ds.X[static_cast<size_t>(s) * D];
    for (int i = 0; i < D; ++i) x[i] = 0.0f;
    for (int g = 0; g < GROUPS; ++g) {
      b[g] = coin(rng) ? 1 : 0;
      present += b[g];
      if (b[g]) {
        const int part = g * PARTS_PER_GROUP + pick(rng);
        const float* pv = &parts[static_cast<size_t>(part) * D];
        for (int i = 0; i < D; ++i) x[i] += pv[i];
      }
    }
    if (present == 0) continue;  // resample empty samples
    for (int i = 0; i < D; ++i) x[i] += noise(rng);
    l2norm(x, D);
    const int q0 = b[0] ^ b[1];
    const int q1 = b[2] ^ b[3];
    ds.y[static_cast<size_t>(s)] = 2 * q0 + q1;  // 4 balanced super-classes
    ++s;
  }
  return ds;
}

// ---------------------------------------------------------------------------
// The stack
// ---------------------------------------------------------------------------
using Layers = std::vector<lat::LateralInhibition>;

std::vector<int> taper_k(int L) {
  std::vector<int> k(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) k[static_cast<size_t>(i)] = (i < 9) ? 4 : (i < 18 ? 3 : 2);
  return k;
}

Layers build_stack(int L, int width, int d_in, const std::vector<int>& k_per,
                   float lr0, uint64_t seed_base, bool conscience, bool decorr) {
  Layers layers;
  layers.reserve(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) {
    const int in = (i == 0) ? d_in : width;
    const float lr = lr0 * (1.0f - static_cast<float>(i + 1) / 50.0f);
    layers.emplace_back(in, width, k_per[static_cast<size_t>(i)], lr, seed_base + static_cast<uint64_t>(i));
    layers.back().set_homeostasis(conscience, decorr);
  }
  return layers;
}

void train_stack(Layers& layers, const Dataset& tr, int epochs, uint64_t seed,
                 bool reinit, int reinit_every) {
  const int L = static_cast<int>(layers.size());
  std::vector<int> idx(static_cast<size_t>(tr.N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::mt19937 rrng(static_cast<uint32_t>(seed ^ 0x9e3779b9u));
  std::vector<std::vector<float>> codes(static_cast<size_t>(L) + 1);
  long seen = 0;
  for (int e = 0; e < epochs; ++e) {
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int s : idx) {
      const float* x0 = &tr.X[static_cast<size_t>(s) * D];
      codes[0].assign(x0, x0 + D);
      for (int i = 0; i < L; ++i)
        layers[static_cast<size_t>(i)].step_code(codes[static_cast<size_t>(i)].data(),
                                                 codes[static_cast<size_t>(i) + 1]);
      ++seen;
      if (reinit && reinit_every > 0 && seen % reinit_every == 0)
        for (int i = 0; i < L; ++i)
          layers[static_cast<size_t>(i)].reinit_dead(
              0.2f / static_cast<float>(layers[static_cast<size_t>(i)].neurons()), rrng);
    }
    std::fprintf(stderr, "    epoch %d/%d done\n", e + 1, epochs);
  }
}

// codes[0] = raw input (D), codes[i] = output of the i-th applied layer.
void encode_stack(Layers& layers, const std::vector<int>& order, const float* x0,
                  std::vector<std::vector<float>>& codes) {
  const int L = static_cast<int>(order.size());
  codes.resize(static_cast<size_t>(L) + 1);
  codes[0].assign(x0, x0 + D);
  for (int i = 0; i < L; ++i)
    layers[static_cast<size_t>(order[static_cast<size_t>(i)])].encode(
        codes[static_cast<size_t>(i)].data(), codes[static_cast<size_t>(i) + 1]);
}

// ---------------------------------------------------------------------------
// Closed-form ridge linear solve (Gauss-Jordan, partial pivot). Result in B.
// A: F*F, B: F*C  ->  B becomes solution X (F*C).
// ---------------------------------------------------------------------------
void solve_ridge(std::vector<double>& A, std::vector<double>& B, int F, int C, double lambda) {
  for (int i = 0; i < F; ++i) A[static_cast<size_t>(i) * F + i] += lambda;
  for (int col = 0; col < F; ++col) {
    int piv = col;
    double best = std::fabs(A[static_cast<size_t>(col) * F + col]);
    for (int r = col + 1; r < F; ++r) {
      const double v = std::fabs(A[static_cast<size_t>(r) * F + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (piv != col) {
      for (int q = 0; q < F; ++q)
        std::swap(A[static_cast<size_t>(col) * F + q], A[static_cast<size_t>(piv) * F + q]);
      for (int q = 0; q < C; ++q)
        std::swap(B[static_cast<size_t>(col) * C + q], B[static_cast<size_t>(piv) * C + q]);
    }
    double d = A[static_cast<size_t>(col) * F + col];
    if (std::fabs(d) < 1e-12) d = (d < 0 ? -1e-12 : 1e-12);
    for (int r = 0; r < F; ++r) {
      if (r == col) continue;
      const double f = A[static_cast<size_t>(r) * F + col] / d;
      if (f == 0.0) continue;
      for (int q = col; q < F; ++q)
        A[static_cast<size_t>(r) * F + q] -= f * A[static_cast<size_t>(col) * F + q];
      for (int q = 0; q < C; ++q)
        B[static_cast<size_t>(r) * C + q] -= f * B[static_cast<size_t>(col) * C + q];
    }
  }
  for (int i = 0; i < F; ++i) {
    double d = A[static_cast<size_t>(i) * F + i];
    if (std::fabs(d) < 1e-12) d = (d < 0 ? -1e-12 : 1e-12);
    for (int q = 0; q < C; ++q) B[static_cast<size_t>(i) * C + q] /= d;
  }
}

int ncc_predict(const std::vector<float>& c, const std::vector<std::vector<float>>& cent) {
  // c is unit-norm; classify by max cosine to each class centroid.
  int best = 0;
  float bv = -1e30f;
  for (int k = 0; k < N_CLASSES; ++k) {
    const auto& m = cent[static_cast<size_t>(k)];
    float dot = 0.0f, nn = 0.0f;
    for (size_t i = 0; i < c.size(); ++i) { dot += c[i] * m[i]; nn += m[i] * m[i]; }
    const float cv = dot / (std::sqrt(nn) + 1e-8f);
    if (cv > bv) { bv = cv; best = k; }
  }
  return best;
}

int lin_predict(const std::vector<float>& c, const std::vector<double>& X, int F, int C) {
  int best = 0;
  double bv = -1e30;
  for (int k = 0; k < C; ++k) {
    double s = 0.0;
    for (int p = 0; p < F; ++p) {
      const double xp = (p < static_cast<int>(c.size())) ? c[static_cast<size_t>(p)] : 1.0;
      s += xp * X[static_cast<size_t>(p) * C + k];
    }
    if (s > bv) { bv = s; best = k; }
  }
  return best;
}

struct Curve {
  std::vector<float> ncc, lin, transferL1, dead, entropy, sparsity, proto_corr, quant;
  std::vector<long> nullc;
};

// Full per-depth evaluation on a FROZEN stack. Measurement only (no learning).
Curve eval_stack(Layers& layers, const std::vector<int>& order, const Dataset& tr,
                 const Dataset& te, bool want_transfer) {
  const int L = static_cast<int>(order.size());
  const int C = N_CLASSES;
  const int F = WIDTH + 1;  // +1 bias
  // FIT accumulators
  std::vector<std::vector<std::vector<float>>> csum(
      static_cast<size_t>(L) + 1,
      std::vector<std::vector<float>>(N_CLASSES, std::vector<float>(WIDTH, 0.0f)));
  std::vector<std::vector<long>> ccount(static_cast<size_t>(L) + 1, std::vector<long>(N_CLASSES, 0));
  std::vector<std::vector<double>> A(static_cast<size_t>(L) + 1,
                                     std::vector<double>(static_cast<size_t>(F) * F, 0.0));
  std::vector<std::vector<double>> B(static_cast<size_t>(L) + 1,
                                     std::vector<double>(static_cast<size_t>(F) * C, 0.0));
  std::vector<std::vector<float>> codes;

  for (int s = 0; s < tr.N; ++s) {
    encode_stack(layers, order, &tr.X[static_cast<size_t>(s) * D], codes);
    const int y = tr.y[static_cast<size_t>(s)];
    for (int d = 0; d <= L; ++d) {
      const auto& c = codes[static_cast<size_t>(d)];
      const int wd = static_cast<int>(c.size());
      for (int j = 0; j < wd; ++j) csum[static_cast<size_t>(d)][static_cast<size_t>(y)][static_cast<size_t>(j)] += c[static_cast<size_t>(j)];
      ccount[static_cast<size_t>(d)][static_cast<size_t>(y)]++;
      auto& Ad = A[static_cast<size_t>(d)];
      auto& Bd = B[static_cast<size_t>(d)];
      for (int p = 0; p < F; ++p) {
        const double xp = (p < wd) ? c[static_cast<size_t>(p)] : 1.0;
        if (xp == 0.0) continue;
        for (int q = 0; q < F; ++q) {
          const double xq = (q < wd) ? c[static_cast<size_t>(q)] : 1.0;
          Ad[static_cast<size_t>(p) * F + q] += xp * xq;
        }
        Bd[static_cast<size_t>(p) * C + y] += xp;
      }
    }
  }

  // finalize centroids + solve linear systems per depth
  std::vector<std::vector<std::vector<float>>> cent = csum;
  for (int d = 0; d <= L; ++d)
    for (int k = 0; k < C; ++k) {
      const long n = std::max(1L, ccount[static_cast<size_t>(d)][static_cast<size_t>(k)]);
      for (int j = 0; j < WIDTH; ++j)
        cent[static_cast<size_t>(d)][static_cast<size_t>(k)][static_cast<size_t>(j)] /= static_cast<float>(n);
    }
  for (int d = 0; d <= L; ++d) solve_ridge(A[static_cast<size_t>(d)], B[static_cast<size_t>(d)], F, C, 1e-3);

  // EVAL on test
  for (auto& l : layers) l.reset_null_codes();
  std::vector<std::vector<long>> hist(static_cast<size_t>(L) + 1, std::vector<long>(WIDTH, 0));
  std::vector<long> ncc_ok(static_cast<size_t>(L) + 1, 0), lin_ok(static_cast<size_t>(L) + 1, 0),
      tr_ok(static_cast<size_t>(L) + 1, 0);
  std::vector<double> sp(static_cast<size_t>(L) + 1, 0.0), qz(static_cast<size_t>(L) + 1, 0.0);
  for (int s = 0; s < te.N; ++s) {
    encode_stack(layers, order, &te.X[static_cast<size_t>(s) * D], codes);
    const int y = te.y[static_cast<size_t>(s)];
    for (int d = 0; d <= L; ++d) {
      const auto& c = codes[static_cast<size_t>(d)];
      const int wd = static_cast<int>(c.size());
      if (ncc_predict(c, cent[static_cast<size_t>(d)]) == y) ncc_ok[static_cast<size_t>(d)]++;
      if (lin_predict(c, B[static_cast<size_t>(d)], F, C) == y) lin_ok[static_cast<size_t>(d)]++;
      if (want_transfer && L >= 1 && ncc_predict(c, cent[1]) == y) tr_ok[static_cast<size_t>(d)]++;
      int wi = 0; float wv = -1e30f; int zeros = 0;
      for (int j = 0; j < wd; ++j) { if (c[static_cast<size_t>(j)] > wv) { wv = c[static_cast<size_t>(j)]; wi = j; } if (c[static_cast<size_t>(j)] == 0.0f) ++zeros; }
      hist[static_cast<size_t>(d)][static_cast<size_t>(wi)]++;
      sp[static_cast<size_t>(d)] += static_cast<double>(zeros) / wd;
      if (d >= 1)
        qz[static_cast<size_t>(d)] += 1.0 - layers[static_cast<size_t>(order[static_cast<size_t>(d) - 1])]
                                               .similarity_to_winner(codes[static_cast<size_t>(d) - 1].data());
    }
  }

  Curve cv;
  auto rs = [&](std::vector<float>& v) { v.assign(static_cast<size_t>(L) + 1, 0.0f); };
  rs(cv.ncc); rs(cv.lin); rs(cv.transferL1); rs(cv.dead); rs(cv.entropy); rs(cv.sparsity); rs(cv.proto_corr); rs(cv.quant);
  cv.nullc.assign(static_cast<size_t>(L) + 1, 0);
  const double nte = static_cast<double>(te.N);
  for (int d = 0; d <= L; ++d) {
    cv.ncc[static_cast<size_t>(d)] = static_cast<float>(ncc_ok[static_cast<size_t>(d)] / nte);
    cv.lin[static_cast<size_t>(d)] = static_cast<float>(lin_ok[static_cast<size_t>(d)] / nte);
    cv.transferL1[static_cast<size_t>(d)] = want_transfer ? static_cast<float>(tr_ok[static_cast<size_t>(d)] / nte) : -1.0f;
    cv.sparsity[static_cast<size_t>(d)] = static_cast<float>(sp[static_cast<size_t>(d)] / nte);
    // dead-unit fraction + winner entropy from the histogram
    int dead = 0; double H = 0.0; long tot = 0;
    for (int j = 0; j < WIDTH; ++j) { tot += hist[static_cast<size_t>(d)][static_cast<size_t>(j)]; if (hist[static_cast<size_t>(d)][static_cast<size_t>(j)] == 0) ++dead; }
    for (int j = 0; j < WIDTH; ++j) {
      const long h = hist[static_cast<size_t>(d)][static_cast<size_t>(j)];
      if (h > 0) { const double p = static_cast<double>(h) / tot; H -= p * std::log(p); }
    }
    cv.dead[static_cast<size_t>(d)] = static_cast<float>(static_cast<double>(dead) / WIDTH);
    cv.entropy[static_cast<size_t>(d)] = static_cast<float>(H / std::log(static_cast<double>(WIDTH)));
    if (d >= 1) {
      cv.proto_corr[static_cast<size_t>(d)] = layers[static_cast<size_t>(order[static_cast<size_t>(d) - 1])].mean_pairwise_proto_cos();
      cv.quant[static_cast<size_t>(d)] = static_cast<float>(qz[static_cast<size_t>(d)] / nte);
      cv.nullc[static_cast<size_t>(d)] = layers[static_cast<size_t>(order[static_cast<size_t>(d) - 1])].null_codes();
    }
  }
  return cv;
}

// Standalone single-layer probe for the width-scaled baseline (any code dim).
void probe_single(lat::LateralInhibition& layer, const Dataset& tr, const Dataset& te,
                  float& ncc_acc, float& lin_acc) {
  const int W = layer.neurons();
  const int C = N_CLASSES;
  const int F = W + 1;
  std::vector<std::vector<float>> cent(N_CLASSES, std::vector<float>(static_cast<size_t>(W), 0.0f));
  std::vector<long> cnt(N_CLASSES, 0);
  std::vector<double> A(static_cast<size_t>(F) * F, 0.0), Bv(static_cast<size_t>(F) * C, 0.0);
  std::vector<float> c;
  for (int s = 0; s < tr.N; ++s) {
    layer.encode(&tr.X[static_cast<size_t>(s) * D], c);
    const int y = tr.y[static_cast<size_t>(s)];
    for (int j = 0; j < W; ++j) cent[static_cast<size_t>(y)][static_cast<size_t>(j)] += c[static_cast<size_t>(j)];
    cnt[static_cast<size_t>(y)]++;
    for (int p = 0; p < F; ++p) {
      const double xp = (p < W) ? c[static_cast<size_t>(p)] : 1.0;
      if (xp == 0.0) continue;
      for (int q = 0; q < F; ++q) { const double xq = (q < W) ? c[static_cast<size_t>(q)] : 1.0; A[static_cast<size_t>(p) * F + q] += xp * xq; }
      Bv[static_cast<size_t>(p) * C + y] += xp;
    }
  }
  for (int k = 0; k < C; ++k) { const long n = std::max(1L, cnt[static_cast<size_t>(k)]); for (int j = 0; j < W; ++j) cent[static_cast<size_t>(k)][static_cast<size_t>(j)] /= static_cast<float>(n); }
  solve_ridge(A, Bv, F, C, 1e-3);
  long no = 0, lo = 0;
  for (int s = 0; s < te.N; ++s) {
    layer.encode(&te.X[static_cast<size_t>(s) * D], c);
    const int y = te.y[static_cast<size_t>(s)];
    if (ncc_predict(c, cent) == y) ++no;
    if (lin_predict(c, Bv, F, C) == y) ++lo;
  }
  ncc_acc = static_cast<float>(static_cast<double>(no) / te.N);
  lin_acc = static_cast<float>(static_cast<double>(lo) / te.N);
}

float best_after1(const std::vector<float>& v) {
  float b = -1.0f;
  for (size_t i = 1; i < v.size(); ++i) b = std::max(b, v[i]);
  return b;
}
int argbest_after1(const std::vector<float>& v) {
  int bi = 1; float b = -1.0f;
  for (int i = 1; i < static_cast<int>(v.size()); ++i) if (v[static_cast<size_t>(i)] > b) { b = v[static_cast<size_t>(i)]; bi = i; }
  return bi;
}

// ---------------------------------------------------------------------------
// OXTA-MEM: remember every layer's code; the readout composes over the UNION.
// Tests whether PRESERVING each layer's contribution (instead of only the
// collapsed top code) makes depth productive — i.e. does the memory-prefix
// readout RISE with depth even though the per-layer code erodes?
// Prefix NCC at depth d uses the concatenation of codes from layers 1..d.
// Full linear uses the whole 1..L concatenation (== same total dim as the
// width-scaled single layer, built incrementally by depth).
// ---------------------------------------------------------------------------
struct MemCurve { std::vector<float> ncc; float full_lin = 0.0f; };

MemCurve memory_readout(Layers& layers, const std::vector<int>& order,
                        const Dataset& tr, const Dataset& te) {
  const int L = static_cast<int>(order.size());
  const int FD = L * WIDTH;     // full concat dim (layers 1..L)
  const int C = N_CLASSES;
  const int F = FD + 1;         // + bias for the linear probe
  std::vector<std::vector<double>> csum(N_CLASSES, std::vector<double>(static_cast<size_t>(FD), 0.0));
  std::vector<long> cnt(N_CLASSES, 0);
  std::vector<double> A(static_cast<size_t>(F) * F, 0.0), Bv(static_cast<size_t>(F) * C, 0.0);
  std::vector<std::vector<float>> codes;
  std::vector<float> feat(static_cast<size_t>(FD));
  auto build_feat = [&](const std::vector<std::vector<float>>& cd) {
    for (int d = 0; d < L; ++d)
      for (int j = 0; j < WIDTH; ++j)
        feat[static_cast<size_t>(d) * WIDTH + j] = cd[static_cast<size_t>(d) + 1][static_cast<size_t>(j)];
  };
  for (int s = 0; s < tr.N; ++s) {
    encode_stack(layers, order, &tr.X[static_cast<size_t>(s) * D], codes);
    build_feat(codes);
    const int y = tr.y[static_cast<size_t>(s)];
    for (int i = 0; i < FD; ++i) csum[static_cast<size_t>(y)][static_cast<size_t>(i)] += feat[static_cast<size_t>(i)];
    cnt[static_cast<size_t>(y)]++;
    for (int p = 0; p < F; ++p) {
      const double xp = (p < FD) ? feat[static_cast<size_t>(p)] : 1.0;
      if (xp == 0.0) continue;
      for (int q = 0; q < F; ++q) { const double xq = (q < FD) ? feat[static_cast<size_t>(q)] : 1.0; A[static_cast<size_t>(p) * F + q] += xp * xq; }
      Bv[static_cast<size_t>(p) * C + y] += xp;
    }
  }
  std::vector<std::vector<float>> cent(N_CLASSES, std::vector<float>(static_cast<size_t>(FD), 0.0f));
  for (int k = 0; k < C; ++k) { const long n = std::max(1L, cnt[static_cast<size_t>(k)]); for (int i = 0; i < FD; ++i) cent[static_cast<size_t>(k)][static_cast<size_t>(i)] = static_cast<float>(csum[static_cast<size_t>(k)][static_cast<size_t>(i)] / n); }
  std::fprintf(stderr, "[oxta-mem] solving full %d-dim linear...\n", FD);
  solve_ridge(A, Bv, F, C, 1e-3);

  std::vector<long> ncc_ok(static_cast<size_t>(L) + 1, 0);
  long lin_ok = 0;
  for (int s = 0; s < te.N; ++s) {
    encode_stack(layers, order, &te.X[static_cast<size_t>(s) * D], codes);
    build_feat(codes);
    const int y = te.y[static_cast<size_t>(s)];
    for (int d = 1; d <= L; ++d) {
      const int dim = d * WIDTH;
      int best = 0; float bv = -1e30f;
      for (int k = 0; k < C; ++k) {
        float dot = 0.0f, nn = 0.0f;
        for (int i = 0; i < dim; ++i) { dot += feat[static_cast<size_t>(i)] * cent[static_cast<size_t>(k)][static_cast<size_t>(i)]; nn += cent[static_cast<size_t>(k)][static_cast<size_t>(i)] * cent[static_cast<size_t>(k)][static_cast<size_t>(i)]; }
        const float cv = dot / (std::sqrt(nn) + 1e-8f);
        if (cv > bv) { bv = cv; best = k; }
      }
      if (best == y) ncc_ok[static_cast<size_t>(d)]++;
    }
    int best = 0; double bv = -1e30;
    for (int k = 0; k < C; ++k) {
      double s2 = 0.0;
      for (int p = 0; p < F; ++p) { const double xp = (p < FD) ? feat[static_cast<size_t>(p)] : 1.0; s2 += xp * Bv[static_cast<size_t>(p) * C + k]; }
      if (s2 > bv) { bv = s2; best = k; }
    }
    if (best == y) ++lin_ok;
  }
  MemCurve mc;
  mc.ncc.assign(static_cast<size_t>(L) + 1, 0.0f);
  for (int d = 1; d <= L; ++d) mc.ncc[static_cast<size_t>(d)] = static_cast<float>(static_cast<double>(ncc_ok[static_cast<size_t>(d)]) / te.N);
  mc.full_lin = static_cast<float>(static_cast<double>(lin_ok) / te.N);
  return mc;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::vector<float> parts = make_parts(101);
  const Dataset tr = gen(12000, parts, 7, 0.10f);
  const Dataset te = gen(3000, parts, 8, 0.10f);
  const std::vector<int> kT = taper_k(N_LAYERS);
  const float chance = 1.0f / N_CLASSES;
  std::vector<int> ident(N_LAYERS);
  std::iota(ident.begin(), ident.end(), 0);

  std::printf("=====================================================================\n");
  std::printf(" 25-LAYER FORWARD-ONLY COMPETITIVE STACK — hierarchical composition?\n");
  std::printf("=====================================================================\n");
  std::printf(" task: super-class = (bG0 XOR bG1, bG2 XOR bG3)  [non-linear of parts]\n");
  std::printf(" data: %d train / %d test, dim=%d, %d parts/%d groups, %d classes, chance=%.3f\n",
              tr.N, te.N, D, P, GROUPS, N_CLASSES, chance);
  std::printf(" stack: %d layers x %d neurons, k-taper 4/3/2, graded sparse codes\n",
              N_LAYERS, WIDTH);
  std::printf(" homeostasis: conscience + dead-unit reinit + winner decorrelation\n\n");

  // -------- CORE STACK --------
  std::fprintf(stderr, "[core] training...\n");
  Layers core = build_stack(N_LAYERS, WIDTH, D, kT, 0.10f, 1000, true, true);
  train_stack(core, tr, 6, 11, true, 2000);
  std::fprintf(stderr, "[core] evaluating...\n");
  const Curve cc = eval_stack(core, ident, tr, te, true);

  std::printf("PER-DEPTH (core stack)  [readout = measurement only, no backprop]\n");
  std::printf(" depth  ncc    lin    L1xfer  dead   entropy sparse protoC  quant  null\n");
  for (int d = 0; d <= N_LAYERS; ++d) {
    if (d == 0)
      std::printf(" %3d*  %.3f  %.3f  %.3f    -      -      %.3f    -      -      -\n",
                  d, cc.ncc[d], cc.lin[d], cc.transferL1[d], cc.sparsity[d]);
    else
      std::printf(" %3d   %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %ld\n",
                  d, cc.ncc[d], cc.lin[d], cc.transferL1[d], cc.dead[d], cc.entropy[d],
                  cc.sparsity[d], cc.proto_corr[d], cc.quant[d], cc.nullc[d]);
  }
  std::printf(" (* depth 0 = raw input)\n\n");

  // -------- BASELINES --------
  std::fprintf(stderr, "[inhib-off] training...\n");
  std::vector<int> kFull(N_LAYERS, WIDTH);
  Layers inhibOff = build_stack(N_LAYERS, WIDTH, D, kFull, 0.10f, 2000, false, false);
  train_stack(inhibOff, tr, 6, 11, false, 0);
  const Curve cInhib = eval_stack(inhibOff, ident, tr, te, false);

  std::fprintf(stderr, "[anti-collapse-off] training...\n");
  Layers acOff = build_stack(N_LAYERS, WIDTH, D, kT, 0.10f, 3000, false, false);
  train_stack(acOff, tr, 6, 11, false, 0);
  const Curve cAc = eval_stack(acOff, ident, tr, te, false);

  std::fprintf(stderr, "[const-k=3] training...\n");
  std::vector<int> kConst(N_LAYERS, 3);
  Layers ck = build_stack(N_LAYERS, WIDTH, D, kConst, 0.10f, 4000, true, true);
  train_stack(ck, tr, 6, 11, true, 2000);
  const Curve cCk = eval_stack(ck, ident, tr, te, false);

  std::fprintf(stderr, "[random-frozen] (no training)...\n");
  Layers rnd = build_stack(N_LAYERS, WIDTH, D, kT, 0.10f, 5000, true, true);
  const Curve cRnd = eval_stack(rnd, ident, tr, te, false);

  std::fprintf(stderr, "[shuffled-depth] (reuse core)...\n");
  std::vector<int> perm = ident;
  std::shuffle(perm.begin(), perm.end(), std::mt19937(424242u));
  const Curve cShuf = eval_stack(core, perm, tr, te, false);

  std::fprintf(stderr, "[width-scaled single layer m=1600] training...\n");
  lat::LateralInhibition wide(D, WIDTH * N_LAYERS, 4, 0.10f, 6000);
  wide.set_homeostasis(true, true);
  {
    std::vector<int> idx(static_cast<size_t>(tr.N));
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(11u);
    std::vector<float> cscratch;
    for (int e = 0; e < 6; ++e) {
      std::shuffle(idx.begin(), idx.end(), rng);
      for (int s : idx) wide.step_code(&tr.X[static_cast<size_t>(s) * D], cscratch);
      std::fprintf(stderr, "    wide epoch %d/6\n", e + 1);
    }
  }
  float wideNcc = 0.0f, wideLin = 0.0f;
  probe_single(wide, tr, te, wideNcc, wideLin);

  std::fprintf(stderr, "[oxta-mem] evaluating memory-of-all-layers readout...\n");
  const MemCurve mem = memory_readout(core, ident, tr, te);

  // -------- SUMMARY + VERDICT (derived from measured numbers) --------
  const float raw = cc.ncc[0];
  const float single = cc.ncc[1];
  const int peak_d = argbest_after1(cc.ncc);
  const float peak = cc.ncc[static_cast<size_t>(peak_d)];
  const float finalAcc = cc.ncc[N_LAYERS];
  const float shufFinal = cShuf.ncc[N_LAYERS];
  const float rndBest = best_after1(cRnd.ncc);
  const float wideBest = std::max(wideNcc, wideLin);

  std::printf("CONTROLS (NCC unless noted):\n");
  std::printf("  raw-input floor (depth 0) .............. %.3f  (chance %.3f)\n", raw, chance);
  std::printf("  single-layer (depth 1) ................. %.3f\n", single);
  std::printf("  CORE best  (depth %2d) .................. %.3f\n", peak_d, peak);
  std::printf("  CORE final (depth 25) ................. %.3f\n", finalAcc);
  std::printf("  inhibition-OFF stack (best/final) ..... %.3f / %.3f  [expect collapse]\n",
              best_after1(cInhib.ncc), cInhib.ncc[N_LAYERS]);
  std::printf("  anti-collapse-OFF (best/final/dead@25). %.3f / %.3f / dead=%.2f\n",
              best_after1(cAc.ncc), cAc.ncc[N_LAYERS], cAc.dead[N_LAYERS]);
  std::printf("  const-k=3 stack (best) ................ %.3f\n", best_after1(cCk.ncc));
  std::printf("  random-frozen stack (best) ........... %.3f\n", rndBest);
  std::printf("  shuffled-depth (final) ............... %.3f  (in-order final %.3f)\n", shufFinal, finalAcc);
  std::printf("  width-scaled 1-layer m=1600 (ncc/lin). %.3f / %.3f\n\n", wideNcc, wideLin);

  std::printf("OXTA-MEM (remember every layer's code, compose over the union):\n");
  std::printf("  memory-prefix NCC over depth 1..d (does it RISE where per-layer eroded?):\n");
  std::printf("    d= 1: %.3f   d= 2: %.3f   d= 3: %.3f   d= 5: %.3f\n",
              mem.ncc[1], mem.ncc[2], mem.ncc[3], mem.ncc[5]);
  std::printf("    d=10: %.3f   d=15: %.3f   d=20: %.3f   d=25: %.3f\n",
              mem.ncc[10], mem.ncc[15], mem.ncc[20], mem.ncc[25]);
  std::printf("  per-layer NCC for contrast:        d=1: %.3f  ...  d=25: %.3f (eroded)\n",
              cc.ncc[1], cc.ncc[N_LAYERS]);
  std::printf("  FULL-memory linear (1600-dim) ........ %.3f\n", mem.full_lin);
  std::printf("    vs width-scaled 1-layer linear ..... %.3f   vs deep per-layer best linear %.3f\n\n",
              wideLin, best_after1(cc.lin));
  const bool mem_rises = (mem.ncc[N_LAYERS] - mem.ncc[1]) >= 0.03f;
  std::printf("  => memory readout %s with depth (d25 %.3f vs d1 %.3f): %s\n\n",
              mem_rises ? "RISES" : "does NOT rise", mem.ncc[N_LAYERS], mem.ncc[1],
              mem_rises ? "each layer ADDS info to memory even as its own code collapses"
                        : "deeper layers add little new info to the memory");

  // honesty booleans, computed straight from the numbers
  const bool beats_single = (peak - single) >= 0.10f;
  const bool beats_raw = (peak - raw) >= 0.20f;
  const bool beats_capacity = peak >= wideBest;
  const bool order_matters = (finalAcc - shufFinal) >= 0.05f;
  const bool learning_real = rndBest < 0.35f;
  const bool degrades = (peak - finalAcc) >= 0.10f && peak_d <= 5;
  const bool healthy = cc.dead[static_cast<size_t>(peak_d)] < 0.25f && cc.entropy[static_cast<size_t>(peak_d)] > 0.6f;

  std::printf("HONESTY CHECKS:\n");
  std::printf("  [1] best beats single-layer by >=0.10 ... %s (%.3f -> %.3f, +%.3f)\n",
              beats_single ? "YES" : "NO ", single, peak, peak - single);
  std::printf("  [2] best beats raw floor by >=0.20 ...... %s (+%.3f)\n", beats_raw ? "YES" : "NO ", peak - raw);
  std::printf("  [3] depth beats equal-capacity width .... %s (deep %.3f vs wide %.3f)\n",
              beats_capacity ? "YES" : "NO ", peak, wideBest);
  std::printf("  [4] order matters (shuffle drops >=0.05). %s (%.3f drop)\n",
              order_matters ? "YES" : "NO ", finalAcc - shufFinal);
  std::printf("  [5] learning is real (rnd < 0.35) ....... %s (%.3f)\n", learning_real ? "YES" : "NO ", rndBest);
  std::printf("  [6] representation stays healthy @peak .. %s (dead %.2f, entropy %.2f)\n",
              healthy ? "YES" : "NO ", cc.dead[static_cast<size_t>(peak_d)], cc.entropy[static_cast<size_t>(peak_d)]);

  std::printf("\nVERDICT: ");
  if (beats_single && beats_raw && beats_capacity && learning_real) {
    std::printf("FEATURES COMPOSE — depth helps (peak at layer %d: %.3f, +%.3f over single).\n",
                peak_d, peak, peak - single);
    if (degrades)
      std::printf("         ...but then DEGRADES toward depth 25 (%.3f). Composition is EARLY, not all-25.\n", finalAcc);
    else if ((peak - finalAcc) < 0.05f && peak_d >= N_LAYERS - 3)
      std::printf("         ...and KEEPS improving to the deep layers (rare, strong result).\n");
    else
      std::printf("         ...then PLATEAUS; marginal value of extra depth ~0 after layer %d.\n", peak_d);
  } else if (degrades) {
    std::printf("DEGRADES — accuracy peaks early (layer %d: %.3f) then falls to %.3f. Stacking erodes info.\n",
                peak_d, peak, finalAcc);
  } else {
    std::printf("PLATEAU — no depth beats single-layer by >=0.10 (best %.3f vs single %.3f). Layer 1 already captures it.\n",
                peak, single);
  }
  std::printf("=====================================================================\n");
  return 0;
}
