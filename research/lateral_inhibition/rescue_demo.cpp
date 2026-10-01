// ============================================================================
// RESCUE experiments — can DEPTH compose forward-only if we fix the bottleneck?
// ============================================================================
// Driven by the adversarial verification of the negative result:
//   * LR-matched (constant 0.10 everywhere) — removes the wide-vs-deep confound.
//   * SKIP-CONCAT: every deep layer also sees the RAW input ([raw, prev_code]).
//                  == "Oxta-mem of the input": diversity so memory isn't redundant.
//   * SOFT-TEMP dense inter-layer code (tau): tests the pure-bottleneck hypothesis.
//   * WIDE LR sweep (m=1600 at lr 0.05/0.074/0.10): bounds the width number.
//   * MEMORY-concat linear on the skip stack (full 1600-dim union readout).
//
// Build: cl /nologo /O2 /EHsc /std:c++17 rescue_demo.cpp /Fe:rescue_demo.exe
#include "lateral_inhibition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int D = 64, WIDTH = 64, N_LAYERS = 25;
constexpr int P = 16, GROUPS = 4, PARTS_PER_GROUP = 4, N_CLASSES = 4;

void l2norm(float* v, int d) {
  float n = 0.0f;
  for (int i = 0; i < d; ++i) n += v[i] * v[i];
  n = std::sqrt(n) + 1e-8f;
  for (int i = 0; i < d; ++i) v[i] /= n;
}

struct Dataset { std::vector<float> X; std::vector<int> y; int N = 0; };

std::vector<float> make_parts(uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> parts(static_cast<size_t>(P) * D);
  for (int p = 0; p < P; ++p) { float* r = &parts[static_cast<size_t>(p) * D]; for (int i = 0; i < D; ++i) r[i] = nd(rng); l2norm(r, D); }
  return parts;
}

Dataset gen(int n, const std::vector<float>& parts, uint64_t seed, float sigma) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::bernoulli_distribution coin(0.5);
  std::uniform_int_distribution<int> pick(0, PARTS_PER_GROUP - 1);
  std::normal_distribution<float> noise(0.0f, sigma);
  Dataset ds; ds.N = n; ds.X.resize(static_cast<size_t>(n) * D); ds.y.resize(static_cast<size_t>(n));
  for (int s = 0; s < n;) {
    int b[GROUPS]; int present = 0; float* x = &ds.X[static_cast<size_t>(s) * D];
    for (int i = 0; i < D; ++i) x[i] = 0.0f;
    for (int g = 0; g < GROUPS; ++g) {
      b[g] = coin(rng) ? 1 : 0; present += b[g];
      if (b[g]) { const int part = g * PARTS_PER_GROUP + pick(rng); const float* pv = &parts[static_cast<size_t>(part) * D]; for (int i = 0; i < D; ++i) x[i] += pv[i]; }
    }
    if (present == 0) continue;
    for (int i = 0; i < D; ++i) x[i] += noise(rng);
    l2norm(x, D);
    ds.y[static_cast<size_t>(s)] = 2 * (b[0] ^ b[1]) + (b[2] ^ b[3]);
    ++s;
  }
  return ds;
}

std::vector<int> taper_k(int L) {
  std::vector<int> k(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) k[static_cast<size_t>(i)] = (i < 9) ? 4 : (i < 18 ? 3 : 2);
  return k;
}

void solve_ridge(std::vector<double>& A, std::vector<double>& B, int F, int C, double lambda) {
  for (int i = 0; i < F; ++i) A[static_cast<size_t>(i) * F + i] += lambda;
  for (int col = 0; col < F; ++col) {
    int piv = col; double best = std::fabs(A[static_cast<size_t>(col) * F + col]);
    for (int r = col + 1; r < F; ++r) { const double v = std::fabs(A[static_cast<size_t>(r) * F + col]); if (v > best) { best = v; piv = r; } }
    if (piv != col) { for (int q = 0; q < F; ++q) std::swap(A[static_cast<size_t>(col) * F + q], A[static_cast<size_t>(piv) * F + q]); for (int q = 0; q < C; ++q) std::swap(B[static_cast<size_t>(col) * C + q], B[static_cast<size_t>(piv) * C + q]); }
    double d = A[static_cast<size_t>(col) * F + col]; if (std::fabs(d) < 1e-12) d = (d < 0 ? -1e-12 : 1e-12);
    for (int r = 0; r < F; ++r) { if (r == col) continue; const double f = A[static_cast<size_t>(r) * F + col] / d; if (f == 0.0) continue; for (int q = col; q < F; ++q) A[static_cast<size_t>(r) * F + q] -= f * A[static_cast<size_t>(col) * F + q]; for (int q = 0; q < C; ++q) B[static_cast<size_t>(r) * C + q] -= f * B[static_cast<size_t>(col) * C + q]; }
  }
  for (int i = 0; i < F; ++i) { double d = A[static_cast<size_t>(i) * F + i]; if (std::fabs(d) < 1e-12) d = (d < 0 ? -1e-12 : 1e-12); for (int q = 0; q < C; ++q) B[static_cast<size_t>(i) * C + q] /= d; }
}

int ncc_predict(const float* c, int dim, const std::vector<std::vector<float>>& cent) {
  int best = 0; float bv = -1e30f;
  for (int k = 0; k < N_CLASSES; ++k) {
    const auto& m = cent[static_cast<size_t>(k)]; float dot = 0.0f, nn = 0.0f;
    for (int i = 0; i < dim; ++i) { dot += c[i] * m[static_cast<size_t>(i)]; nn += m[static_cast<size_t>(i)] * m[static_cast<size_t>(i)]; }
    const float cv = dot / (std::sqrt(nn) + 1e-8f); if (cv > bv) { bv = cv; best = k; }
  }
  return best;
}

int lin_predict(const float* c, int dim, const std::vector<double>& X, int F, int C) {
  int best = 0; double bv = -1e30;
  for (int k = 0; k < C; ++k) { double s = 0.0; for (int p = 0; p < F; ++p) { const double xp = (p < dim) ? c[p] : 1.0; s += xp * X[static_cast<size_t>(p) * C + k]; } if (s > bv) { bv = s; best = k; } }
  return best;
}

// ---- mode-aware stack ----
enum Mode { CODE_ONLY, SKIP, SOFT };
using Layers = std::vector<lat::LateralInhibition>;

Layers build_stack(Mode mode, int L, const std::vector<int>& k, float lr0, uint64_t seed) {
  Layers layers; layers.reserve(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) {
    const int in = (i == 0) ? D : (mode == SKIP ? D + WIDTH : WIDTH);
    layers.emplace_back(in, WIDTH, k[static_cast<size_t>(i)], lr0, seed + static_cast<uint64_t>(i));
    layers.back().set_homeostasis(true, true);
  }
  return layers;
}

// build the input to layer i from raw + previous code
void layer_input(Mode mode, int i, const float* raw, const std::vector<float>& prev, std::vector<float>& buf) {
  if (mode == SKIP && i >= 1) {
    buf.resize(static_cast<size_t>(D) + WIDTH);
    for (int j = 0; j < D; ++j) buf[static_cast<size_t>(j)] = raw[j];
    for (int j = 0; j < WIDTH; ++j) buf[static_cast<size_t>(D) + j] = prev[static_cast<size_t>(j)];
  } else {
    buf.assign(prev.begin(), prev.end());
  }
}

void train(Layers& layers, Mode mode, float tau, const Dataset& tr, int epochs, uint64_t seed) {
  const int L = static_cast<int>(layers.size());
  std::vector<int> idx(static_cast<size_t>(tr.N)); std::iota(idx.begin(), idx.end(), 0);
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::mt19937 rrng(static_cast<uint32_t>(seed ^ 0x9e3779b9u));
  std::vector<float> codes_prev, buf, code_out; long seen = 0;
  for (int e = 0; e < epochs; ++e) {
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int s : idx) {
      const float* raw = &tr.X[static_cast<size_t>(s) * D];
      codes_prev.assign(raw, raw + D);  // codes[0] = raw
      for (int i = 0; i < L; ++i) {
        layer_input(mode, i, raw, codes_prev, buf);
        if (mode == SOFT) layers[static_cast<size_t>(i)].step_code_soft(buf.data(), code_out, tau);
        else layers[static_cast<size_t>(i)].step_code(buf.data(), code_out);
        codes_prev = code_out;
      }
      if (++seen % 2000 == 0)
        for (int i = 0; i < L; ++i) layers[static_cast<size_t>(i)].reinit_dead(0.2f / WIDTH, rrng);
    }
    std::fprintf(stderr, "    epoch %d/%d\n", e + 1, epochs);
  }
}

// produce per-depth codes (codes[0]=raw, codes[i]=output of layer i-1)
void encode(Layers& layers, Mode mode, float tau, const float* raw, std::vector<std::vector<float>>& codes) {
  const int L = static_cast<int>(layers.size());
  codes.resize(static_cast<size_t>(L) + 1);
  codes[0].assign(raw, raw + D);
  std::vector<float> buf;
  for (int i = 0; i < L; ++i) {
    layer_input(mode, i, raw, codes[static_cast<size_t>(i)], buf);
    if (mode == SOFT) layers[static_cast<size_t>(i)].encode_soft(buf.data(), codes[static_cast<size_t>(i) + 1], tau);
    else layers[static_cast<size_t>(i)].encode(buf.data(), codes[static_cast<size_t>(i) + 1]);
  }
}

struct Curve { std::vector<float> ncc, lin, dead, entropy; };

Curve eval(Layers& layers, Mode mode, float tau, const Dataset& tr, const Dataset& te) {
  const int L = static_cast<int>(layers.size()); const int C = N_CLASSES; const int F = WIDTH + 1;
  std::vector<std::vector<std::vector<float>>> csum(static_cast<size_t>(L) + 1, std::vector<std::vector<float>>(N_CLASSES, std::vector<float>(WIDTH, 0.0f)));
  std::vector<std::vector<long>> ccount(static_cast<size_t>(L) + 1, std::vector<long>(N_CLASSES, 0));
  std::vector<std::vector<double>> A(static_cast<size_t>(L) + 1, std::vector<double>(static_cast<size_t>(F) * F, 0.0));
  std::vector<std::vector<double>> B(static_cast<size_t>(L) + 1, std::vector<double>(static_cast<size_t>(F) * C, 0.0));
  std::vector<std::vector<float>> codes;
  for (int s = 0; s < tr.N; ++s) {
    encode(layers, mode, tau, &tr.X[static_cast<size_t>(s) * D], codes);
    const int y = tr.y[static_cast<size_t>(s)];
    for (int d = 0; d <= L; ++d) {
      const auto& c = codes[static_cast<size_t>(d)]; const int wd = static_cast<int>(c.size());
      for (int j = 0; j < wd; ++j) csum[static_cast<size_t>(d)][static_cast<size_t>(y)][static_cast<size_t>(j)] += c[static_cast<size_t>(j)];
      ccount[static_cast<size_t>(d)][static_cast<size_t>(y)]++;
      auto& Ad = A[static_cast<size_t>(d)]; auto& Bd = B[static_cast<size_t>(d)];
      for (int p = 0; p < F; ++p) { const double xp = (p < wd) ? c[static_cast<size_t>(p)] : 1.0; if (xp == 0.0) continue; for (int q = 0; q < F; ++q) { const double xq = (q < wd) ? c[static_cast<size_t>(q)] : 1.0; Ad[static_cast<size_t>(p) * F + q] += xp * xq; } Bd[static_cast<size_t>(p) * C + y] += xp; }
    }
  }
  std::vector<std::vector<std::vector<float>>> cent = csum;
  for (int d = 0; d <= L; ++d) for (int k = 0; k < C; ++k) { const long n = std::max(1L, ccount[static_cast<size_t>(d)][static_cast<size_t>(k)]); for (int j = 0; j < WIDTH; ++j) cent[static_cast<size_t>(d)][static_cast<size_t>(k)][static_cast<size_t>(j)] /= static_cast<float>(n); }
  for (int d = 0; d <= L; ++d) solve_ridge(A[static_cast<size_t>(d)], B[static_cast<size_t>(d)], F, C, 1e-3);
  std::vector<std::vector<long>> hist(static_cast<size_t>(L) + 1, std::vector<long>(WIDTH, 0));
  std::vector<long> no(static_cast<size_t>(L) + 1, 0), lo(static_cast<size_t>(L) + 1, 0);
  for (int s = 0; s < te.N; ++s) {
    encode(layers, mode, tau, &te.X[static_cast<size_t>(s) * D], codes);
    const int y = te.y[static_cast<size_t>(s)];
    for (int d = 0; d <= L; ++d) {
      const auto& c = codes[static_cast<size_t>(d)]; const int wd = static_cast<int>(c.size());
      if (ncc_predict(c.data(), wd, cent[static_cast<size_t>(d)]) == y) no[static_cast<size_t>(d)]++;
      if (lin_predict(c.data(), wd, B[static_cast<size_t>(d)], F, C) == y) lo[static_cast<size_t>(d)]++;
      int wi = 0; float wv = -1e30f; for (int j = 0; j < wd; ++j) if (c[static_cast<size_t>(j)] > wv) { wv = c[static_cast<size_t>(j)]; wi = j; }
      hist[static_cast<size_t>(d)][static_cast<size_t>(wi)]++;
    }
  }
  Curve cv; cv.ncc.assign(static_cast<size_t>(L) + 1, 0); cv.lin.assign(static_cast<size_t>(L) + 1, 0); cv.dead.assign(static_cast<size_t>(L) + 1, 0); cv.entropy.assign(static_cast<size_t>(L) + 1, 0);
  const double nte = te.N;
  for (int d = 0; d <= L; ++d) {
    cv.ncc[static_cast<size_t>(d)] = static_cast<float>(no[static_cast<size_t>(d)] / nte);
    cv.lin[static_cast<size_t>(d)] = static_cast<float>(lo[static_cast<size_t>(d)] / nte);
    int dead = 0; double H = 0.0; long tot = 0;
    for (int j = 0; j < WIDTH; ++j) { tot += hist[static_cast<size_t>(d)][static_cast<size_t>(j)]; if (hist[static_cast<size_t>(d)][static_cast<size_t>(j)] == 0) ++dead; }
    for (int j = 0; j < WIDTH; ++j) { const long h = hist[static_cast<size_t>(d)][static_cast<size_t>(j)]; if (h > 0) { const double p = static_cast<double>(h) / tot; H -= p * std::log(p); } }
    cv.dead[static_cast<size_t>(d)] = static_cast<float>(static_cast<double>(dead) / WIDTH);
    cv.entropy[static_cast<size_t>(d)] = static_cast<float>(H / std::log(static_cast<double>(WIDTH)));
  }
  return cv;
}

// full memory-concat linear (layers 1..L union) for a trained stack
float memory_linear(Layers& layers, Mode mode, float tau, const Dataset& tr, const Dataset& te) {
  const int L = static_cast<int>(layers.size()); const int FD = L * WIDTH; const int C = N_CLASSES; const int F = FD + 1;
  std::vector<double> A(static_cast<size_t>(F) * F, 0.0), B(static_cast<size_t>(F) * C, 0.0);
  std::vector<std::vector<float>> codes; std::vector<float> feat(static_cast<size_t>(FD));
  auto bf = [&](const std::vector<std::vector<float>>& cd) { for (int d = 0; d < L; ++d) for (int j = 0; j < WIDTH; ++j) feat[static_cast<size_t>(d) * WIDTH + j] = cd[static_cast<size_t>(d) + 1][static_cast<size_t>(j)]; };
  for (int s = 0; s < tr.N; ++s) { encode(layers, mode, tau, &tr.X[static_cast<size_t>(s) * D], codes); bf(codes); const int y = tr.y[static_cast<size_t>(s)]; for (int p = 0; p < F; ++p) { const double xp = (p < FD) ? feat[static_cast<size_t>(p)] : 1.0; if (xp == 0.0) continue; for (int q = 0; q < F; ++q) { const double xq = (q < FD) ? feat[static_cast<size_t>(q)] : 1.0; A[static_cast<size_t>(p) * F + q] += xp * xq; } B[static_cast<size_t>(p) * C + y] += xp; } }
  std::fprintf(stderr, "    [mem] solving %d-dim...\n", FD);
  solve_ridge(A, B, F, C, 1e-3);
  long ok = 0;
  for (int s = 0; s < te.N; ++s) { encode(layers, mode, tau, &te.X[static_cast<size_t>(s) * D], codes); bf(codes); const int y = te.y[static_cast<size_t>(s)]; if (lin_predict(feat.data(), FD, B, F, C) == y) ++ok; }
  return static_cast<float>(static_cast<double>(ok) / te.N);
}

void wide_probe(const Dataset& tr, const Dataset& te, float lr, uint64_t seed, float& ncc, float& lin) {
  lat::LateralInhibition w(D, WIDTH * N_LAYERS, 4, lr, seed); w.set_homeostasis(true, true);
  std::vector<int> idx(static_cast<size_t>(tr.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(11u); std::vector<float> c;
  for (int e = 0; e < 6; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) w.step_code(&tr.X[static_cast<size_t>(s) * D], c); }
  const int W = w.neurons(); const int C = N_CLASSES; const int F = W + 1;
  std::vector<std::vector<float>> cent(N_CLASSES, std::vector<float>(static_cast<size_t>(W), 0.0f)); std::vector<long> cnt(N_CLASSES, 0);
  std::vector<double> A(static_cast<size_t>(F) * F, 0.0), B(static_cast<size_t>(F) * C, 0.0);
  for (int s = 0; s < tr.N; ++s) { w.encode(&tr.X[static_cast<size_t>(s) * D], c); const int y = tr.y[static_cast<size_t>(s)]; for (int j = 0; j < W; ++j) cent[static_cast<size_t>(y)][static_cast<size_t>(j)] += c[static_cast<size_t>(j)]; cnt[static_cast<size_t>(y)]++; for (int p = 0; p < F; ++p) { const double xp = (p < W) ? c[static_cast<size_t>(p)] : 1.0; if (xp == 0.0) continue; for (int q = 0; q < F; ++q) { const double xq = (q < W) ? c[static_cast<size_t>(q)] : 1.0; A[static_cast<size_t>(p) * F + q] += xp * xq; } B[static_cast<size_t>(p) * C + y] += xp; } }
  for (int k = 0; k < C; ++k) { const long n = std::max(1L, cnt[static_cast<size_t>(k)]); for (int j = 0; j < W; ++j) cent[static_cast<size_t>(k)][static_cast<size_t>(j)] /= static_cast<float>(n); }
  std::fprintf(stderr, "    [wide lr=%.3f] solving %d-dim...\n", lr, W);
  solve_ridge(A, B, F, C, 1e-3);
  long no = 0, lo = 0;
  for (int s = 0; s < te.N; ++s) { w.encode(&te.X[static_cast<size_t>(s) * D], c); const int y = te.y[static_cast<size_t>(s)]; if (ncc_predict(c.data(), W, cent) == y) ++no; if (lin_predict(c.data(), W, B, F, C) == y) ++lo; }
  ncc = static_cast<float>(static_cast<double>(no) / te.N); lin = static_cast<float>(static_cast<double>(lo) / te.N);
}

void print_curve(const char* name, const Curve& cv) {
  const int L = static_cast<int>(cv.ncc.size()) - 1;
  std::printf("  %s\n   depth: ", name);
  const int ds[] = {0, 1, 2, 3, 5, 8, 12, 18, 25};
  std::printf("ncc  "); for (int d : ds) std::printf("d%-2d %.3f  ", d, cv.ncc[static_cast<size_t>(d)]); std::printf("\n          lin  ");
  for (int d : ds) std::printf("d%-2d %.3f  ", d, cv.lin[static_cast<size_t>(d)]); std::printf("\n          dead@25 %.2f  entropy@25 %.2f\n", cv.dead[static_cast<size_t>(L)], cv.entropy[static_cast<size_t>(L)]);
}
float bestNCC(const Curve& cv) { float b = -1; for (size_t i = 1; i < cv.ncc.size(); ++i) b = std::max(b, cv.ncc[i]); return b; }
float bestLIN(const Curve& cv) { float b = -1; for (size_t i = 1; i < cv.lin.size(); ++i) b = std::max(b, cv.lin[i]); return b; }

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::vector<float> parts = make_parts(101);
  const Dataset tr = gen(12000, parts, 7, 0.10f), te = gen(3000, parts, 8, 0.10f);
  const std::vector<int> kT = taper_k(N_LAYERS);

  std::printf("=====================================================================\n");
  std::printf(" RESCUE: can DEPTH compose forward-only? (LR-matched, bottleneck fixes)\n");
  std::printf("=====================================================================\n");
  std::printf(" task XOR-of-groups, chance 0.250, single-layer ref ~0.387 ncc / ~0.478 lin\n\n");

  std::fprintf(stderr, "[A code-only lr=0.10]\n");
  Layers a = build_stack(CODE_ONLY, N_LAYERS, kT, 0.10f, 1000); train(a, CODE_ONLY, 0, tr, 6, 11);
  const Curve ca = eval(a, CODE_ONLY, 0, tr, te);

  std::fprintf(stderr, "[B skip-concat lr=0.10]\n");
  Layers b = build_stack(SKIP, N_LAYERS, kT, 0.10f, 2000); train(b, SKIP, 0, tr, 6, 11);
  const Curve cb = eval(b, SKIP, 0, tr, te);
  const float bMem = memory_linear(b, SKIP, 0, tr, te);

  std::fprintf(stderr, "[C soft-temp tau=0.4 lr=0.10]\n");
  Layers c = build_stack(SOFT, N_LAYERS, kT, 0.10f, 3000); train(c, SOFT, 0.4f, tr, 6, 11);
  const Curve cc = eval(c, SOFT, 0.4f, tr, te);

  std::fprintf(stderr, "[wide LR sweep]\n");
  float w05n, w05l, w07n, w07l, w10n, w10l;
  wide_probe(tr, te, 0.05f, 6000, w05n, w05l);
  wide_probe(tr, te, 0.074f, 6001, w07n, w07l);
  wide_probe(tr, te, 0.10f, 6002, w10n, w10l);

  std::printf("PER-DEPTH CURVES (ncc + lin at sampled depths):\n");
  print_curve("A) code-only (LR-matched baseline)", ca);
  print_curve("B) SKIP-CONCAT (each layer sees raw + prev code = Oxta-mem of input)", cb);
  print_curve("C) SOFT-TEMP dense code (tau=0.4)", cc);

  std::printf("\nWIDTH-SCALED single layer m=1600 (LR sweep, ncc/lin):\n");
  std::printf("  lr=0.050: %.3f / %.3f    lr=0.074: %.3f / %.3f    lr=0.100: %.3f / %.3f\n",
              w05n, w05l, w07n, w07l, w10n, w10l);

  std::printf("\nSUMMARY (NCC best-over-depth / LIN best-over-depth):\n");
  std::printf("  A code-only ... best ncc %.3f  lin %.3f  | single(d1) ncc %.3f lin %.3f\n", bestNCC(ca), bestLIN(ca), ca.ncc[1], ca.lin[1]);
  std::printf("  B skip-concat . best ncc %.3f  lin %.3f  | d1 ncc %.3f lin %.3f | MEM-concat lin %.3f\n", bestNCC(cb), bestLIN(cb), cb.ncc[1], cb.lin[1], bMem);
  std::printf("  C soft-temp ... best ncc %.3f  lin %.3f  | d1 ncc %.3f lin %.3f\n", bestNCC(cc), bestLIN(cc), cc.ncc[1], cc.lin[1]);
  std::printf("  wide(matched lr=0.074) ncc %.3f lin %.3f\n\n", w07n, w07l);

  // verdicts derived from numbers
  auto depthHelps = [](const Curve& cv) { return (bestNCC(cv) - cv.ncc[1]) >= 0.05f; };
  std::printf("DOES DEPTH HELP (best-depth beats its own layer-1 by >=0.05 ncc)?\n");
  std::printf("  A code-only ... %s (+%.3f)\n", depthHelps(ca) ? "YES" : "no ", bestNCC(ca) - ca.ncc[1]);
  std::printf("  B skip-concat . %s (+%.3f)   [the decisive test]\n", depthHelps(cb) ? "YES" : "no ", bestNCC(cb) - cb.ncc[1]);
  std::printf("  C soft-temp ... %s (+%.3f)\n", depthHelps(cc) ? "YES" : "no ", bestNCC(cc) - cc.ncc[1]);
  std::printf("\n  width confound check: wide lr=0.074 lin %.3f vs lr=0.10 lin %.3f (deep best lin %.3f)\n",
              w07l, w10l, std::max({bestLIN(ca), bestLIN(cb), bestLIN(cc)}));
  std::printf("=====================================================================\n");
  return 0;
}
