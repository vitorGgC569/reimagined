// ============================================================================
// WIDTH METRIC — how much width does forward-only cost to MATCH backprop?
// ============================================================================
// We sweep per-layer width for backprop (Adam MLP) vs Forward-Forward (local,
// no backprop), at fixed depth, on two tasks:
//   B) parity-8 (single hard task) -> the "iso-accuracy width ratio": the FF
//      width needed to match a given backprop width's accuracy. A concrete
//      exchange rate for going forward-only.
//   A) continual A->B -> retention vs width: does MORE width fix backprop's
//      catastrophic forgetting, or is it a mechanism gap width cannot close?
// Params are counted so the cost is reported in width AND parameters.
//
// Build: cl /nologo /O2 /EHsc /std:c++17 width_metric.cpp /Fe:width_metric.exe
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

void l2norm(float* v, int d) { float n = 0; for (int i = 0; i < d; ++i) n += v[i] * v[i]; n = std::sqrt(n) + 1e-8f; for (int i = 0; i < d; ++i) v[i] /= n; }

struct Data { std::vector<float> X; std::vector<int> y; int N = 0, d = 0, C = 0; };

Data gen_parity(int n, float noise, uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::bernoulli_distribution coin(0.5); std::normal_distribution<float> nd(0.0f, noise);
  Data d; d.N = n; d.d = 8; d.C = 2; d.X.resize(static_cast<size_t>(n) * 8); d.y.resize(static_cast<size_t>(n));
  for (int s = 0; s < n; ++s) { int par = 0; float* x = &d.X[static_cast<size_t>(s) * 8]; for (int i = 0; i < 8; ++i) { const int b = coin(rng) ? 1 : 0; par ^= b; x[i] = (b ? 1.0f : -1.0f) + nd(rng); } d.y[static_cast<size_t>(s)] = par; }
  return d;
}
Data make_clusters(int per, int K, int dim, float noise, uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<float>> C(static_cast<size_t>(K), std::vector<float>(static_cast<size_t>(dim)));
  for (int c = 0; c < K; ++c) { for (int i = 0; i < dim; ++i) C[static_cast<size_t>(c)][static_cast<size_t>(i)] = nd(rng); l2norm(C[static_cast<size_t>(c)].data(), dim); }
  Data d; d.N = K * per; d.d = dim; d.C = K; d.X.resize(static_cast<size_t>(d.N) * dim); d.y.resize(static_cast<size_t>(d.N)); int idx = 0;
  for (int c = 0; c < K; ++c) for (int p = 0; p < per; ++p) { float* r = &d.X[static_cast<size_t>(idx) * dim]; for (int i = 0; i < dim; ++i) r[i] = C[static_cast<size_t>(c)][static_cast<size_t>(i)] + noise * nd(rng); l2norm(r, dim); d.y[static_cast<size_t>(idx)] = c; ++idx; }
  return d;
}

// ---- backprop MLP (Adam), arbitrary sizes ----
struct MLP {
  std::vector<int> sz; int L; float lr; std::vector<std::vector<float>> W, b, mW, mb, vW, vb, a, z; long t = 0; const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  MLP(const std::vector<int>& s, float lr_, uint64_t seed) : sz(s), L(static_cast<int>(s.size()) - 1), lr(lr_), W(L), b(L), mW(L), mb(L), vW(L), vb(L), a(static_cast<size_t>(L) + 1), z(L) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    for (int l = 0; l < L; ++l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / fi)); W[static_cast<size_t>(l)].resize(static_cast<size_t>(fo) * fi); for (auto& w : W[static_cast<size_t>(l)]) w = nd(rng); b[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); mW[static_cast<size_t>(l)].assign(static_cast<size_t>(fo) * fi, 0.0f); mb[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); vW[static_cast<size_t>(l)].assign(static_cast<size_t>(fo) * fi, 0.0f); vb[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); }
  }
  const std::vector<float>& forward(const float* x) {
    a[0].assign(x, x + sz[0]);
    for (int l = 0; l < L; ++l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; z[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); a[static_cast<size_t>(l) + 1].assign(static_cast<size_t>(fo), 0.0f);
      for (int o = 0; o < fo; ++o) { const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float s = b[static_cast<size_t>(l)][static_cast<size_t>(o)]; for (int i = 0; i < fi; ++i) s += Wo[i] * a[static_cast<size_t>(l)][static_cast<size_t>(i)]; z[static_cast<size_t>(l)][static_cast<size_t>(o)] = s; a[static_cast<size_t>(l) + 1][static_cast<size_t>(o)] = (l < L - 1) ? (s > 0 ? s : 0.01f * s) : s; } }  // leaky ReLU (anti dead-unit collapse)
    auto& o = a[static_cast<size_t>(L)]; float mx = -1e30f; for (float v : o) mx = std::max(mx, v); float Z = 0; for (float& v : o) { v = std::exp(v - mx); Z += v; } for (float& v : o) v /= Z; return o;
  }
  void step(const float* x, int y) {
    forward(x); ++t; const float c1 = 1 - std::pow(b1, static_cast<float>(t)), c2 = 1 - std::pow(b2, static_cast<float>(t));
    std::vector<float> delta = a[static_cast<size_t>(L)]; delta[static_cast<size_t>(y)] -= 1.0f;
    for (int l = L - 1; l >= 0; --l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; std::vector<float> nd2;
      if (l > 0) { nd2.assign(static_cast<size_t>(fi), 0.0f); for (int o = 0; o < fo; ++o) { const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; const float dl = delta[static_cast<size_t>(o)]; for (int i = 0; i < fi; ++i) nd2[static_cast<size_t>(i)] += Wo[i] * dl; } for (int i = 0; i < fi; ++i) nd2[static_cast<size_t>(i)] *= (z[static_cast<size_t>(l) - 1][static_cast<size_t>(i)] > 0 ? 1.0f : 0.01f); }
      for (int o = 0; o < fo; ++o) { const float dl = delta[static_cast<size_t>(o)]; float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float* mo = &mW[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float* vo = &vW[static_cast<size_t>(l)][static_cast<size_t>(o) * fi];
        for (int i = 0; i < fi; ++i) { const float g = dl * a[static_cast<size_t>(l)][static_cast<size_t>(i)]; mo[i] = b1 * mo[i] + (1 - b1) * g; vo[i] = b2 * vo[i] + (1 - b2) * g * g; Wo[i] -= lr * (mo[i] / c1) / (std::sqrt(vo[i] / c2) + eps); }
        float& mB = mb[static_cast<size_t>(l)][static_cast<size_t>(o)]; float& vB = vb[static_cast<size_t>(l)][static_cast<size_t>(o)]; mB = b1 * mB + (1 - b1) * dl; vB = b2 * vB + (1 - b2) * dl * dl; b[static_cast<size_t>(l)][static_cast<size_t>(o)] -= lr * (mB / c1) / (std::sqrt(vB / c2) + eps); }
      if (l > 0) delta.swap(nd2); }
  }
  void train(const Data& d, int ep, uint64_t seed) { std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); for (int e = 0; e < ep; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) step(&d.X[static_cast<size_t>(s) * d.d], d.y[static_cast<size_t>(s)]); } }
  float acc(const Data& d) { int ok = 0; for (int s = 0; s < d.N; ++s) { const auto& o = forward(&d.X[static_cast<size_t>(s) * d.d]); int bi = 0; float bv = -1e30f; for (int c = 0; c < d.C; ++c) if (o[static_cast<size_t>(c)] > bv) { bv = o[static_cast<size_t>(c)]; bi = c; } if (bi == d.y[static_cast<size_t>(s)]) ++ok; } return static_cast<float>(static_cast<double>(ok) / d.N); }
};
long mlp_params(int D, int w, int depth, int C) { long p = static_cast<long>(w) * (D + 1) + static_cast<long>(C) * (w + 1); for (int i = 1; i < depth; ++i) p += static_cast<long>(w) * (w + 1); return p; }

// ---- Forward-Forward with dual-trace memory (generic) ----
struct FFLayer {
  int d_in, m; float lr, theta = 0.0f, tr = 0.01f, gamma, lambda; bool dual; std::vector<float> W, Ws, b;
  FFLayer(int d_in_, int m_, float lr_, float g, float lam, bool dual_, uint64_t seed) : d_in(d_in_), m(m_), lr(lr_), gamma(g), lambda(lam), dual(dual_), W(static_cast<size_t>(m_) * d_in_), Ws(static_cast<size_t>(m_) * d_in_), b(static_cast<size_t>(m_), 0.0f) {
    std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt(static_cast<float>(d_in_))); for (auto& w : W) w = nd(rng); Ws = W;
  }
  float act(const float* x, std::vector<float>& h, std::vector<float>& xhat) const {
    float nn = 0; for (int i = 0; i < d_in; ++i) nn += x[i] * x[i]; nn = std::sqrt(nn) + 1e-8f; xhat.resize(static_cast<size_t>(d_in)); for (int i = 0; i < d_in; ++i) xhat[static_cast<size_t>(i)] = x[i] / nn; h.assign(static_cast<size_t>(m), 0.0f);
    for (int j = 0; j < m; ++j) { const float* Wj = &W[static_cast<size_t>(j) * d_in]; float s = b[static_cast<size_t>(j)]; for (int i = 0; i < d_in; ++i) s += Wj[i] * xhat[static_cast<size_t>(i)]; h[static_cast<size_t>(j)] = s > 0 ? s : 0.0f; }
    float G = 0; for (int j = 0; j < m; ++j) G += h[static_cast<size_t>(j)] * h[static_cast<size_t>(j)]; return G;
  }
  float fwd(const float* x, std::vector<float>& h) const { std::vector<float> xh; return act(x, h, xh); }
  void step(const float* x, bool pos, std::vector<float>& h) {
    std::vector<float> xhat; const float G = act(x, h, xhat); const float p = 1.0f / (1.0f + std::exp(-(G - theta))); theta = (1 - tr) * theta + tr * G; const float coef = pos ? -(1.0f - p) : p;
    for (int j = 0; j < m; ++j) { if (h[static_cast<size_t>(j)] <= 0) continue; const float g = lr * coef * 2.0f * h[static_cast<size_t>(j)]; float* Wj = &W[static_cast<size_t>(j) * d_in]; for (int i = 0; i < d_in; ++i) Wj[i] -= g * xhat[static_cast<size_t>(i)]; b[static_cast<size_t>(j)] -= g;
      if (dual) { float* Sj = &Ws[static_cast<size_t>(j) * d_in]; for (int i = 0; i < d_in; ++i) { Sj[i] = (1 - gamma) * Sj[i] + gamma * Wj[i]; Wj[i] += lambda * (Sj[i] - Wj[i]); } } }
  }
};
struct FFNet {
  std::vector<FFLayer> L; int D, C, depth; float alpha;
  FFNet(int D_, int C_, int depth_, int width, float lr, float alpha_, float g, float lam, bool dual, uint64_t seed) : D(D_), C(C_), depth(depth_), alpha(alpha_) {
    for (int i = 0; i < depth_; ++i) L.emplace_back((i == 0) ? D_ + C_ : width, width, lr, g, lam, dual, seed + static_cast<uint64_t>(i));
  }
  void embed(const float* x, int lab, std::vector<float>& o) const { o.resize(static_cast<size_t>(D) + C); for (int i = 0; i < D; ++i) o[static_cast<size_t>(i)] = x[i]; for (int c = 0; c < C; ++c) o[static_cast<size_t>(D) + c] = (c == lab) ? alpha : 0.0f; }
  void train(const Data& d, int ep, uint64_t seed) {
    std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); std::uniform_int_distribution<int> roll(1, C - 1); std::vector<float> ip, in, cur, h;
    for (int e = 0; e < ep; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) { const float* x = &d.X[static_cast<size_t>(s) * d.d]; const int y = d.y[static_cast<size_t>(s)]; const int yn = (y + roll(rng)) % C; embed(x, y, ip); embed(x, yn, in); cur = ip; for (auto& l : L) { l.step(cur.data(), true, h); cur = h; } cur = in; for (auto& l : L) { l.step(cur.data(), false, h); cur = h; } } }
  }
  float acc(const Data& d) {
    int ok = 0; std::vector<float> in, cur, h;
    for (int s = 0; s < d.N; ++s) { const float* x = &d.X[static_cast<size_t>(s) * d.d]; const int y = d.y[static_cast<size_t>(s)]; int best = 0; float bv = -1e30f;
      for (int c = 0; c < C; ++c) { embed(x, c, in); cur = in; float G = 0; for (int li = 0; li < depth; ++li) { const float g = L[static_cast<size_t>(li)].fwd(cur.data(), h) - L[static_cast<size_t>(li)].theta; if (li >= 1) G += g; cur = h; } if (G > bv) { bv = G; best = c; } } if (best == y) ++ok; }
    return static_cast<float>(static_cast<double>(ok) / d.N);
  }
};
long ff_params(int D, int w, int depth, int C) { long p = static_cast<long>(w) * (D + C + 1); for (int i = 1; i < depth; ++i) p += static_cast<long>(w) * (w + 1); return p; }

float med3(float a, float b, float c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const int DEPTH = 3;
  std::printf("=====================================================================\n");
  std::printf(" WIDTH METRIC — how much width does forward-only cost vs backprop?\n");
  std::printf("=====================================================================\n");

  // ---------------- PART B: parity-8, iso-accuracy width ratio ----------------
  const Data ptr = gen_parity(12000, 0.30f, 7), pte = gen_parity(3000, 0.30f, 8);
  const int widths[] = {8, 16, 24, 32, 48, 64};
  const int NW = 6;
  std::vector<float> bpA(NW), ffA(NW); std::vector<long> bpP(NW), ffP(NW);
  std::printf("\n[B] PARITY-8 (depth %d, MEDIAN of 3 seeds, leaky-ReLU backprop). chance 0.500\n", DEPTH);
  std::printf("  width | backprop acc (params) | fwd-fwd acc (params)\n");
  for (int wi = 0; wi < NW; ++wi) {
    const int w = widths[wi];
    std::vector<int> mlp_sz = {8, w, w, w, 2};
    float bs[3], fs[3];
    for (int s = 0; s < 3; ++s) { MLP m(mlp_sz, 0.002f, 100 + static_cast<uint64_t>(s)); m.train(ptr, 90, 11 + static_cast<uint64_t>(s)); bs[s] = m.acc(pte); }
    for (int s = 0; s < 3; ++s) { FFNet ff(8, 2, DEPTH, w, 0.02f, 1.0f, 0, 0, false, 200 + static_cast<uint64_t>(s)); ff.train(ptr, 80, 11 + static_cast<uint64_t>(s)); fs[s] = ff.acc(pte); }
    bpA[static_cast<size_t>(wi)] = med3(bs[0], bs[1], bs[2]); ffA[static_cast<size_t>(wi)] = med3(fs[0], fs[1], fs[2]);
    bpP[static_cast<size_t>(wi)] = mlp_params(8, w, DEPTH, 2); ffP[static_cast<size_t>(wi)] = ff_params(8, w, DEPTH, 2);
    std::printf("  %4d  |   %.3f  (%5ld)      |  %.3f  (%5ld)\n", w, bpA[static_cast<size_t>(wi)], bpP[static_cast<size_t>(wi)], ffA[static_cast<size_t>(wi)], ffP[static_cast<size_t>(wi)]);
    std::fprintf(stderr, "[B width %d done]\n", w);
  }
  // iso-accuracy: for each backprop width, the FF width needed to match its acc
  std::printf("\n  ISO-ACCURACY (FF width needed to match backprop@width):\n");
  auto ff_width_for = [&](float target) -> float {
    for (int wi = 0; wi < NW; ++wi) if (ffA[static_cast<size_t>(wi)] >= target) {
      if (wi == 0) return static_cast<float>(widths[0]);
      const float a0 = ffA[static_cast<size_t>(wi) - 1], a1 = ffA[static_cast<size_t>(wi)]; const float w0 = widths[wi - 1], w1 = widths[wi];
      const float frac = (a1 > a0) ? (target - a0) / (a1 - a0) : 0.0f; return w0 + frac * (w1 - w0);
    }
    return -1.0f;  // FF never reaches it within the sweep
  };
  for (int wi = 1; wi < NW; ++wi) {
    const float fw = ff_width_for(bpA[static_cast<size_t>(wi)]);
    if (fw < 0) std::printf("    backprop@%2d (%.3f): FF cannot match within width<=64\n", widths[wi], bpA[static_cast<size_t>(wi)]);
    else std::printf("    backprop@%2d (%.3f): FF needs width ~%.1f  => %.2fx width, ~%.2fx params\n",
                     widths[wi], bpA[static_cast<size_t>(wi)], fw, fw / widths[wi], (static_cast<double>(ff_params(8, static_cast<int>(std::lround(fw)), DEPTH, 2))) / bpP[static_cast<size_t>(wi)]);
  }

  // ---------------- PART A: continual, retention vs width ----------------
  const Data A = make_clusters(300, 4, 16, 0.22f, 101), B = make_clusters(300, 4, 16, 0.22f, 202);
  const int cw[] = {16, 32, 64, 128}; const int NCW = 4;
  std::printf("\n[A] CONTINUAL A->B (depth %d, dim 16, 4 classes). retain A AFTER B; chance 0.250\n", DEPTH);
  std::printf("  width | backprop retain | FF retain | FF+mem retain\n");
  for (int wi = 0; wi < NCW; ++wi) {
    const int w = cw[wi];
    float bp[3], ff[3], fm[3];
    for (int s = 0; s < 3; ++s) {
      MLP m({16, w, w, w, 4}, 0.003f, 100 + static_cast<uint64_t>(s)); m.train(A, 30, 11 + static_cast<uint64_t>(s)); m.train(B, 30, 22 + static_cast<uint64_t>(s)); bp[s] = m.acc(A);
      FFNet f0(16, 4, DEPTH, w, 0.02f, 1.0f, 0, 0, false, 200 + static_cast<uint64_t>(s)); f0.train(A, 30, 11 + static_cast<uint64_t>(s)); f0.train(B, 30, 22 + static_cast<uint64_t>(s)); ff[s] = f0.acc(A);
      FFNet f1(16, 4, DEPTH, w, 0.02f, 1.0f, 0.015f, 0.20f, true, 200 + static_cast<uint64_t>(s)); f1.train(A, 30, 11 + static_cast<uint64_t>(s)); f1.train(B, 30, 22 + static_cast<uint64_t>(s)); fm[s] = f1.acc(A);
    }
    std::printf("  %4d  |     %.3f       |   %.3f   |    %.3f\n", w, med3(bp[0], bp[1], bp[2]), med3(ff[0], ff[1], ff[2]), med3(fm[0], fm[1], fm[2]));
    std::fprintf(stderr, "[A width %d done]\n", w);
  }
  std::printf("\n read [B]: FF's width tax to match backprop. read [A]: does width fix backprop's forgetting?\n");
  std::printf("=====================================================================\n");
  return 0;
}
