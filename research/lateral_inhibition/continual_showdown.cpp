// ============================================================================
// CONTINUAL SHOWDOWN — the COMBINED learner (FF + dual-trace) vs backprop
// ============================================================================
// Unites the two validated pieces into ONE forward-only learner: Forward-Forward
// layers (local learning signal) whose weights carry a dual-trace memory (fast
// plastic + slow consolidated + elastic anchor). Tested on CONTINUAL learning:
// learn task A, then task B (same net), measure how much of A survives.
// Competitors: plain backprop MLP (gold standard, but forgets), and FF without
// the dual-trace memory (to isolate the memory's contribution).
//
// Build: cl /nologo /O2 /EHsc /std:c++17 continual_showdown.cpp /Fe:continual_showdown.exe
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int DIM = 16, K = 4, M = 64, DEPTH = 3;
constexpr int PER = 300;

void l2norm(float* v, int d) { float n = 0; for (int i = 0; i < d; ++i) n += v[i] * v[i]; n = std::sqrt(n) + 1e-8f; for (int i = 0; i < d; ++i) v[i] /= n; }

struct Data { std::vector<float> X; std::vector<int> y; int N = 0; };
Data make_task(int per, float noise, uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<float>> C(static_cast<size_t>(K), std::vector<float>(DIM));
  for (int c = 0; c < K; ++c) { for (int i = 0; i < DIM; ++i) C[static_cast<size_t>(c)][static_cast<size_t>(i)] = nd(rng); l2norm(C[static_cast<size_t>(c)].data(), DIM); }
  Data d; d.N = K * per; d.X.resize(static_cast<size_t>(d.N) * DIM); d.y.resize(static_cast<size_t>(d.N)); int idx = 0;
  for (int c = 0; c < K; ++c) for (int p = 0; p < per; ++p) { float* r = &d.X[static_cast<size_t>(idx) * DIM]; for (int i = 0; i < DIM; ++i) r[i] = C[static_cast<size_t>(c)][static_cast<size_t>(i)] + noise * nd(rng); l2norm(r, DIM); d.y[static_cast<size_t>(idx)] = c; ++idx; }
  return d;
}

// ---- backprop MLP (Adam) ----
struct MLP {
  std::vector<int> sz; int L; float lr; std::vector<std::vector<float>> W, b, mW, mb, vW, vb, a, z; long t = 0; const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  MLP(const std::vector<int>& s, float lr_, uint64_t seed) : sz(s), L(static_cast<int>(s.size()) - 1), lr(lr_), W(L), b(L), mW(L), mb(L), vW(L), vb(L), a(static_cast<size_t>(L) + 1), z(L) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    for (int l = 0; l < L; ++l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / fi)); W[static_cast<size_t>(l)].resize(static_cast<size_t>(fo) * fi); for (auto& w : W[static_cast<size_t>(l)]) w = nd(rng); b[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); mW[static_cast<size_t>(l)].assign(static_cast<size_t>(fo) * fi, 0.0f); mb[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); vW[static_cast<size_t>(l)].assign(static_cast<size_t>(fo) * fi, 0.0f); vb[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); }
  }
  const std::vector<float>& forward(const float* x) {
    a[0].assign(x, x + sz[0]);
    for (int l = 0; l < L; ++l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; z[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); a[static_cast<size_t>(l) + 1].assign(static_cast<size_t>(fo), 0.0f);
      for (int o = 0; o < fo; ++o) { const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float s = b[static_cast<size_t>(l)][static_cast<size_t>(o)]; for (int i = 0; i < fi; ++i) s += Wo[i] * a[static_cast<size_t>(l)][static_cast<size_t>(i)]; z[static_cast<size_t>(l)][static_cast<size_t>(o)] = s; a[static_cast<size_t>(l) + 1][static_cast<size_t>(o)] = (l < L - 1) ? (s > 0 ? s : 0.0f) : s; } }
    auto& o = a[static_cast<size_t>(L)]; float mx = -1e30f; for (float v : o) mx = std::max(mx, v); float Z = 0; for (float& v : o) { v = std::exp(v - mx); Z += v; } for (float& v : o) v /= Z; return o;
  }
  void step(const float* x, int y) {
    forward(x); ++t; const float c1 = 1 - std::pow(b1, static_cast<float>(t)), c2 = 1 - std::pow(b2, static_cast<float>(t));
    std::vector<float> delta = a[static_cast<size_t>(L)]; delta[static_cast<size_t>(y)] -= 1.0f;
    for (int l = L - 1; l >= 0; --l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; std::vector<float> nd2;
      if (l > 0) { nd2.assign(static_cast<size_t>(fi), 0.0f); for (int o = 0; o < fo; ++o) { const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; const float dl = delta[static_cast<size_t>(o)]; for (int i = 0; i < fi; ++i) nd2[static_cast<size_t>(i)] += Wo[i] * dl; } for (int i = 0; i < fi; ++i) nd2[static_cast<size_t>(i)] *= (z[static_cast<size_t>(l) - 1][static_cast<size_t>(i)] > 0 ? 1.0f : 0.0f); }
      for (int o = 0; o < fo; ++o) { const float dl = delta[static_cast<size_t>(o)]; float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float* mo = &mW[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float* vo = &vW[static_cast<size_t>(l)][static_cast<size_t>(o) * fi];
        for (int i = 0; i < fi; ++i) { const float g = dl * a[static_cast<size_t>(l)][static_cast<size_t>(i)]; mo[i] = b1 * mo[i] + (1 - b1) * g; vo[i] = b2 * vo[i] + (1 - b2) * g * g; Wo[i] -= lr * (mo[i] / c1) / (std::sqrt(vo[i] / c2) + eps); }
        float& mB = mb[static_cast<size_t>(l)][static_cast<size_t>(o)]; float& vB = vb[static_cast<size_t>(l)][static_cast<size_t>(o)]; mB = b1 * mB + (1 - b1) * dl; vB = b2 * vB + (1 - b2) * dl * dl; b[static_cast<size_t>(l)][static_cast<size_t>(o)] -= lr * (mB / c1) / (std::sqrt(vB / c2) + eps); }
      if (l > 0) delta.swap(nd2); }
  }
  void train(const Data& d, int ep, uint64_t seed) { std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); for (int e = 0; e < ep; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) step(&d.X[static_cast<size_t>(s) * DIM], d.y[static_cast<size_t>(s)]); } }
  float acc(const Data& d) { int ok = 0; for (int s = 0; s < d.N; ++s) { const auto& o = forward(&d.X[static_cast<size_t>(s) * DIM]); int b2i = 0; float bv = -1e30f; for (int c = 0; c < K; ++c) if (o[static_cast<size_t>(c)] > bv) { bv = o[static_cast<size_t>(c)]; b2i = c; } if (b2i == d.y[static_cast<size_t>(s)]) ++ok; } return static_cast<float>(static_cast<double>(ok) / d.N); }
};

// ---- Forward-Forward layer WITH dual-trace memory ----
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
    for (int j = 0; j < m; ++j) {
      if (h[static_cast<size_t>(j)] <= 0) continue; const float g = lr * coef * 2.0f * h[static_cast<size_t>(j)]; float* Wj = &W[static_cast<size_t>(j) * d_in]; for (int i = 0; i < d_in; ++i) Wj[i] -= g * xhat[static_cast<size_t>(i)]; b[static_cast<size_t>(j)] -= g;
      if (dual) { float* Sj = &Ws[static_cast<size_t>(j) * d_in]; for (int i = 0; i < d_in; ++i) { Sj[i] = (1 - gamma) * Sj[i] + gamma * Wj[i]; Wj[i] += lambda * (Sj[i] - Wj[i]); } }
    }
  }
};
struct FFNet {
  std::vector<FFLayer> L; float alpha;
  FFNet(float lr, float alpha_, float g, float lam, bool dual, uint64_t seed) : alpha(alpha_) { for (int i = 0; i < DEPTH; ++i) L.emplace_back((i == 0) ? DIM + K : M, M, lr, g, lam, dual, seed + static_cast<uint64_t>(i)); }
  void embed(const float* x, int lab, std::vector<float>& o) const { o.resize(static_cast<size_t>(DIM) + K); for (int i = 0; i < DIM; ++i) o[static_cast<size_t>(i)] = x[i]; for (int c = 0; c < K; ++c) o[static_cast<size_t>(DIM) + c] = (c == lab) ? alpha : 0.0f; }
  void train(const Data& d, int ep, uint64_t seed) {
    std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); std::uniform_int_distribution<int> roll(1, K - 1); std::vector<float> ip, in, cur, h;
    for (int e = 0; e < ep; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) { const float* x = &d.X[static_cast<size_t>(s) * DIM]; const int y = d.y[static_cast<size_t>(s)]; const int yn = (y + roll(rng)) % K; embed(x, y, ip); embed(x, yn, in); cur = ip; for (auto& l : L) { l.step(cur.data(), true, h); cur = h; } cur = in; for (auto& l : L) { l.step(cur.data(), false, h); cur = h; } } }
  }
  float acc(const Data& d) {
    int ok = 0; std::vector<float> in, cur, h;
    for (int s = 0; s < d.N; ++s) { const float* x = &d.X[static_cast<size_t>(s) * DIM]; const int y = d.y[static_cast<size_t>(s)]; int best = 0; float bv = -1e30f;
      for (int c = 0; c < K; ++c) { embed(x, c, in); cur = in; float G = 0; for (int li = 0; li < DEPTH; ++li) { const float g = L[static_cast<size_t>(li)].fwd(cur.data(), h) - L[static_cast<size_t>(li)].theta; if (li >= 1) G += g; cur = h; } if (G > bv) { bv = G; best = c; } } if (best == y) ++ok; }
    return static_cast<float>(static_cast<double>(ok) / d.N);
  }
};

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const Data A = make_task(PER, 0.22f, 101), B = make_task(PER, 0.22f, 202);
  std::printf("=====================================================================\n");
  std::printf(" CONTINUAL SHOWDOWN — combined FF+dual-trace vs backprop\n");
  std::printf("=====================================================================\n");
  std::printf(" %d classes, task A and task B (disjoint clusters), dim %d, %d-layer nets\n", K, DIM, DEPTH);
  std::printf(" protocol: learn A -> learn B -> measure. chance %.3f\n\n", 1.0f / K);
  std::printf(" %-28s  A(after A)  B(after B)  A-RETAINED\n", "learner");

  { MLP m({DIM, M, M, K}, 0.003f, 100); m.train(A, 30, 11); const float a1 = m.acc(A); m.train(B, 30, 22); std::printf("  %-28s  %.3f       %.3f       %.3f\n", "backprop MLP (gold std)", a1, m.acc(B), m.acc(A)); }
  { FFNet f(0.02f, 1.0f, 0.0f, 0.0f, false, 200); f.train(A, 30, 11); const float a1 = f.acc(A); f.train(B, 30, 22); std::printf("  %-28s  %.3f       %.3f       %.3f\n", "FF only (no memory)", a1, f.acc(B), f.acc(A)); }
  { FFNet f(0.02f, 1.0f, 0.02f, 0.10f, true, 200); f.train(A, 30, 11); const float a1 = f.acc(A); f.train(B, 30, 22); std::printf("  %-28s  %.3f       %.3f       %.3f\n", "FF + OXTA-MEM (combined)", a1, f.acc(B), f.acc(A)); }
  { FFNet f(0.02f, 1.0f, 0.015f, 0.20f, true, 200); f.train(A, 30, 11); const float a1 = f.acc(A); f.train(B, 30, 22); std::printf("  %-28s  %.3f       %.3f       %.3f\n", "FF + OXTA-MEM (strong)", a1, f.acc(B), f.acc(A)); }

  std::printf("\n A-RETAINED = accuracy on task A AFTER learning B (higher = less forgetting).\n");
  std::printf("=====================================================================\n");
  return 0;
}
