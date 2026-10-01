// ============================================================================
// MNIST SHOWDOWN — real task: backprop vs Forward-Forward (+ dual-trace memory)
// ============================================================================
// Does the forward-only story survive a REAL dataset?
//   Part 1: classification accuracy + WALL-CLOCK training time/epoch (the
//           empirical answer to "is forward training faster?").
//   Part 2: continual (permuted-MNIST): learn MNIST, then a pixel-permuted
//           MNIST, measure retention — does FF+Oxta-mem still beat backprop?
//
// Build: cl /nologo /O2 /EHsc /std:c++17 mnist_showdown.cpp /Fe:mnist_showdown.exe
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int D = 784, C = 10;

uint32_t be32(std::ifstream& f) { unsigned char b[4]; f.read(reinterpret_cast<char*>(b), 4); return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) | (static_cast<uint32_t>(b[2]) << 8) | b[3]; }

struct Data { std::vector<float> X; std::vector<int> y; int N = 0, d = D, C = ::C; };

Data load(const std::string& imgf, const std::string& labf, int limit) {
  std::ifstream fi(imgf, std::ios::binary), fl(labf, std::ios::binary);
  if (!fi || !fl) { std::fprintf(stderr, "FAIL open %s / %s\n", imgf.c_str(), labf.c_str()); return {}; }
  be32(fi); const int n = static_cast<int>(be32(fi)); const int rows = static_cast<int>(be32(fi)), cols = static_cast<int>(be32(fi));
  be32(fl); be32(fl);
  const int N = (limit > 0 && limit < n) ? limit : n; const int pix = rows * cols;
  Data d; d.N = N; d.X.resize(static_cast<size_t>(N) * pix); d.y.resize(static_cast<size_t>(N));
  std::vector<unsigned char> buf(static_cast<size_t>(pix));
  for (int s = 0; s < N; ++s) {
    fi.read(reinterpret_cast<char*>(buf.data()), pix); unsigned char lb; fl.read(reinterpret_cast<char*>(&lb), 1);
    float* x = &d.X[static_cast<size_t>(s) * pix]; float nn = 0;
    for (int i = 0; i < pix; ++i) { x[i] = buf[static_cast<size_t>(i)] / 255.0f; nn += x[i] * x[i]; }
    nn = std::sqrt(nn) + 1e-8f; for (int i = 0; i < pix; ++i) x[i] /= nn;   // unit-L2 images
    d.y[static_cast<size_t>(s)] = lb;
  }
  return d;
}

Data permute(const Data& src, uint64_t seed) {
  std::vector<int> perm(D); std::iota(perm.begin(), perm.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); std::shuffle(perm.begin(), perm.end(), rng);
  Data d = src;
  for (int s = 0; s < src.N; ++s) { const float* x = &src.X[static_cast<size_t>(s) * D]; float* o = &d.X[static_cast<size_t>(s) * D]; for (int i = 0; i < D; ++i) o[static_cast<size_t>(i)] = x[static_cast<size_t>(perm[static_cast<size_t>(i)])]; }
  return d;
}

// ---- backprop MLP (Adam, leaky-ReLU) ----
struct MLP {
  std::vector<int> sz; int L; float lr; std::vector<std::vector<float>> W, b, mW, mb, vW, vb, a, z; long t = 0; const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  MLP(const std::vector<int>& s, float lr_, uint64_t seed) : sz(s), L(static_cast<int>(s.size()) - 1), lr(lr_), W(L), b(L), mW(L), mb(L), vW(L), vb(L), a(static_cast<size_t>(L) + 1), z(L) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    for (int l = 0; l < L; ++l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / fi)); W[static_cast<size_t>(l)].resize(static_cast<size_t>(fo) * fi); for (auto& w : W[static_cast<size_t>(l)]) w = nd(rng); b[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); mW[static_cast<size_t>(l)].assign(static_cast<size_t>(fo) * fi, 0.0f); mb[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); vW[static_cast<size_t>(l)].assign(static_cast<size_t>(fo) * fi, 0.0f); vb[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); }
  }
  const std::vector<float>& forward(const float* x) {
    a[0].assign(x, x + sz[0]);
    for (int l = 0; l < L; ++l) { const int fi = sz[static_cast<size_t>(l)], fo = sz[static_cast<size_t>(l) + 1]; z[static_cast<size_t>(l)].assign(static_cast<size_t>(fo), 0.0f); a[static_cast<size_t>(l) + 1].assign(static_cast<size_t>(fo), 0.0f);
      for (int o = 0; o < fo; ++o) { const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fi]; float s = b[static_cast<size_t>(l)][static_cast<size_t>(o)]; for (int i = 0; i < fi; ++i) s += Wo[i] * a[static_cast<size_t>(l)][static_cast<size_t>(i)]; z[static_cast<size_t>(l)][static_cast<size_t>(o)] = s; a[static_cast<size_t>(l) + 1][static_cast<size_t>(o)] = (l < L - 1) ? (s > 0 ? s : 0.01f * s) : s; } }
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
  void train(const Data& d, int ep, uint64_t seed) { std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); for (int e = 0; e < ep; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) step(&d.X[static_cast<size_t>(s) * D], d.y[static_cast<size_t>(s)]); } }
  float acc(const Data& d) { int ok = 0; for (int s = 0; s < d.N; ++s) { const auto& o = forward(&d.X[static_cast<size_t>(s) * D]); int bi = 0; float bv = -1e30f; for (int c = 0; c < C; ++c) if (o[static_cast<size_t>(c)] > bv) { bv = o[static_cast<size_t>(c)]; bi = c; } if (bi == d.y[static_cast<size_t>(s)]) ++ok; } return static_cast<float>(static_cast<double>(ok) / d.N); }
};

// ---- Forward-Forward with dual-trace memory ----
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
  std::vector<FFLayer> L; int depth; float alpha;
  FFNet(int depth_, int width, float lr, float alpha_, float g, float lam, bool dual, uint64_t seed) : depth(depth_), alpha(alpha_) { for (int i = 0; i < depth_; ++i) L.emplace_back((i == 0) ? D + C : width, width, lr, g, lam, dual, seed + static_cast<uint64_t>(i)); }
  void embed(const float* x, int lab, std::vector<float>& o) const { o.resize(static_cast<size_t>(D) + C); for (int i = 0; i < D; ++i) o[static_cast<size_t>(i)] = x[i]; for (int c = 0; c < C; ++c) o[static_cast<size_t>(D) + c] = (c == lab) ? alpha : 0.0f; }
  void train(const Data& d, int ep, uint64_t seed) {
    std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); std::uniform_int_distribution<int> roll(1, C - 1); std::vector<float> ip, in, cur, h;
    for (int e = 0; e < ep; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) { const float* x = &d.X[static_cast<size_t>(s) * D]; const int y = d.y[static_cast<size_t>(s)]; const int yn = (y + roll(rng)) % C; embed(x, y, ip); embed(x, yn, in); cur = ip; for (auto& l : L) { l.step(cur.data(), true, h); cur = h; } cur = in; for (auto& l : L) { l.step(cur.data(), false, h); cur = h; } } }
  }
  float acc(const Data& d) {
    int ok = 0; std::vector<float> in, cur, h;
    for (int s = 0; s < d.N; ++s) { const float* x = &d.X[static_cast<size_t>(s) * D]; const int y = d.y[static_cast<size_t>(s)]; int best = 0; float bv = -1e30f;
      for (int c = 0; c < C; ++c) { embed(x, c, in); cur = in; float G = 0; for (int li = 0; li < depth; ++li) { const float g = L[static_cast<size_t>(li)].fwd(cur.data(), h) - L[static_cast<size_t>(li)].theta; if (li >= 1) G += g; cur = h; } if (G > bv) { bv = G; best = c; } } if (best == y) ++ok; }
    return static_cast<float>(static_cast<double>(ok) / d.N);
  }
};

double secs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string dir = "mnist/";
  const int N_TRAIN = 8000;
  Data tr = load(dir + "train-images-idx3-ubyte", dir + "train-labels-idx1-ubyte", N_TRAIN);
  Data te = load(dir + "t10k-images-idx3-ubyte", dir + "t10k-labels-idx1-ubyte", 10000);
  if (tr.N == 0 || te.N == 0) { std::printf("could not load MNIST\n"); return 1; }
  std::printf("=====================================================================\n");
  std::printf(" MNIST SHOWDOWN — backprop vs Forward-Forward (real task)\n");
  std::printf("=====================================================================\n");
  std::printf(" %d train / %d test, 784-dim, 10 classes, chance 0.100\n", tr.N, te.N);

  const int WIDTH = 256, DEPTH = 3;
  std::printf(" arch: width %d, depth %d (matched). backprop=Adam, FF=local goodness.\n\n", WIDTH, DEPTH);

  // ---- Continual (permuted MNIST): re-tune Oxta-mem anchor for the real task ----
  Data trP = permute(tr, 777), teP = permute(te, 777);
  std::printf("[2] CONTINUAL (permuted-MNIST): retain original MNIST after learning permuted.\n");
  std::printf("    Oxta-mem anchor sweep (toy lambda=0.20 over-regularized on MNIST).\n");
  const int EP = 12;
  { MLP m({D, WIDTH, WIDTH, C}, 0.001f, 1); m.train(tr, EP, 11); const float a1 = m.acc(te); m.train(trP, EP, 22); std::printf("  backprop              : A %.3f  B %.3f  RETAIN %.3f\n", a1, m.acc(teP), m.acc(te)); }
  { FFNet f(DEPTH, WIDTH, 0.02f, 0.5f, 0, 0, false, 200); f.train(tr, EP, 11); const float a1 = f.acc(te); f.train(trP, EP, 22); std::printf("  FF (no mem)           : A %.3f  B %.3f  RETAIN %.3f\n", a1, f.acc(teP), f.acc(te)); }
  struct Cfg { float g, lam; }; const Cfg cfgs[] = {{0.03f, 0.04f}, {0.05f, 0.08f}, {0.02f, 0.15f}};
  for (const auto& cf : cfgs) { FFNet f(DEPTH, WIDTH, 0.02f, 0.5f, cf.g, cf.lam, true, 200); f.train(tr, EP, 11); const float a1 = f.acc(te); f.train(trP, EP, 22); std::printf("  FF+mem (g=%.2f,l=%.2f)  : A %.3f  B %.3f  RETAIN %.3f\n", cf.g, cf.lam, a1, f.acc(teP), f.acc(te)); }

  std::printf("=====================================================================\n");
  return 0;
}
