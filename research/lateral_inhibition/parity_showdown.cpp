// ============================================================================
// PARITY SHOWDOWN — does depth compose? backprop vs Forward-Forward (local)
// ============================================================================
// Parity-N is the classic depth-requiring task: at NARROW width a single hidden
// layer cannot represent it, but depth can. We pit a real BACKPROP MLP against
// FORWARD-FORWARD (local, no backprop), shallow vs deep, at widths 6 and 16, to
// settle: (1) does depth compose with backprop (the gold standard)? (2) does a
// LOCAL forward-only signal also compose with depth, and how close to backprop?
//
// Build: cl /nologo /O2 /EHsc /std:c++17 parity_showdown.cpp /Fe:parity_showdown.exe
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int N_BITS = 8, NC = 2;

struct Data { std::vector<float> X; std::vector<int> y; int N = 0, d = 0; };

// N bits as +-1 (+ gaussian noise); label = parity (XOR of all bits).
Data gen(int n, float noise, uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::bernoulli_distribution coin(0.5); std::normal_distribution<float> nd(0.0f, noise);
  Data d; d.N = n; d.d = N_BITS; d.X.resize(static_cast<size_t>(n) * N_BITS); d.y.resize(static_cast<size_t>(n));
  for (int s = 0; s < n; ++s) {
    int par = 0; float* x = &d.X[static_cast<size_t>(s) * N_BITS];
    for (int i = 0; i < N_BITS; ++i) { const int bit = coin(rng) ? 1 : 0; par ^= bit; x[i] = (bit ? 1.0f : -1.0f) + nd(rng); }
    d.y[static_cast<size_t>(s)] = par;
  }
  return d;
}

// =================== BACKPROP MLP (ReLU hidden, softmax out, SGD+momentum) ====
struct MLP {
  std::vector<int> sz; int L; float lr;
  std::vector<std::vector<float>> W, b, mW, mb, vW, vb, a, z;  // Adam: m=1st, v=2nd moment
  long t = 0; const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  MLP(const std::vector<int>& sizes, float lr_, float /*unused*/, uint64_t seed)
      : sz(sizes), L(static_cast<int>(sizes.size()) - 1), lr(lr_),
        W(L), b(L), mW(L), mb(L), vW(L), vb(L), a(static_cast<size_t>(L) + 1), z(L) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    for (int l = 0; l < L; ++l) {
      const int fin = sz[static_cast<size_t>(l)], fout = sz[static_cast<size_t>(l) + 1];
      std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / fin));
      W[static_cast<size_t>(l)].resize(static_cast<size_t>(fout) * fin);
      for (auto& w : W[static_cast<size_t>(l)]) w = nd(rng);
      b[static_cast<size_t>(l)].assign(static_cast<size_t>(fout), 0.0f);
      mW[static_cast<size_t>(l)].assign(static_cast<size_t>(fout) * fin, 0.0f);
      mb[static_cast<size_t>(l)].assign(static_cast<size_t>(fout), 0.0f);
      vW[static_cast<size_t>(l)].assign(static_cast<size_t>(fout) * fin, 0.0f);
      vb[static_cast<size_t>(l)].assign(static_cast<size_t>(fout), 0.0f);
    }
  }
  const std::vector<float>& forward(const float* x) {
    a[0].assign(x, x + sz[0]);
    for (int l = 0; l < L; ++l) {
      const int fin = sz[static_cast<size_t>(l)], fout = sz[static_cast<size_t>(l) + 1];
      z[static_cast<size_t>(l)].assign(static_cast<size_t>(fout), 0.0f);
      a[static_cast<size_t>(l) + 1].assign(static_cast<size_t>(fout), 0.0f);
      for (int o = 0; o < fout; ++o) {
        const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fin]; float s = b[static_cast<size_t>(l)][static_cast<size_t>(o)];
        for (int i = 0; i < fin; ++i) s += Wo[i] * a[static_cast<size_t>(l)][static_cast<size_t>(i)];
        z[static_cast<size_t>(l)][static_cast<size_t>(o)] = s;
        a[static_cast<size_t>(l) + 1][static_cast<size_t>(o)] = (l < L - 1) ? (s > 0 ? s : 0.0f) : s;
      }
    }
    // softmax on last
    auto& out = a[static_cast<size_t>(L)]; float mx = -1e30f; for (float v : out) mx = std::max(mx, v);
    float Z = 0; for (float& v : out) { v = std::exp(v - mx); Z += v; } for (float& v : out) v /= Z;
    return out;
  }
  void train_step(const float* x, int y) {
    forward(x);
    ++t; const float bc1 = 1.0f - std::pow(b1, static_cast<float>(t)), bc2 = 1.0f - std::pow(b2, static_cast<float>(t));
    std::vector<float> delta = a[static_cast<size_t>(L)]; delta[static_cast<size_t>(y)] -= 1.0f;  // softmax-CE grad
    for (int l = L - 1; l >= 0; --l) {
      const int fin = sz[static_cast<size_t>(l)], fout = sz[static_cast<size_t>(l) + 1];
      std::vector<float> nd2;
      if (l > 0) {  // propagate delta to a[l] before updating W[l]
        nd2.assign(static_cast<size_t>(fin), 0.0f);
        for (int o = 0; o < fout; ++o) { const float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fin]; const float dl = delta[static_cast<size_t>(o)]; for (int i = 0; i < fin; ++i) nd2[static_cast<size_t>(i)] += Wo[i] * dl; }
        for (int i = 0; i < fin; ++i) nd2[static_cast<size_t>(i)] *= (z[static_cast<size_t>(l) - 1][static_cast<size_t>(i)] > 0 ? 1.0f : 0.0f);
      }
      for (int o = 0; o < fout; ++o) {
        const float dl = delta[static_cast<size_t>(o)];
        float* Wo = &W[static_cast<size_t>(l)][static_cast<size_t>(o) * fin]; float* mo = &mW[static_cast<size_t>(l)][static_cast<size_t>(o) * fin]; float* vo = &vW[static_cast<size_t>(l)][static_cast<size_t>(o) * fin];
        for (int i = 0; i < fin; ++i) {  // Adam
          const float g = dl * a[static_cast<size_t>(l)][static_cast<size_t>(i)];
          mo[i] = b1 * mo[i] + (1 - b1) * g; vo[i] = b2 * vo[i] + (1 - b2) * g * g;
          Wo[i] -= lr * (mo[i] / bc1) / (std::sqrt(vo[i] / bc2) + eps);
        }
        float& mB = mb[static_cast<size_t>(l)][static_cast<size_t>(o)]; float& vB = vb[static_cast<size_t>(l)][static_cast<size_t>(o)];
        mB = b1 * mB + (1 - b1) * dl; vB = b2 * vB + (1 - b2) * dl * dl;
        b[static_cast<size_t>(l)][static_cast<size_t>(o)] -= lr * (mB / bc1) / (std::sqrt(vB / bc2) + eps);
      }
      if (l > 0) delta.swap(nd2);
    }
  }
  void train(const Data& d, int epochs, uint64_t seed) {
    std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed));
    for (int e = 0; e < epochs; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) train_step(&d.X[static_cast<size_t>(s) * d.d], d.y[static_cast<size_t>(s)]); }
  }
  float accuracy(const Data& d) {
    int ok = 0; for (int s = 0; s < d.N; ++s) { const auto& o = forward(&d.X[static_cast<size_t>(s) * d.d]); int b2 = 0; float bv = -1e30f; for (int c = 0; c < NC; ++c) if (o[static_cast<size_t>(c)] > bv) { bv = o[static_cast<size_t>(c)]; b2 = c; } if (b2 == d.y[static_cast<size_t>(s)]) ++ok; }
    return static_cast<float>(static_cast<double>(ok) / d.N);
  }
};

// =================== FORWARD-FORWARD (local, no backprop) =====================
struct FFLayer {
  int d_in, m, k; float lr, theta = 0.0f, tr = 0.01f; std::vector<float> W, b;
  FFLayer(int d_in_, int m_, int k_, float lr_, uint64_t seed) : d_in(d_in_), m(m_), k(std::min(k_, m_)), lr(lr_), W(static_cast<size_t>(m_) * d_in_), b(static_cast<size_t>(m_), 0.0f) {
    std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt(static_cast<float>(d_in_))); for (auto& w : W) w = nd(rng);
  }
  float act(const float* x, std::vector<float>& h, std::vector<float>& xhat) const {
    float nn = 0; for (int i = 0; i < d_in; ++i) nn += x[i] * x[i]; nn = std::sqrt(nn) + 1e-8f;
    xhat.resize(static_cast<size_t>(d_in)); for (int i = 0; i < d_in; ++i) xhat[static_cast<size_t>(i)] = x[i] / nn;
    h.assign(static_cast<size_t>(m), 0.0f);
    for (int j = 0; j < m; ++j) { const float* Wj = &W[static_cast<size_t>(j) * d_in]; float s = b[static_cast<size_t>(j)]; for (int i = 0; i < d_in; ++i) s += Wj[i] * xhat[static_cast<size_t>(i)]; h[static_cast<size_t>(j)] = s > 0 ? s : 0.0f; }
    if (k < m) { std::vector<int> idx(static_cast<size_t>(m)); std::iota(idx.begin(), idx.end(), 0); std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int c) { return h[static_cast<size_t>(a)] > h[static_cast<size_t>(c)]; }); std::vector<char> keep(static_cast<size_t>(m), 0); for (int t = 0; t < k; ++t) keep[static_cast<size_t>(idx[static_cast<size_t>(t)])] = 1; for (int j = 0; j < m; ++j) if (!keep[static_cast<size_t>(j)]) h[static_cast<size_t>(j)] = 0.0f; }
    float G = 0; for (int j = 0; j < m; ++j) G += h[static_cast<size_t>(j)] * h[static_cast<size_t>(j)]; return G;
  }
  float fwd(const float* x, std::vector<float>& h) const { std::vector<float> xh; return act(x, h, xh); }
  void step(const float* x, bool pos, std::vector<float>& h) {
    std::vector<float> xhat; const float G = act(x, h, xhat); const float p = 1.0f / (1.0f + std::exp(-(G - theta))); theta = (1 - tr) * theta + tr * G; const float coef = pos ? -(1.0f - p) : p;
    for (int j = 0; j < m; ++j) { if (h[static_cast<size_t>(j)] <= 0) continue; const float g = lr * coef * 2.0f * h[static_cast<size_t>(j)]; float* Wj = &W[static_cast<size_t>(j) * d_in]; for (int i = 0; i < d_in; ++i) Wj[i] -= g * xhat[static_cast<size_t>(i)]; b[static_cast<size_t>(j)] -= g; }
  }
};
struct FFNet {
  std::vector<FFLayer> L; int depth, width, in0; float alpha;
  FFNet(int depth_, int width_, int d_in, float lr, float alpha_, uint64_t seed) : depth(depth_), width(width_), in0(d_in + NC), alpha(alpha_) {
    for (int i = 0; i < depth_; ++i) L.emplace_back((i == 0) ? d_in + NC : width_, width_, width_, lr, seed + static_cast<uint64_t>(i));
  }
  void embed(const float* x, int d, int label, std::vector<float>& o) const { o.resize(static_cast<size_t>(d) + NC); for (int i = 0; i < d; ++i) o[static_cast<size_t>(i)] = x[i]; for (int c = 0; c < NC; ++c) o[static_cast<size_t>(d) + c] = (c == label) ? alpha : 0.0f; }
  void train(const Data& d, int epochs, uint64_t seed) {
    std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rng(static_cast<uint32_t>(seed)); std::uniform_int_distribution<int> roll(1, NC - 1); std::vector<float> ip, in, cur, h;
    for (int e = 0; e < epochs; ++e) { std::shuffle(idx.begin(), idx.end(), rng); for (int s : idx) { const float* x = &d.X[static_cast<size_t>(s) * d.d]; const int y = d.y[static_cast<size_t>(s)]; const int yn = (y + roll(rng)) % NC; embed(x, d.d, y, ip); embed(x, d.d, yn, in); cur = ip; for (auto& l : L) { l.step(cur.data(), true, h); cur = h; } cur = in; for (auto& l : L) { l.step(cur.data(), false, h); cur = h; } } }
  }
  // classify by accumulated goodness (excluding layer 0 to avoid label-block leak)
  float accuracy(const Data& d, bool exclude0) {
    int ok = 0; std::vector<float> in, cur, h;
    for (int s = 0; s < d.N; ++s) { const float* x = &d.X[static_cast<size_t>(s) * d.d]; const int y = d.y[static_cast<size_t>(s)]; int best = 0; float bv = -1e30f;
      for (int c = 0; c < NC; ++c) { embed(x, d.d, c, in); cur = in; float G = 0; for (int li = 0; li < depth; ++li) { const float g = L[static_cast<size_t>(li)].fwd(cur.data(), h) - L[static_cast<size_t>(li)].theta; if (!(exclude0 && li == 0)) G += g; cur = h; } if (G > bv) { bv = G; best = c; } }
      if (best == y) ++ok; }
    return static_cast<float>(static_cast<double>(ok) / d.N);
  }
};

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const Data tr = gen(12000, 0.30f, 7), te = gen(3000, 0.30f, 8);
  std::printf("=====================================================================\n");
  std::printf(" PARITY-%d SHOWDOWN — does depth compose? backprop vs Forward-Forward\n", N_BITS);
  std::printf("=====================================================================\n");
  std::printf(" parity-%d (chance 0.500), narrow width => 1 layer CANNOT represent it; depth can.\n", N_BITS);
  std::printf(" %d train / %d test, noise 0.30 on +-1 bits.\n\n", tr.N, te.N);

  const int widths[] = {6, 16, 32};
  std::printf(" %-26s  train   test\n", "config");
  for (int w : widths) {
    std::printf(" --- width %d ---\n", w);
    // Backprop shallow (1 hidden) vs deep (3 hidden)
    { MLP m({N_BITS, w, NC}, 0.003f, 0.9f, 100); m.train(tr, 80, 11); std::printf("  %-26s  %.3f   %.3f\n", "backprop  1-hidden", m.accuracy(tr), m.accuracy(te)); }
    { MLP m({N_BITS, w, w, w, NC}, 0.003f, 0.9f, 101); m.train(tr, 80, 11); std::printf("  %-26s  %.3f   %.3f\n", "backprop  3-hidden (DEEP)", m.accuracy(tr), m.accuracy(te)); }
    // Forward-Forward shallow (1 layer) vs deep (3 layers)
    { FFNet f(1, w, N_BITS, 0.02f, 1.0f, 200); f.train(tr, 80, 11); std::printf("  %-26s  %.3f   %.3f\n", "fwd-fwd   1-layer", f.accuracy(tr, false), f.accuracy(te, false)); }
    { FFNet f(3, w, N_BITS, 0.02f, 1.0f, 203); f.train(tr, 80, 11); std::printf("  %-26s  %.3f   %.3f (no-L0 %.3f)\n", "fwd-fwd   3-layer (DEEP)", f.accuracy(tr, true), f.accuracy(te, true), f.accuracy(te, true)); }
    std::fprintf(stderr, "[width %d done]\n", w);
  }
  std::printf("\n read: depth COMPOSES for a method if its DEEP row >> its shallow row.\n");
  std::printf(" backprop = gold standard; FF = local forward-only (no backprop).\n");
  std::printf("=====================================================================\n");
  return 0;
}
