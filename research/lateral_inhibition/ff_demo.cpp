// ============================================================================
// FORWARD-FORWARD — does a LOCAL forward-only learning signal make depth COMPOSE?
// ============================================================================
// The unsupervised competitive stack peaked at layer 1 and eroded: pure WTA has
// no REASON to build the XOR conjunction. Forward-Forward (Hinton 2022) gives
// every layer a LOCAL objective: "goodness" = sum of squared activations, pushed
// HIGH for positive data (input + correct label) and LOW for negative data
// (input + wrong label). Each layer's update is the local gradient of its own
// goodness — NO backprop, no cross-layer gradient. Length-normalizing between
// layers forces each layer to find NEW structure -> the mechanism for depth.
//
// Same XOR-of-part-groups task as the unsupervised study, so results compare
// directly (unsupervised single-layer ref ~0.387; chance 0.25).
// Honest leak check: if layer-1-alone already solves it, depth isn't composing.
//
// Build: cl /nologo /O2 /EHsc /std:c++17 ff_demo.cpp /Fe:ff_demo.exe
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int D = 64, NC = 4, M = 64, N_LAYERS = 25;
constexpr int P = 16, GROUPS = 4, PARTS_PER_GROUP = 4;

void l2norm(float* v, int d) {
  float n = 0.0f; for (int i = 0; i < d; ++i) n += v[i] * v[i];
  n = std::sqrt(n) + 1e-8f; for (int i = 0; i < d; ++i) v[i] /= n;
}

struct Dataset { std::vector<float> X; std::vector<int> y; int N = 0; };

std::vector<float> make_parts(uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> parts(static_cast<size_t>(P) * D);
  for (int p = 0; p < P; ++p) { float* r = &parts[static_cast<size_t>(p) * D]; for (int i = 0; i < D; ++i) r[i] = nd(rng); l2norm(r, D); }
  return parts;
}

Dataset gen(int n, const std::vector<float>& parts, uint64_t seed, float sigma) {
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::bernoulli_distribution coin(0.5);
  std::uniform_int_distribution<int> pick(0, PARTS_PER_GROUP - 1); std::normal_distribution<float> noise(0.0f, sigma);
  Dataset ds; ds.N = n; ds.X.resize(static_cast<size_t>(n) * D); ds.y.resize(static_cast<size_t>(n));
  for (int s = 0; s < n;) {
    int b[GROUPS]; int present = 0; float* x = &ds.X[static_cast<size_t>(s) * D];
    for (int i = 0; i < D; ++i) x[i] = 0.0f;
    for (int g = 0; g < GROUPS; ++g) { b[g] = coin(rng) ? 1 : 0; present += b[g]; if (b[g]) { const int part = g * PARTS_PER_GROUP + pick(rng); const float* pv = &parts[static_cast<size_t>(part) * D]; for (int i = 0; i < D; ++i) x[i] += pv[i]; } }
    if (present == 0) continue;
    for (int i = 0; i < D; ++i) x[i] += noise(rng); l2norm(x, D);
    ds.y[static_cast<size_t>(s)] = 2 * (b[0] ^ b[1]) + (b[2] ^ b[3]); ++s;
  }
  return ds;
}

// ---- Forward-Forward layer (local, forward-only) ----
struct FFLayer {
  int d_in, m, k; float lr, theta = 0.0f, theta_rate = 0.01f;
  std::vector<float> W, b;
  FFLayer(int d_in_, int m_, int k_, float lr_, uint64_t seed)
      : d_in(d_in_), m(m_), k(std::min(k_, m_)), lr(lr_), W(static_cast<size_t>(m_) * d_in_), b(static_cast<size_t>(m_), 0.0f) {
    std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt(static_cast<float>(d_in_)));
    for (auto& w : W) w = nd(rng);
  }
  // ReLU(W * xhat + b), optional top-k WTA. Fills h (size m), returns goodness sum h^2. Also fills xhat.
  float activate(const float* x_in, std::vector<float>& h, std::vector<float>& xhat) const {
    float nn = 0.0f; for (int i = 0; i < d_in; ++i) nn += x_in[i] * x_in[i]; nn = std::sqrt(nn) + 1e-8f;
    xhat.resize(static_cast<size_t>(d_in)); for (int i = 0; i < d_in; ++i) xhat[static_cast<size_t>(i)] = x_in[i] / nn;
    h.assign(static_cast<size_t>(m), 0.0f);
    for (int j = 0; j < m; ++j) { const float* Wj = &W[static_cast<size_t>(j) * d_in]; float s = b[static_cast<size_t>(j)]; for (int i = 0; i < d_in; ++i) s += Wj[i] * xhat[static_cast<size_t>(i)]; h[static_cast<size_t>(j)] = s > 0.0f ? s : 0.0f; }
    if (k < m) {  // lateral inhibition: keep top-k activations
      std::vector<int> idx(static_cast<size_t>(m)); std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int c) { return h[static_cast<size_t>(a)] > h[static_cast<size_t>(c)]; });
      std::vector<char> keep(static_cast<size_t>(m), 0); for (int t = 0; t < k; ++t) keep[static_cast<size_t>(idx[static_cast<size_t>(t)])] = 1;
      for (int j = 0; j < m; ++j) if (!keep[static_cast<size_t>(j)]) h[static_cast<size_t>(j)] = 0.0f;
    }
    float G = 0.0f; for (int j = 0; j < m; ++j) G += h[static_cast<size_t>(j)] * h[static_cast<size_t>(j)];
    return G;
  }
  float forward(const float* x_in, std::vector<float>& h) const { std::vector<float> xhat; return activate(x_in, h, xhat); }
  // Local FF update; returns the (pre-update) activations to pass upward.
  void train_step(const float* x_in, bool is_pos, std::vector<float>& h) {
    std::vector<float> xhat; const float G = activate(x_in, h, xhat);
    const float p = 1.0f / (1.0f + std::exp(-(G - theta)));   // P(positive) by goodness
    theta = (1.0f - theta_rate) * theta + theta_rate * G;     // adaptive midpoint
    const float coef = is_pos ? -(1.0f - p) : p;              // dLoss/dG (descent)
    for (int j = 0; j < m; ++j) {
      if (h[static_cast<size_t>(j)] <= 0.0f) continue;         // inactive/inhibited: no grad
      const float g = lr * coef * 2.0f * h[static_cast<size_t>(j)];
      float* Wj = &W[static_cast<size_t>(j) * d_in];
      for (int i = 0; i < d_in; ++i) Wj[i] -= g * xhat[static_cast<size_t>(i)];
      b[static_cast<size_t>(j)] -= g;
    }
  }
};

using Stack = std::vector<FFLayer>;

Stack build(int L, int k, float lr, uint64_t seed) {
  Stack s; s.reserve(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) s.emplace_back((i == 0) ? D + NC : M, M, k, lr, seed + static_cast<uint64_t>(i));
  return s;
}

// input0 = [ x (D) , onehot(label)*alpha (NC) ]
void embed(const float* x, int label, float alpha, std::vector<float>& out) {
  out.resize(static_cast<size_t>(D) + NC);
  for (int i = 0; i < D; ++i) out[static_cast<size_t>(i)] = x[i];
  for (int c = 0; c < NC; ++c) out[static_cast<size_t>(D) + c] = (c == label) ? alpha : 0.0f;
}

void train(Stack& net, const Dataset& tr, int epochs, float alpha, uint64_t seed) {
  const int L = static_cast<int>(net.size());
  std::vector<int> idx(static_cast<size_t>(tr.N)); std::iota(idx.begin(), idx.end(), 0);
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::uniform_int_distribution<int> roll(1, NC - 1);
  std::vector<float> in_pos, in_neg, cur, h;
  for (int e = 0; e < epochs; ++e) {
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int s : idx) {
      const float* x = &tr.X[static_cast<size_t>(s) * D]; const int y = tr.y[static_cast<size_t>(s)];
      const int yn = (y + roll(rng)) % NC;       // a wrong label
      embed(x, y, alpha, in_pos); embed(x, yn, alpha, in_neg);
      cur = in_pos; for (int l = 0; l < L; ++l) { net[static_cast<size_t>(l)].train_step(cur.data(), true, h); cur = h; }
      cur = in_neg; for (int l = 0; l < L; ++l) { net[static_cast<size_t>(l)].train_step(cur.data(), false, h); cur = h; }
    }
    std::fprintf(stderr, "    epoch %d/%d\n", e + 1, epochs);
  }
}

// Per-depth classification accuracy. For each test x and each candidate label c,
// forward (embedding c) and accumulate (G_l - theta_l). Predict argmax_c.
// acc[d]    = cumulative over layers 0..d-1.
// accNo0[d] = cumulative over layers 1..d-1 (honest: excludes label-leaking L0).
void evaluate(Stack& net, const Dataset& te, float alpha, std::vector<float>& acc,
              std::vector<float>& accNo0, float& layer0_only) {
  const int L = static_cast<int>(net.size());
  acc.assign(static_cast<size_t>(L) + 1, 0.0f); accNo0.assign(static_cast<size_t>(L) + 1, 0.0f);
  std::vector<long> ok(static_cast<size_t>(L) + 1, 0), okN(static_cast<size_t>(L) + 1, 0); long l0 = 0;
  std::vector<float> in, cur, h;
  std::vector<std::vector<float>> g(static_cast<size_t>(NC), std::vector<float>(static_cast<size_t>(L), 0.0f));
  for (int s = 0; s < te.N; ++s) {
    const float* x = &te.X[static_cast<size_t>(s) * D]; const int y = te.y[static_cast<size_t>(s)];
    for (int c = 0; c < NC; ++c) {
      embed(x, c, alpha, in); cur = in;
      for (int l = 0; l < L; ++l) { const float G = net[static_cast<size_t>(l)].forward(cur.data(), h); g[static_cast<size_t>(c)][static_cast<size_t>(l)] = G - net[static_cast<size_t>(l)].theta; cur = h; }
    }
    // layer-0-only leak check
    { int best = 0; float bv = -1e30f; for (int c = 0; c < NC; ++c) if (g[static_cast<size_t>(c)][0] > bv) { bv = g[static_cast<size_t>(c)][0]; best = c; } if (best == y) ++l0; }
    // cumulative per depth
    std::vector<float> cum(static_cast<size_t>(NC), 0.0f), cumN(static_cast<size_t>(NC), 0.0f);
    for (int d = 1; d <= L; ++d) {
      const int l = d - 1;
      for (int c = 0; c < NC; ++c) { cum[static_cast<size_t>(c)] += g[static_cast<size_t>(c)][static_cast<size_t>(l)]; if (l >= 1) cumN[static_cast<size_t>(c)] += g[static_cast<size_t>(c)][static_cast<size_t>(l)]; }
      int b1 = 0; float v1 = -1e30f; for (int c = 0; c < NC; ++c) if (cum[static_cast<size_t>(c)] > v1) { v1 = cum[static_cast<size_t>(c)]; b1 = c; } if (b1 == y) ok[static_cast<size_t>(d)]++;
      int b2 = 0; float v2 = -1e30f; for (int c = 0; c < NC; ++c) if (cumN[static_cast<size_t>(c)] > v2) { v2 = cumN[static_cast<size_t>(c)]; b2 = c; } if (b2 == y) okN[static_cast<size_t>(d)]++;
    }
  }
  const double n = te.N;
  for (int d = 1; d <= L; ++d) { acc[static_cast<size_t>(d)] = static_cast<float>(ok[static_cast<size_t>(d)] / n); accNo0[static_cast<size_t>(d)] = static_cast<float>(okN[static_cast<size_t>(d)] / n); }
  layer0_only = static_cast<float>(l0 / n);
}

void run(const char* name, int k, const Dataset& tr, const Dataset& te, float alpha, int epochs, float lr) {
  std::fprintf(stderr, "[%s] training (k=%d)...\n", name, k);
  Stack net = build(N_LAYERS, k, lr, 4242);
  train(net, tr, epochs, alpha, 11);
  std::vector<float> acc, accNo0; float l0 = 0.0f;
  evaluate(net, te, alpha, acc, accNo0, l0);
  std::printf("%s (k=%d, alpha=%.2f, lr=%.3f, %d epochs)\n", name, k, alpha, lr, epochs);
  const int ds[] = {1, 2, 3, 5, 8, 12, 18, 25};
  std::printf("  acc (layers 0..d):    "); for (int d : ds) std::printf("d%-2d %.3f  ", d, acc[static_cast<size_t>(d)]); std::printf("\n");
  std::printf("  acc (layers 1..d, no-leak): "); for (int d : ds) std::printf("d%-2d %.3f  ", d, accNo0[static_cast<size_t>(d)]); std::printf("\n");
  float bestN = 0.0f; int bestD = 1; for (int d = 2; d <= N_LAYERS; ++d) if (accNo0[static_cast<size_t>(d)] > bestN) { bestN = accNo0[static_cast<size_t>(d)]; bestD = d; }
  std::printf("  layer-0-ONLY (leak check) = %.3f   |   best no-leak = %.3f @ depth %d\n", l0, bestN, bestD);
  const float d2 = accNo0[2];
  std::printf("  => depth helps (no-leak best - depth2): %+.3f  %s\n\n", bestN - d2,
              (bestN - d2) >= 0.05f ? "[DEPTH COMPOSES via local signal]" : "[little depth gain]");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::vector<float> parts = make_parts(101);
  const Dataset tr = gen(12000, parts, 7, 0.10f), te = gen(3000, parts, 8, 0.10f);
  std::printf("=====================================================================\n");
  std::printf(" FORWARD-FORWARD — local signal: does depth compose? (no backprop)\n");
  std::printf("=====================================================================\n");
  std::printf(" XOR-of-groups, 4 classes, chance 0.250 | unsupervised ref: single-layer 0.387\n");
  std::printf(" classify = argmax_label accumulated goodness; no-leak curve excludes layer 0\n\n");

  run("PLAIN-FF", M, tr, te, 0.7f, 15, 0.03f);          // no inhibition (k=m)
  run("FF + LATERAL INHIBITION", 16, tr, te, 0.7f, 15, 0.03f);  // k-WTA

  std::printf("=====================================================================\n");
  return 0;
}
