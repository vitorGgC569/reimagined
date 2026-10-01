// ============================================================================
// OXTA-MEM (per-neuron dual-trace) — forward-only memory vs catastrophic forgetting
// ============================================================================
// Oxta's idea: let a neuron "keep existing" via memory. Each neuron has a FAST
// weight (plastic, current behavior) and a SLOW consolidated memory weight. The
// slow trace is a lazy EMA of the fast one; an elastic ANCHOR pulls the fast
// weight back toward its memory, resisting drift; dead units revive FROM memory
// (come back as what they were), not random. All local, forward-only, no backprop.
//
// Validation = continual learning. Train task A, then task B (same neurons), then
// briefly re-touch A. Measure how much of A survives B (retention) and how fast A
// recovers (savings) — dual-trace vs a vanilla single-trace competitive layer.
//
// Build: cl /nologo /O2 /EHsc /std:c++17 oxtamem_demo.cpp /Fe:oxtamem_demo.exe
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int DIM = 16, K = 6, M = 12;   // input dim, clusters/task, neurons
constexpr int PER = 240;

void l2norm(float* v, int d) {
  float n = 0.0f; for (int i = 0; i < d; ++i) n += v[i] * v[i];
  n = std::sqrt(n) + 1e-8f; for (int i = 0; i < d; ++i) v[i] /= n;
}

struct Data { std::vector<float> X; std::vector<int> y; int N = 0; };

Data make_task(int k, int per, float noise, uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<float>> C(static_cast<size_t>(k), std::vector<float>(DIM));
  for (int c = 0; c < k; ++c) { for (int i = 0; i < DIM; ++i) C[static_cast<size_t>(c)][static_cast<size_t>(i)] = nd(rng); l2norm(C[static_cast<size_t>(c)].data(), DIM); }
  Data d; d.N = k * per; d.X.resize(static_cast<size_t>(d.N) * DIM); d.y.resize(static_cast<size_t>(d.N));
  int idx = 0;
  for (int c = 0; c < k; ++c) for (int p = 0; p < per; ++p) {
    float* r = &d.X[static_cast<size_t>(idx) * DIM];
    for (int i = 0; i < DIM; ++i) r[i] = C[static_cast<size_t>(c)][static_cast<size_t>(i)] + noise * nd(rng);
    l2norm(r, DIM); d.y[static_cast<size_t>(idx)] = c; ++idx;
  }
  return d;
}

// Dual-trace competitive layer. dual=false => vanilla single-trace (the baseline).
struct OxtaMem {
  int in, m, k; float lr, gamma, lambda; bool dual;
  std::vector<float> wf, ws;  // fast, slow consolidated
  std::vector<float> freq;    // for dead-unit revival
  OxtaMem(int in_, int m_, int k_, float lr_, float gamma_, float lambda_, bool dual_, uint64_t seed)
      : in(in_), m(m_), k(k_), lr(lr_), gamma(gamma_), lambda(lambda_), dual(dual_),
        wf(static_cast<size_t>(m_) * in_), ws(static_cast<size_t>(m_) * in_), freq(static_cast<size_t>(m_), 0.0f) {
    std::mt19937 rng(static_cast<uint32_t>(seed)); std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& w : wf) w = nd(rng);
    for (int j = 0; j < m; ++j) norm(wf, j);
    ws = wf;  // memory starts as the fast weights
  }
  float cos(const std::vector<float>& w, int j, const float* x) const {
    const float* wj = &w[static_cast<size_t>(j) * in]; float s = 0.0f; for (int i = 0; i < in; ++i) s += wj[i] * x[i]; return s;
  }
  void norm(std::vector<float>& w, int j) {
    float* wj = &w[static_cast<size_t>(j) * in]; float n = 0.0f; for (int i = 0; i < in; ++i) n += wj[i] * wj[i];
    n = std::sqrt(n) + 1e-8f; for (int i = 0; i < in; ++i) wj[i] /= n;
  }
  std::vector<int> winners(const float* x) const {
    std::vector<int> idx(static_cast<size_t>(m)); std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return cos(wf, a, x) > cos(wf, b, x); });
    idx.resize(static_cast<size_t>(k)); return idx;
  }
  void step(const float* x) {
    const std::vector<int> win = winners(x);
    for (int j = 0; j < m; ++j) freq[static_cast<size_t>(j)] *= 0.999f;
    for (int j : win) {
      freq[static_cast<size_t>(j)] += 0.001f;
      float* f = &wf[static_cast<size_t>(j) * in];
      for (int i = 0; i < in; ++i) f[i] += lr * (x[i] - f[i]);   // competitive move
      norm(wf, j);
      if (dual) {
        float* s = &ws[static_cast<size_t>(j) * in];
        for (int i = 0; i < in; ++i) s[i] = (1.0f - gamma) * s[i] + gamma * f[i];  // lazy consolidation
        norm(ws, j);
        for (int i = 0; i < in; ++i) f[i] += lambda * (s[i] - f[i]);  // elastic anchor to memory
        norm(wf, j);
      }
    }
  }
  // dead-unit revival FROM memory (dual) or random (vanilla)
  void revive(float thresh, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int j = 0; j < m; ++j) if (freq[static_cast<size_t>(j)] < thresh) {
      float* f = &wf[static_cast<size_t>(j) * in];
      if (dual) { const float* s = &ws[static_cast<size_t>(j) * in]; for (int i = 0; i < in; ++i) f[i] = s[i]; }
      else { for (int i = 0; i < in; ++i) f[i] = nd(rng); }
      norm(wf, j); freq[static_cast<size_t>(j)] = 1.0f / m;
    }
  }
  int winner(const float* x, bool useSlow) const {
    const std::vector<float>& w = useSlow ? ws : wf; int best = 0; float bv = -1e30f;
    for (int j = 0; j < m; ++j) { const float v = cos(w, j, x); if (v > bv) { bv = v; best = j; } }
    return best;
  }
};

void train(OxtaMem& L, const Data& d, int epochs, uint64_t seed) {
  std::vector<int> idx(static_cast<size_t>(d.N)); std::iota(idx.begin(), idx.end(), 0);
  std::mt19937 rng(static_cast<uint32_t>(seed)); std::mt19937 rr(static_cast<uint32_t>(seed ^ 0x5bd1e995u));
  long seen = 0;
  for (int e = 0; e < epochs; ++e) {
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int s : idx) { L.step(&d.X[static_cast<size_t>(s) * DIM]); if (++seen % 1500 == 0) L.revive(0.2f / M, rr); }
  }
}

// readout accuracy: label each neuron by majority true cluster that fires it.
float readout(const OxtaMem& L, const Data& d, bool useSlow) {
  std::vector<std::vector<int>> votes(static_cast<size_t>(M), std::vector<int>(static_cast<size_t>(K), 0));
  std::vector<int> win(static_cast<size_t>(d.N));
  for (int s = 0; s < d.N; ++s) { win[static_cast<size_t>(s)] = L.winner(&d.X[static_cast<size_t>(s) * DIM], useSlow); votes[static_cast<size_t>(win[static_cast<size_t>(s)])][static_cast<size_t>(d.y[static_cast<size_t>(s)])]++; }
  std::vector<int> lab(static_cast<size_t>(M), -1);
  for (int j = 0; j < M; ++j) { int bc = -1, bv = -1; for (int c = 0; c < K; ++c) if (votes[static_cast<size_t>(j)][static_cast<size_t>(c)] > bv) { bv = votes[static_cast<size_t>(j)][static_cast<size_t>(c)]; bc = c; } lab[static_cast<size_t>(j)] = bc; }
  int ok = 0; for (int s = 0; s < d.N; ++s) if (lab[static_cast<size_t>(win[static_cast<size_t>(s)])] == d.y[static_cast<size_t>(s)]) ++ok;
  return static_cast<float>(static_cast<double>(ok) / d.N);
}

void scenario(const char* name, bool dual, float gamma, float lambda, const Data& A, const Data& B) {
  OxtaMem L(DIM, M, 1, 0.10f, gamma, lambda, dual, 1234);
  train(L, A, 25, 11);            const float accA1 = readout(L, A, false);
  train(L, B, 25, 22);            const float accB = readout(L, B, false);
  const float accA2f = readout(L, A, false);
  const float accA2s = dual ? readout(L, A, true) : accA2f;
  train(L, A, 4, 33);            const float accA3 = readout(L, A, false);  // brief recovery
  std::printf("  %-26s  A1=%.3f  B=%.3f | retain A(fast)=%.3f", name, accA1, accB, accA2f);
  if (dual) std::printf("  A(memory)=%.3f", accA2s);
  std::printf("  | recover A=%.3f\n", accA3);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const Data A = make_task(K, PER, 0.22f, 101);
  const Data B = make_task(K, PER, 0.22f, 202);
  std::printf("=====================================================================\n");
  std::printf(" OXTA-MEM dual-trace — forward-only memory vs catastrophic forgetting\n");
  std::printf("=====================================================================\n");
  std::printf(" %d neurons, task A=%d clusters, task B=%d clusters (disjoint), dim=%d\n", M, K, K, DIM);
  std::printf(" protocol: train A (25ep) -> train B (25ep) -> brief re-touch A (4ep)\n");
  std::printf(" chance readout ~ %.3f\n\n", 1.0f / K);

  std::printf("RESULTS (higher 'retain A' = less forgetting; higher 'recover A' = savings):\n");
  scenario("vanilla (single-trace)", false, 0.0f, 0.0f, A, B);
  scenario("oxta-mem (mild)", true, 0.02f, 0.08f, A, B);
  scenario("oxta-mem (strong anchor)", true, 0.015f, 0.20f, A, B);

  std::printf("\n note: dual-trace trades a little B-plasticity for A-retention + faster A-recovery;\n");
  std::printf("       'A(memory)' = readout using the SLOW consolidated weights (the memory itself).\n");
  std::printf("=====================================================================\n");
  return 0;
}
