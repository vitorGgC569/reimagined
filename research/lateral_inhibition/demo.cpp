// Validation of forward-only learning via LATERAL INHIBITION.
//
// Thesis: a competitive layer that learns with LOCAL updates (no backprop)
// only works because of lateral inhibition (winner-take-all).  We test it on a
// clustered dataset and compare inhibition ON (k=1) vs OFF (k=all):
//   * ON  -> prototypes SPECIALISE on clusters: low quantisation error, many
//            active neurons, high unsupervised readout accuracy.
//   * OFF -> prototypes COLLAPSE to one mean: high error, 1 active neuron,
//            chance-level accuracy.
// Plus a drift test (continual on-device adaptation) and wall-clock timing.
//
// Build (standalone, no NSOS):  cl /O2 /EHsc /std:c++17 demo.cpp
#include "lateral_inhibition.h"

#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

struct Data {
  std::vector<float> x;       // [N * d]
  std::vector<int> label;     // [N]
  int N = 0, d = 0, K = 0;
};

void normalize(float* v, int d) {
  float n = 0.0f;
  for (int i = 0; i < d; ++i) n += v[i] * v[i];
  n = std::sqrt(n) + 1e-8f;
  for (int i = 0; i < d; ++i) v[i] /= n;
}

// K clusters: each is a random unit direction; points = center + noise, then
// projected back onto the unit sphere.
Data make_clusters(int K, int per, int d, float noise, uint64_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<float>> centers(K, std::vector<float>(d));
  for (int c = 0; c < K; ++c) {
    for (int i = 0; i < d; ++i) centers[c][i] = nd(rng);
    normalize(centers[c].data(), d);
  }
  Data data;
  data.d = d; data.K = K; data.N = K * per;
  data.x.resize(static_cast<size_t>(data.N) * d);
  data.label.resize(static_cast<size_t>(data.N));
  int idx = 0;
  for (int c = 0; c < K; ++c)
    for (int p = 0; p < per; ++p) {
      float* row = &data.x[static_cast<size_t>(idx) * d];
      for (int i = 0; i < d; ++i) row[i] = centers[c][i] + noise * nd(rng);
      normalize(row, d);
      data.label[static_cast<size_t>(idx)] = c;
      ++idx;
    }
  return data;
}

float quant_error(const lat::LateralInhibition& layer, const Data& data) {
  double s = 0.0;
  for (int n = 0; n < data.N; ++n)
    s += 1.0 - layer.similarity_to_winner(&data.x[static_cast<size_t>(n) * data.d]);
  return static_cast<float>(s / std::max(data.N, 1));
}

int active_neurons(const lat::LateralInhibition& layer, const Data& data) {
  std::vector<char> used(static_cast<size_t>(layer.neurons()), 0);
  for (int n = 0; n < data.N; ++n)
    used[static_cast<size_t>(layer.winner(&data.x[static_cast<size_t>(n) * data.d]))] = 1;
  return std::accumulate(used.begin(), used.end(), 0,
                         [](int a, char b) { return a + (b ? 1 : 0); });
}

// Unsupervised readout: label each neuron by the majority true cluster of the
// points that fire it, then measure classification accuracy.  This is the
// "did it learn useful features?" metric.
float readout_accuracy(const lat::LateralInhibition& layer, const Data& data) {
  const int m = layer.neurons();
  std::vector<std::vector<int>> votes(static_cast<size_t>(m),
                                      std::vector<int>(static_cast<size_t>(data.K), 0));
  std::vector<int> win(static_cast<size_t>(data.N));
  for (int n = 0; n < data.N; ++n) {
    win[static_cast<size_t>(n)] = layer.winner(&data.x[static_cast<size_t>(n) * data.d]);
    votes[static_cast<size_t>(win[static_cast<size_t>(n)])]
         [static_cast<size_t>(data.label[static_cast<size_t>(n)])]++;
  }
  std::vector<int> neuron_label(static_cast<size_t>(m), -1);
  for (int j = 0; j < m; ++j) {
    int best = -1, bv = -1;
    for (int c = 0; c < data.K; ++c)
      if (votes[static_cast<size_t>(j)][static_cast<size_t>(c)] > bv) {
        bv = votes[static_cast<size_t>(j)][static_cast<size_t>(c)];
        best = c;
      }
    neuron_label[static_cast<size_t>(j)] = best;
  }
  int correct = 0;
  for (int n = 0; n < data.N; ++n)
    if (neuron_label[static_cast<size_t>(win[static_cast<size_t>(n)])] ==
        data.label[static_cast<size_t>(n)])
      ++correct;
  return static_cast<float>(correct) / std::max(data.N, 1);
}

void train(lat::LateralInhibition& layer, const Data& data, int epochs,
           uint64_t seed, bool verbose) {
  std::vector<int> order(static_cast<size_t>(data.N));
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(seed);
  for (int e = 0; e < epochs; ++e) {
    std::shuffle(order.begin(), order.end(), rng);
    for (int idx : order) layer.step(&data.x[static_cast<size_t>(idx) * data.d]);
    if (verbose && (e % 4 == 0 || e == epochs - 1))
      std::printf("    epoch %2d  quant_err=%.4f  active=%d\n", e,
                  quant_error(layer, data), active_neurons(layer, data));
  }
}

}  // namespace

int main() {
  const int d = 16, K = 6, per = 220, m = 12, epochs = 24;
  const float noise = 0.28f, lr = 0.10f;
  const Data data = make_clusters(K, per, d, noise, /*seed*/ 7);
  const float chance = 1.0f / static_cast<float>(K);

  std::printf("Lateral-inhibition forward-only learning (no backprop)\n");
  std::printf("  data: %d points, dim=%d, %d clusters | neurons=%d  chance=%.3f\n\n",
              data.N, d, K, m, chance);

  // ---- Inhibition ON (k=1: winner-take-all, the cat's focus) ----
  std::printf("[1] Lateral inhibition ON  (k=1, winner-take-all)\n");
  lat::LateralInhibition on(d, m, /*k*/ 1, lr, /*seed*/ 3);
  const auto t0 = std::chrono::steady_clock::now();
  train(on, data, epochs, /*seed*/ 11, /*verbose*/ true);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const float qe_on = quant_error(on, data);
  const int act_on = active_neurons(on, data);
  const float acc_on = readout_accuracy(on, data);
  std::printf("    => quant_err=%.4f  active=%d  readout_acc=%.3f  (%.0f ms, %.0f samples/s)\n\n",
              qe_on, act_on, acc_on, ms,
              1000.0 * epochs * data.N / std::max(ms, 1e-6));

  // ---- Inhibition OFF (k=m: every neuron updates -> collapse) ----
  std::printf("[2] Lateral inhibition OFF (k=%d, no competition)\n", m);
  lat::LateralInhibition off(d, m, /*k*/ m, lr, /*seed*/ 3);
  train(off, data, epochs, /*seed*/ 11, /*verbose*/ false);
  const float qe_off = quant_error(off, data);
  const int act_off = active_neurons(off, data);
  const float acc_off = readout_accuracy(off, data);
  std::printf("    => quant_err=%.4f  active=%d  readout_acc=%.3f\n\n",
              qe_off, act_off, acc_off);

  // ---- Continual on-device adaptation: shift the world, keep learning ----
  std::printf("[3] Drift / continual adaptation (no retrain, no backprop)\n");
  const Data drifted = make_clusters(K, per, d, noise, /*seed*/ 99);  // new clusters
  const float qe_before = quant_error(on, drifted);
  on.set_lr(0.12f);
  train(on, drifted, 12, /*seed*/ 23, /*verbose*/ false);
  const float qe_after = quant_error(on, drifted);
  std::printf("    quant_err on shifted data: %.4f -> %.4f  (adapts online)\n\n",
              qe_before, qe_after);

  // ---- Verdict ----
  const bool learned = acc_on > 0.80f;                 // learned real structure
  const bool inhibition_matters = (acc_on > acc_off + 0.25f) && (qe_on < qe_off * 0.5f);
  const bool adapts = qe_after < qe_before * 0.7f;
  std::printf("VERDICT:\n");
  std::printf("  forward-only learning works ........ %s (readout %.3f vs chance %.3f)\n",
              learned ? "YES" : "no ", acc_on, chance);
  std::printf("  lateral inhibition is THE cause .... %s (acc %.3f vs %.3f; qerr %.3f vs %.3f)\n",
              inhibition_matters ? "YES" : "no ", acc_on, acc_off, qe_on, qe_off);
  std::printf("  continual on-device adaptation ..... %s\n", adapts ? "YES" : "no ");

  const bool ok = learned && inhibition_matters && adapts;
  std::printf("\n%s\n", ok ? "ALL VALIDATIONS PASSED." : "VALIDATION FAILED.");
  return ok ? 0 : 1;
}
