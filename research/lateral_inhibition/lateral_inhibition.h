#pragma once
// ============================================================================
// Lateral Inhibition — forward-only competitive learning (NO backprop)
// ============================================================================
// Oxta's idea: a cat hunting inhibits other neurons and focuses only on the
// prey.  That is LATERAL INHIBITION / winner-take-all: the most-activated
// neurons suppress the rest.  Paired with a LOCAL Hebbian/competitive update,
// it gives a layer that LEARNS features with:
//   * no backpropagation, no global gradient, no stored activations
//   * one streaming forward pass per sample  (CPU / edge native, continual)
//   * weights that SPECIALISE only because of the inhibition (without it they
//     collapse to the same prototype and nothing is learned).
//
// Standalone: depends only on the C++ standard library.  Not wired into NSOS.
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace lat {

class LateralInhibition {
public:
  // n_inputs : input dimension
  // n_neurons: feature detectors ("prototypes")
  // k_winners: how many neurons may fire per input (k=1 => full inhibition /
  //            winner-take-all; k=n_neurons => NO inhibition).
  LateralInhibition(int n_inputs, int n_neurons, int k_winners, float lr = 0.1f,
                    uint64_t seed = 1u)
      : in_(n_inputs), m_(n_neurons), k_(std::max(1, std::min(k_winners, n_neurons))),
        lr_(lr), w_(static_cast<size_t>(n_neurons) * n_inputs),
        freq_(static_cast<size_t>(n_neurons), 0.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : w_) v = nd(rng);
    for (int j = 0; j < m_; ++j) normalize(j);  // prototypes live on the unit sphere
  }

  // Cosine similarity of input x to every prototype.
  void activate(const float* x, std::vector<float>& a) const {
    a.assign(static_cast<size_t>(m_), 0.0f);
    for (int j = 0; j < m_; ++j) a[static_cast<size_t>(j)] = dot(j, x);
  }

  // Lateral inhibition: keep the top-k activations, suppress the rest to zero.
  std::vector<int> inhibit(const std::vector<float>& a) const {
    std::vector<int> idx(static_cast<size_t>(m_));
    for (int i = 0; i < m_; ++i) idx[static_cast<size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + k_, idx.end(),
                      [&](int p, int q) { return a[static_cast<size_t>(p)] >
                                                 a[static_cast<size_t>(q)]; });
    idx.resize(static_cast<size_t>(k_));
    return idx;
  }

  // LOCAL learning rule: each WINNER moves toward the input (competitive
  // learning / online spherical k-means).  Purely local — uses only the
  // neuron's own weights and the input.  No gradient flows between neurons.
  void learn(const float* x, const std::vector<int>& winners) {
    for (int j : winners) {
      float* wj = &w_[static_cast<size_t>(j) * in_];
      for (int i = 0; i < in_; ++i) wj[i] += lr_ * (x[i] - wj[i]);
      normalize(j);
    }
  }

  // One streaming train step.  Returns the winners (the firing neurons).
  std::vector<int> step(const float* x) {
    std::vector<float> a;
    activate(x, a);
    std::vector<int> winners = inhibit(a);
    learn(x, winners);
    return winners;
  }

  // Top-1 winner for inference / assignment.
  int winner(const float* x) const {
    int best = 0;
    float bv = -1e30f;
    for (int j = 0; j < m_; ++j) {
      const float v = dot(j, x);
      if (v > bv) { bv = v; best = j; }
    }
    return best;
  }

  float similarity_to_winner(const float* x) const {
    float bv = -1e30f;
    for (int j = 0; j < m_; ++j) bv = std::max(bv, dot(j, x));
    return bv;
  }

  const float* prototype(int j) const { return &w_[static_cast<size_t>(j) * in_]; }
  int neurons() const { return m_; }
  int inputs() const { return in_; }
  void set_lr(float lr) { lr_ = lr; }

  // ==========================================================================
  // Deep-stack additions (forward-only, local) — used to stack many layers.
  // All purely local: each neuron uses only its own weights, its own win
  // counter, and the input. No backprop, no cross-layer gradient.
  // ==========================================================================

  // Toggle the local homeostasis used when stacking deep:
  //  conscience = leaky win-frequency bias that revives silent neurons,
  //  decorr     = push apart co-active winners that grow too similar.
  void set_homeostasis(bool conscience, bool decorr) {
    beta_ = conscience ? 0.5f : 0.0f;
    use_decorr_ = decorr;
  }

  // One streaming TRAIN step that also emits a graded k-sparse code for the
  // next layer.  Ranking uses the conscience-biased activation; the LEARN rule
  // and the emitted code both use the RAW cosine activation.
  void step_code(const float* x, std::vector<float>& code_out) {
    std::vector<float> a;
    activate(x, a);
    std::vector<float> ab(a);
    const float inv_m = 1.0f / static_cast<float>(m_);
    for (int j = 0; j < m_; ++j)
      ab[static_cast<size_t>(j)] =
          a[static_cast<size_t>(j)] - beta_ * (freq_[static_cast<size_t>(j)] - inv_m);
    const std::vector<int> winners = inhibit(ab);
    // leaky win-frequency update (local conscience counter)
    for (int j = 0; j < m_; ++j)
      freq_[static_cast<size_t>(j)] *= (1.0f - decay_);
    for (int j : winners) freq_[static_cast<size_t>(j)] += decay_;
    learn(x, winners);
    if (use_decorr_) decorrelate(winners, 0.95f, 0.01f);
    build_code(a, winners, code_out);
  }

  // Forward-only ENCODE (no learning, no counter update): used after freezing
  // for all measurement.  Same biased ranking with the frozen freq_.
  void encode(const float* x, std::vector<float>& code_out) const {
    std::vector<float> a;
    activate(x, a);
    std::vector<float> ab(a);
    const float inv_m = 1.0f / static_cast<float>(m_);
    for (int j = 0; j < m_; ++j)
      ab[static_cast<size_t>(j)] =
          a[static_cast<size_t>(j)] - beta_ * (freq_[static_cast<size_t>(j)] - inv_m);
    const std::vector<int> winners = inhibit(ab);
    build_code(a, winners, code_out);
  }

  // Soft-temperature DENSE code variant (tests the information-bottleneck
  // hypothesis): learning is still hard top-k competitive, but the emitted
  // code is a dense softmax(a/tau)*relu(a), L2-normalized — carries far more
  // bits per layer than the k-sparse code.
  void step_code_soft(const float* x, std::vector<float>& code_out, float tau) {
    std::vector<float> a;
    activate(x, a);
    std::vector<float> ab(a);
    const float inv_m = 1.0f / static_cast<float>(m_);
    for (int j = 0; j < m_; ++j)
      ab[static_cast<size_t>(j)] =
          a[static_cast<size_t>(j)] - beta_ * (freq_[static_cast<size_t>(j)] - inv_m);
    const std::vector<int> winners = inhibit(ab);
    for (int j = 0; j < m_; ++j) freq_[static_cast<size_t>(j)] *= (1.0f - decay_);
    for (int j : winners) freq_[static_cast<size_t>(j)] += decay_;
    learn(x, winners);
    if (use_decorr_) decorrelate(winners, 0.95f, 0.01f);
    build_soft_code(a, tau, code_out);
  }
  void encode_soft(const float* x, std::vector<float>& code_out, float tau) const {
    std::vector<float> a;
    activate(x, a);
    build_soft_code(a, tau, code_out);
  }

  // Revive neurons that essentially never win (dead units): reseed prototype,
  // reset its counter to neutral.  Returns how many were reinitialized.
  int reinit_dead(float thresh, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int cnt = 0;
    for (int j = 0; j < m_; ++j) {
      if (freq_[static_cast<size_t>(j)] < thresh) {
        float* wj = &w_[static_cast<size_t>(j) * in_];
        for (int i = 0; i < in_; ++i) wj[i] = nd(rng);
        normalize(j);
        freq_[static_cast<size_t>(j)] = 1.0f / static_cast<float>(m_);
        ++cnt;
      }
    }
    return cnt;
  }

  // Push apart any pair of CURRENT winners whose prototypes are too similar.
  // Local to the co-active set; updates computed from the originals.
  void decorrelate(const std::vector<int>& winners, float tau, float push) {
    const int W = static_cast<int>(winners.size());
    for (int a = 0; a < W; ++a)
      for (int b = a + 1; b < W; ++b) {
        const int j = winners[static_cast<size_t>(a)];
        const int q = winners[static_cast<size_t>(b)];
        if (proto_cos(j, q) > tau) {
          const std::vector<float> oj(prototype(j), prototype(j) + in_);
          const std::vector<float> oq(prototype(q), prototype(q) + in_);
          float* wj = &w_[static_cast<size_t>(j) * in_];
          float* wq = &w_[static_cast<size_t>(q) * in_];
          for (int i = 0; i < in_; ++i) {
            wj[i] = oj[static_cast<size_t>(i)] - push * oq[static_cast<size_t>(i)];
            wq[i] = oq[static_cast<size_t>(i)] - push * oj[static_cast<size_t>(i)];
          }
          normalize(j);
          normalize(q);
        }
      }
  }

  // Average pairwise cosine of all prototypes (redundancy / collapse metric).
  float mean_pairwise_proto_cos() const {
    double s = 0.0;
    long cnt = 0;
    for (int j = 0; j < m_; ++j)
      for (int q = j + 1; q < m_; ++q) { s += proto_cos(j, q); ++cnt; }
    return cnt ? static_cast<float>(s / static_cast<double>(cnt)) : 0.0f;
  }

  const std::vector<float>& freq() const { return freq_; }
  long null_codes() const { return null_codes_; }
  void reset_null_codes() const { null_codes_ = 0; }

private:
  // Graded k-sparse code: keep RAW cosine (clipped at 0) on winners, L2-norm.
  void build_code(const std::vector<float>& a, const std::vector<int>& winners,
                  std::vector<float>& code_out) const {
    code_out.assign(static_cast<size_t>(m_), 0.0f);
    for (int j : winners)
      code_out[static_cast<size_t>(j)] = std::max(0.0f, a[static_cast<size_t>(j)]);
    float n = 0.0f;
    for (float v : code_out) n += v * v;
    n = std::sqrt(n);
    if (n < 1e-8f) {  // null code: feed a valid sphere point downstream
      const float u = 1.0f / std::sqrt(static_cast<float>(m_));
      for (int j = 0; j < m_; ++j) code_out[static_cast<size_t>(j)] = u;
      ++null_codes_;
    } else {
      for (int j = 0; j < m_; ++j) code_out[static_cast<size_t>(j)] /= n;
    }
  }

  float proto_cos(int j, int q) const {
    const float* wj = &w_[static_cast<size_t>(j) * in_];
    const float* wq = &w_[static_cast<size_t>(q) * in_];
    float s = 0.0f;
    for (int i = 0; i < in_; ++i) s += wj[i] * wq[i];
    return s;  // prototypes are unit-norm, so dot == cosine
  }

  // Dense soft code: softmax(a/tau) * relu(a), L2-normalized.
  void build_soft_code(const std::vector<float>& a, float tau,
                       std::vector<float>& code_out) const {
    code_out.assign(static_cast<size_t>(m_), 0.0f);
    float mx = -1e30f;
    for (int j = 0; j < m_; ++j) mx = std::max(mx, a[static_cast<size_t>(j)]);
    double Z = 0.0;
    std::vector<double> p(static_cast<size_t>(m_));
    for (int j = 0; j < m_; ++j) {
      const double e = std::exp(static_cast<double>(a[static_cast<size_t>(j)] - mx) / tau);
      p[static_cast<size_t>(j)] = e;
      Z += e;
    }
    float n = 0.0f;
    for (int j = 0; j < m_; ++j) {
      const float v = static_cast<float>(p[static_cast<size_t>(j)] / Z) *
                      std::max(0.0f, a[static_cast<size_t>(j)]);
      code_out[static_cast<size_t>(j)] = v;
      n += v * v;
    }
    n = std::sqrt(n);
    if (n < 1e-8f) {
      const float u = 1.0f / std::sqrt(static_cast<float>(m_));
      for (int j = 0; j < m_; ++j) code_out[static_cast<size_t>(j)] = u;
      ++null_codes_;
    } else {
      for (int j = 0; j < m_; ++j) code_out[static_cast<size_t>(j)] /= n;
    }
  }

  float dot(int j, const float* x) const {
    const float* wj = &w_[static_cast<size_t>(j) * in_];
    float s = 0.0f;
    for (int i = 0; i < in_; ++i) s += wj[i] * x[i];
    return s;
  }
  void normalize(int j) {
    float* wj = &w_[static_cast<size_t>(j) * in_];
    float n = 0.0f;
    for (int i = 0; i < in_; ++i) n += wj[i] * wj[i];
    n = std::sqrt(n) + 1e-8f;
    for (int i = 0; i < in_; ++i) wj[i] /= n;
  }

  int in_, m_, k_;
  float lr_;
  std::vector<float> w_;
  std::vector<float> freq_;       // leaky win-frequency (conscience), size m_
  float beta_ = 0.5f;             // conscience bias strength (0 => off)
  float decay_ = 0.001f;          // leaky rate of the win-frequency counter
  bool use_decorr_ = true;        // winner-decorrelation on/off
  mutable long null_codes_ = 0;   // count of all-zero (fallback) codes emitted
};

}  // namespace lat
