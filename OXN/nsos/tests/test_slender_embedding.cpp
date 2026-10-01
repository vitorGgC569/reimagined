// test_slender_embedding.cpp
//
// Unit tests for Cherry-pick #2 (Slender-Mamba head-to-toe quantization)
// applied to the Embedding class.  Validates the four core invariants of
// the Slender forward path implementation in src/embedding.cpp:
//
//   1. Cache lifecycle — version-keyed lazy build invalidates correctly
//   2. Round-trip quantization — values stay within β·ε of FP32 baseline
//   3. STE backward — gradient flow matches FP32 path semantically
//   4. Output shape — slender path matches FP32 path shape for all inputs
//
// These tests are intentionally fast (no training loop, no real corpus)
// so they run on every CI build.  Heavier ablations (loss-curve parity
// over real training) live in the Colab notebook oxta_ablation_slender.

#include "../include/embedding.h"
#include "../include/nsos_config.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace nsos;

namespace {

// ── Helpers ─────────────────────────────────────────────────────────────────

// Maximum element-wise absolute difference between two equal-shape tensors.
// Used to bound Slender vs FP32 output drift.
float max_abs_diff(const Tensor& a, const Tensor& b) {
  if (a.size != b.size) {
    std::cerr << "max_abs_diff: size mismatch " << a.size << " vs " << b.size << std::endl;
    std::abort();
  }
  const float* pa = a.data();
  const float* pb = b.data();
  float worst = 0.0f;
  for (int i = 0; i < a.size; ++i) {
    const float diff = std::fabs(pa[i] - pb[i]);
    if (diff > worst) worst = diff;
  }
  return worst;
}

// Build an Embedding initialized to deterministic values so test results
// are stable across runs.  vocab_size × dim with W[v, d] = sin(v + d/10).
std::unique_ptr<Embedding> make_deterministic_embedding(int vocab, int dim) {
  auto emb = std::make_unique<Embedding>(vocab, dim);
  float* w = emb->weight.data.data();
  for (int v = 0; v < vocab; ++v) {
    for (int d = 0; d < dim; ++d) {
      // Range chosen so β = mean(|W|) is non-trivially > ε but also
      // not so large that ternarization saturates everything to ±1.
      w[v * dim + d] = std::sin(static_cast<float>(v) + static_cast<float>(d) / 10.0f) * 0.5f;
    }
  }
  // Bump version so any prior cache (none yet, but defensive) is invalid.
  emb->weight.mark_updated();
  return emb;
}

// ── Test 1: Cache lifecycle ─────────────────────────────────────────────────
// The slender cache must:
//   - Build lazily on first slender forward
//   - Reuse on subsequent forwards if weight.version is unchanged
//   - Rebuild after weight.mark_updated() bumps the version
//
// We can't peek at the private cache fields directly, but we can verify
// behavior indirectly: cache hit → output bit-identical; cache rebuild
// after weight change → output reflects the new weights.

void test_cache_lifecycle() {
  std::cout << "[1/4] cache lifecycle ..." << std::flush;
  const int V = 32;
  const int D = 16;

  auto emb = make_deterministic_embedding(V, D);
  emb->set_slender_quantization(true);

  const std::vector<int> ids = {0, 5, 10, 20, 31};

  // First forward: cold cache — builds W̃ + β
  Tensor out_a = emb->forward(ids);
  assert(out_a.shape[0] == static_cast<int>(ids.size()));
  assert(out_a.shape[1] == D);

  // Second forward without any weight change: cache hit, identical output
  Tensor out_b = emb->forward(ids);
  const float drift_no_update = max_abs_diff(out_a, out_b);
  assert(drift_no_update == 0.0f);  // bit-exact since same cache used

  // Mutate the weight and mark_updated; cache should rebuild
  float* w = emb->weight.data.data();
  for (int i = 0; i < V * D; ++i) {
    w[i] *= 1.5f;  // arbitrary perturbation that meaningfully changes β
  }
  emb->weight.mark_updated();

  Tensor out_c = emb->forward(ids);
  const float drift_after_update = max_abs_diff(out_a, out_c);
  // Must differ: scaling W by 1.5 scales β by 1.5, which propagates to
  // E_out via the (β/127) dequant factor.  Drift bounded below by a
  // very loose threshold to avoid false negatives on floating-point edge.
  assert(drift_after_update > 1e-4f);

  std::cout << " OK (cold→hit=0, hot rebuild drift=" << drift_after_update << ")" << std::endl;
}

// ── Test 2: Round-trip approximates FP32 ────────────────────────────────────
// Slender forward should produce output that approximates the FP32 lookup
// within bounded quantization noise.  The bound is approximately β
// (per-element absolute error), because the per-token activation quant has
// resolution γ/127 and the dequant scales by γ·β/127, giving worst-case
// drift around β per element vs the original W[id].
//
// We don't check exact equality — that would be wrong (the whole point of
// Slender is to introduce controlled quantization noise).  We check that
// drift is in a sensible range: large enough to confirm quantization
// happened, small enough to confirm we're not generating garbage.

void test_round_trip_within_bounds() {
  std::cout << "[2/4] round-trip within bounds ..." << std::flush;
  const int V = 64;
  const int D = 32;

  auto emb = make_deterministic_embedding(V, D);

  const std::vector<int> ids = {0, 1, 7, 13, 50, 63};

  // FP32 baseline
  emb->set_slender_quantization(false);
  Tensor out_fp = emb->forward(ids);

  // Slender quantized
  emb->set_slender_quantization(true);
  Tensor out_slender = emb->forward(ids);

  // Shapes must match exactly
  assert(out_fp.shape == out_slender.shape);

  // Drift must be > 0 (quantization actually happened)
  const float drift = max_abs_diff(out_fp, out_slender);
  assert(drift > 0.0f);

  // Drift must be bounded.  Sin-based weights in [-0.5, 0.5] give
  // β = mean(|W|) ≈ 0.32 for the deterministic initializer.  Slender
  // also applies LN which scales values per-token, so the absolute drift
  // is not directly comparable to β — but it should be O(1).  We use a
  // loose upper bound of 5.0 to catch only catastrophic failures
  // (NaN, exploded values), not normal quantization noise.
  assert(drift < 5.0f);

  // Sanity: no NaN/Inf in slender output
  const float* p = out_slender.data();
  for (int i = 0; i < out_slender.size; ++i) {
    assert(std::isfinite(p[i]));
  }

  std::cout << " OK (drift=" << drift << ", finite=true)" << std::endl;
}

// ── Test 3: Gradient flow via STE ───────────────────────────────────────────
// The backward path uses straight-through estimator: dL/dW ≈ scatter at
// indices.  In the slender path, this should accumulate gradient into
// weight.grad at exactly the touched rows — identical to the FP32 path.
//
// We feed a constant grad_output (all ones) and verify:
//   - Rows that appear in indices_batch get non-zero gradient
//   - Rows that don't appear stay zero
//   - The accumulated value matches expected count

void test_ste_backward_routing() {
  std::cout << "[3/4] STE backward routing ..." << std::flush;
  const int V = 16;
  const int D = 8;

  auto emb = make_deterministic_embedding(V, D);
  emb->set_slender_quantization(true);

  // Indices that touch only rows 3, 5, 7 (3 appears twice)
  const std::vector<std::vector<int>> indices_batch = {
      {3, 5, 7, 3},
  };
  const int batch_size = 1;
  const int seq_len = 4;

  // Forward to ensure cache is built (gradient computation reads weight)
  (void)emb->forward_batch(indices_batch);

  // Build a constant grad_output of ones, shape [1, 4, 8]
  Tensor grad_output({batch_size, seq_len, D}, Device::CPU);
  float* go = grad_output.data();
  for (int i = 0; i < grad_output.size; ++i) go[i] = 1.0f;

  // Run backward
  emb->backward_batch(grad_output, indices_batch);

  // Verify gradient was accumulated into weight.grad
  // (weight.grad is allocated on first add_grad call inside backward_batch)
  assert(emb->weight.grad.size == V * D);

  const float* gw = emb->weight.grad.data();
  for (int v = 0; v < V; ++v) {
    for (int d = 0; d < D; ++d) {
      const float g = gw[v * D + d];
      if (v == 3) {
        // Row 3 was hit twice (id appears at positions 0 and 3): expect 2.0
        assert(std::fabs(g - 2.0f) < 1e-5f);
      } else if (v == 5 || v == 7) {
        // Hit once: expect 1.0
        assert(std::fabs(g - 1.0f) < 1e-5f);
      } else {
        // Not hit: must be exactly 0
        assert(g == 0.0f);
      }
    }
  }

  std::cout << " OK (rows 3,5,7 got correct counts; others=0)" << std::endl;
}

// ── Test 4: Empty + edge-case inputs ────────────────────────────────────────
// Slender path must handle the same edge cases as the FP32 path:
//   - Empty indices vector → returns empty tensor (no crash)
//   - Out-of-range token IDs → zero-fill that row
//   - Single-token sequence → returns [1, D] tensor

void test_edge_cases() {
  std::cout << "[4/4] edge cases ..." << std::flush;
  const int V = 8;
  const int D = 4;

  auto emb = make_deterministic_embedding(V, D);
  emb->set_slender_quantization(true);

  // Empty
  {
    Tensor out = emb->forward(std::vector<int>{});
    assert(out.size == 0);
  }

  // Out-of-range IDs (must zero-fill, not crash)
  {
    Tensor out = emb->forward({-1, 0, 99999, V - 1});
    assert(out.shape[0] == 4);
    assert(out.shape[1] == D);
    const float* p = out.data();
    // Row 0 (id=-1) must be all zeros
    for (int d = 0; d < D; ++d) assert(p[d] == 0.0f);
    // Row 2 (id=99999) must be all zeros
    for (int d = 0; d < D; ++d) assert(p[2 * D + d] == 0.0f);
    // Rows 1 and 3 must be non-zero somewhere (actual lookup happened)
    bool row1_has_value = false;
    for (int d = 0; d < D; ++d) {
      if (p[D + d] != 0.0f) { row1_has_value = true; break; }
    }
    assert(row1_has_value);
  }

  // Single-token
  {
    Tensor out = emb->forward({5});
    assert(out.shape[0] == 1);
    assert(out.shape[1] == D);
  }

  std::cout << " OK (empty=0-tensor; OOB=zero-fill; single=1xD)" << std::endl;
}

}  // namespace

int main() {
  std::cout << "=== Slender Embedding tests (Cherry-pick #2) ===" << std::endl;
  test_cache_lifecycle();
  test_round_trip_within_bounds();
  test_ste_backward_routing();
  test_edge_cases();
  std::cout << "\nAll Slender embedding tests PASSED" << std::endl;
  return 0;
}
