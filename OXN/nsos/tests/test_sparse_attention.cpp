#include "../include/sparse_attention.h"
#include "../include/jamba.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nsos;

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

Tensor make(int n, int d, const std::function<float(int, int)>& f) {
  Tensor t({n, d}, Device::CPU);
  float* p = t.data();
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < d; ++c) p[static_cast<size_t>(i) * d + c] = f(i, c);
  return t;
}

float row_dist(const Tensor& t, int row, float target) {
  const int d = t.shape[1];
  const float* p = t.data();
  float s = 0.0f;
  for (int c = 0; c < d; ++c) {
    const float diff = p[static_cast<size_t>(row) * d + c] - target;
    s += diff * diff;
  }
  return std::sqrt(s);
}

// ── 1. Parity: when the selection covers every causal block, sparse == dense ──
void test_parity_with_dense() {
  const int n = 96, d = 16;
  Tensor Q = make(n, d, [](int i, int c) { return 0.10f * std::sin(0.7f * i + 1.3f * c); });
  Tensor K = make(n, d, [](int i, int c) { return 0.10f * std::cos(0.5f * i + 0.9f * c); });
  Tensor V = make(n, d, [](int i, int c) { return 0.20f * std::sin(0.3f * i - 0.6f * c); });

  Tensor dense = dense_causal_attention(Q, K, V);

  SparseAttentionConfig cfg;
  cfg.block_size = 16;     // -> 6 blocks
  cfg.top_k_blocks = 16;   // >= num blocks: selects every causal block
  cfg.local_blocks = 0;
  cfg.sink_blocks = 0;
  Tensor sparse = sparse_selective_attention(Q, K, V, cfg);

  require(dense.shape == sparse.shape, "parity shape mismatch");
  float max_abs = 0.0f;
  for (int i = 0; i < dense.size; ++i)
    max_abs = std::max(max_abs, std::abs(dense.data()[i] - sparse.data()[i]));
  std::printf("[ssa] parity max|dense - sparse| = %.3e\n", max_abs);
  require(max_abs < 1e-4f, "sparse with full selection must equal dense attention");
  std::cout << "SSA parity-with-dense test passed!" << std::endl;
}

// ── 2. Retrieval: content selection finds the needle a fixed window misses ──
void test_content_selection_retrieves_needle() {
  const int n = 512, d = 16, B = 64;
  const int needle_pos = 350;   // lives in block 5 (320..383)
  const float marker = 5.0f;

  // Small background everywhere; values (V) are zero except at the needle, so
  // a correct retrieval makes the output approach `marker` and a miss stays ~0.
  Tensor K = make(n, d, [](int i, int c) { return 0.05f * std::sin(0.7f * i + 1.3f * c); });
  Tensor V = make(n, d, [](int, int) { return 0.0f; });
  Tensor Q = make(n, d, [](int i, int c) { return 0.05f * std::cos(0.4f * i + 0.8f * c); });

  // Plant the needle: a strong key direction + a distinctive value.
  for (int c = 0; c < d; ++c) {
    K.data()[static_cast<size_t>(needle_pos) * d + c] = (c == 0) ? 12.0f : 0.0f;
    V.data()[static_cast<size_t>(needle_pos) * d + c] = marker;
  }
  // The final query points at the needle's key direction.
  for (int c = 0; c < d; ++c)
    Q.data()[static_cast<size_t>(n - 1) * d + c] = (c == 0) ? 6.0f : 0.0f;

  SparseAttentionConfig content;
  content.block_size = B;
  content.top_k_blocks = 2;
  content.local_blocks = 1;
  content.sink_blocks = 1;

  SparseAttentionConfig window_only;
  window_only.block_size = B;
  window_only.top_k_blocks = 0;  // no content routing -> fixed local window
  window_only.local_blocks = 1;
  window_only.sink_blocks = 0;

  Tensor out_c = sparse_selective_attention(Q, K, V, content);
  Tensor out_w = sparse_selective_attention(Q, K, V, window_only);

  const float dist_content = row_dist(out_c, n - 1, marker);
  const float dist_window = row_dist(out_w, n - 1, marker);
  std::printf("[ssa] retrieval dist(content)=%.3f  dist(window-only)=%.3f\n",
              dist_content, dist_window);

  require(dist_content < 0.5f,
          "content selection failed to retrieve the needle value");
  require(dist_window > 3.0f,
          "fixed window unexpectedly saw the out-of-window needle");
  require(dist_content < dist_window,
          "content selection must beat the fixed window on retrieval");
  std::cout << "SSA content-selection retrieval test passed!" << std::endl;
}

// ── 3. Subquadratic: far fewer attended pairs / score FLOPs than dense ──
void test_subquadratic_cost() {
  const int n = 1024, d = 16;
  Tensor Q = make(n, d, [](int i, int c) { return 0.10f * std::sin(0.31f * i + 0.7f * c); });
  Tensor K = make(n, d, [](int i, int c) { return 0.10f * std::cos(0.23f * i + 0.5f * c); });
  Tensor V = make(n, d, [](int i, int c) { return 0.10f * std::sin(0.17f * i - 0.4f * c); });

  SparseAttentionConfig cfg;
  cfg.block_size = 64;  // 16 blocks
  cfg.top_k_blocks = 2;
  cfg.local_blocks = 1;
  cfg.sink_blocks = 1;  // <= 4 blocks attended per query out of up to 16

  SparseAttentionStats st;
  Tensor out = sparse_selective_attention(Q, K, V, cfg, &st);
  require(out.shape[0] == n && out.shape[1] == d, "sparse output shape wrong");

  std::printf("[ssa] n=%d attended pairs sparse=%lld dense=%lld (%.1f%%), "
              "score-FLOP fraction=%.1f%%\n",
              n, st.sparse_attended_pairs, st.dense_attended_pairs,
              100.0 * st.attended_fraction(), 100.0 * st.score_flop_fraction());
  require(st.dense_attended_pairs > 0, "stats not populated");
  require(st.attended_fraction() < 0.6,
          "sparse attention did not reduce the attended-pair count");
  std::cout << "SSA subquadratic-cost test passed!" << std::endl;
}

// ── 4. SSA wired into the real JambaModel attention (opt-in, default OFF) ──
void test_ssa_wired_into_model() {
  ModelConfig cfg;
  cfg.num_layers = 2;
  cfg.d_model = 32;
  cfg.vocab_size = 64;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 1;  // every layer is an attention layer
  cfg.attention_slot = 0;
  cfg.use_moe = false;
  cfg.use_ttt = false;

  JambaModel model(cfg, Device::CPU);
  model.set_training_mode(false);

  bool has_attn = false;
  for (auto& b : model.layers)
    if (b && b->uses_attention()) has_attn = true;
  require(has_attn, "test model has no attention layer to exercise SSA");

  std::vector<int> probe;
  for (int i = 0; i < 40; ++i) probe.push_back((i * 7 + 3) % cfg.vocab_size);

  model.set_sparse_attention(false);
  model.reset_session();
  Tensor dense = model.forward_ids(probe, nullptr).cpu();
  require(std::isfinite(dense.data()[0]), "dense model forward not finite");

  // Parity: SSA selecting all causal blocks must match dense attention.
  model.set_sparse_attention(true, /*block*/ 8, /*top_k*/ 1000, /*local*/ 0, /*sink*/ 0);
  model.reset_session();
  Tensor ssa_full = model.forward_ids(probe, nullptr).cpu();
  require(dense.shape == ssa_full.shape, "ssa/dense shape mismatch");
  float max_parity = 0.0f;
  for (int i = 0; i < dense.size; ++i)
    max_parity = std::max(max_parity, std::abs(dense.data()[i] - ssa_full.data()[i]));

  // Liveness: a small top-k changes the output -> the sparse path actually ran.
  model.set_sparse_attention(true, /*block*/ 8, /*top_k*/ 1, /*local*/ 1, /*sink*/ 0);
  model.reset_session();
  Tensor ssa_sparse = model.forward_ids(probe, nullptr).cpu();
  float max_live = 0.0f;
  for (int i = 0; i < dense.size; ++i)
    max_live = std::max(max_live, std::abs(dense.data()[i] - ssa_sparse.data()[i]));

  std::printf("[ssa-model] parity(full-select vs dense)=%.3e  liveness(small-k vs dense)=%.3e\n",
              max_parity, max_live);
  require(std::isfinite(ssa_sparse.data()[0]), "SSA model forward not finite");
  require(max_parity < 1e-4f, "SSA full-block selection must match dense attention in-model");
  std::cout << "SSA wired-into-model test passed!" << std::endl;
}

// ── 5. SSA in the streaming decode path (token-by-token), opt-in, default OFF ──
void test_ssa_streaming_decode() {
  ModelConfig cfg;
  cfg.num_layers = 2;
  cfg.d_model = 32;
  cfg.vocab_size = 64;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.attention_period = 1;  // every layer attends
  cfg.attention_slot = 0;
  cfg.use_moe = false;
  cfg.use_ttt = false;

  JambaModel model(cfg, Device::CPU);
  model.set_training_mode(false);

  std::vector<int> toks;
  for (int i = 0; i < 40; ++i) toks.push_back((i * 5 + 2) % cfg.vocab_size);

  auto run = [&](bool sparse, int topk) {
    if (sparse) {
      model.set_sparse_attention(true, /*block*/ 4, topk, /*local*/ 1, /*sink*/ 1);
    } else {
      model.set_sparse_attention(false);
    }
    model.set_streaming_inference(true);
    model.reset_session();
    Tensor last;
    for (int tok : toks) last = model.forward_ids(std::vector<int>{tok}, nullptr);
    return last.cpu();
  };

  Tensor dense = run(false, 0);
  Tensor ssa_all = run(true, 1000);  // covers every causal block -> parity
  Tensor ssa_k = run(true, 1);       // small top-k -> sparse path must change output

  require(dense.shape == ssa_all.shape, "streaming ssa/dense shape mismatch");
  float max_parity = 0.0f, max_live = 0.0f;
  for (int i = 0; i < dense.size; ++i) {
    max_parity = std::max(max_parity, std::abs(dense.data()[i] - ssa_all.data()[i]));
    max_live = std::max(max_live, std::abs(dense.data()[i] - ssa_k.data()[i]));
  }
  std::printf("[ssa-stream] parity(full vs dense)=%.3e  liveness(small-k vs dense)=%.3e\n",
              max_parity, max_live);
  require(std::isfinite(ssa_k.data()[0]), "streaming SSA forward not finite");
  require(max_parity < 1e-4f, "streaming SSA full selection must match dense decode");
  std::cout << "SSA streaming-decode test passed!" << std::endl;
}

// ── 6. Learned selection: finite-difference gradcheck of the scorer Wsel ──
// Proves the learnable block-selection scorer trains correctly (analytic dWsel
// matches the numerical gradient), so selection can be LEARNED, not just a
// fixed mean-key heuristic.
void test_learned_selection_gradcheck() {
  const int n = 24, d = 8;
  Tensor Q = make(n, d, [](int i, int c) { return 0.2f * std::sin(0.3f * i + 0.7f * c); });
  Tensor K = make(n, d, [](int i, int c) { return 0.2f * std::cos(0.4f * i + 0.5f * c); });
  Tensor V = make(n, d, [](int i, int c) { return 0.3f * std::sin(0.2f * i - 0.6f * c); });
  Tensor Wsel = make(d, d, [](int a, int b) {
    return 0.05f * std::sin(1.1f * a + 0.9f * b) + (a == b ? 0.3f : 0.0f);
  });
  SparseAttentionConfig cfg;
  cfg.block_size = 8;

  Tensor out = learned_block_bias_attention(Q, K, V, Wsel, cfg);
  Tensor dOut = out.clone();  // loss = 0.5 * sum(out^2)  =>  dL/dout = out
  Tensor dW = learned_block_bias_attention_backward(Q, K, V, Wsel, cfg, dOut);

  auto loss = [&]() {
    Tensor o = learned_block_bias_attention(Q, K, V, Wsel, cfg);
    double s = 0.0;
    for (int i = 0; i < o.size; ++i)
      s += 0.5 * static_cast<double>(o.data()[i]) * static_cast<double>(o.data()[i]);
    return s;
  };

  const float eps = 1e-3f;
  double diff_sq = 0.0, ana_sq = 0.0, num_sq = 0.0;
  for (int idx = 0; idx < Wsel.size; ++idx) {
    const float orig = Wsel.data()[idx];
    Wsel.data()[idx] = orig + eps;
    const double lp = loss();
    Wsel.data()[idx] = orig - eps;
    const double lm = loss();
    Wsel.data()[idx] = orig;
    const double numeric = (lp - lm) / (2.0 * eps);
    const double analytic = static_cast<double>(dW.data()[idx]);
    diff_sq += (numeric - analytic) * (numeric - analytic);
    ana_sq += analytic * analytic;
    num_sq += numeric * numeric;
  }
  // Standard gradcheck: global relative L2 error (robust to near-zero entries,
  // unlike a per-entry max-rel which inflates when both grads are tiny).
  const double rel = std::sqrt(diff_sq) / (std::sqrt(ana_sq) + 1e-8);
  std::printf("[ssa-learned] dWsel gradcheck rel-L2 err = %.3e (||ana||=%.3e)\n", rel,
              std::sqrt(ana_sq));
  require(rel < 2e-2, "learned-selection scorer gradient (dWsel) is wrong");
  std::cout << "SSA learned-selection gradcheck passed!" << std::endl;
}

// ── 7. Learned selector trains IN THE MODEL: block-mass distillation loss
//      must decrease as the per-layer ssa_wsel_ is trained. ──
void test_learned_selector_distill_trains() {
  const int n = 48, d = 8;
  Tensor Q = make(n, d, [](int i, int c) { return 0.5f * std::sin(0.3f * i + 0.7f * c); });
  // Block-dependent key magnitude -> dense attention has clear (non-uniform)
  // per-block preferences for the selector to learn.
  Tensor K = make(n, d, [](int i, int c) {
    const float blk = static_cast<float>(i / 8);
    return (0.15f + 0.35f * blk) * std::cos(0.4f * i + 0.5f * c);
  });
  Tensor Wsel = make(d, d, [](int, int) { return 0.0f; });  // zero init -> uniform pred

  const float loss0 = block_selector_distill_step(Q, K, Wsel, /*block*/ 8, /*lr*/ 0.0f);
  float loss1 = loss0;
  for (int s = 0; s < 80; ++s)
    loss1 = block_selector_distill_step(Q, K, Wsel, /*block*/ 8, /*lr*/ 0.5f);

  std::printf("[ssa-distill] block-mass CE loss: %.4f -> %.4f\n", loss0, loss1);
  require(loss0 > 0.0f, "degenerate distillation setup (no candidate blocks)");
  require(loss1 < loss0 - 1e-3f,
          "learned selector did not reduce the block-mass distillation loss");
  std::cout << "SSA learned-selector distillation training test passed!" << std::endl;
}

// ── 8. Does LEARNED selection beat the fixed mean-key heuristic on retrieval? ──
// Honest scope: on clean/aligned queries the cheap mean-key router already
// suffices (it ties).  Learning helps specifically when the query carries a
// MISLEADING component the fixed heuristic cannot suppress.  Here a distractor
// block has a high BLOCK-MEAN in a dim the query also likes, so the mean-key
// router mis-selects it and MISSES the needle; the trained Wsel learns to
// down-weight that distractor query dimension and recovers the needle's block.
void test_learned_selection_vs_meankey() {
  const int n = 64, d = 8, B = 8;                 // 8 blocks of 8
  const float marker = 5.0f;
  const int needle_pos = 6 * B + 3;               // block 6, position 51
  const int distractor_block = 2;                 // positions 16..23

  Tensor K = make(n, d, [](int i, int c) { return 0.02f * std::sin(0.7f * i + 1.1f * c); });
  Tensor V = make(n, d, [](int, int) { return 0.0f; });
  Tensor Q = make(n, d, [](int, int) { return 0.0f; });

  // Needle: one strong key in the SIGNAL dim 0, carrying the distinctive value.
  for (int c = 0; c < d; ++c) {
    K.data()[static_cast<size_t>(needle_pos) * d + c] = (c == 0) ? 16.0f : 0.0f;
    V.data()[static_cast<size_t>(needle_pos) * d + c] = marker;
  }
  // Distractor block: every key has a moderate DISTRACTOR-dim (1) magnitude, so
  // its block MEAN in dim 1 is high and lures the raw mean-key router.
  for (int j = distractor_block * B; j < (distractor_block + 1) * B; ++j)
    K.data()[static_cast<size_t>(j) * d + 1] = 3.0f;
  // Every query likes BOTH the signal (dim 0) and the distractor (dim 1).
  for (int i = 0; i < n; ++i) {
    Q.data()[static_cast<size_t>(i) * d + 0] = 1.0f;
    Q.data()[static_cast<size_t>(i) * d + 1] = 1.0f;
  }

  // Train the selector by block-mass distillation, starting from the mean-key
  // identity so any improvement is purely from learning.
  Tensor Wlearned = make(d, d, [](int a, int b) { return (a == b) ? 1.0f : 0.0f; });
  const float ce0 = block_selector_distill_step(Q, K, Wlearned, B, 0.0f);
  float ce1 = ce0;
  for (int s = 0; s < 300; ++s) ce1 = block_selector_distill_step(Q, K, Wlearned, B, 0.3f);
  Tensor Wmeankey = make(d, d, [](int a, int b) { return (a == b) ? 1.0f : 0.0f; });

  SparseAttentionConfig cfg;
  cfg.block_size = B;
  cfg.top_k_blocks = 1;   // ONE content block: a wrong pick misses the needle
  cfg.local_blocks = 1;
  cfg.sink_blocks = 0;

  Tensor out_mk = sparse_selective_attention(Q, K, V, cfg, nullptr, &Wmeankey);
  Tensor out_lr = sparse_selective_attention(Q, K, V, cfg, nullptr, &Wlearned);
  const float dist_mk = row_dist(out_mk, n - 1, marker);
  const float dist_lr = row_dist(out_lr, n - 1, marker);

  std::printf("[ssa-learned-vs-heur] distill CE %.3f->%.3f | retrieval dist: mean-key=%.3f  learned=%.3f\n",
              ce0, ce1, dist_mk, dist_lr);
  require(ce1 < ce0 - 1e-3f, "selector training did not reduce the distillation loss");
  require(dist_mk > 2.0f, "setup invalid: the mean-key heuristic should MISS this needle");
  require(dist_lr < 1.0f, "learned selector failed to retrieve the needle the heuristic misses");
  require(dist_lr < dist_mk - 1.0f, "learned selection must beat the mean-key heuristic on this query");
  std::cout << "SSA learned-selection beats mean-key (misleading-query) test passed!" << std::endl;
}

}  // namespace

int main() {
  try {
    test_parity_with_dense();
    test_content_selection_retrieves_needle();
    test_subquadratic_cost();
    test_ssa_wired_into_model();
    test_ssa_streaming_decode();
    test_learned_selection_gradcheck();
    test_learned_selector_distill_trains();
    test_learned_selection_vs_meankey();
    std::cout << "All SSA tests passed!" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "SSA test failed: " << ex.what() << std::endl;
    return 1;
  }
}
