#ifndef NSOS_SPARSE_ATTENTION_H
#define NSOS_SPARSE_ATTENTION_H

#include "tensor.h"
#include <vector>

namespace nsos {

// ============================================================================
// SSA - Subquadratic Sparse Attention (content-dependent block selection)
// ============================================================================
// NSOS implementation of the SSA idea (subq.ai "how SSA makes long context
// practical").  Instead of comparing every query against every key (O(n^2)),
// each query is scored against cheap per-block key summaries (O(n/B) blocks),
// the top-k most relevant blocks are selected -- plus an always-on local
// window (recent blocks) and attention sinks (initial blocks) -- and EXACT
// attention is computed only over the selected blocks.
//
// Why this matters for NSOS: the Jamba hybrid leans on Mamba2 for long range,
// but an SSM's fixed-capacity state blurs/discards distant information (no
// exact retrieval).  Content-selected sparse attention keeps EXACT retrieval
// while cost grows with the number of selected positions (O(n*k)), not the
// full sequence -- the natural complement to the cheap Mamba stream.
//
// Single head, causal.  Multi-head wires by applying this per head.  The
// default block_size matches the Attention KV-cache page size (64).

struct SparseAttentionConfig {
  int block_size = 64;   // tokens per block (matches the KV-cache page size)
  int top_k_blocks = 4;  // content-selected blocks per query
  int local_blocks = 1;  // always-attend most-recent blocks (sliding window)
  int sink_blocks = 1;   // always-attend initial blocks (attention sinks)
  float scale = 0.0f;    // softmax scale; <= 0 means 1/sqrt(head_dim)
};

struct SparseAttentionStats {
  long long dense_attended_pairs = 0;   // sum_i (i+1)  -- full causal
  long long sparse_attended_pairs = 0;  // pairs actually scored in exact attn
  long long dense_score_flops = 0;      // ~ dense_pairs * 2 * d
  long long sparse_score_flops = 0;     // block scoring + exact attention
  double attended_fraction() const {
    return dense_attended_pairs > 0
               ? static_cast<double>(sparse_attended_pairs) /
                     static_cast<double>(dense_attended_pairs)
               : 0.0;
  }
  double score_flop_fraction() const {
    return dense_score_flops > 0
               ? static_cast<double>(sparse_score_flops) /
                     static_cast<double>(dense_score_flops)
               : 0.0;
  }
};

// Exact dense causal attention reference.  Q, K, V: [n, d] -> [n, d].
Tensor dense_causal_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                              float scale = 0.0f);

// Subquadratic sparse causal attention.  Q, K, V: [n, d] -> [n, d].
// When stats != nullptr it is filled with attended-pair / FLOP accounting.
// `Wsel` (optional, [d, d]): when provided, blocks are scored with the LEARNED
// routing query (Wsel @ q) instead of the raw q -- this is how the trained
// selector (see learned_block_bias_attention) drives hard top-k at inference.
Tensor sparse_selective_attention(const Tensor& Q, const Tensor& K,
                                  const Tensor& V,
                                  const SparseAttentionConfig& cfg,
                                  SparseAttentionStats* stats = nullptr,
                                  const Tensor* Wsel = nullptr);

// ── Learned block selection (training-time, differentiable) ──
// The fixed mean-key router above is training-free.  Here a LEARNABLE scorer
// `Wsel` ([d, d]) projects the query (q_sel = Wsel @ q); the block relevance
// (q_sel . block_mean) ADDITIVELY biases the attention logits, so the model
// LEARNS which blocks matter through the ordinary attention loss (gradients
// flow to Wsel).  At inference this same score drives the hard top-k selection
// of sparse_selective_attention -- i.e. this is how the router is trained.
// Forward + analytic backward (dWsel), validated by finite-difference gradcheck.
Tensor learned_block_bias_attention(const Tensor& Q, const Tensor& K,
                                    const Tensor& V, const Tensor& Wsel,
                                    const SparseAttentionConfig& cfg);

// Backward of learned_block_bias_attention w.r.t. the scorer Wsel.
// dOut is the upstream gradient ([n, d]); returns dWsel ([d, d]).
Tensor learned_block_bias_attention_backward(const Tensor& Q, const Tensor& K,
                                             const Tensor& V, const Tensor& Wsel,
                                             const SparseAttentionConfig& cfg,
                                             const Tensor& dOut);

// One training step of the block selector by block-MASS distillation: the
// scorer (Wsel) is trained so softmax over blocks of (Wsel @ q . block_mean)
// matches the DENSE attention's per-block mass (cross-entropy, exact grad
// pred-mass), then a self-contained SGD step updates Wsel in place.  Single
// head, causal; Q, K: [n, d], Wsel: [d, d].  Returns the CE loss (pre-update).
// This is the correct objective for LEARNED selection (vs the trivial
// output-matching).  Used by Attention's per-layer selector training.
float block_selector_distill_step(const Tensor& Q, const Tensor& K, Tensor& Wsel,
                                  int block_size, float lr);

}  // namespace nsos

#endif  // NSOS_SPARSE_ATTENTION_H
