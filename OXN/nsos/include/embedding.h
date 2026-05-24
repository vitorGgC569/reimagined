#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "autograd.h"
#include "tensor.h"
#include <vector>

namespace nsos {

class Embedding {
public:
  int vocab_size;
  int embedding_dim;
  Parameter weight;

  // RIERASS Cache (Precomputed Sin/Cos table)
  // [embedding_dim, 256] ? No, it depends on TokenID AND Dim.
  // The previous logic was: sin(freq(hash(id, dim)) * id)
  // This is unique per (ID, Dim).
  // Cache size = Vocab * Dim. Too huge.
  // But logic: freq depends on (z >> (d % 64)). z depends on ID.
  // If we want speed, we can precompute the 128-bit hashes for Vocab?
  // z[vocab_size].
  // Yes, 128-bit int per token.
  struct uint128_pod {
    unsigned long long high;
    unsigned long long low;
  };
  std::vector<uint128_pod> anchor_hashes;

  // Optimization: Precomputed sine table for RIERASS
  // Size: vocab_size * embedding_dim
  std::vector<float> sin_table;

  Embedding(int vocab, int dim);

  // Look up embeddings for a batch of token indices
  // Input: [Batch, Seq] indices
  // Output: [Batch, Seq, Dim]
  Tensor forward(const std::vector<int> &indices);
  Tensor forward_batch(const std::vector<std::vector<int>>& indices_batch);

  // Backward pass
  // Accumulates gradient into grad_weight
  void backward(const Tensor &grad_output, const std::vector<int> &indices);
  void backward_batch(const Tensor& grad_output,
                      const std::vector<std::vector<int>>& indices_batch);
  void to(Device dev);
  std::vector<Parameter *> parameters() {
    weight.name = weight.base_name;
    return {&weight};
  }

  // ── Slender-Mamba head-to-toe quantization (Cherry-pick #2, Yu et al. 2025) ──
  // When enabled, applies 1.58-bit ternary quantization to the embedding weight
  // matrix + 8-bit activation quantization to the lookup output, following the
  // formulas in Section 3.3 of the Slender-Mamba paper (COLING 2025).
  //
  // The original BitNet b1.58 protocol only quantizes BitLinear layers in the
  // main body of the model, leaving embedding and projection layers in FP32 —
  // which represents ~46% of parameter bits in models like Mamba-2 170M.
  // Quantizing them head-to-toe yields ~90% parameter-bit reduction with
  // empirically zero average degradation on downstream tasks (Table 2 of paper).
  //
  // IMPORTANT: this is a TRAINING-TIME decision.  Models trained with
  // slender_quantization = false cannot be retroactively converted — they must
  // be retrained from scratch with the flag enabled.  See docs/SLENDER_INTEGRATION.md
  // for the full rationale, fórmulas, and rollout plan.
  //
  // Default: OFF (backward-compatible with all existing checkpoints).
  void set_slender_quantization(bool enabled) { slender_quantization_ = enabled; }
  bool slender_quantization_enabled() const { return slender_quantization_; }

 private:
  // Cherry-pick #2 state.  Default false preserves byte-for-byte the existing
  // FP32 embedding behavior — no risk of regression on the current pipeline.
  bool slender_quantization_ = false;

  // Cached ternary weight matrix.  Recomputed when slender_quantization_ is
  // true AND the underlying weight version changes (or on first use).  Avoids
  // re-quantizing the entire vocab × dim matrix on every forward pass when
  // weights are static between optimizer steps.
  //
  // Stored as a flat int8 vector { -1, 0, +1 } of length vocab_size * embedding_dim.
  // ~ 4900 × 640 = ~3MB for a typical 80M Mamba model.  Trivially fits any device.
  mutable std::vector<int8_t> slender_packed_weights_;
  mutable float slender_beta_ = 0.0f;
  mutable uint64_t slender_weight_version_ = 0;  // Bump when weights update
};

} // namespace nsos

#endif
