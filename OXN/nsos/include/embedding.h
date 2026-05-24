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
  // ── Slender forward helpers (Phase 2 implementation) ──
  // Lazily quantizes the weight matrix to ternary {-1, 0, +1} when the
  // current weight.version differs from slender_cached_weight_version_.
  // Also computes the per-tensor scale β = max(mean(|W|), ε) and stores
  // it in slender_cached_beta_.  Cheap when the cache is hot (single
  // version comparison), O(V·D) when cold (full pass over the weight
  // matrix to build the cached ternary representation).
  //
  // PRECONDITION: weight.data must be on Device::CPU when this is called.
  // Slender on GPU is Phase 7 (future); for now we fall back to CPU when
  // the user opts into slender + GPU — see forward_batch() dispatch.
  void ensure_slender_cache_() const;

  // Computes a single Slender forward over the flat batched indices.
  // Performs: ternary lookup → LayerNorm → per-token 8-bit activation
  // quantization → dequantization back to float, exactly as described
  // in equations 7-13 of Yu et al. 2025 Sec 3.3.
  //
  // Output shape: [batch_size, max_seq_len, embedding_dim] on CPU.
  // Out-of-range token IDs (id < 0 or id >= vocab_size) zero their
  // row, matching the behavior of the existing FP32 path.
  Tensor slender_forward_cpu_(
      const std::vector<std::vector<int>>& indices_batch,
      int batch_size,
      int max_seq_len) const;

  // Cherry-pick #2 (Slender) state.  Default false preserves byte-for-byte the
  // existing FP32 embedding behavior — no risk of regression on the current
  // pipeline without explicit opt-in via set_slender_quantization(true).
  bool slender_quantization_ = false;

  // Cached ternary weight matrix.  Recomputed lazily when (a) the cache is
  // empty or (b) the underlying `weight.version` differs from the cached
  // `slender_cached_weight_version_`.  The version is bumped automatically
  // by Trainer::step() via Parameter::mark_updated() after each optimizer
  // update (see src/trainer.cpp:814,836), so the cache stays coherent
  // without manual invalidation calls.
  //
  // Layout: flat std::vector<int8_t> of length vocab_size * embedding_dim,
  // values strictly ∈ {-1, 0, +1}.  This is NOT bit-packed (that would be
  // ~vocab_dim/4 bytes); we keep it unpacked for fast direct lookup, since
  // total size is small: ~3 MB for 4900 × 640 = 3.1 M entries.  Bit-packing
  // is a future optimization once we measure that the linear scan dominates.
  //
  // β (slender_cached_beta_) is the per-tensor scale: max(mean(|W|), ε) per
  // equation 8 of the paper.  Cached to avoid recomputing on every forward.
  //
  // Thread-safety note: these mutable fields are NOT protected by a mutex.
  // The contract is that forward()/forward_batch() are not called concurrently
  // with weight updates from the optimizer.  This matches the existing
  // contract of Parameter::data — there is no concurrent read/write protection
  // anywhere else in the runtime.  Inference replicas use independent
  // Embedding instances (see InferenceEngine replica creation).
  mutable std::vector<int8_t> slender_cached_weights_;
  mutable float slender_cached_beta_ = 0.0f;
  mutable uint64_t slender_cached_weight_version_ = 0;
};

} // namespace nsos

#endif
