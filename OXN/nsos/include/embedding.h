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

  // Backward pass
  // Accumulates gradient into grad_weight
  void backward(const Tensor &grad_output, const std::vector<int> &indices);
  void to(Device dev);
  std::vector<Parameter *> parameters() {
    weight.name = weight.base_name;
    return {&weight};
  }
};

} // namespace nsos

#endif
