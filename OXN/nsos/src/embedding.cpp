#include "embedding.h"
#include "../include/rierass_core.h"
#include <cmath>
#include <omp.h>

// MSVC compatibility for 128-bit types
#ifdef _MSC_VER
typedef uint64_t uint128_compat;
#else
typedef unsigned __int128 uint128_compat;
#endif

namespace nsos {

Embedding::Embedding(int vocab, int dim)
    : vocab_size(vocab), embedding_dim(dim),
      weight(Tensor::xavier_uniform({vocab, dim}), "embedding.weight") {
  // Sin table disabled for baseline stability
}

Tensor Embedding::forward(const std::vector<int> &indices) {
  int seq_len = (int)indices.size();
  Tensor out({seq_len, embedding_dim}, weight.data.get_device());

  float *out_ptr = out.data();
  const float *w_ptr = weight.data.data();

#pragma omp parallel for
  for (int i = 0; i < seq_len; ++i) {
    int id = indices[i];
    if (id < 0 || id >= vocab_size) {
      for (int d = 0; d < embedding_dim; ++d)
        out_ptr[i * embedding_dim + d] = 0;
      continue;
    }

    const float *w_row = w_ptr + id * embedding_dim;
    float *dest = out_ptr + i * embedding_dim;

    for (int d = 0; d < embedding_dim; ++d) {
      dest[d] = w_row[d];
    }
  }

  return out;
}

void Embedding::backward(const Tensor &grad_output,
                         const std::vector<int> &indices) {
  // grad_output: [Seq, Dim]
  // Accumulate grad into weight
  int seq_len = (int)indices.size();
  Tensor d_w = Tensor::zeros(weight.data.shape, weight.data.get_device());

  float *dw_ptr = d_w.data();
  const float *go_ptr = grad_output.data();

  for (int i = 0; i < seq_len; ++i) {
    int id = indices[i];
    if (id < 0 || id >= vocab_size)
      continue;

    float *dest = dw_ptr + id * embedding_dim;
    const float *src = go_ptr + i * embedding_dim;

#if !defined(_MSC_VER)
#pragma omp simd
#endif
    for (int d = 0; d < embedding_dim; ++d) {
      dest[d] += src[d];
    }
  }
  weight.add_grad(d_w);
}
void Embedding::to(Device dev) { weight.data = weight.data.to(dev); }

} // namespace nsos
