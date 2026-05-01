#include "embedding.h"
#include "../include/rierass_core.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef USE_CUDA
#include "../include/cuda/kernels.cuh"
#include <cuda_runtime.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// MSVC compatibility for 128-bit types
#ifdef _MSC_VER
typedef uint64_t uint128_compat;
#else
typedef unsigned __int128 uint128_compat;
#endif

namespace nsos {

namespace {

#ifdef USE_CUDA
struct CudaIntBuffer {
  int* ptr = nullptr;

  explicit CudaIntBuffer(size_t count) {
    if (count == 0) {
      return;
    }
    if (cudaMalloc(&ptr, count * sizeof(int)) != cudaSuccess) {
      throw std::runtime_error("Embedding CUDA allocation failed");
    }
  }

  ~CudaIntBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  int* get() const { return ptr; }
};
#endif

std::vector<int> flatten_embedding_ids(const std::vector<std::vector<int>>& indices_batch,
                                       int batch_size,
                                       int seq_len) {
  std::vector<int> flat(static_cast<size_t>(batch_size * seq_len), -1);
  for (int batch = 0; batch < batch_size; ++batch) {
    const auto& indices = indices_batch[static_cast<size_t>(batch)];
    const int valid = std::min(seq_len, static_cast<int>(indices.size()));
    for (int i = 0; i < valid; ++i) {
      flat[static_cast<size_t>(batch * seq_len + i)] =
          indices[static_cast<size_t>(i)];
    }
  }
  return flat;
}

} // namespace

Embedding::Embedding(int vocab, int dim)
    : vocab_size(vocab), embedding_dim(dim),
      weight(Tensor::xavier_uniform({vocab, dim}), "embedding.weight") {
  // Sin table disabled for baseline stability
}

Tensor Embedding::forward(const std::vector<int> &indices) {
  if (indices.empty()) {
    return Tensor({0, embedding_dim}, weight.data.get_device());
  }
  Tensor batch = forward_batch({indices});
  return batch.reshape({static_cast<int>(indices.size()), embedding_dim});
}

Tensor Embedding::forward_batch(const std::vector<std::vector<int>>& indices_batch) {
  if (indices_batch.empty()) {
    return Tensor({0, 0, embedding_dim}, weight.data.get_device());
  }

  const int batch_size = static_cast<int>(indices_batch.size());
  int max_seq_len = 0;
  for (const auto& indices : indices_batch) {
    max_seq_len = std::max(max_seq_len, static_cast<int>(indices.size()));
  }
  if (max_seq_len == 0) {
    return Tensor({batch_size, 0, embedding_dim}, weight.data.get_device());
  }

#ifdef USE_CUDA
  if (weight.data.get_device() == Device::GPU) {
    Tensor out({batch_size, max_seq_len, embedding_dim}, Device::GPU);
    std::vector<int> flat_ids =
        flatten_embedding_ids(indices_batch, batch_size, max_seq_len);
    CudaIntBuffer d_ids(flat_ids.size());
    if (cudaMemcpy(d_ids.get(), flat_ids.data(), flat_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("Embedding GPU id upload failed");
    }
    launch_embedding_gather_kernel(out.data(), weight.data.data(), d_ids.get(),
                                   batch_size * max_seq_len, vocab_size,
                                   embedding_dim);
    return out;
  }
#endif

  Tensor weight_host = weight.data;
  Tensor out_host({batch_size, max_seq_len, embedding_dim}, Device::CPU);
  std::fill_n(out_host.data(), out_host.size, 0.0f);
  float *out_ptr = out_host.data();
  const float *w_ptr = weight_host.data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int batch = 0; batch < batch_size; ++batch) {
    const int seq_len = static_cast<int>(indices_batch[static_cast<size_t>(batch)].size());
    for (int i = 0; i < seq_len; ++i) {
      const int id = indices_batch[static_cast<size_t>(batch)][static_cast<size_t>(i)];
      float* dest = out_ptr + ((batch * max_seq_len) + i) * embedding_dim;
      if (id < 0 || id >= vocab_size) {
        std::fill_n(dest, embedding_dim, 0.0f);
        continue;
      }

      const float* w_row = w_ptr + id * embedding_dim;
      for (int d = 0; d < embedding_dim; ++d) {
        dest[d] = w_row[d];
      }
    }
  }

  return out_host;
}

void Embedding::backward(const Tensor &grad_output,
                         const std::vector<int> &indices) {
  if (indices.empty()) {
    return;
  }
  Tensor grad_rank3 =
      grad_output.shape.size() == 3
          ? grad_output
          : grad_output.reshape({1, static_cast<int>(indices.size()), embedding_dim});
  backward_batch(grad_rank3, {indices});
}

void Embedding::backward_batch(const Tensor& grad_output,
                               const std::vector<std::vector<int>>& indices_batch) {
  if (indices_batch.empty()) {
    return;
  }

  if (grad_output.shape.size() != 3 || grad_output.shape[0] != static_cast<int>(indices_batch.size()) ||
      grad_output.shape[2] != embedding_dim) {
    throw std::runtime_error("Embedding::backward_batch expects [batch, seq, dim] gradient");
  }

  const int batch_size = grad_output.shape[0];
  const int seq_len = grad_output.shape[1];
  for (const auto& indices : indices_batch) {
    if (static_cast<int>(indices.size()) > seq_len) {
      throw std::runtime_error(
          "Embedding::backward_batch sequence is longer than gradient sequence length");
    }
  }

#ifdef USE_CUDA
  if (weight.data.get_device() == Device::GPU) {
    Tensor grad_device =
        grad_output.get_device() == Device::GPU ? grad_output : grad_output.to(Device::GPU);
    Tensor d_w = Tensor::zeros(weight.data.shape, Device::GPU);
    std::vector<int> flat_ids =
        flatten_embedding_ids(indices_batch, batch_size, seq_len);
    CudaIntBuffer d_ids(flat_ids.size());
    if (cudaMemcpy(d_ids.get(), flat_ids.data(), flat_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("Embedding GPU id upload failed");
    }
    launch_embedding_scatter_add_kernel(d_w.data(), grad_device.data(), d_ids.get(),
                                        batch_size * seq_len, vocab_size,
                                        embedding_dim);
    weight.add_grad(d_w);
    return;
  }
#endif

  Tensor grad_host = grad_output;
  Tensor d_w_host = Tensor::zeros(weight.data.shape, Device::CPU);
  float *dw_ptr = d_w_host.data();
  const float *go_ptr = grad_host.data();

  for (int batch = 0; batch < batch_size; ++batch) {
    const int valid_len =
        std::min(seq_len, static_cast<int>(indices_batch[static_cast<size_t>(batch)].size()));
    for (int i = 0; i < valid_len; ++i) {
      const int id = indices_batch[static_cast<size_t>(batch)][static_cast<size_t>(i)];
      if (id < 0 || id >= vocab_size) {
        continue;
      }

      float *dest = dw_ptr + id * embedding_dim;
      const float *src = go_ptr + (batch * seq_len + i) * embedding_dim;
      for (int d = 0; d < embedding_dim; ++d) {
        dest[d] += src[d];
      }
    }
  }
  weight.add_grad(d_w_host);
}

void Embedding::to(Device dev) { weight.data = weight.data.to(dev); }

} // namespace nsos
