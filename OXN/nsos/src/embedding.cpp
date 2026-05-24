#include "embedding.h"
#include "../include/nsos_config.h"  // NSOS_DEFAULT_EPSILON for Slender quantization
#include "../include/rierass_core.h"
#include <algorithm>                  // std::max for Slender per-token reductions
#include <cmath>
#include <cstdint>                    // int8_t for Slender ternary weights
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

// ── Slender Phase 2: weight quantization cache ──────────────────────────────
// Lazy-builds the int8 ternary weight matrix W̃ + scalar β when (a) the cache
// is empty OR (b) the underlying weight.version has bumped (signal that the
// optimizer modified the weights since our last cache build).
//
// Implementation of equations 7 and 8 from Yu et al. 2025 Sec 3.3:
//   β = max( (1/(V·D)) · Σ |W_vd|, ε )                    (eq 8)
//   W̃ = clip(Round(W / β), -1, 1)                         (eq 7)
//
// The cache is keyed on weight.version (Parameter::version, bumped by
// Parameter::mark_updated() which the trainer calls after each optimizer
// step — see src/trainer.cpp).  No manual invalidation needed.
void Embedding::ensure_slender_cache_() const {
  // Cache hit: version matches and buffer is the right size
  const size_t expected = static_cast<size_t>(vocab_size) * static_cast<size_t>(embedding_dim);
  if (slender_cached_weight_version_ == weight.version &&
      slender_cached_weights_.size() == expected) {
    return;
  }

  // Cache miss: rebuild.  Weight must be on CPU for this pass.
  if (weight.data.get_device() != Device::CPU) {
    throw std::runtime_error(
        "Slender ensure_slender_cache_ requires weight.data on CPU.  "
        "Slender GPU path is Phase 7 (not yet implemented).  "
        "Either move weights to CPU or disable slender_quantization for now.");
  }

  const float* w_ptr = weight.data.data();

  // Equation 8: β = max( mean(|W|), ε ).  Single pass over the matrix.
  double sum_abs = 0.0;
  for (size_t i = 0; i < expected; ++i) {
    sum_abs += std::fabs(static_cast<double>(w_ptr[i]));
  }
  const double mean_abs = sum_abs / static_cast<double>(expected);
  const double beta = std::max(mean_abs, static_cast<double>(NSOS_DEFAULT_EPSILON));
  slender_cached_beta_ = static_cast<float>(beta);

  // Equation 7: W̃ = clip(round(W / β), -1, 1).  Result is strictly ternary.
  slender_cached_weights_.resize(expected);
  const float inv_beta = static_cast<float>(1.0 / beta);
  for (size_t i = 0; i < expected; ++i) {
    const float quantized = std::round(w_ptr[i] * inv_beta);
    // Clip to {-1, 0, +1}.  std::round returns nearest int (ties-to-even
    // varies by impl, but irrelevant here since we clip immediately).
    int8_t ternary;
    if (quantized >= 1.0f) {
      ternary = 1;
    } else if (quantized <= -1.0f) {
      ternary = -1;
    } else {
      ternary = 0;
    }
    slender_cached_weights_[i] = ternary;
  }

  slender_cached_weight_version_ = weight.version;
}

// ── Slender Phase 2: forward pass on CPU ───────────────────────────────────
// Implements equations 9-13 from Yu et al. 2025 Sec 3.3:
//   E = W̃[x]                                              (eq 9, lookup)
//   Ê = (E - μ(E)) / sqrt(Var(E) + ε)                     (eq 10, LayerNorm)
//   γ_i = max( max_j(|Ê_ij|), ε )                         (eq 11, per-token scale)
//   Ẽ_i = clip(round(Ê_i · Q_b / γ_i), -Q_b, Q_b - 1)    (eq 12, 8-bit quant)
//   E_out = (Ẽ ⊙ γ 1_D) · (β / Q_b)                       (eq 13, dequantize)
//
// Mathematical note on equation 10: the paper applies LayerNorm to E_real =
// E * β.  We can apply LN directly to E (without multiplying by β first)
// because LN is scale-invariant: (E·β - μ(E·β)) / sqrt(Var(E·β) + ε) ≈
// (E - μ(E)) / sqrt(Var(E) + ε) when ε is small compared to β²·Var(E).
// This saves one multiplication per element with negligible numerical impact.
//
// Output layout: [batch_size, max_seq_len, embedding_dim] on CPU.
// Out-of-range token IDs zero-fill (matches existing FP32 path semantics).
Tensor Embedding::slender_forward_cpu_(
    const std::vector<std::vector<int>>& indices_batch,
    int batch_size,
    int max_seq_len) const {
  ensure_slender_cache_();

  const int D = embedding_dim;
  const float beta = slender_cached_beta_;
  const int8_t* W_tilde = slender_cached_weights_.data();
  constexpr float kQb = 127.0f;
  constexpr float kEpsilon = NSOS_DEFAULT_EPSILON;

  Tensor out_host({batch_size, max_seq_len, D}, Device::CPU);
  std::fill_n(out_host.data(), out_host.size, 0.0f);
  float* out_ptr = out_host.data();

  // Per-token thread-local scratch buffers.  Allocating once per thread
  // avoids reallocating on every iteration of the inner loop.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<float> E_real(D);   // ternary lookup * β (for LN scale invariance, we keep E here)
    std::vector<float> E_hat(D);    // post-LN values

#ifdef _OPENMP
#pragma omp for
#endif
    for (int batch = 0; batch < batch_size; ++batch) {
      const auto& seq = indices_batch[static_cast<size_t>(batch)];
      const int seq_len = static_cast<int>(seq.size());

      for (int i = 0; i < seq_len; ++i) {
        const int id = seq[static_cast<size_t>(i)];
        float* dest = out_ptr + ((batch * max_seq_len) + i) * D;

        if (id < 0 || id >= vocab_size) {
          // Zero-fill matches FP32 path behavior for invalid IDs.
          std::fill_n(dest, D, 0.0f);
          continue;
        }

        // Step 1: lookup ternary row.  Keep E in float so subsequent LN
        // is simple; per-element load + cast from int8 is essentially free
        // compared to the LN reduction.
        const int8_t* w_row = W_tilde + static_cast<size_t>(id) * D;
        for (int d = 0; d < D; ++d) {
          E_real[d] = static_cast<float>(w_row[d]);
        }

        // Step 2: LayerNorm.  Compute μ and σ² over the D dimension.
        double sum = 0.0;
        for (int d = 0; d < D; ++d) sum += E_real[d];
        const float mean_E = static_cast<float>(sum / D);

        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
          const float diff = E_real[d] - mean_E;
          sum_sq += static_cast<double>(diff) * static_cast<double>(diff);
        }
        const float var_E = static_cast<float>(sum_sq / D);
        const float inv_norm = 1.0f / std::sqrt(var_E + kEpsilon);

        // Step 3: per-token max |Ê| (γ_i in the paper).  Accumulated in
        // the same pass that writes Ê.
        float gamma = kEpsilon;  // floor at ε to avoid division by zero
        for (int d = 0; d < D; ++d) {
          const float hat = (E_real[d] - mean_E) * inv_norm;
          E_hat[d] = hat;
          const float abs_hat = std::fabs(hat);
          if (abs_hat > gamma) gamma = abs_hat;
        }

        // Step 4: 8-bit quantize-dequantize.  Combined into one pass.
        //   Ẽ_d = clip(round(Ê_d · 127 / γ), -127, 127)        (int8)
        //   out_d = Ẽ_d · γ · (β / 127)                         (float)
        //
        // Algebraically, out_d = round(Ê_d · 127/γ) · γ · β/127, which is
        // a quantize/dequantize round-trip — introduces controlled error
        // matching what hardware inference would do at deploy time.
        const float quant_scale = kQb / gamma;
        const float dequant_scale = gamma * beta / kQb;

        for (int d = 0; d < D; ++d) {
          float q = std::round(E_hat[d] * quant_scale);
          // Clip to int8 range. Use 127 (not 128) on both sides for
          // symmetric quant range matching the paper.
          if (q > 127.0f) q = 127.0f;
          else if (q < -127.0f) q = -127.0f;
          dest[d] = q * dequant_scale;
        }
      }
    }
  }

  return out_host;
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

  // Cherry-pick #2 (Slender) Phase 2 dispatch.  When the slender flag is on,
  // route through the head-to-toe quantized forward.  GPU path for slender is
  // Phase 7 (not yet implemented); for now we require CPU weights when
  // slender is enabled.  This is enforced by ensure_slender_cache_().
  //
  // Empty-sequence shortcut below the dispatch still runs for both paths.
  if (slender_quantization_ && max_seq_len > 0) {
    return slender_forward_cpu_(indices_batch, batch_size, max_seq_len);
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
    launch_embedding_gather_kernel(out.raw_data(), weight.data.raw_data(),
                                   d_ids.get(),
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
  // Cherry-pick #2 (Slender) Phase 3 design note:
  //
  // The same backward code serves BOTH the FP32 path AND the Slender
  // quantized path via the straight-through estimator (STE; Bengio et
  // al. 2013).  Rationale:
  //
  // Slender forward computes y ≈ quant_dequant(W[id]) where the quant/
  // dequant pair is approximately identity (within quantization noise
  // bounded by β/127 per element).  Under STE, we treat the quantization
  // operations as identity for backpropagation:
  //
  //   ∂y/∂W ≈ ∂(W[id])/∂W = scatter at index id
  //
  // Therefore dL/dW = scatter_add(dL/dy, indices) — which is EXACTLY
  // what this FP32 backward implements below.  No Slender-specific
  // branching needed.  The paper itself (Yu et al. 2025 Sec 3.3, last
  // paragraph) confirms: "We employed a straight-through estimator
  // (Bengio et al., 2013) to approximate the gradient during
  // backpropagation."
  //
  // After the optimizer applies dL/dW via Parameter::add_grad + step(),
  // Trainer::step() calls Parameter::mark_updated() which bumps
  // weight.version, which signals our slender cache (in forward) to
  // re-quantize on the next forward call.  This keeps cache coherent
  // without manual invalidation.
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
    launch_embedding_scatter_add_kernel(d_w.raw_data(), grad_device.raw_data(),
                                        d_ids.get(),
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
