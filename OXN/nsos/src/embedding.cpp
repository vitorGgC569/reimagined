#include "embedding.h"
#include "../include/nsos_config.h"  // NSOS_DEFAULT_EPSILON for Slender quantization
#include "../include/rierass_core.h"
#include "../include/nsos/determinism.h"  // deterministic_reductions_enabled()
#include <algorithm>                  // std::max for Slender per-token reductions
#include <cmath>
#include <cstdint>                    // int8_t for Slender ternary weights
#include <cstdlib>
#include <memory>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#ifdef USE_CUDA
#include "../include/cuda/device_buffer.h"
#include "../include/cuda/kernels.cuh"
#include "../include/cuda/pinned_buffer.h"
#include "../include/gpu_backend.h"
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

#ifdef USE_CUDA
struct EmbeddingGpuWorkspace {
  struct Slot {
    cuda_detail::DeviceBuffer<int> device_ids;
    cuda_detail::PinnedHostBuffer<int> pinned_ids;
    cuda_detail::DeviceBuffer<int> device_unique_ids;
    cuda_detail::PinnedHostBuffer<int> pinned_unique_ids;
    cuda_detail::DeviceBuffer<int> device_offsets;
    cuda_detail::PinnedHostBuffer<int> pinned_offsets;
    cuda_detail::DeviceBuffer<int> device_positions;
    cuda_detail::PinnedHostBuffer<int> pinned_positions;
    cudaEvent_t consumption_complete = nullptr;
    bool in_flight = false;
    bool awaiting_consumption_record = false;
    bool poisoned = false;
  };

  Slot slots[2];
  size_t next_slot = 0;

  ~EmbeddingGpuWorkspace() {
    for (auto& slot : slots) {
      bool completion_proven = !slot.poisoned;
      if (slot.awaiting_consumption_record) {
        // A launch exception between upload() and record_consumed() must never
        // permit the backing ID buffer to be freed while stream 0 may still
        // reference it.
        const cudaError_t status =
            cudaStreamSynchronize( nsos::gpu::current_stream());
        record_gpu_stream_synchronization();
        completion_proven =
            completion_proven && status == cudaSuccess;
      } else if (slot.in_flight && slot.consumption_complete != nullptr) {
        const cudaError_t status =
            cudaEventSynchronize(slot.consumption_complete);
        record_gpu_stream_synchronization();
        completion_proven =
            completion_proven && status == cudaSuccess;
      }
      if (!completion_proven) {
        slot.device_ids.abandon();
        slot.pinned_ids.abandon();
        slot.device_unique_ids.abandon();
        slot.pinned_unique_ids.abandon();
        slot.device_offsets.abandon();
        slot.pinned_offsets.abandon();
        slot.device_positions.abandon();
        slot.pinned_positions.abandon();
        slot.consumption_complete = nullptr;
        continue;
      }
      if (slot.consumption_complete != nullptr) {
        const cudaError_t status =
            cudaEventDestroy(slot.consumption_complete);
        if (status != cudaSuccess) {
          (void)cudaGetLastError();
        }
      }
    }
  }

  void reserve(Slot& slot, size_t requested_ids,
               size_t requested_unique_ids = 0,
               size_t requested_offsets = 0,
               size_t requested_positions = 0) {
    if (slot.poisoned) {
      throw std::runtime_error(
          "Embedding GPU staging is poisoned after an unrecoverable "
          "completion error");
    }
    const auto ready = [](size_t requested, size_t device_capacity,
                          size_t pinned_capacity, const int* device,
                          const int* pinned) {
      return requested == 0 ||
             (requested <= device_capacity && requested <= pinned_capacity &&
              device != nullptr && pinned != nullptr);
    };
    if (ready(requested_ids, slot.device_ids.capacity(),
              slot.pinned_ids.capacity(), slot.device_ids.get(),
              slot.pinned_ids.get()) &&
        ready(requested_unique_ids, slot.device_unique_ids.capacity(),
              slot.pinned_unique_ids.capacity(),
              slot.device_unique_ids.get(), slot.pinned_unique_ids.get()) &&
        ready(requested_offsets, slot.device_offsets.capacity(),
              slot.pinned_offsets.capacity(), slot.device_offsets.get(),
              slot.pinned_offsets.get()) &&
        ready(requested_positions, slot.device_positions.capacity(),
              slot.pinned_positions.capacity(), slot.device_positions.get(),
              slot.pinned_positions.get())) {
      return;
    }
    if (slot.awaiting_consumption_record) {
      throw std::logic_error(
          "Embedding GPU staging slot has an uncommitted consumer");
    }
    if (slot.in_flight) {
      const cudaError_t status =
          cudaEventSynchronize(slot.consumption_complete);
      if (status != cudaSuccess) {
        slot.poisoned = true;
        throw std::runtime_error(
            std::string("Embedding ID upload synchronization failed: ") +
            cudaGetErrorString(status));
      }
      slot.in_flight = false;
      record_gpu_stream_synchronization();
    }
    const auto ensure_pair = [](size_t requested,
                                cuda_detail::DeviceBuffer<int>& device,
                                cuda_detail::PinnedHostBuffer<int>& pinned,
                                const char* label) {
      if (requested == 0) {
        return;
      }
      const size_t capacity = std::max(requested, size_t{256});
      if (device.ensure(capacity) == nullptr) {
        throw std::runtime_error(
            std::string("Embedding GPU ") + label + " allocation failed");
      }
      if (pinned.ensure(capacity) == nullptr) {
        throw std::runtime_error(
            std::string("Embedding pinned ") + label +
            " allocation failed");
      }
    };
    ensure_pair(requested_ids, slot.device_ids, slot.pinned_ids, "ID");
    ensure_pair(requested_unique_ids, slot.device_unique_ids,
                slot.pinned_unique_ids, "unique-ID");
    ensure_pair(requested_offsets, slot.device_offsets,
                slot.pinned_offsets, "offset");
    ensure_pair(requested_positions, slot.device_positions,
                slot.pinned_positions, "position");
    if (slot.consumption_complete == nullptr) {
      const cudaError_t status =
          cudaEventCreate(&slot.consumption_complete);
      if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Embedding upload event creation failed: ") +
            cudaGetErrorString(status));
      }
    }
  }

  Slot& upload(const std::vector<int>& ids) {
    if (ids.empty()) {
      throw std::invalid_argument(
          "Embedding GPU staging cannot upload an empty ID vector");
    }
    Slot& slot = slots[next_slot];
    next_slot = (next_slot + 1) % 2;
    reserve(slot, ids.size());
    if (slot.awaiting_consumption_record) {
      throw std::logic_error(
          "Embedding GPU staging slot has an uncommitted consumer");
    }
    if (slot.in_flight) {
      cudaError_t wait_status =
          cudaEventQuery(slot.consumption_complete);
      if (wait_status == cudaErrorNotReady) {
        wait_status =
            cudaEventSynchronize(slot.consumption_complete);
        record_gpu_stream_synchronization();
      }
      if (wait_status != cudaSuccess) {
        slot.poisoned = true;
        throw std::runtime_error(
            std::string("Embedding staging reuse synchronization failed: ") +
            cudaGetErrorString(wait_status));
      }
      slot.in_flight = false;
    }
    std::copy(
        ids.begin(), ids.end(), slot.pinned_ids.get());
    cudaError_t status =
        cudaMemcpyAsync(slot.device_ids.get(), slot.pinned_ids.get(),
                        ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice, nsos::gpu::current_stream());
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("Embedding GPU ID upload failed: ") +
          cudaGetErrorString(status));
    }
    record_gpu_transfer(
        Device::GPU, Device::CPU, ids.size() * sizeof(int));
    slot.awaiting_consumption_record = true;
    return slot;
  }

  Slot& upload_sparse(const std::vector<int>& unique_ids,
                      const std::vector<int>& offsets,
                      const std::vector<int>& positions) {
    if (unique_ids.empty() || offsets.size() != unique_ids.size() + 1 ||
        positions.empty()) {
      throw std::invalid_argument(
          "Embedding sparse staging requires non-empty canonical CSR data");
    }
    Slot& slot = slots[next_slot];
    next_slot = (next_slot + 1) % 2;
    reserve(slot, 0, unique_ids.size(), offsets.size(), positions.size());
    if (slot.awaiting_consumption_record) {
      throw std::logic_error(
          "Embedding GPU staging slot has an uncommitted consumer");
    }
    if (slot.in_flight) {
      cudaError_t wait_status =
          cudaEventQuery(slot.consumption_complete);
      if (wait_status == cudaErrorNotReady) {
        wait_status = cudaEventSynchronize(slot.consumption_complete);
        record_gpu_stream_synchronization();
      }
      if (wait_status != cudaSuccess) {
        slot.poisoned = true;
        throw std::runtime_error(
            std::string("Embedding sparse staging reuse failed: ") +
            cudaGetErrorString(wait_status));
      }
      slot.in_flight = false;
    }

    std::copy(unique_ids.begin(), unique_ids.end(),
              slot.pinned_unique_ids.get());
    std::copy(offsets.begin(), offsets.end(), slot.pinned_offsets.get());
    std::copy(positions.begin(), positions.end(),
              slot.pinned_positions.get());

    const auto upload_array = [&](int* destination, const int* source,
                                  size_t count, const char* label) {
      const cudaError_t status = cudaMemcpyAsync(
          destination, source, count * sizeof(int), cudaMemcpyHostToDevice, nsos::gpu::current_stream());
      if (status != cudaSuccess) {
        if (slot.awaiting_consumption_record) {
          cancel_pending(slot);
        }
        throw std::runtime_error(
            std::string("Embedding sparse ") + label +
            " upload failed: " + cudaGetErrorString(status));
      }
      slot.awaiting_consumption_record = true;
      record_gpu_transfer(Device::GPU, Device::CPU,
                          count * sizeof(int));
    };
    upload_array(slot.device_unique_ids.get(),
                 slot.pinned_unique_ids.get(), unique_ids.size(),
                 "unique-ID");
    upload_array(slot.device_offsets.get(), slot.pinned_offsets.get(),
                 offsets.size(), "offset");
    upload_array(slot.device_positions.get(), slot.pinned_positions.get(),
                 positions.size(), "position");
    return slot;
  }

  void record_consumed(Slot& slot) {
    if (!slot.awaiting_consumption_record) {
      throw std::logic_error(
          "Embedding GPU staging consumer was already committed");
    }
    const cudaError_t status =
        cudaEventRecord(slot.consumption_complete, nsos::gpu::current_stream());
    if (status != cudaSuccess) {
      // Preserve memory safety even when event creation/recording fails.
      const cudaError_t sync_status =
          cudaStreamSynchronize( nsos::gpu::current_stream());
      record_gpu_stream_synchronization();
      if (sync_status != cudaSuccess) {
        slot.poisoned = true;
        throw std::runtime_error(
            std::string("Embedding consumer event record failed: ") +
            cudaGetErrorString(status) +
            "; completion recovery also failed: " +
            cudaGetErrorString(sync_status));
      }
      slot.awaiting_consumption_record = false;
      slot.in_flight = false;
      throw std::runtime_error(
          std::string("Embedding consumer event record failed: ") +
          cudaGetErrorString(status));
    }
    slot.awaiting_consumption_record = false;
    slot.in_flight = true;
  }

  void cancel_pending(Slot& slot) noexcept {
    if (!slot.awaiting_consumption_record) {
      return;
    }
    const cudaError_t status =
        cudaStreamSynchronize( nsos::gpu::current_stream());
    record_gpu_stream_synchronization();
    if (status != cudaSuccess) {
      slot.poisoned = true;
      return;
    }
    slot.awaiting_consumption_record = false;
    slot.in_flight = false;
  }
};
#endif

namespace {

std::vector<int> flatten_embedding_ids(const std::vector<std::vector<int>>& indices_batch,
                                       int batch_size,
                                       int seq_len) {
  if (batch_size < 0 || seq_len < 0 ||
      (seq_len != 0 &&
       batch_size > std::numeric_limits<int>::max() / seq_len)) {
    throw std::overflow_error(
        "Embedding flattened ID plane exceeds the supported range");
  }
  std::vector<int> flat(
      static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len), -1);
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

struct EmbeddingSparseMetadata {
  std::vector<int> unique_ids;
  std::vector<int> offsets;
  std::vector<int> positions;
};

EmbeddingSparseMetadata build_embedding_sparse_metadata(
    const std::vector<int>& flat_ids, int vocab_size) {
  if (flat_ids.size() >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Embedding sparse metadata exceeds 32-bit kernel indexing");
  }

  EmbeddingSparseMetadata metadata;
  metadata.unique_ids.reserve(
      std::min(flat_ids.size(), static_cast<size_t>(vocab_size)));
  std::vector<int> counts;
  counts.reserve(metadata.unique_ids.capacity());
  std::unordered_map<int, size_t> unique_index;
  unique_index.reserve(metadata.unique_ids.capacity());

  for (int token_id : flat_ids) {
    if (token_id < 0 || token_id >= vocab_size) {
      continue;
    }
    const auto [found, inserted] = unique_index.emplace(
        token_id, metadata.unique_ids.size());
    if (inserted) {
      metadata.unique_ids.push_back(token_id);
      counts.push_back(0);
    }
    ++counts[found->second];
  }

  metadata.offsets.resize(metadata.unique_ids.size() + 1, 0);
  for (size_t index = 0; index < counts.size(); ++index) {
    if (counts[index] < 0 ||
        metadata.offsets[index] >
            std::numeric_limits<int>::max() - counts[index]) {
      throw std::overflow_error(
          "Embedding sparse offsets exceed 32-bit kernel indexing");
    }
    metadata.offsets[index + 1] =
        metadata.offsets[index] + counts[index];
  }
  metadata.positions.resize(
      static_cast<size_t>(metadata.offsets.back()));
  std::vector<int> cursor = metadata.offsets;
  for (size_t position = 0; position < flat_ids.size(); ++position) {
    const int token_id = flat_ids[position];
    if (token_id < 0 || token_id >= vocab_size) {
      continue;
    }
    const auto found = unique_index.find(token_id);
    if (found == unique_index.end()) {
      throw std::logic_error(
          "Embedding sparse metadata lost a valid token ID");
    }
    const size_t index = found->second;
    metadata.positions[static_cast<size_t>(cursor[index]++)] =
        static_cast<int>(position);
  }
  for (size_t index = 0; index < counts.size(); ++index) {
    if (cursor[index] != metadata.offsets[index + 1]) {
      throw std::logic_error(
          "Embedding sparse metadata count mismatch");
    }
  }
  return metadata;
}

bool dense_deterministic_embedding_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("NSOS_EMBEDDING_DENSE_DETERMINISTIC");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

} // namespace

Embedding::Embedding(int vocab, int dim)
    : vocab_size(vocab), embedding_dim(dim),
      weight(Tensor::xavier_uniform({vocab, dim}), "weight") {
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

Tensor Embedding::forward_device_ids(const int* device_ids, int count) {
#ifdef USE_CUDA
  if (device_ids == nullptr || count <= 0) {
    throw std::runtime_error("Embedding::forward_device_ids: null/empty ids");
  }
  if (weight.data.get_device() != Device::GPU) {
    throw std::runtime_error(
        "Embedding::forward_device_ids requires GPU-resident weights");
  }
  // The gather kernel writes every output element, including explicit zeros
  // for invalid IDs, so a pre-launch full-buffer memset is redundant.
  Tensor out =
      Tensor::uninitialized({count, embedding_dim}, Device::GPU);
  launch_embedding_gather_kernel(out.raw_data(), weight.data.raw_data(),
                                 device_ids, count, vocab_size, embedding_dim);
  return out;
#else
  (void)device_ids;
  (void)count;
  throw std::runtime_error(
      "Embedding::forward_device_ids requires a GPU build");
#endif
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
    Tensor out = Tensor::uninitialized(
        {batch_size, max_seq_len, embedding_dim}, Device::GPU);
    std::vector<int> flat_ids =
        flatten_embedding_ids(indices_batch, batch_size, max_seq_len);
    if (!gpu_workspace_) {
      gpu_workspace_ = std::make_shared<EmbeddingGpuWorkspace>();
    }
    EmbeddingGpuWorkspace::Slot& slot =
        gpu_workspace_->upload(flat_ids);
    try {
      launch_embedding_gather_kernel(out.raw_data(), weight.data.raw_data(),
                                     slot.device_ids.get(),
                                     batch_size * max_seq_len, vocab_size,
                                     embedding_dim);
      gpu_workspace_->record_consumed(slot);
    } catch (...) {
      gpu_workspace_->cancel_pending(slot);
      throw;
    }
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
    if (!gpu_workspace_) {
      gpu_workspace_ = std::make_shared<EmbeddingGpuWorkspace>();
    }
    const bool deterministic =
        determinism::deterministic_reductions_enabled();
    if (deterministic &&
        !dense_deterministic_embedding_enabled()) {
      EmbeddingSparseMetadata metadata =
          build_embedding_sparse_metadata(flat_ids, vocab_size);
      if (metadata.unique_ids.empty()) {
        weight.add_grad(d_w);
        return;
      }
      EmbeddingGpuWorkspace::Slot& slot =
          gpu_workspace_->upload_sparse(
              metadata.unique_ids, metadata.offsets,
              metadata.positions);
      try {
        const bool launch_enqueued =
            launch_embedding_scatter_add_deterministic_sparse_kernel(
            d_w.raw_data(), grad_device.raw_data(),
            slot.device_unique_ids.get(), slot.device_offsets.get(),
            slot.device_positions.get(),
            static_cast<int>(metadata.unique_ids.size()),
            batch_size * seq_len, vocab_size, embedding_dim);
        if (!launch_enqueued) {
          throw std::runtime_error(
              "Deterministic sparse embedding backward rejected invalid "
              "arguments");
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
          throw std::runtime_error(
              std::string(
                  "Deterministic sparse embedding backward launch failed: ") +
              cudaGetErrorString(launch_status));
        }
        gpu_workspace_->record_consumed(slot);
      } catch (...) {
        gpu_workspace_->cancel_pending(slot);
        throw;
      }
    } else {
      EmbeddingGpuWorkspace::Slot& slot =
          gpu_workspace_->upload(flat_ids);
      try {
        if (deterministic) {
        launch_embedding_scatter_add_deterministic_kernel(
            d_w.raw_data(), grad_device.raw_data(), slot.device_ids.get(),
            batch_size * seq_len, vocab_size, embedding_dim);
        } else {
          launch_embedding_scatter_add_kernel(
              d_w.raw_data(), grad_device.raw_data(), slot.device_ids.get(),
              batch_size * seq_len, vocab_size, embedding_dim);
        }
        gpu_workspace_->record_consumed(slot);
      } catch (...) {
        gpu_workspace_->cancel_pending(slot);
        throw;
      }
    }
    weight.add_grad(d_w);
    return;
  }
#endif

  // Device-safe: copy a GPU grad to host so the ordered (deterministic) scatter
  // loop below can read it.  weight.add_grad(d_w_host) moves the result back to
  // the weight's device.
  if ((grad_output.get_device() == Device::GPU ||
       weight.data.get_device() == Device::GPU) &&
      strict_gpu_execution()) {
    throw std::runtime_error(
        "Strict GPU embedding backward cannot use the deterministic host "
        "scatter path");
  }
  Tensor grad_host =
      grad_output.get_device() == Device::GPU ? grad_output.cpu() : grad_output;
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
