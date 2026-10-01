#ifndef RIERASS_CORE_H
#define RIERASS_CORE_H

#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

// MSVC doesn't support __int128, use uint64_t fallback
#ifdef _MSC_VER
typedef uint64_t uint128_t; // Reduced precision but compiles on MSVC
typedef int64_t int128_t;
#else
typedef __int128_t int128_t;
typedef __uint128_t uint128_t;
#endif

using namespace nsos; // Required for Tensor type

namespace rierass {

// --- Frente 3: 128-bit Precision Anchor ---
// Maps a numeric token ID to a "High-Resolution" vector signature.
// This creates "Galactic Distance" between neighboring integers (3 and 4).

inline void inject_anchor(Tensor &embedding, int token_id, int dim) {
  // Only apply to numeric range tokens (assumed 0-9 or logic tokens)
  // For simulation, we hash ANY token ID to a 128-bit seed.

  // 1. Generate 128-bit Isomorphic Signature
  // We use a mixing function similar to SplitMix64 but extended to 128.
  uint128_t z = (uint128_t)token_id * 0x9E3779B97F4A7C15ULL; // Golden Ratio
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);

  // 2. Project 128-bit space to Float Dimension (Orthogonalization)
  // We use the bits of 'z' to modulate high-frequency sine waves.
  // This ensures that even if token_id differs by 1, 'z' differs wildly
  // (Avalanche), and the resulting vector is nearly orthogonal.

  float *data = embedding.data();
  for (int d = 0; d < dim; ++d) {
    // Frequency derived from 128-bit chunks
    double freq = (double)((z >> (d % 64)) & 0xFF) / 255.0 * 3.14159 * (d + 1);

    // The Anchor: Highly non-linear, deterministic signal
    float anchor_val = std::sin(freq * (double)token_id);

    // Inject!
    // We Replace or Add?
    // "Anchor" implies a strong reference point. We ADD strongly.
    data[d] += anchor_val * 2.0f; // Strong signal
  }

  // Debug
  // if (token_id < 5) std::cout << "[RIERASS] Injected 128-bit Anchor for Token
  // " << token_id << std::endl;
}

// --- Frente 1: WDD Memory (Isomorphic Buffer) ---
// A linear scratchpad that enforces sequential access (Weighted Directional
// Drop). Prevents "Attention Leakage" by forcing the model to write thoughts in
// order.

class IsomorphicBuffer {
public:
  std::vector<Tensor> buffer;
  int write_head;
  int capacity;

  explicit IsomorphicBuffer(int cap = 16) : write_head(0), capacity(cap) {
    if (capacity <= 0) {
      throw std::invalid_argument("IsomorphicBuffer capacity must be positive");
    }
    buffer.reserve(static_cast<size_t>(capacity));
  }

  void write(const Tensor &thought) {
    if (write_head < capacity) {
      if (buffer.size() <= write_head)
        buffer.push_back(thought.clone());
      else
        buffer[write_head] = thought.clone();
      write_head++;
    } else {
      throw std::overflow_error(
          "IsomorphicBuffer capacity exhausted; thought was not stored");
    }
  }

  void reset() {
    write_head = 0;
    buffer.clear();
  }

  Tensor read_linear() {
    if (buffer.empty())
      return Tensor();
    Tensor sequence = get_sequence();
    return sequence.reshape({1, static_cast<int>(sequence.size)});
  }

  // Returns the full sequence as a Tensor [Seq, Dim]
  Tensor get_sequence() {
    if (buffer.empty())
      return Tensor();
    const Tensor &first = buffer.front();
    if (first.shape.size() != 2 || first.shape[0] != 1 || first.size <= 0) {
      throw std::invalid_argument(
          "IsomorphicBuffer expects every thought to have shape [1, D]");
    }
    const int seq = static_cast<int>(buffer.size());
    const int dim = static_cast<int>(first.size);
    Tensor result = Tensor::uninitialized({seq, dim}, first.get_device());
    for (int index = 0; index < seq; ++index) {
      const Tensor &thought = buffer[static_cast<size_t>(index)];
      if (thought.shape.size() != 2 || thought.shape[0] != 1 ||
          thought.size != dim || thought.get_device() != first.get_device()) {
        throw std::invalid_argument(
            "IsomorphicBuffer thoughts must share shape and device");
      }
      nsos::copy_tensor_bytes(
          result.raw_data() + static_cast<size_t>(index) * dim,
          result.get_device(), thought.raw_data(), thought.get_device(),
          static_cast<size_t>(dim) * sizeof(float));
    }
    return result;
  }
};

} // namespace rierass

#endif
