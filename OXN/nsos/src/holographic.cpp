#include "../include/holographic.h"
#include <algorithm>
#include <cmath>
#include <cstring> // Added missing header
#include <iostream>
#include <limits>
#include <random>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

HolographicMemory::HolographicMemory(int dimension) : dim(dimension) {
  item_memory_matrix = Tensor::zeros({0, dim}); // Init empty
}

void HolographicMemory::to(Device dev) {
  item_memory_matrix = item_memory_matrix.to(dev);
  for (auto &pair : concept_memory) {
    pair.second = pair.second.to(dev);
  }
}

void HolographicMemory::add_concept(std::string name, const Tensor &vec) {
  if (capacity == 0) {
    capacity = 128; // Start with a reasonable buffer
    item_memory_matrix = Tensor::zeros({capacity, dim}, vec.get_device());
  }

  Tensor stored_vec =
      vec.get_device() == item_memory_matrix.get_device()
          ? vec
          : vec.to(item_memory_matrix.get_device());
  concept_memory[name] = stored_vec;
  concept_names.push_back(name);

  if (current_size >= capacity) {
    // Grow linearly or by powers of 2? Powers of 2 is standard for O(1)
    // amortized
    int new_capacity = capacity * 2;
    Tensor new_matrix({new_capacity, dim}, item_memory_matrix.get_device());

    // Copy existing data
    if (item_memory_matrix.get_device() == Device::CPU) {
      std::memcpy(new_matrix.data(), item_memory_matrix.data(),
                  (size_t)current_size * dim * sizeof(float));
    } else {
#ifdef USE_CUDA
      cudaMemcpy(new_matrix.data(), item_memory_matrix.data(),
                 (size_t)current_size * dim * sizeof(float),
                 cudaMemcpyDeviceToDevice);
#endif
    }

    item_memory_matrix = new_matrix;
    capacity = new_capacity;
  }

  // Insert at current_size offset
  if (stored_vec.get_device() == Device::CPU) {
    std::memcpy(item_memory_matrix.data() + (size_t)current_size * dim,
                stored_vec.data(), dim * sizeof(float));
  } else {
#ifdef USE_CUDA
    cudaMemcpy(item_memory_matrix.data() + (size_t)current_size * dim,
               stored_vec.data(), dim * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
  }

  current_size++;
}

Tensor HolographicMemory::create_concept(std::string name) {
  auto existing = concept_memory.find(name);
  if (existing != concept_memory.end()) {
    return existing->second;
  }

  // Create random bipolar hypervector (-1, 1)
  Tensor v =
      Tensor::random({dim}, Device::CPU); // Generate on CPU first for stability
  float *d = v.data();
  for (int i = 0; i < dim; ++i) {
    d[i] = (d[i] > 0.0f) ? 1.0f : -1.0f;
  }

  if (item_memory_matrix.get_device() == Device::GPU) {
    v = v.to(Device::GPU);
  }

  add_concept(name, v);
  return v;
}

Tensor HolographicMemory::bind(const Tensor &A, const Tensor &B) {
  // Binding: Element-wise Multiplication (Preserves magnitude)
  return A.mul(B);
}

Tensor HolographicMemory::bundle(const std::vector<Tensor> &vectors) {
  if (vectors.empty())
    return Tensor::zeros({dim});

  const Device device = vectors.front().get_device();
  Tensor result = Tensor::zeros({dim}, Device::CPU);
  float *res_ptr = result.data();

  for (const auto &v : vectors) {
    Tensor current = v.get_device() == Device::GPU ? v.cpu() : v;
    const float *v_ptr = current.data();
    for (int i = 0; i < dim; ++i)
      res_ptr[i] += v_ptr[i];
  }

  return result.to(device);
}

Tensor HolographicMemory::permute(const Tensor &A, int shifts) {
  // Rotation/Shift to encode order
  Tensor source = A.get_device() == Device::GPU ? A.cpu() : A;
  Tensor out = Tensor::zeros({dim}, Device::CPU);
  const float *src = source.data();
  float *dst = out.data();
  const int normalized_shift = ((shifts % dim) + dim) % dim;

  for (int i = 0; i < dim; ++i) {
    dst[(i + normalized_shift) % dim] = src[i];
  }
  return out.to(A.get_device());
}

Tensor
HolographicMemory::encode_sequence(const std::vector<std::string> &tokens) {
  // encode(t1, t2, t3) = Pi^3(t1) + Pi^2(t2) + Pi^1(t3)
  if (tokens.empty())
    return Tensor::zeros({dim});

  std::vector<Tensor> positioned_tokens;
  positioned_tokens.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string t = tokens[i];
    Tensor v;
    if (concept_memory.count(t)) {
      v = concept_memory[t];
    } else {
      v = create_concept(t);
    }

    int shifts = static_cast<int>(i);
    positioned_tokens.push_back(permute(v, shifts));
  }
  return bundle(positioned_tokens);
}

std::string HolographicMemory::query(const Tensor &query_vec) {
  if (current_size == 0)
    return "MEMORY_EMPTY";

  const Tensor query_cpu = query_vec.cpu();
  const Tensor memory_cpu = item_memory_matrix.cpu();
  const float *query_ptr = query_cpu.data();
  const float *memory_ptr = memory_cpu.data();

  float max_score = -std::numeric_limits<float>::infinity();
  int best_idx = -1;

  for (int i = 0; i < current_size; ++i) {
    float score = 0.0f;
    const float *concept_ptr = memory_ptr + static_cast<size_t>(i) * dim;
    for (int j = 0; j < dim; ++j) {
      score += concept_ptr[j] * query_ptr[j];
    }

    if (score > max_score) {
      max_score = score;
      best_idx = i;
    }
  }

  if (best_idx != -1)
    return concept_names[best_idx];
  return "NOT_FOUND";
}

Tensor HolographicMemory::retrieve_vector(const Tensor &query_vec, int top_k,
                                          float temperature) {
  if (current_size == 0)
    return Tensor::zeros({dim}, query_vec.get_device());

  Tensor query_cpu = query_vec.cpu();
  Tensor memory_cpu = item_memory_matrix.cpu();
  const float *query_ptr = query_cpu.data();
  const float *memory_ptr = memory_cpu.data();
  std::vector<std::pair<float, int>> ranked;
  for (int i = 0; i < current_size; ++i) {
    float score = 0.0f;
    const float *concept_ptr = memory_ptr + static_cast<size_t>(i) * dim;
    for (int j = 0; j < dim; ++j) {
      score += concept_ptr[j] * query_ptr[j];
    }
    ranked.push_back({score, i});
  }

  // 3. Find Top-K
  int k = std::min(top_k, current_size);
  std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                    std::greater<std::pair<float, int>>());

  // 4. Softmax Weights on CPU
  std::vector<float> weights(k);
  float sum_exp = 0;
  float max_score = ranked[0].first / temperature;

  for (int i = 0; i < k; ++i) {
    weights[i] = std::exp((ranked[i].first / temperature) - max_score);
    sum_exp += weights[i];
  }

  for (int i = 0; i < k; ++i) {
    weights[i] /= sum_exp;
  }

  Tensor result = Tensor::zeros({dim}, Device::CPU);
  float *result_ptr = result.data();
  for (int i = 0; i < k; ++i) {
    const float weight = weights[i];
    const float *concept_ptr =
        memory_ptr + static_cast<size_t>(ranked[i].second) * dim;
    for (int j = 0; j < dim; ++j) {
      result_ptr[j] += weight * concept_ptr[j];
    }
  }

  return result.to(query_vec.get_device());
}

Tensor HolographicMemory::clean(const Tensor &noisy) {
  // Thresholding back to -1, 1 improves robustness
  if (noisy.get_device() == Device::GPU) {
    // GPU tensors: Use sign function simulation or clamp logic
    // For now, let's keep robust on CPU and transfer if needed,
    // or implement a generic threshold kernel.
    return noisy.clamp(-1.0f, 1.0f); // Fast approximate cleaning
  }

  Tensor v = noisy.clone();
  float *d = v.data();
  for (int i = 0; i < v.size; ++i) {
    d[i] = (d[i] >= 0.0f) ? 1.0f : -1.0f;
  }
  return v;
}

} // namespace nsos
