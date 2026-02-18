#include "../include/holographic.h"
#include <algorithm>
#include <cmath>
#include <cstring> // Added missing header
#include <iostream>
#include <random>

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
  concept_memory[name] = vec;
  concept_names.push_back(name);

  if (capacity == 0) {
    capacity = 128; // Start with a reasonable buffer
    item_memory_matrix = Tensor::zeros({capacity, dim}, vec.get_device());
  }

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
  if (vec.get_device() == Device::CPU) {
    std::memcpy(item_memory_matrix.data() + (size_t)current_size * dim,
                vec.data(), dim * sizeof(float));
  } else {
#ifdef USE_CUDA
    cudaMemcpy(item_memory_matrix.data() + (size_t)current_size * dim,
               vec.data(), dim * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
  }

  current_size++;
}

Tensor HolographicMemory::create_concept(std::string name) {
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

  Tensor result = Tensor::zeros({dim});
  float *res_ptr = result.data();

  for (const auto &v : vectors) {
    const float *v_ptr = v.data();
    for (int i = 0; i < dim; ++i)
      res_ptr[i] += v_ptr[i];
  }

  return clean(result);
}

Tensor HolographicMemory::permute(const Tensor &A, int shifts) {
  // Rotation/Shift to encode order
  Tensor out = Tensor::zeros({dim});
  const float *src = A.data();
  float *dst = out.data();

  for (int i = 0; i < dim; ++i) {
    dst[(i + shifts) % dim] = src[i];
  }
  return out;
}

Tensor
HolographicMemory::encode_sequence(const std::vector<std::string> &tokens) {
  // encode(t1, t2, t3) = Pi^2(t1) * Pi^1(t2) * Pi^0(t3)
  if (tokens.empty())
    return Tensor::zeros({dim});

  Tensor result = Tensor::ones({dim}); // Identity for multiplication/bind
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string t = tokens[i];
    Tensor v;
    if (concept_memory.count(t)) {
      v = concept_memory[t];
    } else {
      v = create_concept(t);
    }

    int shifts = (int)(tokens.size() - 1 - i);
    result = bind(result, permute(v, shifts));
  }
  return result;
}

std::string HolographicMemory::query(const Tensor &query_vec) {
  if (current_size == 0)
    return "MEMORY_EMPTY";

  // Retrieval: Find index of max cosine similarity
  // Similarity is proportional to Dot Product for bipolar vectors.

  // Use only the part of the matrix that contains concepts
  Tensor active_items = item_memory_matrix.slice(0, 0, current_size);
  Tensor scores = active_items.matmul(query_vec.reshape({-1, 1}));
  scores = scores.reshape({current_size});

  // Move scores to CPU for sorting (Top-1)
  Tensor scores_cpu = scores.cpu();
  const float *s_ptr = scores_cpu.data();
  float max_score = -1e9;
  int best_idx = -1;

  for (int i = 0; i < current_size; ++i) {
    if (s_ptr[i] > max_score) {
      max_score = s_ptr[i];
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

  // 1. Calculate scores (Dot Product) on Device
  Tensor active_items = item_memory_matrix.slice(0, 0, current_size);
  Tensor scores = active_items.matmul(query_vec.reshape({-1, 1}));
  scores = scores.reshape({current_size});

  // 2. Move scores to CPU for ranking
  Tensor scores_cpu = scores.cpu();
  const float *s_ptr = scores_cpu.data();
  std::vector<std::pair<float, int>> ranked;
  for (int i = 0; i < current_size; ++i) {
    ranked.push_back({s_ptr[i], i});
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

  // 5. Build a Weight Tensor on Device for fast Weighted Sum
  // Weight Tensor [1, current_size] initialized with zeros
  Tensor weights_tensor = Tensor::zeros({1, current_size}, Device::CPU);
  float *w_ptr = weights_tensor.data();
  for (int i = 0; i < k; ++i) {
    w_ptr[ranked[i].second] = weights[i];
  }

  // Move weights to the query device
  Tensor weights_dev = weights_tensor.to(query_vec.get_device());

  // 6. Weighted Sum via Matmul: [1, current_size] @ [current_size, dim] -> [1,
  // dim]
  Tensor result = weights_dev.matmul(active_items);
  return result.reshape({dim});
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
