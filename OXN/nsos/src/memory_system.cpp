#include "memory_system.h"
#include <cmath>
#include <iostream>

namespace nsos {

MemorySystem::MemorySystem(int chunk_dim) : chunk_size(chunk_dim) {}

void MemorySystem::store_episodic(const Tensor &state) {
  std::lock_guard<std::mutex> lock(memory_mutex);

  // Tiered Clustering Storage (HNSW-Lite)
  float best_dist = 1e9;
  int best_cluster = -1;

  // Find best cluster
  for (size_t i = 0; i < clusters.size(); ++i) {
    // Euclidean distance to centroid (simplistic)
    float dist = 0;
    float *c = clusters[i].centroid.data();
    const float *s = state.data();
    for (int k = 0; k < state.size; ++k)
      dist += (c[k] - s[k]) * (c[k] - s[k]);

    if (dist < best_dist) {
      best_dist = dist;
      best_cluster = (int)i;
    }
  }

  // Threshold to create new cluster
  float CLUSTER_RADIUS = 10.0f; // Tunable
  if (best_cluster == -1 || best_dist > CLUSTER_RADIUS) {
    // Create new
    Cluster c;
    c.centroid = state.clone(); // Clone
    c.items.push_back(state.clone());
    clusters.push_back(c);
  } else {
    // Add to existing
    clusters[best_cluster].items.push_back(state.clone());
    // Update centroid (Moving Average)
    float alpha = 0.1f;
    float *c = clusters[best_cluster].centroid.data();
    const float *s = state.data();
    for (int k = 0; k < state.size; ++k)
      c[k] = (1 - alpha) * c[k] + alpha * s[k];
  }
}

void MemorySystem::add_instruction(const std::string &instr) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  instructional_memory.push_back(instr);
}

Tensor MemorySystem::retrieve(const Tensor &query) {
  std::lock_guard<std::mutex> lock(memory_mutex);

  // 2-Stage Retrieval
  // 1. Find Top-K Clusters
  std::vector<int> relevant_clusters;
  for (size_t i = 0; i < clusters.size(); ++i) {
    // Dot product with centroid
    float dot = 0;
    float *c = clusters[i].centroid.data();
    const float *q = query.data();
    for (int k = 0; k < query.size; ++k)
      dot += c[k] * q[k];

    if (dot > 0.5f)
      relevant_clusters.push_back((int)i); // Similarity threshold
  }

  // 2. Scan Items in Relevant Clusters
  Tensor context = Tensor::zeros(query.shape, query.get_device());
  float total_weight = 0;

  for (int idx : relevant_clusters) {
    for (const auto &mem : clusters[idx].items) {
      float dot = 0;
      for (int i = 0; i < mem.size; ++i)
        dot += query.data()[i] * mem.data()[i];

      float weight = std::exp(dot); // Softmax-ish

      // Accumulate
      for (int i = 0; i < mem.size; ++i)
        context.data()[i] += mem.data()[i] * weight;
      total_weight += weight;
    }
  }

  if (total_weight > 1e-6) {
    for (int i = 0; i < context.size; ++i)
      context.data()[i] /= total_weight;
  }

  return context;
}

} // namespace nsos
