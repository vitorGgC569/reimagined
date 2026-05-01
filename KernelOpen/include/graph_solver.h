#ifndef UHK_GRAPH_SOLVER_H
#define UHK_GRAPH_SOLVER_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// MSVC compatibility for __builtin_clzll
#ifdef _MSC_VER
#include <intrin.h>
inline int portable_clzll(unsigned long long x) {
  unsigned long index;
  if (_BitScanReverse64(&index, x)) {
    return 63 - (int)index;
  }
  return 64;
}
#define __builtin_clzll(x) portable_clzll(x)
#endif

namespace uhk {
namespace graph {

struct CSRGraph {
  int n;
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> values;
};

// --- 1. Chebyshev Spectral Guidance (Heat Map) ---
// Approximates heat diffusion to find topological centrality.
class ChebyshevEngine {
public:
  static std::vector<float> compute_heat_map(const CSRGraph &graph, int source,
                                             int k_iterations = 20) {
    int n = graph.n;
    std::vector<float> heat(n, 0.0f);
    std::vector<float> next_heat(n, 0.0f);
    std::vector<float> inv_deg(n, 0.0f);

    // Precompute Inverse Degrees
    for (int i = 0; i < n; ++i) {
      int degree = graph.row_ptr[i + 1] - graph.row_ptr[i];
      inv_deg[i] = (degree > 0) ? 1.0f / degree : 0.0f;
    }

    heat[source] = 1.0f;

    // Power Iteration (Simplified Chebyshev)
    // Simulate diffusion: Hot nodes are topologically close to source
    for (int iter = 0; iter < k_iterations; ++iter) {
      std::fill(next_heat.begin(), next_heat.end(), 0.0f);

      // Push-based diffusion (simulating symmetric flow for directed graph
      // heuristic)
      for (int u = 0; u < n; ++u) {
        if (heat[u] > 1e-9f) {
          float push_val = (0.5f * heat[u]) * inv_deg[u];
          int start = graph.row_ptr[u];
          int end = graph.row_ptr[u + 1];

          for (int idx = start; idx < end; ++idx) {
            int v = graph.col_ind[idx];
            next_heat[v] += push_val;
          }
          next_heat[u] += 0.5f * heat[u];
        }
      }
      std::swap(heat, next_heat);
    }
    return heat;
  }
};

// --- 2. Layout Engine (WDD) ---
// Reorders graph based on Heat Map to optimize cache locality.
class LayoutEngine {
public:
  std::vector<int> perm;
  std::vector<int> inv_perm;

  void compute_layout(const std::vector<float> &heat_map) {
    int n = heat_map.size();
    perm.resize(n);
    std::iota(perm.begin(), perm.end(), 0);

    // Sort: Hotter (Higher Value) first -> Cache Hot
    std::sort(perm.begin(), perm.end(),
              [&](int a, int b) { return heat_map[a] > heat_map[b]; });

    inv_perm.resize(n);
    for (int i = 0; i < n; ++i) {
      inv_perm[perm[i]] = i;
    }
  }

  CSRGraph reorder(const CSRGraph &graph) {
    int n = graph.n;
    CSRGraph new_graph;
    new_graph.n = n;
    new_graph.row_ptr.resize(n + 1);
    new_graph.col_ind.reserve(graph.col_ind.size());
    new_graph.values.reserve(graph.values.size());

    int current_idx = 0;
    for (int new_u = 0; new_u < n; ++new_u) {
      new_graph.row_ptr[new_u] = current_idx;
      int old_u = perm[new_u];

      int start = graph.row_ptr[old_u];
      int end = graph.row_ptr[old_u + 1];

      for (int idx = start; idx < end; ++idx) {
        int old_v = graph.col_ind[idx];
        int new_v = inv_perm[old_v]; // Remap target
        new_graph.col_ind.push_back(new_v);
        new_graph.values.push_back(graph.values[idx]);
        current_idx++;
      }
    }
    new_graph.row_ptr[n] = current_idx;
    return new_graph;
  }
};

// --- 3. Radix Heap ---
class RadixHeap {
  static const int BUCKETS = 65;
  std::vector<std::vector<int>> buckets;
  unsigned long long last_dist;

public:
  RadixHeap() : buckets(BUCKETS), last_dist(0) {}

  void push(int u, unsigned long long dist) {
    if (dist < last_dist)
      dist = last_dist;
    unsigned long long xor_diff = dist ^ last_dist;
    int idx = (xor_diff == 0) ? 0 : (64 - __builtin_clzll(xor_diff));
    buckets[idx].push_back(u);
  }

  int pop(std::vector<unsigned long long> &dist) {
    if (buckets[0].empty()) {
      int i = 1;
      while (i < BUCKETS && buckets[i].empty())
        i++;
      if (i == BUCKETS)
        return -1;
      unsigned long long min_val = 18446744073709551615ULL;
      for (int u : buckets[i])
        if (dist[u] < min_val)
          min_val = dist[u];
      last_dist = min_val;
      for (int u : buckets[i]) {
        unsigned long long xor_diff = dist[u] ^ last_dist;
        int idx = (xor_diff == 0) ? 0 : (64 - __builtin_clzll(xor_diff));
        buckets[idx].push_back(u);
      }
      buckets[i].clear();
    }
    int u = buckets[0].back();
    buckets[0].pop_back();
    return u;
  }
};

// --- 4. Main Solver (Kimera Core) ---
// Returns: Distances, Permutation used (to map back)
inline std::pair<std::vector<unsigned long long>, std::vector<int>>
solve_optimized(const CSRGraph &raw_graph, int source) {
  // A. Spectral Guidance
  auto heat_map = ChebyshevEngine::compute_heat_map(raw_graph, source, 10);

  // B. Layout Optimization
  LayoutEngine layout;
  layout.compute_layout(heat_map);
  CSRGraph graph = layout.reorder(raw_graph);
  int source_mapped = layout.inv_perm[source];

  // C. SSSP Core
  int n = graph.n;
  unsigned long long INF = 9223372036854775807ULL;
  std::vector<unsigned long long> dist(n, INF);
  RadixHeap pq;

  dist[source_mapped] = 0;
  pq.push(source_mapped, 0);

  while (true) {
    int u = pq.pop(dist);
    if (u == -1)
      break;

    unsigned long long du = dist[u];
    int start = graph.row_ptr[u];
    int end = graph.row_ptr[u + 1];
    int idx = start;

// AVX2 Optimization
#if defined(__AVX2__)
    int n_vec = (end - start) / 4;
    __m256i v_du = _mm256_set1_epi64x(du);
    for (int i = 0; i < n_vec; ++i) {
      __m128i v_idx = _mm_loadu_si128((__m128i *)&graph.col_ind[idx]);
      __m256i v_indices = _mm256_cvtepi32_epi64(v_idx);
      __m128i v_w = _mm_loadu_si128((__m128i *)&graph.values[idx]);
      __m256i v_weights = _mm256_cvtepi32_epi64(v_w);
      __m256i v_new = _mm256_add_epi64(v_du, v_weights);

      __m256i v_curr =
          _mm256_i32gather_epi64((long long int *)&dist[0], v_idx, 8);
      __m256i v_mask = _mm256_cmpgt_epi64(v_curr, v_new);

      int mask = _mm256_movemask_pd(_mm256_castsi256_pd(v_mask));
      if (mask) {
        long long *new_ptr = (long long *)&v_new;
        int *idx_ptr = (int *)&graph.col_ind[idx];
        if (mask & 1) {
          int v = idx_ptr[0];
          if (new_ptr[0] < dist[v]) {
            dist[v] = new_ptr[0];
            pq.push(v, new_ptr[0]);
          }
        }
        if (mask & 2) {
          int v = idx_ptr[1];
          if (new_ptr[1] < dist[v]) {
            dist[v] = new_ptr[1];
            pq.push(v, new_ptr[1]);
          }
        }
        if (mask & 4) {
          int v = idx_ptr[2];
          if (new_ptr[2] < dist[v]) {
            dist[v] = new_ptr[2];
            pq.push(v, new_ptr[2]);
          }
        }
        if (mask & 8) {
          int v = idx_ptr[3];
          if (new_ptr[3] < dist[v]) {
            dist[v] = new_ptr[3];
            pq.push(v, new_ptr[3]);
          }
        }
      }
      idx += 4;
    }
#endif

    for (; idx < end; ++idx) {
      int v = graph.col_ind[idx];
      int w = graph.values[idx];
      if (du + w < dist[v]) {
        dist[v] = du + w;
        pq.push(v, dist[v]);
      }
    }
  }

  // Remap distances back to original indices
  std::vector<unsigned long long> final_dist(n);
  for (int i = 0; i < n; ++i) {
    // dist[i] corresponds to node `perm[i]` in original graph
    // actually, node `u` in reordered graph is `perm[u]` in original
    // So dist[u] is distance to `perm[u]`.
    // We want final_dist[original_idx].
    // original_idx = perm[u].
    final_dist[layout.perm[i]] = dist[i];
  }

  return {final_dist, layout.perm};
}

// Legacy wrapper for compatibility
inline std::vector<unsigned long long> sssp(const CSRGraph &graph, int source) {
  auto res = solve_optimized(graph, source);
  return res.first;
}

} // namespace graph
} // namespace uhk

#endif
