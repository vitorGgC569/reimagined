#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <chrono>
#include <numeric>
#include <tuple>
#include <omp.h>
#include <immintrin.h>

// ============================================================================
// KIMERA PAPER EDITION
// Configurable flags for ablation studies (No-Spectral, Scalar-Fallback)
// ============================================================================

struct Edge { int u, v; int w; };

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<int> values;

    CSRGraph(int n_nodes, const std::vector<Edge>& edges) : n(n_nodes) {
        std::vector<std::vector<std::pair<int, int>>> adj(n);
        m = 0;
        for (const auto& e : edges) {
            adj[e.u].push_back({e.v, e.w});
            m++;
            // Directed usually, but symmetric for Grid/Map benchmarks
            // We assume input handles symmetry if needed
        }
        row_ptr.resize(n + 1);
        col_ind.reserve(m);
        values.reserve(m);
        int current_idx = 0;
        for (int i = 0; i < n; ++i) {
            row_ptr[i] = current_idx;
            for (const auto& pair : adj[i]) {
                col_ind.push_back(pair.first);
                values.push_back(pair.second);
                current_idx++;
            }
        }
        row_ptr[n] = current_idx;
        m = current_idx;
    }
};

// CHEBYSHEV
class ChebyshevEngine {
public:
    static std::vector<float> compute_heat_map(const CSRGraph& graph, int source, int k_iterations) {
        int n = graph.n;
        if (k_iterations <= 0) return std::vector<float>(n, 0.0f); // Disabled

        std::vector<float> heat(n, 0.0f);
        std::vector<float> next_heat(n, 0.0f);
        std::vector<float> inv_deg(n, 0.0f);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int degree = graph.row_ptr[i+1] - graph.row_ptr[i];
            inv_deg[i] = (degree > 0) ? 1.0f / degree : 0.0f;
        }
        heat[source] = 1.0f;

        for (int iter = 0; iter < k_iterations; ++iter) {
            std::fill(next_heat.begin(), next_heat.end(), 0.0f);
            #pragma omp parallel for
            for (int u = 0; u < n; ++u) {
                if (heat[u] > 1e-9) {
                    float push_val = (0.5f * heat[u]) * inv_deg[u];
                    int start = graph.row_ptr[u];
                    int end = graph.row_ptr[u+1];
                    for (int idx = start; idx < end; ++idx) {
                        int v = graph.col_ind[idx];
                        #pragma omp atomic
                        next_heat[v] += push_val;
                    }
                    #pragma omp atomic
                    next_heat[u] += 0.5f * heat[u];
                }
            }
            std::swap(heat, next_heat);
        }
        return heat;
    }
};

// LAYOUT
class LayoutEngine {
public:
    std::vector<int> perm;
    std::vector<int> inv_perm;

    void compute_layout(const std::vector<float>& heat_map, bool active) {
        int n = heat_map.size();
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), 0);

        if (active) {
            std::sort(perm.begin(), perm.end(), [&](int a, int b) {
                return heat_map[a] > heat_map[b];
            });
        }

        inv_perm.resize(n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) inv_perm[perm[i]] = i;
    }

    CSRGraph reorder(const CSRGraph& graph) {
        int n = graph.n;
        std::vector<Edge> edges;
        edges.reserve(graph.m);
        for (int u = 0; u < n; ++u) {
            for(int idx = graph.row_ptr[u]; idx < graph.row_ptr[u+1]; ++idx){
                edges.push_back({inv_perm[u], inv_perm[graph.col_ind[idx]], graph.values[idx]});
            }
        }
        return CSRGraph(n, edges);
    }
};

// RADIX HEAP
class RadixHeap {
    static const int BUCKETS = 65;
    std::vector<std::vector<int>> buckets;
    unsigned long long last_dist;
public:
    RadixHeap() : buckets(BUCKETS), last_dist(0) {}
    void push(int u, unsigned long long dist) {
        if (dist < last_dist) dist = last_dist;
        unsigned long long xor_diff = dist ^ last_dist;
        int idx = (xor_diff == 0) ? 0 : (64 - __builtin_clzll(xor_diff));
        buckets[idx].push_back(u);
    }
    int pop(std::vector<unsigned long long>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) return -1;
            unsigned long long min_val = 18446744073709551615ULL;
            for (int u : buckets[i]) if (dist[u] < min_val) min_val = dist[u];
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

// CORE
class KimeraCore {
public:
    static std::tuple<long long, double> solve(const CSRGraph& graph, int source, bool use_avx) {
        int n = graph.n;
        unsigned long long INF = 9223372036854775807ULL; // LLONG_MAX
        std::vector<unsigned long long> dist(n, INF);
        RadixHeap pq;

        dist[source] = 0;
        pq.push(source, 0);

        long long operations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            unsigned long long du = dist[u];
            operations++;

            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];
            int idx = start;

            // AVX2 Path
            #ifdef __AVX2__
            if (use_avx) {
                int n_vec = (end - start) / 4;
                __m256i v_du = _mm256_set1_epi64x(du);
                for (int i = 0; i < n_vec; ++i) {
                    __m128i v_indices_small = _mm_loadu_si128((__m128i*)&graph.col_ind[idx]);
                    __m256i v_indices = _mm256_cvtepi32_epi64(v_indices_small);
                    __m128i v_weights_small = _mm_loadu_si128((__m128i*)&graph.values[idx]);
                    __m256i v_weights = _mm256_cvtepi32_epi64(v_weights_small);
                    __m256i v_new_dists = _mm256_add_epi64(v_du, v_weights);
                    __m256i v_curr_dists = _mm256_i32gather_epi64((long long int*)&dist[0], v_indices_small, 8);
                    __m256i v_mask = _mm256_cmpgt_epi64(v_curr_dists, v_new_dists);
                    int mask = _mm256_movemask_pd(_mm256_castsi256_pd(v_mask));
                    if (mask) {
                        long long* new_dists_arr = (long long*)&v_new_dists;
                        int* v_arr = (int*)&graph.col_ind[idx];
                        if (mask & 1) { int v = v_arr[0]; if (new_dists_arr[0] < dist[v]) { dist[v] = new_dists_arr[0]; pq.push(v, new_dists_arr[0]); } }
                        if (mask & 2) { int v = v_arr[1]; if (new_dists_arr[1] < dist[v]) { dist[v] = new_dists_arr[1]; pq.push(v, new_dists_arr[1]); } }
                        if (mask & 4) { int v = v_arr[2]; if (new_dists_arr[2] < dist[v]) { dist[v] = new_dists_arr[2]; pq.push(v, new_dists_arr[2]); } }
                        if (mask & 8) { int v = v_arr[3]; if (new_dists_arr[3] < dist[v]) { dist[v] = new_dists_arr[3]; pq.push(v, new_dists_arr[3]); } }
                    }
                    idx += 4;
                }
            }
            #endif

            // Scalar Path
            for (; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                int w = graph.values[idx];
                if (du + w < dist[v]) {
                    dist[v] = du + w;
                    pq.push(v, dist[v]);
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return {operations, ms};
    }
};

CSRGraph load_graph_from_file(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) exit(1);
    int n, m, source;
    infile >> n >> m >> source;
    std::vector<Edge> edges;
    edges.reserve(m);
    int u, v; double w;
    for (int i = 0; i < m; ++i) {
        infile >> u >> v >> w;
        edges.push_back({u, v, (int)w});
    }
    return CSRGraph(n, edges);
}

int main(int argc, char* argv[]) {
    // Usage: ./kimera_paper <file> <k_chebyshev> <use_avx:0|1>
    if (argc < 4) return 1;
    std::string filename = argv[1];
    int k_chebyshev = std::atoi(argv[2]);
    bool use_avx = std::atoi(argv[3]);

    CSRGraph graph = load_graph_from_file(filename);

    // Spectral
    auto t0 = std::chrono::high_resolution_clock::now();
    auto heat_map = ChebyshevEngine::compute_heat_map(graph, 0, k_chebyshev);

    // Layout (Only if heat map is valid/active)
    LayoutEngine layout;
    layout.compute_layout(heat_map, k_chebyshev > 0);
    CSRGraph graph_opt = layout.reorder(graph);

    // Core
    auto [ops, t_core] = KimeraCore::solve(graph_opt, layout.inv_perm[0], use_avx);
    auto t1 = std::chrono::high_resolution_clock::now();

    double t_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << t_total << " " << t_core << " " << ops << std::endl;
    return 0;
}
