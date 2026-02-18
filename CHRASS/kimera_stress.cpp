#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <numeric>
#include <tuple>
#include <omp.h>
#include <random>
#include <immintrin.h>

// ============================================================================
// KIMERA STRESS TEST (INTERNAL GENERATORS)
// ============================================================================

struct CSRGraph {
    int n;
    size_t m; // Use size_t for massive graphs
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<int> values;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

// ============================================================================
// GENERATORS (Internal to avoid I/O)
// ============================================================================
class Generator {
public:
    static CSRGraph generate_chaos(int n, int avg_deg) {
        std::cout << "[Gen] Generating Chaos N=" << n << "..." << std::endl;
        CSRGraph g(n);
        size_t estimated_m = (size_t)n * avg_deg;
        g.row_ptr.resize(n + 1);
        g.col_ind.reserve(estimated_m);
        g.values.reserve(estimated_m);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist_node(0, n - 1);
        std::uniform_int_distribution<int> dist_w(1, 1000);

        int current_idx = 0;
        for (int u = 0; u < n; ++u) {
            g.row_ptr[u] = current_idx;
            for (int k = 0; k < avg_deg; ++k) {
                int v = dist_node(rng);
                int w = dist_w(rng);
                g.col_ind.push_back(v);
                g.values.push_back(w);
                current_idx++;
            }
        }
        g.row_ptr[n] = current_idx;
        g.m = current_idx;
        return g;
    }

    static CSRGraph generate_grid(int n) {
        int side = (int)std::sqrt(n);
        int actual_n = side * side;
        std::cout << "[Gen] Generating Grid " << side << "x" << side << " (N=" << actual_n << ")..." << std::endl;

        CSRGraph g(actual_n);
        g.row_ptr.resize(actual_n + 1);
        // Approx 4 edges per node
        g.col_ind.reserve(actual_n * 4);
        g.values.reserve(actual_n * 4);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist_w(1, 10000); // High weights

        int current_idx = 0;
        for (int r = 0; r < side; ++r) {
            for (int c = 0; c < side; ++c) {
                int u = r * side + c;
                g.row_ptr[u] = current_idx;

                // 4 Neighbors: Up, Down, Left, Right
                int dr[] = {-1, 1, 0, 0};
                int dc[] = {0, 0, -1, 1};

                for (int i = 0; i < 4; ++i) {
                    int nr = r + dr[i];
                    int nc = c + dc[i];
                    if (nr >= 0 && nr < side && nc >= 0 && nc < side) {
                        int v = nr * side + nc;
                        g.col_ind.push_back(v);
                        g.values.push_back(dist_w(rng));
                        current_idx++;
                    }
                }
            }
        }
        g.row_ptr[actual_n] = current_idx;
        g.m = current_idx;
        g.n = actual_n;
        return g;
    }

    static CSRGraph generate_snake(int n) {
        std::cout << "[Gen] Generating Snake (Worst Case Depth) N=" << n << "..." << std::endl;
        CSRGraph g(n);
        g.row_ptr.resize(n + 1);
        g.col_ind.reserve(n);
        g.values.reserve(n);

        int current_idx = 0;
        for (int u = 0; u < n; ++u) {
            g.row_ptr[u] = current_idx;
            if (u < n - 1) {
                g.col_ind.push_back(u + 1);
                g.values.push_back(1); // Unit weight to maximize hops
                current_idx++;
            }
        }
        g.row_ptr[n] = current_idx;
        g.m = current_idx;
        return g;
    }
};

// ============================================================================
// KIMERA COMPONENTS
// ============================================================================

class ChebyshevEngine {
public:
    static std::vector<float> compute_heat_map(const CSRGraph& graph, int source, int k_iterations = 10) {
        int n = graph.n;
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

class LayoutEngine {
public:
    std::vector<int> perm;
    std::vector<int> inv_perm;

    void compute_layout(const std::vector<float>& heat_map) {
        int n = heat_map.size();
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), 0);

        // Parallel sort not needed for 10M, but good habit.
        // std::sort is sequential.
        std::sort(perm.begin(), perm.end(), [&](int a, int b) {
            return heat_map[a] > heat_map[b];
        });

        inv_perm.resize(n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            inv_perm[perm[i]] = i;
        }
    }

    CSRGraph reorder(const CSRGraph& graph) {
        int n = graph.n;
        CSRGraph new_graph(n);
        new_graph.row_ptr.resize(n + 1);
        new_graph.col_ind.reserve(graph.m);
        new_graph.values.reserve(graph.m);

        int current_idx = 0;
        for (int new_u = 0; new_u < n; ++new_u) {
            new_graph.row_ptr[new_u] = current_idx;
            int old_u = perm[new_u];

            int start = graph.row_ptr[old_u];
            int end = graph.row_ptr[old_u+1];

            for (int idx = start; idx < end; ++idx) {
                int old_v = graph.col_ind[idx];
                int new_v = inv_perm[old_v];
                new_graph.col_ind.push_back(new_v);
                new_graph.values.push_back(graph.values[idx]);
                current_idx++;
            }
        }
        new_graph.row_ptr[n] = current_idx;
        new_graph.m = current_idx;
        return new_graph;
    }
};

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
            for (int u : buckets[i]) {
                if (dist[u] < min_val) min_val = dist[u];
            }
            last_dist = min_val;

            for (int u : buckets[i]) {
                unsigned long long d = dist[u];
                unsigned long long xor_diff = d ^ last_dist;
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

class KimeraCore {
public:
    static std::tuple<long long, double> solve(const CSRGraph& graph, int source) {
        int n = graph.n;
        // AVX2 signed compare safety
        unsigned long long INF = 9223372036854775807ULL;
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

            // Explicit AVX2 Vectorization
            int idx = start;
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

                    if (mask & 1) { int v = v_arr[0]; unsigned long long nd = new_dists_arr[0]; if (nd < dist[v]) { dist[v] = nd; pq.push(v, nd); } }
                    if (mask & 2) { int v = v_arr[1]; unsigned long long nd = new_dists_arr[1]; if (nd < dist[v]) { dist[v] = nd; pq.push(v, nd); } }
                    if (mask & 4) { int v = v_arr[2]; unsigned long long nd = new_dists_arr[2]; if (nd < dist[v]) { dist[v] = nd; pq.push(v, nd); } }
                    if (mask & 8) { int v = v_arr[3]; unsigned long long nd = new_dists_arr[3]; if (nd < dist[v]) { dist[v] = nd; pq.push(v, nd); } }
                }
                idx += 4;
            }

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

int main(int argc, char* argv[]) {
    int N = 1000000;
    std::string mode = "GRID";

    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) mode = argv[2];

    std::cout << "=== KIMERA STRESS TEST ===" << std::endl;
    std::cout << "Nodes: " << N << " | Mode: " << mode << std::endl;

    CSRGraph graph(0);
    if (mode == "GRID") graph = Generator::generate_grid(N);
    else if (mode == "CHAOS") graph = Generator::generate_chaos(N, 3);
    else if (mode == "SNAKE") graph = Generator::generate_snake(N);
    else {
        std::cerr << "Unknown mode" << std::endl;
        return 1;
    }

    int source = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. CHEBYSHEV
    std::cout << "[Step 1] Chebyshev Spectral Guidance..." << std::endl;
    auto t_step1_start = std::chrono::high_resolution_clock::now();
    auto heat_map = ChebyshevEngine::compute_heat_map(graph, source, 10);
    double t_step1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_step1_start).count();

    // 2. LAYOUT
    std::cout << "[Step 2] WDD Reordering..." << std::endl;
    auto t_step2_start = std::chrono::high_resolution_clock::now();
    LayoutEngine layout;
    layout.compute_layout(heat_map);
    CSRGraph graph_opt = layout.reorder(graph);
    int source_mapped = layout.inv_perm[source];
    double t_step2 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_step2_start).count();

    // 3. CORE
    std::cout << "[Step 3] Kimera Core (Radix Heap + AVX2)..." << std::endl;
    auto [ops, t_core] = KimeraCore::solve(graph_opt, source_mapped);

    auto t1 = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n>>> RESULTS <<<" << std::endl;
    std::cout << "Total Time: " << t_total << " ms" << std::endl;
    std::cout << "Step1(Chebyshev): " << t_step1 << " ms" << std::endl;
    std::cout << "Step2(Layout): " << t_step2 << " ms" << std::endl;
    std::cout << "Step3(Core): " << t_core << " ms" << std::endl;
    std::cout << "Relaxations: " << ops << std::endl;
    std::cout << "TEPS (Est): " << (double)ops * 1000.0 / t_core << std::endl;

    return 0;
}
