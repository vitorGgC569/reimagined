#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <tuple>
#include <fstream>

// ============================================================================
// DATA STRUCTURES: CSR GRAPH
// ============================================================================
struct Edge {
    int u, v;
    int w;
};

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values; // Double for physics compatibility

    CSRGraph(int n_nodes, const std::vector<Edge>& edges, bool symmetric) : n(n_nodes) {
        std::vector<std::vector<std::pair<int, int>>> adj(n);
        m = 0;
        for (const auto& e : edges) {
            adj[e.u].push_back({e.v, e.w});
            m++;
            if (symmetric) {
                adj[e.v].push_back({e.u, e.w});
                m++;
            }
        }

        row_ptr.resize(n + 1);
        col_ind.reserve(m);
        values.reserve(m);

        int current_idx = 0;
        for (int i = 0; i < n; ++i) {
            row_ptr[i] = current_idx;
            for (const auto& pair : adj[i]) {
                col_ind.push_back(pair.first);
                values.push_back((double)pair.second);
                current_idx++;
            }
        }
        row_ptr[n] = current_idx;
        m = current_idx; // Update m to actual stored edges
    }

    // Empty constructor
    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

// ============================================================================
// 1. PHYSICS ENGINE (THERMO)
// ============================================================================
class ThermoEngine {
public:
    static std::vector<double> compute_heat_map(const CSRGraph& graph, int source, int iterations = 15) {
        int n = graph.n;
        std::vector<double> heat(n, 0.0);
        std::vector<double> next_heat(n, 0.0);
        std::vector<double> inv_degrees(n, 0.0);

        // Precompute inverse degrees (D^-1)
        for (int i = 0; i < n; ++i) {
            int degree = graph.row_ptr[i+1] - graph.row_ptr[i];
            inv_degrees[i] = (degree == 0) ? 1.0 : 1.0 / degree;
        }

        heat[source] = 1.0;

        // Power Iteration: heat = 0.5 * heat + 0.5 * (D^-1 * A * heat)
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp parallel for schedule(static)
            for (int u = 0; u < n; ++u) {
                double neighbor_sum = 0.0;
                for (int idx = graph.row_ptr[u]; idx < graph.row_ptr[u+1]; ++idx) {
                    int v = graph.col_ind[idx];
                    // Diffusion depends on connection weight?
                    // Python prototype used: M = D^-1 * A.
                    // A is weighted in prototype. Let's stick to that for fidelity.
                    neighbor_sum += graph.values[idx] * heat[v];
                }
                // Update rule
                next_heat[u] = 0.5 * heat[u] + 0.5 * (inv_degrees[u] * neighbor_sum);
            }
            std::swap(heat, next_heat);
        }
        return heat;
    }
};

// ============================================================================
// 2. LAYOUT ENGINE (WDD)
// ============================================================================
class LayoutEngine {
public:
    std::vector<int> perm;
    std::vector<int> inv_perm;

    void compute_layout(const std::vector<double>& heat_map) {
        int n = heat_map.size();
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), 0);

        // Sort descending by heat
        std::sort(perm.begin(), perm.end(), [&](int a, int b) {
            return heat_map[a] > heat_map[b];
        });

        inv_perm.resize(n);
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

            // Access old neighbors
            int start = graph.row_ptr[old_u];
            int end = graph.row_ptr[old_u+1];

            for (int idx = start; idx < end; ++idx) {
                int old_v = graph.col_ind[idx];
                int new_v = inv_perm[old_v]; // Remap target
                double w = graph.values[idx];

                new_graph.col_ind.push_back(new_v);
                new_graph.values.push_back(w);
                current_idx++;
            }
        }
        new_graph.row_ptr[n] = current_idx;
        new_graph.m = current_idx;
        return new_graph;
    }
};

// ============================================================================
// 3. CORE (VECTORIZED SPFA)
// ============================================================================
class VectorizedCore {
public:
    static std::tuple<std::vector<double>, long long, double> solve(const CSRGraph& graph, int source, bool track_stats=false) {
        int n = graph.n;
        double inf = 1e14;
        std::vector<double> dist(n, inf);
        std::vector<int> active_nodes;
        std::vector<int> next_active_nodes;

        // Optimizations
        dist[source] = 0;
        active_nodes.reserve(n);
        next_active_nodes.reserve(n);
        active_nodes.push_back(source);

        std::vector<bool> in_queue(n, false);
        in_queue[source] = true;

        long long hops = 0;

        // Timer for core execution
        auto start = std::chrono::high_resolution_clock::now();

        while (!active_nodes.empty()) {
            hops++;
            next_active_nodes.clear();

            // 1. Gather & Relax
            // Vectorized loop potential here (compiler auto-vectorization)
            for (int u : active_nodes) {
                in_queue[u] = false; // Reset for next iteration logic
                double du = dist[u];

                int start = graph.row_ptr[u];
                int end = graph.row_ptr[u+1];

                // Hot loop
                for (int idx = start; idx < end; ++idx) {
                    int v = graph.col_ind[idx];
                    double w = graph.values[idx];
                    double new_dist = du + w;

                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        if (!in_queue[v]) {
                            next_active_nodes.push_back(v);
                            in_queue[v] = true;
                        }
                    }
                }
            }

            active_nodes = next_active_nodes; // Swap buffers
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        return {dist, hops, ms};
    }
};

// ============================================================================
// GENERATORS
// ============================================================================
std::vector<Edge> gen_random_chaos(int n, int avg_deg) {
    std::vector<Edge> edges;
    edges.reserve(n * avg_deg);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_node(0, n - 1);
    std::uniform_int_distribution<int> dist_w(1, 100);

    for (int i = 0; i < n * avg_deg; ++i) {
        int u = dist_node(rng);
        int v = dist_node(rng);
        if (u != v) edges.push_back({u, v, dist_w(rng)});
    }
    return edges;
}

std::vector<Edge> gen_grid_map(int n) {
    std::vector<Edge> edges;
    int side = (int)std::sqrt(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_w(1, 10);

    // Horizontal
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side - 1; ++c) {
            int u = r * side + c;
            int v = r * side + (c + 1);
            int w = dist_w(rng);
            edges.push_back({u, v, w});
        }
    }
    // Vertical
    for (int r = 0; r < side - 1; ++r) {
        for (int c = 0; c < side; ++c) {
            int u = r * side + c;
            int v = (r + 1) * side + c;
            int w = dist_w(rng);
            edges.push_back({u, v, w});
        }
    }
    return edges;
}

std::vector<Edge> gen_layered_dag(int n, int layers) {
    std::vector<Edge> edges;
    int nodes_per_layer = n / layers;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_w(1, 100);

    for (int l = 0; l < layers - 1; ++l) {
        int layer_start = l * nodes_per_layer;
        int next_start = (l + 1) * nodes_per_layer;
        std::uniform_int_distribution<int> dist_next(next_start, next_start + nodes_per_layer - 1);

        for (int i = 0; i < nodes_per_layer; ++i) {
            int u = layer_start + i;
            // 3 random connections
            for (int k = 0; k < 3; ++k) {
                int v = dist_next(rng);
                edges.push_back({u, v, dist_w(rng)});
            }
        }
    }
    return edges;
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================
void run_scenario(std::string name, int n, std::vector<Edge> (*gen_func)(int), int extra_param=0) {
    std::cout << "\n=== SCENARIO: " << name << " (N=" << n << ") ===" << std::endl;

    // 1. Generate
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Edge> edges = extra_param ? ((std::vector<Edge> (*)(int, int))gen_func)(n, extra_param) : gen_func(n);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "   [Gen Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms | Edges: " << edges.size() << "]" << std::endl;

    bool symmetric = (name.find("DAG") == std::string::npos); // DAG is directed
    CSRGraph graph(n, edges, symmetric);

    // 2. RAW V18 (C++)
    std::cout << "   Running V18 Raw (C++ Native)..." << std::endl;
    auto [d_raw, hops_raw, t_raw] = VectorizedCore::solve(graph, 0);
    std::cout << "   -> V18 Time: " << t_raw << " ms (Hops: " << hops_raw << ")" << std::endl;

    // 3. GOD V6 (C++)
    std::cout << "   Running God V6 (Thermo-WDD C++ Native)..." << std::endl;
    auto t_god_start = std::chrono::high_resolution_clock::now();

    // Phase 1: Physics
    auto t_phy_start = std::chrono::high_resolution_clock::now();
    auto heat_map = ThermoEngine::compute_heat_map(graph, 0, 15);
    double t_phy = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_phy_start).count();

    // Phase 2: WDD
    auto t_wdd_start = std::chrono::high_resolution_clock::now();
    LayoutEngine layout;
    layout.compute_layout(heat_map);
    CSRGraph graph_opt = layout.reorder(graph);
    int source_mapped = layout.inv_perm[0];
    double t_wdd = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_wdd_start).count();

    // Phase 3: Core
    auto [d_god, hops_god, t_core] = VectorizedCore::solve(graph_opt, source_mapped);

    double t_god_total = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_god_start).count();

    std::cout << "   -> God V6 Time: " << t_god_total << " ms" << std::endl;
    std::cout << "      (Phy: " << t_phy << " | WDD: " << t_wdd << " | Core: " << t_core << ")" << std::endl;

    if (t_god_total < t_raw) {
        std::cout << "   🏆 WINNER: GOD V6 (" << t_raw / t_god_total << "x faster)" << std::endl;
    } else {
        std::cout << "   🏆 WINNER: V18 RAW (" << t_god_total / t_raw << "x faster)" << std::endl;
    }
}

// ============================================================================
// FILE I/O FOR EXTERNAL BENCHMARKING
// ============================================================================
CSRGraph load_graph_from_file(const std::string& filename, bool symmetric) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    int n, m, source;
    infile >> n >> m >> source;

    std::vector<Edge> edges;
    edges.reserve(m);

    int u, v;
    double w;
    for (int i = 0; i < m; ++i) {
        infile >> u >> v >> w;
        edges.push_back({u, v, (int)w});
    }

    return CSRGraph(n, edges, symmetric);
}

int main(int argc, char* argv[]) {
    // Mode 1: Standalone (Old behavior)
    if (argc < 2) {
        int N = 1000000;
        run_scenario("CHAOS (Random)", N, (std::vector<Edge> (*)(int))gen_random_chaos, 3);
        run_scenario("MAP (2D Grid)", N, gen_grid_map, 0);
        run_scenario("AI MODEL (Layered DAG)", N, (std::vector<Edge> (*)(int))gen_layered_dag, 200);
        return 0;
    }

    // Mode 2: Controlled Benchmark (Load from file)
    // Usage: ./god_algo <graph_file> <mode:RAW|GOD>
    std::string filename = argv[1];
    std::string mode = (argc > 2) ? argv[2] : "RAW";

    // Detect symmetricity from filename (hacky but effective for script control)
    bool symmetric = (filename.find("DAG") == std::string::npos);

    CSRGraph graph = load_graph_from_file(filename, symmetric);
    int source = 0; // Assuming 0 for benchmark consistency

    auto t_start = std::chrono::high_resolution_clock::now();

    if (mode == "RAW") {
        auto [d, hops, t] = VectorizedCore::solve(graph, source);
        // Print only time in ms for parsing
        std::cout << t << std::endl;
    }
    else if (mode == "GOD") {
        // Full V6 Pipeline
        auto heat_map = ThermoEngine::compute_heat_map(graph, source, 15);
        LayoutEngine layout;
        layout.compute_layout(heat_map);
        CSRGraph graph_opt = layout.reorder(graph);
        int source_mapped = layout.inv_perm[source];

        auto [d, hops, t_core] = VectorizedCore::solve(graph_opt, source_mapped);

        // Total time includes physics + layout + core
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << total_ms << std::endl;
    }

    return 0;
}
