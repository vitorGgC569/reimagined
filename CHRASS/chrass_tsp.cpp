#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <iomanip>
#include <chrono>

// ============================================================================
// CHRASS TSP AI EXPERIMENT
// Testing if Spectral Heat Map provides a "Natural" TSP tour.
// ============================================================================

struct Edge { int u, v; int w; };

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<int> values;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}

    void build_grid(int side) {
        m = n * 4;
        row_ptr.resize(n + 1);
        col_ind.reserve(m);
        values.reserve(m);

        int idx = 0;
        for (int r = 0; r < side; ++r) {
            for (int c = 0; c < side; ++c) {
                int u = r * side + c;
                row_ptr[u] = idx;
                int dr[] = {-1, 1, 0, 0};
                int dc[] = {0, 0, -1, 1};
                for (int i = 0; i < 4; ++i) {
                    int nr = r + dr[i];
                    int nc = c + dc[i];
                    if (nr >= 0 && nr < side && nc >= 0 && nc < side) {
                        col_ind.push_back(nr * side + nc);
                        values.push_back(1); // Unit weight
                        idx++;
                    }
                }
            }
        }
        row_ptr[n] = idx;
        m = idx;
    }
};

class ChebyshevEngine {
public:
    static std::vector<float> compute_heat_map(const CSRGraph& graph, int source, int k_iterations) {
        int n = graph.n;
        std::vector<float> heat(n, 0.0f);
        std::vector<float> next_heat(n, 0.0f);
        std::vector<float> inv_deg(n, 0.0f);

        for (int i = 0; i < n; ++i) {
            int degree = graph.row_ptr[i+1] - graph.row_ptr[i];
            inv_deg[i] = (degree > 0) ? 1.0f / degree : 0.0f;
        }
        heat[source] = 1.0f;

        for (int iter = 0; iter < k_iterations; ++iter) {
            std::fill(next_heat.begin(), next_heat.end(), 0.0f);
            for (int u = 0; u < n; ++u) {
                if (heat[u] > 1e-9) {
                    float push_val = (0.5f * heat[u]) * inv_deg[u];
                    int start = graph.row_ptr[u];
                    int end = graph.row_ptr[u+1];
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

long long calculate_tour_cost(const std::vector<int>& tour, int side) {
    long long cost = 0;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        int u = tour[i];
        int v = tour[i+1];
        // Manhattan dist on grid
        int r1 = u / side, c1 = u % side;
        int r2 = v / side, c2 = v % side;
        cost += std::abs(r1 - r2) + std::abs(c1 - c2);
    }
    return cost;
}

int main() {
    int SIDE = 500;
    int N = SIDE * SIDE;
    std::cout << "=== CHRASS TSP AI EXPERIMENT ===" << std::endl;
    std::cout << "Problem: Visit " << N << " cities (Grid 500x500)." << std::endl;

    CSRGraph graph(N);
    graph.build_grid(SIDE);

    // 1. Random Tour
    std::vector<int> tour(N);
    std::iota(tour.begin(), tour.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(tour.begin(), tour.end(), rng);
    long long cost_random = calculate_tour_cost(tour, SIDE);
    std::cout << "Random Tour Cost: " << cost_random << " (Baseline)" << std::endl;

    // 2. Spatial Tour (Sort by X+Y, naive)
    std::sort(tour.begin(), tour.end(), [&](int a, int b) {
        int ra = a / SIDE, ca = a % SIDE;
        int rb = b / SIDE, cb = b % SIDE;
        return (ra + ca) < (rb + cb);
    });
    long long cost_spatial = calculate_tour_cost(tour, SIDE);
    std::cout << "Spatial Tour Cost (X+Y): " << cost_spatial << std::endl;

    // 3. Spectral Tour (Chebyshev Heat Map)
    // Heat diffuses from corner (0)
    auto heat_map = ChebyshevEngine::compute_heat_map(graph, 0, 500);

    std::vector<int> spectral_tour(N);
    std::iota(spectral_tour.begin(), spectral_tour.end(), 0);

    // Sort by Heat (Descending)
    // "Hot" nodes are topologically close to source.
    // Cooling down follows the diffusion gradient.
    std::sort(spectral_tour.begin(), spectral_tour.end(), [&](int a, int b) {
        return heat_map[a] > heat_map[b];
    });

    long long cost_spectral = calculate_tour_cost(spectral_tour, SIDE);
    std::cout << "Spectral Tour Cost (Chebyshev): " << cost_spectral << std::endl;

    double reduction = (double)(cost_random - cost_spectral) / cost_random * 100.0;
    std::cout << "\n>>> AI FEATURE QUALITY <<<" << std::endl;
    std::cout << "Spectral Reduction vs Random: " << std::fixed << std::setprecision(2) << reduction << "%" << std::endl;
    std::cout << "Spectral vs Spatial: " << (cost_spectral < cost_spatial ? "WIN" : "LOSE") << std::endl;

    return 0;
}
