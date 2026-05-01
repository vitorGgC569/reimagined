#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <fstream>

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}

    void build_grid(int side) {
        m = n * 4;
        row_ptr.resize(n + 1);
        col_ind.reserve(m);

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

void save_tour(const std::string& filename, const std::vector<int>& tour, int side) {
    std::ofstream out(filename);
    for (int u : tour) {
        int r = u / side;
        int c = u % side;
        out << c << " " << r << "\n"; // x, y
    }
    out.close();
}

int main() {
    int SIDE = 100; // 100x100 = 10k nodes
    int N = SIDE * SIDE;
    CSRGraph graph(N);
    graph.build_grid(SIDE);

    // 1. Spatial Tour
    std::vector<int> tour_spatial(N);
    std::iota(tour_spatial.begin(), tour_spatial.end(), 0);
    std::sort(tour_spatial.begin(), tour_spatial.end(), [&](int a, int b) {
        int ra = a / SIDE, ca = a % SIDE;
        int rb = b / SIDE, cb = b % SIDE;
        return (ra + ca) < (rb + cb); // Scanline diagonal
    });
    save_tour("tour_spatial.txt", tour_spatial, SIDE);

    // 2. Spectral Tour
    auto heat_map = ChebyshevEngine::compute_heat_map(graph, 0, 200);
    std::vector<int> tour_spectral(N);
    std::iota(tour_spectral.begin(), tour_spectral.end(), 0);
    std::sort(tour_spectral.begin(), tour_spectral.end(), [&](int a, int b) {
        return heat_map[a] > heat_map[b];
    });
    save_tour("tour_spectral.txt", tour_spectral, SIDE);

    // Also output heat map for AI training proof
    std::ofstream out_heat("heat_data.txt");
    for(int i=0; i<N; ++i) {
        int r = i / SIDE;
        int c = i % SIDE;
        out_heat << c << " " << r << " " << heat_map[i] << "\n";
    }
    out_heat.close();

    return 0;
}
