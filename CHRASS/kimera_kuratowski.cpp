#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <fstream>
#include <chrono>
#include <numeric>

// ============================================================================
// KIMERA KURATOWSKI CHALLENGE
// Can we route K3,3 on a 2D Grid without crossings?
// ============================================================================

struct Edge { int u, v; int w; };

struct CSRGraph {
    int n;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<int> values;

    CSRGraph(int n_nodes) : n(n_nodes) {}

    void build_grid(int side) {
        int m_est = n * 4;
        row_ptr.resize(n + 1);
        col_ind.reserve(m_est);
        values.reserve(m_est);

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
                        values.push_back(1);
                        idx++;
                    }
                }
            }
        }
        row_ptr[n] = idx;
    }
};

// Simple Radix Heap for small grid
class RadixHeap {
    std::vector<std::vector<int>> buckets;
    int last_dist;
public:
    RadixHeap() : buckets(33), last_dist(0) {}
    void clear() { for(auto& b: buckets) b.clear(); last_dist = 0; }

    void push(int u, int dist) {
        if (dist < last_dist) dist = last_dist;
        int xor_diff = dist ^ last_dist;
        int idx = (xor_diff == 0) ? 0 : (32 - __builtin_clz(xor_diff));
        buckets[idx].push_back(u);
    }

    int pop(const std::vector<int>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < 33 && buckets[i].empty()) i++;
            if (i == 33) return -1;

            int min_val = 2147483647;
            for (int u : buckets[i]) if (dist[u] < min_val) min_val = dist[u];
            last_dist = min_val;

            for (int u : buckets[i]) {
                int idx = (dist[u] ^ last_dist) == 0 ? 0 : (32 - __builtin_clz(dist[u] ^ last_dist));
                buckets[idx].push_back(u);
            }
            buckets[i].clear();
        }
        int u = buckets[0].back();
        buckets[0].pop_back();
        return u;
    }
};

struct Net { int id; int s; int t; };

int main() {
    int SIDE = 100;
    int N = SIDE * SIDE;
    CSRGraph graph(N);
    graph.build_grid(SIDE);

    // K3,3 Setup
    // Houses: Top row (y=20)
    int H[] = {
        20 * SIDE + 20,
        20 * SIDE + 50,
        20 * SIDE + 80
    };
    // Services: Bottom row (y=80)
    int S[] = {
        80 * SIDE + 20,
        80 * SIDE + 50,
        80 * SIDE + 80
    };

    std::vector<Net> nets;
    int id = 0;
    for(int h=0; h<3; ++h) {
        for(int s=0; s<3; ++s) {
            nets.push_back({id++, H[h], S[s]});
        }
    }

    std::cout << "=== KIMERA KURATOWSKI CHALLENGE ===" << std::endl;
    std::cout << "Attempting to route K3,3 on planar grid..." << std::endl;

    std::mt19937 rng(42);
    int max_routed = 0;
    int iterations = 10000;

    std::vector<int> best_grid_map(N, -1); // Stores net_id for used nodes
    std::vector<bool> occupied(N, false);
    std::vector<int> dist(N);
    std::vector<int> parent(N);
    RadixHeap pq;

    for (int iter = 0; iter < iterations; ++iter) {
        // Shuffle order
        std::shuffle(nets.begin(), nets.end(), rng);

        // Reset Grid State
        std::fill(occupied.begin(), occupied.end(), false);
        // Mark Terminals as used (except for the net routing them)
        // Actually, terminals must be open for their specific net.

        int routed_this_iter = 0;
        std::vector<int> current_grid_map(N, -1);

        for (const auto& net : nets) {
            // SSSP
            std::fill(dist.begin(), dist.end(), 2147483647);
            pq.clear();

            // Start
            if (occupied[net.s]) continue; // Should effectively not happen if logic is right
            dist[net.s] = 0;
            parent[net.s] = -1;
            pq.push(net.s, 0);

            bool found = false;

            while(true) {
                int u = pq.pop(dist);
                if (u == -1) break;
                if (u == net.t) { found = true; break; }

                int d_u = dist[u];
                int start = graph.row_ptr[u];
                int end = graph.row_ptr[u+1];

                for(int idx = start; idx < end; ++idx) {
                    int v = graph.col_ind[idx];

                    // Strict Blocking: Can't use occupied nodes
                    // Exception: Can enter Target
                    if (occupied[v] && v != net.t) continue;

                    if (d_u + 1 < dist[v]) {
                        dist[v] = d_u + 1;
                        parent[v] = u;
                        pq.push(v, dist[v]);
                    }
                }
            }

            if (found) {
                routed_this_iter++;
                // Mark path
                int curr = net.t;
                while (curr != -1) {
                    occupied[curr] = true;
                    current_grid_map[curr] = net.id;
                    curr = parent[curr];
                }
            }
        }

        if (routed_this_iter > max_routed) {
            max_routed = routed_this_iter;
            best_grid_map = current_grid_map;
            std::cout << "Iter " << iter << ": New Max = " << max_routed << "/9" << std::endl;
            if (max_routed == 9) break;
        }
    }

    // Save Output
    std::cout << "Final Result: " << max_routed << "/9 connections." << std::endl;
    std::ofstream out("kuratowski_map.txt");
    out << SIDE << "\n";
    for(int i=0; i<N; ++i) out << best_grid_map[i] << " ";
    out.close();

    return 0;
}
