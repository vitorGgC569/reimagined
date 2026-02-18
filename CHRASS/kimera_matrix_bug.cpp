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
// KIMERA MATRIX BUG CHALLENGE
// Forcing K3,3 on 2D Grid by paying the "Reality Distortion Penalty"
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
    // Houses (Top)
    int H[] = { 20 * SIDE + 20, 20 * SIDE + 50, 20 * SIDE + 80 };
    // Services (Bottom)
    int S[] = { 80 * SIDE + 20, 80 * SIDE + 50, 80 * SIDE + 80 };

    std::vector<Net> nets;
    int id = 0;
    for(int h=0; h<3; ++h) {
        for(int s=0; s<3; ++s) {
            nets.push_back({id++, H[h], S[s]});
        }
    }

    std::cout << "=== KIMERA MATRIX BUG CHALLENGE ===" << std::endl;
    std::cout << "Forcing 9/9 connections via Reality Distortion (Soft Congestion)..." << std::endl;

    std::vector<int> congestion(N, 0);
    std::vector<std::vector<int>> net_paths(nets.size());
    std::vector<int> dist(N);
    std::vector<int> parent(N);
    RadixHeap pq;

    // Rip-up and Reroute Loop
    for (int iter = 0; iter < 10; ++iter) {
        int routed_this_iter = 0;

        for (const auto& net : nets) {
            // Rip-up
            for (int u : net_paths[net.id]) {
                if (congestion[u] > 0) congestion[u]--;
            }
            net_paths[net.id].clear();

            // Reroute (SSSP)
            std::fill(dist.begin(), dist.end(), 2147483647);
            pq.clear();
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

                    // COST FUNCTION: The "Matrix Hack"
                    // Base cost 1.
                    // If congested, add Penalty 10,000.
                    // This allows crossing but makes it VERY expensive.
                    int w = 1 + (congestion[v] * 10000);

                    if (d_u + w < dist[v]) {
                        dist[v] = d_u + w;
                        parent[v] = u;
                        pq.push(v, dist[v]);
                    }
                }
            }

            if (found) {
                routed_this_iter++;
                int curr = net.t;
                while (curr != -1) {
                    congestion[curr]++;
                    net_paths[net.id].push_back(curr);
                    curr = parent[curr];
                }
            }
        }
        std::cout << "Iter " << iter << ": Routed " << routed_this_iter << "/9" << std::endl;
    }

    // Identify "Bug Pixels" (Nodes with congestion > 1)
    int bugs = 0;
    for(int i=0; i<N; ++i) if(congestion[i] > 1) bugs++;

    std::cout << "Found " << bugs << " bugs in the Matrix (Crossings)." << std::endl;

    // Save Output
    std::ofstream out("matrix_bug_map.txt");
    out << SIDE << "\n";

    // Create a map of "Who owns this pixel?"
    // If overlap, mark as BUG (-2)
    std::vector<int> final_map(N, -1);
    for (const auto& net : nets) {
        for (int u : net_paths[net.id]) {
            if (congestion[u] > 1) final_map[u] = -2; // Bug
            else final_map[u] = net.id;
        }
    }

    // Ensure terminals are visible
    for (const auto& net : nets) {
        final_map[net.s] = net.id; // House
        final_map[net.t] = net.id; // Service
    }

    for(int i=0; i<N; ++i) out << final_map[i] << " ";
    out.close();

    return 0;
}
