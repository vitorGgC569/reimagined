#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace pantheon {
namespace physics {

    class TopologyDistillation {
    public:
        // Persistent Homology (0-dim): Connected Components
        // Computes the "persistence barcode" of 0-dim homology via Minimum Spanning Tree (MST) / Single Linkage.
        // Returns Wasserstein distance between Student and Teacher barcodes.

        static float compute_topology_loss(const std::vector<float>& student_dist,
                                         const std::vector<float>& teacher_dist,
                                         int n_samples) {

            auto s_barcodes = compute_0dim_barcodes(student_dist, n_samples);
            auto t_barcodes = compute_0dim_barcodes(teacher_dist, n_samples);

            return compute_wasserstein(s_barcodes, t_barcodes);
        }

    private:
        struct Barcode {
            float death; // Birth is always 0 for 0-dim connected components of points
        };

        // Input: pairwise distance matrix (flattened, size N*N)
        static std::vector<Barcode> compute_0dim_barcodes(const std::vector<float>& dists, int n) {
            // Simplified Kruskal's algorithm logic to find edge weights that merge components.
            // For 0-dim, components die when they merge.
            // We collect all edge weights that cause a merge.

            struct Edge { int u, v; float w; };
            std::vector<Edge> edges;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    edges.push_back({i, j, dists[i*n + j]});
                }
            }

            // Sort edges by weight
            std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
                return a.w < b.w;
            });

            // Union-Find
            std::vector<int> parent(n);
            for(int i=0; i<n; ++i) parent[i] = i;
            auto find = [&](int i) {
                int root = i;
                while (root != parent[root]) root = parent[root];
                return root; // No path compression for simplicity/recursion-safety
            };

            std::vector<Barcode> barcodes;
            int components = n;

            for (const auto& e : edges) {
                int root_u = find(e.u);
                int root_v = find(e.v);

                if (root_u != root_v) {
                    // Merge: One component dies.
                    // We record the death time (edge weight).
                    barcodes.push_back({e.w});
                    parent[root_u] = root_v;
                    components--;
                }
            }

            // The last component never dies (infinite persistence), we ignore it or set huge death.
            // We sort barcodes for Wasserstein matching.
            std::sort(barcodes.begin(), barcodes.end(), [](const Barcode& a, const Barcode& b) {
                return a.death < b.death;
            });

            return barcodes;
        }

        static float compute_wasserstein(const std::vector<Barcode>& s, const std::vector<Barcode>& t) {
            float loss = 0.0f;
            size_t n = std::min(s.size(), t.size());

            for (size_t i = 0; i < n; ++i) {
                float diff = s[i].death - t[i].death;
                loss += diff * diff;
            }

            // If sizes differ (shouldn't for same batch size), penalize
            return loss;
        }
    };

}
}
