#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace Aion {

    // Estrutura CSR Simplificada para o AION (Ingestão)
    struct SimpleCSR {
        int n;
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
    };

    class SpectralLayout {
    public:
        // Output: Mapa de Permutação (Old ID -> New ID)
        // Se invert = true, retorna (New ID -> Old ID)
        static std::vector<int> compute_layout(const SimpleCSR& graph, int source_hint, int k_iterations = 15) {
            int n = graph.n;
            std::vector<float> heat(n, 0.0f);
            std::vector<float> next_heat(n, 0.0f);
            std::vector<float> inv_deg(n, 0.0f);

            // 1. Precompute Degrees
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                int degree = graph.row_ptr[i+1] - graph.row_ptr[i];
                inv_deg[i] = (degree > 0) ? 1.0f / degree : 0.0f;
            }

            // 2. Heat Source
            // Se source_hint for -1 (desconhecido), usar PageRank (todas as fontes)
            if (source_hint >= 0 && source_hint < n) {
                heat[source_hint] = 1.0f;
            } else {
                std::fill(heat.begin(), heat.end(), 1.0f / n);
            }

            // 3. Chebyshev / Power Iteration (Pull-based for cache safety)
            // Simulating Heat Diffusion
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
                            // Atomic add needed for Push model
                            #pragma omp atomic
                            next_heat[v] += push_val;
                        }
                        #pragma omp atomic
                        next_heat[u] += 0.5f * heat[u];
                    }
                }
                std::swap(heat, next_heat);
            }

            // 4. Sort (Compute Permutation)
            std::vector<int> perm(n);
            std::iota(perm.begin(), perm.end(), 0);

            // Sort Descending Heat (Hotter nodes first -> lower ID)
            std::sort(perm.begin(), perm.end(), [&](int a, int b) {
                return heat[a] > heat[b];
            });

            // Return Mapping: Old ID -> New ID
            std::vector<int> mapping(n);
            #pragma omp parallel for
            for (int new_id = 0; new_id < n; ++new_id) {
                mapping[perm[new_id]] = new_id;
            }

            return mapping;
        }
    };
}
