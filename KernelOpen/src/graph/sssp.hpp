#ifndef UHK_GRAPH_SSSP_H
#define UHK_GRAPH_SSSP_H

#include <vector>
#include <tuple>
#include <immintrin.h>
#include <chrono>

namespace uhk {
namespace graph {

    // CSR Graph structure for Solver
    struct GraphCSR {
        int n;
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<int> values;
    };

    // Radix Heap Implementation (Monotonic O(1) PQ)
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

                unsigned long long min_val = ~0ULL;
                for (int u : buckets[i]) {
                    if (dist[u] < min_val) min_val = dist[u];
                }
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

    // CHRASS Solver (SSSP)
    class SSSP {
    public:
        // Returns: Distances, Operations Count, Time (ms)
        static std::tuple<std::vector<unsigned long long>, long long, double> solve(const GraphCSR& graph, int source) {
            int n = graph.n;
            unsigned long long INF = ~0ULL >> 1; // Max signed int64 effectively
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

                // AVX2 Loop (4 edges)
                int n_vec = (end - start) / 4;
                __m256i v_du = _mm256_set1_epi64x(du);

                for (int i = 0; i < n_vec; ++i) {
                    __m128i v_idx_small = _mm_loadu_si128((__m128i*)&graph.col_ind[idx]);
                    __m256i v_indices = _mm256_cvtepi32_epi64(v_idx_small);

                    __m128i v_w_small = _mm_loadu_si128((__m128i*)&graph.values[idx]);
                    __m256i v_weights = _mm256_cvtepi32_epi64(v_w_small);

                    __m256i v_new = _mm256_add_epi64(v_du, v_weights);
                    __m256i v_curr = _mm256_i32gather_epi64((long long int*)dist.data(), v_idx_small, 8);

                    __m256i v_mask = _mm256_cmpgt_epi64(v_curr, v_new);
                    int mask = _mm256_movemask_pd(_mm256_castsi256_pd(v_mask));

                    if (mask) {
                        long long* nd_arr = (long long*)&v_new;
                        int* idx_arr = (int*)&graph.col_ind[idx];
                        if (mask & 1) { int v=idx_arr[0]; if(nd_arr[0] < dist[v]) { dist[v]=nd_arr[0]; pq.push(v, dist[v]); } }
                        if (mask & 2) { int v=idx_arr[1]; if(nd_arr[1] < dist[v]) { dist[v]=nd_arr[1]; pq.push(v, dist[v]); } }
                        if (mask & 4) { int v=idx_arr[2]; if(nd_arr[2] < dist[v]) { dist[v]=nd_arr[2]; pq.push(v, dist[v]); } }
                        if (mask & 8) { int v=idx_arr[3]; if(nd_arr[3] < dist[v]) { dist[v]=nd_arr[3]; pq.push(v, dist[v]); } }
                    }
                    idx += 4;
                }

                // Scalar Tail
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
            return {dist, operations, ms};
        }
    };

}
}
#endif
