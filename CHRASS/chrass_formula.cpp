#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <numeric>
#include <tuple>
#include <immintrin.h>
#include <random>

// ============================================================================
// CHRASS FORMULA (Instrumented for Mathematical Regression)
// Supports simulated bit-width scaling
// ============================================================================

struct Edge { int u, v; int w; };

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<int> values;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

class Generator {
public:
    static CSRGraph generate_chaos(int n, int avg_deg) {
        CSRGraph g(n);
        size_t estimated_m = (size_t)n * avg_deg;
        g.row_ptr.resize(n + 1);
        g.col_ind.reserve(estimated_m);
        g.values.reserve(estimated_m);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist_node(0, n - 1);
        std::uniform_int_distribution<int> dist_w(1, 100);

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
};

// ============================================================================
// INSTRUMENTED CORE
// ============================================================================

class RadixHeapInstrumented {
    static const int BUCKETS = 65;
    std::vector<std::vector<int>> buckets;
    unsigned long long last_dist;
    int bit_load; // Simulate extra words
public:
    long long push_time_ns = 0;
    long long pop_time_ns = 0;
    long long push_count = 0;
    long long pop_count = 0;

    RadixHeapInstrumented(int b_load) : buckets(BUCKETS), last_dist(0), bit_load(b_load) {}

    void push(int u, unsigned long long dist) {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (dist < last_dist) dist = last_dist;
        unsigned long long xor_diff = dist ^ last_dist;
        int idx = (xor_diff == 0) ? 0 : (64 - __builtin_clzll(xor_diff));
        buckets[idx].push_back(u);

        // Simulate bit width cost (linear copy)
        volatile long long dummy = 0;
        for(int i=0; i<bit_load; ++i) dummy += i; // Memory/ALU waste

        auto t1 = std::chrono::high_resolution_clock::now();
        push_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        push_count++;
    }

    int pop(std::vector<unsigned long long>& dist) {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) {
                auto t1 = std::chrono::high_resolution_clock::now();
                pop_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                pop_count++; // Count failed pops?
                return -1;
            }
            unsigned long long min_val = 9223372036854775807ULL;
            for (int u : buckets[i]) if (dist[u] < min_val) min_val = dist[u];
            last_dist = min_val;
            for (int u : buckets[i]) {
                unsigned long long xor_diff = dist[u] ^ last_dist;
                int idx = (xor_diff == 0) ? 0 : (64 - __builtin_clzll(xor_diff));
                buckets[idx].push_back(u);

                // Re-bucket cost proportional to bit-width logic overhead?
                // Actually re-bucketing involves key comparison and bitscan.
                // Comparison cost is O(words).
                volatile long long dummy = 0;
                for(int k=0; k<bit_load; ++k) dummy += k;
            }
            buckets[i].clear();
        }
        int u = buckets[0].back();
        buckets[0].pop_back();

        auto t1 = std::chrono::high_resolution_clock::now();
        pop_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        pop_count++;
        return u;
    }
};

class ChrassCoreInstrumented {
public:
    static void solve(const CSRGraph& graph, int source, int bit_load) {
        int n = graph.n;
        unsigned long long INF = 9223372036854775807ULL;
        std::vector<unsigned long long> dist(n, INF);
        RadixHeapInstrumented pq(bit_load);

        dist[source] = 0;
        pq.push(source, 0);

        long long relax_time_ns = 0;
        long long relax_count = 0;

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            unsigned long long du = dist[u];
            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            auto t_relax_0 = std::chrono::high_resolution_clock::now();

            // Scalar Loop to measure pure complexity logic
            for (int idx = start; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                int w = graph.values[idx];

                // Simulate Bit-width addition cost
                unsigned long long new_dist = du + w;
                volatile long long dummy = 0;
                for(int k=0; k<bit_load; ++k) dummy += k;

                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    // Note: Push is timed inside PQ
                    // We pause relax timer? No, push is part of relaxation overhead
                    pq.push(v, dist[v]);
                }
            }
            auto t_relax_1 = std::chrono::high_resolution_clock::now();
            relax_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_relax_1 - t_relax_0).count();
            // Subtract push time to isolate pure logic?
            // Actually, total relaxation cost includes PQ operations.
            // But we have separate PQ counters.
            // Let's keep relax_time_ns as "Time spent in edge loop".
            relax_count += (end - start);
        }

        // Output CSV Line
        // N, M, BitLoad, RelaxTime, RelaxCount, PushTime, PushCount, PopTime, PopCount
        std::cout << n << "," << graph.m << "," << bit_load << ","
                  << relax_time_ns << "," << relax_count << ","
                  << pq.push_time_ns << "," << pq.push_count << ","
                  << pq.pop_time_ns << "," << pq.pop_count << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // ./chrass_formula <N> <M_factor> <BitLoad>
    if (argc < 4) return 1;
    int N = std::atoi(argv[1]);
    int deg = std::atoi(argv[2]);
    int bit_load = std::atoi(argv[3]);

    CSRGraph graph = Generator::generate_chaos(N, deg);
    ChrassCoreInstrumented::solve(graph, 0, bit_load);
    return 0;
}
