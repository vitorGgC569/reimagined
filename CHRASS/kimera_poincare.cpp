#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <tuple>
#include <immintrin.h>

// ============================================================================
// KIMERA POINCARE (V19-P)
// Storing Exponents (Log Space) for 10^10^120.
// Uses UInt512 to store the Exponent.
// Logic: dist[v] = max(dist[u], w) (Bottleneck Path)
// ============================================================================

// Reuse UInt512 from Shannon
struct UInt512 {
    unsigned long long parts[8];

    UInt512() { std::memset(parts, 0, sizeof(parts)); }

    UInt512(unsigned long long val) {
        std::memset(parts, 0, sizeof(parts));
        parts[0] = val;
    }

    static UInt512 power_of_2(int bit) {
        UInt512 res;
        if (bit < 512) res.parts[bit / 64] = (1ULL << (bit % 64));
        return res;
    }

    static UInt512 max() {
        UInt512 res;
        std::memset(res.parts, 0xFF, sizeof(res.parts));
        return res;
    }

    // No Addition needed for Bottleneck, just Max and XOR/Less
    bool operator<(const UInt512& other) const {
        for (int i = 7; i >= 0; --i) {
            if (parts[i] != other.parts[i]) return parts[i] < other.parts[i];
        }
        return false;
    }

    UInt512 operator^(const UInt512& other) const {
        UInt512 res;
        for (int i = 0; i < 8; ++i) res.parts[i] = parts[i] ^ other.parts[i];
        return res;
    }

    bool operator==(const UInt512& other) const {
        for (int i = 0; i < 8; ++i) if(parts[i] != other.parts[i]) return false;
        return true;
    }

    bool operator>(const UInt512& other) const {
        return other < *this;
    }
};

inline int clz64(unsigned long long x) { return x ? __builtin_clzll(x) : 64; }

int clz512(const UInt512& x) {
    int zeros = 0;
    for (int i = 7; i >= 0; --i) {
        if (x.parts[i] != 0) return zeros + clz64(x.parts[i]);
        zeros += 64;
    }
    return 512;
}

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<UInt512> values; // Exponents
    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

class RadixHeap512 {
    static const int BUCKETS = 513;
    std::vector<std::vector<int>> buckets;
    UInt512 last_dist;
public:
    RadixHeap512() : buckets(BUCKETS), last_dist(0) {}

    void push(int u, UInt512 dist) {
        if (dist < last_dist) dist = last_dist;
        UInt512 xor_diff = dist ^ last_dist;
        int idx;

        bool zero = true;
        for(int i=0; i<8; ++i) if(xor_diff.parts[i]) { zero = false; break; }

        if (zero) idx = 0;
        else idx = 512 - clz512(xor_diff);

        buckets[idx].push_back(u);
    }

    int pop(std::vector<UInt512>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) return -1;

            UInt512 min_val = UInt512::max();
            for (int u : buckets[i]) {
                if (dist[u] < min_val) min_val = dist[u];
            }
            last_dist = min_val;

            for (int u : buckets[i]) {
                UInt512 d = dist[u];
                UInt512 xor_diff = d ^ last_dist;
                int idx;
                bool zero = true;
                for(int k=0; k<8; ++k) if(xor_diff.parts[k]) { zero = false; break; }
                if (zero) idx = 0;
                else idx = 512 - clz512(xor_diff);
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
        UInt512 INF = UInt512::max();
        std::vector<UInt512> dist(n, INF);
        RadixHeap512 pq;

        dist[source] = UInt512(0);
        pq.push(source, UInt512(0));

        long long operations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            UInt512 du = dist[u];
            operations++;

            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            for (int idx = start; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                UInt512 w = graph.values[idx];

                // Bottleneck Logic: new_dist = max(du, w)
                // Approximating 10^A + 10^B ~ 10^max(A, B)
                UInt512 new_dist = (du < w) ? w : du;

                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pq.push(v, new_dist);
                }
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return {operations, ms};
    }
};

CSRGraph generate_poincare_line(int n) {
    CSRGraph g(n);
    g.row_ptr.resize(n + 1);
    g.col_ind.reserve(n);
    g.values.reserve(n);
    // Exponent 10^115 ~ 2^382
    UInt512 huge_exp = UInt512::power_of_2(382);

    int current_idx = 0;
    for (int u = 0; u < n; ++u) {
        g.row_ptr[u] = current_idx;
        if (u < n - 1) {
            g.col_ind.push_back(u + 1);
            g.values.push_back(huge_exp);
            current_idx++;
        }
    }
    g.row_ptr[n] = current_idx;
    g.m = current_idx;
    return g;
}

int main() {
    int N = 100000;
    std::cout << "=== KIMERA POINCARE (Logarithmic 10^10^120) ===" << std::endl;
    std::cout << "Nodes: " << N << std::endl;
    std::cout << "Edge Exponent: 10^115" << std::endl;
    auto g = generate_poincare_line(N);
    auto [ops, ms] = KimeraCore::solve(g, 0);
    std::cout << "Time: " << ms << " ms" << std::endl;
    return 0;
}
