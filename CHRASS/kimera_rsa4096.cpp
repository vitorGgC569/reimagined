#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <tuple>
#include <immintrin.h>

// ============================================================================
// KIMERA RSA-4096 (V19-R4K)
// 64x 64-bit integers = 4096 bits.
// ============================================================================

struct UInt4096 {
    unsigned long long parts[64];

    UInt4096() { std::memset(parts, 0, sizeof(parts)); }

    UInt4096(unsigned long long val) {
        std::memset(parts, 0, sizeof(parts));
        parts[0] = val;
    }

    static UInt4096 power_of_2(int bit) {
        UInt4096 res;
        if (bit < 4096) res.parts[bit / 64] = (1ULL << (bit % 64));
        return res;
    }

    static UInt4096 max() {
        UInt4096 res;
        std::memset(res.parts, 0xFF, sizeof(res.parts));
        return res;
    }

    UInt4096 operator+(const UInt4096& other) const {
        UInt4096 res;
        unsigned __int128 carry = 0;
        for (int i = 0; i < 64; ++i) {
            unsigned __int128 sum = (unsigned __int128)parts[i] + other.parts[i] + carry;
            res.parts[i] = (unsigned long long)sum;
            carry = sum >> 64;
        }
        return res;
    }

    bool operator<(const UInt4096& other) const {
        for (int i = 63; i >= 0; --i) {
            if (parts[i] != other.parts[i]) return parts[i] < other.parts[i];
        }
        return false;
    }

    UInt4096 operator^(const UInt4096& other) const {
        UInt4096 res;
        for (int i = 0; i < 64; ++i) res.parts[i] = parts[i] ^ other.parts[i];
        return res;
    }

    bool operator==(const UInt4096& other) const {
        for (int i = 0; i < 64; ++i) if(parts[i] != other.parts[i]) return false;
        return true;
    }
};

inline int clz64(unsigned long long x) { return x ? __builtin_clzll(x) : 64; }

int clz4096(const UInt4096& x) {
    int zeros = 0;
    for (int i = 63; i >= 0; --i) {
        if (x.parts[i] != 0) return zeros + clz64(x.parts[i]);
        zeros += 64;
    }
    return 4096;
}

// ============================================================================
// COMPONENTS
// ============================================================================
struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<UInt4096> values;
    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

class RadixHeap4096 {
    static const int BUCKETS = 4097;
    std::vector<std::vector<int>> buckets;
    UInt4096 last_dist;
public:
    RadixHeap4096() : buckets(BUCKETS), last_dist(0) {}

    void push(int u, UInt4096 dist) {
        if (dist < last_dist) dist = last_dist;
        UInt4096 xor_diff = dist ^ last_dist;
        int idx;

        bool zero = true;
        for(int i=0; i<64; ++i) if(xor_diff.parts[i]) { zero = false; break; }

        if (zero) idx = 0;
        else idx = 4096 - clz4096(xor_diff);

        buckets[idx].push_back(u);
    }

    int pop(std::vector<UInt4096>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) return -1;

            UInt4096 min_val = UInt4096::max();
            for (int u : buckets[i]) {
                if (dist[u] < min_val) min_val = dist[u];
            }
            last_dist = min_val;

            for (int u : buckets[i]) {
                UInt4096 d = dist[u];
                UInt4096 xor_diff = d ^ last_dist;
                int idx;
                bool zero = true;
                for(int k=0; k<64; ++k) if(xor_diff.parts[k]) { zero = false; break; }
                if (zero) idx = 0;
                else idx = 4096 - clz4096(xor_diff);
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
        UInt4096 INF = UInt4096::max();
        std::vector<UInt4096> dist(n, INF);
        RadixHeap4096 pq;

        dist[source] = UInt4096(0);
        pq.push(source, UInt4096(0));

        long long operations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            UInt4096 du = dist[u];
            operations++;

            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            for (int idx = start; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                UInt4096 w = graph.values[idx];
                UInt4096 new_dist = du + w;
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

CSRGraph generate_line(int n) {
    CSRGraph g(n);
    g.row_ptr.resize(n + 1);
    g.col_ind.reserve(n);
    g.values.reserve(n);
    UInt4096 huge_weight = UInt4096::power_of_2(4080);

    int current_idx = 0;
    for (int u = 0; u < n; ++u) {
        g.row_ptr[u] = current_idx;
        if (u < n - 1) {
            g.col_ind.push_back(u + 1);
            g.values.push_back(huge_weight);
            current_idx++;
        }
    }
    g.row_ptr[n] = current_idx;
    g.m = current_idx;
    return g;
}

int main() {
    int N = 20000;
    std::cout << "=== KIMERA RSA-4096 ===" << std::endl;
    std::cout << "Nodes: " << N << std::endl;
    std::cout << "Edge: 2^4080" << std::endl;
    auto g = generate_line(N);
    auto [ops, ms] = KimeraCore::solve(g, 0);
    std::cout << "Time: " << ms << " ms" << std::endl;
    return 0;
}
