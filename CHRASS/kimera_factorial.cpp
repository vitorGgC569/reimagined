#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <tuple>
#include <immintrin.h>

// ============================================================================
// KIMERA FACTORIAL (V19-F)
// 1000! ~ 10^2567 ~ 2^8530. Using 10240 bits (160x 64-bit).
// ============================================================================

struct UInt10240 {
    unsigned long long parts[160];

    UInt10240() { std::memset(parts, 0, sizeof(parts)); }

    UInt10240(unsigned long long val) {
        std::memset(parts, 0, sizeof(parts));
        parts[0] = val;
    }

    static UInt10240 power_of_2(int bit) {
        UInt10240 res;
        if (bit < 10240) res.parts[bit / 64] = (1ULL << (bit % 64));
        return res;
    }

    static UInt10240 max() {
        UInt10240 res;
        std::memset(res.parts, 0xFF, sizeof(res.parts));
        return res;
    }

    UInt10240 operator+(const UInt10240& other) const {
        UInt10240 res;
        unsigned __int128 carry = 0;
        for (int i = 0; i < 160; ++i) {
            unsigned __int128 sum = (unsigned __int128)parts[i] + other.parts[i] + carry;
            res.parts[i] = (unsigned long long)sum;
            carry = sum >> 64;
        }
        return res;
    }

    bool operator<(const UInt10240& other) const {
        for (int i = 159; i >= 0; --i) {
            if (parts[i] != other.parts[i]) return parts[i] < other.parts[i];
        }
        return false;
    }

    UInt10240 operator^(const UInt10240& other) const {
        UInt10240 res;
        for (int i = 0; i < 160; ++i) res.parts[i] = parts[i] ^ other.parts[i];
        return res;
    }

    bool operator==(const UInt10240& other) const {
        for (int i = 0; i < 160; ++i) if(parts[i] != other.parts[i]) return false;
        return true;
    }
};

inline int clz64(unsigned long long x) { return x ? __builtin_clzll(x) : 64; }

int clz10240(const UInt10240& x) {
    int zeros = 0;
    for (int i = 159; i >= 0; --i) {
        if (x.parts[i] != 0) return zeros + clz64(x.parts[i]);
        zeros += 64;
    }
    return 10240;
}

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<UInt10240> values;
    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

class RadixHeap10240 {
    static const int BUCKETS = 10241;
    std::vector<std::vector<int>> buckets;
    UInt10240 last_dist;
public:
    RadixHeap10240() : buckets(BUCKETS), last_dist(0) {}

    void push(int u, UInt10240 dist) {
        if (dist < last_dist) dist = last_dist;
        UInt10240 xor_diff = dist ^ last_dist;
        int idx;

        bool zero = true;
        for(int i=0; i<160; ++i) if(xor_diff.parts[i]) { zero = false; break; }

        if (zero) idx = 0;
        else idx = 10240 - clz10240(xor_diff);

        buckets[idx].push_back(u);
    }

    int pop(std::vector<UInt10240>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) return -1;

            UInt10240 min_val = UInt10240::max();
            for (int u : buckets[i]) {
                if (dist[u] < min_val) min_val = dist[u];
            }
            last_dist = min_val;

            for (int u : buckets[i]) {
                UInt10240 d = dist[u];
                UInt10240 xor_diff = d ^ last_dist;
                int idx;
                bool zero = true;
                for(int k=0; k<160; ++k) if(xor_diff.parts[k]) { zero = false; break; }
                if (zero) idx = 0;
                else idx = 10240 - clz10240(xor_diff);
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
        UInt10240 INF = UInt10240::max();
        std::vector<UInt10240> dist(n, INF);
        RadixHeap10240 pq;

        dist[source] = UInt10240(0);
        pq.push(source, UInt10240(0));

        long long operations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            UInt10240 du = dist[u];
            operations++;

            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            for (int idx = start; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                UInt10240 w = graph.values[idx];
                UInt10240 new_dist = du + w;
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
    UInt10240 huge_weight = UInt10240::power_of_2(8500); // 2^8500 ~ 10^2558

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
    int N = 10000;
    std::cout << "=== KIMERA FACTORIAL (1000!) ===" << std::endl;
    std::cout << "Nodes: " << N << std::endl;
    std::cout << "Edge: 2^8500" << std::endl;
    auto g = generate_line(N);
    auto [ops, ms] = KimeraCore::solve(g, 0);
    std::cout << "Time: " << ms << " ms" << std::endl;
    return 0;
}
