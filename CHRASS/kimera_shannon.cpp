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
// KIMERA SHANNON (V19-S)
// Scaling to 10^120 (Shannon Number) using 512-bit integers.
// ============================================================================

// 10^120 ~ 2^398. We use 512 bits (8x 64-bit).
struct UInt512 {
    unsigned long long parts[8]; // parts[0] is LSB, parts[7] is MSB

    UInt512() { std::memset(parts, 0, sizeof(parts)); }

    // Constructor from small int
    UInt512(unsigned long long val) {
        std::memset(parts, 0, sizeof(parts));
        parts[0] = val;
    }

    // Constructor for specific bit set (power of 2)
    static UInt512 power_of_2(int bit) {
        UInt512 res;
        if (bit < 512) {
            res.parts[bit / 64] = (1ULL << (bit % 64));
        }
        return res;
    }

    // Max value
    static UInt512 max() {
        UInt512 res;
        std::memset(res.parts, 0xFF, sizeof(res.parts));
        return res;
    }

    // Addition
    UInt512 operator+(const UInt512& other) const {
        UInt512 res;
        unsigned __int128 carry = 0;
        for (int i = 0; i < 8; ++i) {
            unsigned __int128 sum = (unsigned __int128)parts[i] + other.parts[i] + carry;
            res.parts[i] = (unsigned long long)sum;
            carry = sum >> 64;
        }
        return res;
    }

    // Less than
    bool operator<(const UInt512& other) const {
        for (int i = 7; i >= 0; --i) {
            if (parts[i] != other.parts[i]) return parts[i] < other.parts[i];
        }
        return false;
    }

    // XOR
    UInt512 operator^(const UInt512& other) const {
        UInt512 res;
        for (int i = 0; i < 8; ++i) {
            res.parts[i] = parts[i] ^ other.parts[i];
        }
        return res;
    }

    bool operator==(const UInt512& other) const {
        for (int i = 0; i < 8; ++i) if(parts[i] != other.parts[i]) return false;
        return true;
    }
};

// 64-bit CLZ
inline int clz64(unsigned long long x) {
    return x ? __builtin_clzll(x) : 64;
}

// 512-bit CLZ
int clz512(const UInt512& x) {
    int zeros = 0;
    for (int i = 7; i >= 0; --i) {
        if (x.parts[i] != 0) {
            return zeros + clz64(x.parts[i]);
        }
        zeros += 64;
    }
    return 512;
}

// ============================================================================
// GRAPH
// ============================================================================
struct Edge {
    int u, v;
    UInt512 w;
};

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<UInt512> values;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

// ============================================================================
// RADIX HEAP 512
// Buckets: 0 to 512 (Size 513)
// ============================================================================
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

        // If xor_diff == 0, idx = 0
        // If MSB (511) is set, clz=0 -> idx = 512
        // idx = 512 - clz(xor_diff)

        // Check if zero
        bool zero = true;
        for(int i=0; i<8; ++i) if(xor_diff.parts[i]) { zero = false; break; }

        if (zero) {
            idx = 0;
        } else {
            idx = 512 - clz512(xor_diff);
        }

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

// ============================================================================
// CORE
// ============================================================================
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

                UInt512 new_dist = du + w;
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

// ============================================================================
// GENERATOR FOR SHANNON WEIGHTS (10^115)
// ============================================================================
CSRGraph generate_shannon_line(int n) {
    CSRGraph g(n);
    g.row_ptr.resize(n + 1);
    g.col_ind.reserve(n);
    g.values.reserve(n);

    // 10^120 approx 2^398.
    // Edge weight approx 10^115 approx 2^382.
    // Let's set bit 382.
    UInt512 huge_weight = UInt512::power_of_2(382);

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
    int N = 100000; // 100k nodes line
    std::cout << "=== KIMERA SHANNON (10^120 Challenge) ===" << std::endl;
    std::cout << "Nodes: " << N << " (Line Topology)" << std::endl;
    std::cout << "Edge Weight: 2^382 (~10^115)" << std::endl;
    std::cout << "Target Dist: 10^5 * 2^382 > 2^398 (~10^120)" << std::endl;

    CSRGraph g = generate_shannon_line(N);

    auto [ops, ms] = KimeraCore::solve(g, 0);

    std::cout << "Done!" << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "Ops: " << ops << std::endl;

    return 0;
}
