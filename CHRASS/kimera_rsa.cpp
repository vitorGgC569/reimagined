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
// KIMERA RSA (V19-R)
// Scaling to RSA-2048 (2^2048 ~ 10^617) using 2048-bit integers.
// ============================================================================

// 32x 64-bit integers = 2048 bits.
struct UInt2048 {
    unsigned long long parts[32]; // parts[0] is LSB, parts[31] is MSB

    UInt2048() { std::memset(parts, 0, sizeof(parts)); }

    // Constructor from small int
    UInt2048(unsigned long long val) {
        std::memset(parts, 0, sizeof(parts));
        parts[0] = val;
    }

    // Constructor for specific bit set (power of 2)
    static UInt2048 power_of_2(int bit) {
        UInt2048 res;
        if (bit < 2048) {
            res.parts[bit / 64] = (1ULL << (bit % 64));
        }
        return res;
    }

    // Max value
    static UInt2048 max() {
        UInt2048 res;
        std::memset(res.parts, 0xFF, sizeof(res.parts));
        return res;
    }

    // Addition
    UInt2048 operator+(const UInt2048& other) const {
        UInt2048 res;
        unsigned __int128 carry = 0;
        // Unrolling could help, but loop is cleaner
        for (int i = 0; i < 32; ++i) {
            unsigned __int128 sum = (unsigned __int128)parts[i] + other.parts[i] + carry;
            res.parts[i] = (unsigned long long)sum;
            carry = sum >> 64;
        }
        return res;
    }

    // Less than
    bool operator<(const UInt2048& other) const {
        for (int i = 31; i >= 0; --i) {
            if (parts[i] != other.parts[i]) return parts[i] < other.parts[i];
        }
        return false;
    }

    // XOR
    UInt2048 operator^(const UInt2048& other) const {
        UInt2048 res;
        for (int i = 0; i < 32; ++i) {
            res.parts[i] = parts[i] ^ other.parts[i];
        }
        return res;
    }

    bool operator==(const UInt2048& other) const {
        for (int i = 0; i < 32; ++i) if(parts[i] != other.parts[i]) return false;
        return true;
    }
};

// 64-bit CLZ
inline int clz64(unsigned long long x) {
    return x ? __builtin_clzll(x) : 64;
}

// 2048-bit CLZ
int clz2048(const UInt2048& x) {
    int zeros = 0;
    for (int i = 31; i >= 0; --i) {
        if (x.parts[i] != 0) {
            return zeros + clz64(x.parts[i]);
        }
        zeros += 64;
    }
    return 2048;
}

// ============================================================================
// GRAPH
// ============================================================================
struct Edge {
    int u, v;
    UInt2048 w;
};

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<UInt2048> values;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

// ============================================================================
// RADIX HEAP 2048
// Buckets: 0 to 2048 (Size 2049)
// ============================================================================
class RadixHeap2048 {
    static const int BUCKETS = 2049;
    std::vector<std::vector<int>> buckets;
    UInt2048 last_dist;

public:
    RadixHeap2048() : buckets(BUCKETS), last_dist(0) {}

    void push(int u, UInt2048 dist) {
        if (dist < last_dist) dist = last_dist;

        UInt2048 xor_diff = dist ^ last_dist;
        int idx;

        // Check if zero
        bool zero = true;
        for(int i=0; i<32; ++i) if(xor_diff.parts[i]) { zero = false; break; }

        if (zero) {
            idx = 0;
        } else {
            idx = 2048 - clz2048(xor_diff);
        }

        buckets[idx].push_back(u);
    }

    int pop(std::vector<UInt2048>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) return -1;

            UInt2048 min_val = UInt2048::max();
            for (int u : buckets[i]) {
                if (dist[u] < min_val) min_val = dist[u];
            }
            last_dist = min_val;

            for (int u : buckets[i]) {
                UInt2048 d = dist[u];
                UInt2048 xor_diff = d ^ last_dist;

                int idx;
                bool zero = true;
                for(int k=0; k<32; ++k) if(xor_diff.parts[k]) { zero = false; break; }

                if (zero) idx = 0;
                else idx = 2048 - clz2048(xor_diff);

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
        UInt2048 INF = UInt2048::max();
        std::vector<UInt2048> dist(n, INF);

        RadixHeap2048 pq;

        dist[source] = UInt2048(0);
        pq.push(source, UInt2048(0));

        long long operations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            UInt2048 du = dist[u];
            operations++;

            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            for (int idx = start; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                UInt2048 w = graph.values[idx];

                UInt2048 new_dist = du + w;
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
// GENERATOR FOR RSA WEIGHTS (2^2040)
// ============================================================================
CSRGraph generate_rsa_line(int n) {
    CSRGraph g(n);
    g.row_ptr.resize(n + 1);
    g.col_ind.reserve(n);
    g.values.reserve(n);

    // RSA-2048 uses 2048-bit keys.
    // Edge weight approx 2^2040.
    UInt2048 huge_weight = UInt2048::power_of_2(2040);

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
    int N = 50000; // 50k nodes (Large structs slow down cache)
    std::cout << "=== KIMERA RSA (RSA-2048 Challenge) ===" << std::endl;
    std::cout << "Nodes: " << N << " (Line Topology)" << std::endl;
    std::cout << "Edge Weight: 2^2040" << std::endl;
    std::cout << "Target Dist: > 2^2048 (Will exceed RSA modulus range)" << std::endl;

    CSRGraph g = generate_rsa_line(N);

    auto [ops, ms] = KimeraCore::solve(g, 0);

    std::cout << "Done!" << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "Ops: " << ops << std::endl;

    return 0;
}
