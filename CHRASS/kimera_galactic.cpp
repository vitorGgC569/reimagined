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
// KIMERA GALACTIC (V19-G)
// Supporting 10^40 magnitude weights using 192-bit integers.
// ============================================================================

// 10^40 requires ~133 bits. We use 3x 64-bit ints (192 bits).
struct UInt192 {
    unsigned long long lo;
    unsigned long long mid;
    unsigned long long hi;

    UInt192() : lo(0), mid(0), hi(0) {}
    UInt192(unsigned long long l) : lo(l), mid(0), hi(0) {}
    UInt192(unsigned long long h, unsigned long long m, unsigned long long l) : lo(l), mid(m), hi(h) {}

    // Addition with carry
    UInt192 operator+(const UInt192& other) const {
        UInt192 res;
        unsigned __int128 tmp = (unsigned __int128)lo + other.lo;
        res.lo = (unsigned long long)tmp;
        unsigned long long carry = (unsigned long long)(tmp >> 64);

        tmp = (unsigned __int128)mid + other.mid + carry;
        res.mid = (unsigned long long)tmp;
        carry = (unsigned long long)(tmp >> 64);

        res.hi = hi + other.hi + carry;
        return res;
    }

    // Less than
    bool operator<(const UInt192& other) const {
        if (hi != other.hi) return hi < other.hi;
        if (mid != other.mid) return mid < other.mid;
        return lo < other.lo;
    }

    // XOR (for Radix Heap)
    UInt192 operator^(const UInt192& other) const {
        return UInt192(hi ^ other.hi, mid ^ other.mid, lo ^ other.lo);
    }

    bool operator==(const UInt192& other) const {
        return hi == other.hi && mid == other.mid && lo == other.lo;
    }

    bool operator!=(const UInt192& other) const {
        return !(*this == other);
    }
};

// Helper to print roughly
std::ostream& operator<<(std::ostream& os, const UInt192& val) {
    if (val.hi > 0) os << "Large(>2^128)";
    else if (val.mid > 0) os << "Large(>2^64)";
    else os << val.lo;
    return os;
}

// 64-bit CLZ
inline int clz64(unsigned long long x) {
    return x ? __builtin_clzll(x) : 64;
}

// 192-bit CLZ (for Bucket Index)
int clz192(const UInt192& x) {
    if (x.hi) return clz64(x.hi);
    if (x.mid) return 64 + clz64(x.mid);
    if (x.lo) return 128 + clz64(x.lo);
    return 192;
}

// ============================================================================
// GRAPH
// ============================================================================
struct Edge {
    int u, v;
    UInt192 w;
};

struct CSRGraph {
    int n;
    int m;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<UInt192> values;

    CSRGraph(int n_nodes) : n(n_nodes), m(0) {}
};

// ============================================================================
// RADIX HEAP 192
// Buckets: 0 to 192 (Size 193)
// ============================================================================
class RadixHeap192 {
    static const int BUCKETS = 193;
    std::vector<std::vector<int>> buckets;
    UInt192 last_dist;

public:
    RadixHeap192() : buckets(BUCKETS), last_dist(0) {}

    void push(int u, UInt192 dist) {
        if (dist < last_dist) dist = last_dist; // Monotonicity

        UInt192 xor_diff = dist ^ last_dist;
        int idx;

        // Logic: clz returns leading zeros.
        // If xor_diff is 0, idx = 0.
        // If xor_diff has 1 at MSB (bit 191), clz=0. We want bucket ~192?
        // Standard mapping: idx = bit_width - clz(xor_diff)
        // If xor_diff=1 (bit 0), clz=191. idx = 192 - 191 = 1.
        // If xor_diff=MSB, clz=0. idx = 192.

        if (xor_diff == UInt192(0)) {
            idx = 0;
        } else {
            idx = 192 - clz192(xor_diff);
        }

        buckets[idx].push_back(u);
    }

    int pop(std::vector<UInt192>& dist) {
        if (buckets[0].empty()) {
            int i = 1;
            while (i < BUCKETS && buckets[i].empty()) i++;
            if (i == BUCKETS) return -1;

            // Find min
            UInt192 min_val(18446744073709551615ULL, 18446744073709551615ULL, 18446744073709551615ULL);
            for (int u : buckets[i]) {
                if (dist[u] < min_val) min_val = dist[u];
            }
            last_dist = min_val;

            for (int u : buckets[i]) {
                UInt192 d = dist[u];
                UInt192 xor_diff = d ^ last_dist;
                int idx = (xor_diff == UInt192(0)) ? 0 : (192 - clz192(xor_diff));
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
        // INF: Max 192-bit value
        UInt192 INF(18446744073709551615ULL, 18446744073709551615ULL, 18446744073709551615ULL);
        std::vector<UInt192> dist(n, INF);

        RadixHeap192 pq;

        dist[source] = UInt192(0);
        pq.push(source, UInt192(0));

        long long operations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            int u = pq.pop(dist);
            if (u == -1) break;

            UInt192 du = dist[u];
            operations++;

            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            // Scalar Loop (BigInt not AVX friendly)
            for (int idx = start; idx < end; ++idx) {
                int v = graph.col_ind[idx];
                UInt192 w = graph.values[idx];

                UInt192 new_dist = du + w;
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
// GENERATOR FOR GALACTIC WEIGHTS
// ============================================================================
CSRGraph generate_galactic_line(int n) {
    CSRGraph g(n);
    g.row_ptr.resize(n + 1);
    g.col_ind.reserve(n);
    g.values.reserve(n);

    // Creates a line where each edge has weight 10^35
    // 10^35 approx = 2^116. Fits in mid/lo.
    // 100,000 hops * 10^35 = 10^40.

    // 10^35 decomposition:
    // log2(10^35) = 116.2
    // hi=0, mid has upper bits, lo has lower bits.
    // Let's manually construct a "huge number" approx 10^35.
    // 2^116
    UInt192 huge_weight(0, (1ULL << 52), 0);

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
    std::cout << "=== KIMERA GALACTIC (10^40 Challenge) ===" << std::endl;
    std::cout << "Nodes: " << N << " (Line Topology)" << std::endl;
    std::cout << "Edge Weight: ~10^35 (2^116)" << std::endl;
    std::cout << "Target Dist: ~10^40 (2^133)" << std::endl;

    CSRGraph g = generate_galactic_line(N);

    auto [ops, ms] = KimeraCore::solve(g, 0);

    std::cout << "Done!" << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "Ops: " << ops << std::endl;

    return 0;
}
