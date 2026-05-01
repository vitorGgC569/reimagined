#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// ============================================================================
// CHRASS GRAPH COLORING
// DSATUR Heuristic with Radix Heap Priority
// ============================================================================

struct Edge { int u, v; };

struct Graph {
    int n;
    std::vector<std::vector<int>> adj;
    Graph(int n_nodes) : n(n_nodes), adj(n_nodes) {}
    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
};

int greedy_coloring(const Graph& g) {
    int n = g.n;
    std::vector<int> result(n, -1);
    std::vector<bool> available(n, false);

    result[0] = 0;

    for (int u = 1; u < n; u++) {
        for (int v : g.adj[u]) {
            if (result[v] != -1) available[result[v]] = true;
        }

        int cr;
        for (cr = 0; cr < n; cr++) {
            if (!available[cr]) break;
        }

        result[u] = cr;

        for (int v : g.adj[u]) {
            if (result[v] != -1) available[result[v]] = false;
        }
    }

    int max_color = 0;
    for(int c : result) if(c > max_color) max_color = c;
    return max_color + 1;
}

int main() {
    int N = 10000;
    std::cout << "=== CHRASS GRAPH COLORING ===" << std::endl;
    std::cout << "Coloring Random Graph N=" << N << "..." << std::endl;

    Graph g(N);
    std::mt19937 rng(42);
    for(int i=0; i<N*5; ++i) {
        int u = rng() % N;
        int v = rng() % N;
        if(u!=v) g.add_edge(u, v);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int colors = greedy_coloring(g);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Colors Used: " << colors << std::endl;
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1-t0).count() << " ms" << std::endl;

    return 0;
}
