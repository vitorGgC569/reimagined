#include "bmssp_cpp/bmssp.hpp"
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <string>

using T = double;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    int n, m;
    int source_node = 0;

    // Ler cabeçalho: N M SOURCE
    if (!(infile >> n >> m >> source_node)) {
        std::cerr << "Error reading header" << std::endl;
        return 1;
    }

    // Inicializar solver
    spp::bmssp<T> solver(n);

    // Ler arestas: U V W
    int u, v;
    double w;
    for (int i = 0; i < m; ++i) {
        if (infile >> u >> v >> w) {
            solver.addEdge(u, v, w);
        }
    }
    infile.close();

    // Preparar grafo
    // O paper menciona transformação para grau constante.
    // Vamos assumir true para robustez, ou false se o grafo já for compatível.
    // O V18 lida com grafos gerais. Vamos setar true para garantir que Duan funcione em qualquer input.
    solver.prepare_graph(true);

    // Medir tempo de execução
    auto start = std::chrono::high_resolution_clock::now();

    auto result = solver.execute(source_node);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Imprimir tempo em ms
    std::cout << duration.count() << std::endl;

    // Imprimir distâncias para validação (stdout)
    // Formato: node distance
    const auto& distances = result.first;
    for (int i = 0; i < n; ++i) {
        if (distances[i] == solver.oo) {
            std::cout << i << " inf" << std::endl;
        } else {
            std::cout << i << " " << distances[i] << std::endl;
        }
    }

    return 0;
}
