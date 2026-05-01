#ifndef UHK_STREAM_K_H
#define UHK_STREAM_K_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>

namespace uhk {
namespace scheduler {

struct StreamKConfig {
    uint32_t sm_count;      // Número de SMs disponíveis na GPU
    uint32_t tile_m;        // Tamanho do Tile M (ex: 128)
    uint32_t tile_n;        // Tamanho do Tile N (ex: 128)
    uint32_t tile_k;        // Tamanho do Tile K (ex: 32)

    // Phase 2: Distributed Topology Awareness
    uint32_t num_nodes;     // Total de nós no cluster
    uint32_t node_id;       // ID do nó local (0..num_nodes-1)
};

struct TileWork {
    uint32_t tile_idx;      // Índice linear do tile de saída (M, N)
    uint32_t k_start;       // Início da iteração K
    uint32_t k_end;         // Fim da iteração K (exclusivo)
};

// Estrutura que define o trabalho atribuído a um SM
struct SMWorkAssignment {
    uint32_t sm_idx;
    std::vector<TileWork> tiles;
};

// Função de decomposição Stream-K (Topology Aware)
// Phase 2: Distributed Logic
// O algoritmo deve garantir que splits parciais (Split-K) preferencialmente
// residam no mesmo nó para permitir redução local (NVLink) em vez de global (Infiniband).
inline std::vector<SMWorkAssignment> decompose_stream_k(
    int M, int N, int K,
    const StreamKConfig& config
) {
    // 1. Total Global Work
    uint32_t grid_m = (M + config.tile_m - 1) / config.tile_m;
    uint32_t grid_n = (N + config.tile_n - 1) / config.tile_n;
    uint32_t total_output_tiles = grid_m * grid_n;
    uint32_t k_iters = (K + config.tile_k - 1) / config.tile_k;

    // Trabalho Total Global em "units"
    uint64_t total_work_global = (uint64_t)total_output_tiles * k_iters;

    // Dividir trabalho entre NÓS primeiro (Distributed Partitioning)
    // Estratégia simples: Block Partitioning por Nó para minimizar bordas
    uint64_t work_per_node = total_work_global / config.num_nodes;
    uint64_t node_remainder = total_work_global % config.num_nodes;

    uint64_t my_node_work = work_per_node + (config.node_id < node_remainder ? 1 : 0);

    // Calcular offset global de início para este nó
    uint64_t my_global_start_iter = 0;
    for(uint32_t n=0; n < config.node_id; ++n) {
        my_global_start_iter += work_per_node + (n < node_remainder ? 1 : 0);
    }

    // Agora distribuir 'my_node_work' entre os SMs LOCAIS deste nó
    uint64_t work_per_sm = my_node_work / config.sm_count;
    uint64_t sm_remainder = my_node_work % config.sm_count;

    std::vector<SMWorkAssignment> assignments(config.sm_count);

    uint64_t current_iter_in_node = 0; // Relativo ao início do trabalho do nó

    for (uint32_t sm = 0; sm < config.sm_count; ++sm) {
        assignments[sm].sm_idx = sm;

        uint64_t my_sm_work = work_per_sm + (sm < sm_remainder ? 1 : 0);

        // Intervalo absoluto global de iterações para este SM
        uint64_t start_iter_abs = my_global_start_iter + current_iter_in_node;
        uint64_t end_iter_abs = start_iter_abs + my_sm_work;
        current_iter_in_node += my_sm_work;

        // Mapear intervalo [start, end) para (Tile, K)
        uint64_t iter = start_iter_abs;
        while (iter < end_iter_abs) {
            uint32_t tile_idx = (uint32_t)(iter / k_iters);
            uint32_t k_offset = (uint32_t)(iter % k_iters);

            uint32_t work_in_this_tile = std::min((uint64_t)(k_iters - k_offset), end_iter_abs - iter);

            TileWork work_item;
            work_item.tile_idx = tile_idx;
            work_item.k_start = k_offset;
            work_item.k_end = k_offset + work_in_this_tile;

            assignments[sm].tiles.push_back(work_item);

            iter += work_in_this_tile;
        }
    }

    return assignments;
}

} // namespace scheduler
} // namespace uhk

#endif // UHK_STREAM_K_H
