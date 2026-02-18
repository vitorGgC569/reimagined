#ifndef UHK_RDMA_RING_BUFFER_H
#define UHK_RDMA_RING_BUFFER_H

#include <cstdint>
#include <vector>

// =================================================================================
// Global Ring Buffer (RDMA Backend) - Phase 2
// =================================================================================
// Abstração para suporte a "Infinite Fabric".
// Permite que a CPU mapeie memória de Ring Buffers em nós remotos.
// =================================================================================

namespace uhk {
namespace host {

struct RemoteBufferHandle {
    uint32_t node_id;
    uint64_t remote_addr; // Endereço virtual na GPU remota
    uint32_t rkey;        // Chave RDMA para acesso remoto
};

class GlobalRingBufferManager {
public:
    GlobalRingBufferManager(int local_node_id, int num_nodes)
        : local_node_id_(local_node_id), num_nodes_(num_nodes) {}

    // Simulação de registro de buffer remoto (IB Verbs: ibv_reg_mr)
    void register_remote_buffer(int node_id, void* buffer_ptr, size_t size) {
        // Em implementação real, isso cria um Queue Pair (QP)
        RemoteBufferHandle handle;
        handle.node_id = node_id;
        handle.remote_addr = (uint64_t)buffer_ptr;
        handle.rkey = 0xDEADBEEF + node_id; // Mock key
        remote_buffers_.push_back(handle);
    }

    // Escrita RDMA (RDMA Write with Immediate)
    // Dispara comando diretamente na memória da GPU do nó destino
    bool push_remote_command(int target_node, const void* command_packet, size_t size) {
        // Lógica de IB Verbs:
        // ibv_post_send(qp, &wr, &bad_wr);
        // Usamos RC (Reliable Connected) transport.

        // Simulação: Acesso direto à memória (assumindo espaço de endereçamento compartilhado em simulação)
        // Em cluster real, isso falharia sem drivers IB.
        return true;
    }

private:
    int local_node_id_;
    int num_nodes_;
    std::vector<RemoteBufferHandle> remote_buffers_;
};

} // namespace host
} // namespace uhk

#endif // UHK_RDMA_RING_BUFFER_H
