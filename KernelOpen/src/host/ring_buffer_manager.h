#ifndef RING_BUFFER_MANAGER_H
#define RING_BUFFER_MANAGER_H

#include "../common/uhk_types.h"
#include <iostream>
#include <thread>
#include <vector>
#include <cstring>
#include <atomic>

// Simulação de APIs CUDA para ambiente Host-Only (para testes)
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace uhk {
namespace host {

class RingBufferManager {
public:
    // Construtor aceita ponteiro externo (alocado via cudaHostAlloc pelo Runtime)
    RingBufferManager(RingBufferControl* external_control) : control_(external_control), owns_memory_(false) {
        initialize();
    }

    // Fallback construtor (apenas para testes unitários isolados)
    RingBufferManager() : owns_memory_(true) {
        control_ = new RingBufferControl();
        initialize();
    }

    ~RingBufferManager() {
        if (owns_memory_) {
            delete control_;
        }
    }

    void initialize() {
        // Inicializa atomicamente
        std::atomic_init(&control_->tail, 0u);
        std::atomic_init(&control_->head, 0u);

        // Limpa buffer
        std::memset(control_->commands, 0, sizeof(CommandPacket) * RING_BUFFER_SIZE);
    }

    RingBufferControl* get_control_ptr() {
        return control_;
    }

    // Função do Produtor (CPU)
    bool push_command(OpType type, uint64_t param_ptr, uint64_t task_id) {
        uint32_t current_tail = control_->tail.load(std::memory_order_relaxed);
        uint32_t next_tail = (current_tail + 1) % RING_BUFFER_SIZE;
        uint32_t current_head = control_->head.load(std::memory_order_acquire);

        // Verifica se está cheio
        if (next_tail == current_head) {
            return false; // Buffer cheio
        }

        // Escreve o comando
        CommandPacket& packet = control_->commands[current_tail];
        packet.op_type = type;
        packet.param_ptr = param_ptr;
        packet.task_id = task_id;

        // Publica a escrita atualizando o tail
        control_->tail.store(next_tail, std::memory_order_release);
        return true;
    }

    // Função auxiliar para ver status (debug)
    void debug_print() {
        std::cout << "RingBuffer Head: " << control_->head.load()
                  << " Tail: " << control_->tail.load() << std::endl;
    }

private:
    RingBufferControl* control_;
    bool owns_memory_;
};

} // namespace host
} // namespace uhk

#endif // RING_BUFFER_MANAGER_H
