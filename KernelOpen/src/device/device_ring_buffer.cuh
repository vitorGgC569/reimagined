#ifndef UHK_DEVICE_RING_BUFFER_H
#define UHK_DEVICE_RING_BUFFER_H

#include "../common/uhk_types.h"

// Implementação "Metal" usando PTX Inline para consistência de sistema
// Requer GPU com suporte a System Scope Atomics (Pascal+)
// Otimizado para Hopper/Ampere com Zero-Copy

namespace uhk {
namespace device {

// Wrapper para leitura atômica de sistema (LD.GLOBAL.ACQUIRE.SYS)
__device__ __forceinline__ uint32_t atomic_load_system_acquire(const volatile uint32_t* addr) {
    uint32_t ret;
    asm volatile ("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(ret) : "l"(addr) : "memory");
    return ret;
}

// Wrapper para escrita atômica de sistema (ST.GLOBAL.RELEASE.SYS)
__device__ __forceinline__ void atomic_store_system_release(volatile uint32_t* addr, uint32_t val) {
    asm volatile ("st.global.release.sys.u32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}

// Consumidor do Ring Buffer usando semântica de memória estrita
__device__ __forceinline__ bool consume_command(RingBufferControl* control, CommandPacket& out_packet) {
    // 1. Ler o Tail (escrito pela CPU) com consistência de sistema
    // Isso garante que vejamos as escritas mais recentes da CPU que passaram pelo barramento PCIe/C2C
    uint32_t current_tail = atomic_load_system_acquire((const volatile uint32_t*)&control->tail);

    // 2. Ler o Head (local/cacheável, pois só GPU escreve, mas precisa ser atômico para coerência entre SMs se múltiplos consumers)
    // Usamos relaxed aqui pois a dependência de dados vem depois. Se houver apenas 1 warp consumidor global, pode ser normal.
    // Assumindo Múltiplos Consumidores (Persistent Threads competindo), precisamos de atomicCAS ou atomicAdd.
    // A arquitetura proposta é SINGLE CONSUMER LOGIC (Head gerenciado pela GPU).
    // Vamos ler relaxado primeiro para checar vazio.
    uint32_t current_head = atomic_load_system_acquire((const volatile uint32_t*)&control->head);

    if (current_head == current_tail) {
        return false; // Buffer Vazio
    }

    // 3. Garantir que os dados do pacote sejam visíveis antes de processar
    // A barreira 'acquire' acima no tail deve garantir ordenação com os dados dependentes se a CPU usou release.
    // threadfence_system() pode ser necessário se estivermos lendo dados massivos apontados pelo pacote.
    __threadfence_system();

    // 4. Copiar o pacote (Memória Unificada -> Registradores)
    // Otimização: Vectorized Load (128-bit) se alinhado
    // CommandPacket é 16 bytes (128 bits).
    const uint4* packet_ptr = (const uint4*)&control->commands[current_head];
    uint4 raw_packet = *packet_ptr; // Load vectorizado

    // Reinterpretar
    // Copia manual para evitar aliasing strict issues
    CommandPacket* temp_pkt = (CommandPacket*)&raw_packet;
    out_packet = *temp_pkt;

    // 5. Atualizar Head
    // Usamos Release para sinalizar à CPU que o slot está livre
    uint32_t next_head = (current_head + 1) % RING_BUFFER_SIZE;
    atomic_store_system_release((volatile uint32_t*)&control->head, next_head);

    return true;
}

} // namespace device
} // namespace uhk

#endif // UHK_DEVICE_RING_BUFFER_H
