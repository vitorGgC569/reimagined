#ifndef UHK_COMMON_H
#define UHK_COMMON_H

#include <cstdint>
#include <atomic>

// Definições de tamanhos e constantes
constexpr uint32_t RING_BUFFER_SIZE = 1024; // Número de slots de comando
constexpr uint32_t CACHE_LINE_SIZE = 64;    // Para alinhamento e evitar false sharing

// Tipos de Operações Suportadas pelo Kernel Universal
enum class OpType : uint8_t {
    NO_OP = 0,
    GEMM_BITNET = 1,   // Multiplicação de Matriz 1.58-bit
    SCAN_MAMBA = 2,    // Parallel Scan (SSM)
    FLASH_ATTN = 3,    // FlashAttention-3
    GEMM_SYNTHETIC = 4,// V10: Compute-Bound Stress Test (Xorshift + Tensor Core)
    SHUTDOWN = 255     // Comando para encerrar o kernel persistente
};

// Estrutura de Comando (Packet)
// Deve ser alinhada para garantir leituras atômicas eficientes ou consistência
struct alignas(16) CommandPacket {
    OpType op_type;
    uint8_t reserved[7];
    uint64_t param_ptr; // Ponteiro para descritor específico da operação (na memória unificada)
    uint64_t task_id;   // ID para rastreamento
};

// Estrutura de Controle do Ring Buffer
// Reside em Memória Unificada (Pinned Host Memory)
struct RingBufferControl {
    // Ponteiros atômicos para cabeça e cauda
    // alignas(CACHE_LINE_SIZE) para evitar false sharing entre CPU (produtor) e GPU (consumidor)
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> tail; // Escrito pela CPU, Lido pela GPU
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> head; // Escrito pela GPU, Lido pela CPU

    // O buffer de dados em si
    CommandPacket commands[RING_BUFFER_SIZE];

    // Métricas de Performance (Atualizadas pela GPU/Consumidor)
    // Using volatile instead of std::atomic for easier sharing with CUDA Kernel (C++17 style)
    // Host reads, Device writes.
    alignas(CACHE_LINE_SIZE) volatile float throughput; // TOPS
    alignas(CACHE_LINE_SIZE) volatile float latency;    // micro-seconds
    alignas(CACHE_LINE_SIZE) volatile uint32_t active_sms;
    alignas(CACHE_LINE_SIZE) volatile uint64_t tasks_processed;
};

// Descritores de Tarefas Específicas

struct BitNetGEMMParams {
    int M, N, K;
    const void* A_ptr; // Dados empacotados (ternary)
    const void* B_ptr; // Dados empacotados (ternary) ou INT8
    void* C_ptr;       // Saída (FP16/BF16/FP32)
    float alpha;
    float beta;
};

struct StreamKParams {
    uint32_t total_tiles;
    uint32_t sm_count;
    // Parâmetros adicionais para configuração do Stream-K
};

#endif // UHK_COMMON_H
