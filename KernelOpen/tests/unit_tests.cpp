#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include "../src/host/uhk_runtime.h"
#include "../src/common/bitnet_math.h"
#include "../src/common/stream_k.h"

// NÃO INCLUIR Device Code diretamente se ele contém intrínsecos PTX/CUDA reais.
// O teste deve simular o comportamento observando a memória compartilhada.

// Mock para simular device consumption na CPU
void mock_gpu_consumer_thread(RingBufferControl* control, std::atomic<bool>& stop_flag) {
    while (!stop_flag) {
        // Simulação manual da lógica de consumo do device_ring_buffer
        // (Já que o código real agora usa PTX ASM)

        uint32_t head = control->head.load(std::memory_order_relaxed);
        uint32_t tail = control->tail.load(std::memory_order_acquire);

        if (head != tail) {
            // Processar
            CommandPacket packet = control->commands[head];

            // Atualizar head (simulando release da GPU)
            control->head.store((head + 1) % RING_BUFFER_SIZE, std::memory_order_release);

            // if (packet.op_type == OpType::SHUTDOWN) ...
             if (packet.op_type == OpType::SHUTDOWN) {
                break;
            }
        } else {
            std::this_thread::yield();
        }
    }
}

void test_ring_buffer() {
    std::cout << "Running Ring Buffer Test..." << std::endl;
    uhk::runtime::UniversalKernelRuntime runtime;

    std::atomic<bool> stop_mock(false);
    RingBufferControl* control = runtime.get_rb_manager()->get_control_ptr();

    // Inicia thread consumidora (simulando GPU)
    std::thread gpu_thread(mock_gpu_consumer_thread, control, std::ref(stop_mock));

    // Produz comandos
    int dummyA[1], dummyB[1], dummyC[1];
    for (int i = 0; i < 100; ++i) {
        runtime.submit_bitnet_gemm(128, 128, 128, dummyA, dummyB, dummyC);
    }

    // Finaliza
    runtime.shutdown();
    stop_mock = true;
    gpu_thread.join();

    std::cout << "Ring Buffer Test Passed!" << std::endl;
}

void test_bitnet_math() {
    std::cout << "Running BitNet Math Test..." << std::endl;

    // Test Packing
    std::vector<int8_t> weights = {1, -1, 0, 1, -1, 0, 0, 1}; // 8 pesos
    std::vector<uint8_t> packed = uhk::math::pack_ternary_weights(weights);

    assert(packed.size() == 2); // 4 pesos por byte -> 2 bytes

    // Verificar byte 0: 1, -1, 0, 1 -> 01, 10, 00, 01 (binary) -> LSB first or specified?
    // pack logic:
    // w0=1 -> 01
    // w1=-1 -> 10
    // w2=0 -> 00
    // w3=1 -> 01
    // result = (01) | (10<<2) | (00<<4) | (01<<6) = 0x01 | 0x08 | 0x00 | 0x40 = 0x49 (73)
    // Vamos verificar o valor:
    // std::cout << "Packed[0]: " << (int)packed[0] << std::endl;

    // Test Dot Product Logic
    int8_t activations[4] = {10, 10, 10, 10};
    uint32_t a_packed = *(uint32_t*)activations; // Reinterpretação simples para o teste

    // Bloco 1: 1, -1, 0, 1
    // Esperado: 1*10 + (-1)*10 + 0*10 + 1*10 = 10 - 10 + 0 + 10 = 10

    int result = uhk::math::bitnet_dot_4(packed[0], a_packed);
    assert(result == 10);

    std::cout << "BitNet Math Test Passed!" << std::endl;
}

void test_stream_k() {
    std::cout << "Running Stream-K Decomposition Test..." << std::endl;

    uhk::scheduler::StreamKConfig config;
    config.sm_count = 4;
    config.tile_m = 64;
    config.tile_n = 64;
    config.tile_k = 32;
    // Distributed/Phase 2 Params
    config.num_nodes = 1;
    config.node_id = 0;

    // Problema: M=128, N=128, K=64
    // Output Tiles: (128/64) * (128/64) = 2 * 2 = 4 tiles
    // K Iters per tile: 64/32 = 2
    // Total Work Units = 4 tiles * 2 iters = 8 units
    // Work per SM = 8 / 4 = 2 units.
    // Perfect split!

    auto assignments = uhk::scheduler::decompose_stream_k(128, 128, 64, config);

    assert(assignments.size() == 4);
    for (const auto& asmnt : assignments) {
        // Cada SM deve ter 2 unidades de trabalho (pode ser 1 tile full (2 iters) ou partes)
        int total_k = 0;
        for (const auto& t : asmnt.tiles) {
            total_k += (t.k_end - t.k_start);
        }
        assert(total_k == 2);
    }

    std::cout << "Stream-K Logic Test Passed!" << std::endl;
}

int main() {
    test_bitnet_math();
    test_stream_k();
    test_ring_buffer();
    return 0;
}
