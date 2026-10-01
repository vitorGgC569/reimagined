#ifndef UHK_BITNET_MATH_H
#define UHK_BITNET_MATH_H

#include <cstdint>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include "../gpu_backend.h"
#endif

namespace uhk {
namespace math {

// Constantes de codificação 2-bit
constexpr uint8_t TRIT_0 = 0b00;
constexpr uint8_t TRIT_P1 = 0b01;
constexpr uint8_t TRIT_M1 = 0b10;

// -----------------------------------------------------------------------------
// HOST SIDE HELPERS (Packing)
// -----------------------------------------------------------------------------
inline std::vector<uint8_t> pack_ternary_weights(const std::vector<int8_t>& weights) {
    std::vector<uint8_t> packed;
    packed.reserve((weights.size() + 3) / 4);
    uint8_t current_byte = 0;
    int bit_offset = 0;
    for (int8_t w : weights) {
        uint8_t code = TRIT_0;
        if (w == 1) code = TRIT_P1;
        else if (w == -1) code = TRIT_M1;
        current_byte |= (code << bit_offset);
        bit_offset += 2;
        if (bit_offset == 8) {
            packed.push_back(current_byte);
            current_byte = 0;
            bit_offset = 0;
        }
    }
    if (bit_offset > 0) packed.push_back(current_byte);
    return packed;
}

// -----------------------------------------------------------------------------
// DEVICE SIDE IMPLEMENTATION ("METAL")
// -----------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)

// Realização eficiente de Produto Escalar Ternário (1.58-bit)
// Pesos W compactados (2 bits), Ativações A (INT8)
// Lógica: Acc += A[i] * W[i]
// W[i] in {0, 1, -1}
// Otimização SWAR (SIMD Within A Register):
// Usar __dp4a (Dot Product 4-way) se convertermos W on-the-fly,
// OU usar máscaras e bit hacks para somar/subtrair.

// Método 1: DP4A com expansão on-the-fly (Melhor para INT8 Activations)
// Expande 4 pesos de 2 bits para 4 bytes INT8 em 1 registrador.
__device__ __forceinline__ int dot_product_ternary_int8(
    uint32_t packed_weights_8, uint32_t packed_activations_32) {
    // The low byte contains four 2-bit ternary codes.  The activation word
    // contains four signed INT8 lanes in little-endian order.  Code 0b11 is
    // deliberately decoded as zero, matching pack_ternary_weights' domain.
    int result = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const uint32_t code = (packed_weights_8 >> (2 * lane)) & 0x3u;
        const int weight = static_cast<int>(code & 1u) -
                           static_cast<int>(code >> 1u);
        const int activation = static_cast<int>(static_cast<int8_t>(
            (packed_activations_32 >> (8 * lane)) & 0xffu));
        result += weight * activation;
    }
    return result;
}

// Especialização __device__ completa
__device__ __forceinline__ int bitnet_dot_4(uint8_t w_packed, uint32_t a_packed) {
    int acc = 0;

    // Unpack on the fly & DP4A
    // w_packed: [w3 w2 w1 w0] (2 bits each)
    // a_packed: [a3 a2 a1 a0] (8 bits each)

    // Extrair códigos
    uint32_t c0 = (w_packed) & 0x3;
    uint32_t c1 = (w_packed >> 2) & 0x3;
    uint32_t c2 = (w_packed >> 4) & 0x3;
    uint32_t c3 = (w_packed >> 6) & 0x3;

    // Converter para valores int8 (-1, 0, 1)
    // Formula: (c & 1) - (c >> 1)
    int32_t v0 = (int32_t)(c0 & 1) - (int32_t)(c0 >> 1);
    int32_t v1 = (int32_t)(c1 & 1) - (int32_t)(c1 >> 1);
    int32_t v2 = (int32_t)(c2 & 1) - (int32_t)(c2 >> 1);
    int32_t v3 = (int32_t)(c3 & 1) - (int32_t)(c3 >> 1);

    // Reconstruir vetor de pesos de 32 bits (4x int8)
    // Cast para uint8 para garantir truncation correto (ex: -1 -> 0xFF)
    /*
    uint32_t w_vec =
        ((uint8_t)v0) |
        (((uint8_t)v1) << 8) |
        (((uint8_t)v2) << 16) |
        (((uint8_t)v3) << 24);

    // Hardware intrinsic dot product
    // res = a0*w0 + a1*w1 + a2*w2 + a3*w3 + c (=0)
    return __dp4a((int)w_vec, (int)a_packed, 0);
    */
    
    // Manual fallback compatible with all architectures (SM52+)
    // Unpack activations
    int8_t a0 = (int8_t)(a_packed & 0xFF);
    int8_t a1 = (int8_t)((a_packed >> 8) & 0xFF);
    int8_t a2 = (int8_t)((a_packed >> 16) & 0xFF);
    int8_t a3 = (int8_t)((a_packed >> 24) & 0xFF);
    
    return (v0 * a0) + (v1 * a1) + (v2 * a2) + (v3 * a3);
}

// Simulação para Host (Teste)
#else
inline int bitnet_dot_4(uint8_t w_packed, uint32_t a_packed) {
    int8_t* a = (int8_t*)&a_packed;
    int acc = 0;
    for(int i=0; i<4; ++i) {
        uint8_t c = (w_packed >> (i*2)) & 0x3;
        int val = (c & 1) - (c >> 1);
        acc += val * a[i];
    }
    return acc;
}
#endif // __CUDA_ARCH__

} // namespace math
} // namespace uhk

#endif // UHK_BITNET_MATH_H
