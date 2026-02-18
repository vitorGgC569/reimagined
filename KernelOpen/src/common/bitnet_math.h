#ifndef UHK_BITNET_MATH_H
#define UHK_BITNET_MATH_H

#include <cstdint>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
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
__device__ __forceinline__ int dot_product_ternary_int8(uint32_t packed_weights_32, uint32_t packed_activations_32) {
    // packed_weights_32 contém 16 pesos (4 bytes, 4 pesos por byte)
    // packed_activations_32 contém 4 ativações (4 bytes)
    // Isso não bate. Precisamos processar 4 pesos x 4 ativacoes.
    // Entrada:
    //   packed_weights_8: 1 byte contendo 4 pesos (2-bit cada)
    //   packed_activations_32: 4 bytes contendo 4 ativações (int8)

    // Expandir 4 pesos de 2 bits para 4 ints de 8 bits
    // Pesos: 00=0, 01=1, 10=-1 (raw bits) -> mapear para int8 0, 1, -1
    // Mapeamento mágico:
    // 00 -> 0x00
    // 01 -> 0x01
    // 10 -> 0xFF (-1 em complemento de dois)
    // 11 -> N/A

    // Algoritmo de Expansão Bitwise (SWAR):
    // w = byte
    // vals = (w & 0x55) - ((w >> 1) & 0x55) ? Não, codificação não é padrão.
    // Nossa codificação: 00=0, 01=1, 10=-1

    // Expansão via LUT ou Bithacks. Vamos usar PRMT (Permute) ou lógica simples
    // Vamos processar byte a byte do input de pesos.
    // Assumindo input 'packed_weights_byte' (8 bits, 4 pesos) e 'activations' (32 bits, 4 int8)

    // Implementação "Metal" precisa ser correta e rápida.
    // Vamos usar __dp4a. Precisamos construir o registrador "weights_expanded".

    // Extração:
    // p0 = (w >> 0) & 3
    // p1 = (w >> 2) & 3
    // ...

    // Converter 2-bit code para 8-bit value:
    // Code 0 (00) -> 0
    // Code 1 (01) -> 1
    // Code 2 (10) -> -1 (0xFF)
    // Code 3 (11) -> 0

    // Lookup Table em registrador?
    // LUT: 0x0001FF00 (na ordem 11 10 01 00 -> 0 -1 1 0) ??
    // Bytes: 00, FF, 01, 00.
    // Shift dinâmico é lento.

    // Abordagem Aritmética sem branch:
    // map(c): return (c == 2) ? -1 : c;
    // ou: c - (c >= 2 ? 3 : 0) ? Nao.
    // c=0->0, c=1->1, c=2->-1.
    // (c & 1) - ((c >> 1) & 1) * 2 ???
    // 00: 0 - 0 = 0. OK.
    // 01: 1 - 0 = 1. OK.
    // 10: 0 - 1*2 = -2. ERRADO (queremos -1).
    // 10: c=2.

    // Tentar: (c == 1) - (c == 2)
    // Bitwise: (c & 1) - (c >> 1)
    // 00: 0 - 0 = 0
    // 01: 1 - 0 = 1
    // 10: 0 - 1 = -1
    // 11: 1 - 1 = 0
    // BINGO! Formula: (c & 1) - (c >> 1)

    // Agora aplicar SWAR para 4 pesos em 1 uint32.
    // w: byte de entrada (4 pesos)
    // w_expanded:

    // Passo 1: Isolar bits baixos (bit 0 de cada par)
    // mask_lo = 0b01010101 (0x55)
    // lo = w & 0x55

    // Passo 2: Isolar bits altos (bit 1 de cada par)
    // hi = (w >> 1) & 0x55

    // Agora temos lo e hi compactados. Precisamos espaçá-los para bytes.
    // Padrão de bits atual: l3 l2 l1 l0 (cada l é 1 bit na posição certa... espera, estão a cada 2 bits)
    // Temos 8 bits. Queremos espalhar para 32 bits.
    // Podemos usar __prmt ou pdep (Pascal+).
    // Mas vamos simplificar para loop unroll com __dp4a, compilador deve otimizar.

    return 0; // Placeholder para especialização abaixo
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
    uint32_t w_vec =
        ((uint8_t)v0) |
        (((uint8_t)v1) << 8) |
        (((uint8_t)v2) << 16) |
        (((uint8_t)v3) << 24);

    // Hardware intrinsic dot product
    // res = a0*w0 + a1*w1 + a2*w2 + a3*w3 + c (=0)
    return __dp4a(reinterpret_cast<int>(w_vec), reinterpret_cast<int>(a_packed), 0);
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
