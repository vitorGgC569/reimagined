#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace Aion {

    // Pilar C: Compressão (SIMD)
    // Bit-Packing: Compress 32-bit ints into N-bit streams.

    class BitPacker {
    public:
        // Scalar fallback implementation for sandbox safety
        static void pack_scalar(const uint32_t* in, uint64_t* out, size_t n, int bits);

        // Placeholder for AVX-512 implementation
        static void pack_avx512(const uint32_t* in, void* out, size_t n, int bits);
    };
}
