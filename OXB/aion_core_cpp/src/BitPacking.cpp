#include "../include/BitPacking.hpp"
#include <iostream>
#if defined(__AVX2__) || defined(_M_X64) || defined(__x86_64__)
#  include <immintrin.h>
#endif

// __int128 is GCC/Clang extension; MSVC needs manual 128-bit shift register.
// We provide a small portable Buf128 that has the same semantics as the
// original unsigned __int128 usage (only |=, <<, >>=64 by 64 are needed).
#if !defined(__SIZEOF_INT128__) && !defined(__GNUC__) && !defined(__clang__)
struct AionBuf128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
    // buffer |= (val << shift), where shift ∈ [0, 127]
    inline void or_shifted(uint64_t val, int shift) {
        if (shift >= 64) {
            hi |= (val << (shift - 64));
        } else {
            lo |= (val << shift);
            if (shift > 0) hi |= (val >> (64 - shift));
        }
    }
    inline uint64_t low64() const { return lo; }
    inline void shift_right_64() { lo = hi; hi = 0; }
};
#  define AION_USE_PORTABLE_128 1
#endif

namespace Aion {

    // Helper for 2-bit packing (specific optimization)
    // Packs 32 uint32_t values (each < 4) into 2 uint32_t (or 1 uint64_t) - wait.
    // 32 * 2 bits = 64 bits.
    // Input: 32 ints. Output: 1 uint64.

    // pack_2bit_avx2: kept as forward-declared no-op for ABI compatibility.
    // The AVX2 implementation was never finished -- author left dead `break`
    // followed by a scalar stub that never wrote to output.  Real users of
    // 2-bit packing should call BitPacker::pack_scalar(in, out, n, 2).
    // Removed __int128 leftover so MSVC builds.
    void pack_2bit_avx2(const uint32_t* in, uint64_t* out, size_t n) {
        BitPacker::pack_scalar(in, out, n, 2);
    }

    void BitPacker::pack_scalar(const uint32_t* in, uint64_t* out, size_t n, int bits) {
        size_t out_idx = 0;
#ifdef AION_USE_PORTABLE_128
        AionBuf128 buffer;
        int bits_in_buffer = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t val = (uint64_t)(in[i] & ((1ULL << bits) - 1));
            buffer.or_shifted(val, bits_in_buffer);
            bits_in_buffer += bits;
            while (bits_in_buffer >= 64) {
                out[out_idx++] = buffer.low64();
                buffer.shift_right_64();
                bits_in_buffer -= 64;
            }
        }
        if (bits_in_buffer > 0) {
            out[out_idx] = buffer.low64();
        }
#else
        unsigned __int128 buffer = 0;
        int bits_in_buffer = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t val = (uint64_t)(in[i] & ((1ULL << bits) - 1));
            buffer |= ((unsigned __int128)val << bits_in_buffer);
            bits_in_buffer += bits;
            while (bits_in_buffer >= 64) {
                out[out_idx++] = (uint64_t)buffer;
                buffer >>= 64;
                bits_in_buffer -= 64;
            }
        }
        if (bits_in_buffer > 0) {
            out[out_idx] = (uint64_t)buffer;
        }
#endif
    }

    void BitPacker::pack_avx512(const uint32_t* in, void* out, size_t n, int bits) {
        // Fallback to scalar for now as AVX-512 is not available/reliable here
        pack_scalar(in, (uint64_t*)out, n, bits);
    }
}
