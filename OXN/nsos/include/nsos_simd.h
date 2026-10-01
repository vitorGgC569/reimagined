#pragma once
#include <cstdint>
#include <cmath>

namespace nsos {

// Platform Detection
#if defined(__aarch64__) || defined(__ARM_NEON)
    #define NSOS_ARCH_ARM
    #include <arm_neon.h>
// Use the AVX (__m256) packet path only when AVX is actually enabled (-mavx
// defines __AVX__).  Keep the x86 path on MSVC (its build relies on it).  A
// baseline build (-DNSOS_BASELINE_X86=ON, no -mavx) falls back to the scalar
// GENERIC path below so the binary runs on pre-Haswell CPUs.
#elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX__) || defined(_MSC_VER))
    #define NSOS_ARCH_X86
    #include <immintrin.h>
#else
    #define NSOS_ARCH_GENERIC
#endif

// ============================================================================
// BFloat16 Storage Type (SOTA 2026 Standard)
// ============================================================================
// Minimal wrapper to halve memory bandwidth usage.
// Conversion to float happens on-the-fly during compute.

struct alignas(2) bfloat16 {
    uint16_t bits;

    bfloat16() = default;
    bfloat16(float f) {
        // IEEE-754 float: Sign(1) | Exponent(8) | Mantissa(23)
        // BFloat16:       Sign(1) | Exponent(8) | Mantissa(7)
        union { float f; uint32_t i; } u = {f};
        if (f != f) {
            // NaN -> canonical bf16 quiet NaN, preserving the sign bit.  The
            // previous fixed +0x8000 add could turn a NaN into Inf.
            bits = static_cast<uint16_t>((u.i >> 16) | 0x0040u);
            return;
        }
        // Round half to even (the previous fixed +0x8000 bias was half-up, not
        // the "nearest even" the old comment claimed): the bias depends on the
        // resulting LSB.
        const uint32_t lsb = (u.i >> 16) & 1u;
        const uint32_t rounding_bias = 0x7FFFu + lsb;
        bits = static_cast<uint16_t>((u.i + rounding_bias) >> 16);
    }

    operator float() const {
        union { uint32_t i; float f; } u;
        u.i = (uint32_t)bits << 16;
        return u.f;
    }
};

// ============================================================================
// SIMD Abstraction Layer
// ============================================================================
// Unifies AVX2 (256-bit) and NEON (128-bit) under one API.

template <typename T>
struct SimdPacket;

// Float32 Packet
template <>
struct SimdPacket<float> {
#ifdef NSOS_ARCH_X86
    static constexpr int width = 8;
    __m256 v;
    
    SimdPacket() : v(_mm256_setzero_ps()) {}
    SimdPacket(__m256 x) : v(x) {}
    SimdPacket(float val) : v(_mm256_set1_ps(val)) {}
    
    static SimdPacket load(const float* p) { return _mm256_loadu_ps(p); }
    void store(float* p) const { _mm256_storeu_ps(p, v); }
    
    SimdPacket operator+(const SimdPacket& o) const { return _mm256_add_ps(v, o.v); }
    SimdPacket operator*(const SimdPacket& o) const { return _mm256_mul_ps(v, o.v); }
    // FMA: a*b + c
    static SimdPacket fma(const SimdPacket& a, const SimdPacket& b, const SimdPacket& c) {
        return _mm256_fmadd_ps(a.v, b.v, c.v);
    }
#elif defined(NSOS_ARCH_ARM)
    static constexpr int width = 4;
    float32x4_t v;
    
    SimdPacket() : v(vdupq_n_f32(0)) {}
    SimdPacket(float32x4_t x) : v(x) {}
    SimdPacket(float val) : v(vdupq_n_f32(val)) {}
    
    static SimdPacket load(const float* p) { return vld1q_f32(p); }
    void store(float* p) const { vst1q_f32(p, v); }
    
    SimdPacket operator+(const SimdPacket& o) const { return vaddq_f32(v, o.v); }
    SimdPacket operator*(const SimdPacket& o) const { return vmulq_f32(v, o.v); }
    static SimdPacket fma(const SimdPacket& a, const SimdPacket& b, const SimdPacket& c) {
        return vfmaq_f32(c.v, a.v, b.v); // NEON: c + a*b
    }
#else
    static constexpr int width = 1;
    float v;
    SimdPacket(float val) : v(val) {}
    static SimdPacket load(const float* p) { return *p; }
    void store(float* p) const { *p = v; }
    SimdPacket operator+(const SimdPacket& o) const { return v + o.v; }
    SimdPacket operator*(const SimdPacket& o) const { return v * o.v; }
    static SimdPacket fma(const SimdPacket& a, const SimdPacket& b, const SimdPacket& c) { return a.v * b.v + c.v; }
#endif
};

} // namespace nsos
