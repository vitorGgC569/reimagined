#ifndef XOROSHIRO_H
#define XOROSHIRO_H

#include <cstdint>

#include <omp.h>
#include <vector>
#include <memory>
#include <mutex>

// Xoroshiro128++ (Blackman/Vigna)
// State size: 128 bits | Period: 2^128 - 1 | Speed: ~1ns/op

class Xoroshiro128PlusPlus {
    uint64_t s[2];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    // SplitMix64 Seeding
    Xoroshiro128PlusPlus(uint64_t seed = 42) {
        uint64_t z = (seed + 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[0] = z ^ (z >> 31);

        z = (s[0] + 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[1] = z ^ (z >> 31);
    }
    
    // Manual state init (for jump)
    Xoroshiro128PlusPlus(uint64_t s0, uint64_t s1) {
        s[0] = s0; s[1] = s1;
    }

    inline uint64_t next_u64() {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = rotl(s0 + s1, 17) + s0;

        s1 ^= s0;
        s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); 
        s[1] = rotl(s1, 28); 

        return result;
    }

    // Standard Uniform float [0, 1)
    // 2^53 = 9007199254740992
    inline float next_float() {
        // (x >> 11) * (1.0 / 9007199254740992.0)
        return (next_u64() >> 11) * 0x1.0p-53f; 
    }

    // Jump function (equivalent to 2^64 calls)
    // Used to spawn independent streams for parallel threads
    void jump() {
        static const uint64_t JUMP[] = { 0x2bd7a6a6e99c2ddc, 0x0992ccaf6a6fca05 };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            for(int b = 0; b < 64; b++) {
                if (JUMP[i] & (1ULL << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                }
                next_u64();
            }

        s[0] = s0;
        s[1] = s1;
    }
    
    // Long jump (2^96 calls)
    void long_jump() {
        static const uint64_t LONG_JUMP[] = { 0x360fd5f2cf8d5d99, 0x9c6e6877736c46e3 };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            for(int b = 0; b < 64; b++) {
                if (LONG_JUMP[i] & (1ULL << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                }
                next_u64();
            }

        s[0] = s0;
        s[1] = s1;
    }
};

// Thread-Safe RNG Manager for Parallel Execution
class ParallelRNG {
    std::vector<Xoroshiro128PlusPlus> thread_states;
    uint64_t global_seed;
    int max_threads;

public:
    ParallelRNG(uint64_t seed, int threads = 64) : global_seed(seed), max_threads(threads) {
        Xoroshiro128PlusPlus master_rng(seed);
        
        // Pre-generate states. Each thread gets a state jumped by 2^64 from the previous.
        // This guarantees no overlap for 2^64 generations/thread.
        for(int i=0; i<max_threads; ++i) {
            thread_states.push_back(master_rng);
            master_rng.jump(); 
        }
    }

    // Get the RNG instance for the current thread
    inline Xoroshiro128PlusPlus& get() {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid >= max_threads) tid = tid % max_threads; // Fallback wraparound
        return thread_states[tid];
    }
    
    // Reseed all threads consistently
    void reseed(uint64_t new_seed) {
         Xoroshiro128PlusPlus master_rng(new_seed);
         for(int i=0; i<max_threads; ++i) {
            thread_states[i] = master_rng;
            master_rng.jump(); 
        }
    }
};

// Global instance defined in nsos_sdk.cpp or similar
extern std::unique_ptr<ParallelRNG> GLOBAL_PARALLEL_RNG;

#endif
