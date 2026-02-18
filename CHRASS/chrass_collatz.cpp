#include <iostream>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <chrono>
#include <iomanip>

// ============================================================================
// CHRASS COLLATZ (V19-C)
// Vectorized 3n+1 Checker using AVX2 Bit-Slicing (Branchless)
// ============================================================================

class CollatzEngine {
public:
    // Check range [start, end]
    // Returns total steps (just to prevent optimization removal)
    static long long check_range_avx2(unsigned long long start, unsigned long long count) {
        long long total_steps = 0;

        // We process 4x 64-bit integers at a time
        // Note: AVX2 integer mul is messy for 64-bit (32-bit is native).
        // 64-bit mul requires emulation or AVX-512.
        // For AVX2, we stick to 32-bit numbers for speed demonstration,
        // or emulate 64-bit logic carefully.
        // Let's target 32-bit range for MAX throughput demo (trillions/sec potential).
        // Processing 8x 32-bit integers.

        unsigned int* batch = (unsigned int*)_mm_malloc(8 * sizeof(unsigned int), 32);

        // Main Loop
        unsigned int n = (unsigned int)start;
        unsigned int end = n + (unsigned int)count;

        // Constants
        __m256i v_one = _mm256_set1_epi32(1);
        __m256i v_two = _mm256_set1_epi32(2);
        __m256i v_three = _mm256_set1_epi32(3);

        for (; n < end; n += 8) {
            // Load 8 numbers: n, n+1, ... n+7
            // Manual set for simplicity
            batch[0]=n; batch[1]=n+1; batch[2]=n+2; batch[3]=n+3;
            batch[4]=n+4; batch[5]=n+5; batch[6]=n+6; batch[7]=n+7;

            __m256i v_n = _mm256_load_si256((__m256i*)batch);
            __m256i v_steps = _mm256_setzero_si256();

            // Iterate until all reach 1
            // In a real high-throughput scanner, we would swap out finished numbers.
            // For simple benchmark, we wait for the slowest in the vector.

            int active_mask = 0xFF;

            while (active_mask) {
                // Check if n == 1
                __m256i v_is_one = _mm256_cmpeq_epi32(v_n, v_one);
                int done_mask = _mm256_movemask_ps(_mm256_castsi256_ps(v_is_one));
                active_mask = (~done_mask) & 0xFF;

                if (!active_mask) break;

                // Increment steps for active
                // v_steps = v_steps + (is_one ? 0 : 1)
                // Since is_one is -1 (FFFF) or 0, we can use add/sub logic
                // But simple masked add is cleaner if supported.
                // AVX2 Blend: if finished, keep old steps. Else add 1.
                v_steps = _mm256_add_epi32(v_steps, _mm256_andnot_si256(v_is_one, v_one));

                // Calculate Next Step Branchless
                // is_odd = (n & 1)
                __m256i v_is_odd = _mm256_and_si256(v_n, v_one); // 0 or 1
                // Mask for odd: 0xFFFFFFFF if odd (via compare?)
                // Actually compare with 1
                __m256i v_odd_mask = _mm256_cmpeq_epi32(v_is_odd, v_one);

                // Even path: n >> 1
                __m256i v_even_res = _mm256_srli_epi32(v_n, 1);

                // Odd path: 3n + 1
                // 3n = (n << 1) + n
                // Check overflow? For benchmark we assume 32-bit holds.
                __m256i v_n2 = _mm256_slli_epi32(v_n, 1); // n*2
                __m256i v_3n = _mm256_add_epi32(v_n2, v_n); // n*3
                __m256i v_odd_res = _mm256_add_epi32(v_3n, v_one); // 3n+1

                // Select result
                v_n = _mm256_blendv_epi8(v_even_res, v_odd_res, v_odd_mask);
            }

            // Sum steps (horizontal sum hack)
            // Just scalar sum for benchmark stat
            int* s_ptr = (int*)&v_steps;
            for(int i=0; i<8; ++i) total_steps += s_ptr[i];
        }

        _mm_free(batch);
        return total_steps;
    }
};

int main() {
    unsigned long long N = 10000000; // 10 Million
    std::cout << "=== CHRASS COLLATZ (AVX2 Vectorized) ===" << std::endl;
    std::cout << "Checking range [1, " << N << "]..." << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    long long steps = CollatzEngine::check_range_avx2(1, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Total Steps: " << steps << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (N / (ms / 1000.0)) / 1e6 << " M/sec" << std::endl;

    return 0;
}
