#include "simd_dispatch.h"
#include <iostream>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

// Full NEON Implementation for V2 (Optimized)
// Implements 1.58-bit (Ternary) Matmul: Y = X @ W
// W elements are expected to be physically float {-1, 0, 1} but loaded as float.
// Optimized to avoid multiplication completely by using additions/subtractions.
void matmul_158bit_neon(float* res, const float* x, const float* w, int M, int K, int N) {
    // Assumptions: N is multiple of 4 for NEON 128-bit alignment (standard in ML).
    // If not, use padding or scalar cleanup (implemented below).
    
    // Iterate over M (Batch size)
    for (int i = 0; i < M; ++i) {
        
        // Zero out result accumulator row for safety (unless accumulation is desired)
        // memset(&res[i * N], 0, N * sizeof(float)); // Assuming caller handles initialization/bias

        // Iterate over K (Input features)
        for (int k = 0; k < K; ++k) {
            
            // Logically: res[i, :] += x[i, k] * w[k, :]
            // Since w is {-1, 0, 1}, we just add or subtract x[i, k].
            
            float x_val = x[i * K + k]; // Scalar input activation
            if (std::abs(x_val) < 1e-9f) continue; // Sparsity check (skip zero inputs)

            float32x4_t vx = vdupq_n_f32(x_val); // Broadcast x across vector
            float32x4_t vneg_x = vnegq_f32(vx);  // Precompute -x

            int j = 0;
            // Unroll loop 4x (process 16 floats per iter) for pipeline efficiency
            for (; j <= N - 16; j += 16) {
                // Prefetch next cache line
                __builtin_prefetch(&res[i * N + j + 16]);
                __builtin_prefetch(&w[k * N + j + 16]);

                // Block 1 (floats 0-3)
                float32x4_t vacc0 = vld1q_f32(&res[i * N + j]);
                float32x4_t vw0 = vld1q_f32(&w[k * N + j]);
                
                // Compare masks
                uint32x4_t mask_pos0 = vcgtq_f32(vw0, vdupq_n_f32(0.5f));  // W == 1
                uint32x4_t mask_neg0 = vcltq_f32(vw0, vdupq_n_f32(-0.5f)); // W == -1

                // Select: if W=1 add X, if W=-1 add -X, else add 0
                float32x4_t vdelta0 = vbslq_f32(mask_pos0, vx, vbslq_f32(mask_neg0, vneg_x, vdupq_n_f32(0.0f)));
                vacc0 = vaddq_f32(vacc0, vdelta0);
                vst1q_f32(&res[i * N + j], vacc0);

                // Block 2 (floats 4-7)
                float32x4_t vacc1 = vld1q_f32(&res[i * N + j + 4]);
                float32x4_t vw1 = vld1q_f32(&w[k * N + j + 4]);
                uint32x4_t mask_pos1 = vcgtq_f32(vw1, vdupq_n_f32(0.5f));
                uint32x4_t mask_neg1 = vcltq_f32(vw1, vdupq_n_f32(-0.5f));
                float32x4_t vdelta1 = vbslq_f32(mask_pos1, vx, vbslq_f32(mask_neg1, vneg_x, vdupq_n_f32(0.0f)));
                vacc1 = vaddq_f32(vacc1, vdelta1);
                vst1q_f32(&res[i * N + j + 4], vacc1);

                // Block 3 (floats 8-11)
                float32x4_t vacc2 = vld1q_f32(&res[i * N + j + 8]);
                float32x4_t vw2 = vld1q_f32(&w[k * N + j + 8]);
                uint32x4_t mask_pos2 = vcgtq_f32(vw2, vdupq_n_f32(0.5f));
                uint32x4_t mask_neg2 = vcltq_f32(vw2, vdupq_n_f32(-0.5f));
                float32x4_t vdelta2 = vbslq_f32(mask_pos2, vx, vbslq_f32(mask_neg2, vneg_x, vdupq_n_f32(0.0f)));
                vacc2 = vaddq_f32(vacc2, vdelta2);
                vst1q_f32(&res[i * N + j + 8], vacc2);

                // Block 4 (floats 12-15)
                float32x4_t vacc3 = vld1q_f32(&res[i * N + j + 12]);
                float32x4_t vw3 = vld1q_f32(&w[k * N + j + 12]);
                uint32x4_t mask_pos3 = vcgtq_f32(vw3, vdupq_n_f32(0.5f));
                uint32x4_t mask_neg3 = vcltq_f32(vw3, vdupq_n_f32(-0.5f));
                float32x4_t vdelta3 = vbslq_f32(mask_pos3, vx, vbslq_f32(mask_neg3, vneg_x, vdupq_n_f32(0.0f)));
                vacc3 = vaddq_f32(vacc3, vdelta3);
                vst1q_f32(&res[i * N + j + 12], vacc3);
            }

            // Handle remaining blocks of 4
            for (; j <= N - 4; j += 4) {
                float32x4_t vacc = vld1q_f32(&res[i * N + j]);
                float32x4_t vw = vld1q_f32(&w[k * N + j]);
                
                uint32x4_t mask_pos = vcgtq_f32(vw, vdupq_n_f32(0.5f));
                uint32x4_t mask_neg = vcltq_f32(vw, vdupq_n_f32(-0.5f));
                
                // Select delta using bitwise select
                float32x4_t vdelta = vbslq_f32(mask_pos, vx, vbslq_f32(mask_neg, vneg_x, vdupq_n_f32(0.0f)));
                
                // Accumulate
                vacc = vaddq_f32(vacc, vdelta);
                vst1q_f32(&res[i * N + j], vacc);
            }
            
            // Scalar Cleanup (for N % 4 != 0)
            for (; j < N; ++j) {
                float w_val = w[k * N + j];
                if (w_val > 0.5f) res[i * N + j] += x_val;
                else if (w_val < -0.5f) res[i * N + j] -= x_val;
            }
        }
    }
}
#endif
