#include "../include/LinearModel.hpp"
#include <cmath>
#include <iostream>
#include <immintrin.h>

namespace Aion {

    LinearModel::LinearModel() : m(0), b(0), max_error(0) {}

    void LinearModel::train(const std::vector<double>& keys, const std::vector<double>& offsets) {
        size_t n = keys.size();
        if (n == 0) return;
        if (n == 1) {
            m = 0; b = offsets[0]; return;
        }

        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (size_t i = 0; i < n; ++i) {
            sum_x += keys[i];
            sum_y += offsets[i];
            sum_xy += keys[i] * offsets[i];
            sum_xx += keys[i] * keys[i];
        }

        m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        b = (sum_y - m * sum_x) / n;

        double max_err = 0;
        // Optimization: Use scalar loop for error finding (memory bound usually)
        for (size_t i = 0; i < n; ++i) {
            double pred = predict(keys[i]);
            double diff = std::abs(offsets[i] - pred);
            if (diff > max_err) max_err = diff;
        }
        max_error = static_cast<uint64_t>(std::ceil(max_err));
    }

    // Optimized Prediction with FMA (AVX2/FMA3)
    // Note: The signature takes a single key, so intrinsic optimization happens
    // when processing BATCHES (which we should add to API) or if the compiler autovectorizes loops calling this.
    // To strictly "implement intrinsics", we should expose a batch predict method or ensure the loop in benchmark uses them.
    // However, the user asked to optimize the LinearModel.
    // Let's keep the scalar `predict` fast (it compiles to FMA scalar usually).

    double LinearModel::predict(double key) const {
        // Compilers usually emit vfmadd213sd for this with -O3 -march=native
        return m * key + b;
    }

    // New: Batch Prediction for explicit SIMD usage
    void LinearModel::predict_batch(const double* keys, double* results, size_t n) const {
        size_t i = 0;

        // AVX2 (4 doubles per register)
#if defined(__AVX2__) || defined(__FMA__)
        __m256d mm_m = _mm256_set1_pd(m);
        __m256d mm_b = _mm256_set1_pd(b);

        for (; i + 4 <= n; i += 4) {
            __m256d k = _mm256_loadu_pd(keys + i);
            // res = m * k + b
            __m256d res = _mm256_fmadd_pd(mm_m, k, mm_b);
            _mm256_storeu_pd(results + i, res);
        }
#endif
        // Tail
        for (; i < n; ++i) {
            results[i] = m * keys[i] + b;
        }
    }
}
