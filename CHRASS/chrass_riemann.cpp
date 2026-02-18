#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <immintrin.h>
#include <omp.h>

// ============================================================================
// CHRASS RIEMANN ENGINE (V19-Z)
// Optimizing Riemann-Siegel summation via Hardware Isomorphism.
// Goal: Maximize Terms/Second using Linear Memory & AVX2.
// ============================================================================

// Constants
const double PI = 3.14159265358979323846;

// Precomputed tables for N (Hardware Isomorphism: Linear Memory)
// We precompute log(n) and 1/sqrt(n) because they are static.
// The dynamic part is 't'.
struct PrecomputedData {
    int n_max;
    std::vector<double> log_n;
    std::vector<double> inv_sqrt_n;

    PrecomputedData(int max_val) : n_max(max_val) {
        log_n.resize(max_val + 1);
        inv_sqrt_n.resize(max_val + 1);

        // Alignment is handled by std::vector usually, but for AVX strictness
        // we rely on compiler or use aligned_alloc (omitted for brevity in prototype)

        #pragma omp parallel for
        for (int i = 1; i <= n_max; ++i) {
            log_n[i] = std::log((double)i);
            inv_sqrt_n[i] = 1.0 / std::sqrt((double)i);
        }
    }
};

// AVX2 Cosine approximation (Taylor Series or Intrinsics if SVML available)
// Standard C++ cos is slow inside hot loops.
// For this proof-of-concept, we assume the compiler vectorizes std::cos with -Ofast
// or we implement a simple 4-lane cosine.
// Here we stick to scalar std::cos for correctness benchmark baseline,
// relying on -O3 -ffast-math to auto-vectorize.

class ZetaEngine {
public:
    // Riemann-Siegel Theta function approximation
    static double theta(double t) {
        return (t / 2.0) * std::log(t / (2.0 * PI)) - (t / 2.0) - (PI / 8.0) + (1.0 / (48.0 * t));
    }

    // Z(t) Calculation
    // Z(t) = 2 * Sum_{n=1}^{K} [ cos(theta(t) - t * ln(n)) / sqrt(n) ]
    // K = floor(sqrt(t / 2pi))
    static double compute_z(double t, const PrecomputedData& data) {
        int K = (int)std::sqrt(t / (2.0 * PI));
        if (K > data.n_max) K = data.n_max; // Safety cap

        double theta_t = theta(t);
        double sum = 0.0;

        // Vectorized Loop Candidate
        // We process data linearly: log_n[i] and inv_sqrt_n[i] are contiguous.
        // Cache misses are minimal.

        // Manual unroll to help compiler
        int i = 1;

        // AVX2 explicit logic would go here.
        // For portable prototype, we use OpenMP SIMD
        #pragma omp simd reduction(+:sum)
        for (i = 1; i <= K; ++i) {
            double phase = theta_t - t * data.log_n[i];
            sum += std::cos(phase) * data.inv_sqrt_n[i];
        }

        return 2.0 * sum;
    }
};

int main() {
    // Setup
    // t approx 10^12 -> K approx 400,000.
    // Let's optimize for t around 10^9 first.
    double t_start = 1000000000.0;
    int K_max = (int)std::sqrt(t_start / (2.0 * PI)) + 1000;

    std::cout << "=== CHRASS RIEMANN ZETA ENGINE ===" << std::endl;
    std::cout << "Target t: " << std::scientific << t_start << std::endl;
    std::cout << "Max Terms (K): " << K_max << std::endl;
    std::cout << "Initializing Memory Layout..." << std::endl;

    auto t_init_start = std::chrono::high_resolution_clock::now();
    PrecomputedData data(K_max);
    auto t_init_end = std::chrono::high_resolution_clock::now();
    std::cout << "Init Time: " << std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count() << " ms" << std::endl;

    // Search for a zero near t_start
    // A zero happens when Z(t) changes sign.

    std::cout << "Scanning for Zeros..." << std::endl;

    double t = t_start;
    double step = 0.1;
    double prev_Z = ZetaEngine::compute_z(t, data);

    int zeros_found = 0;
    auto t_scan_start = std::chrono::high_resolution_clock::now();

    // Increased steps to run for ~25 seconds
    int steps = 20000;
    for (int i = 0; i < steps; ++i) {
        t += step;
        double curr_Z = ZetaEngine::compute_z(t, data);

        if (prev_Z * curr_Z < 0) {
            std::cout << "Zero crossing detected between "
                      << std::fixed << std::setprecision(4) << (t - step)
                      << " and " << t << " | Z approx " << curr_Z << std::endl;
            zeros_found++;
        }
        prev_Z = curr_Z;
    }

    auto t_scan_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_scan_end - t_scan_start).count();

    // Performance Metrics
    double total_terms = (double)steps * std::sqrt(t_start / (2.0 * PI));
    std::cout << "\n>>> PERFORMANCE <<<" << std::endl;
    std::cout << "Total Terms: " << std::scientific << total_terms << std::endl;
    std::cout << "Time: " << total_ms << " ms" << std::endl;
    std::cout << "Terms/Sec: " << total_terms / (total_ms / 1000.0) << std::endl;

    return 0;
}
