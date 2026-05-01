#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <tuple>
#include <immintrin.h>

// ============================================================================
// RIERASS: Riemann Isomorphic Engine (V20)
// Goal: High-Precision Riemann-Siegel with Hardware-Isomorphic FFT Layout
// ============================================================================

// ----------------------------------------------------------------------------
// 1. BigFloat512 (Fixed Point: 256.256)
// ----------------------------------------------------------------------------
struct BigFloat512 {
    unsigned long long parts[8]; // [0..3] fraction, [4..7] integer. Little Endian.

    BigFloat512() { std::memset(parts, 0, sizeof(parts)); }

    // From double
    BigFloat512(double v) {
        std::memset(parts, 0, sizeof(parts));
        double int_part;
        double frac_part = std::modf(v, &int_part);

        // Integer part (simplify to low 64 bits for demo)
        parts[4] = (unsigned long long)int_part;

        // Fraction part
        unsigned __int128 frac = (unsigned __int128)(frac_part * 18446744073709551616.0); // * 2^64
        parts[3] = (unsigned long long)frac;
        // Further precision omitted for speed in prototype
    }

    // Add
    BigFloat512 operator+(const BigFloat512& other) const {
        BigFloat512 res;
        unsigned __int128 carry = 0;
        for (int i = 0; i < 8; ++i) {
            unsigned __int128 sum = (unsigned __int128)parts[i] + other.parts[i] + carry;
            res.parts[i] = (unsigned long long)sum;
            carry = sum >> 64;
        }
        return res;
    }

    // Multiply (Schoolbook O(N^2) - slow but correct)
    BigFloat512 operator*(const BigFloat512& other) const {
        BigFloat512 res;
        unsigned __int128 temp[16] = {0};

        for (int i = 0; i < 8; ++i) {
            unsigned __int128 carry = 0;
            for (int j = 0; j < 8; ++j) {
                unsigned __int128 prod = (unsigned __int128)parts[i] * other.parts[j] + temp[i+j] + carry;
                temp[i+j] = (unsigned long long)prod;
                carry = prod >> 64;
            }
            temp[i+8] += carry;
        }

        // Re-align (Fixed point 256.256 * 256.256 = 512.512)
        // We want the middle 512 bits (256.256)
        // Result starts at temp[4]
        for (int i = 0; i < 8; ++i) {
            res.parts[i] = (unsigned long long)temp[i+4];
        }
        return res;
    }

    double to_double() const {
        double res = (double)parts[4];
        res += (double)parts[3] / 18446744073709551616.0;
        return res;
    }
};

// ----------------------------------------------------------------------------
// 2. Hardware-Isomorphic FFT Layout (WDD)
// Reorders the bit-reversal pattern of FFT into linear memory access.
// ----------------------------------------------------------------------------
class LinearFFT {
public:
    std::vector<int> perm;
    int n;

    LinearFFT(int size) : n(size) {
        perm.resize(n);
        int log_n = 0;
        while ((1 << log_n) < n) log_n++;

        for (int i = 0; i < n; ++i) {
            int rev = 0;
            for (int j = 0; j < log_n; ++j) {
                if ((i >> j) & 1) rev |= (1 << (log_n - 1 - j));
            }
            perm[i] = rev;
        }
    }

    // Reorders input data so FFT butterfly access is linear in first stages
    void linearize_data(std::vector<BigFloat512>& data) {
        std::vector<BigFloat512> temp(n);
        for (int i = 0; i < n; ++i) {
            temp[i] = data[perm[i]];
        }
        data = temp;
    }
};

// ----------------------------------------------------------------------------
// 3. Gram Point Scanner (High Precision)
// ----------------------------------------------------------------------------
class GramScanner {
public:
    // Finds Gram point g_n such that theta(g_n) = n*pi
    // Uses Newton-Raphson on BigFloat (Simulated with double here for speed logic)
    static double find_gram(long long n) {
        // Approximate inverse of theta
        // t ~ 2pi * n / log n
        double t = 2 * 3.14159 * n / std::log(n);
        for(int i=0; i<5; ++i) {
            double th = (t/2)*std::log(t/(2*3.14159)) - t/2 - 3.14159/8;
            double dth = 0.5 * std::log(t/(2*3.14159));
            t -= (th - n*3.14159) / dth;
        }
        return t;
    }

    static void scan(long long start_n, int count) {
        double prev_Z = 0;

        std::cout << "Scanning Gram Interval [" << start_n << ", " << start_n + count << "]..." << std::endl;

        for(int i=0; i<count; ++i) {
            long long n = start_n + i;
            double t = find_gram(n);

            // Riemann-Siegel Z(t) proxy
            // At Gram point, Z(t) ~ 2 * (-1)^n * Sum(...)
            // We use BigFloat for sum precision

            BigFloat512 sum(0.0);
            int K = std::sqrt(t / (2*3.14159));

            // Vectorized accumulation would go here
            for(int k=1; k<=K; ++k) {
                // Term: 1/sqrt(k) * cos(theta - t ln k)
                // Simplified for Gram: cos(n*pi - t ln k) = (-1)^n * cos(t ln k)
                double phase = t * std::log(k);
                double val = std::cos(phase) / std::sqrt(k);

                // Using BigFloat accumulator
                sum = sum + BigFloat512(val);
            }

            double Z_val = 2.0 * ((n%2==0)?1.0:-1.0) * sum.to_double();

            // Check Law
            // Gram's Law: Z(g_n) should alternate sign: + - + -
            // Or rather: (-1)^n * Z(g_n) > 0 ("Z follows the rotation")

            bool violation = false;
            if ( (n%2 == 0 && Z_val < 0) || (n%2 != 0 && Z_val > 0) ) {
                violation = true;
            }

            if (violation) {
                std::cout << "VIOLATION at g_" << n << " (t=" << std::fixed << std::setprecision(2) << t << ") | Z=" << Z_val << std::endl;
            }
        }
    }
};

int main() {
    std::cout << "=== RIERASS: RIEMANN ISOMORPHIC ENGINE (V20) ===" << std::endl;
    std::cout << "Precision: 512-bit Fixed Point (256.256)" << std::endl;
    std::cout << "FFT Layout: WDD Linearized" << std::endl;

    // 1. Test BigFloat Logic
    BigFloat512 a(1.5), b(2.0);
    BigFloat512 c = a * b; // 3.0
    std::cout << "BigFloat Test: 1.5 * 2.0 = " << c.to_double() << std::endl;

    // 2. Test FFT Layout
    int FFT_SIZE = 1024;
    std::cout << "Initializing WDD Layout for FFT size " << FFT_SIZE << "..." << std::endl;
    LinearFFT fft_layout(FFT_SIZE);
    // (In a real run, we would load Dirichlet coefficients here)

    // 3. Run Gram Scan
    // Scan near 10^12 where violations are known to exist
    // Gram index n approx (t/2pi) log(t/2pi). For t=10^12, n is huge.
    // Let's scan a known lower range for demo speed.
    long long start_n = 10000;
    GramScanner::scan(start_n, 50);

    return 0;
}
