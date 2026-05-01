#ifndef UHK_ANALOG_RERAM_SIMULATOR_H
#define UHK_ANALOG_RERAM_SIMULATOR_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

namespace uhk {
namespace analog {

// =============================================================================
// Phase 5: Real Analog Physics Emulator
// =============================================================================
// Simulates V=IR with Gaussian Thermal Noise.
// =============================================================================

class ReRAMCrossbar {
public:
    ReRAMCrossbar(int rows, int cols) : rows_(rows), cols_(cols), dist_(0.0, 0.05) {
        rng_.seed(42);
        conductance_matrix_.resize(rows * cols, 0.0f);
    }

    void program_weights(const std::vector<float>& weights) {
        for (size_t i = 0; i < weights.size(); ++i) {
            conductance_matrix_[i] = std::abs(weights[i]) * 100e-6f; // 0-100 uS
        }
    }

    std::vector<float> compute_analog(const std::vector<float>& input_voltages, double& energy_joules) {
        std::vector<float> output_currents(cols_, 0.0f);

        for (int c = 0; c < cols_; ++c) {
            float current_sum = 0.0f;
            for (int r = 0; r < rows_; ++r) {
                float v = input_voltages[r];
                float g = conductance_matrix_[r * cols_ + c];

                // Add Noise (Thermal/Shot noise in analog domain)
                float noise = dist_(rng_) * 1e-6f;
                float i_cell = (v * g) + noise;

                current_sum += i_cell;

                // Real Energy accumulation (V*I*t)
                energy_joules += std::abs(v * i_cell) * 10e-9;
            }
            output_currents[c] = current_sum;
        }
        return output_currents;
    }

private:
    int rows_, cols_;
    std::vector<float> conductance_matrix_;
    std::mt19937 rng_;
    std::normal_distribution<float> dist_; // Gaussian Noise
};

} // namespace analog
} // namespace uhk

#endif // UHK_ANALOG_RERAM_SIMULATOR_H
