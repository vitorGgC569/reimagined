#ifndef INTERFERENCE_ENGINE_H
#define INTERFERENCE_ENGINE_H

#include <vector>
#include <complex>

namespace uhk {
namespace photonic {

using Complex = std::complex<float>;

// Mach-Zehnder Interferometer (MZI) Mesh Simulator
class InterferenceEngine {
public:
    InterferenceEngine(int size) : size_(size) {
        // Initialize identity matrix (perfect transmission)
        mesh_.resize(size * size, {0.0f, 0.0f});
        for (int i = 0; i < size; ++i) {
            mesh_[i * size + i] = {1.0f, 0.0f};
        }
    }

    // Apply a phase shift to a specific arm (simulating MZI tuning)
    void phase_shift(int row, int col, float theta) {
        if (row < size_ && col < size_) {
            mesh_[row * size_ + col] *= std::polar(1.0f, theta);
        }
    }

    std::vector<Complex> compute(const std::vector<Complex>& input_vector) {
        if (input_vector.size() != size_) return {};

        std::vector<Complex> output(size_, {0.0f, 0.0f});

        // Matrix-Vector Multiplication (Optical Passive)
        for (int i = 0; i < size_; ++i) {
            for (int j = 0; j < size_; ++j) {
                output[i] += mesh_[i * size_ + j] * input_vector[j];
            }
        }
        return output;
    }

private:
    int size_;
    std::vector<Complex> mesh_;
};

} // namespace photonic
} // namespace uhk

#endif // INTERFERENCE_ENGINE_H
