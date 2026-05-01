#pragma once
#include <vector>
#include <cstdint>
#include <cstddef> // Fix size_t

namespace Aion {

    // Pilar A: A "Mente" (RMI)
    // Modelo Linear Simples: y = mx + b
    class LinearModel {
    public:
        double m;
        double b;
        uint64_t max_error;

        LinearModel();

        void train(const std::vector<double>& keys, const std::vector<double>& offsets);

        double predict(double key) const;

        // New Batch Prediction for SIMD
        void predict_batch(const double* keys, double* results, size_t n) const;
    };

}
