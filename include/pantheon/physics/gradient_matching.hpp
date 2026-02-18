#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

namespace pantheon {
namespace physics {

    class GradientMatcher {
    public:
        // Computes Euclidean distance (L2 norm) between two gradient vectors
        // Returns the loss value
        static float compute_loss(const std::vector<float>& teacher_grads, const std::vector<float>& student_grads) {
            if (teacher_grads.size() != student_grads.size()) {
                throw std::runtime_error("Gradient size mismatch");
            }

            float sum_sq = 0.0f;
            for (size_t i = 0; i < teacher_grads.size(); ++i) {
                float diff = teacher_grads[i] - student_grads[i];
                sum_sq += diff * diff;
            }
            return sum_sq; // Or sqrt(sum_sq) depending on definition. Usually MSE uses mean sum sq.
                           // GKD often uses L2 norm squared. We'll return sum squared for now.
        }
    };

}
}
