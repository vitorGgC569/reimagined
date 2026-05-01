#pragma once

#include <vector>
#include <cmath>

namespace pantheon {
namespace cognition {

    class TheoryOfMind {
    public:
        // ToM Loss: Belief Prediction
        // Student must predict the "Belief State" (Hidden state) of an agent.
        // true_belief: The actual hidden state vector of the other agent (from Teacher).
        // pred_belief: The student's inference of that state.

        static float compute_tom_loss(const std::vector<float>& true_belief,
                                    const std::vector<float>& pred_belief) {
            float loss = 0.0f;
            for(size_t i=0; i<true_belief.size(); ++i) {
                // KL Divergence or Cross Entropy usually, L2 for simplicity here
                float diff = true_belief[i] - pred_belief[i];
                loss += diff * diff;
            }
            return loss;
        }
    };

}
}
