#pragma once

#include <vector>
#include <cmath>

namespace pantheon {
namespace frontier {

    class CausalInvariance {
    public:
        // Invariance Loss: E[ (Rep(x) - Rep(do(x)))^2 ]
        // Ensures representation is invariant to interventions on nuisance variables.
        // original: Representation of x
        // intervention: Representation of x with style/environment changed

        static float compute_invariance_loss(const std::vector<float>& original,
                                           const std::vector<float>& intervention) {
            float loss = 0.0f;
            for (size_t i = 0; i < original.size(); ++i) {
                float diff = original[i] - intervention[i];
                loss += diff * diff;
            }
            return loss;
        }
    };

}
}
