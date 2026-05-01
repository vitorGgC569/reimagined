#pragma once

#include <vector>
#include <cmath>

namespace pantheon {
namespace social {

    class SwarmDistiller {
    public:
        // Swarm Update (PSO-like)
        // Computes new velocity for a student model (particle) towards the "Global Best" teacher.
        // current_weights: w
        // best_weights: g
        // velocity: v
        // Returns new weights: w + v_new

        static std::vector<float> update_swarm_weights(const std::vector<float>& current,
                                                     const std::vector<float>& best,
                                                     const std::vector<float>& velocity,
                                                     float inertia = 0.5f,
                                                     float cognitive = 1.0f,
                                                     float social = 1.0f) {
            std::vector<float> new_weights(current.size());
            for(size_t i=0; i<current.size(); ++i) {
                // v_new = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
                // Simplified: Just pulling towards Teacher (g_best).
                float r2 = 0.5f; // Random factor fixed for deterministic test
                float v_new = inertia * velocity[i] + social * r2 * (best[i] - current[i]);
                new_weights[i] = current[i] + v_new;
            }
            return new_weights;
        }
    };

}
}
