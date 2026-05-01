#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace pantheon {
namespace social {

    class GameTheory {
    public:
        // Nash Distiller
        // Computes Regret against a Nash Equilibrium strategy.
        // nash_policy: The optimal probability distribution over actions.
        // student_policy: The student's policy.

        static float compute_nash_regret(const std::vector<float>& nash_policy,
                                       const std::vector<float>& student_policy) {
            // Regret -> Cross Entropy or KL Divergence with Nash
            // If Nash is optimal, deviation is regret.
            float regret = 0.0f;
            for(size_t i=0; i<nash_policy.size(); ++i) {
                float n = nash_policy[i];
                float s = student_policy[i];
                if (s > 1e-9f) {
                    regret -= n * std::log(s);
                }
            }
            return regret;
        }
    };

}
}
