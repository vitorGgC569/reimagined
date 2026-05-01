#pragma once

#include <vector>
#include <random>
#include <numeric>

namespace pantheon {
namespace frontier {

    class PrivacyEngine {
    public:
        // PATE: Private Aggregation of Teacher Ensembles
        // Adds Laplace noise to vote counts.
        // votes: vector of counts per class
        // epsilon: privacy budget (inverse of noise scale)

        static std::vector<float> aggregate_pate(const std::vector<int>& votes, float epsilon) {
            std::vector<float> noisy_votes(votes.size());
            std::mt19937 gen(42);
            // Laplace distribution: P(x) = 1/2b * exp(-|x|/b), b = 1/epsilon
            std::exponential_distribution<float> exp_dist(epsilon);

            for(size_t i=0; i<votes.size(); ++i) {
                // Laplace(b) ~ Exp(1/b) - Exp(1/b)
                float n1 = exp_dist(gen);
                float n2 = exp_dist(gen);
                float laplace_noise = n1 - n2;

                noisy_votes[i] = votes[i] + laplace_noise;
            }
            return noisy_votes;
        }
    };

}
}
