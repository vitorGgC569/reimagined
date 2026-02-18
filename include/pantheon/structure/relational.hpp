#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

namespace pantheon {
namespace structure {

    class RelationalDistillation {
    public:
        // Distance-wise distillation loss
        // Measures if relative distances between samples are preserved
        static float compute_distance_loss(const std::vector<float>& s, const std::vector<float>& t, int batch_size, int dim) {
            auto dist_matrix_s = compute_pdist(s, batch_size, dim);
            auto dist_matrix_t = compute_pdist(t, batch_size, dim);

            // Normalize (often done by mean)
            float mean_s = mean(dist_matrix_s);
            float mean_t = mean(dist_matrix_t);

            float loss = 0.0f;
            for (size_t i = 0; i < dist_matrix_s.size(); ++i) {
                float diff = (dist_matrix_s[i] / mean_s) - (dist_matrix_t[i] / mean_t);
                loss += std::abs(diff); // Huber or L2 usually, keeping simple L1/L2
            }
            return loss;
        }

    private:
        static std::vector<float> compute_pdist(const std::vector<float>& feat, int batch_size, int dim) {
            std::vector<float> pdist;
            pdist.reserve(batch_size * batch_size);

            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < batch_size; ++j) {
                    float dist = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        float diff = feat[i*dim + k] - feat[j*dim + k];
                        dist += diff * diff;
                    }
                    pdist.push_back(std::sqrt(dist));
                }
            }
            return pdist;
        }

        static float mean(const std::vector<float>& v) {
            float sum = 0.0f;
            for (float f : v) sum += f;
            return v.empty() ? 0.0f : sum / v.size();
        }
    };

}
}
