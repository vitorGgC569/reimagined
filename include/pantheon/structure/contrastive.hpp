#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace pantheon {
namespace structure {

    class ContrastiveDistillation {
    public:
        // Computes Contrastive Loss (InfoNCE style) for a batch
        // Simplified implementation: assumes positive pairs are at same index
        // z_s: Student features (Batch x Dim) - flattened
        // z_t: Teacher features (Batch x Dim) - flattened
        // temperature: scaling factor
        static float compute_loss(const std::vector<float>& z_s, const std::vector<float>& z_t, int batch_size, int dim, float temperature = 0.07f) {
            if (z_s.size() != z_t.size() || z_s.size() != (size_t)(batch_size * dim)) {
                throw std::runtime_error("Feature dimension mismatch in CRD");
            }

            float total_loss = 0.0f;

            for (int i = 0; i < batch_size; ++i) {
                // Positive pair (s_i, t_i)
                float pos_sim = cosine_similarity(z_s, z_t, i, i, dim);
                float numerator = std::exp(pos_sim / temperature);

                float denominator = 0.0f;
                // Negative pairs (s_i, t_j) for all j
                for (int j = 0; j < batch_size; ++j) {
                    float sim = cosine_similarity(z_s, z_t, i, j, dim);
                    denominator += std::exp(sim / temperature);
                }

                total_loss += -std::log(numerator / denominator);
            }

            return total_loss / batch_size;
        }

    private:
        static float cosine_similarity(const std::vector<float>& A, const std::vector<float>& B, int idx_A, int idx_B, int dim) {
            float dot = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;

            int offset_A = idx_A * dim;
            int offset_B = idx_B * dim;

            for (int k = 0; k < dim; ++k) {
                float a = A[offset_A + k];
                float b = B[offset_B + k];
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }

            if (norm_a == 0 || norm_b == 0) return 0.0f;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }
    };

}
}
