#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace pantheon {
namespace response {

    class LogitDistillation {
    public:
        // NTCE-KD: Non-Target Class Enhanced Knowledge Distillation
        // Weights non-target classes (incorrect classes) higher to capture dark knowledge.
        static float compute_ntce_loss(const std::vector<float>& student_logits,
                                     const std::vector<float>& teacher_logits,
                                     int target_class,
                                     float temperature = 1.0f,
                                     float beta = 2.0f) { // beta: amplification factor for non-targets
            if (student_logits.size() != teacher_logits.size()) {
                throw std::runtime_error("Logit size mismatch");
            }

            auto s_probs = softmax(student_logits, temperature);
            auto t_probs = softmax(teacher_logits, temperature);

            float loss = 0.0f;
            for (size_t i = 0; i < s_probs.size(); ++i) {
                float weight = (i == (size_t)target_class) ? 1.0f : beta;
                // KL Divergence term: T * log(T/S)
                // We focus on the cross-entropy part -T * log(S) weighted
                // Standard KD minimizes KL(T||S), which involves -sum(T * log(S))

                // Avoid log(0)
                float s_val = std::max(s_probs[i], 1e-9f);
                loss -= weight * t_probs[i] * std::log(s_val);
            }
            return loss;
        }

        // Universal Logit Distillation (ULD) via Sinkhorn Distance (Optimal Transport)
        // Computes Wasserstein distance between two distributions (logits -> softmax)
        static float compute_optimal_transport_loss(const std::vector<float>& student_logits,
                                                  const std::vector<float>& teacher_logits,
                                                  float temperature = 1.0f,
                                                  int max_iter = 100,
                                                  float epsilon = 0.1f) {
            auto s_probs = softmax(student_logits, temperature);
            auto t_probs = softmax(teacher_logits, temperature);

            // Cost matrix C (assuming semantic distance, e.g., index distance or uniform if unknown)
            // Ideally, we need embeddings to calculate cost.
            // For pure logits without embeddings, we assume a cost based on index difference is NOT useful.
            // ULD typically uses word embeddings distance.
            // Here we will implement the Sinkhorn algorithm assuming a uniform cost matrix
            // except diagonal (0 cost) to simulate "identity" matching,
            // OR we accept a Cost Matrix.
            // For valid standalone testing, we'll generate a dummy cost matrix: |i - j| (ordinal distance).

            size_t n = s_probs.size();
            std::vector<std::vector<float>> C(n, std::vector<float>(n));
            for(size_t i=0; i<n; ++i) {
                for(size_t j=0; j<n; ++j) {
                    C[i][j] = std::abs((float)i - (float)j); // Simple ordinal distance
                }
            }

            return sinkhorn(s_probs, t_probs, C, epsilon, max_iter);
        }

    private:
        static std::vector<float> softmax(const std::vector<float>& logits, float T) {
            std::vector<float> probs(logits.size());
            float max_val = *std::max_element(logits.begin(), logits.end());
            float sum = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i) {
                probs[i] = std::exp((logits[i] - max_val) / T);
                sum += probs[i];
            }
            for (size_t i = 0; i < logits.size(); ++i) {
                probs[i] /= sum;
            }
            return probs;
        }

        static float sinkhorn(const std::vector<float>& r, const std::vector<float>& c,
                            const std::vector<std::vector<float>>& M, float lambda, int max_iter) {
            // Sinkhorn-Knopp algorithm
            // r: distribution 1 (student)
            // c: distribution 2 (teacher)
            // M: Cost matrix
            // lambda: regularization (epsilon)

            size_t n = r.size();
            std::vector<std::vector<float>> K(n, std::vector<float>(n));
            for(size_t i=0; i<n; ++i)
                for(size_t j=0; j<n; ++j)
                    K[i][j] = std::exp(-M[i][j] / lambda);

            std::vector<float> u(n, 1.0f);
            std::vector<float> v(n, 1.0f);

            for (int iter = 0; iter < max_iter; ++iter) {
                // Update u
                for (size_t i = 0; i < n; ++i) {
                    float sum_kv = 0.0f;
                    for (size_t j = 0; j < n; ++j) sum_kv += K[i][j] * v[j];
                    if (sum_kv > 0) u[i] = r[i] / sum_kv;
                }

                // Update v
                for (size_t j = 0; j < n; ++j) {
                    float sum_ktu = 0.0f;
                    for (size_t i = 0; i < n; ++i) sum_ktu += K[i][j] * u[i];
                    if (sum_ktu > 0) v[j] = c[j] / sum_ktu;
                }
            }

            // Transport Distance: sum(P_ij * M_ij)
            float dist = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float P_ij = u[i] * K[i][j] * v[j];
                    dist += P_ij * M[i][j];
                }
            }
            return dist;
        }
    };

}
}
