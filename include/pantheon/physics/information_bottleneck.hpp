#pragma once

#include <vector>
#include <cmath>
#include <numeric>

namespace pantheon {
namespace physics {

    class InformationBottleneck {
    public:
        // Variational Information Bottleneck (VIB) Loss
        // Minimizes: -I(Z;Y) + beta * I(Z;X)
        // In distillation: -I(S;T) + beta * I(S;X)
        // I(S;T) -> Maximize mutual info between Student and Teacher representations.
        // I(S;X) -> Minimize mutual info between Student and Input (Compression).

        // Simplified VIB for Gaussian representations:
        // L = MSE(mean_s, mean_t) + KL(N(mean_s, var_s) || N(0, I))
        // Here we implement the compression term: KL(S || Prior).
        // And the prediction term is handled by standard KD or CRD.
        // We return the IB regularization term: sum( -0.5 * (1 + log(var) - mean^2 - var) )

        static float compute_ib_compression_loss(const std::vector<float>& means,
                                               const std::vector<float>& log_vars,
                                               float beta = 1.0f) {

            float kl_div = 0.0f;
            size_t dim = means.size();

            for (size_t i = 0; i < dim; ++i) {
                float mu = means[i];
                float log_var = log_vars[i];
                float var = std::exp(log_var);

                // KL(N(mu, var) || N(0, 1))
                // = -0.5 * (1 + log(var) - mu^2 - var)
                kl_div += -0.5f * (1.0f + log_var - (mu * mu) - var);
            }

            return beta * (kl_div / dim);
        }
    };

}
}
