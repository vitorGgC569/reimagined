#include "../include/chrass_layer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

namespace nsos {

ChrassLayer::ChrassLayer(int dimension, const std::vector<float>& adjacency)
    : dim(dimension), t(0), bias(Tensor::zeros({dimension}), "bias") {

    // CSR Construction
    row_ptr.push_back(0);

    for(int r=0; r<dim; ++r) {
        long double row_sum = 0.0;
        for(int c=0; c<dim; ++c) {
            float val = adjacency[r * dim + c];
            if (std::abs(val) > 1e-6) row_sum += (long double)std::abs(val);
        }

        long double scale = (row_sum > 1e-12) ? (1.0 / row_sum) : 1.0;

        for(int c=0; c<dim; ++c) {
            float val = adjacency[r * dim + c];
            if (std::abs(val) > 1e-6) {
                float safe_weight = (float)((long double)val * scale);
                values.push_back(safe_weight);
                col_indices.push_back(c);
                m.push_back(0.0f);
                v.push_back(0.0f);
                grad_values.push_back(0.0f);
            }
        }
        row_ptr.push_back((int)values.size());
    }
}

Tensor ChrassLayer::forward(const Tensor& x) {
    int batch = x.shape[0];
    Tensor output = Tensor::zeros({batch, dim}, x.get_device());

    const float* in_ptr = x.data();
    float* out_ptr = output.data();
    const float* b_ptr = bias.data();

    // Parallelize batch
    #pragma omp parallel for
    for(int b=0; b<batch; ++b) {
        const float* in_row = in_ptr + b * dim;
        float* out_row = out_ptr + b * dim;

        for(int r=0; r<dim; ++r) {
            float sum = 0.0f;
            int start = row_ptr[r];
            int end = row_ptr[r+1];

            // CSR Loop (Linear Memory Access for Weights)
            for(int i=start; i<end; ++i) {
                sum += values[i] * in_row[col_indices[i]];
            }
            sum += b_ptr[r];

            // Aggressive sanitize for stability
            if (std::isnan(sum) || std::isinf(sum)) sum = 0.0f;
            if (sum > 100.0f) sum = 100.0f;
            if (sum < -100.0f) sum = -100.0f;
            out_row[r] = sum;
        }
    }
    return output;
}

void ChrassLayer::backward(const Tensor& grad_output, const Tensor& input) {
    // Sparse Backward Pass
    int batch = input.shape[0];
    const float* grad_ptr = grad_output.data();
    const float* in_ptr = input.data();

    // Reset Grads
    std::fill(grad_values.begin(), grad_values.end(), 0.0f);

    // Accumulate Grads (Serial for safety, parallelizable with atomic)
    for(int b=0; b<batch; ++b) {
        const float* g_row = grad_ptr + b * dim;
        const float* in_row = in_ptr + b * dim;

        for(int r=0; r<dim; ++r) {
            float g = g_row[r];
            int start = row_ptr[r];
            int end = row_ptr[r+1];

            for(int i=start; i<end; ++i) {
                grad_values[i] += g * in_row[col_indices[i]];
            }
        }
    }
}

void ChrassLayer::step(float lr) {
    t++;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;

    // Sparse AdamW
    int size = (int)values.size();
    float bc1 = 1.0f - std::pow(beta1, (float)t);
    float bc2 = 1.0f - std::pow(beta2, (float)t);

    for(int i=0; i<size; ++i) {
        float g = grad_values[i];

        // Update moments
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;

        // Apply Weight Decay
        values[i] -= lr * weight_decay * values[i];

        // Update Weight
        values[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

        // Sanitize weight
        if (std::isnan(values[i])) values[i] = 0.0f;
        if (values[i] > 10.0f) values[i] = 10.0f;
        if (values[i] < -10.0f) values[i] = -10.0f;
    }
}

std::vector<float> ChrassLayer::to_dense() {
    std::vector<float> dense(dim * dim, 0.0f);
    for(int r=0; r<dim; ++r) {
        int start = row_ptr[r];
        int end = row_ptr[r+1];
        for(int i=start; i<end; ++i) {
            dense[r * dim + col_indices[i]] = values[i];
        }
    }
    return dense;
}

} // namespace nsos
