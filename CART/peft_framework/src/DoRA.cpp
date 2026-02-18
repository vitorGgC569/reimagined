#include "../include/DoRA.h"
#include "../include/TensorOps.h"
#include <iostream>
#include <random>
#include <cmath>

DoRALayer::DoRALayer(int input_dims, int output_dims, int rank)
    : m_input_dims(input_dims), m_output_dims(output_dims), m_rank(rank) {
    m_A = std::make_unique<Tensor>(rank, input_dims);
    m_B = std::make_unique<Tensor>(output_dims, rank);
    m_m = std::make_unique<Tensor>(1, output_dims);

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0 / std::sqrt(input_dims));

    for (int i = 0; i < m_A->getRows(); ++i) {
        for (int j = 0; j < m_A->getCols(); ++j) {
            m_A->at(i, j) = distribution(generator);
        }
    }
    for (int i = 0; i < m_B->getRows(); ++i) {
        for (int j = 0; j < m_B->getCols(); ++j) {
            m_B->at(i, j) = 0.0f;
        }
    }
    // Initialize m to 1 temporarily; it should ideally match the norm of W0.
    // We will set it properly when W0 is set if possible, or here.
    for (int j = 0; j < m_output_dims; ++j) {
        m_m->at(0, j) = 1.0f;
    }
}

void DoRALayer::setBaseWeights(const Tensor& weights) {
    m_W0 = std::make_unique<Tensor>(weights);

    // Initialize m to the column-wise L2 norm of W0
    // W0 is (output_dim, input_dim) usually in this repo based on LoRA
    // Let's verify LoRA W0 usage.
    // LoRA: W_eff = W0 + BA.
    // Dimensions: W0 is (out, in).
    // Row-wise or Col-wise?
    // LoRA forward: input * W_eff^T.
    // Usually weights are (out_features, in_features).
    // So each row is a neuron's weights.
    // Norm should be per output neuron (per row of W).

    for (int i = 0; i < m_output_dims; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < m_input_dims; ++j) {
            float val = m_W0->at(i, j);
            sum_sq += val * val;
        }
        m_m->at(0, i) = std::sqrt(sum_sq);
    }
}

Tensor DoRALayer::forward(const Tensor& input) {
    if (!m_W0) {
        throw std::runtime_error("Base weights W0 must be set before forward pass.");
    }
    m_last_input = std::make_unique<Tensor>(input);

    // V = W0 + B * A
    Tensor delta_W = TensorOps::multiply(*m_B, *m_A);
    Tensor V = TensorOps::add(*m_W0, delta_W);
    m_V = std::make_unique<Tensor>(V);

    // Calculate Norm of V (per row, since W is out x in)
    // and normalize V -> V_bar
    Tensor V_bar(m_output_dims, m_input_dims);
    for (int i = 0; i < m_output_dims; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < m_input_dims; ++j) {
            sum_sq += V.at(i, j) * V.at(i, j);
        }
        float norm = std::sqrt(sum_sq) + 1e-9f; // avoid div by zero

        // Apply DoRA scaling: W' = m * (V / ||V||)
        float scale = m_m->at(0, i) / norm;
        for (int j = 0; j < m_input_dims; ++j) {
            V_bar.at(i, j) = V.at(i, j) * scale;
        }
    }
    m_V_norm = std::make_unique<Tensor>(V_bar);

    // Forward: Y = X * W'^T
    Tensor W_prime_T = V_bar.transpose();
    return TensorOps::multiply(input, W_prime_T);
}

void DoRALayer::backward(const Tensor& upstream_grad) {
    if (!m_last_input || !m_V) {
        throw std::runtime_error("Forward pass must be performed before backward pass.");
    }

    // Simplified Backward for DoRA
    // Exact derivative involves Jacobian of normalization.
    // Approximate: dL/dV approx (m/||V||) * dL/dW' - ...
    // Research mode implementation: We will compute gradients for m and V.

    // 1. Gradient w.r.t W' (the effective weight)
    // dL/dW' = upstream_grad^T * input
    Tensor ug_T = upstream_grad.transpose();
    Tensor grad_W_prime = TensorOps::multiply(ug_T, *m_last_input);

    // 2. Gradient w.r.t m and V
    // W'_{i,j} = m_i * V_{i,j} / ||V_i||
    // dL/dm_i = sum_j (dL/dW'_{i,j} * V_{i,j} / ||V_i||)

    Tensor grad_m_val(1, m_output_dims);
    Tensor grad_V(m_output_dims, m_input_dims);

    for (int i = 0; i < m_output_dims; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < m_input_dims; ++j) {
            sum_sq += m_V->at(i, j) * m_V->at(i, j);
        }
        float norm = std::sqrt(sum_sq) + 1e-9f;
        float inv_norm = 1.0f / norm;
        float norm_sq = norm * norm;

        float grad_m_i = 0.0f;

        // Calculate dot product (grad_W'_i . V_i) needed for both dm and dV
        float dot_grad_V = 0.0f;
        for (int j = 0; j < m_input_dims; ++j) {
            dot_grad_V += grad_W_prime.at(i, j) * m_V->at(i, j);
        }

        grad_m_i = dot_grad_V * inv_norm;
        grad_m_val.at(0, i) = grad_m_i;

        // dL/dV_{i,j} = (m_i / ||V_i||) * dL/dW'_{i,j} - (m_i / ||V_i||^3) * V_{i,j} * (grad_W'_i . V_i)
        float term1_scale = m_m->at(0, i) * inv_norm;
        float term2_scale = m_m->at(0, i) * dot_grad_V / (norm_sq * norm);

        for (int j = 0; j < m_input_dims; ++j) {
             grad_V.at(i, j) = term1_scale * grad_W_prime.at(i, j) - term2_scale * m_V->at(i, j);
        }
    }
    m_grad_m = std::make_unique<Tensor>(grad_m_val);

    // 3. Backprop through V = W0 + B*A
    // V = W0 + BA
    // dL/dB = dL/dV * A^T
    Tensor A_T = m_A->transpose();
    Tensor grad_B_val = TensorOps::multiply(grad_V, A_T);
    m_grad_B = std::make_unique<Tensor>(grad_B_val);

    // dL/dA = B^T * dL/dV
    Tensor B_T = m_B->transpose();
    Tensor grad_A_val = TensorOps::multiply(B_T, grad_V);
    m_grad_A = std::make_unique<Tensor>(grad_A_val);
}

void DoRALayer::update(float learning_rate) {
    if (!m_grad_A || !m_grad_B || !m_grad_m) return;

    // Update A
    for (int i = 0; i < m_A->getRows(); ++i) {
        for (int j = 0; j < m_A->getCols(); ++j) {
            m_A->at(i, j) -= learning_rate * m_grad_A->at(i, j);
        }
    }
    // Update B
    for (int i = 0; i < m_B->getRows(); ++i) {
        for (int j = 0; j < m_B->getCols(); ++j) {
            m_B->at(i, j) -= learning_rate * m_grad_B->at(i, j);
        }
    }
    // Update m
    for (int j = 0; j < m_m->getCols(); ++j) {
        m_m->at(0, j) -= learning_rate * m_grad_m->at(0, j);
    }
}
