#include "../include/IA3.h"
#include "../include/TensorOps.h"
#include <iostream>
#include <random>

IA3Layer::IA3Layer(int input_dim, int output_dim)
    : m_input_dim(input_dim), m_output_dim(output_dim) {
    m_L = std::make_unique<Tensor>(1, output_dim);

    // Initialize scaling vector to 1.0 (identity scaling initially)
    for (int j = 0; j < m_output_dim; ++j) {
        m_L->at(0, j) = 1.0f;
    }
}

void IA3Layer::setBaseWeights(const Tensor& weights) {
    m_W0 = std::make_unique<Tensor>(weights);
}

Tensor IA3Layer::forward(const Tensor& input) {
    if (!m_W0) {
        throw std::runtime_error("Base weights W0 must be set before forward pass.");
    }
    m_last_input = std::make_unique<Tensor>(input);

    // Y_pre = X * W0^T
    Tensor W0_T = m_W0->transpose();
    Tensor pre_scale = TensorOps::multiply(input, W0_T);
    m_pre_scale_output = std::make_unique<Tensor>(pre_scale);

    // Y = Y_pre * L (element-wise broadcasting)
    Tensor result(pre_scale.getRows(), pre_scale.getCols());
    for (int i = 0; i < pre_scale.getRows(); ++i) {
        for (int j = 0; j < pre_scale.getCols(); ++j) {
            result.at(i, j) = pre_scale.at(i, j) * m_L->at(0, j);
        }
    }
    return result;
}

Tensor IA3Layer::backward(const Tensor& upstream_grad) {
    if (!m_pre_scale_output) {
        throw std::runtime_error("Forward pass must be performed before backward pass.");
    }

    // dL/dL_vector = sum(upstream_grad * Y_pre, axis=0)
    Tensor grad_L_val(1, m_output_dim);
    for (int j = 0; j < m_output_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < upstream_grad.getRows(); ++i) {
            sum += upstream_grad.at(i, j) * m_pre_scale_output->at(i, j);
        }
        grad_L_val.at(0, j) = sum;
    }
    m_grad_L = std::make_unique<Tensor>(grad_L_val);

    // dL/dX = (upstream_grad * L) @ W0^T
    // Forward: Y = (X @ W0) * L  (broadcasting L over batch)
    //   pre_scale[i,k] = sum_n X[i,n] * W0[k,n]   (W0 is [output, input])
    //   Y[i,k]         = pre_scale[i,k] * L[0,k]
    // Backward:
    //   dY/dX[i,n]     = sum_k W0[k,n] * L[0,k] * upstream_grad[i,k]
    int batch = upstream_grad.getRows();
    Tensor grad_X(batch, m_input_dim);
    if (m_W0) {
        for (int i = 0; i < batch; ++i) {
            for (int n = 0; n < m_input_dim; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < m_output_dim; ++k) {
                    sum += m_W0->at(k, n) * m_L->at(0, k) * upstream_grad.at(i, k);
                }
                grad_X.at(i, n) = sum;
            }
        }
    } else {
        // No base weights -> can't propagate; return zeros, sized for the
        // caller's convenience (matches batch x input_dim).
        for (int i = 0; i < batch; ++i)
            for (int n = 0; n < m_input_dim; ++n)
                grad_X.at(i, n) = 0.0f;
    }
    return grad_X;
}

void IA3Layer::update(float learning_rate) {
    if (!m_grad_L) return;

    // L = L - lr * grad_L
    for (int j = 0; j < m_L->getCols(); ++j) {
        m_L->at(0, j) -= learning_rate * m_grad_L->at(0, j);
    }
}
