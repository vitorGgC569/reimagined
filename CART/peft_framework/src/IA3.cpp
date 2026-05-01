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

void IA3Layer::backward(const Tensor& upstream_grad) {
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

    // We don't compute grad_X or grad_W0 because W0 is frozen and we usually don't backprop
    // further in these PEFT benchmarks (or it's implied).
    // But strictly speaking dL/dX should be computed if we stack layers.
    // For now, focusing on the parameter gradient.
}

void IA3Layer::update(float learning_rate) {
    if (!m_grad_L) return;

    // L = L - lr * grad_L
    for (int j = 0; j < m_L->getCols(); ++j) {
        m_L->at(0, j) -= learning_rate * m_grad_L->at(0, j);
    }
}
