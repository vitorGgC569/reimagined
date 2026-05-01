#include "../include/FullFinetuning.h"
#include "../include/TensorOps.h"
#include <iostream>
#include <random>

FullFinetuningLayer::FullFinetuningLayer(int input_dim, int output_dim)
    : m_input_dim(input_dim), m_output_dim(output_dim) {
    m_weights = std::make_unique<Tensor>(output_dim, input_dim);
    m_bias = std::make_unique<Tensor>(1, output_dim);

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0 / std::sqrt(input_dim));

    for (int i = 0; i < m_weights->getRows(); ++i) {
        for (int j = 0; j < m_weights->getCols(); ++j) {
            m_weights->at(i, j) = distribution(generator);
        }
    }
    // Initialize bias to 0
    for (int i = 0; i < m_bias->getRows(); ++i) {
        for (int j = 0; j < m_bias->getCols(); ++j) {
            m_bias->at(i, j) = 0.0f;
        }
    }
}

Tensor FullFinetuningLayer::forward(const Tensor& input) {
    m_last_input = std::make_unique<Tensor>(input);

    // Y = X * W^T
    Tensor W_T = m_weights->transpose();
    Tensor output = TensorOps::multiply(input, W_T);

    // Add bias. Note: TensorOps::add expects same dimensions.
    // If input is (batch, output_dim) and bias is (1, output_dim),
    // we need to broadcast bias.
    // Since current TensorOps::add throws if dimensions differ, we iterate manually here
    // or assume batch size 1 for now if we strictly follow TensorOps.
    // But let's do it manually to be safe for batch > 1.

    Tensor result = output;
    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            result.at(i, j) += m_bias->at(0, j);
        }
    }
    return result;
}

void FullFinetuningLayer::backward(const Tensor& upstream_grad) {
    // dL/dW = upstream_grad^T * input
    Tensor ug_T = upstream_grad.transpose();
    Tensor grad_w_val = TensorOps::multiply(ug_T, *m_last_input);
    m_grad_W = std::make_unique<Tensor>(grad_w_val);

    // dL/dB = sum(upstream_grad, axis=0)
    Tensor grad_b_val(1, m_output_dim);
    for (int j = 0; j < m_output_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < upstream_grad.getRows(); ++i) {
            sum += upstream_grad.at(i, j);
        }
        grad_b_val.at(0, j) = sum;
    }
    m_grad_B = std::make_unique<Tensor>(grad_b_val);
}
