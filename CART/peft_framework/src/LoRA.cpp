#include "LoRA.h"
#include <iostream>
#include <random>
#include "TensorOps.h"

LoRALayer::LoRALayer(int input_dims, int output_dims, int rank) : m_rank(rank) {
    m_A = std::make_unique<Tensor>(rank, input_dims);
    m_B = std::make_unique<Tensor>(output_dims, rank);

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
}

LoRALayer::LoRALayer(std::unique_ptr<Tensor> A, std::unique_ptr<Tensor> B) {
    if (!A || !B) {
        throw std::invalid_argument("A and B must not be null.");
    }
    m_A = std::make_unique<Tensor>(*A);
    m_B = std::make_unique<Tensor>(*B);
    m_rank = m_B->getCols();
}

void LoRALayer::setBaseWeights(const Tensor& weights) {
    m_W0 = std::make_unique<Tensor>(weights);
}

Tensor LoRALayer::forward(const Tensor& input) {
    if (!m_W0) {
        throw std::runtime_error("Base weights W0 must be set before forward pass.");
    }
    m_last_input = std::make_unique<Tensor>(input);

    Tensor delta_W = TensorOps::multiply(*m_B, *m_A);
    Tensor W_eff = TensorOps::add(*m_W0, delta_W);
    Tensor W_eff_T = W_eff.transpose();

    return TensorOps::multiply(input, W_eff_T);
}

void LoRALayer::backward(const Tensor& upstream_grad) {
    if (!m_last_input) {
        throw std::runtime_error("Forward pass must be performed before backward pass.");
    }
    Tensor upstream_grad_T = upstream_grad.transpose();
    Tensor B_T = m_B->transpose();
    Tensor temp_grad_A = TensorOps::multiply(B_T, upstream_grad_T);
    m_grad_A = std::make_unique<Tensor>(TensorOps::multiply(temp_grad_A, *m_last_input));

    Tensor A_T = m_A->transpose();
    Tensor temp_grad_B = TensorOps::multiply(*m_last_input, A_T);
    m_grad_B = std::make_unique<Tensor>(TensorOps::multiply(upstream_grad_T, temp_grad_B));
}

void LoRALayer::update(float learning_rate) {
    if (!m_grad_A || !m_grad_B) return;

    // A = A - lr * grad_A
    for (int i = 0; i < m_A->getRows(); ++i) {
        for (int j = 0; j < m_A->getCols(); ++j) {
            m_A->at(i, j) -= learning_rate * m_grad_A->at(i, j);
        }
    }

    // B = B - lr * grad_B
    for (int i = 0; i < m_B->getRows(); ++i) {
        for (int j = 0; j < m_B->getCols(); ++j) {
            m_B->at(i, j) -= learning_rate * m_grad_B->at(i, j);
        }
    }
}
