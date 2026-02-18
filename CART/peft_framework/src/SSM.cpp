#include "SSM.h"
#include "TensorOps.h"
#include <random>
#include <memory>
#include <vector>

SSMLayer::SSMLayer(int state_dim, int input_dim)
    : m_state_dim(state_dim), m_input_dim(input_dim) {

    m_A = std::make_unique<Tensor>(m_state_dim, m_state_dim);
    m_B = std::make_unique<Tensor>(m_state_dim, m_input_dim);
    m_C = std::make_unique<Tensor>(m_state_dim, m_state_dim);

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 0.01);

    for (int i = 0; i < m_A->getRows(); ++i)
        for (int j = 0; j < m_A->getCols(); ++j)
            m_A->at(i, j) = distribution(generator);

    for (int i = 0; i < m_B->getRows(); ++i)
        for (int j = 0; j < m_B->getCols(); ++j)
            m_B->at(i, j) = distribution(generator);

    for (int i = 0; i < m_C->getRows(); ++i)
        for (int j = 0; j < m_C->getCols(); ++j)
            m_C->at(i, j) = distribution(generator);
}

Tensor SSMLayer::forward(const Tensor& sequence) {
    int sequence_length = sequence.getRows();
    if (sequence.getCols() != m_input_dim) {
        throw std::invalid_argument("Input sequence has incorrect dimension.");
    }

    m_history_x.clear();
    m_history_h.clear();

    auto h_t = std::make_unique<Tensor>(m_state_dim, 1);
    for(int i = 0; i < m_state_dim; ++i) h_t->at(i, 0) = 0.0f;
    m_history_h.push_back(std::make_unique<Tensor>(*h_t));

    Tensor output_sequence(sequence_length, m_state_dim);

    for (int t = 0; t < sequence_length; ++t) {
        auto x_t = std::make_unique<Tensor>(m_input_dim, 1);
        for(int i = 0; i < m_input_dim; ++i) x_t->at(i, 0) = sequence.at(t, i);

        Tensor term1 = TensorOps::multiply(*m_A, *m_history_h.back());
        Tensor term2 = TensorOps::multiply(*m_B, *x_t);
        h_t = std::make_unique<Tensor>(TensorOps::add(term1, term2));

        Tensor y_t = TensorOps::multiply(*m_C, *h_t);

        for(int i = 0; i < m_state_dim; ++i) output_sequence.at(t, i) = y_t.at(i, 0);

        m_history_x.push_back(std::move(x_t));
        m_history_h.push_back(std::move(h_t));
    }

    return output_sequence;
}

void SSMLayer::backward(const Tensor& upstream_grad_sequence) {
    int seq_len = upstream_grad_sequence.getRows();

    m_grad_A = std::make_unique<Tensor>(m_state_dim, m_state_dim);
    m_grad_B = std::make_unique<Tensor>(m_state_dim, m_input_dim);
    m_grad_C = std::make_unique<Tensor>(m_state_dim, m_state_dim);

    Tensor C_T = m_C->transpose();
    Tensor A_T = m_A->transpose();

    Tensor dh_next(m_state_dim, 1);

    for (int t = seq_len - 1; t >= 0; --t) {
        Tensor dy_t(m_state_dim, 1);
        for(int i=0; i<m_state_dim; ++i) dy_t.at(i,0) = upstream_grad_sequence.at(t, i);

        Tensor h_t_T = m_history_h[t+1]->transpose();
        *m_grad_C = TensorOps::add(*m_grad_C, TensorOps::multiply(dy_t, h_t_T));

        Tensor dh_t = TensorOps::add(TensorOps::multiply(C_T, dy_t), dh_next);

        Tensor x_t_T = m_history_x[t]->transpose();
        *m_grad_B = TensorOps::add(*m_grad_B, TensorOps::multiply(dh_t, x_t_T));

        Tensor h_prev_T = m_history_h[t]->transpose();
        *m_grad_A = TensorOps::add(*m_grad_A, TensorOps::multiply(dh_t, h_prev_T));

        dh_next = TensorOps::multiply(A_T, dh_t);
    }
}
