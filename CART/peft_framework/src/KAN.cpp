#include "KAN.h"
#include <random>
#include <stdexcept>

// --- Spline Implementation ---

Spline::Spline(int order) {
    m_control_points.resize(order + 1);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (size_t i = 0; i < m_control_points.size(); ++i) {
        m_control_points[i] = distribution(generator);
    }
}

float Spline::eval(float x) const {
    float result = 0.0f;
    float x_pow = 1.0f;
    for (float coeff : m_control_points) {
        result += coeff * x_pow;
        x_pow *= x;
    }
    return result;
}


// --- KANLayer Implementation ---

KANLayer::KANLayer(int input_dims, int output_dims, int spline_order)
    : m_input_dims(input_dims),
      m_output_dims(output_dims),
      m_spline_order(spline_order) {

    m_splines.resize(m_output_dims);
    for (int i = 0; i < m_output_dims; ++i) {
        for (int j = 0; j < m_input_dims; ++j) {
            m_splines[i].emplace_back(m_spline_order);
        }
    }
}

Tensor KANLayer::forward(const Tensor& input) {
    if (input.getCols() != m_input_dims || input.getRows() != 1) {
        throw std::invalid_argument("Input tensor must have dimensions [1, input_dims].");
    }

    m_last_input = std::make_unique<Tensor>(input); // Store for backward pass

    Tensor output(1, m_output_dims);

    for (int i = 0; i < m_output_dims; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m_input_dims; ++j) {
            sum += m_splines[i][j].eval(input.at(0, j));
        }
        output.at(0, i) = sum;
    }

    return output;
}

void KANLayer::backward(const Tensor& upstream_grad) {
    if (!m_last_input) {
        throw std::runtime_error("Forward pass must be performed before backward pass.");
    }

    // Resize gradient storage
    m_spline_grads.assign(m_output_dims, std::vector<std::vector<float>>(m_input_dims, std::vector<float>(m_spline_order + 1, 0.0f)));

    for (int i = 0; i < m_output_dims; ++i) { // For each output dimension
        float dL_dyi = upstream_grad.at(0, i);
        for (int j = 0; j < m_input_dims; ++j) { // For each input dimension (and corresponding spline)
            float x_j = m_last_input->at(0, j);
            float x_pow_k = 1.0f;
            for (int k = 0; k <= m_spline_order; ++k) { // For each control point
                // dL/d_ck = dL/dyi * dyi/d_fij * d_fij/d_ck
                // d_fij/d_ck = x_j^k
                m_spline_grads[i][j][k] = dL_dyi * x_pow_k;
                x_pow_k *= x_j;
            }
        }
    }
}
