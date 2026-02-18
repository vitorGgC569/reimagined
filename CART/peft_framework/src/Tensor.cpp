#include "Tensor.h"
#include <algorithm>

Tensor::Tensor(int rows, int cols) : m_rows(rows), m_cols(cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Tensor dimensions must be non-negative.");
    }
    if (rows * cols > 0) {
        m_data = std::make_unique<float[]>(rows * cols);
    }
}

Tensor::Tensor(const Tensor& other)
    : m_rows(other.m_rows), m_cols(other.m_cols) {
    if (m_rows * m_cols > 0) {
        m_data = std::make_unique<float[]>(m_rows * m_cols);
        std::copy(other.m_data.get(), other.m_data.get() + (m_rows * m_cols), m_data.get());
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : m_rows(other.m_rows),
      m_cols(other.m_cols),
      m_data(std::move(other.m_data)) {
    other.m_rows = 0;
    other.m_cols = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    m_rows = other.m_rows;
    m_cols = other.m_cols;
    if (m_rows * m_cols > 0) {
        m_data = std::make_unique<float[]>(m_rows * m_cols);
        std::copy(other.m_data.get(), other.m_data.get() + (m_rows * m_cols), m_data.get());
    } else {
        m_data.reset();
    }

    return *this;
}

Tensor Tensor::transpose() const {
    Tensor transposed(m_cols, m_rows);
    for (int i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            transposed.at(j, i) = this->at(i, j);
        }
    }
    return transposed;
}

float& Tensor::at(int row, int col) {
    if (row >= m_rows || col >= m_cols || row < 0 || col < 0) {
        throw std::out_of_range("Tensor access out of bounds.");
    }
    return m_data[row * m_cols + col];
}

const float& Tensor::at(int row, int col) const {
    if (row >= m_rows || col >= m_cols || row < 0 || col < 0) {
        throw std::out_of_range("Tensor access out of bounds.");
    }
    return m_data[row * m_cols + col];
}
