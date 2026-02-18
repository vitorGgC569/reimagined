#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>

class Tensor {
public:
    Tensor(int rows = 0, int cols = 0);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);

    Tensor transpose() const;

    float& at(int row, int col);
    const float& at(int row, int col) const;

    int getRows() const { return m_rows; }
    int getCols() const { return m_cols; }

private:
    int m_rows;
    int m_cols;
    std::unique_ptr<float[]> m_data;
};

#endif // TENSOR_H
