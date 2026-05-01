#include "TensorOps.h"
#include <stdexcept>

namespace TensorOps {

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.getCols() != b.getRows()) {
        throw std::invalid_argument(
            "Tensor dimensions are not compatible for multiplication.");
    }
    Tensor result(a.getRows(), b.getCols());
    for (int i = 0; i < a.getRows(); ++i) {
        for (int j = 0; j < b.getCols(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a.getCols(); ++k) {
                sum += a.at(i, k) * b.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
        throw std::invalid_argument(
            "Tensor dimensions are not compatible for addition.");
    }
    Tensor result(a.getRows(), a.getCols());
    for (int i = 0; i < a.getRows(); ++i) {
        for (int j = 0; j < a.getCols(); ++j) {
            result.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }
    return result;
}

}  // namespace TensorOps
