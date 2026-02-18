#ifndef LINALG_H
#define LINALG_H

#include "Tensor.h"
#include <tuple>

namespace LinAlg {

/**
 * @brief [SIMPLIFIED] Computes a truncated SVD using power iteration.
 * @note This is a simplified, non-robust implementation for research purposes.
 *       It may fail to find multiple singular values.
 */
std::tuple<Tensor, Tensor, Tensor> simplifiedSVD(const Tensor& matrix, int rank, int num_iterations = 10);

std::pair<Tensor, Tensor> qrDecomposition(const Tensor& matrix);

} // namespace LinAlg

#endif // LINALG_H
