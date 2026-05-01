#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "Tensor.h"

namespace TensorOps {

Tensor multiply(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);

} // namespace TensorOps

#endif // TENSOR_OPS_H
