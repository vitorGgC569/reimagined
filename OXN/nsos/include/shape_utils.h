#ifndef SHAPE_UTILS_H
#define SHAPE_UTILS_H

#include "tensor.h"
#include <string>

namespace nsos {

void ensure_3d(const Tensor &t, const std::string &name);
void check_matmul(const Tensor &a, const Tensor &b);
void ensure_2d_flat(const Tensor &t);

} // namespace nsos

#endif
