#include "../include/numerical_guard.h"
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

void NumericalGuard::check_nan(const Tensor& t, const std::string& name) {
    // Check logic
}

} // namespace nsos
