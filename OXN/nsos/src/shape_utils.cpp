#include "shape_utils.h"
#include "tensor.h"
#include <iostream>
#include <vector>

namespace nsos {

using namespace std;

// Ensure 3D Shape [B, L, D]
void ensure_3d(const Tensor &t, const std::string &name) {
  if (t.shape.size() != 3) {
    std::cerr << "Error: " << name << " must be 3D. Got " << t.shape.size()
              << "D" << std::endl;
    // In strict mode, throw. For now, warn.
  }
}

// Check matrix multiplication compatibility
void check_matmul(const Tensor &a, const Tensor &b) {
  if (a.shape.back() != b.shape[0]) {
    std::cerr << "Error: Matmul mismatch. " << a.shape.back()
              << " != " << b.shape[0] << std::endl;
  }
}

void ensure_2d_flat(const Tensor &t) {
  if (t.shape.size() != 2) {
    // Warn
  }
}

} // namespace nsos
