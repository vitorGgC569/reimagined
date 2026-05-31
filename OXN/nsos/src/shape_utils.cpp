#include "shape_utils.h"
#include "tensor.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace nsos {

using namespace std;

// Ensure 3D Shape [B, L, D].  Validators now enforce (throw) instead of
// printing to cerr and continuing — a silent warning is not a validation.
void ensure_3d(const Tensor &t, const std::string &name) {
  if (t.shape.size() != 3) {
    throw std::invalid_argument(name + " must be 3D, got " +
                                std::to_string(t.shape.size()) + "D");
  }
}

// Check matrix multiplication compatibility.  Supports batched B: for
// A[..., M, K] @ B[..., K, N] the contracted dim of B is its second-to-last
// dim (only equal to B.shape[0] when B is exactly 2D — the previous code was
// wrong for batched operands).
void check_matmul(const Tensor &a, const Tensor &b) {
  const int a_rank = static_cast<int>(a.shape.size());
  const int b_rank = static_cast<int>(b.shape.size());
  if (a_rank < 2 || b_rank < 2) {
    throw std::invalid_argument("matmul operands must be at least 2D");
  }
  const int b_contract = b.shape[b_rank - 2];
  if (a.shape.back() != b_contract) {
    throw std::invalid_argument(
        "matmul inner-dimension mismatch: " + std::to_string(a.shape.back()) +
        " != " + std::to_string(b_contract));
  }
}

void ensure_2d_flat(const Tensor &t) {
  if (t.shape.size() != 2) {
    throw std::invalid_argument("expected a 2D tensor, got " +
                                std::to_string(t.shape.size()) + "D");
  }
}

} // namespace nsos
