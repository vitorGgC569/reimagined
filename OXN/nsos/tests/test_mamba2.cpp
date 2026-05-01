#include "../include/mamba2.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nsos;

void test_ssd_recurrence() {
  int L = 5;
  int D = 4;
  int N = 2;
  int H = 2;

  Mamba2SSD layer(D, N, H);

  Tensor u = Tensor::ones({L, D});
  Tensor y = layer.forward(u);

  assert(y.shape[0] == L);
  assert(y.shape[1] == D);

  // Check if output is not zero (basic activity check)
  assert(y.data()[0] != 0.0f);

  Context ctx;
  Tensor grad = Tensor::ones({L, D});
  Tensor grad_input = layer.backward(grad, ctx);
  assert(grad_input.shape[0] == L);
  assert(grad_input.shape[1] == D);

  bool has_grad = false;
  for (auto *param : layer.parameters()) {
    if (!param || param->grad.size == 0)
      continue;

    float grad_sum = 0.0f;
    for (int i = 0; i < param->grad.size; ++i)
      grad_sum += std::abs(param->grad.data()[i]);

    if (grad_sum > 0.0f) {
      has_grad = true;
      break;
    }
  }

  assert(has_grad && "Mamba2SSD parameters should receive backward gradients");

  std::cout << "Mamba2 SSD Forward test passed!" << std::endl;
}

int main() {
  test_ssd_recurrence();
  return 0;
}
