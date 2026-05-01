#include "../include/kan.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace nsos;

int main() {
  BitFastKANLayer layer(4, 3, 6);

  Tensor x({2, 5, 4}, Device::CPU);
  for (int i = 0; i < x.size; ++i) {
    x.data()[i] = static_cast<float>(i % 7) * 0.1f;
  }

  Tensor y = layer.forward(x);
  assert(y.shape[0] == 2);
  assert(y.shape[1] == 5);
  assert(y.shape[2] == 3);

  Tensor grad = Tensor::ones({2, 5, 3});
  Tensor grad_x = layer.backward(grad);
  assert(grad_x.shape == x.shape);

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

  assert(has_grad && "KAN parameters should receive gradients");
  std::cout << "BitFastKANLayer test passed!" << std::endl;
  return 0;
}
