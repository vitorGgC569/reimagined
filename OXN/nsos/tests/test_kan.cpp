#include "../include/kan.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>

using namespace nsos;

int main() {
  for (int invalid : {0,-1,std::numeric_limits<int>::max()}) {
    bool rejected=false;
    try { BitFastKANLayer malformed(invalid,3,6); }
    catch (const std::invalid_argument&) { rejected=true; }
    if (!rejected) throw std::runtime_error("KAN invalid/overflow geometry accepted");
  }
  BitFastKANLayer layer(4, 3, 6);
  bool wrong_feature_rejected=false;
  try { layer.forward(Tensor::ones({2,4,5},Device::CPU)); }
  catch (const std::invalid_argument&) { wrong_feature_rejected=true; }
  if (!wrong_feature_rejected) throw std::runtime_error("KAN reinterpreted rank3 feature mismatch");

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
