#include "../include/bitlinear.h"
#include <cassert>
#include <iostream>

using namespace nsos;

void test_quantization() {
  Tensor w({2, 2});
  // Manual data set since we removed direct vector access
  w.at({0, 0}) = 0.8f;
  w.at({0, 1}) = -0.9f;
  w.at({1, 0}) = 0.1f;
  w.at({1, 1}) = -0.2f;

  // Needs instance now due to RNG state
  BitLinear layer(2, 2);
  Tensor q = layer.quantize_weights(w);

  // Mean abs = (0.8+0.9+0.1+0.2)/4 = 0.5
  // 0.8/0.5 = 1.6 -> 1
  // -0.9/0.5 = -1.8 -> -1
  // 0.1/0.5 = 0.2 -> 0
  // -0.2/0.5 = -0.4 -> 0

  // Note: With stochastic rounding, exact values may vary slightly if close to
  // boundary, but these specific values (0.2, -0.4) are far from 0.5 boundary,
  // so they should likely stick to 0. However, if logic is floor(x) + p, 0.2
  // means floor(0.2)=0, p=0.2. 20% chance of being 1. This breaks the
  // deterministic test.

  // We update the test to check validity of range {-1, 0, 1} instead of exact
  // values since stochastic rounding is now active.

  for (int i = 0; i < 4; ++i) {
    float v = q.data()[i];
    assert(v == -1.0f || v == 0.0f || v == 1.0f);
  }

  std::cout << "Quantization test passed (Stochastic Check)!" << std::endl;
}

void test_forward() {
  BitLinear layer(4, 2);
  Tensor x = Tensor::ones({1, 4});
  Tensor y = layer.forward(x);

  assert(y.shape[0] == 1);
  assert(y.shape[1] == 2);

  std::cout << "Forward test passed!" << std::endl;
}

int main() {
  test_quantization();
  test_forward();
  return 0;
}
