#include "../include/mamba2.h"
#include "tensor.h"
#include <cassert>
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

  std::cout << "Mamba2 SSD Forward test passed!" << std::endl;
}

int main() {
  test_ssd_recurrence();
  return 0;
}
