#include "../include/jamba.h"
#include "tensor.h"
#include <cassert>
#include <iostream>

using namespace nsos;

void test_jamba_structure() {
  int layers = 16;
  int d_model = 8;

  JambaModel model(layers, d_model);

  // Check layer types
  // Layer 7 (index) -> 8th layer -> Attention
  assert(model.layers[7]->is_attention == true);
  // Layer 0 -> Mamba
  assert(model.layers[0]->is_attention == false);

  // Check if KAN experts are initialized (indirectly via
  // compilation/construction check) We assume if it runs, it's correct
  // structure.

  Tensor x = Tensor::random({4, d_model}); // Sequence length 4
  Tensor y = model.forward(x);

  assert(y.shape == x.shape);
  std::cout << "Jamba Architecture test passed!" << std::endl;
}

int main() {
  test_jamba_structure();
  return 0;
}
