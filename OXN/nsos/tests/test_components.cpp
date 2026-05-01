#include "../include/components.h"
#include "../include/jamba.h"
#include "../include/tensor.h" // Added this line
#include <cassert>
#include <iostream>

using namespace nsos;

void test_muon() {
  Tensor param({2, 2});
  param.at({0, 0}) = 1.0f;
  param.at({0, 1}) = 2.0f;
  param.at({1, 0}) = 3.0f;
  param.at({1, 1}) = 4.0f;

  Tensor grad({2, 2});
  grad.at({0, 0}) = 0.1f;
  grad.at({0, 1}) = 0.1f;
  grad.at({1, 0}) = 0.1f;
  grad.at({1, 1}) = 0.1f;

  MuonOptimizer opt({2, 2}, 0.02f);
  opt.step(param, grad);

  // Check if param changed
  assert(param.data()[0] != 1.0f);
  std::cout << "Muon Optimizer test passed!" << std::endl;
}

void test_mcts() {
  // Need a dummy JambaModel to prevent SegFault
  int d_model = 64;
  JambaModel model(2, d_model, 128); // 2 layers, 64 dim, 128 vocab

  Tensor initial_state = Tensor::zeros({1, d_model});

  MCTS mcts(initial_state, &model);

  mcts.expand(mcts.root.get());
  mcts.search(10);

  int action = mcts.get_best_action();
  assert(action != -1);

  std::cout << "MCTS test passed! Best action: " << action << std::endl;
}

int main() {
  test_muon();
  test_mcts();
  return 0;
}
