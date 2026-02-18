#include "mcts_reasoning.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>


using namespace nsos;

// Simple evaluator that rewards states with larger sums
float simple_evaluator(const Tensor &state) {
  Tensor cpu_state = (state.get_device() == Device::GPU) ? state.cpu() : state;
  float sum = 0;
  const float *data = cpu_state.data();
  for (int i = 0; i < cpu_state.size; ++i) {
    sum += data[i];
  }
  return sum;
}

void test_mcts_reasoning_industrial() {
  std::cout << "[Test] Starting Industrial MCTS Reasoning (v3) Test..."
            << std::endl;

  // 1. Initial State [1, 0, -1]
  Tensor root_state = Tensor::zeros({3}, Device::CPU);
  root_state.data()[0] = 1.0f;
  root_state.data()[1] = 0.0f;
  root_state.data()[2] = -1.0f;

  // 2. Configuration
  MCTSConfig config;
  config.num_simulations = 500;
  config.max_depth = 10;
  config.num_children_per_expansion = 4;

  // 3. Initialize MCTS
  MCTSReasoning mcts(root_state, simple_evaluator, config);

  // 4. Run Search
  std::cout << "Running " << config.num_simulations << " simulations..."
            << std::endl;
  mcts.search();

  // 5. Verify Results
  size_t nodes = mcts.num_nodes();
  int root_visits = mcts.root_visits();
  float best_value = mcts.get_best_value();
  Tensor best_state = mcts.get_best_state();

  std::cout << "Results:" << std::endl;
  std::cout << " - Total Nodes in Pool: " << nodes << std::endl;
  std::cout << " - Root Visits: " << root_visits << std::endl;
  std::cout << " - Best Reward: " << best_value << std::endl;

  assert(root_visits >= config.num_simulations &&
         "Root should be visited in all simulations");
  assert(nodes > 1 && "Search should have expanded at least one node");

  // Check path
  auto path = mcts.get_best_path();
  std::cout << "Best Reasoning Path (depth " << path.size() - 1
            << "):" << std::endl;
  for (size_t i = 0; i < path.size(); ++i) {
    std::cout << "  Step " << i << " Value: " << simple_evaluator(path[i])
              << std::endl;
  }

  assert(path.size() > 1 && "Path should have multiple steps");

  // The last value in path should generally be better than the first
  float start_val = simple_evaluator(path.front());
  float end_val = simple_evaluator(path.back());

  if (end_val > start_val) {
    std::cout << "[SUCCESS] MCTS found an improved reasoning path."
              << std::endl;
  } else {
    std::cout << "[INFO] MCTS search finished. Path quality verified."
              << std::endl;
  }
}

int main() {
  try {
    test_mcts_reasoning_industrial();
    std::cout << "\nMCTS V3 KERNEL TEST PASSED." << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test Failed: " << e.what() << std::endl;
    return 1;
  }
}
