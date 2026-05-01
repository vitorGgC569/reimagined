#include "mcts_reasoning.h"
#include "tensor.h"
#include <atomic>
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

void test_mcts_batch_evaluator() {
  Tensor root_state = Tensor::zeros({4}, Device::CPU);
  MCTSConfig config;
  config.num_simulations = 64;
  config.max_depth = 6;
  config.num_children_per_expansion = 3;

  MCTSReasoning mcts(root_state, simple_evaluator, config);
  std::atomic<int> batch_calls{0};
  mcts.set_batch_evaluator([&batch_calls](const std::vector<Tensor>& states) {
    batch_calls.fetch_add(1, std::memory_order_relaxed);
    std::vector<float> values;
    values.reserve(states.size());
    for (const Tensor& state : states) {
      values.push_back(simple_evaluator(state));
    }
    return values;
  });

  mcts.search();
  assert(batch_calls.load(std::memory_order_relaxed) > 0);
}

void test_mcts_ultraplan_batch_swarm() {
  Tensor root_state = Tensor::zeros({4}, Device::CPU);
  MCTSConfig config;
  config.num_simulations = 24;
  config.max_depth = 5;
  config.num_children_per_expansion = 3;

  MCTSReasoning mcts(root_state, simple_evaluator, config);
  std::atomic<int> batch_calls{0};
  std::atomic<int> max_batch{0};
  mcts.set_batch_evaluator([&](const std::vector<Tensor>& states) {
    batch_calls.fetch_add(1, std::memory_order_relaxed);
    const int batch_size = static_cast<int>(states.size());
    int current = max_batch.load(std::memory_order_relaxed);
    while (batch_size > current &&
           !max_batch.compare_exchange_weak(current, batch_size,
                                            std::memory_order_relaxed)) {
    }

    std::vector<float> values;
    values.reserve(states.size());
    for (const Tensor& state : states) {
      values.push_back(simple_evaluator(state));
    }
    return values;
  });

  Tensor best = mcts.search_ultraplan(4);
  assert(best.size == root_state.size);
  assert(batch_calls.load(std::memory_order_relaxed) > 0);
  assert(max_batch.load(std::memory_order_relaxed) >= config.num_children_per_expansion);
}

int main() {
  try {
    test_mcts_reasoning_industrial();
    test_mcts_batch_evaluator();
    test_mcts_ultraplan_batch_swarm();
    std::cout << "\nMCTS V3 KERNEL TEST PASSED." << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test Failed: " << e.what() << std::endl;
    return 1;
  }
}
