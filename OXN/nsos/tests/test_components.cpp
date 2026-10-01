#include "../include/components.h"
#include "../include/fabric.h"
#include "../include/mcts_reasoning.h"
#include "../include/nsos_mpi.h"
#include "../include/sprecher_kan.h"
#include "../include/tensor.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nsos;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

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

  require(std::isfinite(param.data()[0]),
          "Muon produced a non-finite parameter");
  require(param.data()[0] != 1.0f,
          "Muon did not update the parameter");
  std::cout << "Muon Optimizer test passed!" << std::endl;
}

void test_mcts() {
  constexpr int state_size = 8;
  Tensor initial_state = Tensor::zeros({state_size}, Device::CPU);

  MCTSConfig config;
  config.num_simulations = 10;
  config.max_depth = 4;
  config.num_children_per_expansion = 2;
  config.use_noise = false;

  auto evaluator = [](const Tensor& state) {
    float sum = 0.0f;
    for (int index = 0; index < state.size; ++index) {
      sum += state.data()[index];
    }
    return sum;
  };
  MCTSReasoning mcts(initial_state, evaluator, config);

  mcts.search();

  require(mcts.root_visits() >= config.num_simulations,
          "MCTS did not backpropagate all simulations");
  require(mcts.num_nodes() > 1, "MCTS did not expand the root");
  Tensor best_state = mcts.get_best_state();
  require(best_state.size == state_size,
          "MCTS returned a state with the wrong shape");

  std::cout << "MCTS Reasoning test passed!" << std::endl;
}

void test_sprecher() {
  SprecherKAN block(16, 8, 32);
  Tensor output =
      block.forward(Tensor::random({2, 16}, Device::CPU));
  require(output.shape.dims == std::vector<int>({2, 8}),
          "SprecherKAN returned the wrong shape");
  for (int index = 0; index < output.size; ++index) {
    require(std::isfinite(output.data()[index]),
            "SprecherKAN returned NaN or Inf");
  }
  std::cout << "SprecherKAN test passed!" << std::endl;
}

void test_single_process_fabric() {
  Fabric fabric;
  Tensor value = Tensor::ones({10}, Device::CPU);
  fabric.all_reduce(value);
  for (int index = 0; index < value.size; ++index) {
    require(value.data()[index] == 1.0f,
            "single-process Fabric all_reduce changed data");
  }
  fabric.broadcast(value, 0);
#ifndef USE_MPI
  bool send_rejected = false;
  try {
    fabric.send(value, 0);
  } catch (const std::logic_error&) {
    send_rejected = true;
  }
  require(send_rejected,
          "non-MPI Fabric accepted a simulated point-to-point send");
#endif
  std::cout << "Fabric single-process test passed!" << std::endl;
}

void test_multinode_orchestrator_provider() {
  MultiNodeOrchestrator orchestrator;
  require(orchestrator.get_rank() == 0,
          "single-process orchestrator rank is not zero");
  require(orchestrator.get_world_size() == 1,
          "single-process orchestrator world size is not one");
  Tensor gradients = Tensor::ones({4}, Device::CPU);
  orchestrator.sync_gradients(gradients);
  orchestrator.broadcast_params(gradients);
  for (int index = 0; index < gradients.size; ++index) {
    require(gradients.data()[index] == 1.0f,
            "single-process orchestrator changed an identity payload");
  }
  std::cout << "MultiNodeOrchestrator provider test passed!" << std::endl;
}

int main() {
  try {
    test_muon();
    test_mcts();
    test_sprecher();
    test_single_process_fabric();
    test_multinode_orchestrator_provider();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Component test failed: " << error.what() << std::endl;
    return 1;
  }
}
