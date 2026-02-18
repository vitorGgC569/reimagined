#include "components.h"
#include "dynamic_chrass.h"
#include "holographic.h"
#include "jamba.h"
#include "nsos_sdk.h"
#include "tensor.h"
#include <iostream>

using namespace nsos;
#include <cassert>
#include <iostream>
#include <string>

// Test V2.0 Features: MCTS and Self-Healing
void test_mcts_reasoning() {
  std::cout << "[Test] Starting MCTS Reasoning Test..." << std::endl;

  // 1. Create Model
  int d_model = 64; // Small model for test
  int vocab = 128;
  JambaModel model(2, d_model, vocab);

  // 2. Create MCTS
  // Random state
  Tensor state = Tensor::zeros({1, d_model}, Device::CPU);
  MCTS mcts(state, &model);

  // 3. Search
  mcts.search(10); // Run 10 iterations

  // 4. Check Tree
  assert(mcts.root->children.size() > 0 &&
         "MCTS Root should have children after expansion");
  int visits = 0;
  for (auto &c : mcts.root->children)
    visits += c->visits;

  // Note: Root visits are updated during backprop from leaf to root.
  // In our impl: 'while(node) { visits++ ... }'.
  // 10 iterations -> Root visits should be 10? Or 10 + initial.
  // MCTS::search loop runs 10 times. Each time backprop increments root.
  assert(mcts.root->visits >= 10 && "Root should be visited");

  std::cout << "[Test] MCTS Reasoning Verified. Best Action: "
            << mcts.get_best_action() << std::endl;
}

void test_self_healing() {
  std::cout << "[Test] Starting Self-Healing Test..." << std::endl;

  nsos::InferenceEngine engine;
  nsos::ModelConfig config;
  config.d_model = 64;
  config.num_layers = 2;
  config.vocab_size = 128;

  bool loaded = engine.load_model("dummy_path", config);
  // load_model creates new model if path missing (warning printed)

  // Case 1: Valid Math
  // Prompt: "1+1=", Response: "2"
  // Statement: "1+1=2" -> Valid
  bool triggered = engine.self_heal("1+1=", "2");
  assert(triggered == false && "Self-Heal triggered on correct math!");

  // Case 2: Invalid Math
  // Prompt: "1+1=", Response: "3"
  // Statement: "1+1=3" -> Invalid
  triggered = engine.self_heal("1+1=", "3");
  assert(triggered == true && "Self-Heal FAILED to trigger on incorrect math!");

  std::cout << "[Test] Self-Healing Verified." << std::endl;
}

int main() {
  try {
    test_mcts_reasoning();
    test_self_healing();
    std::cout << "\nALL V2.0 TESTS PASSED." << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test Failed: " << e.what() << std::endl;
    return 1;
  }
}
