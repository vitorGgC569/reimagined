#include "../include/jamba.h"
#include "../include/numerical_guard.h"
#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nsos;

void test_moe_gradient_flow() {
  std::cout << "[TEST] MoE Gradient Flow..." << std::endl;

  // 1. Initialize Model (2 layers: Layer 0 should be MoE)
  JambaModel model(2, 64, 128, Device::CPU);

  // 2. Forward Pass
  std::vector<int> tokens = {1, 2, 3, 4};
  Context ctx;
  Tensor out = model.forward_ids(tokens, &ctx);

  std::cout << "  [INFO] Forward pass done." << std::endl;
  if (NumericalGuard::has_numerical_issues(out)) {
    std::cout << "  [WARN] Output has NaNs/Infs!" << std::endl;
  }
  std::cout << "  [INFO] NumericalGuard fixes: "
            << (int)NumericalGuard::get_total_fixes() << std::endl;

  // 3. Backward Pass
  Tensor grad = Tensor::ones(out.shape, Device::CPU);
  model.backward(grad, ctx);
  std::cout << "  [INFO] Backward pass done." << std::endl;

  // 4. Verify Gradients exist on Experts
  bool expert_has_grad = false;
  float max_expert_grad = 0.0f;

  auto &moe_layer = model.layers[1]; // Layer 1 is MoE
  for (size_t i = 0; i < moe_layer->experts.size(); ++i) {
    Parameter &w = moe_layer->experts[i]->base_weight;
    if (w.grad.size > 0) {
      float norm = w.grad.norm();
      if (std::isnan(norm)) {
        std::cout << "  [ERR] Expert " << i << " gradient is NaN!" << std::endl;
      } else if (norm > 0) {
        expert_has_grad = true;
        max_expert_grad = std::max(max_expert_grad, norm);
      }
    }
  }

  if (expert_has_grad) {
    std::cout << "  [SUCCESS] KAN Experts have gradients! Max norm: "
              << max_expert_grad << std::endl;
  } else {
    std::cout << "  [FAIL] No gradients found on KAN Experts (or they are NaN)."
              << std::endl;
    exit(1);
  }

  // 5. Verify Router Gradients
  bool router_has_grad = false;
  if (moe_layer->router) {
    float router_grad_norm = moe_layer->router->gate->weight.grad.norm();
    if (router_grad_norm > 0) {
      router_has_grad = true;
      std::cout << "  [SUCCESS] MoE Router has gradients (norm: "
                << router_grad_norm << ")" << std::endl;
    } else if (std::isnan(router_grad_norm)) {
      std::cout << "  [ERR] MoE Router gradient is NaN!" << std::endl;
    }
  }

  if (!router_has_grad) {
    std::cout << "  [FAIL] No gradients on MoE Roouter." << std::endl;
    exit(1);
  }
}

int main() {
  try {
    test_moe_gradient_flow();
    std::cout << "ALL MOE CHECKS PASSED" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
