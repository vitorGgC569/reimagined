// MoE gradient-flow test, rewritten against the CURRENT API.
// (The old version referenced moe_layer->experts[i]->base_weight — a stale
// KAN-expert API that no longer exists; real experts are expert_gate_up /
// expert_down BitLinears, and MoE is opt-in via ModelConfig::use_moe.)
#include "../include/jamba.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nsos;

int main() {
  try {
    std::cout << "[TEST] MoE gradient flow (current expert_gate_up/expert_down API)\n";

    ModelConfig cfg;
    cfg.num_layers = 6;
    cfg.d_model = 64;
    cfg.vocab_size = 128;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 4;
    cfg.use_moe = true;
    cfg.num_experts = 4;
    cfg.num_experts_per_token = 2;
    cfg.moe_period = 2;
    cfg.moe_slot = 1;  // blocks where (idx % period == slot) become MoE
    JambaModel model(cfg, Device::CPU);

    std::vector<int> moe_layers;
    for (size_t i = 0; i < model.layers.size(); ++i)
      if (model.layers[i] && model.layers[i]->uses_moe()) moe_layers.push_back(static_cast<int>(i));
    std::cout << "  MoE layers created: " << moe_layers.size() << "\n";
    if (moe_layers.empty()) { std::cerr << "  [FAIL] use_moe=true produced no MoE blocks\n"; return 1; }

    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    Context ctx;
    Tensor out = model.forward_ids(tokens, &ctx);
    Tensor grad = Tensor::ones(out.shape, Device::CPU);
    model.backward(grad, ctx);

    bool expert_grad = false, router_grad = false, nan = false;
    float max_norm = 0.0f;
    auto check = [&](const std::unique_ptr<BitLinear>& l) {
      if (!l || l->weight.grad.size == 0) return;
      const float n = l->weight.grad.norm();
      if (std::isnan(n)) nan = true;
      else if (n > 0) { expert_grad = true; max_norm = std::max(max_norm, n); }
    };
    for (int li : moe_layers) {
      auto& blk = model.layers[static_cast<size_t>(li)];
      for (auto& e : blk->expert_gate_up) check(e);
      for (auto& e : blk->expert_down) check(e);
      if (blk->router && blk->router->gate && blk->router->gate->weight.grad.size > 0) {
        const float n = blk->router->gate->weight.grad.norm();
        if (std::isnan(n)) nan = true; else if (n > 0) router_grad = true;
      }
    }

    if (nan) { std::cerr << "  [FAIL] NaN gradient in MoE\n"; return 1; }
    if (!expert_grad) { std::cerr << "  [FAIL] no expert gradients flowed through MoE\n"; return 1; }

    // The router gate is trained via the MoE aux-loss path (Trainer::
    // apply_moe_aux_regularization), not the plain main-loss backward, so its
    // gradient is reported as info rather than a hard requirement here.
    std::cout << "  [OK] expert grads flow (max norm " << max_norm << "); router main-loss grad "
              << (router_grad ? "present" : "absent (trained via aux loss in Trainer)") << "\n";
    std::cout << "ALL MOE CHECKS PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
}
