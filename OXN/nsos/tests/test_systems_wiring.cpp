// Validates the newly-wired subsystems end-to-end against the real model:
//   - KAN FFN (ModelConfig::use_kan): replaces the dense FFN, gradients flow,
//     params are trained; use_kan=false preserves the dense FFN (no regression).
//   - Self-Healer (generate_with_self_healing): reachable from the live model.
#include "../include/jamba.h"
#include "../include/self_healer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

using namespace nsos;

static bool test_kan_wired() {
  std::cout << "[KAN] ";
  ModelConfig cfg;
  cfg.num_layers = 3; cfg.d_model = 64; cfg.vocab_size = 128;
  cfg.n_heads = 8; cfg.n_kv_heads = 4;
  cfg.mamba2_faithful = false;  // isolate the legacy dense-FFN/KAN switch
  cfg.use_kan = true;
  JambaModel model(cfg, Device::CPU);

  int kan_blocks = 0, dense_blocks = 0;
  for (auto& b : model.layers) { if (b->kan_ffn) ++kan_blocks; if (b->ffn_gate_up) ++dense_blocks; }
  std::cout << "kan_blocks=" << kan_blocks << " dense_blocks=" << dense_blocks << "; ";
  if (kan_blocks == 0) { std::cout << "FAIL: no KAN FFN instantiated\n"; return false; }
  if (dense_blocks != 0) { std::cout << "FAIL: dense FFN still present under use_kan\n"; return false; }

  std::vector<int> tok = {1, 2, 3, 4, 5};
  Context ctx;
  Tensor out = model.forward_ids(tok, &ctx);
  Tensor g = Tensor::ones(out.shape, Device::CPU);
  model.backward(g, ctx);

  std::set<Parameter*> mp;
  for (auto* p : model.parameters()) mp.insert(p);
  bool kan_grad = false, kan_in_params = false, nan = false; float mx = 0.0f;
  for (auto& b : model.layers) {
    if (!b->kan_ffn) continue;
    for (auto* p : b->kan_ffn->parameters()) {
      if (mp.count(p)) kan_in_params = true;
      if (p->grad.size > 0) {
        const float n = p->grad.norm();
        if (std::isnan(n)) nan = true; else if (n > 0) { kan_grad = true; mx = std::max(mx, n); }
      }
    }
  }
  std::cout << "grad_maxnorm=" << mx << " in_params=" << kan_in_params << "; ";
  if (nan) { std::cout << "FAIL: NaN KAN grad\n"; return false; }
  if (!kan_grad) { std::cout << "FAIL: no gradient flowed through KAN\n"; return false; }
  if (!kan_in_params) { std::cout << "FAIL: KAN params not in model.parameters()\n"; return false; }
  std::cout << "OK (forward+backward+grads+params)\n";
  return true;
}

static bool test_kan_off_preserves_dense() {
  ModelConfig cfg;
  cfg.num_layers = 2; cfg.d_model = 64; cfg.vocab_size = 128;
  cfg.n_heads = 8; cfg.n_kv_heads = 4;
  cfg.mamba2_faithful = false;  // legacy mode is the dense-FFN baseline
  cfg.use_kan = false;
  JambaModel model(cfg, Device::CPU);
  int kan = 0, dense = 0;
  for (auto& b : model.layers) { if (b->kan_ffn) ++kan; if (b->ffn_gate_up) ++dense; }
  std::cout << "[KAN-off] kan_blocks=" << kan << " dense_blocks=" << dense << " -> "
            << ((kan == 0 && dense > 0) ? "OK (dense FFN preserved)\n" : "FAIL\n");
  return kan == 0 && dense > 0;
}

static bool test_self_healer_wired() {
  std::cout << "[SelfHealer] ";
  JambaModel model(2, 64, 64, Device::CPU);
  HealingConfig hc; hc.max_retries = 2; hc.consistency_samples = 2; hc.eos_token_id = 63;
  HealingReport rep;
  std::vector<int> prompt = {1, 2, 3};
  std::vector<int> out = generate_with_self_healing(model, prompt, 8, rep, hc);
  std::cout << "generated " << out.size() << " tokens, attempts=" << rep.attempts << "; ";
  if (out.size() < prompt.size()) { std::cout << "FAIL: healer returned no generation\n"; return false; }
  std::cout << "OK (healer reachable from the live model)\n";
  return true;
}

int main() {
  bool ok = true;
  try {
    ok &= test_kan_wired();
    ok &= test_kan_off_preserves_dense();
    ok &= test_self_healer_wired();
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
  std::cout << (ok ? "ALL WIRING VALIDATIONS PASSED\n" : "SOME VALIDATIONS FAILED\n");
  return ok ? 0 : 1;
}
