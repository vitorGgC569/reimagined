// test_moe_router.cpp
// Test suite for MoE (Mixture of Experts) router functionality
// Verifies that the MoE system is properly active and routing data

#include "../include/jamba.h"
#include "../include/tensor.h"
#include <chrono>
#include <cmath>
#include <iostream>

using namespace nsos;

int passed = 0;
int failed = 0;

void report(const char *name, bool success, long long ms,
            const char *extra = nullptr) {
  if (success) {
    std::cout << "[TEST] " << name << "... PASS (" << ms << "ms)";
    if (extra)
      std::cout << " " << extra;
    std::cout << std::endl;
    passed++;
  } else {
    std::cout << "[TEST] " << name << "... FAIL";
    if (extra)
      std::cout << ": " << extra;
    std::cout << std::endl;
    failed++;
  }
}

// ============================================================================
// Test MoE Router - Verify router outputs valid indices and weights
// ============================================================================
void test_moe_router_output() {
  auto start = std::chrono::steady_clock::now();
  bool ok = false;
  std::string msg;

  try {
    // Create router with 16 experts, top-2 selection
    MoERouter router(64, 16, 2);

    // Input tensor: [batch=4, d_model=64]
    Tensor input = Tensor::random({4, 64}, Device::CPU);

    auto [logits, weights] = router.forward(input);

    // The public router contract returns dense logits plus a dense expert
    // weight matrix whose non-selected entries are zeroed by top-k.
    bool shape_ok = (logits.shape.size() == 2 && logits.shape[0] == 4 &&
                     logits.shape[1] == 16 && weights.shape.size() == 2 &&
                     weights.shape[0] == 4 && weights.shape[1] == 16);

    if (!shape_ok) {
      msg = "Wrong output shapes";
    } else {
      bool logits_valid = true;
      bool sparsity_valid = true;
      for (int b = 0; b < 4; ++b) {
        int non_zero = 0;
        for (int expert = 0; expert < 16; ++expert) {
          logits_valid =
              logits_valid && std::isfinite(logits.get({b, expert}));
          const float weight = weights.get({b, expert});
          if (weight > 0.0f) {
            ++non_zero;
          }
        }
        sparsity_valid = sparsity_valid && non_zero == 2;
      }

      // Check weights sum to ~1.0 per sample
      bool weights_valid = true;
      for (int b = 0; b < 4; ++b) {
        float sum = 0;
        for (int expert = 0; expert < 16; ++expert) {
          sum += weights.get({b, expert});
        }
        if (std::abs(sum - 1.0f) > 0.01f) {
          weights_valid = false;
        }
      }

      ok = logits_valid && sparsity_valid && weights_valid;
      if (!logits_valid)
        msg = "Router logits contain NaN/Inf";
      else if (!sparsity_valid)
        msg = "Top-k mask did not retain exactly two experts";
      else if (!weights_valid)
        msg = "Weights don't sum to 1.0";
    }
  } catch (const std::exception &e) {
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("MoE Router Output Validation", ok,
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count(),
         msg.empty() ? nullptr : msg.c_str());
}

// ============================================================================
// Test MoE Block Integration - Verify MoE layer in JambaBlock
// ============================================================================
void test_moe_block_integration() {
  auto start = std::chrono::steady_clock::now();
  bool ok = false;
  std::string msg;

  try {
    // Create JambaBlock with MoE enabled (is_moe=true)
    JambaBlock block(
        64, false, true, false, 0, 4,
        4, 2, 4, 2, false);

    // Input tensor: [batch=2, seq=8, d_model=64]
    Tensor input = Tensor::random({2, 8, 64}, Device::CPU);

    Context ctx;
    Tensor output = block.forward(input, &ctx);

    // Check output shape matches input shape
    bool shape_ok = (output.shape.size() == 3 && output.shape[0] == 2 &&
                     output.shape[1] == 8 && output.shape[2] == 64);

    // Check for NaN/Inf in output
    bool stable = true;
    for (int i = 0; i < std::min<int64_t>(100, output.size); ++i) {
      if (std::isnan(output.data()[i]) || std::isinf(output.data()[i])) {
        stable = false;
        break;
      }
    }

    ok = shape_ok && stable;
    if (!shape_ok)
      msg = "Wrong output shape";
    else if (!stable)
      msg = "Output contains NaN/Inf";
  } catch (const std::exception &e) {
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("MoE Block Integration", ok,
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count(),
         msg.empty() ? nullptr : msg.c_str());
}

// ============================================================================
// Test MoE Load Balancing - Verify different experts are selected
// ============================================================================
void test_moe_load_balancing() {
  auto start = std::chrono::steady_clock::now();
  bool ok = false;
  std::string msg;

  try {
    MoERouter router(64, 8, 2);
    router.gate->set_exact_linear_mode(true);
    router.gate->set_training_mode(false);
    Tensor gate_weight = Tensor::zeros({8, 64}, Device::CPU);
    for (int expert = 0; expert < 8; ++expert) {
      gate_weight.data()[expert * 64 + expert] = 5.0f;
    }
    router.gate->weight.copy_data_from(gate_weight);
    std::vector<int> expert_counts(8, 0);

    // Every row has a unique, deliberately dominant expert. This proves
    // diversity without a probabilistic assertion over random initialization.
    Tensor input = Tensor::zeros({8, 64}, Device::CPU);
    for (int row = 0; row < 8; ++row) {
      input.data()[row * 64 + row] = 1.0f;
    }
    auto [logits, weights] = router.forward(input);
    (void)logits;
    bool dominant_experts_selected = true;
    for (int row = 0; row < 8; ++row) {
      for (int expert = 0; expert < 8; ++expert) {
        if (weights.get({row, expert}) > 0.0f) {
          expert_counts[static_cast<size_t>(expert)]++;
        }
      }
      dominant_experts_selected =
          dominant_experts_selected &&
          weights.get({row, row}) > 0.0f;
    }

    int used_experts = 0;
    for (int i = 0; i < 8; ++i) {
      if (expert_counts[i] > 0)
        used_experts++;
    }

    ok = dominant_experts_selected && used_experts == 8;
    if (!ok) {
      msg = "Deterministic routing selected " +
            std::to_string(used_experts) +
            " experts; expected every dominant expert";
    }
  } catch (const std::exception &e) {
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("MoE Load Balancing", ok,
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count(),
         msg.empty() ? nullptr : msg.c_str());
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "NSOS/OXN MOE ROUTER TEST SUITE" << std::endl;
  std::cout << "============================================================"
            << std::endl;

  auto total_start = std::chrono::steady_clock::now();

  // Run all tests
  test_moe_router_output();
  test_moe_block_integration();
  test_moe_load_balancing();

  auto total_end = std::chrono::steady_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_end - total_start)
                      .count();

  std::cout << "\n============================================================"
            << std::endl;
  std::cout << "RESULTS: " << passed << " passed, " << failed << " failed"
            << std::endl;
  std::cout << "Total time: " << total_ms << "ms" << std::endl;
  std::cout << "============================================================"
            << std::endl;

  return failed > 0 ? 1 : 0;
}
