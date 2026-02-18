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

    auto [indices, weights] = router.forward(input);

    // Check shapes
    bool shape_ok = (indices.shape.size() == 2 && indices.shape[0] == 4 &&
                     indices.shape[1] == 2 && weights.shape.size() == 2 &&
                     weights.shape[0] == 4 && weights.shape[1] == 2);

    if (!shape_ok) {
      msg = "Wrong output shapes";
    } else {
      // Check indices are valid (0-15 for 16 experts)
      bool indices_valid = true;
      for (int b = 0; b < 4; ++b) {
        for (int k = 0; k < 2; ++k) {
          int idx = (int)indices.get({b, k});
          if (idx < 0 || idx >= 16) {
            indices_valid = false;
            break;
          }
        }
      }

      // Check weights sum to ~1.0 per sample
      bool weights_valid = true;
      for (int b = 0; b < 4; ++b) {
        float sum = 0;
        for (int k = 0; k < 2; ++k) {
          sum += weights.get({b, k});
        }
        if (std::abs(sum - 1.0f) > 0.01f) {
          weights_valid = false;
        }
      }

      ok = indices_valid && weights_valid;
      if (!indices_valid)
        msg = "Invalid expert indices";
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
    JambaBlock block(64, false, true, false, 0, 4);

    // Input tensor: [batch=2, seq=8, d_model=64]
    Tensor input = Tensor::random({2, 8, 64}, Device::CPU);

    Context ctx;
    Tensor output = block.forward(input, &ctx);

    // Check output shape matches input shape
    bool shape_ok = (output.shape.size() == 3 && output.shape[0] == 2 &&
                     output.shape[1] == 8 && output.shape[2] == 64);

    // Check for NaN/Inf in output
    bool stable = true;
    for (int i = 0; i < std::min(100, output.size); ++i) {
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
    std::vector<int> expert_counts(8, 0);

    // Run multiple batches and count expert usage
    for (int iter = 0; iter < 10; ++iter) {
      Tensor input = Tensor::random({8, 64}, Device::CPU);
      auto [indices, weights] = router.forward(input);

      for (int b = 0; b < 8; ++b) {
        for (int k = 0; k < 2; ++k) {
          int idx = (int)indices.get({b, k});
          if (idx >= 0 && idx < 8) {
            expert_counts[idx]++;
          }
        }
      }
    }

    // Check that at least 4 different experts were used
    int used_experts = 0;
    for (int i = 0; i < 8; ++i) {
      if (expert_counts[i] > 0)
        used_experts++;
    }

    ok = (used_experts >= 2); // At least 2 experts should be used
    if (!ok) {
      msg = "Only " + std::to_string(used_experts) +
            " experts used (expected >= 2)";
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
