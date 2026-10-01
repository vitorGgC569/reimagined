// test_thread_safety.cpp
// Thread safety stress tests for OXN/NSOS
// Tests concurrent access to critical components

#include "../include/jamba.h"
#include "../include/memory_system.h"
#include "../include/numerical_guard.h"
#include "../include/tensor.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

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
// Test MemorySystem Thread Safety
// ============================================================================
void test_memory_system_thread_safety() {
  auto start = std::chrono::steady_clock::now();
  bool ok = true;
  std::string msg;

  try {
    MemorySystem mem(64);
    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;

    // Spawn multiple threads that store and retrieve concurrently
    for (int t = 0; t < 8; ++t) {
      threads.emplace_back([&mem, &error_count, t]() {
        try {
          for (int i = 0; i < 50; ++i) {
            // Store random state
            Tensor state = Tensor::random({64}, Device::CPU);
            mem.store_episodic(state);

            // Retrieve with query
            Tensor query = Tensor::random({64}, Device::CPU);
            Tensor result = mem.retrieve(query);

            // Check for NaN in result
            for (int j = 0; j < std::min<int64_t>(10, result.size); ++j) {
              if (std::isnan(result.data()[j])) {
                error_count++;
                return;
              }
            }
          }
        } catch (...) {
          error_count++;
        }
      });
    }

    // Wait for all threads
    for (auto &t : threads) {
      t.join();
    }

    ok = (error_count == 0);
    if (!ok) {
      msg = std::to_string(error_count.load()) + " errors in concurrent access";
    }
  } catch (const std::exception &e) {
    ok = false;
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("MemorySystem Thread Safety", ok,
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count(),
         msg.empty() ? nullptr : msg.c_str());
}

// ============================================================================
// Test NumericalGuard Thread Safety
// ============================================================================
void test_numerical_guard_thread_safety() {
  auto start = std::chrono::steady_clock::now();
  bool ok = true;
  std::string msg;

  try {
    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;

    // Spawn threads that sanitize tensors concurrently
    for (int t = 0; t < 8; ++t) {
      threads.emplace_back([&error_count, t]() {
        try {
          for (int i = 0; i < 100; ++i) {
            // Create tensor with some NaN/Inf values
            Tensor test({32, 32}, Device::CPU);
            float *data = test.data();
            for (int j = 0; j < test.size; ++j) {
              data[j] = (j % 7 == 0) ? NAN : (float)(j % 100) / 10.0f;
            }

            // Sanitize should handle this safely
            Tensor clean = NumericalGuard::sanitize(test, "thread_test");

            // Verify no NaN in output
            for (int j = 0; j < clean.size; ++j) {
              if (std::isnan(clean.data()[j])) {
                error_count++;
                return;
              }
            }
          }
        } catch (...) {
          error_count++;
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    ok = (error_count == 0);
    if (!ok) {
      msg = std::to_string(error_count.load()) +
            " errors in concurrent sanitization";
    }
  } catch (const std::exception &e) {
    ok = false;
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("NumericalGuard Thread Safety", ok,
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count(),
         msg.empty() ? nullptr : msg.c_str());
}

// ============================================================================
// Test Tensor Operations Thread Safety
// ============================================================================
void test_tensor_ops_thread_safety() {
  auto start = std::chrono::steady_clock::now();
  bool ok = true;
  std::string msg;

  try {
    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 8; ++t) {
      threads.emplace_back([&error_count]() {
        try {
          for (int i = 0; i < 50; ++i) {
            Tensor a = Tensor::random({16, 32}, Device::CPU);
            Tensor b = Tensor::random({32, 16}, Device::CPU);

            // Matmul should be thread-safe
            Tensor c = a.matmul(b);

            // Verify shape
            if (c.shape.size() != 2 || c.shape[0] != 16 || c.shape[1] != 16) {
              error_count++;
              return;
            }

            // Check for NaN
            for (int j = 0; j < std::min<int64_t>(10, c.size); ++j) {
              if (std::isnan(c.data()[j])) {
                error_count++;
                return;
              }
            }
          }
        } catch (...) {
          error_count++;
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    ok = (error_count == 0);
    if (!ok) {
      msg = std::to_string(error_count.load()) +
            " errors in concurrent tensor ops";
    }
  } catch (const std::exception &e) {
    ok = false;
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("Tensor Ops Thread Safety", ok,
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count(),
         msg.empty() ? nullptr : msg.c_str());
}

// ============================================================================
// Test JambaModel Concurrent Inference
// ============================================================================
void test_jamba_concurrent_inference() {
  auto start = std::chrono::steady_clock::now();
  bool ok = true;
  std::string msg;

  try {
    // Single model, multiple inference threads
    JambaModel model(2, 64, 128, Device::CPU);
    model.set_training_mode(false);
    std::atomic<int> error_count{0};
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
      threads.emplace_back([&model, &error_count, &success_count, t]() {
        try {
          for (int i = 0; i < 5; ++i) {
            Context ctx;
            std::vector<int> tokens = {t * 10 + i, t * 10 + i + 1,
                                       t * 10 + i + 2, t * 10 + i + 3};

            Tensor out = model.forward_ids(tokens, &ctx);

            // Check output validity
            bool valid = true;
            for (int j = 0; j < std::min<int64_t>(10, out.size); ++j) {
              if (std::isnan(out.data()[j]) || std::isinf(out.data()[j])) {
                valid = false;
                break;
              }
            }

            if (valid)
              success_count++;
            else
              error_count++;
          }
        } catch (...) {
          error_count++;
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    ok = (error_count == 0);
    if (!ok) {
      msg = std::to_string(error_count.load()) + " errors, " +
            std::to_string(success_count.load()) + " successes";
    } else {
      msg = std::to_string(success_count.load()).c_str();
      msg = "(" + std::to_string(success_count.load()) +
            " successful inferences)";
    }
  } catch (const std::exception &e) {
    ok = false;
    msg = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  report("Jamba Concurrent Inference", ok,
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
  std::cout << "NSOS/OXN THREAD SAFETY STRESS TEST SUITE" << std::endl;
  std::cout << "============================================================"
            << std::endl;

  auto total_start = std::chrono::steady_clock::now();

  // Run all tests
  test_memory_system_thread_safety();
  test_numerical_guard_thread_safety();
  test_tensor_ops_thread_safety();
  test_jamba_concurrent_inference();

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
