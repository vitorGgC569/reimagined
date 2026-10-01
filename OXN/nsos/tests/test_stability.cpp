#include "../include/tensor.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nsos;

#include "../include/autograd.h"
#include "../include/jamba.h"

// Macro for simple assertions
#define ASSERT_TRUE(condition)                                                 \
  if (!(condition)) {                                                          \
    std::cerr << "Assertion failed: " << #condition << " at " << __FILE__      \
              << ":" << __LINE__ << std::endl;                                 \
    std::exit(1);                                                              \
  }

void test_stress_alloc() {
  std::cout << "[Stability] Starting Allocation Stress Test (100k Tensors)..."
            << std::endl;
  for (int i = 0; i < 100000; ++i) {
    Tensor t = Tensor::random({10, 10}, Device::CPU);
    Tensor t2 = t.add(t);
    ASSERT_TRUE(t2.data()[0] == t.data()[0] * 2);
  }
  std::cout << "[Stability] Allocation Stress Passed." << std::endl;
}

void test_stress_model() {
  std::cout << "[Stability] Starting Model Stress Test (100 iters)..."
            << std::endl;
  JambaModel model(2, 16, 32); // Small model
  Tensor x = Tensor::random({1, 10, 16});

  for (int i = 0; i < 100; ++i) {
    for (Parameter* parameter : model.parameters()) {
      if (parameter) {
        parameter->zero_grad();
      }
    }
    Context ctx;
    Tensor out = model.forward(x, &ctx);
    ASSERT_TRUE(out.shape[2] == 32); // Expect vocab_size (32)

    Tensor grad = Tensor::ones(out.shape);
    model.backward(grad, ctx);

    auto params = model.parameters();
    ASSERT_TRUE(params.size() > 0);
    // Ensure materialized gradients are not NaN
    for (auto *p : params) {
      if (p->grad.size > 0) {
        ASSERT_TRUE(!std::isnan(p->grad.data()[0]));
      }
    }
  }
  std::cout << "[Stability] Model Stress Passed." << std::endl;
}

void test_slice_torture() {
  std::cout << "[Stability] Starting Slice Torture..." << std::endl;
  Tensor t = Tensor::random({100, 100});
  for (int i = 0; i < 1000; ++i) {
    int start = i % 50;
    int end = start + 10;
    Tensor s = t.slice(0, start, end);
    Tensor s2 = s.slice(1, 0, 10); // Slice of slice
    ASSERT_TRUE(s2.shape[0] == 10);
    ASSERT_TRUE(s2.shape[1] == 10);

    const float original = t.get({start, 0});
    s.data()[0] = 999.0f;
    ASSERT_TRUE(t.get({start, 0}) == original);
  }
  std::cout << "[Stability] Slice Torture Passed." << std::endl;
}

int main() {
  std::cout << "=== NSOS STABILITY SUITE ===" << std::endl;
  test_stress_alloc();
  test_slice_torture();
  test_stress_model();
  std::cout << "=== ALL TESTS PASSED ===" << std::endl;
  return 0;
}
