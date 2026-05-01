#include "tensor.h"
#include "ttt_layer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nsos;

void test_ttt_kernel_adaptation() {
  std::cout << "[Test] Starting TTT Kernel Adaptation Test..." << std::endl;

  int dim = 64;
  int hidden = 128;
  float lr = 0.01f;

  TTTLayer layer(dim, hidden, lr);
  layer.set_use_hamiltonian(true);
  layer.set_temperature(0.05f);
  layer.set_friction(0.8f);

  // Constant input
  Tensor x({1, 10, dim}, Device::CPU);
  float *x_data = x.data();
  for (int i = 0; i < x.size; ++i)
    x_data[i] = 1.0f;

  std::cout << "Forward Pass 1..." << std::endl;
  Tensor y1 = layer.forward(x);

  // Check for NaNs
  float *y_ptr = y1.data();
  for (int i = 0; i < y1.size; ++i) {
    assert(!std::isnan(y_ptr[i]) && "NaN in TTT output!");
  }

  // We expect the output at t=0 and t=9 to be different due to weight
  // adaptation even with constant input.
  float diff = 0.0f;
  for (int i = 0; i < dim; ++i) {
    diff += std::abs(y_ptr[i] - y_ptr[9 * dim + i]);
  }

  std::cout << "Adaptation diff (t0 vs t9): " << diff << std::endl;
  assert(diff > 0.0f && "TTT Layer should adapt weights over the sequence!");

  std::cout << "[Test] TTT Kernel Adaptation Verified." << std::endl;
}

void test_ttt_kernel_backward() {
  std::cout << "[Test] Starting TTT Kernel Backward Test..." << std::endl;

  int dim = 32;
  int hidden = 64;
  TTTLayer layer(dim, hidden, 0.001f);

  Tensor x({1, 5, dim}, Device::CPU);
  std::fill(x.data(), x.data() + x.size, 0.5f);

  Tensor y = layer.forward(x);

  Tensor grad_y({1, 5, dim}, Device::CPU);
  std::fill(grad_y.data(), grad_y.data() + grad_y.size, 0.1f);

  std::cout << "Backward Pass..." << std::endl;
  Tensor grad_x = layer.backward(grad_y);

  assert(grad_x.shape == x.shape && "Inconsistent grad_x shape!");

  // Verify parameters got gradients
  auto params = layer.parameters();
  bool has_grad = false;
  for (auto *p : params) {
    if (p->grad.size > 0) {
      float sum_g = 0.0f;
      float *g_ptr = p->grad.data();
      for (int i = 0; i < p->grad.size; ++i)
        sum_g += std::abs(g_ptr[i]);

      if (sum_g > 0.0f) {
        std::cout << "Param received gradients: sum=" << sum_g << std::endl;
        has_grad = true;
      }
    }
  }

  assert(has_grad && "Parameters should receive meta-learning gradients!");
  std::cout << "[Test] TTT Kernel Backward Verified." << std::endl;
}

int main() {
  try {
    test_ttt_kernel_adaptation();
    test_ttt_kernel_backward();
    std::cout << "\nALL TTT KERNEL TESTS PASSED." << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test Failed: " << e.what() << std::endl;
    return 1;
  }
}
