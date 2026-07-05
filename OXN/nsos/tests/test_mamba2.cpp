#include "../include/mamba2.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nsos;

void test_ssd_recurrence() {
  int L = 5;
  int D = 4;
  int N = 2;
  int H = 2;

  Mamba2SSD layer(D, N, H);

  Tensor u = Tensor::ones({L, D});
  Tensor y = layer.forward(u);

  assert(y.shape[0] == L);
  assert(y.shape[1] == D);

  // Check if output is not zero (basic activity check)
  assert(y.data()[0] != 0.0f);

  Context ctx;
  Tensor grad = Tensor::ones({L, D});
  Tensor grad_input = layer.backward(grad, ctx);
  assert(grad_input.shape[0] == L);
  assert(grad_input.shape[1] == D);

  bool has_grad = false;
  for (auto *param : layer.parameters()) {
    if (!param || param->grad.size == 0)
      continue;

    float grad_sum = 0.0f;
    for (int i = 0; i < param->grad.size; ++i)
      grad_sum += std::abs(param->grad.data()[i]);

    if (grad_sum > 0.0f) {
      has_grad = true;
      break;
    }
  }

  assert(has_grad && "Mamba2SSD parameters should receive backward gradients");

  std::cout << "Mamba2 SSD Forward test passed!" << std::endl;
}

void test_proper_diagonal_batched_streaming_decode() {
  constexpr int D = 8;
  MambaConfig config;
  config.proper_selective_ssm = true;
  config.proper_state_expansion = false;
  config.conv_kernel = 3;
  Mamba2SSD layer(D, 4, 2, config);
  layer.set_streaming_mode(true);

  std::vector<MambaStreamSnapshot> snapshots;
  std::vector<Tensor> expected;
  for (int batch = 0; batch < 2; ++batch) {
    layer.reset();
    for (int step = 0; step < 3; ++step) {
      Tensor token({1, 1, D});
      for (int c = 0; c < D; ++c) {
        token.data()[c] =
            0.03f * static_cast<float>(1 + batch * 17 + step * D + c);
      }
      (void)layer.forward(token);
    }
    MambaStreamSnapshot snapshot = layer.snapshot_streaming_state();
    snapshots.push_back(snapshot);

    layer.restore_streaming_state(snapshot);
    Tensor next({1, 1, D});
    for (int c = 0; c < D; ++c) {
      next.data()[c] = -0.04f * static_cast<float>(1 + batch * D + c);
    }
    expected.push_back(layer.forward(next).cpu());
  }

  layer.restore_streaming_state_batch(snapshots);
  Tensor batch_next({2, 1, D});
  for (int batch = 0; batch < 2; ++batch) {
    for (int c = 0; c < D; ++c) {
      batch_next.data()[batch * D + c] =
          -0.04f * static_cast<float>(1 + batch * D + c);
    }
  }
  Tensor actual = layer.forward(batch_next).cpu();
  assert(actual.shape.dims == std::vector<int>({2, 1, D}));
  for (int batch = 0; batch < 2; ++batch) {
    for (int c = 0; c < D; ++c) {
      const float want = expected[static_cast<size_t>(batch)].data()[c];
      const float got = actual.data()[batch * D + c];
      assert(std::abs(got - want) < 1e-5f);
    }
  }

  std::cout << "Proper diagonal batched streaming test passed!" << std::endl;
}

int main() {
  test_ssd_recurrence();
  test_proper_diagonal_batched_streaming_decode();
  return 0;
}
