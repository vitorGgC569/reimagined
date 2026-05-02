#include "../include/bitlinear.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/trainer.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nsos;

void test_quantization() {
  Tensor w({2, 2});
  // Manual data set since we removed direct vector access
  w.at({0, 0}) = 0.8f;
  w.at({0, 1}) = -0.9f;
  w.at({1, 0}) = 0.1f;
  w.at({1, 1}) = -0.2f;

  // Needs instance now due to RNG state
  BitLinear layer(2, 2);
  Tensor q = layer.quantize_weights(w);

  // Mean abs = (0.8+0.9+0.1+0.2)/4 = 0.5
  // 0.8/0.5 = 1.6 -> 1
  // -0.9/0.5 = -1.8 -> -1
  // 0.1/0.5 = 0.2 -> 0
  // -0.2/0.5 = -0.4 -> 0

  // Note: With stochastic rounding, exact values may vary slightly if close to
  // boundary, but these specific values (0.2, -0.4) are far from 0.5 boundary,
  // so they should likely stick to 0. However, if logic is floor(x) + p, 0.2
  // means floor(0.2)=0, p=0.2. 20% chance of being 1. This breaks the
  // deterministic test.

  // We update the test to check validity of range {-1, 0, 1} instead of exact
  // values since stochastic rounding is now active.

  for (int i = 0; i < 4; ++i) {
    float v = q.data()[i];
    assert(v == -1.0f || v == 0.0f || v == 1.0f);
  }

  std::cout << "Quantization test passed (Stochastic Check)!" << std::endl;
}

void test_forward() {
  BitLinear layer(4, 2);
  Tensor x = Tensor::ones({1, 4});
  Tensor y = layer.forward(x);

  assert(y.shape[0] == 1);
  assert(y.shape[1] == 2);

  std::cout << "Forward test passed!" << std::endl;
}

void test_packed_fused_affine_matches_unfused_release() {
  BitLinear layer(8, 4, true);
  float* w = layer.weight.data.data();
  for (int i = 0; i < layer.weight.data.size; ++i) {
    w[i] = (i % 5 == 0) ? 0.9f : ((i % 3 == 0) ? -0.8f : 0.1f);
  }
  float* mag = layer.magnitude.data.data();
  float* bias = layer.bias.data.data();
  for (int i = 0; i < 4; ++i) {
    mag[i] = 0.75f + 0.1f * static_cast<float>(i);
    bias[i] = -0.2f + 0.05f * static_cast<float>(i);
  }
  layer.repack_weights();
  layer.set_reference_path(false);

  Tensor x({3, 8});
  for (int i = 0; i < x.size; ++i) {
    x.data()[i] = static_cast<float>((i % 7) - 3) * 0.125f;
  }

  Tensor fused = layer.forward(x);
  BitLinearPackedState state = layer.export_packed_state();

  BitLinear unfused(8, 4, true);
  unfused.import_packed_state(state, Device::CPU, true);
  unfused.set_reference_path(false);
  unfused.set_use_loqa(true);
  Tensor reference = unfused.forward(x);

  assert(fused.shape == reference.shape);
  for (int i = 0; i < fused.size; ++i) {
    assert(std::abs(fused.data()[i] - reference.data()[i]) < 1e-5f);
  }

  std::cout << "Packed fused affine release test passed!" << std::endl;
}

void test_progressive_qat_scheduler() {
  JambaModel model(1, 16, 32);
  Trainer trainer(&model, 0.001f);

  TrainPhaseScheduler scheduler;
  scheduler.progressive_qat_enabled = true;
  scheduler.semantic_warmup_steps = 0;
  scheduler.qat_start_step = 1;
  scheduler.quantized_precision_bits = 2;
  scheduler.ternary_regularization = 1e-3f;
  trainer.configure_progressive_qat(scheduler);

  auto layers = model.collect_bitlinear_layers();
  assert(!layers.empty());
  assert(layers.front()->reference_path_enabled());

  std::vector<int> tokens = {1, 2, 3, 4, 5};
  (void)trainer.train_step(tokens, {});
  (void)trainer.train_step(tokens, {});

  layers = model.collect_bitlinear_layers();
  assert(trainer.progressive_qat_active());
  assert(layers.front()->reference_path_enabled());
  std::cout << "Progressive QAT scheduler test passed!" << std::endl;
}

void test_progressive_qat_short_run_scaling() {
  JambaModel model(1, 16, 32);
  Trainer trainer(&model, 0.001f);
  trainer.total_training_steps = 40;

  TrainPhaseScheduler scheduler;
  scheduler.progressive_qat_enabled = true;
  scheduler.semantic_warmup_steps = 2048;
  scheduler.qat_start_step = 4096;
  scheduler.quantized_precision_bits = 2;
  scheduler.ternary_regularization = 1e-3f;
  trainer.configure_progressive_qat(scheduler);

  auto layers = model.collect_bitlinear_layers();
  assert(!layers.empty());
  assert(layers.front()->reference_path_enabled());

  std::vector<int> tokens = {1, 2, 3, 4, 5};
  for (int step = 0; step < 28; ++step) {
    (void)trainer.train_step(tokens, {});
  }

  layers = model.collect_bitlinear_layers();
  assert(trainer.progressive_qat_active());
  assert(layers.front()->reference_path_enabled());
  std::cout << "Progressive QAT short-run scaling test passed!" << std::endl;
}

void test_supervised_batch_qat_heterogeneous_regression() {
  JambaModel model(1, 16, 64);
  Trainer trainer(&model, 0.001f);
  trainer.total_training_steps = 16;

  TrainPhaseScheduler scheduler;
  scheduler.progressive_qat_enabled = true;
  scheduler.semantic_warmup_steps = 0;
  scheduler.qat_start_step = 1;
  scheduler.quantized_precision_bits = 2;
  scheduler.ternary_regularization = 1e-3f;
  trainer.configure_progressive_qat(scheduler);

  std::vector<std::vector<int>> prompts = {
      {1, 2, 3},
      {4, 5},
      {6, 7, 8, 9},
  };
  std::vector<std::vector<int>> answers = {
      {10, 11},
      {12, 13, 14},
      {15, 16, 17},
  };

  for (int step = 0; step < 3; ++step) {
    (void)trainer.train_supervised_batch(prompts, answers);
  }

  auto layers = model.collect_bitlinear_layers();
  assert(trainer.progressive_qat_active());
  assert(!layers.empty());
  assert(layers.front()->reference_path_enabled());
  std::cout << "Supervised batch heterogeneous QAT regression test passed!"
            << std::endl;
}

void test_progressive_qat_gpu_training_keeps_reference_path() {
  if (!gpu_custom_kernels_supported()) {
    std::cout << "GPU not available; skipping GPU QAT reference-path test."
              << std::endl;
    return;
  }

  JambaModel model(1, 16, 64, Device::GPU);
  model.to(Device::GPU);
  Trainer trainer(&model, 0.001f);
  trainer.total_training_steps = 8;

  TrainPhaseScheduler scheduler;
  scheduler.progressive_qat_enabled = true;
  scheduler.semantic_warmup_steps = 0;
  scheduler.qat_start_step = 1;
  scheduler.quantized_precision_bits = 2;
  scheduler.ternary_regularization = 1e-3f;
  trainer.configure_progressive_qat(scheduler);

  std::vector<std::vector<int>> prompts = {
      {1, 2, 3},
      {4, 5},
  };
  std::vector<std::vector<int>> answers = {
      {10, 11},
      {12, 13, 14},
  };

  for (int step = 0; step < 2; ++step) {
    (void)trainer.train_supervised_batch(prompts, answers);
  }

  auto layers = model.collect_bitlinear_layers();
  assert(trainer.progressive_qat_active());
  assert(!layers.empty());
  assert(layers.front()->reference_path_enabled());
  std::cout << "GPU training keeps reference path under QAT test passed!"
            << std::endl;
}

void test_supervised_batch_qat_phase6_pattern_regression() {
  JambaModel model(2, 32, 128);
  Trainer trainer(&model, 0.001f);
  trainer.total_training_steps = 8;

  TrainPhaseScheduler scheduler;
  scheduler.progressive_qat_enabled = true;
  scheduler.semantic_warmup_steps = 0;
  scheduler.qat_start_step = 1;
  scheduler.quantized_precision_bits = 2;
  scheduler.ternary_regularization = 1e-4f;
  trainer.configure_progressive_qat(scheduler);
  trainer.global_step_count = 1;
  trainer.repetition_unlikelihood_scale = 0.02f;
  trainer.eos_token_id = 127;

  std::vector<std::vector<int>> prompts = {
      {1, 2, 3, 4, 5},
      {6, 7, 8, 9},
      {10, 11, 12, 13, 14, 15},
      {16, 17, 18, 19},
  };
  std::vector<std::vector<int>> answers = {
      {20, 127},
      {21, 22, 127},
      {23, 127},
      {24, 25, 127},
  };

  for (int step = 0; step < 2; ++step) {
    (void)trainer.train_supervised_batch(prompts, answers);
  }

  auto layers = model.collect_bitlinear_layers();
  assert(!layers.empty());
  assert(layers.front()->reference_path_enabled());
  std::cout << "Phase6-pattern QAT regression test passed!" << std::endl;
}

int main() {
  test_quantization();
  test_forward();
  test_packed_fused_affine_matches_unfused_release();
  test_progressive_qat_scheduler();
  test_progressive_qat_short_run_scaling();
  test_supervised_batch_qat_heterogeneous_regression();
  test_progressive_qat_gpu_training_keeps_reference_path();
  test_supervised_batch_qat_phase6_pattern_regression();
  return 0;
}
