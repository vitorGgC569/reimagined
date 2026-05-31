#include "../include/bitlinear.h"
#include "../include/bitnet_adapter.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/trainer.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace nsos;

// True when QAT has routed at least one (quantizable, non-sensitive) BitLinear
// off the float reference path onto the real packed ternary kernel.  Robust to
// which layer is first and to sensitive Mamba projections that stay FP32.
static bool any_layer_quantized(JambaModel &model) {
  for (BitLinear *layer : model.collect_bitlinear_layers()) {
    if (layer && !layer->reference_path_enabled()) {
      return true;
    }
  }
  return false;
}

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
  // QAT active on CPU routes the forward through the real packed ternary
  // kernel, so the reference (float) path is OFF during quantized training.
  trainer.global_step_count = trainer.total_training_steps;
  trainer.configure_progressive_qat(scheduler);
  layers = model.collect_bitlinear_layers();
  assert(any_layer_quantized(model));
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
  trainer.global_step_count = trainer.total_training_steps;
  trainer.configure_progressive_qat(scheduler);
  layers = model.collect_bitlinear_layers();
  assert(any_layer_quantized(model));
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
  trainer.global_step_count = trainer.total_training_steps;
  trainer.configure_progressive_qat(scheduler);
  layers = model.collect_bitlinear_layers();
  assert(any_layer_quantized(model));
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
  trainer.global_step_count = trainer.total_training_steps;
  trainer.configure_progressive_qat(scheduler);
  layers = model.collect_bitlinear_layers();
  assert(any_layer_quantized(model));
  std::cout << "Phase6-pattern QAT regression test passed!" << std::endl;
}

// ── Canonical NSOS ternary rule: pack() must agree with quantize_weights() ──
// Guards the consolidation: packed weights == quantize_weights() codes ==
// QAT regularizer target.  One quantization rule everywhere.
void test_canonical_ternary_rule_consistency() {
  const int in = 24, out = 6;
  BitLinear layer(in, out, true);
  float *w = layer.weight.data.data();
  for (int i = 0; i < layer.weight.data.size; ++i) {
    // Spread across the dead/active bands so quantization is non-trivial.
    w[i] = std::sin(0.37f * static_cast<float>(i + 1)) * 1.2f;
  }
  layer.weight.mark_updated();
  layer.repack_weights();

  BitLinearPackedState state = layer.export_packed_state();
  std::vector<int8_t> unpacked;
  BitNetAdapter::unpack_weights_microsoft_style_to_i8(state.packed_weights, out,
                                                      in, unpacked);
  Tensor canonical = layer.quantize_weights(layer.weight.data);
  assert(static_cast<int>(unpacked.size()) >= out * in);
  for (int i = 0; i < out * in; ++i) {
    const int packed_code = static_cast<int>(unpacked[static_cast<size_t>(i)]);
    const int canonical_code = static_cast<int>(canonical.data()[i]);
    assert(packed_code == canonical_code);
  }
  std::cout << "Canonical ternary rule consistency test passed (pack == "
               "quantize_weights)!"
            << std::endl;
}

// ── STE: a quantized training step must move the FP32 latent weights ──
// Proves the real packed kernel runs in the forward AND the straight-through
// estimator produces a gradient that the optimizer applies to the latent
// weights — i.e. we actually train in BitNet, not FP32 + PTQ.
void test_quantized_training_updates_latent_weights() {
  JambaModel model(2, 32, 64);
  Trainer trainer(&model, 0.01f);
  trainer.total_training_steps = 32;

  TrainPhaseScheduler scheduler;
  scheduler.progressive_qat_enabled = true;
  scheduler.semantic_warmup_steps = 0;
  scheduler.qat_start_step = 1;
  scheduler.quantized_precision_bits = 2;
  scheduler.ternary_regularization = 1e-4f;
  trainer.configure_progressive_qat(scheduler);
  trainer.global_step_count = 1;  // quantized phase active

  auto layers = model.collect_bitlinear_layers();
  assert(!layers.empty());
  BitLinear *probe = layers.front();

  Tensor before = probe->weight.data.clone();

  std::vector<int> prompt = {1, 2, 3, 4};
  std::vector<int> answer = {5, 6, 7};
  const float loss = trainer.train_supervised(prompt, answer);

  // Reference path must be OFF: the real packed ternary kernel ran in forward.
  layers = model.collect_bitlinear_layers();
  assert(any_layer_quantized(model));
  // Loss must be finite (no NaN/Inf through the STE path).
  assert(std::isfinite(loss));
  // The STE gradient must have moved the latent weights (training happened).
  const float *b = before.data();
  const float *a = probe->weight.data.data();
  float max_diff = 0.0f;
  for (int i = 0; i < probe->weight.data.size; ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  assert(max_diff > 0.0f);
  std::cout << "Quantized (STE) training updates latent weights test passed!"
            << std::endl;
}

// ── Smooth backward math: finite-difference gradcheck on the reference path ──
// Validates the shared chain-rule math (rmsnorm / magnitude / matmul) that the
// STE path reuses.  Analytic gradient must match the numerical gradient.
void test_reference_path_gradcheck() {
  const int in = 6, out = 4, rows = 3;
  BitLinear layer(in, out, true);
  float *w = layer.weight.data.data();
  for (int i = 0; i < layer.weight.data.size; ++i) {
    w[i] = 0.05f * std::sin(1.7f * static_cast<float>(i + 1));
  }
  float *mag = layer.magnitude.data.data();
  for (int j = 0; j < out; ++j) mag[j] = 1.0f + 0.1f * static_cast<float>(j);
  float *bias = layer.bias.data.data();
  for (int j = 0; j < out; ++j) bias[j] = 0.02f * static_cast<float>(j);

  Tensor x({rows, in});
  for (int i = 0; i < x.size; ++i) {
    x.data()[i] = std::cos(0.07f * static_cast<float>(i + 3));
  }

  auto loss_of = [&]() {
    Tensor y = layer.forward(x);
    double s = 0.0;
    for (int i = 0; i < y.size; ++i) {
      s += 0.5 * static_cast<double>(y.data()[i]) *
           static_cast<double>(y.data()[i]);
    }
    return s;
  };

  // loss = 0.5 * sum(y^2)  =>  dL/dy = y.
  Tensor y = layer.forward(x);
  Tensor dy = y.clone();
  layer.weight.zero_grad();
  layer.magnitude.zero_grad();
  layer.bias.zero_grad();
  layer.backward(dy);

  const float eps = 1e-3f;
  auto check = [&](Parameter &P, const char *name) {
    float max_rel = 0.0f;
    for (int i = 0; i < P.data.size; ++i) {
      const float orig = P.data.data()[i];
      P.data.data()[i] = orig + eps;
      const double lp = loss_of();
      P.data.data()[i] = orig - eps;
      const double lm = loss_of();
      P.data.data()[i] = orig;
      const float numeric = static_cast<float>((lp - lm) / (2.0 * eps));
      const float analytic = P.grad.size > 0 ? P.grad.data()[i] : 0.0f;
      const float denom =
          std::max(1e-3f, std::max(std::abs(numeric), std::abs(analytic)));
      max_rel = std::max(max_rel, std::abs(numeric - analytic) / denom);
    }
    std::printf("[gradcheck] %-10s max rel err = %.3e\n", name, max_rel);
    assert(max_rel < 5e-2f);
  };
  check(layer.weight, "weight");
  check(layer.magnitude, "magnitude");
  check(layer.bias, "bias");
  std::cout << "Reference-path BitLinear gradcheck passed!" << std::endl;
}

int main() {
  test_quantization();
  test_forward();
  test_packed_fused_affine_matches_unfused_release();
  test_canonical_ternary_rule_consistency();
  test_reference_path_gradcheck();
  test_quantized_training_updates_latent_weights();
  test_progressive_qat_scheduler();
  test_progressive_qat_short_run_scaling();
  test_supervised_batch_qat_heterogeneous_regression();
  test_progressive_qat_gpu_training_keeps_reference_path();
  test_supervised_batch_qat_phase6_pattern_regression();
  return 0;
}
