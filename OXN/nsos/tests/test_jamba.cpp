#include "../include/jamba.h"
#include "../include/trainer.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace nsos;

namespace {

void assert_tensor_close(const Tensor& lhs, const Tensor& rhs, float tolerance = 1e-3f) {
  Tensor lhs_cpu = lhs.cpu();
  Tensor rhs_cpu = rhs.cpu();
  assert(lhs_cpu.size == rhs_cpu.size);
  for (int i = 0; i < lhs_cpu.size; ++i) {
    const float diff = std::abs(lhs_cpu.data()[i] - rhs_cpu.data()[i]);
    assert(diff < tolerance);
  }
}

Tensor concat_rows(const Tensor& a, const Tensor& b) {
  Tensor a_cpu = a.cpu();
  Tensor b_cpu = b.cpu();
  const int cols = a.shape[1];
  assert(b.shape[1] == cols);
  Tensor out({a.shape[0] + b.shape[0], cols}, Device::CPU);
  std::memcpy(out.data(), a_cpu.data(), static_cast<size_t>(a.size) * sizeof(float));
  std::memcpy(out.data() + a.size, b_cpu.data(), static_cast<size_t>(b.size) * sizeof(float));
  return out;
}

Tensor make_padded_batch(const std::vector<Tensor>& sequences) {
  assert(!sequences.empty());
  const int cols = sequences.front().shape[1];
  int max_rows = 0;
  for (const auto& sequence : sequences) {
    assert(sequence.shape.size() == 2);
    assert(sequence.shape[1] == cols);
    max_rows = std::max(max_rows, sequence.shape[0]);
  }

  Tensor out({static_cast<int>(sequences.size()), max_rows, cols}, Device::CPU);
  std::fill_n(out.data(), out.size, 0.0f);
  for (int batch = 0; batch < static_cast<int>(sequences.size()); ++batch) {
    Tensor seq_cpu = sequences[static_cast<size_t>(batch)].cpu();
    for (int row = 0; row < seq_cpu.shape[0]; ++row) {
      const size_t dst_offset =
          ((static_cast<size_t>(batch) * max_rows) + row) * static_cast<size_t>(cols);
      const size_t src_offset = static_cast<size_t>(row) * static_cast<size_t>(cols);
      std::memcpy(out.data() + dst_offset,
                  seq_cpu.data() + src_offset,
                  static_cast<size_t>(cols) * sizeof(float));
    }
  }
  return out;
}

} // namespace

void test_jamba_structure() {
  int layers = 16;
  int d_model = 8;
  int vocab_size = 32;

  JambaModel model(layers, d_model, vocab_size);

  // Stable schedule: attention only appears on every 8th layer.
  assert(model.layers[7]->uses_attention() == true);
  assert(model.layers[0]->uses_attention() == false);

  // Deep blocks should expose MoE as an optional capability.
  assert(model.layers[5]->uses_moe() == true);

  Tensor x = Tensor::random({4, d_model}); // Sequence length 4
  Tensor y = model.forward(x);

  assert(y.shape[0] == 4);
  assert(y.shape[1] == vocab_size);
  std::cout << "Jamba Architecture test passed!" << std::endl;
}

void test_attention_gqa_streaming() {
  const int d_model = 32;
  Attention attention(d_model, 4, 512, 2);
  assert(attention.num_query_heads() == 4);
  assert(attention.num_kv_heads() == 2);

  Tensor x = Tensor::random({130, d_model});
  Tensor full = attention.forward(x, nullptr);
  Tensor expected_last = full.slice(0, 129, 130);

  attention.reset();
  attention.set_streaming_mode(true);
  Tensor last;
  for (int i = 0; i < 130; ++i) {
    last = attention.forward(x.slice(0, i, i + 1), nullptr);
  }
  attention.set_streaming_mode(false);

  assert_tensor_close(expected_last, last);

  std::cout << "Jamba GQA streaming test passed!" << std::endl;
}

void test_attention_snapshot_branching() {
  const int d_model = 32;
  Attention attention(d_model, 4, 512, 2);
  Tensor prefix = Tensor::random({96, d_model});
  Tensor branch_a = Tensor::random({2, d_model});
  Tensor branch_b = Tensor::random({2, d_model});

  attention.set_streaming_mode(true);
  for (int i = 0; i < prefix.shape[0]; ++i) {
    (void)attention.forward(prefix.slice(0, i, i + 1), nullptr);
  }
  AttentionCacheSnapshot snapshot = attention.snapshot_cache();

  Tensor last_a = attention.forward(branch_a.slice(0, 0, 1), nullptr);
  last_a = attention.forward(branch_a.slice(0, 1, 2), nullptr);

  attention.restore_cache(snapshot);
  Tensor last_b = attention.forward(branch_b.slice(0, 0, 1), nullptr);
  last_b = attention.forward(branch_b.slice(0, 1, 2), nullptr);
  attention.set_streaming_mode(false);

  attention.reset();
  Tensor expected_a =
      attention.forward(concat_rows(prefix, branch_a), nullptr).slice(0, prefix.shape[0] + 1,
                                                                      prefix.shape[0] + 2);
  attention.reset();
  Tensor expected_b =
      attention.forward(concat_rows(prefix, branch_b), nullptr).slice(0, prefix.shape[0] + 1,
                                                                      prefix.shape[0] + 2);

  assert_tensor_close(last_a, expected_a);
  assert_tensor_close(last_b, expected_b);
  std::cout << "Attention cache snapshot branching test passed!" << std::endl;
}

void test_attention_gpu_prefill_parity() {
#ifdef USE_CUDA
  const int d_model = 32;
  Attention attention(d_model, 4, 512, 2);
  Tensor x = Tensor::random({48, d_model});

  Tensor cpu_out = attention.forward(x, nullptr);
  attention.to(Device::GPU);
  Tensor gpu_out = attention.forward(x.to(Device::GPU), nullptr).cpu();

  assert_tensor_close(cpu_out, gpu_out, 2e-3f);
  std::cout << "Attention GPU prefill parity test passed!" << std::endl;
#else
  std::cout << "Attention GPU prefill parity test skipped." << std::endl;
#endif
}

void test_mamba_session_fork_restore() {
  JambaModel model(4, 16, 64);
  assert(model.supports_streaming_inference());
  model.set_streaming_inference(true);

  const std::vector<int> prefix = {1, 2, 3, 4, 5};
  (void)model.forward_ids(prefix, nullptr);
  JambaSessionSnapshot snapshot = model.fork_session();
  assert(snapshot.input_ids == prefix);

  Tensor branch_a = model.forward_ids({6}, nullptr);
  model.restore_session(snapshot);
  Tensor branch_b = model.forward_ids({7}, nullptr);

  model.reset_session();
  model.set_streaming_inference(true);
  (void)model.forward_ids(prefix, nullptr);
  Tensor expected_a = model.forward_ids({6}, nullptr);

  model.reset_session();
  model.set_streaming_inference(true);
  (void)model.forward_ids(prefix, nullptr);
  Tensor expected_b = model.forward_ids({7}, nullptr);

  assert_tensor_close(branch_a, expected_a, 1e-4f);
  assert_tensor_close(branch_b, expected_b, 1e-4f);
  std::cout << "Jamba session fork/restore test passed!" << std::endl;
}

void test_mamba_streaming_prefill_matches_incremental() {
  JambaModel model(4, 16, 64);
  assert(model.supports_streaming_inference());

  const std::vector<int> prefix = {1, 2, 3, 4, 5, 6};
  const std::vector<int> next = {7};

  model.reset_session();
  model.set_streaming_inference(true);
  (void)model.forward_ids(prefix, nullptr);
  Tensor full_prefill_next = model.forward_ids(next, nullptr).cpu();

  model.reset_session();
  model.set_streaming_inference(true);
  for (int token : prefix) {
    (void)model.forward_ids({token}, nullptr);
  }
  Tensor incremental_next = model.forward_ids(next, nullptr).cpu();

  assert_tensor_close(full_prefill_next, incremental_next, 1e-4f);
  std::cout << "Mamba streaming prefill parity test passed!" << std::endl;
}

void test_mamba_batched_streaming_decode() {
  JambaModel model(4, 16, 64);
  assert(model.supports_streaming_inference());
  assert(model.supports_batched_streaming_inference());

  const std::vector<std::vector<int>> prompts = {{1, 2, 3, 4}, {5, 6, 7}};
  const std::vector<int> next_tokens = {8, 9};
  std::vector<JambaSessionSnapshot> snapshots;
  std::vector<Tensor> expected_logits;

  model.set_streaming_inference(true);
  for (size_t i = 0; i < prompts.size(); ++i) {
    model.reset_session();
    model.set_streaming_inference(true);
    (void)model.forward_ids(prompts[i], nullptr);
    JambaSessionSnapshot snapshot = model.fork_session();
    snapshots.push_back(snapshot);

    model.restore_session(snapshot);
    Tensor logits = model.forward_ids({next_tokens[i]}, nullptr).cpu();
    if (logits.shape.size() == 3) {
      logits = logits.reshape({1, logits.shape[2]});
    }
    expected_logits.push_back(std::move(logits));
  }

  model.restore_session_batch(snapshots);
  Tensor batch_logits =
      model.forward_ids_batch({{next_tokens[0]}, {next_tokens[1]}}, nullptr).cpu();
  assert(batch_logits.shape.size() == 3);
  assert(batch_logits.shape[0] == 2);
  assert(batch_logits.shape[1] == 1);

  const int vocab = batch_logits.shape[2];
  for (int row = 0; row < 2; ++row) {
    Tensor row_logits({1, vocab}, Device::CPU);
    std::memcpy(row_logits.data(),
                batch_logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab),
                static_cast<size_t>(vocab) * sizeof(float));
    assert_tensor_close(row_logits, expected_logits[static_cast<size_t>(row)], 1e-4f);
  }

  auto roundtrip = model.fork_session_batch();
  assert(roundtrip.size() == prompts.size());
  std::cout << "Mamba batched streaming decode test passed!" << std::endl;
}

void test_attention_batched_streaming_decode() {
  ModelConfig config;
  config.num_layers = 4;
  config.d_model = 32;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 1;
  config.attention_slot = 0;
  config.use_moe = false;
  config.use_ttt = false;
  config.use_exact_attention_training = false;

  JambaModel model(config, Device::CPU);
  assert(model.supports_streaming_inference());
  assert(model.supports_batched_streaming_inference());

  const std::vector<std::vector<int>> prompts = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const std::vector<int> next_tokens = {9, 10};
  std::vector<JambaSessionSnapshot> snapshots;
  std::vector<Tensor> expected_logits;

  model.set_streaming_inference(true);
  for (size_t i = 0; i < prompts.size(); ++i) {
    model.reset_session();
    model.set_streaming_inference(true);
    for (int token : prompts[i]) {
      (void)model.forward_ids({token}, nullptr);
    }
    JambaSessionSnapshot snapshot = model.fork_session();
    snapshots.push_back(snapshot);

    model.restore_session(snapshot);
    Tensor logits = model.forward_ids({next_tokens[i]}, nullptr).cpu();
    if (logits.shape.size() == 3) {
      logits = logits.reshape({1, logits.shape[2]});
    }
    expected_logits.push_back(std::move(logits));
  }

  model.restore_session_batch(snapshots);
  Tensor batch_logits =
      model.forward_ids_batch({{next_tokens[0]}, {next_tokens[1]}}, nullptr).cpu();
  assert(batch_logits.shape.size() == 3);
  assert(batch_logits.shape[0] == 2);
  assert(batch_logits.shape[1] == 1);

  const int vocab = batch_logits.shape[2];
  for (int row = 0; row < 2; ++row) {
    Tensor row_logits({1, vocab}, Device::CPU);
    std::memcpy(row_logits.data(),
                batch_logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab),
                static_cast<size_t>(vocab) * sizeof(float));
    assert_tensor_close(row_logits, expected_logits[static_cast<size_t>(row)], 1e-4f);
  }

  auto roundtrip = model.fork_session_batch();
  assert(roundtrip.size() == prompts.size());
  std::cout << "Attention batched streaming decode test passed!" << std::endl;
}

void test_ttt_batched_streaming_decode() {
  ModelConfig config;
  config.num_layers = 4;
  config.d_model = 32;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 64;
  config.attention_slot = 63;
  config.use_moe = false;
  config.use_ttt = true;
  config.ttt_period = 1;
  config.ttt_slot = 0;
  config.use_exact_attention_training = false;

  JambaModel model(config, Device::CPU);
  assert(model.supports_streaming_inference());
  assert(model.supports_batched_streaming_inference());

  const std::vector<std::vector<int>> prompts = {{2, 3, 4}, {7, 8, 9}};
  const std::vector<int> next_tokens = {5, 10};
  std::vector<JambaSessionSnapshot> snapshots;
  std::vector<Tensor> expected_logits;

  model.set_streaming_inference(true);
  for (size_t i = 0; i < prompts.size(); ++i) {
    model.reset_session();
    model.set_streaming_inference(true);
    for (int token : prompts[i]) {
      (void)model.forward_ids({token}, nullptr);
    }
    JambaSessionSnapshot snapshot = model.fork_session();
    snapshots.push_back(snapshot);

    model.restore_session(snapshot);
    Tensor logits = model.forward_ids({next_tokens[i]}, nullptr).cpu();
    if (logits.shape.size() == 3) {
      logits = logits.reshape({1, logits.shape[2]});
    }
    expected_logits.push_back(std::move(logits));
  }

  model.restore_session_batch(snapshots);
  Tensor batch_logits =
      model.forward_ids_batch({{next_tokens[0]}, {next_tokens[1]}}, nullptr).cpu();
  assert(batch_logits.shape.size() == 3);
  assert(batch_logits.shape[0] == 2);
  assert(batch_logits.shape[1] == 1);

  const int vocab = batch_logits.shape[2];
  for (int row = 0; row < 2; ++row) {
    Tensor row_logits({1, vocab}, Device::CPU);
    std::memcpy(row_logits.data(),
                batch_logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab),
                static_cast<size_t>(vocab) * sizeof(float));
    assert_tensor_close(row_logits, expected_logits[static_cast<size_t>(row)], 1e-4f);
  }

  auto roundtrip = model.fork_session_batch();
  assert(roundtrip.size() == prompts.size());
  std::cout << "TTT batched streaming decode test passed!" << std::endl;
}

void test_jamba_rank3_batch_forward_backward() {
  JambaModel model(4, 16, 64);
  const std::vector<std::vector<int>> batch_ids = {{1, 2, 3, 4}, {5, 6}};

  Tensor batched = model.forward_ids_batch(batch_ids, nullptr);
  assert(batched.shape.size() == 3);
  assert(batched.shape[0] == 2);
  assert(batched.shape[1] == 4);
  assert(batched.shape[2] == 64);

  for (int batch = 0; batch < static_cast<int>(batch_ids.size()); ++batch) {
    model.reset_session();
    Tensor single = model.forward_ids(batch_ids[static_cast<size_t>(batch)], nullptr);
    Tensor batched_slice = batched.slice(0, batch, batch + 1).reshape({4, 64});
    Tensor valid_slice = batched_slice.slice(0, 0, static_cast<int>(batch_ids[static_cast<size_t>(batch)].size()));
    assert_tensor_close(valid_slice, single, 1e-4f);
    for (int row = static_cast<int>(batch_ids[static_cast<size_t>(batch)].size());
         row < batched_slice.shape[0];
         ++row) {
      for (int col = 0; col < batched_slice.shape[1]; ++col) {
        assert(std::abs(batched_slice.data()[row * batched_slice.shape[1] + col]) < 1e-6f);
      }
    }
  }

  model.reset_session();
  Context ctx;
  batched = model.forward_ids_batch(batch_ids, &ctx);
  Tensor grad = Tensor::zeros(batched.shape.dims, batched.get_device());
  Tensor grad_cpu = grad.cpu();
  for (int batch = 0; batch < static_cast<int>(batch_ids.size()); ++batch) {
    const auto& sequence = batch_ids[static_cast<size_t>(batch)];
    for (int row = 0; row < static_cast<int>(sequence.size()); ++row) {
      const int target = (sequence[static_cast<size_t>(row)] + 1) % 64;
      grad_cpu.data()[((batch * batched.shape[1] + row) * batched.shape[2]) + target] =
          1.0f / static_cast<float>(sequence.size());
    }
  }
  grad.copy_from(grad_cpu);
  model.backward_external(grad, ctx);
  assert(model.embedding->weight.grad.size == model.embedding->weight.data.size);
  assert(model.embedding->weight.grad.norm() > 0.0f);
  std::cout << "Jamba rank-3 batch forward/backward test passed!" << std::endl;
}

void test_attention_rank3_heterogeneous_backward() {
  Attention attention(32, 4, 512, 2);
  Tensor seq_a = Tensor::random({5, 32});
  Tensor seq_b = Tensor::random({3, 32});
  Tensor batch = make_padded_batch({seq_a, seq_b});

  attention.set_training_mode(true);
  attention.set_batch_valid_lengths({5, 3});
  Tensor batch_out = attention.forward(batch, nullptr);
  (void)batch_out;

  Tensor grad_batch({2, 5, 32}, Device::CPU);
  std::fill_n(grad_batch.data(), grad_batch.size, 0.0f);
  for (int row = 0; row < 5; ++row) {
    grad_batch.data()[(row * 32) + (row % 32)] = 0.25f;
  }
  for (int row = 0; row < 3; ++row) {
    grad_batch.data()[((5 + row) * 32) + ((row + 7) % 32)] = 0.5f;
  }

  Tensor dx_batch = attention.backward(grad_batch, nullptr).cpu();
  Tensor dx_batch_a = dx_batch.slice(0, 0, 1).reshape({5, 32});
  Tensor dx_batch_b_full = dx_batch.slice(0, 1, 2).reshape({5, 32});
  Tensor dx_batch_b = dx_batch_b_full.slice(0, 0, 3);

  attention.set_batch_valid_lengths({});
  Tensor grad_a = grad_batch.slice(0, 0, 1).reshape({5, 32});
  (void)attention.forward(seq_a, nullptr);
  Tensor dx_a = attention.backward(grad_a, nullptr).cpu();

  Tensor grad_b = grad_batch.slice(0, 1, 2).reshape({5, 32}).slice(0, 0, 3);
  (void)attention.forward(seq_b, nullptr);
  Tensor dx_b = attention.backward(grad_b, nullptr).cpu();

  assert_tensor_close(dx_batch_a, dx_a, 1e-3f);
  assert_tensor_close(dx_batch_b, dx_b, 1e-3f);
  for (int row = 3; row < 5; ++row) {
    for (int col = 0; col < 32; ++col) {
      assert(std::abs(dx_batch_b_full.data()[row * 32 + col]) < 1e-5f);
    }
  }
  std::cout << "Attention heterogeneous rank-3 backward test passed!" << std::endl;
}

void test_sparse_router_topk() {
  MoERouter router(32, 8, 2);
  Tensor x = Tensor::random({4, 32});
  auto [logits, weights] = router.forward(x);
  (void)logits;

  Tensor weights_cpu = weights.cpu();
  for (int row = 0; row < weights_cpu.shape[0]; ++row) {
    int non_zero = 0;
    float sum = 0.0f;
    for (int expert = 0; expert < weights_cpu.shape[1]; ++expert) {
      const float value = weights_cpu.data()[row * weights_cpu.shape[1] + expert];
      if (value > 1e-6f) {
        ++non_zero;
        sum += value;
      }
    }
    assert(non_zero <= 2);
    assert(std::abs(sum - 1.0f) < 1e-4f);
  }
  std::cout << "Sparse router top-k test passed!" << std::endl;

#ifdef USE_CUDA
  router.to(Device::GPU);
  auto [gpu_logits, gpu_weights] = router.forward(x.to(Device::GPU));
  (void)gpu_logits;
  Tensor gpu_weights_cpu = gpu_weights.cpu();
  for (int row = 0; row < gpu_weights_cpu.shape[0]; ++row) {
    int non_zero = 0;
    float sum = 0.0f;
    for (int expert = 0; expert < gpu_weights_cpu.shape[1]; ++expert) {
      const float value = gpu_weights_cpu.data()[row * gpu_weights_cpu.shape[1] + expert];
      if (value > 1e-6f) {
        ++non_zero;
        sum += value;
      }
    }
    assert(non_zero <= 2);
    assert(std::abs(sum - 1.0f) < 1e-4f);
  }
  std::cout << "Sparse router GPU top-k test passed!" << std::endl;
#endif
}

void test_embedding_heterogeneous_batch_device_safe() {
  Embedding embedding(32, 16);
  const std::vector<std::vector<int>> ids = {{1, 2, 3, 4}, {5, 6}};

  Tensor batch = embedding.forward_batch(ids).cpu();
  assert(batch.shape.size() == 3);
  assert(batch.shape[0] == 2);
  assert(batch.shape[1] == 4);
  assert(batch.shape[2] == 16);

  Tensor single_a = embedding.forward(ids[0]).cpu();
  Tensor single_b = embedding.forward(ids[1]).cpu();
  assert_tensor_close(batch.slice(0, 0, 1).reshape({4, 16}), single_a);
  assert_tensor_close(batch.slice(0, 1, 2).reshape({4, 16}).slice(0, 0, 2), single_b);
  for (int col = 0; col < 16; ++col) {
    assert(std::abs(batch.data()[(1 * 4 + 2) * 16 + col]) < 1e-6f);
    assert(std::abs(batch.data()[(1 * 4 + 3) * 16 + col]) < 1e-6f);
  }

  Tensor grad({2, 4, 16}, Device::CPU);
  std::fill_n(grad.data(), grad.size, 0.0f);
  for (int i = 0; i < grad.size; ++i) {
    grad.data()[i] = 0.1f + static_cast<float>(i % 7) * 0.01f;
  }
  embedding.backward_batch(grad, ids);
  assert(embedding.weight.grad);
  assert(embedding.weight.grad->norm() > 0.0f);

#ifdef USE_CUDA
  Embedding gpu_embedding(32, 16);
  gpu_embedding.to(Device::GPU);
  gpu_embedding.weight.data.copy_from(embedding.weight.data.to(Device::GPU));
  Tensor gpu_batch = gpu_embedding.forward_batch(ids).cpu();
  assert_tensor_close(batch, gpu_batch, 1e-4f);

  Tensor grad_gpu = grad.to(Device::GPU);
  gpu_embedding.backward_batch(grad_gpu, ids);
  assert(gpu_embedding.weight.grad);
  assert(gpu_embedding.weight.grad->cpu().norm() > 0.0f);
#endif

  std::cout << "Embedding heterogeneous device-safe batch test passed!" << std::endl;
}

void test_fullstack_aux_training_smoke() {
  try {
  ModelConfig config;
  config.num_layers = 6;
  config.d_model = 32;
  config.vocab_size = 96;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.attention_period = 3;
  config.attention_slot = 1;
  config.use_moe = true;
  config.num_experts = 4;
  config.num_experts_per_token = 2;
  config.moe_period = 3;
  config.moe_slot = 2;
  config.use_ttt = true;
  config.ttt_period = 3;
  config.ttt_slot = 0;
  config.use_exact_attention_training = true;

  JambaModel model(config, Device::CPU);
  assert(model.layers[0]->uses_ttt());
  assert(model.layers[1]->uses_attention());
  assert(model.layers[2]->uses_moe());

  Trainer trainer(&model, 5e-4f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 2.0f;
  trainer.phase_scheduler.auxiliary_stack_enabled = true;
  trainer.phase_scheduler.auxiliary_session_adapt_enabled = true;
  trainer.phase_scheduler.auxiliary_reasoning_enabled = true;
  trainer.phase_scheduler.auxiliary_memory_enabled = true;
  trainer.phase_scheduler.auxiliary_reasoning_iterations = 2;
  trainer.phase_scheduler.auxiliary_reasoning_simulations = 12;
  trainer.phase_scheduler.auxiliary_memory_blend = 0.4f;

  const std::vector<std::vector<int>> prompt_batch = {
      {1, 2, 3, 4, 5},
      {6, 7, 8, 9},
  };
  const std::vector<std::vector<int>> answer_batch = {
      {10, 11},
      {12, 13, 14},
  };

  const float loss = trainer.train_supervised_batch(prompt_batch, answer_batch);
  assert(std::isfinite(loss));
  assert(loss > 0.0f);
  assert(model.embedding->weight.grad.size == model.embedding->weight.data.size);
  std::cout << "Full-stack auxiliary training smoke test passed!" << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "Full-stack auxiliary training smoke failed: " << ex.what() << std::endl;
    throw std::runtime_error(ex.what());
  }
}

int main() {
  test_jamba_structure();
  test_attention_gqa_streaming();
  test_attention_snapshot_branching();
  test_attention_gpu_prefill_parity();
  test_mamba_session_fork_restore();
  test_mamba_streaming_prefill_matches_incremental();
  test_mamba_batched_streaming_decode();
  test_attention_batched_streaming_decode();
  test_ttt_batched_streaming_decode();
  test_jamba_rank3_batch_forward_backward();
  test_attention_rank3_heterogeneous_backward();
  test_sparse_router_topk();
  test_embedding_heterogeneous_batch_device_safe();
  test_fullstack_aux_training_smoke();
  return 0;
}
