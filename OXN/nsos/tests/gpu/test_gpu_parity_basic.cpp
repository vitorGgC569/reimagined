// Tensor-add / matmul / RMSNorm CPU↔GPU parity.  These three are the
// smallest end-to-end checks of the CUDA elementwise, tiled-matmul and
// reduction kernels respectively.  Kept together because they share
// the same trivial fixture (a 2×3 / 3×2 pair of constant tensors) and
// each runs in milliseconds; splitting further would obscure the
// "basic kernels still work" signal.

#include "gpu_parity_common.h"
#include "layer_audit.h"
#include "tensor.h"

using nsos::Tensor;
using nsos::Device;
using nsos::LayerAuditCollector;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

int main() {
  return run_parity("basic", [] {
    // Deterministic operands.  i+1 in [1..6] for `a`, i-2 in [-2..3] for `b`.
    Tensor a({2, 3}, Device::CPU);
    Tensor b({3, 2}, Device::CPU);
    for (int i = 0; i < a.size; ++i) a.data()[i] = static_cast<float>(i + 1);
    for (int i = 0; i < b.size; ++i) b.data()[i] = static_cast<float>(i - 2);

    // tensor_add: A + A on both devices.
    {
      const Tensor cpu_add = a.add(a);
      const Tensor gpu_add = a.to(Device::GPU).add(a.to(Device::GPU)).cpu();
      cuda_sync_or_throw("basic/tensor_add");
      assert_close(cpu_add, gpu_add, 1e-4f, "tensor_add");
    }

    // matmul: 2x3 · 3x2 → 2x2 on both devices.
    {
      const Tensor cpu_mm = a.matmul(b);
      const Tensor gpu_mm = a.to(Device::GPU).matmul(b.to(Device::GPU)).cpu();
      cuda_sync_or_throw("basic/matmul");
      assert_close(cpu_mm, gpu_mm, 1e-4f, "matmul");
    }

    // Native BLAS transpose flags replace full transpose materializations.
    {
      Tensor tn_b({2, 4}, Device::CPU);
      Tensor nt_w({4, 3}, Device::CPU);
      for (int i = 0; i < tn_b.size; ++i) {
        tn_b.data()[i] = static_cast<float>(i - 3) * 0.125f;
      }
      for (int i = 0; i < nt_w.size; ++i) {
        nt_w.data()[i] = static_cast<float>(i + 2) * 0.0625f;
      }
      const Tensor tn_reference = a.transpose().matmul(tn_b);
      const Tensor tn_gpu =
          nsos::matmul_tn(a.to(Device::GPU), tn_b.to(Device::GPU));
      const Tensor nt_reference = a.matmul(nt_w.transpose());
      const Tensor nt_gpu =
          nsos::matmul_nt(a.to(Device::GPU), nt_w.to(Device::GPU));
      cuda_sync_or_throw("basic/transpose_flag_gemms");
      assert_close(tn_reference, tn_gpu, 1e-4f, "matmul_tn");
      assert_close(nt_reference, nt_gpu, 1e-4f, "matmul_nt");
    }

    // Fused value*SiLU(gate) emits both backward gradients in one launch.
    {
      Tensor value({2, 3}, Device::CPU);
      Tensor gate({2, 3}, Device::CPU);
      Tensor grad({2, 3}, Device::CPU);
      for (int i = 0; i < value.size; ++i) {
        value.data()[i] = static_cast<float>(i - 2) * 0.3f;
        gate.data()[i] = static_cast<float>(3 - i) * 0.2f;
        grad.data()[i] = static_cast<float>(i + 1) * 0.1f;
      }
      const Tensor cpu_forward = Tensor::silu_gate(value, gate);
      const auto cpu_backward =
          Tensor::silu_gate_backward(grad, value, gate);
      const Tensor value_gpu = value.to(Device::GPU);
      const Tensor gate_gpu = gate.to(Device::GPU);
      const Tensor grad_gpu = grad.to(Device::GPU);
      const Tensor gpu_forward =
          Tensor::silu_gate(value_gpu, gate_gpu);
      const auto gpu_backward =
          Tensor::silu_gate_backward(grad_gpu, value_gpu, gate_gpu);
      cuda_sync_or_throw("basic/silu_gate");
      assert_close(cpu_forward, gpu_forward, 2e-6f, "silu_gate_forward");
      assert_close(cpu_backward.first, gpu_backward.first, 2e-6f,
                   "silu_gate_value_grad");
      assert_close(cpu_backward.second, gpu_backward.second, 2e-6f,
                   "silu_gate_gate_grad");
    }

    // The internal CE scalar must remain device-resident until reporting.
    {
      Tensor logits({3, 5}, Device::CPU);
      for (int i = 0; i < logits.size; ++i) {
        logits.data()[i] =
            0.4f * std::sin(0.3f * static_cast<float>(i + 1));
      }
      const std::vector<int> targets = {0, 3, 1};
      const auto cpu_ce = logits.cross_entropy(targets);
      nsos::reset_gpu_transfer_stats();
      auto gpu_ce = logits.to(Device::GPU).cross_entropy_device(targets);
      cuda_sync_or_throw("basic/cross_entropy_device");
      const auto before_reporting = nsos::gpu_transfer_stats();
      if (before_reporting.d2h_calls != 0 || before_reporting.d2h_bytes != 0) {
        throw std::runtime_error(
            "cross_entropy_device performed an early D2H transfer");
      }
      const Tensor gpu_loss = gpu_ce.first.cpu();
      const Tensor gpu_grad = gpu_ce.second.cpu();
      if (std::abs(gpu_loss.data()[0] - cpu_ce.first) > 2e-6f) {
        throw std::runtime_error("cross_entropy_device loss mismatch");
      }
      assert_close(cpu_ce.second, gpu_grad, 2e-6f,
                   "cross_entropy_device_gradient");
    }

    {
      const auto pool = nsos::pool_stats();
      if (pool.reserved_bytes != pool.allocated_bytes + pool.cached_bytes ||
          pool.peak_allocated_bytes < pool.allocated_bytes ||
          pool.peak_reserved_bytes < pool.reserved_bytes ||
          pool.cached_fragmentation_ratio < 0.0 ||
          pool.cached_fragmentation_ratio > 1.0) {
        throw std::runtime_error("GPU pool telemetry invariants failed");
      }
    }

    // rmsnorm: row-wise normalization on a 2x3 tensor.  RMSNorm allocates
    // shared memory and uses block-wide reduction on the GPU; the CPU
    // path is a straight scalar loop, so any divergence here points at
    // the warp/block reduction kernel.
    {
      const Tensor cpu_norm = a.rmsnorm();
      const Tensor gpu_norm = a.to(Device::GPU).rmsnorm().cpu();
      cuda_sync_or_throw("basic/rmsnorm");
      assert_close(cpu_norm, gpu_norm, 1e-4f, "rmsnorm");
    }

    // mse_loss previously dereferenced GPU pointers in a host loop. Cover
    // both its scalar reduction and device-resident gradient.
    {
      const Tensor target = a.mul(0.5f);
      const auto cpu_mse = a.mse_loss(target);
      const auto gpu_mse =
          a.to(Device::GPU).mse_loss(target.to(Device::GPU));
      cuda_sync_or_throw("basic/mse_loss");
      if (std::abs(cpu_mse.first - gpu_mse.first) > 1e-4f) {
        throw std::runtime_error(
            "mse_loss scalar mismatch: cpu=" +
            std::to_string(cpu_mse.first) +
            " gpu=" + std::to_string(gpu_mse.first));
      }
      assert_close(cpu_mse.second, gpu_mse.second, 1e-4f,
                   "mse_loss_gradient");
    }

    // Audit statistics must reduce natively on the GPU. The old path copied
    // every activation to the host; the corrected path transfers one compact
    // record regardless of tensor size.
    {
      Tensor audit_input({4096}, Device::CPU);
      for (int i = 0; i < audit_input.size; ++i) {
        audit_input.data()[i] =
            static_cast<float>((i % 15) - 7);
      }
      const auto cpu_stats =
          LayerAuditCollector::summarize_tensor(audit_input);
      const Tensor gpu_input = audit_input.to(Device::GPU);
      nsos::reset_gpu_transfer_stats();
      const auto gpu_stats =
          LayerAuditCollector::summarize_tensor(gpu_input);
      const auto transfer = nsos::gpu_transfer_stats();
      if (transfer.d2h_calls != 1 ||
          transfer.d2h_bytes >=
              static_cast<uint64_t>(audit_input.size) *
                  sizeof(float)) {
        throw std::runtime_error(
            "GPU tensor audit copied a full activation to host");
      }
      if (cpu_stats.min != gpu_stats.min ||
          cpu_stats.max != gpu_stats.max ||
          std::abs(cpu_stats.mean - gpu_stats.mean) > 1e-12 ||
          std::abs(cpu_stats.stddev - gpu_stats.stddev) > 1e-12 ||
          std::abs(cpu_stats.l2_norm - gpu_stats.l2_norm) > 1e-12 ||
          cpu_stats.zero_count != gpu_stats.zero_count ||
          cpu_stats.positive_count != gpu_stats.positive_count ||
          cpu_stats.negative_count != gpu_stats.negative_count ||
          !gpu_stats.finite) {
        throw std::runtime_error(
            "GPU tensor audit statistics diverged from CPU");
      }

      const Tensor attention = gpu_input.mul(0.5f);
      const Tensor ffn = gpu_input.mul(-0.25f);
      const Tensor mamba_contribution = gpu_input.mul(0.8f);
      const Tensor attention_contribution =
          gpu_input.mul(0.1f);
      const Tensor ffn_contribution =
          gpu_input.mul(0.05f);
      LayerAuditCollector collector;
      collector.begin_run("gpu_hybrid_compact_audit");
      collector.set_phase("parity");
      collector.set_enabled(true);
      nsos::reset_gpu_transfer_stats();
      collector.record_hybrid_interaction(
          0, "forward", gpu_input, attention, ffn,
          mamba_contribution, attention_contribution,
          ffn_contribution);
      const auto hybrid_transfer =
          nsos::gpu_transfer_stats();
      if (hybrid_transfer.d2h_calls != 1 ||
          hybrid_transfer.d2h_bytes >=
              static_cast<uint64_t>(audit_input.size) *
                  sizeof(float)) {
        throw std::runtime_error(
            "GPU hybrid audit copied full branches to host");
      }
      const auto records =
          collector.hybrid_interaction_records();
      if (records.size() != 1 ||
          records.front().ffn_signal.elements !=
              audit_input.size ||
          std::abs(records.front().signal_cosine - 1.0) >
              1e-10 ||
          std::abs(
              records.front().mamba_ffn_signal_cosine +
              1.0) > 1e-10) {
        throw std::runtime_error(
            "GPU hybrid audit metrics are invalid");
      }
    }
  });
}
