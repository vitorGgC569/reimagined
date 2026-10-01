#include "gpu_parity_common.h"

#include "bitlinear.h"
#include "tensor.h"

#include <cmath>
#include <stdexcept>

using namespace nsos;
using namespace nsos::gpu_parity_test;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  return run_parity("lowp_weight_cache", [] {
    Tensor input_cpu({8, 16}, Device::CPU);
    Tensor weight_cpu({24, 16}, Device::CPU);
    for (int index = 0; index < input_cpu.size; ++index) {
      input_cpu.data()[index] =
          0.25f * std::sin(0.07f * static_cast<float>(index + 1));
    }
    for (int index = 0; index < weight_cpu.size; ++index) {
      weight_cpu.data()[index] =
          0.2f * std::cos(0.05f * static_cast<float>(index + 3));
    }
    Tensor input = input_cpu.to(Device::GPU);
    Tensor weight = weight_cpu.to(Device::GPU);

    set_matmul_precision_mode(2);
    reset_lowp_weight_cache_stats();
    try {
      Tensor first = matmul_nt_cached_weight(input, weight, 1);
      Tensor second = matmul_nt_cached_weight(input, weight, 1);
      cuda_sync_or_throw("lowp_weight_cache/repeat");
      assert_close(first, second, 0.0f, "lowp cache exact repeat");

      const auto repeated = lowp_weight_cache_stats();
      if (repeated.budget_bytes == 0) {
        require(repeated.hits == 0 && repeated.misses == 0,
                "disabled lowp cache recorded hit/miss state");
        require(repeated.budget_bypasses >= 2 &&
                    repeated.resident_bytes == 0,
                "zero-budget lowp cache did not bypass conversion storage");
        set_matmul_precision_mode(0);
        return;
      }
      require(repeated.misses == 1 && repeated.hits >= 1,
              "lowp cache did not miss once then hit");
      require(repeated.resident_bytes > 0 &&
                  repeated.resident_bytes <= repeated.budget_bytes,
              "lowp cache resident budget invariant failed");

      // A tied alias has the same shared storage owner and view, so it must hit.
      Tensor tied_alias = weight;
      (void)matmul_nt_cached_weight(input, tied_alias, 1);
      cuda_sync_or_throw("lowp_weight_cache/tied_alias");
      const auto tied = lowp_weight_cache_stats();
      require(tied.hits > repeated.hits,
              "tied lowp weight alias did not reuse the conversion");

      // Mutate the same storage and advance its exact content version. The
      // cache must recast in place, not return stale converted bytes.
      Tensor changed = weight_cpu.clone();
      for (int index = 0; index < changed.size; ++index) {
        changed.data()[index] += 0.03125f;
      }
      weight.copy_from(changed.to(Device::GPU));
      Tensor refreshed = matmul_nt_cached_weight(input, weight, 2);
      cuda_sync_or_throw("lowp_weight_cache/version_refresh");
      const auto refresh_stats = lowp_weight_cache_stats();
      require(refresh_stats.version_refreshes == 1,
              "lowp cache did not refresh exactly once after mutation");
      Tensor first_host = first.cpu();
      Tensor refreshed_host = refreshed.cpu();
      bool changed_output = false;
      for (int index = 0; index < first_host.size; ++index) {
        changed_output = changed_output ||
            first_host.data()[index] != refreshed_host.data()[index];
      }
      require(changed_output, "lowp cache returned stale output after mutation");

      // Precision-policy epoch is part of the key. Switching away and back
      // cannot reuse a conversion produced under the earlier policy epoch.
      const uint64_t misses_before_policy_switch = refresh_stats.misses;
      set_matmul_precision_mode(0);
      set_matmul_precision_mode(2);
      (void)matmul_nt_cached_weight(input, weight, 2);
      cuda_sync_or_throw("lowp_weight_cache/policy_epoch");
      const auto policy_stats = lowp_weight_cache_stats();
      require(policy_stats.misses > misses_before_policy_switch,
              "lowp cache ignored the precision-policy epoch");

      // BitLinear owns a second, higher-level QAT inference cache. A valid
      // in-place Parameter mutation must refresh that ternary materialization
      // by content version even when shape, storage and device are unchanged.
      Tensor qat_weight_cpu({24, 16}, Device::CPU);
      for (int index = 0; index < qat_weight_cpu.size; ++index) {
        switch (index % 4) {
          case 0: qat_weight_cpu.data()[index] = 1.0f; break;
          case 1: qat_weight_cpu.data()[index] = -1.0f; break;
          case 2: qat_weight_cpu.data()[index] = 0.125f; break;
          default: qat_weight_cpu.data()[index] = -0.125f; break;
        }
      }
      BitLinear qat_layer(16, 24, false);
      qat_layer.to(Device::GPU);
      qat_layer.set_exact_linear_mode(true);
      qat_layer.set_reference_path(false);
      qat_layer.set_training_mode(false);
      qat_layer.weight.copy_data_from(qat_weight_cpu.to(Device::GPU));
      const Tensor qat_before = qat_layer.forward(input);
      (void)qat_layer.forward(input);  // exercise the cached repeat explicitly

      Tensor mutated_qat_weight = qat_weight_cpu.mul(-1.0f);
      qat_layer.weight.copy_data_from(mutated_qat_weight.to(Device::GPU));
      const Tensor qat_after = qat_layer.forward(input);

      BitLinear qat_reference(16, 24, false);
      qat_reference.to(Device::GPU);
      qat_reference.set_exact_linear_mode(true);
      qat_reference.set_reference_path(false);
      qat_reference.set_training_mode(false);
      qat_reference.weight.copy_data_from(
          mutated_qat_weight.to(Device::GPU));
      const Tensor qat_expected = qat_reference.forward(input);
      cuda_sync_or_throw("lowp_weight_cache/qat_version_refresh");
      assert_close(qat_after, qat_expected, 0.0f,
                   "QAT inference cache version refresh");
      const Tensor qat_before_host = qat_before.cpu();
      const Tensor qat_after_host = qat_after.cpu();
      bool qat_output_changed = false;
      for (int index = 0; index < qat_before_host.size; ++index) {
        qat_output_changed = qat_output_changed ||
            qat_before_host.data()[index] != qat_after_host.data()[index];
      }
      require(qat_output_changed,
              "QAT inference cache returned stale ternary weights");

      bool rejected_zero_version = false;
      try {
        (void)matmul_nt_cached_weight(input, weight, 0);
      } catch (const std::invalid_argument&) {
        rejected_zero_version = true;
      }
      require(rejected_zero_version,
              "lowp cache accepted an unversioned mutable weight");
    } catch (...) {
      set_matmul_precision_mode(0);
      throw;
    }
    set_matmul_precision_mode(0);
  });
}
