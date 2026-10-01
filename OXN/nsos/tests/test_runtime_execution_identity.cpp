#include "runtime_execution_identity.h"
#include "nsos/sha256.h"
#include "optimizer_runtime_policy.h"
#include "training_runtime_policy.h"
#include "tensor.h"
#include "jamba.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nsos;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void set_environment(const char* name, const char* value) {
#ifdef _WIN32
  if (_putenv_s(name, value) != 0) {
    throw std::runtime_error("cannot set runtime identity test environment");
  }
#else
  if (setenv(name, value, 1) != 0) {
    throw std::runtime_error("cannot set runtime identity test environment");
  }
#endif
}

std::vector<RuntimeExecutionIdentity::Field> identity_fields() {
  return {
      {"identity.schema_version", "3"},
      {"checkpoint.gradient_policy", "retain_full_history_v1"},
      {"determinism.reductions", "true"},
      {"device.execution", "cpu"},
      {"gemm.accumulator", "fp32"},
      {"gemm.algorithm", "cpu_reference"},
      {"gemm.lowp_weight_cache_budget_bytes", "0"},
      {"gemm.lowp_weight_cache_key", "test_key"},
      {"gemm.lowp_weight_cache_schema", "1"},
      {"gemm.provider", "cpu"},
      {"gpu.backend", "cpu"},
      {"gpu.strict_execution", "false"},
      {"gpu.vendor", "none"},
      {"mamba.history_layout", "row_channel_state_v1"},
      {"mamba.scan_geometry", "head_channel_v1"},
      {"optimizer.update_policy",
       "deterministic_multi_tensor_chunked_update_v3_chunk_8192_"
       "separate_finite_status_v1"},
      {"optimizer.finite_check_policy",
       "multi_tensor_chunked_boolean_v1_chunk_8192"},
      {"precision.matmul", "fp32"},
  };
}

}  // namespace

int main() {
  require(
      integrity::sha256_hex(nullptr, 0) ==
          "e3b0c44298fc1c149afbf4c8996fb924"
          "27ae41e4649b934ca495991b7852b855",
      "SHA-256 empty-message known vector mismatch");
  const std::string abc = "abc";
  require(
      integrity::sha256_hex(abc.data(), abc.size()) ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad",
      "SHA-256 abc known vector mismatch");

  auto fields = identity_fields();
  RuntimeExecutionIdentity canonical =
      validated_runtime_execution_identity(fields);
  std::reverse(fields.begin(), fields.end());
  RuntimeExecutionIdentity reordered =
      validated_runtime_execution_identity(fields);
  require(runtime_execution_identity_equal(canonical, reordered),
          "identity canonicalization depends on input order");
  require(canonical.digest() == reordered.digest(),
          "identity digest depends on input order");

  auto changed_fields = identity_fields();
  for (auto& field : changed_fields) {
    if (field.first == "precision.matmul") field.second = "fp16";
  }
  RuntimeExecutionIdentity changed =
      validated_runtime_execution_identity(changed_fields);
  require(!runtime_execution_identity_equal(canonical, changed),
          "identity accepted a precision mismatch");
  const std::string diagnostic =
      runtime_execution_identity_mismatch(canonical, changed);
  require(diagnostic.find("precision.matmul") != std::string::npos &&
              diagnostic.find("fp32") != std::string::npos &&
              diagnostic.find("fp16") != std::string::npos,
          "identity mismatch diagnostic is not field-actionable");

  bool duplicate_rejected = false;
  auto duplicate = identity_fields();
  duplicate.emplace_back("precision.matmul", "fp32");
  try {
    (void)validated_runtime_execution_identity(std::move(duplicate));
  } catch (const std::runtime_error&) {
    duplicate_rejected = true;
  }
  require(duplicate_rejected, "identity accepted a duplicate field");

  bool mutation_rejected = false;
  {
    RuntimeExecutionPolicyLease outer;
    RuntimeExecutionPolicyLease nested;
    try {
      set_matmul_precision_mode(matmul_precision_mode());
    } catch (const std::runtime_error&) {
      mutation_rejected = true;
    }
  }
  require(mutation_rejected,
          "runtime policy mutated inside an active operation lease");
  set_matmul_precision_mode(matmul_precision_mode());

  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNKED", "0");
  require(!optimizer_policy::deterministic_adamw_chunked_enabled(),
          "deterministic AdamW rollback policy was ignored");
  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNKED", "1");
  require(optimizer_policy::deterministic_adamw_chunked_enabled(),
          "deterministic AdamW production policy was ignored");
  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNKED", "false");
  bool invalid_adamw_boolean_rejected = false;
  try {
    (void)optimizer_policy::deterministic_adamw_chunked_enabled();
  } catch (const std::invalid_argument&) {
    invalid_adamw_boolean_rejected = true;
  }
  require(invalid_adamw_boolean_rejected,
          "deterministic AdamW accepted an ambiguous boolean policy");
  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNKED", "1");
  set_environment("NSOS_DETERMINISTIC_FINITE_GATE_DEFERRED", "0");
  require(!optimizer_policy::deterministic_finite_gate_deferred_enabled(),
          "deterministic finite-gate rollback policy was ignored");
  set_environment("NSOS_DETERMINISTIC_FINITE_GATE_DEFERRED", "1");
  require(optimizer_policy::deterministic_finite_gate_deferred_enabled(),
          "deterministic finite-gate production policy was ignored");
  set_environment("NSOS_OPTIMIZER_FINITE_CHUNKED", "0");
  require(!optimizer_policy::optimizer_finite_chunked_enabled(),
          "optimizer finite-scan rollback policy was ignored");
  set_environment("NSOS_OPTIMIZER_FINITE_CHUNKED", "1");
  require(optimizer_policy::optimizer_finite_chunked_enabled(),
          "optimizer finite-scan production policy was ignored");
  set_environment("NSOS_OPTIMIZER_FINITE_CHUNKED", "01");
  bool invalid_finite_boolean_rejected = false;
  try {
    (void)optimizer_policy::optimizer_finite_chunked_enabled();
  } catch (const std::invalid_argument&) {
    invalid_finite_boolean_rejected = true;
  }
  require(invalid_finite_boolean_rejected,
          "optimizer finite scan accepted an ambiguous boolean policy");
  set_environment("NSOS_OPTIMIZER_FINITE_CHUNKED", "1");
  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNK_ELEMENTS", "31");
  bool invalid_chunk_rejected = false;
  try {
    (void)optimizer_policy::deterministic_adamw_chunk_elements();
  } catch (const std::invalid_argument&) {
    invalid_chunk_rejected = true;
  }
  require(invalid_chunk_rejected,
          "deterministic AdamW accepted an invalid chunk geometry");
  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNK_ELEMENTS", "32");
  require(optimizer_policy::deterministic_adamw_chunk_elements() == 32u,
          "deterministic AdamW did not canonicalize its chunk geometry");
  set_environment("NSOS_DETERMINISTIC_ADAMW_CHUNK_ELEMENTS", "8192");
  set_environment("NSOS_OPTIMIZER_FINITE_CHUNK_ELEMENTS", "31");
  bool invalid_finite_chunk_rejected = false;
  try {
    (void)optimizer_policy::optimizer_finite_chunk_elements();
  } catch (const std::invalid_argument&) {
    invalid_finite_chunk_rejected = true;
  }
  require(invalid_finite_chunk_rejected,
          "optimizer finite scan accepted an invalid chunk geometry");
  set_environment("NSOS_OPTIMIZER_FINITE_CHUNK_ELEMENTS", "8192");
  const std::pair<const char*, bool(*)()> redesign_policies[] = {
      {"NSOS_MAMBA_BOUNDARY_HISTORY", training_policy::boundary_history},
      {"NSOS_ATTN_TILED_TRAINING", training_policy::tiled_attention},
      {"NSOS_MOE_ORDERED_DEVICE", training_policy::ordered_moe},
      {"NSOS_MOE_GROUPED_TRAINING", training_policy::grouped_moe_training},
      {"NSOS_MOE_WMMA_TRAINING", training_policy::moe_wmma_training},
      {"NSOS_TTT_DEVICE_RECURRENCE", training_policy::device_ttt},
      {"NSOS_TTT_FULL_BPTT", training_policy::full_ttt_bptt},
      {"NSOS_KAN_RECOMPUTE_TRAINING", training_policy::kan_recompute_training},
      {"NSOS_KAN_WMMA_TRAINING", training_policy::kan_wmma_training},
      {"NSOS_DEVICE_GRAD_CLIP", optimizer_policy::device_gradient_clip_enabled},
  };
  for (const auto& policy : redesign_policies) {
    set_environment(policy.first, "0");
    require(!policy.second(), "redesign rollback policy was ignored");
    set_environment(policy.first, "1");
    require(policy.second(), "redesign opt-in policy was ignored");
    set_environment(policy.first, "1invalid");
    bool rejected = false;
    try { (void)policy.second(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "redesign accepted a partially parsed boolean");
    set_environment(policy.first, "0");
  }
  ModelConfig cpu_config;
  cpu_config.d_model = 64; cpu_config.num_layers = 1; cpu_config.vocab_size = 64;
  cpu_config.n_heads = 4; cpu_config.n_kv_heads = 2; cpu_config.mamba_head_dim = 32;
  JambaModel cpu_model(cpu_config, Device::CPU);
  set_environment("NSOS_MOE_WMMA_TRAINING", "1");
  bool cpu_wmma_rejected = false;
  try { (void)capture_runtime_execution_identity(cpu_model, 32, false); }
  catch (const std::runtime_error& error) {
    cpu_wmma_rejected = std::string(error.what()).find("fully GPU-resident") != std::string::npos;
  }
  set_environment("NSOS_MOE_WMMA_TRAINING", "0");
  require(cpu_wmma_rejected, "CPU identity pretended to honor GPU WMMA policy");
  set_environment("NSOS_KAN_RECOMPUTE_TRAINING", "1");
  bool cpu_kan_rejected=false;
  try { (void)capture_runtime_execution_identity(cpu_model,32,false); }
  catch (const std::runtime_error& error) {
    cpu_kan_rejected=std::string(error.what()).find("fully GPU-resident")!=std::string::npos;
  }
  set_environment("NSOS_KAN_RECOMPUTE_TRAINING", "0");
  require(cpu_kan_rejected,"CPU identity pretended to honor GPU KAN recompute policy");
  cpu_config.use_moe = true; cpu_config.moe_period = 1; cpu_config.moe_slot = 0;
  cpu_config.num_experts = 4; cpu_config.num_experts_per_token = 1;
  JambaModel sparse_model(cpu_config, Device::CPU);
  const auto sparse_identity = capture_runtime_execution_identity(sparse_model, 32, false);
  auto legacy_sparse_fields = sparse_identity.fields;
  const auto sparse_field = std::find_if(legacy_sparse_fields.begin(), legacy_sparse_fields.end(),
      [](const auto& field) { return field.first == "optimizer.sparse_gradient_policy"; });
  require(sparse_field != legacy_sparse_fields.end() && sparse_field->second == "explicit_group_contribution_host_v1",
          "sparse correction is not recorded in execution identity");
  legacy_sparse_fields.erase(sparse_field);
  const auto legacy_sparse_identity = validated_runtime_execution_identity(legacy_sparse_fields);
  require(!runtime_execution_identity_equal(sparse_identity, legacy_sparse_identity),
          "legacy sparse trajectory is interchangeable with corrected activity");
  require(runtime_execution_identity_mismatch(sparse_identity, legacy_sparse_identity).find("optimizer.sparse_gradient_policy") != std::string::npos,
          "sparse compatibility diagnostic is not actionable");
  return 0;
}
