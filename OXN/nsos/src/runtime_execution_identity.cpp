#include "cuda/mamba3_projection_wmma.cuh"
#include "cuda/mamba3_layer_kernels.cuh"
#include "../include/runtime_execution_identity.h"
#include "../include/gpu_gemm_provider.h"

#include "../include/jamba.h"
#include "../include/gpu_backend.h"
#include "../include/nsos/determinism.h"
#include "../include/nsos/sha256.h"
#include "../include/optimizer_runtime_policy.h"
#include "../include/muon_math.h"
#include "../include/training_runtime_policy.h"
#include "../include/tensor.h"

#ifdef USE_CUDA
#include "../include/cuda/mamba_kernels.cuh"
#include "../include/cuda/moe_training_wmma.cuh"
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>

namespace nsos {
namespace {

constexpr std::size_t kMaximumIdentityFields = 128;
constexpr std::size_t kMaximumIdentityKeyBytes = 128;
constexpr std::size_t kMaximumIdentityValueBytes = 4096;

std::shared_mutex& runtime_policy_mutex() {
    static std::shared_mutex mutex;
    return mutex;
}

thread_local std::size_t runtime_policy_lease_depth = 0;

const char* boolean_text(bool value) noexcept {
    return value ? "true" : "false";
}

bool environment_exactly_one(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

bool environment_default_on(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

std::string environment_value(const char* name,
                              const char* default_value) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0'
               ? std::string(default_value)
               : std::string(value);
}

std::string precision_name(int mode) {
    switch (mode) {
        case 0:
            return "fp32";
        case 1:
            return "bf16";
        case 2:
            return "fp16";
        default:
            throw std::logic_error(
                "Runtime identity observed an invalid matmul precision mode");
    }
}

#ifdef USE_CUDA
struct GpuLibraryVersions {
    int runtime = 0;
    int driver = 0;
    int blas = 0;
};

const GpuLibraryVersions& gpu_library_versions() {
    // Identity is checked on every protected training operation. Querying and
    // constructing a BLAS handle there would itself become a hot-path
    // synchronization/allocation cost, so resolve immutable process library
    // versions exactly once after the selected GPU context exists.
    static const GpuLibraryVersions versions = [] {
        GpuLibraryVersions result;
        if (cudaRuntimeGetVersion(&result.runtime) != cudaSuccess ||
            cudaDriverGetVersion(&result.driver) != cudaSuccess ||
            result.runtime <= 0 || result.driver <= 0) {
            throw std::runtime_error(
                "Cannot capture GPU runtime/driver versions for execution "
                "identity");
        }
        cublasHandle_t version_handle = nullptr;
        if (cublasCreate(&version_handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Cannot create BLAS handle for execution identity");
        }
        const cublasStatus_t query_status =
            cublasGetVersion(version_handle, &result.blas);
        const cublasStatus_t destroy_status =
            cublasDestroy(version_handle);
        if (query_status != CUBLAS_STATUS_SUCCESS ||
            destroy_status != CUBLAS_STATUS_SUCCESS || result.blas <= 0) {
            throw std::runtime_error(
                "Cannot capture BLAS version for execution identity");
        }
        return result;
    }();
    return versions;
}
#endif

void add_field(RuntimeExecutionIdentity& identity,
               std::string key,
               std::string value) {
    identity.fields.emplace_back(std::move(key), std::move(value));
}

std::string canonical_fields(
    const std::vector<RuntimeExecutionIdentity::Field>& fields) {
    std::ostringstream output;
    for (const auto& [key, value] : fields) {
        output << key.size() << ':' << key
               << value.size() << ':' << value;
    }
    return output.str();
}

const RuntimeExecutionIdentity::Field* find_field(
    const RuntimeExecutionIdentity& identity,
    const std::string& key) noexcept {
    const auto found = std::lower_bound(
        identity.fields.begin(), identity.fields.end(), key,
        [](const RuntimeExecutionIdentity::Field& field,
           const std::string& candidate) {
            return field.first < candidate;
        });
    return found != identity.fields.end() && found->first == key
               ? &*found
               : nullptr;
}

}  // namespace

RuntimeExecutionPolicyLease::RuntimeExecutionPolicyLease() {
    if (runtime_policy_lease_depth == 0) {
        lock_.emplace(runtime_policy_mutex());
    }
    ++runtime_policy_lease_depth;
}

RuntimeExecutionPolicyLease::~RuntimeExecutionPolicyLease() {
    if (runtime_policy_lease_depth > 0) {
        --runtime_policy_lease_depth;
    }
}

RuntimeExecutionPolicyMutationGuard::RuntimeExecutionPolicyMutationGuard() {
    if (runtime_policy_lease_depth != 0) {
        throw std::runtime_error(
            "Runtime execution policy cannot mutate inside an active "
            "training/checkpoint operation");
    }
    lock_.emplace(runtime_policy_mutex());
}

RuntimeExecutionPolicyMutationGuard::~RuntimeExecutionPolicyMutationGuard() =
    default;

std::string RuntimeExecutionIdentity::canonical_text() const {
    return canonical_fields(fields);
}

std::string RuntimeExecutionIdentity::digest() const {
    const std::string canonical = canonical_text();
    return integrity::sha256_hex(canonical.data(), canonical.size());
}

RuntimeExecutionIdentity validated_runtime_execution_identity(
    std::vector<RuntimeExecutionIdentity::Field> fields) {
    if (fields.empty() || fields.size() > kMaximumIdentityFields) {
        throw std::runtime_error(
            "Runtime execution identity has an invalid field count");
    }
    std::sort(fields.begin(), fields.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    for (std::size_t index = 0; index < fields.size(); ++index) {
        const auto& [key, value] = fields[index];
        if (key.empty() || key.size() > kMaximumIdentityKeyBytes ||
            value.size() > kMaximumIdentityValueBytes ||
            key.find('\0') != std::string::npos ||
            value.find('\0') != std::string::npos) {
            throw std::runtime_error(
                "Runtime execution identity contains an invalid field");
        }
        if (index > 0 && fields[index - 1].first == key) {
            throw std::runtime_error(
                "Runtime execution identity contains duplicate field '" +
                key + "'");
        }
    }
    RuntimeExecutionIdentity identity{std::move(fields)};
    const auto* schema = find_field(identity, "identity.schema_version");
    if (schema == nullptr || schema->second != "3") {
        throw std::runtime_error(
            "Unsupported runtime execution identity schema");
    }
    static constexpr std::array<const char*, 16> required = {
        "checkpoint.gradient_policy",
        "determinism.reductions",
        "device.execution",
        "gemm.accumulator",
        "gemm.algorithm",
        "gemm.lowp_weight_cache_budget_bytes",
        "gemm.lowp_weight_cache_key",
        "gemm.lowp_weight_cache_schema",
        "gemm.provider",
        "gpu.backend",
        "gpu.strict_execution",
        "gpu.vendor",
        "mamba.history_layout",
        "mamba.scan_geometry",
        "optimizer.update_policy",
        "precision.matmul"};
    for (const char* key : required) {
        if (find_field(identity, key) == nullptr) {
            throw std::runtime_error(
                std::string("Runtime execution identity is missing '") +
                key + "'");
        }
    }
    return identity;
}

RuntimeExecutionIdentity capture_runtime_execution_identity(
    JambaModel& model,
    int optimizer_state_bits,
    bool dynamic_loss_scaling_enabled,
    int gradient_accumulation_steps,
    bool token_scheduler) {
    if (gradient_accumulation_steps < 1) {
        throw std::invalid_argument(
            "Runtime identity requires gradient_accumulation_steps >= 1");
    }
    if (optimizer_state_bits != 4 && optimizer_state_bits != 32) {
        throw std::invalid_argument(
            "Runtime identity requires optimizer_state_bits 4 or 32");
    }

    RuntimeExecutionIdentity identity;
    add_field(identity, "identity.schema_version", "3");
    add_field(identity, "precision.matmul",
              precision_name(matmul_precision_mode()));
    add_field(identity, "precision.accumulator", "fp32");
    add_field(identity, "precision.master_weights", "fp32");
    add_field(identity, "precision.gradients", "fp32");
    add_field(identity, "precision.dynamic_loss_scaling",
              boolean_text(dynamic_loss_scaling_enabled));
    add_field(identity, "determinism.reductions",
              boolean_text(
                  determinism::deterministic_reductions_enabled()));
    add_field(identity, "gpu.strict_execution",
              boolean_text(strict_gpu_execution()));

    bool has_gpu_parameter = false;
    bool has_cpu_parameter = false;
    bool has_sparse_gradient_parameter = false;
    for (const Parameter* parameter : model.parameters()) {
        if (parameter == nullptr) {
            continue;
        }
        has_sparse_gradient_parameter |= parameter->tracks_gradient_contributions();
        if (parameter->data.get_device() == Device::GPU) {
            has_gpu_parameter = true;
        } else {
            has_cpu_parameter = true;
        }
    }
    const std::string execution =
        has_gpu_parameter && has_cpu_parameter
            ? "heterogeneous"
            : (has_gpu_parameter ? "gpu" : "cpu");
    add_field(identity, "device.execution", execution);
    add_field(identity, "gpu.backend", gpu::backend_name());
    add_field(identity, "gpu.vendor", gpu::vendor_name());
    add_field(identity, "gpu.sync_after_launch",
              boolean_text(environment_exactly_one("NSOS_CUDA_SYNC")));
#ifdef NSOS_CUDA_PTDS
    add_field(identity, "gpu.per_thread_default_stream", "true");
#else
    add_field(identity, "gpu.per_thread_default_stream", "false");
#endif

    std::string device_index = "none";
    std::string device_name = "none";
    std::string device_architecture = "none";
    std::string device_warp_size = "0";
    std::string runtime_version = "0";
    std::string driver_version = "0";
    std::string blas_version = "0";
#ifdef USE_CUDA
    if (has_gpu_parameter) {
        int active_device = -1;
        const cudaError_t status = cudaGetDevice(&active_device);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Cannot capture active ") +
                gpu::backend_name() + " device for runtime identity: " +
                cudaGetErrorString(status));
        }
        const auto devices = gpu::enumerate_devices();
        const auto found = std::find_if(
            devices.begin(), devices.end(),
            [active_device](const gpu::DeviceInfo& info) {
                return info.index == active_device;
            });
        if (found == devices.end() || !found->compiled) {
            throw std::runtime_error(
                "Active GPU is absent from the compiled runtime identity");
        }
        device_index = std::to_string(found->index);
        device_name = found->name;
        device_architecture = found->architecture;
        device_warp_size = std::to_string(found->warp_size);
        const GpuLibraryVersions& versions = gpu_library_versions();
        runtime_version = std::to_string(versions.runtime);
        driver_version = std::to_string(versions.driver);
        blas_version = std::to_string(versions.blas);
    }
#else
    if (has_gpu_parameter) {
        throw std::runtime_error(
            "GPU parameter reached a CPU-only runtime identity build");
    }
#endif
    add_field(identity, "gpu.device_index", device_index);
    add_field(identity, "gpu.device_name", device_name);
    add_field(identity, "gpu.device_architecture", device_architecture);
    add_field(identity, "gpu.warp_size", device_warp_size);
    add_field(identity, "gpu.runtime_version", runtime_version);
    add_field(identity, "gpu.driver_version", driver_version);
    add_field(identity, "gpu.blas_version", blas_version);

    const bool deterministic =
        determinism::deterministic_reductions_enabled();
    const bool fused_optimizer =
        environment_default_on("NSOS_FUSED_OPT");
    const bool deterministic_multi_tensor_optimizer =
        environment_default_on(
            "NSOS_DETERMINISTIC_MULTI_TENSOR_OPT");
    const bool deterministic_chunked_optimizer =
        optimizer_policy::deterministic_adamw_chunked_enabled();
    const bool deterministic_deferred_finite_gate =
        optimizer_policy::deterministic_finite_gate_deferred_enabled();
    const std::string deterministic_optimizer_policy =
        deterministic && deterministic_multi_tensor_optimizer
            ? (deterministic_chunked_optimizer
                   ? "deterministic_multi_tensor_chunked_update_v3_chunk_" +
                         std::to_string(
                             optimizer_policy::
                                 deterministic_adamw_chunk_elements())
                   : "deterministic_multi_tensor_flat_update_v2") +
                  (deterministic_deferred_finite_gate
                       ? "_combined_finite_norm_status_v1"
                       : "_separate_finite_status_v1")
            : "inactive";
    add_field(identity, "optimizer.update_policy",
              optimizer_policy::device_sparse_adam_enabled()
                  ? (deterministic
                         ? "device_sparse_transaction_deterministic_v1_chunk_" + std::to_string(optimizer_policy::deterministic_adamw_chunk_elements())
                         : "device_sparse_transaction_folded_v1")
                  : deterministic
                  ? (deterministic_multi_tensor_optimizer
                         ? deterministic_optimizer_policy
                         : "deterministic_per_tensor_update_v1")
                  : (fused_optimizer
                         ? "fused_multi_tensor_v1"
                         : "ordered_per_tensor_v1"));
    add_field(identity, "optimizer.state_bits",
              std::to_string(optimizer_state_bits));
    if(optimizer_policy::muon_enabled()) {
        if(optimizer_state_bits!=32 || !has_gpu_parameter || has_cpu_parameter || matmul_precision_mode()==2)
            throw std::invalid_argument("Muon FP32 policy requires a fully GPU FP32/BF16 model with FP32 optimizer state");
        add_field(identity,"optimizer.algorithm",muon::kIdentity);
        add_field(identity,"optimizer.muon_reference",muon::kReferenceCommit);
        add_field(identity,"optimizer.muon_reference_sha256",muon::kReferenceSha256);
        add_field(identity,"optimizer.muon_coefficients","3.4445,-4.7750,2.0315;steps5;eps1e-7;momentum0.95;nesterov1");
        add_field(identity,"optimizer.muon_partition","hidden_layers_matrix_weight_except_router_embedding_head_v1");
        add_field(identity,"optimizer.muon_lr_fp32_bits",optimizer_policy::muon_learning_rate_identity());
        add_field(identity,"optimizer.muon_lr_schedule","aux_adam_schedule_ratio_times_composed_parameter_scale_v1");
        add_field(identity,"optimizer.muon_state","m_fp32_momentum_v_exact_zero_lazy_v1");
    }
    if(optimizer_policy::fused_optimizer_epilogue_enabled()) {
        if(optimizer_state_bits!=32 || !has_gpu_parameter || has_cpu_parameter)
            throw std::invalid_argument("Fused optimizer epilogue requires a fully GPU model and FP32 optimizer state");
        if(matmul_precision_mode()==2 && gradient_accumulation_steps>1)
            throw std::invalid_argument("Fused dense Adam FP16 accumulation requires a validated per-microbatch unscale policy");
        add_field(identity,"optimizer.fused_epilogue","post_vjp_global_gate_update_clear_next_group_zero_elision_v1");
    }
    if(optimizer_policy::muon_enabled() || optimizer_policy::fused_optimizer_epilogue_enabled())
        add_field(identity,"optimizer.device_adam_update_arithmetic",
                  "serial_fp32_lr_times_mhat_before_denominator_divide_v2");
    if (model.model_config().mamba3_enabled)
        add_field(identity, "optimizer.explicit_no_decay", "mamba3_dt_bias_D_v1");
    if(optimizer_policy::device_sparse_adam_enabled()) {
        // Generic contribution tracking (e.g. Mamba3) does not imply MoE.
        // Owner identity is topology-stable and independent of buffer presence.
        const bool has_moe=std::any_of(model.layers.begin(),model.layers.end(),[](const auto& block){return block && block->uses_moe();});
        if(optimizer_state_bits!=32 || !has_gpu_parameter || has_cpu_parameter)
            throw std::invalid_argument("Device optimizer transaction requires fully GPU FP32 moments");
        if(optimizer_policy::legacy_device_sparse_adam_enabled() && !has_moe)
            throw std::invalid_argument("NSOS_MOE_DEVICE_ADAM=1 requires at least one MoE block");
        if(has_moe && (!training_policy::grouped_moe_training() || !training_policy::ordered_moe()))
            throw std::invalid_argument("MoE device optimizer requires grouped ordered MoE");
        add_field(identity,"optimizer.sparse_gradient_policy",optimizer_policy::kDeviceSparseGradientIdentity);
        add_field(identity,"optimizer.device_sparse_transaction","predicate_finite_lazy_snapshot_versions_v1");
        if(has_moe)add_field(identity,"moe.qat_regularization_membership","task_union_active_only_v1");
        else add_field(identity,"optimizer.dense_gradient_bank","explicit_dense_contribution_finite_lazy_snapshot_versions_v1");
    } else if(has_sparse_gradient_parameter) {
        add_field(identity,"optimizer.sparse_gradient_policy","explicit_group_contribution_host_v1");
    }
    add_field(identity, "optimizer.norm_accumulator",
              deterministic ? "fp64_fixed_order" : "fp32_backend_order");
    // Optional fields preserve legacy fingerprints when a redesign is off.
    // Changing any active arithmetic/retention policy must fail closed on resume.
    if (training_policy::head_cce()) {
        add_field(identity, "head.loss_policy", training_policy::head_cce_identity);
    }
    if (optimizer_policy::device_sparse_adam_enabled()) {
        add_field(identity,"optimizer.activity_norm_policy",deterministic?"device_activity_fp64_tree256_v1":"device_activity_fp32_folded_v1");
    }
    const auto mamba3_projection_policy=mamba3_projection::policy();
    if(mamba3_projection_policy!=mamba3_projection::Policy::ExactFP32) {
        if(!model.model_config().mamba3_enabled)
            throw std::invalid_argument("Mamba3 WMMA projection policy requires Mamba3 architecture");
        if(!has_gpu_parameter || has_cpu_parameter)
            throw std::invalid_argument("Mamba3 WMMA projection policy requires fully GPU model");
#ifdef USE_CUDA
        if(!mamba3_projection::supported(mamba3_projection_policy))
            throw std::invalid_argument("Mamba3 WMMA projection policy requires compiled RDNA3 wave32");
#else
        throw std::invalid_argument("Mamba3 WMMA projection policy requires GPU build");
#endif
        add_field(identity,"mamba3.projection_arithmetic",mamba3_projection::identity(mamba3_projection_policy));
    }
    const auto attention_provider = training_policy::attention_provider();
    if (attention_provider != attention_training::Policy::FP32) {
        add_field(identity, "attention.training_provider", attention_training::policy_identity(attention_provider));
        add_field(identity, "attention.training_abi", attention_training::integration_identity);
        add_field(identity, "attention.arithmetic_abi", attention_rdna::identity);
        add_field(identity, "attention.training_tape", "owned_prefix_rope_qkv_stats_single_use_recycle_v1");
        add_field(identity, "attention.training_status", "sticky_parameter_ledger_event_bridge_finite_gate_v1");
    }
    if (optimizer_policy::device_gradient_clip_enabled()) {
        add_field(identity, "optimizer.clip_policy", "device_fp64_chunk8192_tree256_v1");
    }
    if (!model.model_config().mamba3_enabled && training_policy::boundary_history()) {
        add_field(identity, "mamba.history_retention", "entering_boundary_chunk32_recompute_v1");
    }
    if (training_policy::tiled_attention()) {
        add_field(identity, "attention.training_policy", "exact_fp32_online_tile8_ordered_v1");
    }
    if (training_policy::ordered_moe()) {
        add_field(identity, "moe.dispatch_policy", "stable_expert_row_order_device_combine_v1");
    }
    if (training_policy::grouped_moe_training()) {
        add_field(identity, "moe.training_compute_policy", "device_segmented_tile16_active_qat_v2");
    }
    if (training_policy::moe_wmma_training()) {
        if (!has_gpu_parameter || has_cpu_parameter)
            throw std::runtime_error("NSOS_MOE_WMMA_TRAINING requires a fully GPU-resident model");
#ifdef USE_CUDA
        if (!moe_training_wmma_supported())
            throw std::runtime_error("NSOS_MOE_WMMA_TRAINING unavailable: compiled rocWMMA/RDNA3 wave32 required");
#else
        throw std::runtime_error("NSOS_MOE_WMMA_TRAINING requires a GPU build");
#endif
        add_field(identity, "moe.training_wmma_policy", training_policy::moe_wmma_identity);
    }
    if (training_policy::full_ttt_bptt()) {
        add_field(identity, "ttt.training_policy", "full_sequence_bptt_isolated_boundary32_column_q_v2");
    } else if (training_policy::device_ttt()) {
        add_field(identity, "ttt.training_policy", "device_recurrence_fp64_norm_truncated_v1");
    }
    if (training_policy::kan_recompute_training()) {
        if (!has_gpu_parameter || has_cpu_parameter)
            throw std::runtime_error("NSOS_KAN_RECOMPUTE_TRAINING requires a fully GPU-resident model");
#ifndef USE_CUDA
        throw std::runtime_error("NSOS_KAN_RECOMPUTE_TRAINING requires a GPU build");
#endif
        add_field(identity, "kan.training_policy", training_policy::kan_recompute_identity);
    }
    if (training_policy::kan_wmma_training()) {
        if (!training_policy::kan_recompute_training())
            throw std::runtime_error("NSOS_KAN_WMMA_TRAINING requires NSOS_KAN_RECOMPUTE_TRAINING");
#ifdef USE_CUDA
        if (!moe_training_wmma_supported())
            throw std::runtime_error("KAN WMMA requires compiled RDNA3 wave32 rocWMMA support");
#else
        throw std::runtime_error("KAN WMMA requires a GPU build");
#endif
        add_field(identity, "kan.wmma_policy", training_policy::kan_wmma_identity);
    }
    const bool finite_check_chunked =
        optimizer_policy::optimizer_finite_chunked_enabled();
    add_field(identity, "optimizer.finite_check_policy",
              finite_check_chunked
                  ? "multi_tensor_chunked_boolean_v1_chunk_" +
                        std::to_string(
                            optimizer_policy::
                                optimizer_finite_chunk_elements())
                  : "multi_tensor_flat_binary_search_v1");

    const ModelConfig& config = model.model_config();
    const auto mamba3_scan_provider=mamba3_block::gpu_provider_from_environment();
    if(mamba3_scan_provider!=mamba3_block::GpuProvider::DenseReference) {
        if(!config.mamba3_enabled || !has_gpu_parameter || has_cpu_parameter)
            throw std::invalid_argument("Mamba3 parallel/Flash requires fully GPU Mamba3 model");
        add_field(identity,"mamba3.scan_provider",
            mamba3_block::is_flash_provider(mamba3_scan_provider)
                ? (mamba3_block::is_hierarchical_provider(mamba3_scan_provider) ? "flash_fp32_hierarchical_v1" : mamba3_scan_provider==mamba3_block::GpuProvider::FlashFp32ReplayLdsV2 ? "flash_fp32_replay_lds_v2" : "flash_fp32_v1") : "parallel_fp32_v1");
        add_field(identity,"mamba3.scan_tile_tokens",std::to_string(mamba3_block::parallel_tile));
        if(mamba3_block::is_replay_lds_provider(mamba3_scan_provider)) {
            add_field(identity,"mamba3.state_prefix",mamba3_block::is_hierarchical_provider(mamba3_scan_provider)?"tile32_hs_arity32_hs_prefix_fmaf_parent_fixup_v1":"tile32_hillis_steele_sequential_chunk_carry_v1");
            add_field(identity,"mamba3.state_suffix",mamba3_block::is_hierarchical_provider(mamba3_scan_provider)?"tile32_hs_arity32_hs_suffix_fmaf_parent_fixup_v1":"tile32_hillis_steele_sequential_reverse_carry_v1");
            if(mamba3_block::is_hierarchical_provider(mamba3_scan_provider)) {
                add_field(identity,"mamba3.hierarchy_arity","32");
                add_field(identity,"mamba3.hierarchy_scratch","owned_pair_tree_fp32_reused_after_forward_v1");
                add_field(identity,"mamba3.boundary_arithmetic","inclusive_chunk_pair_fmaf_public_seed_v1");
            }
            add_field(identity,"mamba3.backward_replay","t4_p1_n128_stride129_explicit_scalar_seam_halos_v2");
            add_field(identity,"mamba3.phase_order","serial_token_fp32_wrap_v1");
        }
    }
    if (config.mamba3_enabled) {
        add_field(identity, "mamba3.implementation", "mamba3_dense_fp32_siso_mimo_n128_v1");
        add_field(identity, "mamba3.upstream", "e9594ce1c732d97440f0332fdc43170a2294dbfa");
        add_field(identity, "mamba3.precision", "fp32_dense_projections_fp64_norm_vjp_radial");
        add_field(identity, "mamba3.history",
            mamba3_block::is_flash_provider(mamba3_scan_provider)
                ? (mamba3_block::is_hierarchical_provider(mamba3_scan_provider)?"hier32_tile32_boundaries_t4_lds_scalar_halos_v1":mamba3_scan_provider==mamba3_block::GpuProvider::FlashFp32ReplayLdsV2 ? "tile32_boundaries_t4_lds_explicit_halos_v2" : "tile32_boundaries_replay_v1") : "dense_bh_time_pn_v1");
        add_field(identity, "mamba3.publication", "explicit_all_batch_status_audit_v1");
    }
    add_field(identity, "checkpoint.gradient_policy",
              config.mamba3_enabled
                  ? (mamba3_block::is_flash_provider(mamba3_scan_provider)
                        ? (mamba3_block::is_hierarchical_provider(mamba3_scan_provider)?"retain_mamba3_hier32_tile32_boundary_replay_lds_v1":mamba3_scan_provider==mamba3_block::GpuProvider::FlashFp32ReplayLdsV2 ? "retain_mamba3_tile32_boundary_replay_lds_v2" : "retain_mamba3_tile32_boundary_replay_v1")
                        : "retain_mamba3_full_history_v1")
                  : config.use_gradient_checkpointing
                  ? "selective_faithful_recompute_v1"
                  : "retain_full_history_v1");
    add_field(identity, "checkpoint.snapshot_transport",
              "host_owned_one_inflight_v1");
    add_field(identity, "embedding.deterministic_policy",
              environment_exactly_one(
                  "NSOS_EMBEDDING_DENSE_DETERMINISTIC")
                  ? "dense_vocab_gather_v1"
                  : "present_id_csr_ordered_v1");
    if (config.mamba3_enabled) {
        add_field(identity, "mamba.history_layout",
                  mamba3_block::is_flash_provider(mamba3_scan_provider)
                      ? (mamba3_block::is_hierarchical_provider(mamba3_scan_provider)?"mamba3_hier32_tile32_boundaries_bh_pn_v1":"mamba3_tile32_boundaries_bh_pn_v1")
                      : "mamba3_dense_bh_time_pn_v1");
        add_field(identity, "mamba.scan_geometry",
                  mamba3_scan_provider==mamba3_block::GpuProvider::DenseReference
                      ? "mamba3_serial_batch_owner_v1"
                      : mamba3_block::is_hierarchical_provider(mamba3_scan_provider)?"mamba3_tile32_local_arity32_tree_128threads_4cells_v1":"mamba3_tile32_affine_prefix_128threads_4cells_v1");
    } else {
#ifdef USE_CUDA
    add_field(identity, "mamba.history_layout",
              cuda::faithful_state_major_history_enabled()
                  ? "row_state_channel_training_v2"
                  : "row_channel_state_reference_v1");
#else
    add_field(identity, "mamba.history_layout",
              "row_channel_state_reference_v1");
#endif
    add_field(identity, "mamba.scan_geometry",
              environment_exactly_one(
                  "NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY")
                  ? "linear_256_v1"
                  : (!environment_default_on(
                         "NSOS_MAMBA_DETERMINISTIC_HEAD_WAVE_GEOMETRY")
                         ? "head_channel_specialized_with_linear_fallback_v1"
                         : "head_channel_specialized_deterministic_head_wave_"
                           "with_linear_fallback_v2"));
    }
    // Compile-time state-width specialization of the faithful forward scan.
    // Recorded only while active so the strict default keeps its existing
    // digest and previously written checkpoints continue to resume. The
    // unrolled kernel contracts multiply-add pairs differently, so a
    // fast-mode checkpoint is not bitwise interchangeable with a strict one
    // and must not resume as if it were.
    // Contrato de acumulação de gradiente. Emitido apenas fora do padrão para
    // não alterar o digest dos checkpoints existentes; presente, torna um
    // checkpoint A>1 não-retomável como A=1, que é uma trajetória distinta.
    if (gradient_accumulation_steps > 1) {
        add_field(identity, "optimizer.gradient_accumulation_steps",
                  std::to_string(gradient_accumulation_steps));
        add_field(identity, "optimizer.gradient_accumulation",
                  "mean_before_clip");
        add_field(identity, "optimizer.gradient_accumulation_version", "1");
    }
    // Unidade do scheduler. A ausência significa o contrato legado em passos.
    if (token_scheduler) {
        add_field(identity, "scheduler.unit", "tokens");
    }
    if (!config.mamba3_enabled) {
    if (environment_exactly_one("NSOS_MAMBA_FAITHFUL_FIXED_STATE")) {
        add_field(identity, "mamba.forward_state_width",
                  "compile_time_fixed_v1");
    }
    // Time-parallel forward scan. Also recorded only while active. The chunk
    // entry carry is accumulated as D_c*carry + end_local rather than step by
    // step, so a chunked-forward checkpoint is mathematically equivalent but
    // not bitwise interchangeable with a sequential one, and the chunk width
    // is part of that contract.
    if (environment_exactly_one("NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD")) {
        add_field(identity, "mamba.forward_scan_decomposition",
                  "time_chunked_affine_carry_v1");
#ifdef USE_CUDA
        add_field(identity, "mamba.forward_chunk_size",
                  std::to_string(cuda::faithful_forward_chunk_size()));
#endif
    }
    add_field(identity, "mamba.deterministic_carry_policy",
              environment_exactly_one(
                  "NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY")
                  ? "inactive_linear_geometry"
                  : (environment_default_on(
                         "NSOS_MAMBA_DETERMINISTIC_HEAD_WAVE_GEOMETRY") &&
                             environment_exactly_one(
                                 "NSOS_MAMBA_DETERMINISTIC_SHARED_CARRY")
                         ? "eligible_wave_block_shared_v1"
                         : "thread_local_reference_v1"));
    add_field(identity, "mamba.decay_policy",
              environment_default_on("NSOS_MAMBA_CHUNKED_BACKWARD")
                  ? "precomputed_chunked_eligible_terms_v3"
                  : (environment_default_on(
                         "NSOS_MAMBA_PRECOMPUTE_DECAY")
                         ? "precomputed_row_head_terms_reused_v2"
                         : "per_channel_reference_v1"));
    add_field(identity, "mamba.deterministic_backward_policy",
              environment_default_on("NSOS_MAMBA_CHUNKED_BACKWARD")
                  ? "time_chunked_affine_boundary_when_multitile_v2"
                  : (!environment_exactly_one(
                         "NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY") &&
                             environment_default_on(
                                 "NSOS_MAMBA_DETERMINISTIC_HEAD_WAVE_GEOMETRY") &&
                             environment_exactly_one(
                                 "NSOS_MAMBA_STATE_PARALLEL_BACKWARD")
                         ? "state_parallel_exact_order_v1"
                         : "fused_channel_state_reference_v1"));
    add_field(identity, "mamba.backward_chunk_size",
              environment_default_on("NSOS_MAMBA_CHUNKED_BACKWARD")
#ifdef USE_CUDA
                  ? std::to_string(cuda::faithful_backward_chunk_size())
#else
                  ? "unsupported_without_gpu_backend"
#endif
                  : "inactive");
#ifdef USE_CUDA
    add_field(identity, "mamba.chunk_lds_layout",
              environment_default_on("NSOS_MAMBA_CHUNKED_BACKWARD")
                  ? (cuda::faithful_chunk_lds_state_major_enabled()
                         ? "state_lane_bank_coalesced_v1"
                         : "lane_state_reference_v1")
                  : "inactive");
#else
    add_field(identity, "mamba.chunk_lds_layout",
              "inactive_without_gpu_backend");
#endif
    add_field(identity, "mamba.silu_gate_policy",
              "fused_gpu_reference_cpu_v1");
    }
    add_field(identity, "loss.cross_entropy_policy",
              "one_exp_device_scalar_v1");

    const gpu::GemmProviderPolicy gemm_policy =
        gpu::gemm_provider_policy(has_gpu_parameter);
    add_field(identity, "gemm.provider", gemm_policy.provider_name);
    add_field(identity, "gemm.algorithm", gemm_policy.algorithm_policy);
    add_field(identity, "gemm.lt_compiled",
              boolean_text(gemm_policy.lt_compiled));
    add_field(identity, "gemm.lt_promoted",
              boolean_text(gemm_policy.lt_promoted));
    add_field(identity, "gemm.accumulator", "fp32");
    add_field(identity, "gemm.transpose_policy", "native_blas_ops_v1");
    const char* matmul_tn_policy = "native_blas_transpose_v1";
    if (environment_exactly_one("NSOS_MATMUL_TN_MATERIALIZE")) {
        matmul_tn_policy = "materialized_reference_v1";
    } else if (matmul_precision_mode() != 0) {
#if defined(NSOS_GPU_BACKEND_HIP)
        matmul_tn_policy = "fused_lowp_cast_transpose_gemm_ex_v2";
#else
        matmul_tn_policy =
            "native_gemm_strided_batched_ex_transpose_v2";
#endif
    }
    add_field(identity, "gemm.matmul_tn_policy", matmul_tn_policy);
    add_field(identity, "gemm.matmul_nt_mixed_policy",
              environment_exactly_one(
                  "NSOS_MATMUL_NT_MIXED_MATERIALIZE")
                  ? "materialized_reference_v1"
                  : "native_gemm_ex_transpose_v1");
    add_field(identity, "gemm.lowp_weight_cache_schema", "1");
    add_field(identity, "gemm.lowp_weight_cache_key",
              "storage_owner_offset_shape_device_mode_policy_epoch_"
              "content_version_v1");
    add_field(identity, "gemm.lowp_weight_cache_budget_bytes",
              std::to_string(lowp_weight_cache_budget_bytes()));

    if (!config.mamba3_enabled) {
#ifdef USE_CUDA
    add_field(identity, "mamba.parallel_selective_scan",
              boolean_text(cuda::mamba_parallel_scan_enabled()));
    add_field(identity, "mamba.faithful_warp_aggregation",
              boolean_text(cuda::faithful_warp_aggregation_enabled()));
    add_field(identity, "mamba.faithful_reduced_conv",
              boolean_text(cuda::faithful_reduced_conv_enabled()));
    add_field(identity, "mamba.faithful_k4_conv",
              boolean_text(cuda::faithful_k4_conv_enabled()));
#else
    add_field(identity, "mamba.parallel_selective_scan", "false");
    add_field(identity, "mamba.faithful_warp_aggregation", "false");
    add_field(identity, "mamba.faithful_reduced_conv", "false");
    add_field(identity, "mamba.faithful_k4_conv", "false");
#endif
    }

    // These controls are not all active for Mamba-only, but recording their
    // normalized values keeps hybrid/direct-API resumes auditable as well.
    add_field(identity, "control.gpu_memory",
              environment_value("NSOS_GPU_MEMORY", "default"));
    add_field(identity, "control.async_d2d",
              environment_value("NSOS_ASYNC_D2D", "default"));
    add_field(identity, "control.uninitialized_allocations",
              environment_value("NSOS_UNINIT", "default"));
    add_field(identity, "control.allow_host_fallback",
              environment_value("NSOS_GPU_ALLOW_HOST_FALLBACK", "default"));
    add_field(identity, "control.attention_backward_host",
              environment_value("NSOS_ATTN_BWD_HOST", "default"));
    add_field(identity, "control.rul_host",
              environment_value("NSOS_RUL_HOST", "default"));
    add_field(identity, "control.moe_router_grad",
              environment_value("NSOS_MOE_ROUTER_GRAD", "default"));
    add_field(identity, "control.moe_fp_router",
              environment_value("NSOS_MOE_FP_ROUTER", "default"));
    add_field(identity, "control.criticality_regularizer",
              environment_value("NSOS_CRIT_REG", "0"));
    add_field(identity, "control.criticality_regularizer_eta",
              environment_value("NSOS_CRIT_REG_ETA", "0.2"));
    add_field(identity, "control.criticality_regularizer_every",
              environment_value("NSOS_CRIT_REG_EVERY", "10"));
    add_field(identity, "control.criticality_lr",
              environment_value("NSOS_CRIT_LR", "0"));
    add_field(identity, "control.criticality_lr_eta",
              environment_value("NSOS_CRIT_LR_ETA", "0.25"));
    add_field(identity, "control.train_chunk_size",
              environment_value("NSOS_TRAIN_CHUNK_SIZE", "default"));
    add_field(identity, "telemetry.train_timing",
              boolean_text(
                  environment_exactly_one("NSOS_TRAIN_TIMING")));

    return validated_runtime_execution_identity(
        std::move(identity.fields));
}

bool runtime_execution_identity_equal(
    const RuntimeExecutionIdentity& expected,
    const RuntimeExecutionIdentity& actual) noexcept {
    return expected.fields == actual.fields;
}

std::string runtime_execution_identity_mismatch(
    const RuntimeExecutionIdentity& expected,
    const RuntimeExecutionIdentity& actual) {
    std::ostringstream message;
    message << "Runtime execution identity mismatch";
    std::size_t expected_index = 0;
    std::size_t actual_index = 0;
    std::size_t differences = 0;
    while (expected_index < expected.fields.size() ||
           actual_index < actual.fields.size()) {
        const auto* expected_field =
            expected_index < expected.fields.size()
                ? &expected.fields[expected_index]
                : nullptr;
        const auto* actual_field =
            actual_index < actual.fields.size()
                ? &actual.fields[actual_index]
                : nullptr;
        if (expected_field != nullptr &&
            (actual_field == nullptr ||
             expected_field->first < actual_field->first)) {
            message << "; " << expected_field->first << ": expected='"
                    << expected_field->second << "', actual=<missing>";
            ++expected_index;
            ++differences;
        } else if (actual_field != nullptr &&
                   (expected_field == nullptr ||
                    actual_field->first < expected_field->first)) {
            message << "; " << actual_field->first
                    << ": expected=<missing>, actual='"
                    << actual_field->second << "'";
            ++actual_index;
            ++differences;
        } else {
            if (expected_field->second != actual_field->second) {
                message << "; " << expected_field->first << ": expected='"
                        << expected_field->second << "', actual='"
                        << actual_field->second << "'";
                ++differences;
            }
            ++expected_index;
            ++actual_index;
        }
        if (differences == 8) {
            message << "; additional differences omitted";
            break;
        }
    }
    return message.str();
}

}  // namespace nsos
