#include "../include/jamba.h"
#include "../include/layer_audit.h"
#include "../include/nsos/determinism.h"
#include "../include/nsos_sdk.h"
#include "../include/tensor.h"
#include "../include/trainer.h"
#include "../include/gpu_backend.h"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <cstring>
#include <limits>
#include <vector>

#ifdef USE_CUDA
#include "../include/cuda/kernels.cuh"  // nsos_bench_d2h_copy
#endif

namespace py = pybind11;
using namespace nsos;

namespace {

py::array_t<float> tensor_to_numpy(Tensor& tensor) {
    // Inspect the authoritative extent before cloning/copying. The default
    // rank-zero sentinel has size 0; a real Tensor({}) scalar has size 1.
    if (tensor.size == 0) {
        std::vector<py::ssize_t> empty_shape;
        for (int dim : tensor.shape.dims)
            empty_shape.push_back(static_cast<py::ssize_t>(dim));
        if (empty_shape.empty()) empty_shape.push_back(0);
        return py::array_t<float>(empty_shape);
    }
    auto* holder = new Tensor(tensor.get_device() == Device::GPU ? tensor.cpu() : tensor.clone());
    py::capsule base(holder, [](void* ptr) {
        delete static_cast<Tensor*>(ptr);
    });

    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    shape.reserve(holder->shape.dims.size());
    strides.reserve(holder->shape.strides.size());
    for (int dim : holder->shape.dims) {
        shape.push_back(static_cast<py::ssize_t>(dim));
    }
    for (size_t stride : holder->shape.strides) {
        strides.push_back(static_cast<py::ssize_t>(stride * sizeof(float)));
    }

    return py::array(py::buffer_info(holder->data(),
                                     sizeof(float),
                                     py::format_descriptor<float>::format(),
                                     static_cast<py::ssize_t>(shape.size()),
                                     shape,
                                     strides),
                     base);
}

template <typename T>
auto trainer_getter(T Trainer::*member) {
    return [member](const Trainer& trainer) {
        return trainer.synchronized_read(member);
    };
}

template <typename T>
auto trainer_setter(T Trainer::*member) {
    return [member](Trainer& trainer, const T& value) {
        trainer.synchronized_write(member, value);
    };
}

}  // namespace

PYBIND11_MODULE(nsos_ext, m) {
    m.doc() = "NSOS Python bindings";

    m.def("fast_gpu_supported", []() {
#ifdef USE_CUDA
        int device_id = -1;
        if (!gpu::select_preferred_device(&device_id, nullptr)) {
            return false;
        }
#if defined(NSOS_GPU_BACKEND_CUDA)
#ifndef NSOS_CUDA_MIN_ARCH
#define NSOS_CUDA_MIN_ARCH 0
#endif
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
            return false;
        }
        const int device_arch = props.major * 10 + props.minor;
        if (NSOS_CUDA_MIN_ARCH > 0 && device_arch < NSOS_CUDA_MIN_ARCH) {
            return false;
        }
#endif
        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        return cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS;
#else
        return false;
#endif
    });
    m.def("gpu_backend_name", []() {
        return std::string(gpu::backend_name());
    });
    m.def("gpu_vendor_name", []() {
        return std::string(gpu::vendor_name());
    });
    m.def("attention_training_runtime", []() {
        py::dict result;
        result["provider"] = attention_training::policy_identity(attention_training::policy());
        result["integration_abi"] = attention_training::integration_identity;
        result["arithmetic_abi"] = attention_rdna::identity;
        const auto c = attention_training::dispatch_counters();
        result["forwards"] = c[0]; result["backwards"] = c[1]; result["finite_merges"] = c[2];
        return result;
    });
    m.def("gpu_dispatch_counters", []() {
        const auto counts = gpu::dispatch_counters();
        const char* names[] = {"bitnet_gemv", "bitnet_gemm", "sparse_moe", "grouped_projection",
            "tiled_attention", "compact_attention", "mamba_epilogue", "graph_replay", "grouped_moe_training",
            "grouped_moe_wmma_gemm", "grouped_moe_gradient_commit", "kan_recompute", "kan_wmma_gemm",
            "mamba3_siso_forward", "mamba3_siso_backward",
            "mamba3_preprocess_forward", "mamba3_preprocess_backward", "mamba3_projection_wmma"};
        static_assert(sizeof(names) / sizeof(names[0]) == static_cast<unsigned>(gpu::DispatchPath::Count));
        py::dict result;
        for (size_t i = 0; i < counts.size(); ++i) result[names[i]] = counts[i];
        return result;
    }, "Host dispatch attempts; capture records are counted once, graph launches separately.");
    m.def("selected_gpu_device", []() {
        int selected = -1;
        std::string error;
        if (!gpu::select_preferred_device(&selected, &error)) {
            throw std::runtime_error(
                "Could not select GPU: " + error);
        }
        return selected;
    });
    m.def("gpu_devices", []() {
        py::list result;
        for (const auto& device : gpu::enumerate_devices()) {
            py::dict item;
            item["index"] = device.index;
            item["name"] = device.name;
            item["architecture"] = device.architecture;
            item["total_memory"] = device.total_memory;
            item["warp_size"] = device.warp_size;
            item["integrated"] = device.integrated;
            item["compiled"] = device.compiled;
            item["fp16"] = device.fp16;
            item["bf16"] = device.bf16;
            result.append(std::move(item));
        }
        return result;
    });
    m.def("pool_stats", []() {
        const auto st = pool_stats();
        py::dict d;
        d["cached_bytes"] = st.cached_bytes;
        d["live_bytes"] = st.live_bytes;
        d["device_cached_bytes"] = st.device_cached_bytes;
        d["managed_cached_bytes"] = st.managed_cached_bytes;
        d["device_live_bytes"] = st.device_live_bytes;
        d["managed_live_bytes"] = st.managed_live_bytes;
        d["quarantined_bytes"] = st.quarantined_bytes;
        d["retained_release_bytes"] =
            st.retained_release_bytes;
        d["cached_blocks"] = st.cached_blocks;
        d["live_blocks"] = st.live_blocks;
        d["quarantined_blocks"] = st.quarantined_blocks;
        d["retained_release_blocks"] =
            st.retained_release_blocks;
        d["bins"] = st.bins;
        d["allocated_bytes"] = st.allocated_bytes;
        d["reserved_bytes"] = st.reserved_bytes;
        d["peak_allocated_bytes"] = st.peak_allocated_bytes;
        d["peak_reserved_bytes"] = st.peak_reserved_bytes;
        d["largest_live_block_bytes"] =
            st.largest_live_block_bytes;
        d["largest_cached_block_bytes"] =
            st.largest_cached_block_bytes;
        d["cached_fragmentation_ratio"] =
            st.cached_fragmentation_ratio;
        d["managed_pressure_probes"] =
            st.managed_pressure_probes;
        d["managed_advice_failures"] =
            st.managed_advice_failures;
        d["cross_stream_domain_frees"] =
            st.cross_stream_domain_frees;
        d["release_failures"] = st.release_failures;
        d["unknown_deallocation_attempts"] =
            st.unknown_deallocation_attempts;
        d["capture_contract_violations"] =
            st.capture_contract_violations;
        d["pool_enabled"] = st.pool_enabled;
        d["capture_active"] = st.capture_active;
        return d;
    });
    m.def("gpu_sparse_adam_dispatch_counters", []() {
        const auto c=gpu_sparse_adam_dispatch_counters();py::dict d;
        d["adamw_device_commits"]=c.adamw_device_commits;
        d["adamw_fused_epilogue_commits"]=c.adamw_fused_epilogue_commits;
        d["muon_adam_commits"]=c.muon_adam_commits;
        d["muon_matrix_directions"]=c.muon_matrix_directions;
        d["fused_gradient_clear_commits"]=c.fused_gradient_clear_commits;
        return d;
    });
    m.def("gpu_transfer_stats", []() {
        const auto st = gpu_transfer_stats();
        py::dict d;
        d["h2d_calls"] = st.h2d_calls;
        d["h2d_bytes"] = st.h2d_bytes;
        d["d2h_calls"] = st.d2h_calls;
        d["d2h_bytes"] = st.d2h_bytes;
        d["d2d_calls"] = st.d2d_calls;
        d["d2d_bytes"] = st.d2d_bytes;
        d["h2h_calls"] = st.h2h_calls;
        d["h2h_bytes"] = st.h2h_bytes;
        d["device_synchronizations"] =
            st.device_synchronizations;
        d["stream_synchronizations"] =
            st.stream_synchronizations;
        return d;
    });
    m.def("reset_gpu_transfer_stats", []() {
        reset_gpu_transfer_stats();
    });
    m.def("lowp_weight_cache_stats", []() {
        const auto st = lowp_weight_cache_stats();
        py::dict d;
        d["hits"] = st.hits;
        d["misses"] = st.misses;
        d["version_refreshes"] = st.version_refreshes;
        d["evictions"] = st.evictions;
        d["budget_bypasses"] = st.budget_bypasses;
        d["allocation_failures"] = st.allocation_failures;
        d["resident_bytes"] = st.resident_bytes;
        d["peak_resident_bytes"] = st.peak_resident_bytes;
        d["budget_bytes"] = st.budget_bytes;
        return d;
    });
    m.def("reset_lowp_weight_cache_stats", []() {
        reset_lowp_weight_cache_stats();
    });
    // Per-thread-default-stream build flag (NSOS_CUDA_PTDS) — prerequisite of
    // the CUDA-graph decode path (NSOS_CUDA_GRAPH_DECODE).
    m.def("cuda_ptds_build", []() {
#ifdef NSOS_CUDA_PTDS
        return true;
#else
        return false;
#endif
    });
    // D2H per-token copy micro-benchmark: average microseconds per 4-byte
    // device->host copy through PAGEABLE vs PINNED host staging.  Returns a
    // dict; raises on GPU runtime errors / CPU builds.
    m.def("bench_d2h_copy", [](int iters) {
#ifdef USE_CUDA
        double pageable_us = 0.0;
        double pinned_us = 0.0;
        if (nsos_bench_d2h_copy(iters, &pageable_us, &pinned_us) == 0) {
            throw std::runtime_error(
                std::string("bench_d2h_copy: ") + NSOS_GPU_BACKEND_NAME +
                " runtime error");
        }
        py::dict d;
        d["iters"] = iters;
        d["pageable_us"] = pageable_us;
        d["pinned_us"] = pinned_us;
        return d;
#else
        (void)iters;
        throw std::runtime_error("bench_d2h_copy requires a GPU build");
#endif
    }, py::arg("iters") = 2000);
    m.def("release_cached_memory", []() { release_cached_memory(); });
    m.def("set_seed", [](uint64_t seed) {
        determinism::DeterminismManager::instance().set_global_seed(seed);
    });
    m.def("get_seed", []() {
        return determinism::DeterminismManager::instance().get_global_seed();
    });
    m.def("get_determinism_report", []() {
        return determinism::DeterminismManager::instance().get_determinism_report();
    });
    // Opt-in bit-reproducible gradients (NSOS_DETERMINISTIC).  Routes the
    // atomicAdd GPU fast-paths (Mamba grad_A, embedding scatter, MoE scatter)
    // to their ordered host/per-expert implementations.  Toggle at runtime.
    m.def("set_deterministic_reductions", [](bool enabled) {
        determinism::set_deterministic_reductions(enabled);
    });
    m.def("deterministic_reductions_enabled", []() {
        return determinism::deterministic_reductions_enabled();
    });

    // Vendor-neutral mixed-precision GEMM control with runtime capability gate.
    // 0=FP32 (default), 1=BF16, 2=FP16.  Master weights/optimizer stay FP32;
    // only GEMM inputs are cast.  Lets the training plane flip BF16 on a modern
    // GPU for 2-4x throughput.  Accepts ints or the strings "fp32"/"bf16"/"fp16".
    m.def("set_matmul_precision_mode", [](int mode) {
        set_matmul_precision_mode(mode);
    }, py::arg("mode"));
    m.def("set_matmul_precision", [](const std::string& s) {
        int mode = 0;
        if (s == "bf16" || s == "BF16") mode = 1;
        else if (s == "fp16" || s == "FP16") mode = 2;
        else if (s == "fp32" || s == "FP32") mode = 0;
        else throw std::invalid_argument(
            "precision must be fp32, fp16, or bf16");
        set_matmul_precision_mode(mode);
    }, py::arg("precision"));
    m.def("matmul_precision_mode", []() { return matmul_precision_mode(); });
    m.def("set_strict_gpu_execution", &set_strict_gpu_execution,
          py::arg("enabled"));
    m.def("strict_gpu_execution", &strict_gpu_execution);

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::enum_<HybridComposition>(m, "HybridComposition")
        .value("LEGACY_REPLACEMENT",
               HybridComposition::LegacyReplacement)
        .value("PARALLEL_GATED",
               HybridComposition::ParallelGated)
        .export_values();

    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("architecture_schema_version", &ModelConfig::architecture_schema_version)
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("d_model", &ModelConfig::d_model)
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("n_heads", &ModelConfig::n_heads)
        .def_readwrite("n_kv_heads", &ModelConfig::n_kv_heads)
        .def_readwrite("sliding_window", &ModelConfig::sliding_window)
        .def_readwrite("attention_period", &ModelConfig::attention_period)
        .def_readwrite("attention_slot", &ModelConfig::attention_slot)
        .def_readwrite("hybrid_composition", &ModelConfig::hybrid_composition)
        .def_readwrite("force_mamba_last_layer", &ModelConfig::force_mamba_last_layer)
        .def_readwrite("faithful_attention_linears", &ModelConfig::faithful_attention_linears)
        .def_readwrite("hybrid_mamba_gate_init", &ModelConfig::hybrid_mamba_gate_init)
        .def_readwrite("hybrid_attention_gate_init", &ModelConfig::hybrid_attention_gate_init)
        .def_readwrite("hybrid_ffn_gate_init", &ModelConfig::hybrid_ffn_gate_init)
        .def_readwrite("rope_theta", &ModelConfig::rope_theta)
        .def_readwrite("mamba_proper_ssm", &ModelConfig::mamba_proper_ssm)
        .def_readwrite("mamba_state_expansion", &ModelConfig::mamba_state_expansion)
        .def_readwrite("mamba_d_state", &ModelConfig::mamba_d_state)
        .def_readwrite("mamba_conv_kernel", &ModelConfig::mamba_conv_kernel)
        .def_readwrite("mamba2_faithful", &ModelConfig::mamba2_faithful)
        .def_readwrite("mamba3_enabled", &ModelConfig::mamba3_enabled)
        .def_readwrite("mamba3_schema_version", &ModelConfig::mamba3_schema_version)
        .def_readwrite("mamba3_state_dim", &ModelConfig::mamba3_state_dim)
        .def_readwrite("mamba3_mimo", &ModelConfig::mamba3_mimo)
        .def_readwrite("mamba3_mimo_rank", &ModelConfig::mamba3_mimo_rank)
        .def_readwrite("mamba3_outproj_norm", &ModelConfig::mamba3_outproj_norm)
        .def_readwrite("mamba3_rope_fraction", &ModelConfig::mamba3_rope_fraction)
        .def_readwrite("mamba3_norm_eps", &ModelConfig::mamba3_norm_eps)
        .def_readwrite("mamba3_a_floor", &ModelConfig::mamba3_a_floor)
        .def_readwrite("mamba_expand", &ModelConfig::mamba_expand)
        .def_readwrite("mamba_head_dim", &ModelConfig::mamba_head_dim)
        .def_readwrite("mamba_n_groups", &ModelConfig::mamba_n_groups)
        .def_readwrite("tie_word_embeddings", &ModelConfig::tie_word_embeddings)
        .def_readwrite("num_experts", &ModelConfig::num_experts)
        .def_readwrite("num_experts_per_token", &ModelConfig::num_experts_per_token)
        .def_readwrite("use_moe", &ModelConfig::use_moe)
        .def_readwrite("use_kan", &ModelConfig::use_kan)
        .def_readwrite("moe_period", &ModelConfig::moe_period)
        .def_readwrite("moe_slot", &ModelConfig::moe_slot)
        // Cherry-pick #4: Nemotron K·m invariant — override of default
        // expert FFN intermediate dim (m).  Default 0 = dm * 4 (existing behavior).
        // See OXN/nsos/docs/NEMOTRON_KM_INTEGRATION.md.
        .def_readwrite("moe_expert_hidden_dim", &ModelConfig::moe_expert_hidden_dim)
        .def_readwrite("use_ttt", &ModelConfig::use_ttt)
        .def_readwrite("ttt_period", &ModelConfig::ttt_period)
        .def_readwrite("ttt_slot", &ModelConfig::ttt_slot)
        // CHRASS topological injection (2026-05-25 wiring).
        // See OXN/nsos/docs/CHRASS_VALIDATION_REPORT.md.
        .def_readwrite("use_chrass", &ModelConfig::use_chrass)
        .def_readwrite("chrass_density", &ModelConfig::chrass_density)
        .def_readwrite("chrass_seed", &ModelConfig::chrass_seed)
        // Exact logit L2.  The old pantheon_vib name remains read/write only
        // for backward compatibility with old experiment scripts.
        .def_readwrite("logit_l2_beta", &ModelConfig::logit_l2_beta)
        .def_readwrite("pantheon_vib_beta", &ModelConfig::pantheon_vib_beta)
        // Slender embedding head-to-toe quantization (2026-05-25 wiring).
        // See OXN/nsos/docs/SLENDER_INTEGRATION.md.
        .def_readwrite("use_slender_embedding", &ModelConfig::use_slender_embedding)
        .def_readwrite("use_gradient_checkpointing", &ModelConfig::use_gradient_checkpointing)
        .def_readwrite("dropout", &ModelConfig::dropout)
        .def_readwrite("mcts_simulations", &ModelConfig::mcts_simulations)
        .def_readwrite("mcts_depth", &ModelConfig::mcts_depth)
        .def_readwrite("checkpoint_path", &ModelConfig::checkpoint_path)
        .def_readwrite("max_context_tokens", &ModelConfig::max_context_tokens)
        .def_readwrite("default_batch_size", &ModelConfig::default_batch_size)
        .def_readwrite("use_cuda", &ModelConfig::use_cuda)
        .def_readwrite("use_exact_attention_training", &ModelConfig::use_exact_attention_training)
        .def_readwrite("use_flash_attn", &ModelConfig::use_flash_attn);

    py::class_<GenerationOptions>(m, "GenerationOptions")
        .def(py::init<>())
        .def_readwrite("max_tokens", &GenerationOptions::max_tokens)
        .def_readwrite("min_new_tokens", &GenerationOptions::min_new_tokens)
        .def_readwrite("temperature", &GenerationOptions::temperature)
        .def_readwrite("top_p", &GenerationOptions::top_p)
        .def_readwrite("top_k", &GenerationOptions::top_k)
        .def_readwrite("eos_token_id", &GenerationOptions::eos_token_id)
        .def_readwrite("max_context_tokens", &GenerationOptions::max_context_tokens)
        .def_readwrite("suppress_control_tokens_at_start", &GenerationOptions::suppress_control_tokens_at_start)
        .def_readwrite("repetition_penalty", &GenerationOptions::repetition_penalty)
        .def_readwrite("no_repeat_ngram_size", &GenerationOptions::no_repeat_ngram_size)
        .def_readwrite("stream", &GenerationOptions::stream);

    py::class_<GenerationMetrics>(m, "GenerationMetrics")
        .def(py::init<>())
        .def_readwrite("prompt_tokens_total", &GenerationMetrics::prompt_tokens_total)
        .def_readwrite("prompt_tokens_used", &GenerationMetrics::prompt_tokens_used)
        .def_readwrite("generated_tokens", &GenerationMetrics::generated_tokens)
        .def_readwrite("batch_size", &GenerationMetrics::batch_size)
        .def_readwrite("elapsed_ms", &GenerationMetrics::elapsed_ms)
        .def_readwrite("prefill_ms", &GenerationMetrics::prefill_ms)
        .def_readwrite("decode_ms", &GenerationMetrics::decode_ms)
        .def_readwrite("sampler_ms", &GenerationMetrics::sampler_ms)
        .def_readwrite("prompt_tokens_per_sec", &GenerationMetrics::prompt_tokens_per_sec)
        .def_readwrite("decode_tokens_per_sec", &GenerationMetrics::decode_tokens_per_sec)
        .def_readwrite("total_tokens_per_sec", &GenerationMetrics::total_tokens_per_sec)
        .def_readwrite("used_streaming", &GenerationMetrics::used_streaming)
        .def_readwrite("loaded_from_pack", &GenerationMetrics::loaded_from_pack)
        .def_readwrite("mamba_fast_path_hits", &GenerationMetrics::mamba_fast_path_hits)
        .def_readwrite("mamba_fast_path_fallbacks", &GenerationMetrics::mamba_fast_path_fallbacks)
        .def_readwrite(
            "mamba_stream_priming_gpu_calls",
            &GenerationMetrics::mamba_stream_priming_gpu_calls)
        .def_readwrite(
            "mamba_stream_priming_host_fallbacks",
            &GenerationMetrics::mamba_stream_priming_host_fallbacks)
        .def_readwrite("mamba_last_fallback_reason", &GenerationMetrics::mamba_last_fallback_reason);

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::vector<int>, Device, float>(),
             py::arg("shape"),
             py::arg("device") = Device::CPU,
             py::arg("fill") = 0.0f)
        .def_property_readonly("shape", [](const Tensor& tensor) { return tensor.shape.dims; })
        .def_property_readonly("device", [](const Tensor& tensor) { return tensor.device; })
        .def_property_readonly("size", [](const Tensor& tensor) { return tensor.size; })
        .def("to", &Tensor::to)
        .def("cpu", &Tensor::cpu)
        .def("clone", &Tensor::clone)
        .def("numpy", &tensor_to_numpy)
        .def_static("from_numpy", [](py::array_t<float, py::array::c_style | py::array::forcecast> array) {
            std::vector<int> shape;
            for (auto dim : array.request().shape) {
                if (dim <= 0 || dim > std::numeric_limits<int>::max())
                    throw std::invalid_argument("Tensor.from_numpy requires positive INT_MAX bounded dimensions");
                shape.push_back(static_cast<int>(dim));
            }
            if (shape.empty() || array.size() > std::numeric_limits<int>::max())
                throw std::invalid_argument("Tensor.from_numpy requires a non-scalar INT_MAX bounded array");
            Tensor tensor(shape, Device::CPU);
            std::memcpy(tensor.data(), array.data(), static_cast<size_t>(array.size()) * sizeof(float));
            return tensor;
        }, py::arg("array"))
        .def("add", &Tensor::add)
        .def("sub", &Tensor::sub)
        .def("mul", py::overload_cast<const Tensor&>(&Tensor::mul, py::const_))
        .def("matmul", &Tensor::matmul)
        .def("transpose", py::overload_cast<>(&Tensor::transpose, py::const_))
        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("softmax", &Tensor::softmax, py::arg("dim") = -1)
        .def("reshape", &Tensor::reshape)
        .def("sum", &Tensor::sum, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("slice", &Tensor::slice)
        .def("clamp", &Tensor::clamp)
        .def("cross_entropy", &Tensor::cross_entropy)
        .def("mse_loss", &Tensor::mse_loss)
        .def("norm", &Tensor::norm)
        .def("zero_grad", &Tensor::zero_grad)
        .def_static("zeros",
                    py::overload_cast<const std::vector<int>&, Device>(&Tensor::zeros),
                    py::arg("shape"),
                    py::arg("device") = Device::CPU)
        .def_static("ones",
                    py::overload_cast<const std::vector<int>&, Device>(&Tensor::ones),
                    py::arg("shape"),
                    py::arg("device") = Device::CPU)
        .def_static("random",
                    py::overload_cast<const std::vector<int>&, Device>(&Tensor::random),
                    py::arg("shape"),
                    py::arg("device") = Device::CPU)
        .def_static("eye", &Tensor::eye, py::arg("n"), py::arg("device") = Device::CPU)
        .def_static("from_scalar", &Tensor::from_scalar, py::arg("value"), py::arg("device") = Device::CPU);

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("load", &Tokenizer::load, py::arg("path"))
        .def("load_text", &Tokenizer::load_text, py::arg("path"))
        .def("load_ox3", &Tokenizer::load_ox3, py::arg("path"))
        .def("load_pack", &Tokenizer::load_pack, py::arg("path"))
        .def("save_pack", &Tokenizer::save_pack, py::arg("path"))
        .def("add_special_tokens", &Tokenizer::add_special_tokens, py::arg("tokens"))
        .def("encode", &Tokenizer::encode, py::arg("text"))
        .def("decode", &Tokenizer::decode, py::arg("ids"))
        .def_property_readonly("vocab_size", [](const Tokenizer& tokenizer) {
            return tokenizer.vocab_size;
        });

    py::class_<Context>(m, "Context")
        .def(py::init<int>(), py::arg("layers") = 128);

    py::class_<Parameter>(m, "Parameter")
        .def_property_readonly("name", [](const Parameter& parameter) { return parameter.name; })
        .def_property_readonly("base_name", [](const Parameter& parameter) { return parameter.base_name; })
        .def_property_readonly("trainable", [](const Parameter& parameter) {
            return parameter.trainable;
        })
        .def_property_readonly("version", [](const Parameter& parameter) {
            return parameter.version;
        })
        .def_property_readonly("gradient_present", &Parameter::has_gradient)
        .def_property_readonly("gradient_contributions_tracked", &Parameter::tracks_gradient_contributions)
        .def_property_readonly(
            "data", [](Parameter& parameter) -> Tensor& {
                return parameter.data;
            }, py::return_value_policy::reference_internal)
        .def_property_readonly(
            "grad", [](Parameter& parameter) -> Tensor& {
                return parameter.grad;
            }, py::return_value_policy::reference_internal)
        .def("copy_data_from", &Parameter::copy_data_from,
             py::arg("source"))
        .def("zero_grad", &Parameter::zero_grad);

    // Backend microbenchmark surfaces. These are intentionally narrow: they
    // expose the same deterministic parameter mirroring and forward paths used
    // by the native benchmark without turning internal training state into a
    // mutable Python API.
    py::class_<BitLinear>(m, "BitLinear")
        .def(py::init<int, int, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("bias") = true)
        .def("forward", &BitLinear::forward, py::arg("input"),
             py::call_guard<py::gil_scoped_release>())
        .def("to", &BitLinear::to, py::arg("device"))
        .def("parameters", &BitLinear::parameters,
             py::return_value_policy::reference_internal)
        .def("set_precision_mode", &BitLinear::set_precision_mode,
             py::arg("bits"))
        .def("set_reference_path", &BitLinear::set_reference_path,
             py::arg("enabled"))
        .def("set_gpu_packed_inference",
             &BitLinear::set_gpu_packed_inference,
             py::arg("enabled"))
        .def("repack_weights", &BitLinear::repack_weights);

    py::class_<Mamba3Config>(m, "Mamba3Config")
        .def(py::init<>())
        .def_readwrite("schema_version", &Mamba3Config::schema_version)
        .def_readwrite("expand", &Mamba3Config::expand)
        .def_readwrite("head_dim", &Mamba3Config::head_dim)
        .def_readwrite("state_dim", &Mamba3Config::state_dim)
        .def_readwrite("n_groups", &Mamba3Config::n_groups)
        .def_readwrite("mimo", &Mamba3Config::mimo)
        .def_readwrite("mimo_rank", &Mamba3Config::mimo_rank)
        .def_readwrite("outproj_norm", &Mamba3Config::outproj_norm)
        .def_readwrite("rope_fraction", &Mamba3Config::rope_fraction)
        .def_readwrite("norm_eps", &Mamba3Config::norm_eps)
        .def_readwrite("a_floor", &Mamba3Config::a_floor)
        .def_readwrite("dt_min", &Mamba3Config::dt_min)
        .def_readwrite("dt_max", &Mamba3Config::dt_max)
        .def_readwrite("dt_init_floor", &Mamba3Config::dt_init_floor)
        .def_readwrite("seed", &Mamba3Config::seed);
    py::class_<Mamba3State>(m, "Mamba3State")
        .def(py::init<>())
        .def_readwrite("phase", &Mamba3State::phase)
        .def_readwrite("ssm", &Mamba3State::ssm)
        .def_readwrite("k", &Mamba3State::k)
        .def_readwrite("v", &Mamba3State::v);
    py::class_<Mamba3SessionSnapshot>(m, "Mamba3SessionSnapshot")
        .def(py::init<>())
        .def_readwrite("schema_version", &Mamba3SessionSnapshot::schema_version)
        .def_readwrite("configuration", &Mamba3SessionSnapshot::configuration)
        .def_readwrite("enabled", &Mamba3SessionSnapshot::enabled)
        .def_readwrite("state", &Mamba3SessionSnapshot::state);
    py::class_<Mamba3Backward>(m, "Mamba3Backward")
        .def_readonly("input", &Mamba3Backward::input)
        .def_readonly("initial_state", &Mamba3Backward::initial_state)
        .def_readonly("parameters", &Mamba3Backward::parameters)
        .def_readonly("owner", &Mamba3Backward::owner);
    py::class_<Mamba3Tape,std::shared_ptr<Mamba3Tape>>(m, "Mamba3Tape")
        .def("output", &Mamba3Tape::output)
        .def("snapshot_final_state", &Mamba3Tape::snapshot_final_state)
        .def("backward", &Mamba3Tape::backward, py::arg("gradient"),py::arg("final_seed")=Mamba3State{})
        .def("cancel", &Mamba3Tape::cancel)
        .def("consumed", &Mamba3Tape::consumed)
        .def("audit_status", &Mamba3Tape::audit_status)
        .def("workspace_bytes", &Mamba3Tape::workspace_bytes);
    py::class_<Mamba3Layer>(m, "Mamba3Layer")
        .def(py::init<int,const Mamba3Config&>(),py::arg("d_model"),py::arg("config")=Mamba3Config{})
        .def("forward", &Mamba3Layer::forward,py::arg("input"),py::arg("context")=nullptr,py::arg("valid_lengths")=std::vector<int>{})
        .def("backward", &Mamba3Layer::backward)
        .def("forward_owned", &Mamba3Layer::forward_owned,py::arg("input"),py::arg("initial")=Mamba3State{},py::arg("valid_lengths")=std::vector<int>{})
        .def("backward_owned", &Mamba3Layer::backward_owned,py::arg("tape"),py::arg("gradient"),py::arg("final_seed")=Mamba3State{})
        .def("publish", &Mamba3Layer::publish)
        .def("cancel_pending", &Mamba3Layer::cancel_pending)
        .def("reset", &Mamba3Layer::reset)
        .def("to", &Mamba3Layer::to)
        .def("parameters", &Mamba3Layer::parameters,py::return_value_policy::reference_internal)
        .def("set_training_mode", &Mamba3Layer::set_training_mode)
        .def("set_streaming_mode", &Mamba3Layer::set_streaming_mode)
        .def("snapshot_streaming_state", &Mamba3Layer::snapshot_streaming_state,py::arg("device_resident")=false)
        .def("restore_streaming_state", &Mamba3Layer::restore_streaming_state)
        .def("snapshot_streaming_state_batch", &Mamba3Layer::snapshot_streaming_state_batch,py::arg("device_resident")=false)
        .def("restore_streaming_state_batch", &Mamba3Layer::restore_streaming_state_batch)
        .def("configuration_identity", &Mamba3Layer::configuration_identity)
        .def("save_checkpoint", &Mamba3Layer::save_checkpoint,py::arg("path"),py::arg("include_session")=false)
        .def("load_checkpoint", &Mamba3Layer::load_checkpoint,py::arg("path"),py::arg("restore_session")=false);
    py::class_<Mamba2SSD>(m, "Mamba2SSD")
        .def(py::init<int, int, int>(),
             py::arg("d_model"),
             py::arg("d_state"),
             py::arg("n_heads"))
        .def(
            "forward",
            [](Mamba2SSD& layer, const Tensor& input, Context* context) {
                return layer.forward(input, context);
            },
            py::arg("input"),
            py::arg("context") = nullptr,
            py::call_guard<py::gil_scoped_release>())
        .def("to", &Mamba2SSD::to, py::arg("device"))
        .def("parameters", &Mamba2SSD::parameters,
             py::return_value_policy::reference_internal)
        .def("reset", &Mamba2SSD::reset);

    py::class_<MoERouter>(m, "MoERouter")
        .def(py::init<int, int, int>(),
             py::arg("d_model"),
             py::arg("num_experts") = 256,
             py::arg("top_k") = 8)
        .def(
            "forward",
            [](MoERouter& router, const Tensor& input) {
                return router.forward(input);
            },
            py::arg("input"),
            py::call_guard<py::gil_scoped_release>())
        .def("to", &MoERouter::to, py::arg("device"))
        .def("parameters", &MoERouter::parameters,
             py::return_value_policy::reference_internal);

    py::class_<TensorAuditStats>(m, "TensorAuditStats")
        .def(py::init<>())
        .def_readwrite("shape", &TensorAuditStats::shape)
        .def_readwrite("elements", &TensorAuditStats::elements)
        .def_readwrite("min", &TensorAuditStats::min)
        .def_readwrite("max", &TensorAuditStats::max)
        .def_readwrite("mean", &TensorAuditStats::mean)
        .def_readwrite("stddev", &TensorAuditStats::stddev)
        .def_readwrite("l2_norm", &TensorAuditStats::l2_norm)
        .def_readwrite("max_abs", &TensorAuditStats::max_abs)
        .def_readwrite("nan_count", &TensorAuditStats::nan_count)
        .def_readwrite("inf_count", &TensorAuditStats::inf_count)
        .def_readwrite("zero_count", &TensorAuditStats::zero_count)
        .def_readwrite("subnormal_count", &TensorAuditStats::subnormal_count)
        .def_readwrite("positive_count", &TensorAuditStats::positive_count)
        .def_readwrite("negative_count", &TensorAuditStats::negative_count)
        .def_readwrite("finite", &TensorAuditStats::finite);

    py::class_<RouterAuditStats>(m, "RouterAuditStats")
        .def(py::init<>())
        .def_readwrite("rows", &RouterAuditStats::rows)
        .def_readwrite("num_experts", &RouterAuditStats::num_experts)
        .def_readwrite("top_k", &RouterAuditStats::top_k)
        .def_readwrite("topk_counts", &RouterAuditStats::topk_counts)
        .def_readwrite("expert_loads", &RouterAuditStats::expert_loads)
        .def_readwrite("entropy", &RouterAuditStats::entropy);

    py::class_<LayerAuditRecord>(m, "LayerAuditRecord")
        .def(py::init<>())
        .def_readwrite("sequence", &LayerAuditRecord::sequence)
        .def_readwrite("run_id", &LayerAuditRecord::run_id)
        .def_readwrite("phase", &LayerAuditRecord::phase)
        .def_readwrite("pass_name", &LayerAuditRecord::pass)
        .def_readwrite("block_type", &LayerAuditRecord::block_type)
        .def_readwrite("tensor_role", &LayerAuditRecord::tensor_role)
        .def_readwrite("step", &LayerAuditRecord::step)
        .def_readwrite("layer_index", &LayerAuditRecord::layer_index)
        .def_readwrite("input", &LayerAuditRecord::input)
        .def_readwrite("output", &LayerAuditRecord::output)
        .def_readwrite("latency_ms", &LayerAuditRecord::latency_ms)
        .def_readwrite("grad_l2_norm", &LayerAuditRecord::grad_l2_norm)
        .def_readwrite("has_router", &LayerAuditRecord::has_router)
        .def_readwrite("router", &LayerAuditRecord::router);

    py::class_<TokenContextAuditRecord>(m, "TokenContextAuditRecord")
        .def(py::init<>())
        .def_readwrite("sequence", &TokenContextAuditRecord::sequence)
        .def_readwrite("run_id", &TokenContextAuditRecord::run_id)
        .def_readwrite("phase", &TokenContextAuditRecord::phase)
        .def_readwrite("step", &TokenContextAuditRecord::step)
        .def_readwrite("batch_size", &TokenContextAuditRecord::batch_size)
        .def_readwrite("prompt_tokens_total", &TokenContextAuditRecord::prompt_tokens_total)
        .def_readwrite("prompt_tokens_used", &TokenContextAuditRecord::prompt_tokens_used)
        .def_readwrite("context_limit", &TokenContextAuditRecord::context_limit)
        .def_readwrite("truncated", &TokenContextAuditRecord::truncated)
        .def_readwrite("token_ids_sample", &TokenContextAuditRecord::token_ids_sample);

    py::class_<TrainingStepAuditRecord>(m, "TrainingStepAuditRecord")
        .def(py::init<>())
        .def_readwrite("sequence", &TrainingStepAuditRecord::sequence)
        .def_readwrite("run_id", &TrainingStepAuditRecord::run_id)
        .def_readwrite("phase", &TrainingStepAuditRecord::phase)
        .def_readwrite("step", &TrainingStepAuditRecord::step)
        .def_readwrite("loss", &TrainingStepAuditRecord::loss)
        .def_readwrite("grad_l2_norm", &TrainingStepAuditRecord::grad_l2_norm)
        .def_readwrite("parameter_count", &TrainingStepAuditRecord::parameter_count);

    py::class_<ParameterAuditRecord>(m, "ParameterAuditRecord")
        .def(py::init<>())
        .def_readwrite("sequence", &ParameterAuditRecord::sequence)
        .def_readwrite("run_id", &ParameterAuditRecord::run_id)
        .def_readwrite("phase", &ParameterAuditRecord::phase)
        .def_readwrite("step", &ParameterAuditRecord::step)
        .def_readwrite("name", &ParameterAuditRecord::name)
        .def_readwrite("base_name", &ParameterAuditRecord::base_name)
        .def_readwrite("layer_index", &ParameterAuditRecord::layer_index)
        .def_readwrite("component", &ParameterAuditRecord::component)
        .def_readwrite("role", &ParameterAuditRecord::role)
        .def_readwrite("trainable", &ParameterAuditRecord::trainable)
        .def_readwrite("optimizer_applied", &ParameterAuditRecord::optimizer_applied)
        .def_readwrite("version_before", &ParameterAuditRecord::version_before)
        .def_readwrite("version_after", &ParameterAuditRecord::version_after)
        .def_readwrite("weight_before", &ParameterAuditRecord::weight_before)
        .def_readwrite("gradient", &ParameterAuditRecord::gradient)
        .def_readwrite("update", &ParameterAuditRecord::update)
        .def_readwrite("weight_after", &ParameterAuditRecord::weight_after)
        .def_readwrite("grad_to_weight_ratio", &ParameterAuditRecord::grad_to_weight_ratio)
        .def_readwrite("update_to_weight_ratio", &ParameterAuditRecord::update_to_weight_ratio)
        .def_readwrite("update_to_grad_ratio", &ParameterAuditRecord::update_to_grad_ratio)
        .def_readwrite("gradient_update_cosine", &ParameterAuditRecord::gradient_update_cosine)
        .def_readwrite("changed_elements", &ParameterAuditRecord::changed_elements)
        .def_readwrite("ternary_elements_before", &ParameterAuditRecord::ternary_elements_before)
        .def_readwrite("ternary_elements_after", &ParameterAuditRecord::ternary_elements_after)
        .def_readwrite("weight_sha256_before", &ParameterAuditRecord::weight_sha256_before)
        .def_readwrite("gradient_sha256", &ParameterAuditRecord::gradient_sha256)
        .def_readwrite("update_sha256", &ParameterAuditRecord::update_sha256)
        .def_readwrite("weight_sha256_after", &ParameterAuditRecord::weight_sha256_after);

    py::class_<HybridInteractionAuditRecord>(
        m, "HybridInteractionAuditRecord")
        .def(py::init<>())
        .def_readwrite("sequence", &HybridInteractionAuditRecord::sequence)
        .def_readwrite("run_id", &HybridInteractionAuditRecord::run_id)
        .def_readwrite("phase", &HybridInteractionAuditRecord::phase)
        .def_readwrite("pass_name", &HybridInteractionAuditRecord::pass)
        .def_readwrite("step", &HybridInteractionAuditRecord::step)
        .def_readwrite("layer_index", &HybridInteractionAuditRecord::layer_index)
        .def_readwrite("mamba_signal", &HybridInteractionAuditRecord::mamba_signal)
        .def_readwrite("attention_signal", &HybridInteractionAuditRecord::attention_signal)
        .def_readwrite("mamba_contribution", &HybridInteractionAuditRecord::mamba_contribution)
        .def_readwrite("attention_contribution", &HybridInteractionAuditRecord::attention_contribution)
        .def_readwrite("combined_contribution", &HybridInteractionAuditRecord::combined_contribution)
        .def_readwrite("signal_cosine", &HybridInteractionAuditRecord::signal_cosine)
        .def_readwrite("contribution_cosine", &HybridInteractionAuditRecord::contribution_cosine)
        .def_readwrite("attention_to_mamba_signal_ratio",
                       &HybridInteractionAuditRecord::attention_to_mamba_signal_ratio)
        .def_readwrite("attention_to_mamba_contribution_ratio",
                       &HybridInteractionAuditRecord::attention_to_mamba_contribution_ratio)
        .def_readwrite("cancellation_fraction",
                       &HybridInteractionAuditRecord::cancellation_fraction);

    py::class_<LayerAuditSummary>(m, "LayerAuditSummary")
        .def(py::init<>())
        .def_readwrite("phase", &LayerAuditSummary::phase)
        .def_readwrite("records", &LayerAuditSummary::records)
        .def_readwrite("forward_records", &LayerAuditSummary::forward_records)
        .def_readwrite("backward_records", &LayerAuditSummary::backward_records)
        .def_readwrite("router_records", &LayerAuditSummary::router_records)
        .def_readwrite("token_contexts", &LayerAuditSummary::token_contexts)
        .def_readwrite("training_steps", &LayerAuditSummary::training_steps)
        .def_readwrite("parameter_records", &LayerAuditSummary::parameter_records)
        .def_readwrite("changed_parameter_records", &LayerAuditSummary::changed_parameter_records)
        .def_readwrite("hybrid_interaction_records", &LayerAuditSummary::hybrid_interaction_records)
        .def_readwrite("parameter_nan", &LayerAuditSummary::parameter_nan)
        .def_readwrite("parameter_inf", &LayerAuditSummary::parameter_inf)
        .def_readwrite("total_nan", &LayerAuditSummary::total_nan)
        .def_readwrite("total_inf", &LayerAuditSummary::total_inf)
        .def_readwrite("max_latency_ms", &LayerAuditSummary::max_latency_ms)
        .def_readwrite("max_l2_norm", &LayerAuditSummary::max_l2_norm)
        .def_readwrite("layers_seen", &LayerAuditSummary::layers_seen)
        .def_readwrite("stored_records", &LayerAuditSummary::stored_records)
        .def_readwrite("dropped_records", &LayerAuditSummary::dropped_records)
        .def_readwrite("truncated_contexts", &LayerAuditSummary::truncated_contexts)
        .def_readwrite("router_entropy_count", &LayerAuditSummary::router_entropy_count)
        .def_readwrite("router_entropy_min", &LayerAuditSummary::router_entropy_min)
        .def_readwrite("router_entropy_max", &LayerAuditSummary::router_entropy_max)
        .def_readwrite("router_entropy_mean", &LayerAuditSummary::router_entropy_mean)
        .def_readwrite("router_num_experts_max", &LayerAuditSummary::router_num_experts_max)
        .def_readwrite("hybrid_signal_cosine_mean",
                       &LayerAuditSummary::hybrid_signal_cosine_mean)
        .def_readwrite("hybrid_contribution_cosine_mean",
                       &LayerAuditSummary::hybrid_contribution_cosine_mean)
        .def_readwrite("hybrid_attention_to_mamba_mean",
                       &LayerAuditSummary::hybrid_attention_to_mamba_mean)
        .def_readwrite("hybrid_cancellation_max",
                       &LayerAuditSummary::hybrid_cancellation_max)
        .def_readwrite(
            "hybrid_forward_interaction_records",
            &LayerAuditSummary::hybrid_forward_interaction_records)
        .def_readwrite(
            "hybrid_forward_signal_cosine_mean",
            &LayerAuditSummary::hybrid_forward_signal_cosine_mean)
        .def_readwrite(
            "hybrid_forward_contribution_cosine_mean",
            &LayerAuditSummary::hybrid_forward_contribution_cosine_mean)
        .def_readwrite(
            "hybrid_forward_attention_to_mamba_mean",
            &LayerAuditSummary::hybrid_forward_attention_to_mamba_mean)
        .def_readwrite(
            "hybrid_forward_cancellation_max",
            &LayerAuditSummary::hybrid_forward_cancellation_max)
        .def_readwrite(
            "hybrid_backward_interaction_records",
            &LayerAuditSummary::hybrid_backward_interaction_records)
        .def_readwrite(
            "hybrid_backward_signal_cosine_mean",
            &LayerAuditSummary::hybrid_backward_signal_cosine_mean)
        .def_readwrite(
            "hybrid_backward_contribution_cosine_mean",
            &LayerAuditSummary::hybrid_backward_contribution_cosine_mean)
        .def_readwrite(
            "hybrid_backward_attention_to_mamba_mean",
            &LayerAuditSummary::hybrid_backward_attention_to_mamba_mean)
        .def_readwrite(
            "hybrid_backward_cancellation_max",
            &LayerAuditSummary::hybrid_backward_cancellation_max)
        .def("healthy", &LayerAuditSummary::healthy);

    py::class_<LayerAuditCollector>(m, "LayerAuditCollector")
        .def(py::init<>())
        .def("set_enabled", &LayerAuditCollector::set_enabled, py::arg("enabled"))
        .def("enabled", &LayerAuditCollector::enabled)
        .def("reset", &LayerAuditCollector::reset)
        .def("begin_run", &LayerAuditCollector::begin_run, py::arg("run_id"))
        .def("set_phase", &LayerAuditCollector::set_phase, py::arg("phase"))
        .def("set_step", &LayerAuditCollector::set_step, py::arg("step"))
        .def("set_storage_policy",
             &LayerAuditCollector::set_storage_policy,
             py::arg("summary_only"),
             py::arg("record_sample_rate"),
             py::arg("max_records_per_phase"),
             py::arg("store_token_contexts"))
        .def("set_parameter_audit_policy",
             &LayerAuditCollector::set_parameter_audit_policy,
             py::arg("enabled"),
             py::arg("step_sample_rate") = 1,
             py::arg("max_records") = 0,
             py::arg("max_snapshot_bytes") =
                 static_cast<size_t>(1024) * 1024 * 1024)
        .def("summary_only", &LayerAuditCollector::summary_only)
        .def("record_sample_rate", &LayerAuditCollector::record_sample_rate)
        .def("max_records_per_phase", &LayerAuditCollector::max_records_per_phase)
        .def("store_token_contexts", &LayerAuditCollector::store_token_contexts)
        .def("parameter_audit_enabled", &LayerAuditCollector::parameter_audit_enabled)
        .def("parameter_step_sample_rate", &LayerAuditCollector::parameter_step_sample_rate)
        .def("max_parameter_records", &LayerAuditCollector::max_parameter_records)
        .def("max_parameter_snapshot_bytes", &LayerAuditCollector::max_parameter_snapshot_bytes)
        .def("records", &LayerAuditCollector::records)
        .def("token_contexts", &LayerAuditCollector::token_contexts)
        .def("training_steps", &LayerAuditCollector::training_steps)
        .def("parameter_records", &LayerAuditCollector::parameter_records)
        .def("hybrid_interaction_records",
             &LayerAuditCollector::hybrid_interaction_records)
        .def("summarize_phase", &LayerAuditCollector::summarize_phase, py::arg("phase"))
        .def("compare_phase_health",
             [](const LayerAuditCollector& collector,
                const std::string& lhs_phase,
                const std::string& rhs_phase) {
                 std::string reason;
                 const bool healthy = collector.compare_phase_health(lhs_phase, rhs_phase, &reason);
                 py::dict result;
                 result["healthy"] = healthy;
                 result["reason"] = reason;
                 return result;
             },
             py::arg("lhs_phase"),
             py::arg("rhs_phase"))
        .def("write_json", &LayerAuditCollector::write_json, py::arg("path"));

    py::class_<ReasoningProposal>(m, "ReasoningProposal")
        .def(py::init<>())
        .def_readwrite("state", &ReasoningProposal::state)
        .def_readwrite("prior", &ReasoningProposal::prior);
    py::class_<ReasoningVerification>(m, "ReasoningVerification")
        .def(py::init<>())
        .def_readwrite("score", &ReasoningVerification::score)
        .def_readwrite("evidence", &ReasoningVerification::evidence);
    py::class_<ReasoningPolicy>(m, "ReasoningPolicy")
        .def(py::init<>())
        .def_readwrite("policy_id", &ReasoningPolicy::policy_id)
        .def_readwrite("verifier_id", &ReasoningPolicy::verifier_id)
        .def_readwrite("propose", &ReasoningPolicy::propose)
        .def_readwrite("verify", &ReasoningPolicy::verify);
    py::class_<VerifiedReasoningReport>(m, "VerifiedReasoningReport")
        .def_readonly("policy_id", &VerifiedReasoningReport::policy_id)
        .def_readonly("verifier_id", &VerifiedReasoningReport::verifier_id)
        .def_readonly("evidence", &VerifiedReasoningReport::evidence)
        .def_readonly("evaluated_states", &VerifiedReasoningReport::evaluated_states)
        .def_readonly("proposal_calls", &VerifiedReasoningReport::proposal_calls)
        .def_readonly("baseline_score", &VerifiedReasoningReport::baseline_score)
        .def_readonly("best_score", &VerifiedReasoningReport::best_score)
        .def_readonly("elapsed_ms", &VerifiedReasoningReport::elapsed_ms);
    m.def("exact_token_verifier", &exact_token_verifier,
          py::arg("expected"), py::arg("decode"));

    py::class_<JambaModel>(m, "JambaModel")
        .def(py::init<int, int, int, Device>(),
             py::arg("num_layers"),
             py::arg("d_model"),
             py::arg("vocab_size") = 128000,
             py::arg("device") = Device::CPU)
        .def(py::init<const ModelConfig&, Device>(),
             py::arg("config"),
             py::arg("device") = Device::CPU)
        // GIL released around the pure-C++ compute body (pybind marshals
        // args/return with the GIL held).  Safe because the only Python callback
        // path is Trainer::train_loop's callback (left GIL-holding below); the
        // audit collector (LayerAuditCollector) is a concrete C++ type, so
        // forward/backward never re-enter Python mid-call.
        .def("forward", py::overload_cast<const Tensor&, Context*>(&JambaModel::forward),
             py::arg("x"), py::arg("ctx") = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_ids_decode_graph", &JambaModel::forward_ids_decode_graph,
             py::arg("token"),
             "HIP/CUDA graph decode step (opt-in NSOS_GPU_GRAPH_DECODE=1). "
             "Returns the step logits, or an empty "
             "Tensor when the graph path is unavailable — fall back to "
             "forward_ids([token]). Captured logits alias reusable graph storage; "
             "clone before retaining them across the next replay.")
        .def("decode_graph_active", &JambaModel::decode_graph_active)
        .def("decode_graph_status", &JambaModel::decode_graph_status)
        .def("forward_ids", &JambaModel::forward_ids, py::arg("ids"), py::arg("ctx") = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_ids_batch", &JambaModel::forward_ids_batch, py::arg("batch_ids"), py::arg("ctx") = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_trunk", &JambaModel::forward_trunk, py::arg("ids"), py::arg("ctx") = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_embedding", &JambaModel::forward_embedding, py::arg("x"), py::arg("ctx") = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("reason", &JambaModel::reason, py::arg("x"), py::arg("num_simulations") = 100,
             py::call_guard<py::gil_scoped_release>())
        .def("set_reasoning_policy", &JambaModel::set_reasoning_policy,
             py::call_guard<py::gil_scoped_release>())
        .def("clear_reasoning_policy", &JambaModel::clear_reasoning_policy,
             py::call_guard<py::gil_scoped_release>())
        .def("has_reasoning_policy", &JambaModel::has_reasoning_policy,
             py::call_guard<py::gil_scoped_release>())
        .def("last_reasoning_report", &JambaModel::last_reasoning_report,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_thought", &JambaModel::forward_thought,
             py::call_guard<py::gil_scoped_release>())
        .def("run_reasoning_loop", py::overload_cast<const Tensor&, int>(&JambaModel::run_reasoning_loop),
             py::call_guard<py::gil_scoped_release>())
        .def("backward_external", &JambaModel::backward_external,
             py::call_guard<py::gil_scoped_release>())
        .def("backward_embedding", &JambaModel::backward_embedding,
             py::call_guard<py::gil_scoped_release>())
        .def("backward", &JambaModel::backward,
             py::call_guard<py::gil_scoped_release>())
        .def("reset_session", &JambaModel::reset_session)
        .def("runtime_telemetry", [](const JambaModel& model) {
            const auto t = model.runtime_telemetry();
            py::dict d;
            d["mamba_fast_path_hits"] = t.mamba_fast_path_hits;
            d["mamba_fast_path_fallbacks"] = t.mamba_fast_path_fallbacks;
            d["faithful_forward_gpu_calls"] =
                t.faithful_forward_gpu_calls;
            d["faithful_forward_host_fallbacks"] =
                t.faithful_forward_host_fallbacks;
            d["faithful_backward_gpu_calls"] =
                t.faithful_backward_gpu_calls;
            d["faithful_backward_host_fallbacks"] =
                t.faithful_backward_host_fallbacks;
            d["faithful_streaming_gpu_calls"] =
                t.faithful_streaming_gpu_calls;
            d["faithful_streaming_host_fallbacks"] =
                t.faithful_streaming_host_fallbacks;
            d["stream_priming_gpu_calls"] =
                t.stream_priming_gpu_calls;
            d["stream_priming_host_fallbacks"] =
                t.stream_priming_host_fallbacks;
            d["faithful_recompute_forwards"] =
                t.faithful_recompute_forwards;
            d["faithful_selective_history_recomputes"] =
                t.faithful_selective_history_recomputes;
            d["faithful_full_block_recompute_forwards"] =
                t.faithful_full_block_recompute_forwards;
            d["faithful_warp_aggregated_backward_calls"] =
                t.faithful_warp_aggregated_backward_calls;
            d["faithful_deterministic_backward_calls"] =
                t.faithful_deterministic_backward_calls;
            d["faithful_scalar_atomic_backward_calls"] =
                t.faithful_scalar_atomic_backward_calls;
            d["faithful_reduced_conv_backward_calls"] =
                t.faithful_reduced_conv_backward_calls;
            d["faithful_generic_atomic_conv_backward_calls"] =
                t.faithful_generic_atomic_conv_backward_calls;
            d["faithful_peak_state_history_bytes"] =
                t.faithful_peak_state_history_bytes;
            d["faithful_grouped_projection_forward_calls"] =
                t.faithful_grouped_projection_forward_calls;
            d["faithful_grouped_projection_backward_calls"] =
                t.faithful_grouped_projection_backward_calls;
            d["faithful_grouped_projection_cache_rebuilds"] =
                t.faithful_grouped_projection_cache_rebuilds;
            d["faithful_grouped_projection_full_forward_calls"] =
                t.faithful_grouped_projection_full_forward_calls;
            d["faithful_grouped_projection_sensitive_forward_calls"] =
                t.faithful_grouped_projection_sensitive_forward_calls;
            d["faithful_grouped_projection_full_backward_calls"] =
                t.faithful_grouped_projection_full_backward_calls;
            d["faithful_grouped_projection_sensitive_backward_calls"] =
                t.faithful_grouped_projection_sensitive_backward_calls;
            d["mamba_last_fallback_reason"] = t.mamba_last_fallback_reason;
            py::list layers;
            for (const auto& layer : t.mamba_layers) {
                py::dict item;
                item["layer_index"] = layer.layer_index;
                item["fast_path_hits"] = layer.fast_path_hits;
                item["fast_path_fallbacks"] =
                    layer.fast_path_fallbacks;
                item["faithful_forward_gpu_calls"] =
                    layer.faithful_forward_gpu_calls;
                item["faithful_forward_host_fallbacks"] =
                    layer.faithful_forward_host_fallbacks;
                item["faithful_backward_gpu_calls"] =
                    layer.faithful_backward_gpu_calls;
                item["faithful_backward_host_fallbacks"] =
                    layer.faithful_backward_host_fallbacks;
                item["faithful_streaming_gpu_calls"] =
                    layer.faithful_streaming_gpu_calls;
                item["faithful_streaming_host_fallbacks"] =
                    layer.faithful_streaming_host_fallbacks;
                item["stream_priming_gpu_calls"] =
                    layer.stream_priming_gpu_calls;
                item["stream_priming_host_fallbacks"] =
                    layer.stream_priming_host_fallbacks;
                item["faithful_recompute_forwards"] =
                    layer.faithful_recompute_forwards;
                item["faithful_selective_history_recomputes"] =
                    layer.faithful_selective_history_recomputes;
                item["faithful_full_block_recompute_forwards"] =
                    layer.faithful_full_block_recompute_forwards;
                item["faithful_warp_aggregated_backward_calls"] =
                    layer.faithful_warp_aggregated_backward_calls;
                item["faithful_deterministic_backward_calls"] =
                    layer.faithful_deterministic_backward_calls;
                item["faithful_scalar_atomic_backward_calls"] =
                    layer.faithful_scalar_atomic_backward_calls;
                item["faithful_reduced_conv_backward_calls"] =
                    layer.faithful_reduced_conv_backward_calls;
                item["faithful_generic_atomic_conv_backward_calls"] =
                    layer.faithful_generic_atomic_conv_backward_calls;
                item["faithful_peak_state_history_bytes"] =
                    layer.faithful_peak_state_history_bytes;
                item["faithful_grouped_projection_forward_calls"] =
                    layer.faithful_grouped_projection_forward_calls;
                item["faithful_grouped_projection_backward_calls"] =
                    layer.faithful_grouped_projection_backward_calls;
                item["faithful_grouped_projection_cache_rebuilds"] =
                    layer.faithful_grouped_projection_cache_rebuilds;
                item["faithful_grouped_projection_full_forward_calls"] =
                    layer.faithful_grouped_projection_full_forward_calls;
                item["faithful_grouped_projection_sensitive_forward_calls"] =
                    layer.faithful_grouped_projection_sensitive_forward_calls;
                item["faithful_grouped_projection_full_backward_calls"] =
                    layer.faithful_grouped_projection_full_backward_calls;
                item["faithful_grouped_projection_sensitive_backward_calls"] =
                    layer.faithful_grouped_projection_sensitive_backward_calls;
                item["last_fallback_reason"] =
                    layer.last_fallback_reason;
                layers.append(std::move(item));
            }
            d["mamba_layers"] = std::move(layers);
            py::list mamba3_layers;
            for(const auto& layer:t.mamba3_layers) {
                py::dict item;
                item["implementation"] = mamba3_block::identity;
                item["cpu_forward"] = layer.cpu_forward;
                item["gpu_forward"] = layer.gpu_forward;
                item["projection_wmma_gemms"] = layer.projection_wmma_gemms;
                item["gpu_reference_forward"] = layer.gpu_reference_forward;
                item["gpu_reference_backward"] = layer.gpu_reference_backward;
                item["gpu_parallel_forward"] = layer.gpu_parallel_forward;
                item["gpu_parallel_backward"] = layer.gpu_parallel_backward;
                item["gpu_flash_forward"] = layer.gpu_flash_forward;
                item["gpu_flash_backward"] = layer.gpu_flash_backward;
                item["cpu_backward"] = layer.cpu_backward;
                item["gpu_backward"] = layer.gpu_backward;
                item["cancelled"] = layer.cancelled;
                item["peak_workspace_bytes"] = layer.peak_workspace_bytes;
                mamba3_layers.append(std::move(item));
            }
            d["mamba3_layers"] = std::move(mamba3_layers);
            return d;
        })
        .def("set_hamiltonian_mode", &JambaModel::set_hamiltonian_mode)
        .def("session_adapt", &JambaModel::session_adapt,
             py::call_guard<py::gil_scoped_release>())
        .def("run_simd_inference", &JambaModel::run_simd_inference,
             py::call_guard<py::gil_scoped_release>())
        // Disk I/O (no Python callback) -> release the GIL so a multi-MB/GB
        // pack save/load doesn't block other Python threads.
        .def("save", &JambaModel::save, py::call_guard<py::gil_scoped_release>())
        .def("load", &JambaModel::load, py::arg("path"), py::arg("strict") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("set_reference_path", &JambaModel::set_reference_path, py::arg("enabled"))
        // Enable SSA on every attention layer (mirror of InferenceEngine's
        // binding so a bare JambaModel can be configured symmetrically).
        .def("set_sparse_attention", &JambaModel::set_sparse_attention,
             py::arg("enabled"), py::arg("block_size") = 64, py::arg("top_k_blocks") = 8,
             py::arg("local_blocks") = 1, py::arg("sink_blocks") = 1)
        .def("release_full_precision_linear_weights",
             &JambaModel::release_full_precision_linear_weights)
        .def("save_edge_linear_pack", &JambaModel::save_edge_linear_pack, py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_edge_linear_pack",
             &JambaModel::load_edge_linear_pack,
             py::arg("path"),
             py::arg("release_full_precision") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("supports_streaming_inference", &JambaModel::supports_streaming_inference)
        .def("set_streaming_inference", &JambaModel::set_streaming_inference, py::arg("enabled"))
        .def("set_training_mode", &JambaModel::set_training_mode, py::arg("enabled"))
        .def("training_mode", &JambaModel::training_mode)
        .def("set_audit_collector",
             &JambaModel::set_audit_collector,
             py::arg("collector"),
             py::keep_alive<1, 2>())
        .def("model_config", &JambaModel::model_config, py::return_value_policy::reference_internal)
        .def("to", &JambaModel::to)
        .def("ternary_weight_parameters", [](JambaModel& model) {
            (void)model.parameters();  // validate the frozen canonical registry
            std::vector<Parameter*> result;
            for (BitLinear* layer : model.collect_bitlinear_layers()) {
                if (layer && !layer->quantization_sensitive() &&
                    layer->has_full_precision_weight()) {
                    result.push_back(&layer->weight);
                }
            }
            return result;
        }, py::return_value_policy::reference_internal)
        .def("parameter_aliases", &JambaModel::parameter_aliases)
        .def("parameters", &JambaModel::parameters, py::return_value_policy::reference_internal);

    py::class_<TrainPhaseScheduler>(m, "TrainPhaseScheduler")
        .def(py::init<>())
        .def_readwrite("progressive_qat_enabled", &TrainPhaseScheduler::progressive_qat_enabled)
        .def_readwrite("semantic_warmup_steps", &TrainPhaseScheduler::semantic_warmup_steps)
        .def_readwrite("qat_start_step", &TrainPhaseScheduler::qat_start_step)
        .def_readwrite("quantized_precision_bits", &TrainPhaseScheduler::quantized_precision_bits)
        .def_readwrite("ternary_regularization", &TrainPhaseScheduler::ternary_regularization)
        .def_readwrite("auxiliary_stack_enabled", &TrainPhaseScheduler::auxiliary_stack_enabled)
        .def_readwrite("auxiliary_session_adapt_enabled",
                       &TrainPhaseScheduler::auxiliary_session_adapt_enabled)
        .def_readwrite("auxiliary_reasoning_enabled",
                       &TrainPhaseScheduler::auxiliary_reasoning_enabled)
        .def_readwrite("auxiliary_memory_enabled", &TrainPhaseScheduler::auxiliary_memory_enabled)
        .def_readwrite("auxiliary_reasoning_iterations",
                       &TrainPhaseScheduler::auxiliary_reasoning_iterations)
        .def_readwrite("auxiliary_reasoning_simulations",
                       &TrainPhaseScheduler::auxiliary_reasoning_simulations)
        .def_readwrite("auxiliary_memory_blend", &TrainPhaseScheduler::auxiliary_memory_blend)
        .def_readwrite("auxiliary_every_steps", &TrainPhaseScheduler::auxiliary_every_steps)
        .def_readwrite("auxiliary_prompt_max_tokens",
                       &TrainPhaseScheduler::auxiliary_prompt_max_tokens)
        .def_readwrite("auxiliary_answer_max_tokens",
                       &TrainPhaseScheduler::auxiliary_answer_max_tokens)
        .def_readwrite("auxiliary_memory_scope",
                       &TrainPhaseScheduler::auxiliary_memory_scope);

    py::class_<AuxiliaryStackStats>(m, "AuxiliaryStackStats")
        .def(py::init<>())
        .def_readwrite("bucket_count", &AuxiliaryStackStats::bucket_count)
        .def_readwrite("due_count", &AuxiliaryStackStats::due_count)
        .def_readwrite("applied_count", &AuxiliaryStackStats::applied_count)
        .def_readwrite("reasoning_count", &AuxiliaryStackStats::reasoning_count)
        .def_readwrite("memory_count", &AuxiliaryStackStats::memory_count)
        .def_readwrite("session_adapt_count", &AuxiliaryStackStats::session_adapt_count)
        .def_readwrite("sample_count", &AuxiliaryStackStats::sample_count)
        .def_readwrite("prompt_tokens", &AuxiliaryStackStats::prompt_tokens)
        .def_readwrite("answer_tokens", &AuxiliaryStackStats::answer_tokens)
        .def_readwrite("prompt_state_norm", &AuxiliaryStackStats::prompt_state_norm)
        .def_readwrite("target_state_norm", &AuxiliaryStackStats::target_state_norm)
        .def_readwrite("reason_delta_norm", &AuxiliaryStackStats::reason_delta_norm)
        .def_readwrite("reason_cosine", &AuxiliaryStackStats::reason_cosine)
        .def_readwrite("memory_delta_norm", &AuxiliaryStackStats::memory_delta_norm)
        .def_readwrite("memory_cosine", &AuxiliaryStackStats::memory_cosine)
        .def_readwrite("final_target_delta_norm", &AuxiliaryStackStats::final_target_delta_norm);

    py::class_<TrainingObjectiveStats>(m, "TrainingObjectiveStats")
        .def(py::init<>())
        .def_readwrite("supervised_cross_entropy",
                       &TrainingObjectiveStats::supervised_cross_entropy)
        .def_readwrite("repetition_unlikelihood",
                       &TrainingObjectiveStats::repetition_unlikelihood)
        .def_readwrite("logit_l2", &TrainingObjectiveStats::logit_l2)
        .def_readwrite("sparse_selector", &TrainingObjectiveStats::sparse_selector)
        .def_readwrite("qat_regularization",
                       &TrainingObjectiveStats::qat_regularization)
        .def_readwrite("moe_auxiliary", &TrainingObjectiveStats::moe_auxiliary)
        .def_readwrite("criticality_regularization",
                       &TrainingObjectiveStats::criticality_regularization)
        .def_readwrite("total", &TrainingObjectiveStats::total);

    py::class_<TrainingStepTelemetry>(m, "TrainingStepTelemetry")
        .def(py::init<>())
        .def_readonly("enabled", &TrainingStepTelemetry::enabled)
        .def_readonly("global_step", &TrainingStepTelemetry::global_step)
        .def_readonly("bucket_count", &TrainingStepTelemetry::bucket_count)
        .def_readonly("wall_ms", &TrainingStepTelemetry::wall_ms)
        .def_readonly("preparation_ms", &TrainingStepTelemetry::preparation_ms)
        .def_readonly("inter_bucket_ms", &TrainingStepTelemetry::inter_bucket_ms)
        .def_readonly("forward_ms", &TrainingStepTelemetry::forward_ms)
        .def_readonly("loss_ms", &TrainingStepTelemetry::loss_ms)
        .def_readonly("backward_ms", &TrainingStepTelemetry::backward_ms)
        .def_readonly("optimizer_ms", &TrainingStepTelemetry::optimizer_ms)
        .def_readonly("unaccounted_ms", &TrainingStepTelemetry::unaccounted_ms);

    py::class_<TrainingCheckpointSnapshot,
               std::shared_ptr<TrainingCheckpointSnapshot>>(
        m, "TrainingCheckpointSnapshot")
        .def_property_readonly(
            "global_step", &TrainingCheckpointSnapshot::global_step)
        .def("write", &TrainingCheckpointSnapshot::write,
             py::arg("model_path"), py::arg("state_path"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<JambaModel*, float>(), py::arg("model"),
             py::arg("learning_rate") = 0.001f,
             py::keep_alive<1, 2>())
        .def_property("learning_rate",
                      trainer_getter(&Trainer::learning_rate),
                      trainer_setter(&Trainer::learning_rate))
        .def_property("beta1", trainer_getter(&Trainer::beta1),
                      trainer_setter(&Trainer::beta1))
        .def_property("beta2", trainer_getter(&Trainer::beta2),
                      trainer_setter(&Trainer::beta2))
        .def_property("eps", trainer_getter(&Trainer::eps),
                      trainer_setter(&Trainer::eps))
        .def_property("weight_decay",
                      trainer_getter(&Trainer::weight_decay),
                      trainer_setter(&Trainer::weight_decay))
        .def_property("max_grad_norm",
                      trainer_getter(&Trainer::max_grad_norm),
                      trainer_setter(&Trainer::max_grad_norm))
        .def_property("min_learning_rate_scale",
                      trainer_getter(&Trainer::min_learning_rate_scale),
                      trainer_setter(&Trainer::min_learning_rate_scale))
        .def_property("first_token_loss_scale",
                      trainer_getter(&Trainer::first_token_loss_scale),
                      trainer_setter(&Trainer::first_token_loss_scale))
        .def_property("eos_loss_scale",
                      trainer_getter(&Trainer::eos_loss_scale),
                      trainer_setter(&Trainer::eos_loss_scale))
        .def_property(
            "repetition_unlikelihood_scale",
            trainer_getter(&Trainer::repetition_unlikelihood_scale),
            trainer_setter(&Trainer::repetition_unlikelihood_scale))
        .def_property("moe_aux_loss_scale",
                      trainer_getter(&Trainer::moe_aux_loss_scale),
                      trainer_setter(&Trainer::moe_aux_loss_scale))
        .def_property("logit_l2_beta",
                      trainer_getter(&Trainer::logit_l2_beta),
                      trainer_setter(&Trainer::logit_l2_beta))
        .def_property("pantheon_vib_beta",
                      trainer_getter(&Trainer::pantheon_vib_beta),
                      trainer_setter(&Trainer::pantheon_vib_beta))
        .def_property("warmup_steps",
                      trainer_getter(&Trainer::warmup_steps),
                      trainer_setter(&Trainer::warmup_steps))
        .def_property("global_step_count",
                      trainer_getter(&Trainer::global_step_count),
                      trainer_setter(&Trainer::global_step_count))
        .def_property("total_training_steps",
                      trainer_getter(&Trainer::total_training_steps),
                      trainer_setter(&Trainer::total_training_steps))
        // Unidade do scheduler: "steps" (legado) ou "tokens".  Em "tokens" o
        // LR anda por tokens commitados, de modo que braços com acumulação
        // diferente recebem a mesma quantidade de dados antes do LR de pico.
        .def_property(
            "scheduler_unit",
            [](const Trainer& t) {
                return t.scheduler_unit == Trainer::SchedulerUnit::Tokens
                           ? "tokens"
                           : "steps";
            },
            [](Trainer& t, const std::string& unit) {
                if (unit == "tokens") {
                    if (!t.token_counters_complete)
                        throw std::invalid_argument("Token scheduler requires complete historical token counters");
                    t.scheduler_unit = Trainer::SchedulerUnit::Tokens;
                } else if (unit == "steps") {
                    t.scheduler_unit = Trainer::SchedulerUnit::Steps;
                } else {
                    throw std::invalid_argument(
                        "scheduler_unit must be 'steps' or 'tokens'");
                }
            })
        // Contrato declarado de acumulação: precisa ser fixado ANTES do
        // treino, porque alimenta a identidade de runtime.  commit exige que
        // o A pedido coincida com este valor.
        .def_property("gradient_accumulation_steps",
                      trainer_getter(&Trainer::gradient_accumulation_steps),
                      trainer_setter(&Trainer::gradient_accumulation_steps))
        .def_property("warmup_tokens",
                      trainer_getter(&Trainer::warmup_tokens),
                      trainer_setter(&Trainer::warmup_tokens))
        .def_property("training_tokens",
                      trainer_getter(&Trainer::training_tokens),
                      trainer_setter(&Trainer::training_tokens))
        .def_property("decay_tokens",
                      trainer_getter(&Trainer::decay_tokens),
                      trainer_setter(&Trainer::decay_tokens))
        // Telemetria (tudo que passou pelo forward) versus estado de treino
        // (apenas grupos que commitaram).  O scheduler anda pelo segundo.
        .def_property_readonly(
            "tokens_processed",
            [](const Trainer& t) { return t.tokens_processed; })
        .def_property_readonly(
            "tokens_committed",
            [](const Trainer& t) { return t.tokens_committed; })
        .def_property_readonly(
            "token_counters_complete",
            [](const Trainer& t) { return t.token_counters_complete; })
        .def_property_readonly(
            "pending_accumulation_microbatches",
            [](const Trainer& t) {
                return t.pending_accumulation_microbatches;
            })
        // Regime efetivo do último commit.  Se a maioria dos updates estiver
        // sendo clipada, quem governa o passo é o clipper, não o AdamW.
        .def_property_readonly(
            "last_grad_norm_pre_clip",
            [](const Trainer& t) { return t.last_grad_norm_pre_clip; })
        .def_property_readonly(
            "last_grad_norm_post_clip",
            [](const Trainer& t) { return t.last_grad_norm_post_clip; })
        .def_property_readonly(
            "last_update_was_clipped",
            [](const Trainer& t) { return t.last_update_was_clipped; })
        .def_property_readonly(
            "last_accumulation_steps",
            [](const Trainer& t) { return t.last_accumulation_steps; })
        .def_property("eos_token_id",
                      trainer_getter(&Trainer::eos_token_id),
                      trainer_setter(&Trainer::eos_token_id))
        // 4-bit optimizer states (Li et al. 2023): set to 4 to cut optimizer
        // memory ~8x on the CPU training path (default 32 = FP32 m/v).
        .def_property("optimizer_state_bits",
                      trainer_getter(&Trainer::optimizer_state_bits),
                      trainer_setter(&Trainer::optimizer_state_bits))
        .def_property(
            "phase_scheduler",
            trainer_getter(&Trainer::phase_scheduler),
            [](Trainer& trainer, const TrainPhaseScheduler& scheduler) {
                trainer.configure_progressive_qat(scheduler);
            })
        .def_property_readonly(
            "last_auxiliary_stats",
            trainer_getter(&Trainer::last_auxiliary_stats))
        .def_property_readonly(
            "last_objective_stats",
            trainer_getter(&Trainer::last_objective_stats))
        .def_property_readonly(
            "last_step_telemetry",
            trainer_getter(&Trainer::last_step_telemetry))
        .def("configure_progressive_qat", &Trainer::configure_progressive_qat, py::arg("scheduler"))
        .def("progressive_qat_active", &Trainer::progressive_qat_active)
        .def("save_training_state", &Trainer::save_training_state,
             py::arg("state_path"), py::arg("model_path"))
        .def("load_training_state", &Trainer::load_training_state,
             py::arg("state_path"), py::arg("model_path"),
             py::arg("allow_legacy_runtime_identity") = false,
             py::arg("allow_legacy_progress_state") = false)
        .def("execution_identity", [](const Trainer& trainer) {
            py::dict result;
            for (const auto& [key, value] :
                 trainer.execution_identity_fields()) {
                result[py::str(key)] = py::str(value);
            }
            return result;
        })
        .def("execution_identity_digest",
             &Trainer::execution_identity_digest)
        .def("validate_execution_identity",
             &Trainer::validate_execution_identity)
        .def("capture_checkpoint_snapshot",
             &Trainer::capture_checkpoint_snapshot,
             py::call_guard<py::gil_scoped_release>())
        // GIL released around native training. train_loop reacquires it only
        // at completed-step safe points for signal polling/Python callbacks.
        .def("train_step", &Trainer::train_step,
             py::call_guard<py::gil_scoped_release>())
        // Gradients-only (no optimizer step) — for the criticality instrument's
        // per-layer gradient SNR / backward-Lyapunov probe (OXTA-CRIT §6).
        .def("accumulate_gradients", &Trainer::accumulate_gradients,
             py::arg("tokens"), py::arg("targets"),
             py::call_guard<py::gil_scoped_release>())
        // Acumulação explícita de gradiente.  O contrato é
        //   for _ in range(A): accumulate_microbatch(...)
        //   commit_optimizer_step(A)
        // com abort_gradient_accumulation() em qualquer caminho de exceção.
        // commit divide por A ANTES do clipping; sem isso o gradiente ficaria
        // A vezes maior e o clipper passaria a governar o passo.
        .def("accumulate_microbatch", &Trainer::accumulate_microbatch,
             py::arg("tokens"), py::arg("targets"),
             py::call_guard<py::gil_scoped_release>())
        .def("commit_optimizer_step", &Trainer::commit_optimizer_step,
             py::arg("accumulation_steps"),
             py::call_guard<py::gil_scoped_release>())
        .def("abort_gradient_accumulation",
             &Trainer::abort_gradient_accumulation,
             py::call_guard<py::gil_scoped_release>())
        // §6 closed loop: per-layer lr multiplier (DEPTH axis from the Python
        // SNR instrument; SPACE axis is the in-loop C++ NSOS_CRIT_LR controller).
        .def("set_lr_scale_by_name", &Trainer::set_lr_scale_by_name,
             py::arg("name"), py::arg("scale"))
        .def("clear_lr_scales", &Trainer::clear_lr_scales)
        .def("request_cancellation",
             &Trainer::request_cancellation)
        .def("clear_cancellation",
             &Trainer::clear_cancellation)
        .def("cancellation_requested",
             &Trainer::cancellation_requested)
        .def("optimizer_state_poisoned",
             &Trainer::optimizer_state_poisoned)
        .def("train_supervised",
             &Trainer::train_supervised,
             py::arg("prompt_tokens"),
             py::arg("answer_tokens"),
             py::call_guard<py::gil_scoped_release>())
        .def("train_supervised_batch",
             &Trainer::train_supervised_batch,
             py::arg("prompt_batch"),
             py::arg("answer_batch"),
             py::call_guard<py::gil_scoped_release>())
        .def("train_loop",
             [](Trainer& trainer,
                const std::vector<int>& tokens,
                int epochs,
                int batch_size,
                int seq_len,
                py::object callback,
                int max_steps,
                int start_step) {
                 py::function python_callback;
                 if (!callback.is_none()) {
                     python_callback =
                         callback.cast<py::function>();
                 }
                 std::function<void(int, float)> safe_point =
                     [&python_callback](int step, float loss) {
                         py::gil_scoped_acquire acquire;
                         if (PyErr_CheckSignals() != 0) {
                             throw py::error_already_set();
                         }
                         if (python_callback) {
                             python_callback(step, loss);
                         }
                     };
                 {
                     // Other Python threads can now request cooperative
                     // cancellation even while a long native step owns the
                     // trainer transaction. The safe-point callback
                     // reacquires the GIL once per completed step.
                     py::gil_scoped_release release;
                     trainer.train_loop(
                         tokens, epochs, batch_size, seq_len,
                         std::move(safe_point), max_steps,
                         start_step);
                 }
             },
             py::arg("tokens"),
             py::arg("epochs"),
             py::arg("batch_size"),
             py::arg("seq_len"),
             py::arg("callback") = py::none(),
             py::arg("max_steps") = -1,
             py::arg("start_step") = 0);

    py::class_<ModelLoadOptions>(m, "ModelLoadOptions")
        .def(py::init<>())
        .def_readwrite("use_cuda", &ModelLoadOptions::use_cuda)
        .def_readwrite("default_batch_size",
                       &ModelLoadOptions::default_batch_size)
        .def_readwrite("mcts_simulations",
                       &ModelLoadOptions::mcts_simulations)
        .def_readwrite("mcts_depth", &ModelLoadOptions::mcts_depth)
        .def_readwrite("checkpoint_path",
                       &ModelLoadOptions::checkpoint_path);

    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load_tokenizer", &InferenceEngine::load_tokenizer, py::arg("path"))
        .def("forward_logits", &InferenceEngine::forward_logits, py::arg("ids"),
             py::call_guard<py::gil_scoped_release>())
        .def("tokenize", &InferenceEngine::tokenize, py::arg("text"))
        .def("detokenize", &InferenceEngine::detokenize, py::arg("ids"))
        .def("model_config", &InferenceEngine::model_config)
        .def("parameter_count", &InferenceEngine::parameter_count)
        .def("load_model",
             py::overload_cast<const std::string&>(&InferenceEngine::load_model),
             py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_model",
             py::overload_cast<const std::string&, const ModelConfig&>(&InferenceEngine::load_model),
             py::arg("path"),
             py::arg("config"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_model",
             py::overload_cast<const std::string&, const ModelLoadOptions&>(
                 &InferenceEngine::load_model),
             py::arg("path"),
             py::arg("options"),
             py::call_guard<py::gil_scoped_release>())
        // Decode/training loops are pure C++ -> release the GIL so other Python
        // threads (and async I/O) run concurrently.
        .def("generate",
             py::overload_cast<const std::string&, int, float>(&InferenceEngine::generate),
             py::arg("prompt"),
             py::arg("max_tokens") = 50,
             py::arg("temperature") = 0.7f,
             py::call_guard<py::gil_scoped_release>())
        .def("generate_ex",
             py::overload_cast<const std::string&, const GenerationOptions&>(&InferenceEngine::generate),
             py::arg("prompt"),
             py::arg("options"),
             py::call_guard<py::gil_scoped_release>())
        .def("generate_batch", &InferenceEngine::generate_batch,
             py::arg("prompts"),
             py::arg("options") = GenerationOptions{},
             py::call_guard<py::gil_scoped_release>())
        .def("train_step",
             py::overload_cast<const std::vector<int>&, const std::vector<int>&>(&InferenceEngine::train_step),
             py::arg("input"),
             py::arg("target") = std::vector<int>{},
             py::call_guard<py::gil_scoped_release>())
        .def("train_text",
             py::overload_cast<const std::string&>(&InferenceEngine::train_step),
             py::arg("text"),
             py::call_guard<py::gil_scoped_release>())
        .def("request_training_cancellation",
             &InferenceEngine::request_training_cancellation)
        .def("clear_training_cancellation",
             &InferenceEngine::clear_training_cancellation)
        .def("training_cancellation_requested",
             &InferenceEngine::training_cancellation_requested)
        .def("training_optimizer_state_poisoned",
             &InferenceEngine::training_optimizer_state_poisoned)
        .def("configure_progressive_qat",
             [](InferenceEngine& e, const TrainPhaseScheduler& s) {
                 if (!e.trainer)
                     throw std::runtime_error(
                         "configure_progressive_qat: engine nao inicializado (chame load_model primeiro)");
                 e.trainer->configure_progressive_qat(s);
             },
             py::arg("scheduler"))
        .def("progressive_qat_active",
             [](InferenceEngine& e) {
                 return e.trainer ? e.trainer->progressive_qat_active() : false;
             })
        .def("set_sparse_attention",
             [](InferenceEngine& e, bool enabled, int block_size, int top_k_blocks,
                int local_blocks, int sink_blocks) {
                 if (!e.model)
                     throw std::runtime_error(
                         "set_sparse_attention: modelo nao inicializado (chame load_model primeiro)");
                 e.model->set_sparse_attention(enabled, block_size, top_k_blocks,
                                               local_blocks, sink_blocks);
             },
             py::arg("enabled"), py::arg("block_size") = 64, py::arg("top_k_blocks") = 8,
             py::arg("local_blocks") = 1, py::arg("sink_blocks") = 1)
        .def("self_heal", py::overload_cast<>(&InferenceEngine::self_heal))
        .def("self_heal_response",
             py::overload_cast<const std::string&, const std::string&>(&InferenceEngine::self_heal),
             py::arg("prompt"),
             py::arg("response"))
        .def("save_checkpoint", &InferenceEngine::save_checkpoint, py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_model_pack", &InferenceEngine::save_model_pack, py::arg("directory"))
        .def("get_memory_usage", &InferenceEngine::get_memory_usage)
        .def("last_generation_metrics",
             [](const InferenceEngine& engine) { return engine.last_generation_metrics(); })
        // Pacote A.3: route BitLinear forward through the GPU __dp4a
        // packed path on every linear layer in the model.  Idempotent.
        // Returns true if the underlying model was available.
        .def("set_gpu_packed_inference",
             [](InferenceEngine& engine, bool enabled) {
                 if (!engine.model) return false;
                 engine.model->set_gpu_packed_inference(enabled);
                 return true;
             },
             py::arg("enabled"))
        // Pacote A.1: at decode time, route MoE through top-k=k instead
        // of the trained top-k.  Pass 0 to clear and restore the
        // trained top-k.  Returns true if a model is loaded.
        .def("set_moe_inference_top_k",
             [](InferenceEngine& engine, int k) {
                 if (!engine.model) return false;
                 engine.model->set_moe_inference_top_k(k);
                 return true;
             },
             py::arg("k"));
}
