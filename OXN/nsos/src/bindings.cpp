#include "../include/jamba.h"
#include "../include/layer_audit.h"
#include "../include/nsos/determinism.h"
#include "../include/nsos_sdk.h"
#include "../include/tensor.h"
#include "../include/trainer.h"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace py = pybind11;
using namespace nsos;

namespace {

py::array_t<float> tensor_to_numpy(Tensor& tensor) {
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

}  // namespace

PYBIND11_MODULE(nsos_ext, m) {
    m.doc() = "NSOS Python bindings";

    m.def("fast_gpu_supported", []() {
#ifdef USE_CUDA
#ifndef NSOS_CUDA_MIN_ARCH
#define NSOS_CUDA_MIN_ARCH 0
#endif
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
            return false;
        }
        int device_id = 0;
        if (cudaGetDevice(&device_id) != cudaSuccess) {
            device_id = 0;
        }
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
            return false;
        }
        const int device_arch = props.major * 10 + props.minor;
        if (NSOS_CUDA_MIN_ARCH > 0 && device_arch < NSOS_CUDA_MIN_ARCH) {
            return false;
        }
        if (cudaFree(nullptr) != cudaSuccess) {
            return false;
        }
        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        cublasDestroy(handle);
        return true;
#else
        return false;
#endif
    });
    m.def("set_seed", [](uint64_t seed) {
        determinism::DeterminismManager::instance().set_global_seed(seed);
    });
    m.def("get_seed", []() {
        return determinism::DeterminismManager::instance().get_global_seed();
    });
    m.def("get_determinism_report", []() {
        return determinism::DeterminismManager::instance().get_determinism_report();
    });

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("d_model", &ModelConfig::d_model)
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("n_heads", &ModelConfig::n_heads)
        .def_readwrite("n_kv_heads", &ModelConfig::n_kv_heads)
        .def_readwrite("sliding_window", &ModelConfig::sliding_window)
        .def_readwrite("attention_period", &ModelConfig::attention_period)
        .def_readwrite("attention_slot", &ModelConfig::attention_slot)
        .def_readwrite("num_experts", &ModelConfig::num_experts)
        .def_readwrite("num_experts_per_token", &ModelConfig::num_experts_per_token)
        .def_readwrite("use_moe", &ModelConfig::use_moe)
        .def_readwrite("moe_period", &ModelConfig::moe_period)
        .def_readwrite("moe_slot", &ModelConfig::moe_slot)
        .def_readwrite("use_ttt", &ModelConfig::use_ttt)
        .def_readwrite("ttt_period", &ModelConfig::ttt_period)
        .def_readwrite("ttt_slot", &ModelConfig::ttt_slot)
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
        .def_readwrite("mamba_last_fallback_reason", &GenerationMetrics::mamba_last_fallback_reason);

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::vector<int>, Device, float>(),
             py::arg("shape"),
             py::arg("device") = Device::CPU,
             py::arg("fill") = 0.0f)
        .def_property_readonly("shape", [](const Tensor& tensor) { return tensor.shape.dims; })
        .def_property_readonly("device", [](const Tensor& tensor) { return tensor.device; })
        .def("to", &Tensor::to)
        .def("cpu", &Tensor::cpu)
        .def("clone", &Tensor::clone)
        .def("numpy", &tensor_to_numpy)
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
        .def_readwrite("data", &Parameter::data)
        .def_readwrite("grad", &Parameter::grad)
        .def("zero_grad", &Parameter::zero_grad);

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

    py::class_<LayerAuditSummary>(m, "LayerAuditSummary")
        .def(py::init<>())
        .def_readwrite("phase", &LayerAuditSummary::phase)
        .def_readwrite("records", &LayerAuditSummary::records)
        .def_readwrite("forward_records", &LayerAuditSummary::forward_records)
        .def_readwrite("backward_records", &LayerAuditSummary::backward_records)
        .def_readwrite("router_records", &LayerAuditSummary::router_records)
        .def_readwrite("token_contexts", &LayerAuditSummary::token_contexts)
        .def_readwrite("training_steps", &LayerAuditSummary::training_steps)
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
        .def("summary_only", &LayerAuditCollector::summary_only)
        .def("record_sample_rate", &LayerAuditCollector::record_sample_rate)
        .def("max_records_per_phase", &LayerAuditCollector::max_records_per_phase)
        .def("store_token_contexts", &LayerAuditCollector::store_token_contexts)
        .def("records", &LayerAuditCollector::records)
        .def("token_contexts", &LayerAuditCollector::token_contexts)
        .def("training_steps", &LayerAuditCollector::training_steps)
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

    py::class_<JambaModel>(m, "JambaModel")
        .def(py::init<int, int, int, Device>(),
             py::arg("num_layers"),
             py::arg("d_model"),
             py::arg("vocab_size") = 128000,
             py::arg("device") = Device::CPU)
        .def(py::init<const ModelConfig&, Device>(),
             py::arg("config"),
             py::arg("device") = Device::CPU)
        .def("forward", py::overload_cast<const Tensor&, Context*>(&JambaModel::forward),
             py::arg("x"), py::arg("ctx") = nullptr)
        .def("forward_ids", &JambaModel::forward_ids, py::arg("ids"), py::arg("ctx") = nullptr)
        .def("forward_ids_batch", &JambaModel::forward_ids_batch, py::arg("batch_ids"), py::arg("ctx") = nullptr)
        .def("forward_trunk", &JambaModel::forward_trunk, py::arg("ids"), py::arg("ctx") = nullptr)
        .def("forward_embedding", &JambaModel::forward_embedding, py::arg("x"), py::arg("ctx") = nullptr)
        .def("reason", &JambaModel::reason, py::arg("x"), py::arg("num_simulations") = 100)
        .def("forward_thought", &JambaModel::forward_thought)
        .def("run_reasoning_loop", py::overload_cast<const Tensor&, int>(&JambaModel::run_reasoning_loop))
        .def("backward_external", &JambaModel::backward_external)
        .def("backward_embedding", &JambaModel::backward_embedding)
        .def("backward", &JambaModel::backward)
        .def("reset_session", &JambaModel::reset_session)
        .def("set_hamiltonian_mode", &JambaModel::set_hamiltonian_mode)
        .def("session_adapt", &JambaModel::session_adapt)
        .def("run_simd_inference", &JambaModel::run_simd_inference)
        .def("save", &JambaModel::save)
        .def("load", &JambaModel::load, py::arg("path"), py::arg("strict") = true)
        .def("set_reference_path", &JambaModel::set_reference_path, py::arg("enabled"))
        .def("release_full_precision_linear_weights",
             &JambaModel::release_full_precision_linear_weights)
        .def("save_edge_linear_pack", &JambaModel::save_edge_linear_pack, py::arg("path"))
        .def("load_edge_linear_pack",
             &JambaModel::load_edge_linear_pack,
             py::arg("path"),
             py::arg("release_full_precision") = true)
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

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<JambaModel*, float>(), py::arg("model"), py::arg("learning_rate") = 0.001f)
        .def_readwrite("learning_rate", &Trainer::learning_rate)
        .def_readwrite("beta1", &Trainer::beta1)
        .def_readwrite("beta2", &Trainer::beta2)
        .def_readwrite("eps", &Trainer::eps)
        .def_readwrite("weight_decay", &Trainer::weight_decay)
        .def_readwrite("max_grad_norm", &Trainer::max_grad_norm)
        .def_readwrite("min_learning_rate_scale", &Trainer::min_learning_rate_scale)
        .def_readwrite("first_token_loss_scale", &Trainer::first_token_loss_scale)
        .def_readwrite("eos_loss_scale", &Trainer::eos_loss_scale)
        .def_readwrite("repetition_unlikelihood_scale", &Trainer::repetition_unlikelihood_scale)
        .def_readwrite("moe_aux_loss_scale", &Trainer::moe_aux_loss_scale)
        .def_readwrite("warmup_steps", &Trainer::warmup_steps)
        .def_readwrite("global_step_count", &Trainer::global_step_count)
        .def_readwrite("total_training_steps", &Trainer::total_training_steps)
        .def_readwrite("eos_token_id", &Trainer::eos_token_id)
        .def_readwrite("phase_scheduler", &Trainer::phase_scheduler)
        .def_readwrite("last_auxiliary_stats", &Trainer::last_auxiliary_stats)
        .def("configure_progressive_qat", &Trainer::configure_progressive_qat, py::arg("scheduler"))
        .def("progressive_qat_active", &Trainer::progressive_qat_active)
        .def("train_step", &Trainer::train_step)
        .def("train_supervised",
             &Trainer::train_supervised,
             py::arg("prompt_tokens"),
             py::arg("answer_tokens"))
        .def("train_supervised_batch",
             &Trainer::train_supervised_batch,
             py::arg("prompt_batch"),
             py::arg("answer_batch"))
        .def("train_loop",
             &Trainer::train_loop,
             py::arg("tokens"),
             py::arg("epochs"),
             py::arg("batch_size"),
             py::arg("seq_len"),
             py::arg("callback") = nullptr,
             py::arg("max_steps") = -1);

    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load_model",
             py::overload_cast<const std::string&, const ModelConfig&>(&InferenceEngine::load_model),
             py::arg("path"),
             py::arg("config") = ModelConfig{})
        .def("generate",
             py::overload_cast<const std::string&, int, float>(&InferenceEngine::generate),
             py::arg("prompt"),
             py::arg("max_tokens") = 50,
             py::arg("temperature") = 0.7f)
        .def("generate_ex",
             py::overload_cast<const std::string&, const GenerationOptions&>(&InferenceEngine::generate),
             py::arg("prompt"),
             py::arg("options"))
        .def("generate_batch", &InferenceEngine::generate_batch,
             py::arg("prompts"),
             py::arg("options") = GenerationOptions{})
        .def("train_step",
             py::overload_cast<const std::vector<int>&, const std::vector<int>&>(&InferenceEngine::train_step),
             py::arg("input"),
             py::arg("target") = std::vector<int>{})
        .def("train_text",
             py::overload_cast<const std::string&>(&InferenceEngine::train_step),
             py::arg("text"))
        .def("self_heal", py::overload_cast<>(&InferenceEngine::self_heal))
        .def("self_heal_response",
             py::overload_cast<const std::string&, const std::string&>(&InferenceEngine::self_heal),
             py::arg("prompt"),
             py::arg("response"))
        .def("save_checkpoint", &InferenceEngine::save_checkpoint, py::arg("path"))
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
