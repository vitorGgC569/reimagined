#include "gpu_moe_training.h"
#include "bitlinear.h"
#include "training_runtime_policy.h"
#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "gpu_execution.h"
#include "cuda/moe_training_kernels.cuh"
#include "cuda/moe_training_wmma.cuh"
#endif

namespace nsos {
namespace {
void check_product(long long n, const char* label) {
    if (n <= 0 || n > std::numeric_limits<int>::max())
        throw std::length_error(std::string("Grouped MoE indexing limit: ") + label);
}
#ifdef USE_CUDA
void check_gpu(cudaError_t result, const char* label) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(result));
}
#endif
}

void GpuMoeTraining::begin_device_accumulation(const std::vector<BitLinear*>& up,
    const std::vector<BitLinear*>& down) {
    if (activity_ || pending_ || up.empty() || up.size() != down.size() || up.size() > 1024)
        throw std::logic_error("Invalid/nested device MoE accumulation group");
    auto owner = std::make_shared<GpuGradientActivity>(static_cast<int>(up.size()));
    std::vector<Parameter*> parameters;
    // Whole-bank preflight before publishing any bindings. Reserve physical
    // gradients for all candidates; logical presence lives ONLY in the device.
    for (size_t e = 0; e < up.size(); ++e) for (BitLinear* op : {up[e], down[e]}) {
        if (!op || !op->supports_gpu_grouped_training())
            throw std::invalid_argument("Unsupported device MoE expert");
        op->track_gradient_contributions();
        GpuMoeTrainingLinearView view{}; Tensor effective, scale;
        op->prepare_gpu_grouped_training_view(view, effective, scale, true);
        for (Parameter* p : {&op->weight, view.bias ? &op->bias : nullptr,
                view.magnitude ? &op->magnitude : nullptr}) {
            if (!p) continue;
            if (p->has_device_gradient_activity() || p->has_gradient() ||
                p->data.get_device() != Device::GPU ||
                (p->grad.size && (p->grad.shape != p->data.shape || p->grad.get_device() != Device::GPU)))
                throw std::logic_error("Device MoE group requires clean, unbound GPU parameters");
            if (!p->grad.size) p->grad = Tensor::uninitialized(p->data.shape.dims, Device::GPU);
            parameters.push_back(p);
        }
    }
    auto unique = parameters; std::sort(unique.begin(), unique.end(), std::less<Parameter*>{});
    if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
        throw std::logic_error("Shared expert parameters require an explicit merged domain");
    for (size_t e = 0; e < up.size(); ++e) for (BitLinear* op : {up[e], down[e]}) {
        op->weight.bind_device_gradient_activity(owner, static_cast<int>(e));
        if (std::find(parameters.begin(), parameters.end(), &op->bias) != parameters.end())
            op->bias.bind_device_gradient_activity(owner, static_cast<int>(e));
        if (std::find(parameters.begin(), parameters.end(), &op->magnitude) != parameters.end())
            op->magnitude.bind_device_gradient_activity(owner, static_cast<int>(e));
    }
    bound_parameters_ = std::move(parameters); bound_up_ = up; bound_down_ = down;
    activity_ = std::move(owner);
}

void GpuMoeTraining::finish_device_accumulation(bool abort, bool materialize_for_audit) {
    if (!activity_) throw std::logic_error("No device MoE accumulation group");
#ifdef USE_CUDA
    activity_->assert_lane();
    if (abort) activity_->abort();
    // Group reset/unbind is an explicit lifetime boundary, never a tape clear.
    check_gpu(cudaStreamSynchronize(gpu::current_stream()), "MoE group finish");
    record_gpu_stream_synchronization();
#endif
    std::vector<unsigned char> contributed(static_cast<size_t>(activity_->experts()),0);
#ifdef USE_CUDA
    if(materialize_for_audit && !abort) {
        check_gpu(cudaMemcpy(contributed.data(), activity_->view().accumulated, contributed.size(), cudaMemcpyDeviceToHost), "MoE explicit audit activity");
        record_gpu_transfer(Device::CPU, Device::GPU, contributed.size());
    }
#endif
    discard_backward_state();
    for (auto* p : bound_parameters_) {
        const int expert=p->device_gradient_binding().expert;
        p->unbind_device_gradient_activity(activity_,materialize_for_audit && !abort && contributed[expert]);
    }
    bound_parameters_.clear(); bound_up_.clear(); bound_down_.clear(); activity_.reset();
}

void GpuMoeTraining::prepare_device_gradient_commit(LinearState& s) {
    s.gradient_views.assign(experts_, {});
    for (int e = 0; e < experts_; ++e) {
        auto* op = s.layers[e]; const auto& v = s.views[e];
        auto prepare = [&](Parameter& p) {
            const auto& b = p.device_gradient_binding();
            if (b.owner != activity_ || b.expert != e || p.grad.shape != p.data.shape ||
                p.grad.get_device() != Device::GPU || !p.grad.size)
                throw std::logic_error("Device MoE gradient destination/binding changed");
            return p.grad.raw_data();
        };
        auto& target = s.gradient_views[e]; target.weight = prepare(op->weight);
        if (v.bias) target.bias = prepare(op->bias);
        if (v.magnitude) target.magnitude = prepare(op->magnitude);
    }
#ifdef USE_CUDA
    auto* device = s.device_gradient_views.ensure(experts_);
    if (!device) throw std::bad_alloc();
    if (s.uploaded_gradient_device != device || s.uploaded_gradient_views != s.gradient_views) {
        // Descriptor replacement must not race a consumer on a nondefault lane.
        check_gpu(cudaStreamSynchronize(gpu::current_stream()), "MoE descriptor replacement");
        record_gpu_stream_synchronization();
        const size_t bytes = s.gradient_views.size() * sizeof(GpuMoeGradientView);
        check_gpu(cudaMemcpy(device, s.gradient_views.data(), bytes, cudaMemcpyHostToDevice), "MoE device commit descriptors");
        record_gpu_transfer(Device::GPU, Device::CPU, bytes);
        s.uploaded_gradient_views = s.gradient_views; s.uploaded_gradient_device = device;
    }
#endif
}

void GpuMoeTraining::prepare_linear(LinearState& s, const std::vector<BitLinear*>& layers) {
    s.layers = layers;
    s.views.assign(layers.size(), {});
    s.effective_weights.resize(layers.size()); s.qat_scales.resize(layers.size());
    s.weight_versions.resize(layers.size()); s.magnitude_versions.resize(layers.size());
    s.bias_versions.resize(layers.size());
    for (size_t e = 0; e < layers.size(); ++e) {
        // Standalone grouped operators use the same sparse parameter contract
        // as Jamba-owned experts. Registration is idempotent across forwards.
        layers[e]->track_gradient_contributions();
        layers[e]->prepare_gpu_grouped_training_view(s.views[e], s.effective_weights[e], s.qat_scales[e], true);
        s.weight_versions[e] = layers[e]->weight.version;
        s.magnitude_versions[e] = layers[e]->magnitude.version;
        s.bias_versions[e] = layers[e]->bias.version;
    }
#ifdef USE_CUDA
    auto* device_views = s.device_views.ensure(layers.size());
    if (!device_views) throw std::bad_alloc();
    // Immutable host metadata only. Blocking copy guarantees staging lifetime;
    // no GPU routing values are read to schedule the grouped computation.
    const size_t bytes = layers.size() * sizeof(GpuMoeTrainingLinearView);
    if (s.uploaded_device != device_views || s.uploaded_views != s.views) {
        check_gpu(cudaStreamSynchronize(gpu::current_stream()), "MoE linear descriptor replacement");
        record_gpu_stream_synchronization();
        check_gpu(cudaMemcpy(device_views, s.views.data(), bytes, cudaMemcpyHostToDevice),
                  "Grouped MoE descriptor upload");
        record_gpu_transfer(Device::GPU, Device::CPU, bytes);
        s.uploaded_views = s.views;
        s.uploaded_device = device_views;
    }
    if (std::any_of(s.views.begin(), s.views.end(), [](const auto& view) { return view.qat_scale != nullptr; })) {
        const int elements = s.views.front().inputs * s.views.front().outputs;
        const int blocks = std::min(4096, (elements - 1) / 256 + 1);
        float* partials = s.qat_partials.ensure(static_cast<size_t>(blocks) * layers.size());
        if (!partials) throw std::bad_alloc();
        if (!launch_moe_training_prepare_qat(device_views, offsets_.get(), partials, experts_, elements))
            throw std::runtime_error("Grouped MoE active-expert QAT preparation failed");
    }
#endif
    const int in = s.views.front().inputs, out = s.views.front().outputs;
    s.normalized = Tensor::uninitialized({capacity_, in}, Device::GPU);
    const bool quantized = std::any_of(s.views.begin(), s.views.end(),
        [](const auto& view) { return view.activation_bits != 0; });
    s.prepared = quantized ? Tensor::uninitialized({capacity_, in}, Device::GPU) : s.normalized;
    s.inverse_rms = Tensor::uninitialized({capacity_}, Device::GPU);
    s.pre = Tensor::uninitialized({capacity_, out}, Device::GPU);
    s.post = Tensor::uninitialized({capacity_, out}, Device::GPU);
}

void GpuMoeTraining::allocate_gradients(LinearState& s) {
    const int in = s.views.front().inputs, out = s.views.front().outputs;
    s.grad_out = Tensor::uninitialized({capacity_, out}, Device::GPU);
    s.grad_pre = Tensor::uninitialized({capacity_, out}, Device::GPU);
    s.grad_input = Tensor::uninitialized({capacity_, in}, Device::GPU);
    s.grad_weight = Tensor::uninitialized({experts_, out, in}, Device::GPU);
    s.grad_bias = Tensor::uninitialized({experts_, out}, Device::GPU);
    s.grad_magnitude = Tensor::uninitialized({experts_, out}, Device::GPU);
}

void GpuMoeTraining::verify_parameters(const LinearState& s) const {
    for (size_t e = 0; e < s.layers.size(); ++e) {
        const auto* op = s.layers[e]; const auto& v = s.views[e];
        if (!op->supports_gpu_grouped_training() ||
            (op->input_norm_strategy() != NormStrategy::NONE) != (v.rms_input != 0) ||
            op->exact_linear_mode() != (v.magnitude == nullptr) ||
            op->reference_path_enabled() != (v.qat_scale == nullptr) ||
            (v.activation_bits && op->precision_bits != v.activation_bits) ||
            op->weight.version != s.weight_versions[e] || op->weight.data.raw_data() != v.latent_weight ||
            op->weight.data.get_device() != Device::GPU ||
            op->input_features() != v.inputs || op->output_features() != v.outputs ||
            (v.magnitude && (op->magnitude.version != s.magnitude_versions[e] ||
                op->magnitude.data.raw_data() != v.magnitude)) ||
            (v.bias && (op->bias.version != s.bias_versions[e] || op->bias.data.raw_data() != v.bias)))
            throw std::runtime_error("Grouped MoE parameters changed between forward and backward");
    }
}

void GpuMoeTraining::prepare_gradient_commit(LinearState& s, const std::vector<int>& counts) {
    s.gradient_views.assign(experts_, {});
    for (int e = 0; e < experts_; ++e) {
        if (counts[e] == 0) continue;
        auto* op = s.layers[e]; const auto& v = s.views[e];
        auto& destination = s.gradient_views[e];
        auto prepare = [&](Parameter& p, unsigned bit) {
            if (p.data.get_device() != Device::GPU ||
                (p.grad.size && (p.grad.shape != p.data.shape ||
                 p.grad.get_device() != Device::GPU)))
                throw std::logic_error("Grouped MoE gradient destination shape/device mismatch");
            if (p.has_gradient()) destination.add_mask |= bit;
            // Allocate only for actual contributions, never for empty experts.
            // First contribution overwrites even a retained, nonzero buffer.
            if (!p.grad.size) p.grad = Tensor::uninitialized(p.data.shape.dims, Device::GPU);
            return p.grad.raw_data();
        };
        destination.weight = prepare(op->weight, 1);
        if (v.bias) destination.bias = prepare(op->bias, 2);
        if (v.magnitude) destination.magnitude = prepare(op->magnitude, 4);
    }
#ifdef USE_CUDA
    auto* device = s.device_gradient_views.ensure(experts_);
    if (!device) throw std::bad_alloc();
    if (s.uploaded_gradient_device != device || s.uploaded_gradient_views != s.gradient_views) {
        const size_t bytes = s.gradient_views.size() * sizeof(GpuMoeGradientView);
        check_gpu(cudaMemcpy(device, s.gradient_views.data(), bytes, cudaMemcpyHostToDevice),
                  "Grouped MoE gradient destination upload");
        record_gpu_transfer(Device::GPU, Device::CPU, bytes);
        s.uploaded_gradient_views = s.gradient_views;
        s.uploaded_gradient_device = device;
    }
#endif
}

void GpuMoeTraining::enqueue_gradient_commit(LinearState& s) {
#ifdef USE_CUDA
    const auto& v = s.views.front();
    if (!launch_moe_training_accumulate_gradients(s.device_gradient_views.get(), offsets_.get(),
        s.grad_weight.raw_data(), s.grad_bias.raw_data(), s.grad_magnitude.raw_data(),
        experts_, v.inputs * v.outputs, v.outputs))
        throw std::runtime_error("Grouped MoE gradient accumulation launch failed");
    gpu::record_dispatch(gpu::DispatchPath::GroupedMoeGradientCommit);
#else
    throw std::runtime_error("Grouped MoE gradient accumulation requires a GPU build");
#endif
}

void GpuMoeTraining::publish_gradient_commit(LinearState& s, const std::vector<int>& counts) {
    for (int e = 0; e < experts_; ++e) {
        if (counts[e] == 0) continue;
        auto* op = s.layers[e]; const auto& v = s.views[e];
        op->weight.mark_gradient_contribution();
        if (v.bias) op->bias.mark_gradient_contribution();
        if (v.magnitude) op->magnitude.mark_gradient_contribution();
    }
}

Tensor GpuMoeTraining::forward(const Tensor& input, const Tensor& routing,
    const std::vector<BitLinear*>& up, const std::vector<BitLinear*>& down, int top_k) {
    // Trainer may pre-bind experts before the existing Jamba grouped owner is
    // constructed. Adopt that explicit domain without changing Jamba's ABI.
    if (!activity_ && !up.empty() && up.front() && up.front()->weight.has_device_gradient_activity()) {
        activity_ = up.front()->weight.device_gradient_binding().owner;
        bound_up_ = up; bound_down_ = down;
        if (activity_->experts() != static_cast<int>(up.size()) || up.size() != down.size())
            throw std::logic_error("Adopted MoE domain geometry mismatch");
        for (size_t e = 0; e < up.size(); ++e) for (auto* op : {up[e], down[e]}) {
            if (!op) throw std::logic_error("Null expert in adopted domain");
            for (auto* p : {&op->weight, &op->bias, &op->magnitude})
                if (p->device_gradient_binding().owner == activity_) bound_parameters_.push_back(p);
        }
        auto self = weak_from_this().lock();
        if (!self) throw std::logic_error("Adopted device MoE producer requires shared ownership");
        activity_->set_producer(self);
    }
    if (activity_ && (up != bound_up_ || down != bound_down_))
        throw std::logic_error("MoE expert registry changed inside device accumulation");
#ifdef USE_CUDA
    if (activity_) activity_->assert_lane();
#endif
    discard_backward_state();
    if (input.get_device() != Device::GPU || routing.get_device() != Device::GPU ||
        (input.shape.size() != 2 && input.shape.size() != 3) || input.shape.back() <= 0 ||
        up.empty() || up.size() != down.size() || up.size() > 1024 ||
        top_k <= 0 || static_cast<size_t>(top_k) > up.size())
        throw std::invalid_argument("Invalid grouped MoE training input/routing geometry");
    experts_ = static_cast<int>(up.size()); dim_ = input.shape.back();
    rows_ = input.size / dim_;
    if (routing.shape.dims != std::vector<int>{rows_, experts_})
        throw std::invalid_argument("Grouped MoE routing must be [rows,experts]");
    if (!up.front() || !down.front()) throw std::invalid_argument("Null grouped MoE expert");
    hidden_dim_ = up.front()->output_features();
    // HIP/CUDA grid.y is limited to 65535. Validate before constructing grids
    // and before multiplying three unconstrained dimensions.
    constexpr int max_tiled_y = 65535 * 16;
    if (rows_ <= 0 || hidden_dim_ <= 0 || rows_ > max_tiled_y ||
        dim_ > max_tiled_y || hidden_dim_ > max_tiled_y)
        throw std::length_error("Grouped MoE tile grid geometry exceeds supported limits");
    matmul_mode_ = matmul_precision_mode();
    wmma_enabled_ = training_policy::moe_wmma_training();
#ifdef USE_CUDA
    if (wmma_enabled_ && !moe_training_wmma_supported())
        throw std::runtime_error("NSOS_MOE_WMMA_TRAINING requires compiled rocWMMA and a supported RDNA3 wave32 device");
#endif
    if (matmul_mode_ < 0 || matmul_mode_ > 2)
        throw std::invalid_argument("Grouped MoE supports FP32/BF16/FP16 GEMM operands");
    check_product(static_cast<long long>(rows_) * top_k, "active capacity");
    capacity_ = rows_ * top_k;
    check_product(static_cast<long long>(capacity_) * std::max(dim_, hidden_dim_), "activations");
    const long long expert_elements = static_cast<long long>(dim_) * hidden_dim_;
    check_product(expert_elements, "expert weight");
    check_product(experts_ * expert_elements, "expert gradients");
    check_product(static_cast<long long>(rows_) * experts_, "inverse map");
    for (int e = 0; e < experts_; ++e) {
        if (!up[e] || !down[e] || !up[e]->supports_gpu_grouped_training() ||
            !down[e]->supports_gpu_grouped_training() || up[e]->input_features() != dim_ ||
            up[e]->output_features() != hidden_dim_ || down[e]->input_features() != hidden_dim_ ||
            down[e]->output_features() != dim_)
            throw std::invalid_argument("Grouped MoE expert geometry/adapter is unsupported; no silent fallback");
    }
    input_shape_ = input.shape.dims;
    input_ = input.reshape({rows_, dim_}).clone();
    scales_ = Tensor::uninitialized({capacity_}, Device::GPU);
    hidden_ = Tensor::uninitialized({capacity_, hidden_dim_}, Device::GPU);
#ifdef USE_CUDA
    if (!counts_.ensure(experts_) || !offsets_.ensure(experts_ + 2) ||
        !permutation_.ensure(capacity_) || !inverse_.ensure(static_cast<size_t>(rows_) * experts_))
        throw std::bad_alloc();
    if (!launch_moe_training_route(routing.raw_data(), counts_.get(), offsets_.get(),
        permutation_.get(), inverse_.get(), scales_.raw_data(), rows_, experts_, capacity_))
        throw std::runtime_error("Grouped MoE routing launch failed");
    prepare_linear(up_, up); prepare_linear(down_, down);
    if (!launch_moe_training_linear_forward(up_.device_views.get(), offsets_.get(), permutation_.get(),
        input_.raw_data(), true, up_.normalized.raw_data(), up_.prepared.raw_data(), up_.inverse_rms.raw_data(),
        up_.pre.raw_data(), up_.post.raw_data(), hidden_.raw_data(), rows_, capacity_, experts_,
        dim_, hidden_dim_, matmul_mode_, wmma_enabled_) ||
        !launch_moe_training_linear_forward(down_.device_views.get(), offsets_.get(), permutation_.get(),
        hidden_.raw_data(), false, down_.normalized.raw_data(), down_.prepared.raw_data(), down_.inverse_rms.raw_data(),
        down_.pre.raw_data(), down_.post.raw_data(), nullptr, rows_, capacity_, experts_,
        hidden_dim_, dim_, matmul_mode_, wmma_enabled_))
        throw std::runtime_error("Grouped MoE expert forward launch failed");
    Tensor output = Tensor::uninitialized({rows_, dim_}, Device::GPU);
    if (!launch_moe_training_combine(down_.post.raw_data(), inverse_.get(), offsets_.get(),
        scales_.raw_data(), output.raw_data(), rows_, dim_, experts_))
        throw std::runtime_error("Grouped MoE combine launch failed");
    pending_ = true;
    gpu::record_dispatch(gpu::DispatchPath::GroupedMoeTraining);
    return output.reshape(input_shape_);
#else
    throw std::runtime_error("Grouped MoE requires a GPU build");
#endif
}

std::vector<int> GpuMoeTraining::materialize_counts() {
    if (!pending_) throw std::runtime_error("Grouped MoE has no pending routing state");
#ifdef USE_CUDA
    std::vector<int> host(experts_ + 2);
    const size_t bytes = host.size() * sizeof(int);
    const auto stream = gpu::current_stream();
    check_gpu(cudaMemcpyAsync(host.data(), offsets_.get(), bytes, cudaMemcpyDeviceToHost, stream),
              "Grouped MoE late registry counts");
    record_gpu_transfer(Device::CPU, Device::GPU, bytes);
    check_gpu(cudaStreamSynchronize(stream), "Grouped MoE registry synchronization");
    record_gpu_stream_synchronization();
    if (host.back() != 0 || host.front() != 0 || host[experts_] < 0 || host[experts_] > capacity_)
        throw std::runtime_error("Grouped MoE routing exceeds sparse capacity or violates offsets");
    std::vector<int> counts(experts_);
    for (int e = 0; e < experts_; ++e) {
        if (host[e] < 0 || host[e + 1] < host[e] || host[e + 1] > host[experts_])
            throw std::runtime_error("Grouped MoE offsets are not monotonic");
        counts[e] = host[e + 1] - host[e];
    }
    return counts;
#else
    throw std::runtime_error("Grouped MoE requires a GPU build");
#endif
}

std::pair<Tensor, Tensor> GpuMoeTraining::backward(const Tensor& grad, bool router_gradient) try {
    if (!pending_ || grad.get_device() != Device::GPU || grad.shape.dims != input_shape_)
        throw std::invalid_argument("Grouped MoE backward requires its matching forward layout/device");
    if (wmma_enabled_ != training_policy::moe_wmma_training() || matmul_mode_ != matmul_precision_mode())
        throw std::runtime_error("Grouped MoE compute policy changed between forward and backward");
    verify_parameters(up_); verify_parameters(down_);
    allocate_gradients(up_); allocate_gradients(down_);
#ifdef USE_CUDA
    if (!launch_moe_training_linear_backward(down_.device_views.get(), offsets_.get(), permutation_.get(),
        scales_.raw_data(), grad.raw_data(), true, nullptr, down_.normalized.raw_data(),
        down_.prepared.raw_data(), down_.inverse_rms.raw_data(), down_.pre.raw_data(),
        down_.grad_out.raw_data(), down_.grad_pre.raw_data(), down_.grad_input.raw_data(),
        down_.grad_weight.raw_data(), down_.grad_bias.raw_data(), down_.grad_magnitude.raw_data(),
        rows_, capacity_, experts_, hidden_dim_, dim_, matmul_mode_, wmma_enabled_) ||
        !launch_moe_training_linear_backward(up_.device_views.get(), offsets_.get(), permutation_.get(),
        nullptr, down_.grad_input.raw_data(), false, up_.post.raw_data(), up_.normalized.raw_data(),
        up_.prepared.raw_data(), up_.inverse_rms.raw_data(), up_.pre.raw_data(),
        up_.grad_out.raw_data(), up_.grad_pre.raw_data(), up_.grad_input.raw_data(),
        up_.grad_weight.raw_data(), up_.grad_bias.raw_data(), up_.grad_magnitude.raw_data(),
        rows_, capacity_, experts_, dim_, hidden_dim_, matmul_mode_, wmma_enabled_))
        throw std::runtime_error("Grouped MoE expert backward launch failed");
    Tensor input_grad = Tensor::uninitialized({rows_, dim_}, Device::GPU);
    if (!launch_moe_training_combine(up_.grad_input.raw_data(), inverse_.get(), offsets_.get(),
        nullptr, input_grad.raw_data(), rows_, dim_, experts_))
        throw std::runtime_error("Grouped MoE input gradient combine failed");
    Tensor router_grad;
    if (router_gradient) {
        router_grad = Tensor::uninitialized({rows_, experts_}, Device::GPU);
        if (!launch_moe_training_router_grad(grad.raw_data(), down_.post.raw_data(), inverse_.get(),
            offsets_.get(), router_grad.raw_data(), rows_, dim_, experts_))
            throw std::runtime_error("Grouped MoE router gradient launch failed");
    }
    if (activity_) {
        try {
            activity_->assert_lane();
            prepare_device_gradient_commit(up_); prepare_device_gradient_commit(down_);
            // Reject even overlapping ranges, not only identical starts.
            std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
            for (Parameter* p : bound_parameters_) {
                const auto start = reinterpret_cast<uintptr_t>(p->grad.raw_data());
                ranges.emplace_back(start, start + static_cast<size_t>(p->grad.size) * sizeof(float));
            }
            std::sort(ranges.begin(), ranges.end());
            for (size_t i = 1; i < ranges.size(); ++i)
                if (ranges[i].first < ranges[i-1].second) throw std::logic_error("Overlapping MoE gradient destinations");
            auto a = activity_->view();
            if (!launch_sparse_optimizer_activity_update(a, offsets_.get(), capacity_, offsets_.get() + experts_ + 1))
                throw std::runtime_error("MoE device contribution union failed");
            for (auto* s : {&up_, &down_}) {
                const auto& v = s->views.front();
                if (!launch_moe_training_accumulate_gradients_activity(s->device_gradient_views.get(), a,
                    s->grad_weight.raw_data(), s->grad_bias.raw_data(), s->grad_magnitude.raw_data(),
                    v.inputs * v.outputs, v.outputs, activity_->gradient_scale)) throw std::runtime_error("MoE device gradient commit failed");
                gpu::record_dispatch(gpu::DispatchPath::GroupedMoeGradientCommit);
            }
        } catch (...) { activity_->abort(); discard_backward_state(); throw; }
        input_grad = input_grad.reshape(input_shape_); discard_backward_state();
        return {input_grad, router_grad};
    }
    std::vector<int> counts;
    try { counts = materialize_counts(); }
    catch (...) { discard_backward_state(); throw; }
    try {
        // Preflight both banks before any destination is written/published.
        // Scratch is kept alive by this owner until the queued stream uses end.
        if (std::any_of(counts.begin(), counts.end(), [](int count) { return count != 0; })) {
            prepare_gradient_commit(up_, counts); prepare_gradient_commit(down_, counts);
            // Expert destinations must be independent: otherwise concurrent
            // bank/segment writes would replace the old ordered accumulation.
            gradient_destinations_.clear();
            for (const auto* state : {&up_, &down_}) for (const auto& view : state->gradient_views)
                for (float* pointer : {view.weight, view.bias, view.magnitude})
                    if (pointer) gradient_destinations_.push_back(pointer);
            std::sort(gradient_destinations_.begin(), gradient_destinations_.end(), std::less<float*>{});
            if (std::adjacent_find(gradient_destinations_.begin(), gradient_destinations_.end()) !=
                gradient_destinations_.end())
                throw std::logic_error("Grouped MoE gradient destinations alias; parallel accumulation is unsupported");
            enqueue_gradient_commit(up_); enqueue_gradient_commit(down_);
            publish_gradient_commit(up_, counts); publish_gradient_commit(down_, counts);
        }
    } catch (...) { discard_backward_state(); throw; }
    input_grad = input_grad.reshape(input_shape_);
    discard_backward_state();
    return {input_grad, router_grad};
#else
    throw std::runtime_error("Grouped MoE requires a GPU build");
#endif
}

catch (...) {
#ifdef USE_CUDA
    if (activity_) { try { activity_->abort(); } catch (...) {} }
#endif
    try { discard_backward_state(); } catch (...) {} throw;
}

GpuMoeTraining::~GpuMoeTraining() {
#ifdef USE_CUDA
    if (pending_) { try { discard_backward_state(); } catch (...) {
        (void)cudaStreamSynchronize(gpu::current_stream());
    } }
    // Parameters' shared domains may survive this object. Tape events retain
    // only scratch; teardown drains its own events, never device-wide work.
    for (auto& tape : retired_tapes_) {
        (void)cudaEventSynchronize(tape.ready); (void)cudaEventDestroy(tape.ready);
    }
#endif
}

Tensor GpuMoeTraining::add_device_qat_regularization(float base_coefficient, int accumulation_steps) {
    if (!activity_ || pending_ || !std::isfinite(base_coefficient) || base_coefficient < 0 || accumulation_steps < 1)
        throw std::invalid_argument("Active-only QAT requires a completed task group and finite coefficient");
#ifdef USE_CUDA
    activity_->assert_lane();
    qat_group_loss_ = Tensor::uninitialized({1}, Device::GPU);
    check_gpu(cudaMemsetAsync(qat_group_loss_.raw_data(), 0, sizeof(float), gpu::current_stream()), "MoE QAT loss clear");
    if (!qat_union_offsets_.ensure(activity_->experts()+2)) throw std::bad_alloc();
    for (auto* s : {&up_, &down_}) {
        if (s->views.empty()) throw std::logic_error("MoE QAT group has no task microbatch");
        s->regularization_weights.resize(s->layers.size()); s->regularization_scales.resize(s->layers.size());
        s->regularization_views = s->views;
        const int elements = s->views.front().inputs * s->views.front().outputs;
        const int blocks = (std::min)(4096, (elements-1)/256+1);
        for (size_t e = 0; e < s->layers.size(); ++e) {
            auto& view = s->regularization_views[e];
            if (s->layers[e]->quantization_sensitive()) { view.qat_scale = nullptr; continue; }
            auto& target = s->regularization_weights[e]; auto& scale = s->regularization_scales[e];
            if (target.shape != s->layers[e]->weight.data.shape) target = Tensor::uninitialized(s->layers[e]->weight.data.shape.dims, Device::GPU);
            if (!scale.size) scale = Tensor::uninitialized({1}, Device::GPU);
            view.weight = target.raw_data(); view.qat_scale = scale.raw_data();
        }
        auto* device = s->regularization_device_views.ensure(s->layers.size());
        auto* partials = s->qat_partials.ensure(static_cast<size_t>(blocks)*s->layers.size());
        auto* loss_partials = s->qat_loss_partials.ensure(static_cast<size_t>(blocks)*s->layers.size());
        if (!device || !partials || !loss_partials) throw std::bad_alloc();
        // Regularizer staging survives this upload and every queued consumer;
        // replacing a previous group's descriptors is an ordered boundary.
        check_gpu(cudaStreamSynchronize(gpu::current_stream()), "MoE QAT descriptors"); record_gpu_stream_synchronization();
        const size_t bytes = s->regularization_views.size()*sizeof(GpuMoeTrainingLinearView);
        check_gpu(cudaMemcpy(device, s->regularization_views.data(), bytes, cudaMemcpyHostToDevice), "MoE QAT regularizer upload");
        record_gpu_transfer(Device::GPU, Device::CPU, bytes);
        if (!launch_moe_training_active_qat_regularization(device, s->device_gradient_views.get(),
            activity_->view(), qat_union_offsets_.get(), partials, loss_partials, qat_group_loss_.raw_data(),
            elements, base_coefficient, accumulation_steps))
            throw std::runtime_error("MoE active-only QAT regularization launch failed");
    }
    return qat_group_loss_;
#else
    throw std::runtime_error("Device QAT requires GPU build");
#endif
}

void GpuMoeTraining::discard_backward_state() {
#ifdef USE_CUDA
    for (auto it = retired_tapes_.begin(); it != retired_tapes_.end();) {
        const auto status = cudaEventQuery(it->ready);
        if (status == cudaSuccess) { check_gpu(cudaEventDestroy(it->ready), "MoE tape event release"); it = retired_tapes_.erase(it); }
        else if (status == cudaErrorNotReady) ++it;
        else check_gpu(status, "MoE tape event query");
    }
    RetiredTape retired{};
    auto retain = [&](const Tensor& value) { if (value.size) retired.storage.push_back(value); };
    retain(input_); retain(hidden_); retain(scales_);
    for (const auto* s : {&up_, &down_})
        for (const Tensor* t : {&s->normalized, &s->prepared, &s->inverse_rms, &s->pre, &s->post,
                &s->grad_out, &s->grad_pre, &s->grad_input, &s->grad_weight, &s->grad_bias, &s->grad_magnitude}) retain(*t);
    if (!retired.storage.empty()) {
#if defined(NSOS_GPU_BACKEND_HIP)
        check_gpu(hipEventCreateWithFlags(&retired.ready, hipEventDisableTiming), "MoE tape event create");
#else
        check_gpu(cudaEventCreateWithFlags(&retired.ready, cudaEventDisableTiming), "MoE tape event create");
#endif
        const auto result = cudaEventRecord(retired.ready, gpu::current_stream());
        if (result != cudaSuccess) { (void)cudaEventDestroy(retired.ready); check_gpu(result, "MoE tape event record"); }
        retired_tapes_.push_back(std::move(retired));
    }
#endif
    pending_ = false;
    input_ = Tensor(); hidden_ = Tensor(); scales_ = Tensor(); input_shape_.clear();
    for (LinearState* s : {&up_, &down_}) {
        // Effective QAT storage is reusable workspace, not a valid tape or
        // cached arithmetic result. Active experts are re-prepared every forward.
        s->normalized = Tensor(); s->prepared = Tensor(); s->inverse_rms = Tensor();
        s->pre = Tensor(); s->post = Tensor(); s->grad_out = Tensor(); s->grad_pre = Tensor();
        s->grad_input = Tensor(); s->grad_weight = Tensor(); s->grad_bias = Tensor(); s->grad_magnitude = Tensor();
    }
}
} // namespace nsos
