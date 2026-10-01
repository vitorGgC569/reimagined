#pragma once
#include "tensor.h"
#include "nsos_context.h"
#include "gpu_gradient_activity.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace nsos {

// Only diagnostics may opt into a host materialization of device membership.
// Optimizer code never enters this scope; ordinary host APIs fail closed.
inline thread_local bool device_gradient_audit_scope = false;
class DeviceGradientAuditScope {
    bool previous_;
public:
    DeviceGradientAuditScope() : previous_(device_gradient_audit_scope) { device_gradient_audit_scope = true; }
    ~DeviceGradientAuditScope() { device_gradient_audit_scope = previous_; }
    DeviceGradientAuditScope(const DeviceGradientAuditScope&) = delete;
    DeviceGradientAuditScope& operator=(const DeviceGradientAuditScope&) = delete;
};

// Unified Context using nsos_context.h logic
// No redefinition here.

class Parameter {
public:
    Tensor data;
    Tensor grad;
    std::string name;
    std::string base_name;
    bool trainable = true;
    // Parameter names are assembled while a model registry is constructed and
    // then frozen.  A frozen name is an immutable model identity used by
    // checkpoints, optimizer sidecars and audit manifests; nested-module
    // enumeration must never rewrite it.
    long long name_epoch = -1;
    uint64_t version = 1;

    Parameter() = default;
    Parameter(Tensor d, std::string n, bool is_trainable = true)
        : data(d),
          name(n),
          base_name(n),
          trainable(is_trainable) {}

    void assign_relative_name(const std::string& relative_name) {
        if (name_frozen_) {
            return;
        }
        if (relative_name.empty()) {
            throw std::invalid_argument("Parameter relative name cannot be empty");
        }
        base_name = relative_name;
        name = relative_name;
    }

    void freeze_name() {
        if (name.empty()) {
            throw std::logic_error("Cannot freeze an unnamed parameter");
        }
        name_frozen_ = true;
    }

    bool name_frozen() const noexcept {
        return name_frozen_;
    }

    // Mutate values without replacing the Tensor object. Preserving storage
    // identity is required for tied parameters (embedding/value head) and
    // captured GPU graphs. The version bump invalidates every derived
    // BitLinear/optimizer cache on its next use.
    void copy_data_from(const Tensor& source) {
        if (data.size == 0) {
            throw std::logic_error(
                "Cannot copy into an uninitialized parameter '" + name + "'");
        }
        if (source.shape != data.shape ||
            source.get_device() != data.get_device()) {
            throw std::invalid_argument(
                "Parameter data copy must preserve shape and device for '" +
                name + "'");
        }
        data.copy_from(source);
        mark_updated();
    }
    
    // Sparse parameters must distinguish allocated storage from a contribution
    // to the current accumulation group. Dense/public legacy parameters retain
    // the historical direct-.grad assignment API until explicitly registered.
    // Enable at construction/registration, never infer activity from values:
    // a contributed all-zero gradient still participates in Adam and decay.
    void track_gradient_contributions() noexcept {
        // add_grad/zero_grad maintain contribution status even before sparse
        // registration. Adopting a previously used dense operator must not
        // revive its cleared buffer merely because storage still exists.
        gradient_contributions_tracked_ = true;
    }

    bool tracks_gradient_contributions() const noexcept {
        return gradient_contributions_tracked_;
    }

    bool has_gradient() const {
        if(device_gradient_.owner && device_gradient_audit_scope) {
#ifdef USE_CUDA
            auto owner=device_gradient_.owner;owner->assert_lane();
            unsigned char contributed=0;int issue=0;
            const auto stream=gpu::current_stream();
            const auto first=cudaMemcpyAsync(&contributed,owner->predicate(device_gradient_.expert),1,cudaMemcpyDeviceToHost,stream);
            const auto second=cudaMemcpyAsync(&issue,owner->issue(),sizeof(int),cudaMemcpyDeviceToHost,stream);
            const auto done=cudaStreamSynchronize(stream);
            if(first!=cudaSuccess || second!=cudaSuccess || done!=cudaSuccess)throw std::runtime_error("Device gradient audit materialization failed");
            record_gpu_transfer(Device::CPU,Device::GPU,1+sizeof(int));record_gpu_stream_synchronization();
            return grad.size>0 && !issue && contributed;
#else
            throw std::logic_error("Device gradient audit requires GPU backend");
#endif
        }
        require_host_gradient_api();
        return grad.size > 0 &&
            (!gradient_contributions_tracked_ || gradient_contributed_);
    }

    // Called once when opening/discarding a group, not between microbatches.
    // The fused zero launcher also calls this without releasing grad storage.
    void reset_gradient_activity() {
        require_host_gradient_api();
        gradient_contributed_ = false;
    }

    // External code writing an existing sparse .grad buffer must publish its
    // contribution explicitly after the write has succeeded.
    void mark_gradient_contribution() {
        require_host_gradient_api();
        if (grad.size != data.size || grad.size == 0 ||
            grad.shape != data.shape || grad.get_device() != data.get_device()) {
            throw std::logic_error("Invalid contributed gradient for '" + name + "'");
        }
        gradient_contributed_ = true;
    }

    // Runtime clones preserve active/inactive status as well as buffer values.
    // Portable checkpoint snapshots intentionally have no gradient storage.
    void copy_gradient_activity_from(const Parameter& source) {
        require_host_gradient_api(); source.require_host_gradient_api();
        gradient_contributions_tracked_ = source.gradient_contributions_tracked_;
        // Preserve the actual contribution bit even for dense legacy sources.
        // has_gradient() deliberately reports their retained storage as active;
        // using it here would revive a cleared dense clone on later registration.
        gradient_contributed_ = grad.size > 0 && source.gradient_contributed_;
    }

    void zero_grad() {
        require_host_gradient_api();
        if (grad.size == 0) {
            reset_gradient_activity();
            return;
        }
#ifdef USE_CUDA
        if (grad.get_device() == Device::GPU) {
            // Async zero on the execution lane: no host sync (raw_data), no
            // full-device drain.  The grad is consumed by backward/optimizer
            // kernels on the same stream (ordered after this); host access goes
            // through data().  The old cudaMemset + cudaDeviceSynchronize ran
            // PER PARAMETER every step -> hundreds of full-device drains/step.
            const cudaError_t status =
                cudaMemsetAsync(grad.raw_data(), 0,
                                static_cast<size_t>(grad.size) * sizeof(float),
                                nsos::gpu::current_stream());
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(NSOS_GPU_BACKEND_NAME) +
                    " gradient zero failed: " + cudaGetErrorString(status));
            }
            reset_gradient_activity();
            return;
        }
#endif
        std::fill_n(grad.data(), grad.size, 0.0f);
        reset_gradient_activity();
    }
    
    void add_grad(const Tensor& g) {
        require_host_gradient_api();
        Tensor incoming = g;
        if (incoming.size == 0) {
            return;
        }
        if (incoming.size == data.size && incoming.shape != data.shape) {
            incoming = incoming.reshape(data.shape.dims);
        }
        if (incoming.get_device() != data.device) {
            incoming = incoming.to(data.device);
        }
        if (incoming.size != data.size || incoming.shape != data.shape) {
            throw std::runtime_error(
                "Gradient shape mismatch for parameter '" + name + "'");
        }
        if (grad.size == 0) {
            // First contribution already is the complete accumulated value.
            // Clone it directly instead of zero-filling a full buffer and then
            // launching a second elementwise add. The allocation remains
            // stable for every subsequent step.
            grad = incoming.clone();
            gradient_contributed_ = true;
            return;
        }
        if (grad.size != data.size || grad.shape != data.shape ||
            grad.get_device() != data.get_device()) {
            throw std::logic_error(
                "Stored gradient invariant violated for parameter '" +
                name + "'");
        }
        // Stable in-place accumulation on both CPU and GPU. This keeps each
        // parameter's grad buffer address constant across the step and removes
        // one temporary Tensor allocation/kernel-output buffer per add_grad.
        if (gradient_contributions_tracked_ && !gradient_contributed_) {
            // The first contribution is the complete value for this group.
            // Copy into stable storage instead of adding to retained zeros.
            grad.copy_from(incoming);
        } else {
            grad.add_inplace_(incoming);
        }
        gradient_contributed_ = true;
    }

    void mark_updated() {
        ++version;
    }

    // Explicit registration reserves storage; it does NOT publish contribution.
    // Legacy host APIs fail closed while bound rather than guessing membership.
    void bind_device_gradient_activity(std::shared_ptr<GpuGradientActivity> owner, int expert) {
        if (!owner || expert < 0 || expert >= owner->experts() ||
            data.get_device() != Device::GPU || grad.shape != data.shape || grad.size != data.size)
            throw std::invalid_argument("Invalid device gradient binding for '" + name + "'");
        if (device_gradient_.owner && (device_gradient_.owner != owner || device_gradient_.expert != expert))
            throw std::logic_error("Parameter already bound to another activity domain");
        device_gradient_ = {std::move(owner), expert};
        gradient_contributions_tracked_ = true; gradient_contributed_ = false;
    }
    const DeviceGradientBinding& device_gradient_binding() const noexcept { return device_gradient_; }
    bool has_device_gradient_activity() const noexcept { return bool(device_gradient_.owner); }
    // Only the group owner may detach after an ordered optimizer/abort boundary.
    void unbind_device_gradient_activity(const std::shared_ptr<GpuGradientActivity>& owner, bool contributed = false) {
        if (device_gradient_.owner != owner) throw std::logic_error("Device activity owner mismatch");
        device_gradient_ = {}; gradient_contributed_ = contributed;
    }
private:
    void require_host_gradient_api() const {
        if (device_gradient_.owner) throw std::logic_error(
            "Parameter '" + name + "' uses device activity; use the explicit device transaction API");
    }
    DeviceGradientBinding device_gradient_;
    bool name_frozen_ = false;
    bool gradient_contributions_tracked_ = false;
    bool gradient_contributed_ = false;
};

} // namespace nsos
