#pragma once
#include "tensor.h"
#include "nsos_context.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace nsos {

// Unified Context using nsos_context.h logic
// No redefinition here.

class Parameter {
public:
    Tensor data;
    Tensor grad;
    std::string name;
    std::string base_name;
    uint64_t version = 1;

    Parameter() = default;
    Parameter(Tensor d, std::string n) : data(d), name(n), base_name(n) {}
    
    void zero_grad() {
        if (grad.size == 0) return;
#ifdef USE_CUDA
        if (grad.get_device() == Device::GPU) {
            // Async zero on the default stream: no host sync (raw_data), no
            // full-device drain.  The grad is consumed by backward/optimizer
            // kernels on the same stream (ordered after this); host access goes
            // through data().  The old cudaMemset + cudaDeviceSynchronize ran
            // PER PARAMETER every step -> hundreds of full-device drains/step.
            cudaMemsetAsync(grad.raw_data(), 0,
                            static_cast<size_t>(grad.size) * sizeof(float), 0);
            return;
        }
#endif
        std::fill_n(grad.data(), grad.size, 0.0f);
    }
    
    void add_grad(const Tensor& g) {
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
        if (grad.size == 0) grad = Tensor::zeros(data.shape.dims, data.device);
        if (incoming.shape != grad.shape) {
            if (incoming.size == grad.size) {
                incoming = incoming.reshape(grad.shape.dims);
            } else {
                throw std::runtime_error("Gradient shape mismatch for parameter '" + name + "'");
            }
        }
        // In-place accumulation (no per-call allocation) on CPU; GPU falls back
        // to the tensor add path which handles device memory.  This is on the
        // optimizer hot path (every parameter, every backward).
        if (grad.get_device() == Device::CPU &&
            incoming.get_device() == Device::CPU && grad.size == incoming.size) {
            float* gp = grad.data();
            const float* ip = incoming.data();
            const int n = grad.size;
            for (int i = 0; i < n; ++i) gp[i] += ip[i];
        } else {
            // Reassign instead of copy_from: grad.add() already produced the
            // summed tensor on-device, so pointing grad at it avoids an extra
            // GPU->GPU copy + stream sync per parameter on every backward.  The
            // old buffer returns to the caching pool; the optimizer keys its
            // m/v state on the Parameter*, not the grad buffer address, so the
            // reassignment is safe.
            grad = grad.add(incoming);
        }
    }

    void mark_updated() {
        ++version;
    }
};

} // namespace nsos
