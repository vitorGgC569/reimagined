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
            cudaMemset(grad.data(), 0, static_cast<size_t>(grad.size) * sizeof(float));
            cudaDeviceSynchronize();
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
        Tensor new_grad = grad.add(incoming);
        grad.copy_from(new_grad);
    }

    void mark_updated() {
        ++version;
    }
};

} // namespace nsos
