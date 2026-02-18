#pragma once
#include "tensor.h"
#include "nsos_context.h"
#include <vector>
#include <memory>

namespace nsos {

// Unified Context using nsos_context.h logic
// No redefinition here.

class Parameter {
public:
    Tensor data;
    Tensor grad;
    std::string name;
    std::string base_name;

    Parameter() = default;
    Parameter(Tensor d, std::string n) : data(d), name(n), base_name(n) {}
    
    void zero_grad() {
        if (grad.size > 0) std::fill_n(grad.data(), grad.size, 0.0f);
    }
    
    void add_grad(const Tensor& g) {
        if (grad.size == 0) grad = Tensor::zeros(data.shape.dims, data.device);
        Tensor new_grad = grad.add(g);
        grad.copy_from(new_grad);
    }
};

} // namespace nsos
