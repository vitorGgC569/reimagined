#include "../include/kan.h"
#include <cmath>
#include <iostream>

namespace nsos {

BitFastKANLayer::BitFastKANLayer(int in, int out, int grid)
    : input_dim(in), output_dim(out), grid_size(grid) {
    base_weight = Tensor::random({out, in});
    rbf_weight = Tensor::random({out, in * grid});
}

Tensor BitFastKANLayer::forward(const Tensor& x) {
    return Tensor::zeros({x.shape[0], output_dim}, x.device);
}

Tensor BitFastKANLayer::backward(const Tensor& grad) {
    return Tensor::zeros({grad.shape[0], input_dim}, grad.device);
}

void BitFastKANLayer::to(Device dev) {
    base_weight = base_weight.to(dev);
    rbf_weight = rbf_weight.to(dev);
}

std::vector<Parameter*> BitFastKANLayer::parameters() {
    return {};
}

} // namespace nsos
