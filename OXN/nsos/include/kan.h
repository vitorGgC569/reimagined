#pragma once
#include "autograd.h"
#include <vector>
#include <memory>

namespace nsos {

class BitFastKANLayer {
public:
    int input_dim;
    int output_dim;
    int grid_size;
    Tensor base_weight;
    Tensor rbf_weight;

    BitFastKANLayer(int in, int out, int grid = 5);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad);
    void to(Device dev);
    std::vector<Parameter*> parameters();
};

} // namespace nsos
