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
    Parameter base_weight;
    Parameter rbf_weight;
    Parameter bias;

    BitFastKANLayer(int in, int out, int grid = 5);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad);
    void to(Device dev);
    std::vector<Parameter*> parameters();

private:
    Tensor saved_input_;
    Tensor saved_basis_;
    std::vector<float> centers_;
    std::vector<float> widths_;
    // Device copies of the (fixed) RBF grid, lazily uploaded on first GPU use
    // so the CUDA basis kernels can read them.  Mutable: populated from the
    // const compute_basis path.
    mutable Tensor centers_dev_;
    mutable Tensor widths_dev_;

    Tensor compute_basis(const Tensor& flat_input) const;
    void ensure_grid_on_device() const;
};

} // namespace nsos
