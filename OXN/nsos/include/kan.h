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

    // N5: when true, the base/RBF weight matmuls use ternary fake-quant
    // (straight-through) so the KAN FFN honors the 1.58-bit invariant like every
    // other BitLinear in the model — instead of full-precision matmuls that
    // broke head-to-toe quantization at the edge.  Off in the bare ctor (keeps
    // the FD gradcheck on the float path); JambaBlock turns it on for the model.
    void set_quantized(bool enabled) { quantized_ = enabled; }
    bool quantized() const { return quantized_; }

private:
    bool quantized_ = false;
    float base_scale_ = 1.0f;
    float rbf_scale_ = 1.0f;
    Tensor saved_base_eff_;   // ternary-dequant base_weight used by the forward
    Tensor saved_rbf_eff_;    // ternary-dequant rbf_weight used by the forward
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
