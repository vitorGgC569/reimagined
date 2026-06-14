#include "../include/kan.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "../include/cuda/kan_kernels.cuh"
#endif

namespace nsos {

namespace {

float safe_width(float width) { return std::max(width, 1e-3f); }

} // namespace

BitFastKANLayer::BitFastKANLayer(int in, int out, int grid)
    : input_dim(in),
      output_dim(out),
      grid_size(std::max(grid, 2)),
      base_weight(Tensor::kaiming_uniform({out, in}, Device::CPU), "kan.base_weight"),
      rbf_weight(Tensor::xavier_uniform({out, in * std::max(grid, 2)}, Device::CPU),
                 "kan.rbf_weight"),
      bias(Tensor::zeros({out}, Device::CPU), "kan.bias") {
    centers_.resize(grid_size);
    widths_.resize(grid_size);

    const float step = grid_size > 1 ? 2.0f / static_cast<float>(grid_size - 1) : 2.0f;
    for (int g = 0; g < grid_size; ++g) {
        centers_[g] = -1.0f + static_cast<float>(g) * step;
        widths_[g] = step;
    }
}

Tensor BitFastKANLayer::compute_basis(const Tensor& flat_input) const {
    if (flat_input.shape.size() != 2 || flat_input.shape[1] != input_dim) {
        throw std::runtime_error("BitFastKANLayer basis expects [rows, input_dim]");
    }

    const int rows = flat_input.shape[0];
    Tensor basis({rows, input_dim * grid_size}, flat_input.get_device());

#ifdef USE_CUDA
    if (flat_input.get_device() == Device::GPU) {
        ensure_grid_on_device();
        cuda::launch_kan_rbf_basis_forward(
            flat_input.raw_data(), centers_dev_.raw_data(), widths_dev_.raw_data(),
            basis.raw_data(), rows, input_dim, grid_size);
        return basis;
    }
#endif

    const float* x_ptr = flat_input.data();
    float* basis_ptr = basis.data();

    for (int row = 0; row < rows; ++row) {
        for (int feature = 0; feature < input_dim; ++feature) {
            const float value = x_ptr[row * input_dim + feature];
            for (int g = 0; g < grid_size; ++g) {
                const float width = safe_width(widths_[g]);
                const float diff = (value - centers_[g]) / width;
                basis_ptr[row * input_dim * grid_size + feature * grid_size + g] =
                    std::exp(-0.5f * diff * diff);
            }
        }
    }

    return basis;
}

void BitFastKANLayer::ensure_grid_on_device() const {
#ifdef USE_CUDA
    if (centers_dev_.size == grid_size &&
        centers_dev_.get_device() == Device::GPU &&
        widths_dev_.size == grid_size &&
        widths_dev_.get_device() == Device::GPU) {
        return;
    }
    Tensor c({grid_size}, Device::CPU);
    Tensor w({grid_size}, Device::CPU);
    for (int g = 0; g < grid_size; ++g) {
        c.data()[g] = centers_[g];
        w.data()[g] = widths_[g];
    }
    centers_dev_ = c.to(Device::GPU);
    widths_dev_ = w.to(Device::GPU);
#endif
}

Tensor BitFastKANLayer::forward(const Tensor& x) {
    if (x.shape.size() != 2 && x.shape.size() != 3) {
        throw std::runtime_error("BitFastKANLayer expects rank-2 or rank-3 input");
    }

    std::vector<int> output_shape = x.shape.dims;
    output_shape.back() = output_dim;

    saved_input_ = x.shape.size() == 2 ? x : x.reshape({static_cast<int>(x.size / input_dim), input_dim});
    saved_basis_ = compute_basis(saved_input_);

    Tensor base = saved_input_.matmul(base_weight.data.transpose());
    Tensor enriched = saved_basis_.matmul(rbf_weight.data.transpose());
    Tensor output = base.add(enriched).add(bias.data);
    return x.shape.size() == 2 ? output : output.reshape(output_shape);
}

Tensor BitFastKANLayer::backward(const Tensor& grad) {
    if (saved_input_.size == 0 || saved_basis_.size == 0) {
        throw std::runtime_error("BitFastKANLayer backward called before forward");
    }

    Tensor grad_2d =
        grad.shape.size() == 2 ? grad : grad.reshape({static_cast<int>(grad.size / output_dim), output_dim});
    if (grad_2d.shape[1] != output_dim) {
        throw std::runtime_error("BitFastKANLayer gradient dimension mismatch");
    }

    base_weight.add_grad(grad_2d.transpose().matmul(saved_input_));
    rbf_weight.add_grad(grad_2d.transpose().matmul(saved_basis_));
    bias.add_grad(grad_2d.sum(0));

    Tensor grad_input = grad_2d.matmul(base_weight.data);
    Tensor grad_basis = grad_2d.matmul(rbf_weight.data);

    const int rows = saved_input_.shape[0];

#ifdef USE_CUDA
    if (saved_input_.get_device() == Device::GPU) {
        ensure_grid_on_device();
        // grad_input already holds grad_2d @ base_weight; the kernel adds the
        // RBF term in place.  grad_basis is on GPU (matmul of GPU operands).
        cuda::launch_kan_rbf_basis_backward(
            saved_input_.raw_data(), grad_basis.raw_data(),
            centers_dev_.raw_data(), widths_dev_.raw_data(),
            grad_input.raw_data(), rows, input_dim, grid_size);
        return grad.shape.size() == 2
                   ? grad_input
                   : grad_input.reshape({grad.shape[0], grad.shape[1], input_dim});
    }
#endif

    const float* x_ptr = saved_input_.data();
    const float* grad_basis_ptr = grad_basis.data();
    float* grad_input_ptr = grad_input.data();

    for (int row = 0; row < rows; ++row) {
        for (int feature = 0; feature < input_dim; ++feature) {
            const float value = x_ptr[row * input_dim + feature];
            float rbf_dx = 0.0f;
            for (int g = 0; g < grid_size; ++g) {
                const float width = safe_width(widths_[g]);
                const float diff = (value - centers_[g]) / width;
                const float basis = std::exp(-0.5f * diff * diff);
                const float deriv = basis * (centers_[g] - value) / (width * width);
                const int basis_index =
                    row * input_dim * grid_size + feature * grid_size + g;
                rbf_dx += grad_basis_ptr[basis_index] * deriv;
            }
            grad_input_ptr[row * input_dim + feature] += rbf_dx;
        }
    }

    return grad.shape.size() == 2
               ? grad_input
               : grad_input.reshape({grad.shape[0], grad.shape[1], input_dim});
}

void BitFastKANLayer::to(Device dev) {
    base_weight.data = base_weight.data.to(dev);
    rbf_weight.data = rbf_weight.data.to(dev);
    bias.data = bias.data.to(dev);
    if (base_weight.grad.size > 0)
        base_weight.grad = base_weight.grad.to(dev);
    if (rbf_weight.grad.size > 0)
        rbf_weight.grad = rbf_weight.grad.to(dev);
    if (bias.grad.size > 0)
        bias.grad = bias.grad.to(dev);
    if (saved_input_.size > 0)
        saved_input_ = saved_input_.to(dev);
    if (saved_basis_.size > 0)
        saved_basis_ = saved_basis_.to(dev);
}

std::vector<Parameter*> BitFastKANLayer::parameters() {
    return {&base_weight, &rbf_weight, &bias};
}

} // namespace nsos
