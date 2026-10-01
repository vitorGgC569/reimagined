#include "../include/kan.h"
#include "../include/bitnet_gpu_dispatch.h"  // N5: qat_fake_quant_ternary / STE clip

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "training_runtime_policy.h"
#include "gpu_execution.h"

#ifdef USE_CUDA
#include "../include/cuda/kan_kernels.cuh"
#include "cuda/moe_training_wmma.cuh"
#endif

namespace nsos {

namespace {

float safe_width(float width) { return std::max(width, 1e-3f); }
std::vector<int> checked_kan_weight_shape(int in,int out,int grid,bool rbf) {
    const long long features=static_cast<long long>(in)*std::max(grid,2);
    if (in<=0 || out<=0 || features>std::numeric_limits<int>::max()-15 ||
        features*out>std::numeric_limits<int>::max())
        throw std::invalid_argument("KAN weight geometry exceeds positive int indexing");
    return {out,rbf?static_cast<int>(features):in};
}

} // namespace

BitFastKANLayer::BitFastKANLayer(int in, int out, int grid)
    : input_dim(in),
      output_dim(out),
      grid_size(std::max(grid, 2)),
      base_weight(Tensor::kaiming_uniform(checked_kan_weight_shape(in,out,grid,false), Device::CPU), "kan.base_weight"),
      rbf_weight(Tensor::xavier_uniform(checked_kan_weight_shape(in,out,grid,true), Device::CPU),
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
    if (static_cast<long long>(rows)*input_dim*grid_size>std::numeric_limits<int>::max())
        throw std::length_error("KAN materialized basis exceeds int indexing");
    Tensor basis=Tensor::uninitialized({rows, input_dim * grid_size}, flat_input.get_device());

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
    clear_tape();
    if (x.shape.size() != 2 && x.shape.size() != 3) {
        throw std::runtime_error("BitFastKANLayer expects rank-2 or rank-3 input");
    }
    if (x.shape.back()!=input_dim || x.size<=0 || grid_size!=static_cast<int>(centers_.size()) ||
        grid_size!=static_cast<int>(widths_.size()) ||
        base_weight.data.shape.dims!=std::vector<int>{output_dim,input_dim} ||
        rbf_weight.data.shape.dims!=std::vector<int>{output_dim,input_dim*grid_size} ||
        bias.data.shape.dims!=std::vector<int>{output_dim} ||
        x.get_device()!=base_weight.data.get_device() || x.get_device()!=rbf_weight.data.get_device() ||
        x.get_device()!=bias.data.get_device())
        throw std::invalid_argument("KAN input/parameter geometry or device mismatch");
    saved_recompute_=training_policy::kan_recompute_training();
    saved_wmma_=training_policy::kan_wmma_training();
    if (saved_wmma_ && !saved_recompute_)
        throw std::runtime_error("NSOS_KAN_WMMA_TRAINING requires NSOS_KAN_RECOMPUTE_TRAINING");
    if (saved_recompute_ && x.get_device()!=Device::GPU)
        throw std::runtime_error("NSOS_KAN_RECOMPUTE_TRAINING requires GPU execution");
    saved_precision_=matmul_precision_mode();
#ifdef USE_CUDA
    if (saved_wmma_ && !moe_training_wmma_supported())
        throw std::runtime_error("KAN WMMA requires compiled RDNA3 wave32 rocWMMA support");
#endif
    const int rows=x.size/input_dim;
    if (saved_recompute_ && (rows>65535*16 || output_dim>65535*16))
        throw std::length_error("KAN recompute tile geometry exceeds device grid limits");

    std::vector<int> output_shape = x.shape.dims;
    output_shape.back() = output_dim;

    saved_input_ = x.shape.size() == 2 ? x : x.reshape({static_cast<int>(x.size / input_dim), input_dim});
    if (saved_recompute_) saved_input_=saved_input_.clone(); // tape owns immutable inputs
    else saved_basis_ = compute_basis(saved_input_);

    // N5: optional ternary fake-quant (STE) of the base/RBF weights so the KAN
    // FFN is 1.58-bit like the rest of the model.  Device-aware; off → exact
    // float path (unchanged, FD-gradcheckable).
    Tensor bw = base_weight.data;
    Tensor rw = rbf_weight.data;
    if (quantized_) {
#ifdef USE_CUDA
        if (saved_recompute_) {
            auto prepare=[&](const Tensor& latent,Tensor& effective,Tensor& scale,Tensor& partials) {
                effective=Tensor::uninitialized(latent.shape.dims,Device::GPU);
                scale=Tensor::uninitialized({1},Device::GPU);
                const int blocks=static_cast<int>(std::min<int64_t>(4096,(latent.size-1)/256+1));
                partials=Tensor::uninitialized({blocks},Device::GPU);
                if (!cuda::launch_kan_prepare_ternary(latent.raw_data(),effective.raw_data(),scale.raw_data(),
                    partials.raw_data(),latent.size)) throw std::runtime_error("KAN device QAT launch failed");
            };
            prepare(bw,saved_base_eff_,base_scale_dev_,base_partials_);
            prepare(rw,saved_rbf_eff_,rbf_scale_dev_,rbf_partials_);
            bw=saved_base_eff_; rw=saved_rbf_eff_;
        } else
#endif
        {
        base_scale_ = tensor_abs_mean(bw) + 1e-8f;  // absmean (BitNet b1.58)
        rbf_scale_ = tensor_abs_mean(rw) + 1e-8f;
        bw = qat_fake_quant_ternary(bw, base_scale_);
        rw = qat_fake_quant_ternary(rw, rbf_scale_);
        saved_base_eff_ = bw;
        saved_rbf_eff_ = rw;
        }
    }
    Tensor base = matmul_nt(saved_input_, bw);
    Tensor output;
#ifdef USE_CUDA
    if (saved_recompute_) {
        ensure_grid_on_device();
        output=Tensor::uninitialized({rows,output_dim},Device::GPU);
        if (!cuda::launch_kan_rbf_projection(saved_input_.raw_data(),rw.raw_data(),centers_dev_.raw_data(),
            widths_dev_.raw_data(),base.raw_data(),bias.data.raw_data(),output.raw_data(),
            rows,input_dim,output_dim,grid_size,saved_precision_,saved_wmma_)) throw std::runtime_error("KAN implicit RBF projection launch failed");
        gpu::record_dispatch(gpu::DispatchPath::KanRecompute);
    } else
#endif
    {
        Tensor enriched = matmul_nt(saved_basis_, rw);
        output=base.add(enriched).add(bias.data);
    }
    saved_output_shape_=output_shape;
    saved_geometry_={input_dim,output_dim,grid_size};
    saved_versions_={base_weight.version,rbf_weight.version,bias.version};
    saved_addresses_={base_weight.data.raw_data(),rbf_weight.data.raw_data(),bias.data.raw_data()};
    saved_quantized_=quantized_; pending_=true;
    return x.shape.size() == 2 ? output : output.reshape(output_shape);
}

Tensor BitFastKANLayer::backward(const Tensor& grad) {
    if (!pending_ || saved_input_.size == 0) {
        throw std::runtime_error("BitFastKANLayer backward called before forward");
    }
    if (saved_quantized_!=quantized_ || saved_recompute_!=training_policy::kan_recompute_training() ||
        saved_wmma_!=training_policy::kan_wmma_training() ||
        saved_precision_!=matmul_precision_mode() ||
        saved_geometry_!=std::array<int,3>{input_dim,output_dim,grid_size} ||
        base_weight.data.shape.dims!=std::vector<int>{output_dim,input_dim} ||
        rbf_weight.data.shape.dims!=std::vector<int>{output_dim,input_dim*grid_size} ||
        bias.data.shape.dims!=std::vector<int>{output_dim} ||
        base_weight.data.get_device()!=saved_input_.get_device() ||
        rbf_weight.data.get_device()!=saved_input_.get_device() ||
        bias.data.get_device()!=saved_input_.get_device() ||
        saved_versions_!=std::array<uint64_t,3>{base_weight.version,rbf_weight.version,bias.version} ||
        saved_addresses_!=std::array<const float*,3>{base_weight.data.raw_data(),rbf_weight.data.raw_data(),bias.data.raw_data()})
        throw std::runtime_error("KAN parameters/compute policy changed between forward and backward");
    if (grad.shape.dims!=saved_output_shape_ || grad.get_device()!=saved_input_.get_device())
        throw std::invalid_argument("KAN backward must match forward shape/device");

    Tensor grad_2d =
        grad.shape.size() == 2 ? grad : grad.reshape({static_cast<int>(grad.size / output_dim), output_dim});
    if (grad_2d.shape[1] != output_dim) {
        throw std::runtime_error("BitFastKANLayer gradient dimension mismatch");
    }

    // N5: weight grads (STE through the ternary fake-quant — identical formula,
    // assigned onto the FP32 latent weights); STE-clip saturated entries.
    Tensor dbase = matmul_tn(grad_2d, saved_input_);
    Tensor drbf;
#ifdef USE_CUDA
    if (saved_recompute_) {
        drbf=Tensor::uninitialized(rbf_weight.data.shape.dims,Device::GPU);
        if (!cuda::launch_kan_rbf_weight_backward(saved_input_.raw_data(),grad_2d.raw_data(),centers_dev_.raw_data(),
            widths_dev_.raw_data(),drbf.raw_data(),saved_input_.shape[0],input_dim,output_dim,grid_size,saved_precision_,saved_wmma_))
            throw std::runtime_error("KAN recomputed weight backward launch failed");
    } else
#endif
    drbf=matmul_tn(grad_2d,saved_basis_);
    if (quantized_) {
        if (saved_recompute_) {
            qat_ste_clip_weight_grad_device_scale(dbase,base_weight.data,base_scale_dev_);
            qat_ste_clip_weight_grad_device_scale(drbf,rbf_weight.data,rbf_scale_dev_);
        } else {
        qat_ste_clip_weight_grad(dbase, base_weight.data, base_scale_);
        qat_ste_clip_weight_grad(drbf, rbf_weight.data, rbf_scale_);
        }
    }
    base_weight.add_grad(dbase);
    rbf_weight.add_grad(drbf);
    bias.add_grad(grad_2d.sum(0));

    // Input/basis grads flow through the EFFECTIVE (fake-quantized) weights the
    // forward actually multiplied when quantization is on.
    const Tensor& base_for_dx =
        (quantized_ && saved_base_eff_.size > 0) ? saved_base_eff_ : base_weight.data;
    const Tensor& rbf_for_dx =
        (quantized_ && saved_rbf_eff_.size > 0) ? saved_rbf_eff_ : rbf_weight.data;
    Tensor grad_input = grad_2d.matmul(base_for_dx);
    Tensor grad_basis;
    if (!saved_recompute_) grad_basis=grad_2d.matmul(rbf_for_dx);

    const int rows = saved_input_.shape[0];

#ifdef USE_CUDA
    if (saved_input_.get_device() == Device::GPU) {
        ensure_grid_on_device();
        // grad_input already holds grad_2d @ base_weight; the kernel adds the
        // RBF term in place.  grad_basis is on GPU (matmul of GPU operands).
        if (saved_recompute_) {
            if (!cuda::launch_kan_rbf_input_backward(saved_input_.raw_data(),grad_2d.raw_data(),rbf_for_dx.raw_data(),
                centers_dev_.raw_data(),widths_dev_.raw_data(),grad_input.raw_data(),rows,input_dim,output_dim,grid_size,saved_precision_,saved_wmma_))
                throw std::runtime_error("KAN recomputed input backward launch failed");
        } else cuda::launch_kan_rbf_basis_backward(
            saved_input_.raw_data(), grad_basis.raw_data(),
            centers_dev_.raw_data(), widths_dev_.raw_data(),
            grad_input.raw_data(), rows, input_dim, grid_size);
        clear_tape();
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

    clear_tape();
    return grad.shape.size() == 2
               ? grad_input
               : grad_input.reshape({grad.shape[0], grad.shape[1], input_dim});
}

void BitFastKANLayer::to(Device dev) {
    clear_tape();
    base_weight.data = base_weight.data.to(dev);
    rbf_weight.data = rbf_weight.data.to(dev);
    bias.data = bias.data.to(dev);
    if (base_weight.grad.size > 0)
        base_weight.grad = base_weight.grad.to(dev);
    if (rbf_weight.grad.size > 0)
        rbf_weight.grad = rbf_weight.grad.to(dev);
    if (bias.grad.size > 0)
        bias.grad = bias.grad.to(dev);
}

void BitFastKANLayer::clear_tape() {
    pending_=false;
    saved_input_=Tensor(); saved_basis_=Tensor();
    saved_base_eff_=Tensor(); saved_rbf_eff_=Tensor();
    base_scale_dev_=Tensor(); rbf_scale_dev_=Tensor();
    base_partials_=Tensor(); rbf_partials_=Tensor();
    saved_output_shape_.clear();
}

std::vector<Parameter*> BitFastKANLayer::parameters() {
    return {&base_weight, &rbf_weight, &bias};
}

} // namespace nsos
