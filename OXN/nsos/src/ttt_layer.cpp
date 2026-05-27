#include "../include/ttt_layer.h"
#include "../include/nsos_arena.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

namespace {

void prefix_parameter_names(std::vector<Parameter*>& params, const std::string& prefix) {
    for (auto* param : params) {
        if (!param) {
            continue;
        }
        const std::string current = param->name.empty() ? param->base_name : param->name;
        const std::string leaf = param->base_name.empty() ? current : param->base_name;
        param->base_name = leaf;
        param->name =
            (current.find('.') == std::string::npos) ? (prefix + leaf) : (prefix + current);
    }
}

Tensor flatten_ttt_input(const Tensor& x, int dim) {
    if (x.shape.size() == 2 && x.shape[1] == dim) {
        return x;
    }
    if (x.shape.size() == 3 && x.shape[2] == dim) {
        return x.reshape({x.shape[0] * x.shape[1], dim});
    }
    throw std::runtime_error("TTTLayer expects rank-2 or rank-3 input with last dim == dim");
}

float clamp_value(float value, float max_abs) {
    if (max_abs <= 0.0f) {
        return value;
    }
    return std::max(-max_abs, std::min(max_abs, value));
}

void copy_float_bytes_device_safe(float* dst,
                                  Device dst_device,
                                  const float* src,
                                  Device src_device,
                                  size_t bytes) {
    if (bytes == 0 || dst == src) {
        return;
    }
#ifdef USE_CUDA
    if (dst_device == Device::GPU || src_device == Device::GPU) {
        cudaMemcpyKind kind = cudaMemcpyDefault;
        if (dst_device == Device::GPU && src_device == Device::GPU) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (dst_device == Device::GPU && src_device == Device::CPU) {
            kind = cudaMemcpyHostToDevice;
        } else if (dst_device == Device::CPU && src_device == Device::GPU) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyHostToHost;
        }
        const cudaError_t status = cudaMemcpy(dst, src, bytes, kind);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("TTTLayer cudaMemcpy failed: ") +
                                     cudaGetErrorString(status));
        }
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

} // namespace

TTTLayer::TTTLayer(int dim, int hidden, float lr, uint64_t seed)
    : dim(dim),
      hidden(hidden),
      learning_rate(lr),
      // Seed propagation: each BitLinear gets seed+offset for distinct init
      // but reproducible across instances.  seed=0 falls back to un-seeded.
      w_k(std::make_unique<BitLinear>(dim, hidden, true,
                                      seed == 0 ? 0u : seed + 1u)),
      w_v(std::make_unique<BitLinear>(dim, hidden, true,
                                      seed == 0 ? 0u : seed + 2u)),
      w_out(std::make_unique<BitLinear>(hidden, dim, true,
                                        seed == 0 ? 0u : seed + 3u)),
      state_(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed),
      cached_gaussian_(0.0f) {
    momentum_ = Tensor::zeros({hidden, dim}, Device::CPU);
    grad_accum_ = Tensor::zeros({hidden, dim}, Device::CPU);
}

void TTTLayer::initialize_from_meta(const Tensor& x, const Tensor& y) {
    Tensor flat_x = flatten_ttt_input(x, dim);
    Tensor flat_y = flatten_ttt_input(y, dim);
    const Device target_device = flat_x.get_device();
    Tensor force = compute_force(flat_x, flat_y);
    if (max_grad_norm_ > 0.0f) {
        force = force.clamp(-max_grad_norm_, max_grad_norm_);
    }

    Tensor momentum =
        momentum_.get_device() == target_device ? momentum_ : momentum_.to(target_device);
    Tensor adaptation =
        grad_accum_.get_device() == target_device ? grad_accum_ : grad_accum_.to(target_device);

    const float step = learning_rate * std::max(temperature_, 1e-3f);
    const float decay = std::clamp(friction_, 0.0f, 0.999f);
    if (use_hamiltonian_) {
        momentum = momentum.mul(decay).add(force.mul(1.0f - decay));
        adaptation = adaptation.sub(force.add(momentum.mul(0.5f)).mul(step));
    } else {
        momentum = momentum.mul(decay).add(force);
        adaptation = adaptation.mul(decay).sub(force.mul(step));
    }

    momentum_ =
        momentum_.get_device() == target_device ? momentum : momentum.to(momentum_.get_device());
    grad_accum_ = grad_accum_.get_device() == target_device ? adaptation
                                                            : adaptation.to(grad_accum_.get_device());
}

Tensor TTTLayer::compute_force(const Tensor& x, const Tensor& target) {
    Tensor k = w_k->forward(x);
    Tensor v = w_v->forward(x);
    Tensor pred = w_out->forward(v);
    Tensor diff = pred.sub(target);
    return k.transpose().matmul(diff).mul(-1.0f);
}

Tensor TTTLayer::forward(const Tensor& x) {
    Tensor flat = flatten_ttt_input(x, dim);
    const int rows = flat.shape[0];
    const Device device = flat.get_device();

    if (!training_mode_) {
        Tensor keys = w_k->forward(flat);
        Tensor values = w_v->forward(flat);
        Tensor base = w_out->forward(values);
        Tensor adaptation =
            grad_accum_.get_device() == flat.get_device() ? grad_accum_ : grad_accum_.to(flat.get_device());
        Tensor correction = keys.matmul(adaptation);
        Tensor output = base.add(correction);
        saved_input_ = Tensor();
        saved_keys_ = Tensor();
        saved_values_ = Tensor();
        saved_pre_adaptation_ = Tensor();
        saved_errors_ = Tensor();
        if (x.shape.size() == 2) {
            return output;
        }
        return output.reshape({x.shape[0], x.shape[1], dim});
    }

    saved_input_ = flat;
    saved_keys_ = w_k->forward(flat);
    saved_values_ = w_v->forward(flat);
    Tensor base = w_out->forward(saved_values_);
    Tensor adaptation =
        grad_accum_.get_device() == device ? grad_accum_.clone() : grad_accum_.to(device);
    Tensor momentum =
        momentum_.get_device() == device ? momentum_.clone() : momentum_.to(device);

    saved_pre_adaptation_ = Tensor::zeros({rows, hidden, dim}, device);
    saved_errors_ = Tensor::zeros({rows, dim}, device);
    Tensor output({rows, dim}, device);

    const float adapt_decay = std::clamp(friction_, 0.0f, 0.999f);
    const float step = learning_rate * std::max(temperature_, 1e-3f);

    for (int row = 0; row < rows; ++row) {
        copy_float_bytes_device_safe(
            saved_pre_adaptation_.data() + static_cast<size_t>(row) * static_cast<size_t>(hidden) *
                                               static_cast<size_t>(dim),
            saved_pre_adaptation_.get_device(),
            adaptation.data(),
            adaptation.get_device(),
            static_cast<size_t>(hidden) * static_cast<size_t>(dim) * sizeof(float));

        Tensor row_key = saved_keys_.slice(0, row, row + 1);
        Tensor row_base = base.slice(0, row, row + 1);
        Tensor row_input = flat.slice(0, row, row + 1);
        Tensor row_output = row_base.add(row_key.matmul(adaptation));
        Tensor row_error = row_output.sub(row_input);

        copy_float_bytes_device_safe(
            output.data() + static_cast<size_t>(row) * static_cast<size_t>(dim),
            output.get_device(),
            row_output.data(),
            row_output.get_device(),
            static_cast<size_t>(dim) * sizeof(float));
        copy_float_bytes_device_safe(
            saved_errors_.data() + static_cast<size_t>(row) * static_cast<size_t>(dim),
            saved_errors_.get_device(),
            row_error.data(),
            row_error.get_device(),
            static_cast<size_t>(dim) * sizeof(float));

        const float error_norm = row_error.norm();
        float scale = 1.0f;
        if (max_grad_norm_ > 0.0f) {
            if (error_norm > max_grad_norm_) {
                scale = max_grad_norm_ / (error_norm + 1e-6f);
            }
        }

        Tensor local_grad = row_key.transpose().matmul(row_error).mul(scale);
        if (use_hamiltonian_) {
            momentum = momentum.mul(adapt_decay).add(local_grad.mul(1.0f - adapt_decay));
            adaptation = adaptation.sub(local_grad.add(momentum.mul(0.25f)).mul(step));
        } else {
            momentum = momentum.mul(adapt_decay).add(local_grad);
            adaptation = adaptation.mul(adapt_decay).sub(local_grad.mul(step));
        }
    }

    momentum_ = momentum_.get_device() == device ? momentum : momentum.to(momentum_.get_device());
    grad_accum_ =
        grad_accum_.get_device() == device ? adaptation : adaptation.to(grad_accum_.get_device());

    if (x.shape.size() == 2) {
        return output;
    }
    return output.reshape({x.shape[0], x.shape[1], dim});
}

Tensor TTTLayer::backward(const Tensor& g) {
    if (saved_input_.size == 0 || saved_keys_.size == 0 || saved_values_.size == 0) {
        throw std::runtime_error("TTTLayer backward called before forward");
    }

    Tensor grad_2d = g.shape.size() == 2 ? g : g.reshape({g.size / dim, dim});
    if (grad_2d.shape[1] != dim) {
        throw std::runtime_error("TTTLayer grad output must end with dim");
    }

    const int rows = saved_input_.shape[0];
    Tensor grad_keys({rows, hidden}, grad_2d.get_device());

    for (int row = 0; row < rows; ++row) {
        Tensor pre_adaptation =
            saved_pre_adaptation_.slice(0, row, row + 1).reshape({hidden, dim});
        Tensor grad_row = grad_2d.slice(0, row, row + 1);
        Tensor grad_key_row = grad_row.matmul(pre_adaptation.transpose());
        copy_float_bytes_device_safe(
            grad_keys.data() + static_cast<size_t>(row) * static_cast<size_t>(hidden),
            grad_keys.get_device(),
            grad_key_row.data(),
            grad_key_row.get_device(),
            static_cast<size_t>(hidden) * sizeof(float));
    }

    Tensor grad_values = w_out->backward(grad_2d);
    Tensor grad_from_values = w_v->backward(grad_values);
    Tensor grad_from_keys = w_k->backward(grad_keys);
    Tensor grad_input = grad_from_values.add(grad_from_keys);

    return g.shape.size() == 2 ? grad_input
                               : grad_input.reshape({g.shape[0], g.shape[1], dim});
}

void TTTLayer::reset() {
    const Device device = grad_accum_.get_device();
    momentum_ = Tensor::zeros({hidden, dim}, device);
    grad_accum_ = Tensor::zeros({hidden, dim}, device);
    saved_input_ = Tensor();
    saved_keys_ = Tensor();
    saved_values_ = Tensor();
    saved_pre_adaptation_ = Tensor();
    saved_errors_ = Tensor();
}

void TTTLayer::to(Device d) {
    w_k->to(d);
    w_v->to(d);
    w_out->to(d);
    momentum_ = momentum_.to(d);
    grad_accum_ = grad_accum_.to(d);
    if (saved_input_.size > 0)
        saved_input_ = saved_input_.to(d);
    if (saved_keys_.size > 0)
        saved_keys_ = saved_keys_.to(d);
    if (saved_values_.size > 0)
        saved_values_ = saved_values_.to(d);
    if (saved_pre_adaptation_.size > 0)
        saved_pre_adaptation_ = saved_pre_adaptation_.to(d);
    if (saved_errors_.size > 0)
        saved_errors_ = saved_errors_.to(d);
}

std::vector<Parameter*> TTTLayer::parameters() {
    std::vector<Parameter*> res;
    auto p1 = w_k->parameters();
    prefix_parameter_names(p1, "w_k.");
    res.insert(res.end(), p1.begin(), p1.end());
    auto p2 = w_v->parameters();
    prefix_parameter_names(p2, "w_v.");
    res.insert(res.end(), p2.begin(), p2.end());
    auto p3 = w_out->parameters();
    prefix_parameter_names(p3, "w_out.");
    res.insert(res.end(), p3.begin(), p3.end());
    return res;
}

void TTTLayer::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (w_k) out.push_back(w_k.get());
    if (w_v) out.push_back(w_v.get());
    if (w_out) out.push_back(w_out.get());
}

void TTTLayer::set_training_mode(bool enabled) { training_mode_ = enabled; }

void TTTLayer::set_use_hamiltonian(bool enabled) { use_hamiltonian_ = enabled; }
void TTTLayer::set_temperature(float t) { temperature_ = std::max(t, 1e-3f); }
void TTTLayer::set_friction(float f) { friction_ = std::clamp(f, 0.0f, 0.999f); }
void TTTLayer::set_max_grad_norm(float m) { max_grad_norm_ = std::max(m, 0.0f); }
void TTTLayer::set_checkpoint_interval(int i) { checkpoint_interval_ = std::max(i, 1); }
Tensor TTTLayer::get_current_adaptation() { return grad_accum_; }

TTTSessionSnapshot TTTLayer::snapshot_state() const {
    TTTSessionSnapshot snapshot;
    snapshot.has_state = momentum_.size > 0 || grad_accum_.size > 0;
    snapshot.momentum = momentum_.size > 0 ? momentum_.clone() : Tensor();
    snapshot.adaptation = grad_accum_.size > 0 ? grad_accum_.clone() : Tensor();
    return snapshot;
}

void TTTLayer::restore_state(const TTTSessionSnapshot& snapshot) {
    if (!snapshot.has_state) {
        reset();
        return;
    }
    if (snapshot.momentum.size > 0) {
        momentum_ = snapshot.momentum.clone();
    }
    if (snapshot.adaptation.size > 0) {
        grad_accum_ = snapshot.adaptation.clone();
    }
}

} // namespace nsos
