#include "../include/ttt_layer.h"
#include "../include/training_runtime_policy.h"
#include "../include/jamba_utils.h"
#include "../include/nsos_arena.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <limits>

#ifdef USE_CUDA
#include "../include/gpu_backend.h"
#include "../include/cuda/kernels.cuh"
#endif

namespace nsos {

namespace {

int checked_ttt_dimension(int dim, int hidden, float lr) {
    if (dim <= 0 || hidden <= 0 || static_cast<long long>(dim) * hidden > std::numeric_limits<int>::max() ||
        !std::isfinite(lr) || lr < 0)
        throw std::invalid_argument("Invalid TTT dimensions or learning rate");
    return dim;
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
        record_gpu_transfer(dst_device, src_device, bytes);
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
    record_gpu_transfer(dst_device, src_device, bytes);
}

} // namespace

TTTLayer::TTTLayer(int dim, int hidden, float lr, uint64_t seed)
    : dim(checked_ttt_dimension(dim, hidden, lr)),
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
    if (!std::isfinite(learning_rate) || learning_rate < 0 ||
        !std::isfinite(learning_rate * temperature_))
        throw std::invalid_argument("Invalid TTT meta adaptation step");
    clear_backward_state();
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
    return matmul_tn(k, diff).mul(-1.0f);
}

Tensor TTTLayer::forward(const Tensor& x) {
    clear_backward_state();
    Tensor flat = flatten_ttt_input(x, dim);
    const int rows = flat.shape[0];
    const Device device = flat.get_device();
    if (!std::isfinite(learning_rate) || learning_rate < 0 ||
        !std::isfinite(temperature_) || !std::isfinite(friction_) || !std::isfinite(max_grad_norm_))
        throw std::invalid_argument("TTT coefficients must be finite, with nonnegative learning rate");
    if (training_mode_ && training_policy::full_ttt_bptt()) return forward_full(x, flat);

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

    saved_pre_adaptation_ = Tensor::uninitialized({rows, hidden, dim}, device);
    saved_errors_ = Tensor::uninitialized({rows, dim}, device);
    Tensor output = Tensor::uninitialized({rows, dim}, device);

    const float adapt_decay = std::clamp(friction_, 0.0f, 0.999f);
    const float step = learning_rate * std::max(temperature_, 1e-3f);

#ifdef USE_CUDA
    if (device == Device::GPU && training_policy::device_ttt()) {
        Tensor scale = Tensor::uninitialized({1}, device);
        if (!launch_ttt_device_forward(flat.raw_data(), saved_keys_.raw_data(), base.raw_data(),
            adaptation.raw_data(), momentum.raw_data(), saved_pre_adaptation_.raw_data(),
            saved_errors_.raw_data(), output.raw_data(), scale.raw_data(), rows, hidden, dim,
            adapt_decay, step, max_grad_norm_, use_hamiltonian_))
            throw std::runtime_error("TTT device recurrence launch failed");
        saved_device_recurrence_ = true;
    } else
#endif
    for (int row = 0; row < rows; ++row) {
        copy_float_bytes_device_safe(
            saved_pre_adaptation_.raw_data() +
                static_cast<size_t>(row) * static_cast<size_t>(hidden) *
                    static_cast<size_t>(dim),
            saved_pre_adaptation_.get_device(),
            adaptation.raw_data(),
            adaptation.get_device(),
            static_cast<size_t>(hidden) * static_cast<size_t>(dim) * sizeof(float));

        Tensor row_key = saved_keys_.slice(0, row, row + 1);
        Tensor row_base = base.slice(0, row, row + 1);
        Tensor row_input = flat.slice(0, row, row + 1);
        Tensor row_output = row_base.add(row_key.matmul(adaptation));
        Tensor row_error = row_output.sub(row_input);

        copy_float_bytes_device_safe(
            output.raw_data() +
                static_cast<size_t>(row) * static_cast<size_t>(dim),
            output.get_device(),
            row_output.raw_data(),
            row_output.get_device(),
            static_cast<size_t>(dim) * sizeof(float));
        copy_float_bytes_device_safe(
            saved_errors_.raw_data() +
                static_cast<size_t>(row) * static_cast<size_t>(dim),
            saved_errors_.get_device(),
            row_error.raw_data(),
            row_error.get_device(),
            static_cast<size_t>(dim) * sizeof(float));

        const float error_norm = row_error.norm();
        float scale = 1.0f;
        if (max_grad_norm_ > 0.0f) {
            if (error_norm > max_grad_norm_) {
                scale = max_grad_norm_ / (error_norm + 1e-6f);
            }
        }

        Tensor local_grad = matmul_tn(row_key, row_error).mul(scale);
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

// ---------------------------------------------------------------------------
// TRUNCATED BACKWARD — precise scope of what is and is NOT differentiated.
//
// The training forward (above) runs an inner test-time gradient-descent loop:
// for each row r it produces
//     output[r] = w_out(values[r]) + keys[r] @ A_r
// where A_r ("adaptation") is the running test-time state.  A_r is updated
// AFTER output[r] is read, using local_grad = keys[r]^T @ error[r] and the
// momentum recurrence (lines ~207-214).  Critically, A_r therefore depends on
// keys[<r], values[<r] and w_out from EARLIER rows.
//
// This backward computes EXACTLY these terms, all of which are correct:
//   * d/d(values) through base = w_out(values):  grad_values = w_out.backward(g)
//     then grad_from_values = w_v.backward(grad_values).        [exact]
//   * d/d(keys) through the keys[r] @ A_r term, holding A_r fixed:
//     grad_key_row = g[r] @ saved_pre_adaptation_[r]^T, then
//     grad_from_keys = w_k.backward(grad_keys).                 [exact]
//   * grad_input = grad_from_values + grad_from_keys.           [exact sum]
//
// What is DELIBERATELY TRUNCATED (stop-gradient): the dependence of A_r on the
// parameters via the inner update loop.  saved_pre_adaptation_[r] is treated as
// a constant; gradients do NOT flow back through local_grad / the momentum
// recurrence into keys[<r], values[<r] or w_out at earlier rows.  This is a
// first-order / truncated-BPTT-through-the-inner-loop approximation (the same
// stop-gradient family used by first-order MAML and TTT references).  It is a
// known, intentional approximation — NOT a silent identity backward: every term
// that is returned is a real, correct gradient; only the inner-loop sensitivity
// is omitted.  Do not "fix" this by faking the missing terms.
// ---------------------------------------------------------------------------
Tensor TTTLayer::backward(const Tensor& g) {
    if (saved_input_.size == 0 || saved_keys_.size == 0 || saved_values_.size == 0) {
        throw std::runtime_error("TTTLayer backward called before forward");
    }
    if (saved_full_bptt_) return backward_full(g);

    Tensor grad_2d = g.shape.size() == 2 ? g : g.reshape({static_cast<int>(g.size / dim), dim});
    if (grad_2d.shape[1] != dim) {
        throw std::runtime_error("TTTLayer grad output must end with dim");
    }

    const int rows = saved_input_.shape[0];
    // The per-row loop below slices grad_2d at every saved forward row; if the
    // incoming gradient has fewer rows than the saved forward pass, those
    // slices would read out of bounds.  Require an exact row match.
    if (grad_2d.shape[0] != rows) {
        throw std::runtime_error(
            "TTTLayer backward row count does not match the saved forward pass");
    }
    Tensor grad_keys({rows, hidden}, grad_2d.get_device());

#ifdef USE_CUDA
    if (saved_device_recurrence_) {
        if (grad_2d.get_device() != Device::GPU ||
            !launch_ttt_device_key_backward(grad_2d.raw_data(), saved_pre_adaptation_.raw_data(),
                grad_keys.raw_data(), rows, hidden, dim))
            throw std::runtime_error("TTT device key backward launch failed");
    } else
#endif
    for (int row = 0; row < rows; ++row) {
        Tensor pre_adaptation =
            saved_pre_adaptation_.slice(0, row, row + 1).reshape({hidden, dim});
        Tensor grad_row = grad_2d.slice(0, row, row + 1);
        Tensor grad_key_row = matmul_nt(grad_row, pre_adaptation);
        copy_float_bytes_device_safe(
            grad_keys.raw_data() +
                static_cast<size_t>(row) * static_cast<size_t>(hidden),
            grad_keys.get_device(),
            grad_key_row.raw_data(),
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
    clear_backward_state();
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
    if (saved_momentum_boundaries_.size > 0)
        saved_momentum_boundaries_ = saved_momentum_boundaries_.to(d);
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

void TTTLayer::set_training_mode(bool enabled) {
    if (training_mode_ != enabled) clear_backward_state();
    training_mode_ = enabled;
}

void TTTLayer::set_use_hamiltonian(bool enabled) { use_hamiltonian_ = enabled; }
void TTTLayer::set_temperature(float t) {
    if (!std::isfinite(t)) throw std::invalid_argument("TTT temperature must be finite");
    temperature_ = std::max(t, 1e-3f);
}
void TTTLayer::set_friction(float f) {
    if (!std::isfinite(f)) throw std::invalid_argument("TTT friction must be finite");
    friction_ = std::clamp(f, 0.0f, 0.999f);
}
void TTTLayer::set_max_grad_norm(float m) {
    if (!std::isfinite(m)) throw std::invalid_argument("TTT clipping limit must be finite");
    max_grad_norm_ = std::max(m, 0.0f);
}
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
    const std::vector<int> expected{hidden, dim};
    if (snapshot.momentum.shape.dims != expected || snapshot.adaptation.shape.dims != expected)
        throw std::invalid_argument("TTT session snapshot must contain both matching state matrices");
    const Device device = grad_accum_.get_device();
    Tensor momentum = snapshot.momentum.get_device() == device ? snapshot.momentum.clone() : snapshot.momentum.to(device);
    Tensor adaptation = snapshot.adaptation.get_device() == device ? snapshot.adaptation.clone() : snapshot.adaptation.to(device);
    momentum_ = std::move(momentum);
    grad_accum_ = std::move(adaptation);
    clear_backward_state();
}

void TTTLayer::clear_backward_state() {
    saved_input_ = Tensor(); saved_keys_ = Tensor(); saved_values_ = Tensor();
    saved_pre_adaptation_ = Tensor(); saved_errors_ = Tensor(); saved_momentum_boundaries_ = Tensor();
    saved_full_bptt_ = false; saved_device_recurrence_ = false;
    saved_full_batch_ = 0; saved_full_seq_ = 0; saved_full_lengths_.clear();
}

void TTTLayer::set_batch_valid_lengths(const std::vector<int>& lengths) { batch_valid_lengths_ = lengths; }

size_t TTTLayer::saved_state_history_bytes() const noexcept {
    return (static_cast<size_t>(saved_pre_adaptation_.size) + saved_momentum_boundaries_.size) * sizeof(float);
}

Tensor TTTLayer::forward_full(const Tensor& x, const Tensor& flat) {
    const int batch = x.shape.size() == 3 ? x.shape[0] : 1;
    const int seq = x.shape.size() == 3 ? x.shape[1] : flat.shape[0];
    if (batch <= 0 || seq <= 0 || dim <= 0 || hidden <= 0 ||
        !w_k || !w_v || !w_out || w_k->input_features() != dim || w_k->output_features() != hidden ||
        w_v->input_features() != dim || w_v->output_features() != hidden ||
        w_out->input_features() != hidden || w_out->output_features() != dim)
        throw std::invalid_argument("Invalid full TTT layer geometry");
    const int chunks = 1 + (seq - 1) / 32;
    const long long cells = static_cast<long long>(hidden) * dim;
    const long long retained = static_cast<long long>(batch) * chunks * cells;
    if (cells > std::numeric_limits<int>::max() / 32 || retained > std::numeric_limits<int>::max())
        throw std::length_error("Full TTT history exceeds tensor indexing limits");
    std::vector<int> lengths(batch, seq);
    if (!batch_valid_lengths_.empty()) {
        if (batch_valid_lengths_.size() != static_cast<size_t>(batch))
            throw std::invalid_argument("TTT valid lengths must match batch size");
        lengths = batch_valid_lengths_;
        for (int length : lengths) if (length < 0 || length > seq)
            throw std::invalid_argument("TTT valid length outside sequence bounds");
    }
    const Device device = flat.get_device();
    const bool independent_batch = x.shape.size() == 3;
    const std::vector<int> state_shape{hidden, dim};
    if (momentum_.shape.dims != state_shape || grad_accum_.shape.dims != state_shape)
        throw std::invalid_argument("TTT session state geometry was externally changed");
    saved_full_config_ = {hidden, dim, 32, std::clamp(friction_, 0.0f, 0.999f),
                         learning_rate * std::max(temperature_, 1e-3f), max_grad_norm_, use_hamiltonian_};
    if (!std::isfinite(saved_full_config_.step)) throw std::invalid_argument("TTT adaptation step overflow");
    saved_input_ = flat;
    saved_keys_ = w_k->forward(flat);
    saved_values_ = w_v->forward(flat);
    Tensor base = w_out->forward(saved_values_);
    saved_pre_adaptation_ = Tensor::uninitialized({batch * chunks, hidden, dim}, device);
    saved_momentum_boundaries_ = Tensor::uninitialized({batch * chunks, hidden, dim}, device);
    const bool padded = std::any_of(lengths.begin(), lengths.end(), [seq](int n) { return n != seq; });
    saved_errors_ = padded ? Tensor::zeros({batch * seq, dim}, device)
                           : Tensor::uninitialized({batch * seq, dim}, device);
    Tensor output = padded ? Tensor::zeros({batch * seq, dim}, device)
                           : Tensor::uninitialized({batch * seq, dim}, device);
    for (int b = 0; b < batch; ++b) {
        // Rank-3 (including B=1) is independent training data, not sessions.
        // No sample can consume another sample's state or commit shared state.
        Tensor adaptation = independent_batch ? Tensor::zeros(state_shape, device)
            : grad_accum_.get_device() == device ? grad_accum_.clone() : grad_accum_.to(device);
        Tensor momentum = independent_batch ? Tensor::zeros(state_shape, device)
            : momentum_.get_device() == device ? momentum_.clone() : momentum_.to(device);
        ttt::forward_sequence(flat, saved_keys_, base, adaptation, momentum,
            saved_pre_adaptation_, saved_momentum_boundaries_, saved_errors_, output,
            b * seq, lengths[b], b * chunks, saved_full_config_);
        if (!independent_batch) { grad_accum_ = std::move(adaptation); momentum_ = std::move(momentum); }
    }
    saved_full_batch_ = batch; saved_full_seq_ = seq; saved_full_lengths_ = std::move(lengths);
    saved_full_bptt_ = true;
    return output.reshape(x.shape.dims);
}

Tensor TTTLayer::backward_full(const Tensor& g) {
    if (dim != saved_full_config_.dim || hidden != saved_full_config_.hidden ||
        g.get_device() != saved_input_.get_device() ||
        (g.shape.size() != 2 && g.shape.size() != 3) || g.shape.back() != dim ||
        g.size != saved_input_.size ||
        (g.shape.size() == 3 && (g.shape[0] != saved_full_batch_ || g.shape[1] != saved_full_seq_)))
        throw std::invalid_argument("Full TTT gradient does not match the saved forward layout/device");
    Tensor flat_grad = g.reshape(saved_input_.shape.dims);
    const int rows = saved_input_.shape[0];
    const Device device = g.get_device();
    const bool padded = std::any_of(saved_full_lengths_.begin(), saved_full_lengths_.end(),
                                   [this](int n) { return n != saved_full_seq_; });
    Tensor keys_grad = padded ? Tensor::zeros({rows, hidden}, device) : Tensor::uninitialized({rows, hidden}, device);
    Tensor base_grad = padded ? Tensor::zeros({rows, dim}, device) : Tensor::uninitialized({rows, dim}, device);
    Tensor direct_grad = padded ? Tensor::zeros({rows, dim}, device) : Tensor::uninitialized({rows, dim}, device);
    const int chunks = 1 + (saved_full_seq_ - 1) / saved_full_config_.chunk;
    for (int b = 0; b < saved_full_batch_; ++b)
        ttt::backward_sequence(saved_keys_, saved_errors_, saved_pre_adaptation_, saved_momentum_boundaries_,
            flat_grad, keys_grad, base_grad, direct_grad,
            b * saved_full_seq_, saved_full_lengths_[b], b * chunks, saved_full_config_);
    Tensor values_grad = w_out->backward(base_grad);
    Tensor input_grad = w_v->backward(values_grad).add(w_k->backward(keys_grad)).add(direct_grad);
    clear_backward_state();
    return input_grad.reshape(g.shape.dims);
}

} // namespace nsos
