#include "../include/mamba2.h"
#include "../include/cuda/mamba_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

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

bool can_use_gpu_mamba_scan(const Tensor& x, const Tensor& delta,
                            const Tensor& a_data) {
#ifdef USE_CUDA
    return x.get_device() == Device::GPU &&
           delta.get_device() == Device::GPU &&
           a_data.get_device() == Device::GPU &&
           x.size > 0 &&
           delta.size == x.size &&
           a_data.size == x.shape.back();
#else
    (void)x;
    (void)delta;
    (void)a_data;
    return false;
#endif
}

bool can_use_gpu_mamba_single_token(const Tensor& x,
                                    const Tensor& delta,
                                    const Tensor& a_data,
                                    const Tensor& state,
                                    const Tensor& output) {
#ifdef USE_CUDA
    return x.get_device() == Device::GPU &&
           delta.get_device() == Device::GPU &&
           a_data.get_device() == Device::GPU &&
           state.get_device() == Device::GPU &&
           output.get_device() == Device::GPU &&
           x.size > 0 &&
           delta.size == x.size &&
           state.size == x.size &&
           output.size == x.size &&
           a_data.size == x.shape.back();
#else
    (void)x;
    (void)delta;
    (void)a_data;
    (void)state;
    (void)output;
    return false;
#endif
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
        const cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
        }
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

} // namespace

thread_local Mamba2SSD::ThreadBuffers Mamba2SSD::buffers_{};

Mamba2SSD::Mamba2SSD(int d_model_value, int d_state_value, int n_heads_value,
                     const MambaConfig& config)
    : d_model(d_model_value),
      d_state(d_state_value),
      n_heads(std::max(n_heads_value, 1)),
      d_head(std::max(d_model_value / std::max(n_heads_value, 1), 1)),
      config_(config),
      in_proj_robust(d_model_value, d_model_value, true),
      in_proj_sensitive(d_model_value, d_model_value, true),
      out_proj(d_model_value, d_model_value, true),
      A(Tensor::ones({d_model_value}, Device::CPU), "mamba.A"),
      D(Tensor::ones({d_model_value}, Device::CPU), "mamba.D") {}

float Mamba2SSD::softplus_stable(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

float Mamba2SSD::sigmoid_stable(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

float Mamba2SSD::silu_stable(float x) {
    return x * sigmoid_stable(x);
}

float Mamba2SSD::d_silu_stable(float x, float sigmoid_x) {
    return sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
}

Tensor Mamba2SSD::ssd_forward(const Tensor& x, const Tensor& delta,
                              const Tensor& A_data, const Tensor& B_data,
                              const Tensor& C_data, Context* ctx,
                              bool save_history) {
    (void)ctx;

    const bool rank_2 = x.shape.size() == 2;
    const int batch = rank_2 ? 1 : x.shape[0];
    const int seq = rank_2 ? x.shape[0] : x.shape[1];
    const int dim = x.shape.back();

    Tensor output(rank_2 ? std::vector<int>{seq, dim}
                         : std::vector<int>{batch, seq, dim},
                  x.get_device());

    Tensor x_host = x.get_device() == Device::GPU ? x.cpu() : x;
    Tensor delta_host = delta.get_device() == Device::GPU ? delta.cpu() : delta;
    Tensor a_host = A_data.get_device() == Device::GPU ? A_data.cpu() : A_data;
    Tensor b_host = B_data.get_device() == Device::GPU ? B_data.cpu() : B_data;
    Tensor c_host = C_data.get_device() == Device::GPU ? C_data.cpu() : C_data;
    Tensor output_host(rank_2 ? std::vector<int>{seq, dim}
                              : std::vector<int>{batch, seq, dim},
                       Device::CPU);
    Tensor history_host;
    if (save_history) {
        history_host = Tensor(rank_2 ? std::vector<int>{seq, dim}
                                     : std::vector<int>{batch, seq, dim},
                              Device::CPU);
    }

    const float* x_ptr = x_host.data();
    const float* dt_ptr = delta_host.data();
    const float* a_ptr = a_host.data();
    const float* b_ptr = b_host.data();
    const float* c_ptr = c_host.data();
    float* out_ptr = output_host.data();
    float* history_ptr = save_history ? history_host.data() : nullptr;

    auto& state = buffers_.ssm_state;
    state.assign(dim, 0.0f);

    for (int b = 0; b < batch; ++b) {
        std::fill(state.begin(), state.end(), 0.0f);
        for (int t = 0; t < seq; ++t) {
            const int token_index = rank_2 ? t : (b * seq + t);
            const float* x_t = x_ptr + token_index * dim;
            const float* dt_t = dt_ptr + token_index * dim;
            const float* b_t = b_ptr + token_index * dim;
            const float* c_t = c_ptr + token_index * dim;
            float* y_t = out_ptr + token_index * dim;
            float* s_t = history_ptr ? history_ptr + token_index * dim : nullptr;

            for (int d = 0; d < dim; ++d) {
                const float decay = std::exp(
                    -softplus_stable(dt_t[d]) * std::max(a_ptr[d], 1e-3f));
                state[d] = state[d] * decay + b_t[d] * x_t[d];
                if (s_t) {
                    s_t[d] = state[d];
                }
                y_t[d] = std::tanh(state[d]) * c_t[d];
            }
        }
    }

    if (save_history) {
        saved_state_history_ =
            x.get_device() == Device::GPU ? history_host.to(Device::GPU) : history_host;
    } else {
        saved_state_history_ = Tensor();
    }
    return x.get_device() == Device::GPU ? output_host.to(Device::GPU) : output_host;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
Mamba2SSD::ssd_backward(const Tensor& grad_y, const Tensor& x,
                        const Tensor& delta, const Tensor& A_data,
                        const Tensor& B_data, const Tensor& C_data) {
    const bool rank_2 = x.shape.size() == 2;
    const int batch = rank_2 ? 1 : x.shape[0];
    const int seq = rank_2 ? x.shape[0] : x.shape[1];
    const int dim = x.shape.back();

    Tensor state_history =
        (saved_state_history_.size == x.size && saved_state_history_.shape == x.shape &&
         config_.save_intermediates)
            ? saved_state_history_
            : Tensor();
    if (state_history.size == 0) {
        (void)ssd_forward(x, delta, A_data, B_data, C_data, nullptr, true);
        state_history = saved_state_history_;
    }

    Tensor grad_y_host = grad_y.get_device() == Device::GPU ? grad_y.cpu() : grad_y;
    Tensor x_host = x.get_device() == Device::GPU ? x.cpu() : x;
    Tensor delta_host = delta.get_device() == Device::GPU ? delta.cpu() : delta;
    Tensor a_host = A_data.get_device() == Device::GPU ? A_data.cpu() : A_data;
    Tensor b_host = B_data.get_device() == Device::GPU ? B_data.cpu() : B_data;
    Tensor c_host = C_data.get_device() == Device::GPU ? C_data.cpu() : C_data;
    Tensor history_host =
        state_history.get_device() == Device::GPU ? state_history.cpu() : state_history;

    Tensor grad_x_cpu(x.shape.dims, Device::CPU);
    Tensor grad_delta_cpu(delta.shape.dims, Device::CPU);
    Tensor grad_A_cpu = Tensor::zeros(A_data.shape.dims, Device::CPU);
    Tensor grad_B_cpu(B_data.shape.dims, Device::CPU);
    Tensor grad_C_cpu(C_data.shape.dims, Device::CPU);

    const float* grad_y_ptr = grad_y_host.data();
    const float* x_ptr = x_host.data();
    const float* state_ptr = history_host.data();
    const float* dt_ptr = delta_host.data();
    const float* a_ptr = a_host.data();
    const float* b_ptr = b_host.data();
    const float* c_ptr = c_host.data();
    float* grad_x_ptr = grad_x_cpu.data();
    float* grad_delta_ptr = grad_delta_cpu.data();
    float* grad_a_ptr = grad_A_cpu.data();
    float* grad_b_ptr = grad_B_cpu.data();
    float* grad_c_ptr = grad_C_cpu.data();

    for (int b = 0; b < batch; ++b) {
        auto& grad_state_next = buffers_.grad_state;
        grad_state_next.assign(dim, 0.0f);
        for (int t = seq - 1; t >= 0; --t) {
            const int token_index = rank_2 ? t : (b * seq + t);
            const int prev_index = rank_2 ? (t - 1) : (b * seq + t - 1);

            for (int d = 0; d < dim; ++d) {
                const float state_t = state_ptr[token_index * dim + d];
                const float prev_state = t == 0 ? 0.0f : state_ptr[prev_index * dim + d];
                const float a_value = std::max(a_ptr[d], 1e-3f);
                const float b_value = b_ptr[token_index * dim + d];
                const float c_value = c_ptr[token_index * dim + d];
                const float candidate = std::tanh(state_t);
                const float decay =
                    std::exp(-softplus_stable(dt_ptr[token_index * dim + d]) * a_value);

                grad_c_ptr[token_index * dim + d] +=
                    grad_y_ptr[token_index * dim + d] * candidate;
                const float grad_candidate =
                    grad_y_ptr[token_index * dim + d] * c_value;
                const float grad_state =
                    grad_candidate * (1.0f - candidate * candidate) + grad_state_next[d];

                grad_x_ptr[token_index * dim + d] += grad_state * b_value;
                grad_b_ptr[token_index * dim + d] +=
                    grad_state * x_ptr[token_index * dim + d];

                const float grad_decay = grad_state * prev_state;
                grad_state_next[d] = grad_state * decay;

                const float decay_pre = grad_decay * decay;
                grad_delta_ptr[token_index * dim + d] +=
                    decay_pre * (-a_value) * sigmoid_stable(dt_ptr[token_index * dim + d]);

                if (a_ptr[d] > 1e-3f) {
                    grad_a_ptr[d] +=
                        decay_pre * (-softplus_stable(dt_ptr[token_index * dim + d]));
                }
            }
        }
    }

    Tensor grad_x =
        x.get_device() == Device::GPU ? grad_x_cpu.to(Device::GPU) : grad_x_cpu;
    Tensor grad_delta =
        delta.get_device() == Device::GPU ? grad_delta_cpu.to(Device::GPU) : grad_delta_cpu;
    Tensor grad_A =
        A_data.get_device() == Device::GPU ? grad_A_cpu.to(Device::GPU) : grad_A_cpu;
    Tensor grad_B =
        B_data.get_device() == Device::GPU ? grad_B_cpu.to(Device::GPU) : grad_B_cpu;
    Tensor grad_C =
        C_data.get_device() == Device::GPU ? grad_C_cpu.to(Device::GPU) : grad_C_cpu;
    return {grad_x, grad_delta, grad_A, grad_B, grad_C};
}

Tensor Mamba2SSD::apply_gating(const Tensor& y_ssd, const Tensor& x,
                               const Tensor& z, Tensor* grad_buffer) {
    (void)x;
    Tensor gate = z.sigmoid();
    if (grad_buffer) {
        grad_buffer->copy_from(gate);
    }
    return y_ssd.mul(gate);
}

void Mamba2SSD::update_streaming_state_from_history(const Tensor& input) {
    if (!streaming_inference_ || saved_state_history_.size == 0 ||
        (input.shape.size() != 2 && input.shape.size() != 3)) {
        return;
    }

    const bool rank_2 = input.shape.size() == 2;
    const int batch = rank_2 ? 1 : input.shape[0];
    const int seq = rank_2 ? input.shape[0] : input.shape[1];
    if (batch <= 0 || seq <= 0 || input.shape.back() != d_model) {
        return;
    }

    Tensor history_host =
        saved_state_history_.get_device() == Device::GPU ? saved_state_history_.cpu()
                                                         : saved_state_history_;
    Tensor next_state(batch == 1 ? std::vector<int>{1, d_model}
                                 : std::vector<int>{batch, d_model},
                      Device::CPU);
    const float* history_ptr = history_host.data();
    float* state_ptr = next_state.data();
    for (int row = 0; row < batch; ++row) {
        const int token_index = rank_2 ? (seq - 1) : (row * seq + seq - 1);
        std::memcpy(state_ptr + static_cast<size_t>(row) * static_cast<size_t>(d_model),
                    history_ptr + static_cast<size_t>(token_index) * static_cast<size_t>(d_model),
                    static_cast<size_t>(d_model) * sizeof(float));
    }

    Tensor target_state =
        input.get_device() == Device::GPU ? next_state.to(Device::GPU) : next_state;
    streaming_state_ = std::make_shared<Tensor>(std::move(target_state));
}

Tensor Mamba2SSD::forward(const Tensor& u, Context* ctx) {
    if (u.shape.size() != 2 && u.shape.size() != 3) {
        throw std::runtime_error("Mamba2SSD expects rank-2 or rank-3 input");
    }

    const bool single_token_rank2 =
        u.shape.size() == 2 && u.shape[0] == 1 && u.shape[1] == d_model;
    const bool single_token_rank1 =
        u.shape.size() == 1 && u.shape[0] == d_model;
    const bool single_token_rank3 =
        u.shape.size() == 3 && u.shape[1] == 1 && u.shape[2] == d_model;

    if (streaming_inference_ && (single_token_rank1 || single_token_rank2 || single_token_rank3)) {
        Tensor input = single_token_rank1 ? u.reshape({1, d_model}) : u;
        const int batch = input.shape.size() == 3 ? input.shape[0] : 1;
        const std::vector<int> state_shape =
            batch == 1 ? std::vector<int>{1, d_model} : std::vector<int>{batch, d_model};
        const Device state_device = input.get_device();
        if (!streaming_state_ ||
            streaming_state_->shape.dims != state_shape ||
            streaming_state_->get_device() != state_device) {
            streaming_state_ = std::make_shared<Tensor>(Tensor::zeros(state_shape, state_device));
        } else if (streaming_state_.use_count() > 1) {
            streaming_state_ = std::make_shared<Tensor>(streaming_state_->clone());
        }

        Tensor x_proj = in_proj_robust.forward(input);
        Tensor gate = in_proj_sensitive.forward(input);
        Tensor delta = gate.sigmoid();
        Tensor selective_b = x_proj.sigmoid();
        Tensor selective_c = gate.sigmoid();
        Tensor x_scaled = x_proj.mul(selective_b);
        Tensor y_ssd(state_shape, input.get_device());

#ifdef USE_CUDA
        if (can_use_gpu_mamba_single_token(x_scaled, delta, A.data, *streaming_state_, y_ssd)) {
            ++gpu_fast_path_hits_;
            last_fallback_reason_.clear();
            Tensor x_flat = x_scaled.reshape({batch, d_model});
            Tensor delta_flat = delta.reshape({batch, d_model});
            cuda::launch_mamba_single_token_update(
                x_flat.data(),
                delta_flat.data(),
                A.data.data(),
                streaming_state_->data(),
                y_ssd.data(),
                batch,
                d_model);
            y_ssd = y_ssd.mul(selective_c.reshape(state_shape));
        } else
#endif
        {
            ++gpu_fast_path_fallbacks_;
#ifdef USE_CUDA
            last_fallback_reason_ = "selective_bc_cpu_fallback";
#else
            last_fallback_reason_ = "cuda_unavailable";
#endif
            Tensor x_proj_host = x_proj.get_device() == Device::GPU ? x_proj.cpu() : x_proj;
            Tensor delta_host = delta.get_device() == Device::GPU ? delta.cpu() : delta;
            Tensor a_host = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
            Tensor b_host = selective_b.get_device() == Device::GPU ? selective_b.cpu() : selective_b;
            Tensor c_host = selective_c.get_device() == Device::GPU ? selective_c.cpu() : selective_c;
            Tensor state_host =
                streaming_state_->get_device() == Device::GPU ? streaming_state_->cpu() : *streaming_state_;
            Tensor y_ssd_host(state_shape, Device::CPU);

            const float* x_ptr = x_proj_host.data();
            const float* delta_ptr = delta_host.data();
            const float* a_ptr = a_host.data();
            const float* b_ptr = b_host.data();
            const float* c_ptr = c_host.data();
            float* state_ptr = state_host.data();
            float* y_ptr = y_ssd_host.data();

            for (int row = 0; row < batch; ++row) {
                for (int d = 0; d < d_model; ++d) {
                    const int index = row * d_model + d;
                    const float a_value = std::max(a_ptr[d], 1e-3f);
                    const float decay =
                        std::exp(-softplus_stable(delta_ptr[index]) * a_value);
                    const float next_state =
                        state_ptr[index] * decay + b_ptr[index] * x_ptr[index];
                    state_ptr[index] = next_state;
                    y_ptr[index] = std::tanh(next_state) * c_ptr[index];
                }
            }

            *streaming_state_ =
                streaming_state_->get_device() == Device::GPU ? state_host.to(Device::GPU) : state_host;
            y_ssd = input.get_device() == Device::GPU ? y_ssd_host.to(Device::GPU) : y_ssd_host;
        }

        if (input.shape.size() == 3) {
            y_ssd = y_ssd.reshape({batch, 1, d_model});
        } else {
            y_ssd = y_ssd.reshape({1, d_model});
        }
        Tensor gated = y_ssd.mul(selective_c);
        Tensor projected = out_proj.forward(gated);
        Tensor skip = input.mul(D.data);
        Tensor result = projected.add(skip);
        if (single_token_rank1) {
            return result.reshape({d_model});
        }
        return result;
    }

    saved_input_ = u;
    saved_x_proj_ = in_proj_robust.forward(u);
    saved_gate_ = in_proj_sensitive.forward(u);

    saved_delta_ = saved_gate_.sigmoid();
    saved_B_ = saved_x_proj_.sigmoid();
    saved_C_ = saved_gate_.sigmoid();
    Tensor y_ssd =
        ssd_forward(saved_x_proj_, saved_delta_, A.data, saved_B_, saved_C_, ctx, true);
    saved_ssd_ = y_ssd;
    update_streaming_state_from_history(u);
    Tensor gated = y_ssd.mul(saved_C_);
    Tensor projected = out_proj.forward(gated);
    Tensor skip = saved_input_.mul(D.data);
    return projected.add(skip);
}

Tensor Mamba2SSD::backward(const Tensor& grad_output, Context& ctx) {
    (void)ctx;
    if (saved_input_.size == 0 || saved_x_proj_.size == 0 || saved_gate_.size == 0) {
        throw std::runtime_error("Mamba2SSD backward called before forward");
    }

    const int dim = saved_input_.shape.back();
    const int rows = saved_input_.size / dim;
    Tensor grad_skip = grad_output.mul(D.data);
    Tensor grad_D = grad_output.mul(saved_input_).reshape({rows, dim}).sum(0, false);
    D.add_grad(grad_D);

    Tensor grad_gated = out_proj.backward(grad_output);
    Tensor gate_sigmoid = saved_gate_.sigmoid();
    Tensor gate_complement =
        Tensor::ones(gate_sigmoid.shape.dims, gate_sigmoid.get_device()).sub(gate_sigmoid);
    Tensor grad_y_ssd = grad_gated.mul(gate_sigmoid);
    Tensor grad_gate_raw =
        grad_gated.mul(saved_ssd_).mul(gate_sigmoid).mul(gate_complement);

    auto [grad_x_proj_ssd, grad_delta, grad_A, grad_B, grad_C] =
        ssd_backward(grad_y_ssd, saved_x_proj_, saved_delta_, A.data, saved_B_, saved_C_);
    A.add_grad(grad_A);
    Tensor delta_complement =
        Tensor::ones(saved_delta_.shape.dims, saved_delta_.get_device()).sub(saved_delta_);
    grad_gate_raw = grad_gate_raw.add(
        grad_delta.mul(saved_delta_).mul(delta_complement));
    Tensor grad_b_raw = grad_B.mul(saved_B_).mul(
        Tensor::ones(saved_B_.shape.dims, saved_B_.get_device()).sub(saved_B_));
    Tensor grad_c_raw = grad_C.mul(saved_C_).mul(
        Tensor::ones(saved_C_.shape.dims, saved_C_.get_device()).sub(saved_C_));

    Tensor grad_from_xproj = in_proj_robust.backward(grad_x_proj_ssd.add(grad_b_raw));
    Tensor grad_from_gate = in_proj_sensitive.backward(grad_gate_raw.add(grad_c_raw));
    return grad_from_xproj.add(grad_from_gate).add(grad_skip).reshape(saved_input_.shape.dims);
}

void Mamba2SSD::reset() {
    saved_input_ = Tensor();
    saved_x_proj_ = Tensor();
    saved_gate_ = Tensor();
    saved_delta_ = Tensor();
    saved_B_ = Tensor();
    saved_C_ = Tensor();
    saved_ssd_ = Tensor();
    saved_state_history_ = Tensor();
    streaming_state_.reset();
}

void Mamba2SSD::reset_runtime_telemetry() {
    gpu_fast_path_hits_ = 0;
    gpu_fast_path_fallbacks_ = 0;
    last_fallback_reason_.clear();
}

void Mamba2SSD::to(Device dev) {
    in_proj_robust.to(dev);
    in_proj_sensitive.to(dev);
    out_proj.to(dev);
    A.data = A.data.to(dev);
    D.data = D.data.to(dev);
}

std::vector<Parameter*> Mamba2SSD::parameters() {
    std::vector<Parameter*> params;
    auto robust = in_proj_robust.parameters();
    prefix_parameter_names(robust, "in_proj_robust.");
    params.insert(params.end(), robust.begin(), robust.end());
    auto sensitive = in_proj_sensitive.parameters();
    prefix_parameter_names(sensitive, "in_proj_sensitive.");
    params.insert(params.end(), sensitive.begin(), sensitive.end());
    auto out = out_proj.parameters();
    prefix_parameter_names(out, "out_proj.");
    params.insert(params.end(), out.begin(), out.end());
    A.base_name = "A";
    A.name = "A";
    params.push_back(&A);
    D.base_name = "D";
    D.name = "D";
    params.push_back(&D);
    return params;
}

void Mamba2SSD::set_streaming_mode(bool enabled) {
    streaming_inference_ = enabled;
    if (!enabled) {
        streaming_state_.reset();
    } else if (!streaming_state_ || streaming_state_->shape.back() != d_model) {
        streaming_state_ =
            std::make_shared<Tensor>(Tensor::zeros({1, d_model}, A.data.get_device()));
    }
}

MambaStreamSnapshot Mamba2SSD::snapshot_streaming_state() const {
    MambaStreamSnapshot snapshot;
    snapshot.enabled = streaming_inference_;
    snapshot.state = streaming_state_;
    return snapshot;
}

void Mamba2SSD::restore_streaming_state(const MambaStreamSnapshot& snapshot) {
    streaming_inference_ = snapshot.enabled;
    streaming_state_ = snapshot.state;
    if (streaming_inference_ &&
        (!streaming_state_ || streaming_state_->shape.back() != d_model)) {
        streaming_state_ =
            std::make_shared<Tensor>(Tensor::zeros({1, d_model}, A.data.get_device()));
    }
}

std::vector<MambaStreamSnapshot> Mamba2SSD::snapshot_streaming_state_batch() const {
    std::vector<MambaStreamSnapshot> snapshots;
    const int batch = streaming_batch_size();
    if (batch <= 0) {
        return snapshots;
    }

    snapshots.resize(static_cast<size_t>(batch));
    Tensor state =
        (streaming_state_ && streaming_state_->get_device() == Device::GPU) ? streaming_state_->cpu()
                                                                             : (streaming_state_ ? *streaming_state_ : Tensor());
    const float* state_ptr = state.data();
    const Device target_device = streaming_state_ ? streaming_state_->get_device() : A.data.get_device();
    for (int row = 0; row < batch; ++row) {
        snapshots[static_cast<size_t>(row)].enabled = streaming_inference_;
        Tensor row_state({1, d_model}, Device::CPU);
        std::memcpy(row_state.data(),
                    state_ptr + static_cast<size_t>(row) * static_cast<size_t>(d_model),
                    static_cast<size_t>(d_model) * sizeof(float));
        if (target_device == Device::GPU) {
            row_state = row_state.to(Device::GPU);
        }
        snapshots[static_cast<size_t>(row)].state =
            std::make_shared<Tensor>(std::move(row_state));
    }
    return snapshots;
}

void Mamba2SSD::restore_streaming_state_batch(const std::vector<MambaStreamSnapshot>& snapshots) {
    if (snapshots.empty()) {
        streaming_state_.reset();
        return;
    }

    streaming_inference_ = snapshots.front().enabled;
    const int batch = static_cast<int>(snapshots.size());
    const Device target_device = A.data.get_device();
    Tensor merged({batch, d_model}, target_device);
    for (int row = 0; row < batch; ++row) {
        Tensor row_state;
        if (snapshots[static_cast<size_t>(row)].state &&
            snapshots[static_cast<size_t>(row)].state->size > 0) {
            row_state = *snapshots[static_cast<size_t>(row)].state;
            if (row_state.shape.size() == 1) {
                row_state = row_state.reshape({1, d_model});
            }
            if (row_state.shape != TensorShape({1, d_model})) {
                throw std::runtime_error("Invalid Mamba streaming snapshot shape");
            }
            if (row_state.get_device() != target_device) {
                row_state = row_state.to(target_device);
            }
        } else {
            row_state = Tensor::zeros({1, d_model}, target_device);
        }
        copy_float_bytes_device_safe(
            merged.data() + static_cast<size_t>(row) * static_cast<size_t>(d_model),
            merged.get_device(),
            row_state.data(),
            row_state.get_device(),
            static_cast<size_t>(d_model) * sizeof(float));
    }
    streaming_state_ = std::make_shared<Tensor>(std::move(merged));
}

int Mamba2SSD::streaming_batch_size() const {
    if (!streaming_state_ || streaming_state_->size == 0) {
        return 0;
    }
    if (streaming_state_->shape.size() == 1) {
        return 1;
    }
    if (streaming_state_->shape.size() == 2) {
        return streaming_state_->shape[0];
    }
    return 0;
}

void Mamba2SSD::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    out.push_back(&in_proj_robust);
    out.push_back(&in_proj_sensitive);
    out.push_back(&out_proj);
}

} // namespace nsos
