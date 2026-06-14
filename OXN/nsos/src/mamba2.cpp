#include "../include/mamba2.h"
#include "../include/cuda/mamba_kernels.cuh"
#include "../include/jamba_utils.h"
#include "../include/nsos/determinism.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nsos {

namespace {

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
      D(Tensor::ones({d_model_value}, Device::CPU), "mamba.D") {
  // dt/B/C come from the "sensitive" projection; keep it on the float path
  // during QAT (mixed precision: ternary robust + out_proj, higher-precision
  // SSM scan parameters).  See trainer.cpp apply_progressive_qat_phase.
  in_proj_sensitive.set_quantization_sensitive(true);

  // OXTA-CRIT Lei 2 (docs/OXTA_CRIT_THEORY.md): A = ones gives a fully
  // DEGENERATE memory spectrum — every channel forgets at the same rate
  // (tau ~ 1/A identical everywhere; measured by scripts/criticality_probe.py:
  // tau_log_spread = 0.0).  S4D-style log-spaced A spans a hierarchy of
  // timescales (tau from 1 to 1/A_min tokens at nominal dt) so slow channels
  // exist from step 0 instead of having to be discovered by gradient.
  // Opt-in (default OFF preserves behavior): NSOS_MAMBA_A_LOGSPACED=1,
  // optional NSOS_MAMBA_A_MIN (default 0.01 -> tau up to ~100 tokens).
  // Read per-construction (not static) so an A/B harness can flip the env
  // between model constructions in one process.  Deterministic (no RNG).
  if (const char* env = std::getenv("NSOS_MAMBA_A_LOGSPACED");
      env != nullptr && env[0] == '1') {
    float a_min = 0.01f;
    if (const char* mn = std::getenv("NSOS_MAMBA_A_MIN"); mn != nullptr) {
      const float parsed = std::strtof(mn, nullptr);
      if (parsed > 1e-3f && parsed < 1.0f) {
        a_min = parsed;
      }
    }
    float* a_ptr = A.data.data();
    const int n = A.data.size;
    const float log_min = std::log(a_min);
    for (int d = 0; d < n; ++d) {
      const float frac = (n > 1) ? static_cast<float>(d) / static_cast<float>(n - 1) : 0.0f;
      a_ptr[d] = std::exp(log_min * frac);  // d=0 -> 1.0 ... d=n-1 -> a_min
    }
  }

  // ── Proper selective-SSM components (opt-in) ───────────────────────────────
  // Allocated only when requested so the default path's parameter set / pack
  // layout is unchanged.  Independent projections give x, z(gate), B, C, dt
  // their own capacity (kills the delta==C degeneracy); the depthwise causal
  // conv is initialized to an identity pass-through (last tap = 1) so a freshly
  // enabled proper path starts numerically close to "no conv" and learns the
  // local mixing from there.
  if (config_.proper_selective_ssm) {
    conv_kernel_ = std::max(config_.conv_kernel, 1);
    x_proj_  = std::make_unique<BitLinear>(d_model, d_model, true);
    z_proj_  = std::make_unique<BitLinear>(d_model, d_model, true);
    B_proj_  = std::make_unique<BitLinear>(d_model, d_model, true);
    C_proj_  = std::make_unique<BitLinear>(d_model, d_model, true);
    dt_proj_ = std::make_unique<BitLinear>(d_model, d_model, true);
    // dt/B/C carry the SSM's selectivity — keep them off the ternary path during
    // QAT (mixed precision), exactly as the legacy in_proj_sensitive does.
    dt_proj_->set_quantization_sensitive(true);
    B_proj_->set_quantization_sensitive(true);
    C_proj_->set_quantization_sensitive(true);
    conv_weight_ = Parameter(Tensor::zeros({d_model, conv_kernel_}, Device::CPU),
                             "conv1d_weight");
    float* cw = conv_weight_.data.data();
    for (int c = 0; c < d_model; ++c) {
      cw[c * conv_kernel_ + (conv_kernel_ - 1)] = 1.0f; // identity: newest tap
    }
  }
}

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

#ifdef USE_CUDA
    // GPU fast-path: all five inputs must already live on the device,
    // shapes must match the canonical [B,Seq,D] layout the kernel
    // expects, and dim must equal x.shape.back().  Otherwise we fall
    // through to the CPU path (which copies as needed).  This avoids
    // accidentally promoting/copying when callers pass mixed devices.
    const bool all_on_gpu = x.get_device() == Device::GPU &&
                            delta.get_device() == Device::GPU &&
                            A_data.get_device() == Device::GPU &&
                            B_data.get_device() == Device::GPU &&
                            C_data.get_device() == Device::GPU;
    const bool shapes_match_simple_dim =
        delta.size == x.size && B_data.size == x.size &&
        C_data.size == x.size && A_data.size == dim;
    if (all_on_gpu && shapes_match_simple_dim && batch > 0 && seq > 0 &&
        dim > 0) {
        // Reshape to canonical [B, Seq, D] for the kernel.  Reshape is
        // a metadata-only operation when underlying storage is contiguous
        // (which it is for tensors freshly allocated on GPU here).
        const std::vector<int> canonical_shape{batch, seq, dim};
        Tensor x_canon = rank_2 ? x.reshape(canonical_shape) : x;
        Tensor dt_canon =
            rank_2 ? delta.reshape(canonical_shape) : delta;
        Tensor B_canon =
            rank_2 ? B_data.reshape(canonical_shape) : B_data;
        Tensor C_canon =
            rank_2 ? C_data.reshape(canonical_shape) : C_data;

        // Output buffer with canonical layout; we reshape back to the
        // caller's expected layout at the return site.
        Tensor y_canon(canonical_shape, Device::GPU);

        // state_history is required for ssd_backward.  Allocate when the
        // forward pass is asked to save it; otherwise pass nullptr and
        // skip the per-step write inside the kernel.
        Tensor history;
        float* history_ptr = nullptr;
        if (save_history) {
            history = Tensor(canonical_shape, Device::GPU);
            history_ptr = history.raw_data();
        }

        cuda::launch_mamba_selective_scan_forward(
            x_canon.raw_data(), dt_canon.raw_data(), A_data.raw_data(),
            B_canon.raw_data(), C_canon.raw_data(), y_canon.raw_data(),
            history_ptr, batch, seq, dim);
        ++gpu_fast_path_hits_;  // telemetria do caminho BATCHED (treino)

        // saved_state_history_ matches the original layout so callers
        // (notably ssd_backward) can compare shapes safely.
        if (save_history) {
            saved_state_history_ =
                rank_2 ? history.reshape(std::vector<int>{seq, dim}) : history;
        } else {
            saved_state_history_ = Tensor();
        }

        return rank_2 ? y_canon.reshape(std::vector<int>{seq, dim}) : y_canon;
    }
#endif

    if (x.get_device() == Device::GPU) {
        ++gpu_fast_path_fallbacks_;  // scan batched caiu no host com tensores GPU
        last_fallback_reason_ = "ssd_forward_host_fallback";
    }
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

#ifdef USE_CUDA
    // GPU fast-path mirroring ssd_forward.  Same eligibility rules:
    // every input must already be on the device and shapes must match
    // the canonical [B,Seq,D] layout.  state_history must also be on
    // the device so we don't have to round-trip it.
    const bool all_on_gpu = grad_y.get_device() == Device::GPU &&
                            x.get_device() == Device::GPU &&
                            delta.get_device() == Device::GPU &&
                            A_data.get_device() == Device::GPU &&
                            B_data.get_device() == Device::GPU &&
                            C_data.get_device() == Device::GPU &&
                            state_history.get_device() == Device::GPU;
    const bool shapes_match_simple_dim =
        delta.size == x.size && B_data.size == x.size &&
        C_data.size == x.size && A_data.size == dim &&
        grad_y.size == x.size && state_history.size == x.size;
    // Deterministic mode: the GPU backward accumulates grad_A via atomicAdd
    // (order-nondeterministic); fall through to the ordered host loop below.
    if (all_on_gpu && shapes_match_simple_dim && batch > 0 && seq > 0 &&
        dim > 0 && !determinism::deterministic_reductions_enabled()) {
        const std::vector<int> canonical_shape{batch, seq, dim};

        // Reshape (metadata-only on contiguous storage) to canonical 3D.
        Tensor grad_y_canon =
            rank_2 ? grad_y.reshape(canonical_shape) : grad_y;
        Tensor x_canon = rank_2 ? x.reshape(canonical_shape) : x;
        Tensor dt_canon =
            rank_2 ? delta.reshape(canonical_shape) : delta;
        Tensor B_canon =
            rank_2 ? B_data.reshape(canonical_shape) : B_data;
        Tensor C_canon =
            rank_2 ? C_data.reshape(canonical_shape) : C_data;
        Tensor sh_canon = rank_2
                              ? state_history.reshape(canonical_shape)
                              : state_history;

        // Output gradient buffers — match the input shapes so callers
        // can use them without reshape gymnastics.  Initialize with
        // zeros only for grad_A; the kernel writes the others fully.
        Tensor grad_x_t(x.shape.dims, Device::GPU);
        Tensor grad_delta_t(delta.shape.dims, Device::GPU);
        Tensor grad_A_t = Tensor::zeros(A_data.shape.dims, Device::GPU);
        Tensor grad_B_t(B_data.shape.dims, Device::GPU);
        Tensor grad_C_t(C_data.shape.dims, Device::GPU);

        cuda::launch_mamba_selective_scan_backward(
            grad_y_canon.raw_data(), x_canon.raw_data(), dt_canon.raw_data(),
            A_data.raw_data(), B_canon.raw_data(), C_canon.raw_data(),
            sh_canon.raw_data(), grad_x_t.raw_data(), grad_delta_t.raw_data(),
            grad_A_t.raw_data(), grad_B_t.raw_data(), grad_C_t.raw_data(),
            batch, seq, dim);

        return {grad_x_t, grad_delta_t, grad_A_t, grad_B_t, grad_C_t};
    }
#endif

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

// ===========================================================================
// Proper selective-SSM path (config_.proper_selective_ssm).  Host math.
// ===========================================================================

void Mamba2SSD::conv1d_causal_forward(const float* in, const float* weight,
                                      float* out, int batch, int seq, int dim,
                                      int K) {
    // out[b,t,c] = sum_{j=0..K-1} weight[c,j] * in[b, t-(K-1)+j, c], in[<0]=0.
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            const size_t row = static_cast<size_t>(b) * seq + t;
            for (int c = 0; c < dim; ++c) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const int src_t = t - (K - 1) + j;
                    if (src_t < 0) continue;
                    const size_t src_row = static_cast<size_t>(b) * seq + src_t;
                    acc += weight[static_cast<size_t>(c) * K + j] *
                           in[src_row * dim + c];
                }
                out[row * dim + c] = acc;
            }
        }
    }
}

void Mamba2SSD::conv1d_causal_backward(const float* grad_out, const float* in,
                                       const float* weight, float* grad_in,
                                       float* grad_weight, int batch, int seq,
                                       int dim, int K) {
    // grad_in and grad_weight must be pre-zeroed by the caller.
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            const size_t row = static_cast<size_t>(b) * seq + t;
            for (int c = 0; c < dim; ++c) {
                const float go = grad_out[row * dim + c];
                for (int j = 0; j < K; ++j) {
                    const int src_t = t - (K - 1) + j;
                    if (src_t < 0) continue;
                    const size_t src_row = static_cast<size_t>(b) * seq + src_t;
                    grad_weight[static_cast<size_t>(c) * K + j] +=
                        go * in[src_row * dim + c];
                    grad_in[src_row * dim + c] +=
                        go * weight[static_cast<size_t>(c) * K + j];
                }
            }
        }
    }
}

Tensor Mamba2SSD::forward_proper(const Tensor& u) {
    const bool rank_2 = u.shape.size() == 2;
    const int batch = rank_2 ? 1 : u.shape[0];
    const int seq = rank_2 ? u.shape[0] : u.shape[1];
    const int dim = u.shape.back();
    if (dim != d_model) {
        throw std::runtime_error("Mamba2SSD proper path: last dim != d_model");
    }
    const int rows = batch * seq;
    const int K = conv_kernel_;

    Tensor u_host = u.get_device() == Device::GPU ? u.cpu() : u;
    Tensor u_flat = u_host.reshape({rows, dim});

    // Independent projections (host).  Each BitLinear caches its own input.
    Tensor xv = x_proj_->forward(u_flat).reshape({rows, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({rows, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({rows, dim});
    Tensor Ct = C_proj_->forward(u_flat).reshape({rows, dim});
    Tensor dt = dt_proj_->forward(u_flat).reshape({rows, dim});

    // Causal depthwise conv on x, then SiLU (Mamba's local mixing).
    Tensor conv_pre({rows, dim}, Device::CPU);
    conv1d_causal_forward(xv.data(), conv_weight_.data.data(), conv_pre.data(),
                          batch, seq, dim, K);
    Tensor xc({rows, dim}, Device::CPU);
    {
        const float* cp = conv_pre.data();
        float* xp = xc.data();
        const int n = rows * dim;
        for (int i = 0; i < n; ++i) xp[i] = silu_stable(cp[i]);
    }

    Tensor a_host = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    const float* a_ptr = a_host.data();

    Tensor h_hist({rows, dim}, Device::CPU);
    Tensor y_ssd({rows, dim}, Device::CPU);
    const float* b_ptr = Bt.data();
    const float* c_ptr = Ct.data();
    const float* dt_ptr = dt.data();
    const float* xc_ptr = xc.data();
    float* h_ptr = h_hist.data();
    float* y_ptr = y_ssd.data();

    // Diagonal selective recurrence: h_t = decay_t*h_{t-1} + B_t*xc_t;
    // y_t = h_t * C_t  (C applied exactly ONCE, linear readout).
    for (int b = 0; b < batch; ++b) {
        std::vector<float> state(static_cast<size_t>(dim), 0.0f);
        for (int t = 0; t < seq; ++t) {
            const size_t row = static_cast<size_t>(b) * seq + t;
            for (int c = 0; c < dim; ++c) {
                const size_t idx = row * dim + c;
                const float a_value = std::max(a_ptr[c], 1e-3f);
                const float decay =
                    std::exp(-softplus_stable(dt_ptr[idx]) * a_value);
                state[static_cast<size_t>(c)] =
                    decay * state[static_cast<size_t>(c)] + b_ptr[idx] * xc_ptr[idx];
                h_ptr[idx] = state[static_cast<size_t>(c)];
                y_ptr[idx] = state[static_cast<size_t>(c)] * c_ptr[idx];
            }
        }
    }

    // Separate SiLU gate: y = y_ssd * silu(z).
    Tensor gated({rows, dim}, Device::CPU);
    {
        const float* zp = z.data();
        float* gp = gated.data();
        const int n = rows * dim;
        for (int i = 0; i < n; ++i) gp[i] = y_ptr[i] * silu_stable(zp[i]);
    }

    Tensor projected = out_proj.forward(gated).reshape({rows, dim});
    Tensor d_host = D.data.get_device() == Device::GPU ? D.data.cpu() : D.data;
    const float* d_ptr = d_host.data();
    const float* u_ptr = u_flat.data();
    const float* pj = projected.data();
    Tensor result({rows, dim}, Device::CPU);
    float* rp = result.data();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < dim; ++c) {
            const size_t idx = static_cast<size_t>(r) * dim + c;
            rp[idx] = pj[idx] + u_ptr[idx] * d_ptr[c];
        }
    }

    pp_u_ = u_flat;
    pp_xv_ = xv;
    pp_conv_pre_ = conv_pre;
    pp_xc_ = xc;
    pp_z_ = z;
    pp_B_ = Bt;
    pp_C_ = Ct;
    pp_dt_ = dt;
    pp_h_hist_ = h_hist;
    pp_y_ssd_ = y_ssd;
    pp_batch_ = batch;
    pp_seq_ = seq;
    proper_active_ = true;

    Tensor out_shaped = rank_2 ? result : result.reshape({batch, seq, dim});
    return u.get_device() == Device::GPU ? out_shaped.to(Device::GPU) : out_shaped;
}

Tensor Mamba2SSD::backward_proper(const Tensor& grad_output) {
    const int batch = pp_batch_;
    const int seq = pp_seq_;
    const int dim = d_model;
    const int K = conv_kernel_;
    const int rows = batch * seq;

    Tensor g_host =
        grad_output.get_device() == Device::GPU ? grad_output.cpu() : grad_output;
    Tensor g = g_host.reshape({rows, dim});
    const float* gp = g.data();

    Tensor a_host = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    Tensor d_host = D.data.get_device() == Device::GPU ? D.data.cpu() : D.data;
    const float* a_ptr = a_host.data();
    const float* d_ptr = d_host.data();
    const float* u_ptr = pp_u_.data();

    // Skip path: result = out_proj(...) + u*D.
    Tensor grad_D = Tensor::zeros({dim}, Device::CPU);
    float* gD = grad_D.data();
    Tensor g_u = Tensor::zeros({rows, dim}, Device::CPU);
    float* gu = g_u.data();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < dim; ++c) {
            const size_t idx = static_cast<size_t>(r) * dim + c;
            gD[c] += gp[idx] * u_ptr[idx];
            gu[idx] += gp[idx] * d_ptr[c];
        }
    }

    // out_proj backward -> grad wrt gated.
    Tensor g_gated = out_proj.backward(g).reshape({rows, dim});
    const float* gg = g_gated.data();

    // Gate y = y_ssd * silu(z).
    const float* z_ptr = pp_z_.data();
    const float* yssd_ptr = pp_y_ssd_.data();
    Tensor g_yssd({rows, dim}, Device::CPU);
    Tensor g_z({rows, dim}, Device::CPU);
    float* gy = g_yssd.data();
    float* gz = g_z.data();
    {
        const int n = rows * dim;
        for (int i = 0; i < n; ++i) {
            const float zz = z_ptr[i];
            const float sz = sigmoid_stable(zz);
            gy[i] = gg[i] * (zz * sz);                 // * silu(z)
            gz[i] = gg[i] * yssd_ptr[i] * d_silu_stable(zz, sz);
        }
    }

    // Scan backward.  G_t = g_yssd_t*C_t + decay_{t+1}*G_{t+1}; carry = decay*G.
    const float* h_ptr = pp_h_hist_.data();
    const float* c_ptr = pp_C_.data();
    const float* b_ptr = pp_B_.data();
    const float* dt_ptr = pp_dt_.data();
    const float* xc_ptr = pp_xc_.data();
    Tensor gB = Tensor::zeros({rows, dim}, Device::CPU);
    Tensor gC = Tensor::zeros({rows, dim}, Device::CPU);
    Tensor gDt = Tensor::zeros({rows, dim}, Device::CPU);
    Tensor gXc = Tensor::zeros({rows, dim}, Device::CPU);
    Tensor grad_A = Tensor::zeros({dim}, Device::CPU);
    float* gBp = gB.data();
    float* gCp = gC.data();
    float* gDtp = gDt.data();
    float* gXcp = gXc.data();
    float* gA = grad_A.data();
    for (int b = 0; b < batch; ++b) {
        std::vector<float> carry(static_cast<size_t>(dim), 0.0f);
        for (int t = seq - 1; t >= 0; --t) {
            const size_t row = static_cast<size_t>(b) * seq + t;
            for (int c = 0; c < dim; ++c) {
                const size_t idx = row * dim + c;
                const float h_t = h_ptr[idx];
                const float h_prev =
                    t == 0 ? 0.0f
                           : h_ptr[(static_cast<size_t>(b) * seq + (t - 1)) * dim + c];
                const float a_value = std::max(a_ptr[c], 1e-3f);
                const float sp = softplus_stable(dt_ptr[idx]);
                const float decay = std::exp(-sp * a_value);
                gCp[idx] += gy[idx] * h_t;
                const float grad_h = gy[idx] * c_ptr[idx] + carry[static_cast<size_t>(c)];
                gBp[idx] += grad_h * xc_ptr[idx];
                gXcp[idx] += grad_h * b_ptr[idx];
                const float grad_decay = grad_h * h_prev;
                gDtp[idx] += grad_decay * decay * (-a_value) *
                             sigmoid_stable(dt_ptr[idx]);
                if (a_ptr[c] > 1e-3f) {
                    gA[c] += grad_decay * decay * (-sp);
                }
                carry[static_cast<size_t>(c)] = grad_h * decay;
            }
        }
    }

    // xc = silu(conv_pre) -> grad_conv_pre.
    const float* cp = pp_conv_pre_.data();
    Tensor grad_conv_pre({rows, dim}, Device::CPU);
    {
        float* gcp = grad_conv_pre.data();
        const int n = rows * dim;
        for (int i = 0; i < n; ++i) {
            const float v = cp[i];
            gcp[i] = gXcp[i] * d_silu_stable(v, sigmoid_stable(v));
        }
    }

    // conv1d backward.
    Tensor grad_xv = Tensor::zeros({rows, dim}, Device::CPU);
    Tensor grad_conv_w = Tensor::zeros({dim, K}, Device::CPU);
    conv1d_causal_backward(grad_conv_pre.data(), pp_xv_.data(),
                           conv_weight_.data.data(), grad_xv.data(),
                           grad_conv_w.data(), batch, seq, dim, K);
    conv_weight_.add_grad(grad_conv_w);

    // Project grads back to the layer input and accumulate projection params.
    auto accumulate = [&](BitLinear* proj, const Tensor& grad) {
        Tensor dxu = proj->backward(grad).reshape({rows, dim});
        const float* s = dxu.data();
        const int n = rows * dim;
        for (int i = 0; i < n; ++i) gu[i] += s[i];
    };
    accumulate(x_proj_.get(), grad_xv);
    accumulate(B_proj_.get(), gB);
    accumulate(C_proj_.get(), gC);
    accumulate(dt_proj_.get(), gDt);
    accumulate(z_proj_.get(), g_z);

    A.add_grad(grad_A);
    D.add_grad(grad_D);

    const bool rank_2 = grad_output.shape.size() == 2;
    Tensor gu_shaped = rank_2 ? g_u : g_u.reshape({batch, seq, dim});
    return grad_output.get_device() == Device::GPU ? gu_shaped.to(Device::GPU)
                                                   : gu_shaped;
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

    // Proper selective-SSM path (opt-in).  Handles rank-2/3 incl. seq==1; it
    // recomputes the full scan each call (no incremental streaming cache yet),
    // which is correct for training/prefill.  Bypasses the legacy streaming and
    // batched scan entirely so the corrected math is the single source of truth
    // when the flag is on.
    if (config_.proper_selective_ssm) {
        (void)ctx;
        return forward_proper(u);
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
                x_flat.raw_data(),
                delta_flat.raw_data(),
                A.data.raw_data(),
                streaming_state_->raw_data(),
                y_ssd.raw_data(),
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
    if (config_.proper_selective_ssm) {
        if (!proper_active_) {
            throw std::runtime_error(
                "Mamba2SSD proper backward called before forward_proper");
        }
        return backward_proper(grad_output);
    }
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
    // Proper-path forward caches.
    proper_active_ = false;
    pp_u_ = Tensor();
    pp_xv_ = Tensor();
    pp_conv_pre_ = Tensor();
    pp_xc_ = Tensor();
    pp_z_ = Tensor();
    pp_B_ = Tensor();
    pp_C_ = Tensor();
    pp_dt_ = Tensor();
    pp_h_hist_ = Tensor();
    pp_y_ssd_ = Tensor();
}

void Mamba2SSD::reset_runtime_telemetry() {
    gpu_fast_path_hits_ = 0;
    gpu_fast_path_fallbacks_ = 0;
    last_fallback_reason_.clear();
}

void Mamba2SSD::to(Device dev) {
    if (config_.proper_selective_ssm) {
        // Proper path v1 computes the conv/scan on the host and keeps its
        // projections (incl. out_proj) on CPU so the host math is device-
        // consistent; forward_proper bridges GPU inputs/outputs (it .cpu()s the
        // input and .to()s the result back).  A/D may live on either device —
        // forward/backward_proper copy them host-side.  A GPU-resident proper
        // path is a tracked follow-up (lands with the N-state CUDA kernel).
        A.data = A.data.to(dev);
        D.data = D.data.to(dev);
        return;
    }
    in_proj_robust.to(dev);
    in_proj_sensitive.to(dev);
    out_proj.to(dev);
    A.data = A.data.to(dev);
    D.data = D.data.to(dev);
}

std::vector<Parameter*> Mamba2SSD::parameters() {
    std::vector<Parameter*> params;
    auto add_proj = [&](BitLinear& bl, const char* prefix) {
        auto p = bl.parameters();
        prefix_parameter_names(p, prefix);
        params.insert(params.end(), p.begin(), p.end());
    };
    if (config_.proper_selective_ssm) {
        // Proper path: independent x/z/B/C/dt projections + shared out_proj +
        // depthwise conv kernel.  The legacy in_proj_robust/sensitive are NOT
        // exposed here (they are unused on this path), so the optimizer never
        // touches dead parameters.
        add_proj(*x_proj_, "x_proj.");
        add_proj(*z_proj_, "z_proj.");
        add_proj(*B_proj_, "B_proj.");
        add_proj(*C_proj_, "C_proj.");
        add_proj(*dt_proj_, "dt_proj.");
        add_proj(out_proj, "out_proj.");
        conv_weight_.base_name = "conv1d_weight";
        conv_weight_.name = "conv1d_weight";
        params.push_back(&conv_weight_);
    } else {
        add_proj(in_proj_robust, "in_proj_robust.");
        add_proj(in_proj_sensitive, "in_proj_sensitive.");
        add_proj(out_proj, "out_proj.");
    }
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
    if (config_.proper_selective_ssm) {
        out.push_back(x_proj_.get());
        out.push_back(z_proj_.get());
        out.push_back(B_proj_.get());
        out.push_back(C_proj_.get());
        out.push_back(dt_proj_.get());
        out.push_back(&out_proj);
        return;
    }
    out.push_back(&in_proj_robust);
    out.push_back(&in_proj_sensitive);
    out.push_back(&out_proj);
}

} // namespace nsos
