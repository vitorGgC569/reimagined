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

  // N-state (full Mamba-2 SSD) uses a PER-HEAD decay A ∈ R^H instead of the
  // per-channel A ∈ R^{d_model}.  Resize BEFORE the optional log-spaced init so
  // that init fills the correct length.
  const bool proper_nstate_ctor =
      config_.proper_selective_ssm && config_.proper_state_expansion;
  if (proper_nstate_ctor) {
    A.data = Tensor::ones({std::max(n_heads, 1)}, Device::CPU);
  }

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
    if (proper_nstate_ctor) {
      // Full SSD: dt and A are per-head; B and C are per-head N-dimensional.
      const int N = std::max(d_state, 1);
      dt_proj_ = std::make_unique<BitLinear>(d_model, std::max(n_heads, 1), true);
      B_proj_  = std::make_unique<BitLinear>(d_model, std::max(n_heads, 1) * N, true);
      C_proj_  = std::make_unique<BitLinear>(d_model, std::max(n_heads, 1) * N, true);
    } else {
      B_proj_  = std::make_unique<BitLinear>(d_model, d_model, true);
      C_proj_  = std::make_unique<BitLinear>(d_model, d_model, true);
      dt_proj_ = std::make_unique<BitLinear>(d_model, d_model, true);
    }
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
    const Device dev = u.get_device();  // GPU-first: stay on the input's device

    Tensor u_flat = rank_2 ? u : u.reshape({rows, dim});

    // Independent projections — BitLinear is device-agnostic (GPU or CPU).
    Tensor xv = x_proj_->forward(u_flat).reshape({rows, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({rows, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({rows, dim});
    Tensor Ct = C_proj_->forward(u_flat).reshape({rows, dim});
    Tensor dt = dt_proj_->forward(u_flat).reshape({rows, dim});

    // Causal depthwise conv on x (GPU kernel on device; ordered host loop on CPU).
    Tensor conv_pre(std::vector<int>{rows, dim}, dev);
#ifdef USE_CUDA
    if (dev == Device::GPU) {
        cuda::launch_conv1d_causal_forward(xv.raw_data(),
                                           conv_weight_.data.raw_data(),
                                           conv_pre.raw_data(), batch, seq, dim, K);
    } else
#endif
    {
        conv1d_causal_forward(xv.data(), conv_weight_.data.data(),
                              conv_pre.data(), batch, seq, dim, K);
    }

    // xc = silu(conv_pre) = conv_pre * sigmoid(conv_pre) — device-agnostic.
    Tensor xc = conv_pre.mul(conv_pre.sigmoid());

    // Diagonal selective recurrence: h_t = decay_t*h_{t-1} + B_t*xc_t;
    // y_t = h_t * C_t (linear readout, C once).  GPU kernel on device; the
    // sequential recurrence runs as an ordered host loop on CPU.
    Tensor h_hist(std::vector<int>{rows, dim}, dev);
    Tensor y_ssd(std::vector<int>{rows, dim}, dev);
#ifdef USE_CUDA
    if (dev == Device::GPU) {
        cuda::launch_mamba_proper_scan_forward(
            xc.raw_data(), dt.raw_data(), A.data.raw_data(), Bt.raw_data(),
            Ct.raw_data(), y_ssd.raw_data(), h_hist.raw_data(), batch, seq, dim);
    } else
#endif
    {
        const float* a_ptr = A.data.data();
        const float* b_ptr = Bt.data();
        const float* c_ptr = Ct.data();
        const float* dt_ptr = dt.data();
        const float* xc_ptr = xc.data();
        float* h_ptr = h_hist.data();
        float* y_ptr = y_ssd.data();
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
                        decay * state[static_cast<size_t>(c)] +
                        b_ptr[idx] * xc_ptr[idx];
                    h_ptr[idx] = state[static_cast<size_t>(c)];
                    y_ptr[idx] = state[static_cast<size_t>(c)] * c_ptr[idx];
                }
            }
        }
    }

    // Separate SiLU gate, out projection and D skip — all device-agnostic.
    Tensor gated = y_ssd.mul(z.mul(z.sigmoid()));
    Tensor projected = out_proj.forward(gated).reshape({rows, dim});
    Tensor skip = u_flat.mul(D.data);  // D is [dim], broadcast over rows
    Tensor result = projected.add(skip);

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

    // Prime the incremental stream: carry the final SSD state + the last (K-1)
    // x_proj taps so subsequent single tokens decode in O(1) (forward_proper_step).
    if (streaming_inference_) {
        Tensor h_h = dev == Device::GPU ? h_hist.cpu() : h_hist;
        Tensor xv_h = dev == Device::GPU ? xv.cpu() : xv;
        pp_stream_state_.assign(static_cast<size_t>(dim), 0.0f);
        const float* hp = h_h.data();
        for (int c = 0; c < dim; ++c) {
            pp_stream_state_[static_cast<size_t>(c)] =
                hp[static_cast<size_t>(rows - 1) * dim + c];
        }
        const int taps = std::max(K - 1, 0);
        pp_stream_ring_.assign(static_cast<size_t>(taps) * dim, 0.0f);
        const float* xvp = xv_h.data();
        for (int s = 0; s < taps; ++s) {
            const int src_row = rows - taps + s;
            if (src_row < 0) continue;
            for (int c = 0; c < dim; ++c) {
                pp_stream_ring_[static_cast<size_t>(s) * dim + c] =
                    xvp[static_cast<size_t>(src_row) * dim + c];
            }
        }
        pp_stream_active_ = true;
    }

    return rank_2 ? result : result.reshape({batch, seq, dim});
}

Tensor Mamba2SSD::backward_proper(const Tensor& grad_output) {
    const int batch = pp_batch_;
    const int seq = pp_seq_;
    const int dim = d_model;
    const int K = conv_kernel_;
    const int rows = batch * seq;
    const Device dev = grad_output.get_device();  // GPU-first: stay on device

    Tensor g = grad_output.shape.size() == 2 ? grad_output
                                             : grad_output.reshape({rows, dim});

    // SiLU derivative d/dx[x·σ(x)] = σ(x)·(1 + x·(1 - σ(x))) — device-agnostic.
    auto dsilu = [&](const Tensor& pre) {
        Tensor s = pre.sigmoid();
        Tensor one = Tensor::ones(std::vector<int>{rows, dim}, dev);
        return s.mul(one.add(pre.mul(one.sub(s))));
    };

    // Skip path: result = out_proj(...) + u*D.  grad_D = Σ_rows(g·u); the input
    // gradient via the skip is g·D (broadcast).  Device-agnostic Tensor ops.
    Tensor grad_D = g.mul(pp_u_).sum(0, false);
    Tensor g_u = g.mul(D.data);

    // out_proj backward -> grad wrt gated.
    Tensor g_gated = out_proj.backward(g).reshape({rows, dim});

    // Gate y = y_ssd · silu(z).
    Tensor silu_z = pp_z_.mul(pp_z_.sigmoid());
    Tensor g_yssd = g_gated.mul(silu_z);
    Tensor g_z = g_gated.mul(pp_y_ssd_).mul(dsilu(pp_z_));

    // Scan backward (linear readout): GPU kernel on device, ordered host loop on
    // CPU.  Produces grads wrt the scan input xc, dt, A, B and C.
    Tensor g_xc = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor g_dt = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor g_B = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor g_C = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor grad_A = Tensor::zeros({dim}, dev);
#ifdef USE_CUDA
    // Deterministic mode: the GPU scan backward accumulates grad_A via atomicAdd
    // (order-nondeterministic); fall to the ordered host loop below.
    if (dev == Device::GPU && !determinism::deterministic_reductions_enabled()) {
        cuda::launch_mamba_proper_scan_backward(
            g_yssd.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
            A.data.raw_data(), pp_B_.raw_data(), pp_C_.raw_data(),
            pp_h_hist_.raw_data(), g_xc.raw_data(), g_dt.raw_data(),
            grad_A.raw_data(), g_B.raw_data(), g_C.raw_data(), batch, seq, dim);
    } else
#endif
    {
        const float* gy = g_yssd.data();
        const float* h_ptr = pp_h_hist_.data();
        const float* c_ptr = pp_C_.data();
        const float* b_ptr = pp_B_.data();
        const float* dt_ptr = pp_dt_.data();
        const float* xc_ptr = pp_xc_.data();
        const float* a_ptr = A.data.data();
        float* gBp = g_B.data();
        float* gCp = g_C.data();
        float* gDtp = g_dt.data();
        float* gXcp = g_xc.data();
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
                    gCp[idx] = gy[idx] * h_t;
                    const float grad_h =
                        gy[idx] * c_ptr[idx] + carry[static_cast<size_t>(c)];
                    gBp[idx] = grad_h * xc_ptr[idx];
                    gXcp[idx] = grad_h * b_ptr[idx];
                    const float grad_decay = grad_h * h_prev;
                    gDtp[idx] = grad_decay * decay * (-a_value) *
                                sigmoid_stable(dt_ptr[idx]);
                    if (a_ptr[c] > 1e-3f) {
                        gA[c] += grad_decay * decay * (-sp);
                    }
                    carry[static_cast<size_t>(c)] = grad_h * decay;
                }
            }
        }
    }

    // xc = silu(conv_pre) -> grad_conv_pre (device-agnostic).
    Tensor grad_conv_pre = g_xc.mul(dsilu(pp_conv_pre_));

    // conv1d backward: GPU kernel on device, ordered host loop on CPU.
    Tensor grad_xv = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor grad_conv_w = Tensor::zeros({dim, K}, dev);
#ifdef USE_CUDA
    if (dev == Device::GPU && !determinism::deterministic_reductions_enabled()) {
        cuda::launch_conv1d_causal_backward(
            grad_conv_pre.raw_data(), pp_xv_.raw_data(),
            conv_weight_.data.raw_data(), grad_xv.raw_data(),
            grad_conv_w.raw_data(), batch, seq, dim, K);
    } else
#endif
    {
        conv1d_causal_backward(grad_conv_pre.data(), pp_xv_.data(),
                               conv_weight_.data.data(), grad_xv.data(),
                               grad_conv_w.data(), batch, seq, dim, K);
    }
    conv_weight_.add_grad(grad_conv_w);

    // Project grads back to the layer input and accumulate projection params.
    // All BitLinear backwards are device-agnostic; the adds run on `dev`.
    g_u = g_u.add(x_proj_->backward(grad_xv).reshape({rows, dim}));
    g_u = g_u.add(B_proj_->backward(g_B).reshape({rows, dim}));
    g_u = g_u.add(C_proj_->backward(g_C).reshape({rows, dim}));
    g_u = g_u.add(dt_proj_->backward(g_dt).reshape({rows, dim}));
    g_u = g_u.add(z_proj_->backward(g_z).reshape({rows, dim}));

    A.add_grad(grad_A);
    D.add_grad(grad_D);

    const bool rank_2 = grad_output.shape.size() == 2;
    return rank_2 ? g_u : g_u.reshape({batch, seq, dim});
}

// ===========================================================================
// Full Mamba-2 SSD with N-dimensional state expansion (proper_state_expansion).
// State h ∈ R^{H×P×N}; B,C per-head N-dim; dt,A per-head; linear readout
//   y_{t,h,p} = Σ_n h_{t,h,p,n} C_{t,h,n}.  Scan + BPTT on host (v1); the conv
// and projections/gate use the device-aware kernels/Tensor ops.  All gradients
// hand-derived; gradchecked (check_mamba2_nstate).
// ===========================================================================
Tensor Mamba2SSD::forward_proper_nstate(const Tensor& u) {
    const bool rank_2 = u.shape.size() == 2;
    const int batch = rank_2 ? 1 : u.shape[0];
    const int seq = rank_2 ? u.shape[0] : u.shape[1];
    const int dim = u.shape.back();
    if (dim != d_model) {
        throw std::runtime_error("Mamba2SSD nstate: last dim != d_model");
    }
    const int H = std::max(n_heads, 1);
    const int P = d_head;
    const int N = std::max(d_state, 1);
    if (H * P != dim) {
        throw std::runtime_error("Mamba2SSD nstate: d_model must equal n_heads*d_head");
    }
    const int rows = batch * seq;
    const int K = conv_kernel_;
    const Device dev = u.get_device();

    Tensor u_flat = rank_2 ? u : u.reshape({rows, dim});

    // Projections (device-agnostic).  x,z:[rows,dim]; B,C:[rows,H*N]; dt:[rows,H].
    Tensor xv = x_proj_->forward(u_flat).reshape({rows, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({rows, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({rows, H * N});
    Tensor Ct = C_proj_->forward(u_flat).reshape({rows, H * N});
    Tensor dt = dt_proj_->forward(u_flat).reshape({rows, H});

    // Causal conv on x then SiLU.
    Tensor conv_pre(std::vector<int>{rows, dim}, dev);
#ifdef USE_CUDA
    if (dev == Device::GPU) {
        cuda::launch_conv1d_causal_forward(xv.raw_data(),
                                           conv_weight_.data.raw_data(),
                                           conv_pre.raw_data(), batch, seq, dim, K);
    } else
#endif
    {
        conv1d_causal_forward(xv.data(), conv_weight_.data.data(),
                              conv_pre.data(), batch, seq, dim, K);
    }
    Tensor xc = conv_pre.mul(conv_pre.sigmoid());

    // ── N-state SSD scan ── GPU-resident kernel (state in registers) on device;
    // ordered host loop on CPU or when N exceeds the kernel's MAX_N.  Full state
    // history saved for BPTT.
    Tensor y_ssd(std::vector<int>{rows, dim}, dev);
    Tensor hist(std::vector<int>{rows, H, P, N}, dev);
    bool nstate_scan_done = false;
#ifdef USE_CUDA
    if (dev == Device::GPU && N <= cuda::mamba_nstate_max_n()) {
        cuda::launch_mamba_nstate_forward(
            xc.raw_data(), dt.raw_data(), A.data.raw_data(), Bt.raw_data(),
            Ct.raw_data(), y_ssd.raw_data(), hist.raw_data(), batch, seq, H, P, N);
        nstate_scan_done = true;
    }
#endif
    if (!nstate_scan_done) {
    Tensor xc_h = xc.get_device() == Device::GPU ? xc.cpu() : xc;
    Tensor Bt_h = Bt.get_device() == Device::GPU ? Bt.cpu() : Bt;
    Tensor Ct_h = Ct.get_device() == Device::GPU ? Ct.cpu() : Ct;
    Tensor dt_h = dt.get_device() == Device::GPU ? dt.cpu() : dt;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    Tensor y_h(std::vector<int>{rows, dim}, Device::CPU);
    Tensor hist_h(std::vector<int>{rows, H, P, N}, Device::CPU);
    const float* xcp = xc_h.data();
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();
    float* yp = y_h.data();
    float* histp = hist_h.data();
    const size_t HPN = static_cast<size_t>(H) * P * N;
    std::vector<float> state(HPN, 0.0f);
    for (int b = 0; b < batch; ++b) {
        std::fill(state.begin(), state.end(), 0.0f);
        for (int t = 0; t < seq; ++t) {
            const int row = b * seq + t;
            for (int h = 0; h < H; ++h) {
                const float a_value = std::max(ap[h], 1e-3f);
                const float decay =
                    std::exp(-softplus_stable(dtp[row * H + h]) * a_value);
                for (int p = 0; p < P; ++p) {
                    const int chan = h * P + p;
                    const float xcv = xcp[static_cast<size_t>(row) * dim + chan];
                    float y_acc = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        const size_t sidx = (static_cast<size_t>(h) * P + p) * N + n;
                        const float bval = bp[static_cast<size_t>(row) * (H * N) + h * N + n];
                        const float cval = cp[static_cast<size_t>(row) * (H * N) + h * N + n];
                        const float hv = decay * state[sidx] + bval * xcv;
                        state[sidx] = hv;
                        histp[static_cast<size_t>(row) * HPN + sidx] = hv;
                        y_acc += hv * cval;
                    }
                    yp[static_cast<size_t>(row) * dim + chan] = y_acc;
                }
            }
        }
    }
        y_ssd = dev == Device::GPU ? y_h.to(Device::GPU) : y_h;
        hist = dev == Device::GPU ? hist_h.to(Device::GPU) : hist_h;
    }

    // Gate + out_proj + skip (device-agnostic).
    Tensor gated = y_ssd.mul(z.mul(z.sigmoid()));
    Tensor projected = out_proj.forward(gated).reshape({rows, dim});
    Tensor skip = u_flat.mul(D.data);
    Tensor result = projected.add(skip);

    pp_u_ = u_flat;
    pp_xv_ = xv;
    pp_conv_pre_ = conv_pre;
    pp_xc_ = xc;
    pp_z_ = z;
    pp_B_ = Bt;
    pp_C_ = Ct;
    pp_dt_ = dt;
    pp_y_ssd_ = y_ssd;
    pp_state_hist_ = hist;
    pp_batch_ = batch;
    pp_seq_ = seq;
    proper_active_ = true;

    // Prime the incremental stream (nstate): carry the final H*P*N state + last
    // (K-1) x_proj taps for O(1) single-token decode (forward_proper_nstate_step).
    if (streaming_inference_) {
        Tensor st_h = dev == Device::GPU ? hist.cpu() : hist;
        Tensor xv_h = dev == Device::GPU ? xv.cpu() : xv;
        const size_t HPN = static_cast<size_t>(H) * P * N;
        pp_stream_state_.assign(HPN, 0.0f);
        const float* hp = st_h.data();
        for (size_t k = 0; k < HPN; ++k) {
            pp_stream_state_[k] = hp[static_cast<size_t>(rows - 1) * HPN + k];
        }
        const int taps = std::max(K - 1, 0);
        pp_stream_ring_.assign(static_cast<size_t>(taps) * dim, 0.0f);
        const float* xvp = xv_h.data();
        for (int s = 0; s < taps; ++s) {
            const int src_row = rows - taps + s;
            if (src_row < 0) continue;
            for (int c = 0; c < dim; ++c) {
                pp_stream_ring_[static_cast<size_t>(s) * dim + c] =
                    xvp[static_cast<size_t>(src_row) * dim + c];
            }
        }
        pp_stream_active_ = true;
    }

    return rank_2 ? result : result.reshape({batch, seq, dim});
}

Tensor Mamba2SSD::backward_proper_nstate(const Tensor& grad_output) {
    const int batch = pp_batch_;
    const int seq = pp_seq_;
    const int dim = d_model;
    const int K = conv_kernel_;
    const int rows = batch * seq;
    const int H = std::max(n_heads, 1);
    const int P = d_head;
    const int N = std::max(d_state, 1);
    const Device dev = grad_output.get_device();
    Tensor g = grad_output.shape.size() == 2 ? grad_output
                                             : grad_output.reshape({rows, dim});

    auto dsilu = [&](const Tensor& pre) {
        Tensor s = pre.sigmoid();
        Tensor one = Tensor::ones(std::vector<int>{rows, dim}, dev);
        return s.mul(one.add(pre.mul(one.sub(s))));
    };

    // Skip + out_proj + gate (device-agnostic).
    Tensor grad_D = g.mul(pp_u_).sum(0, false);
    Tensor g_u = g.mul(D.data);
    Tensor g_gated = out_proj.backward(g).reshape({rows, dim});
    Tensor silu_z = pp_z_.mul(pp_z_.sigmoid());
    Tensor g_yssd = g_gated.mul(silu_z);
    Tensor g_z = g_gated.mul(pp_y_ssd_).mul(dsilu(pp_z_));

    // ── N-state SSD scan backward ── GPU-resident kernel on device; ordered
    // host loop on CPU or when N exceeds the kernel's MAX_N.
    Tensor gXc(std::vector<int>{rows, dim}, dev);
    Tensor gB(std::vector<int>{rows, H * N}, dev);
    Tensor gC(std::vector<int>{rows, H * N}, dev);
    Tensor gDt(std::vector<int>{rows, H}, dev);
    Tensor gA(std::vector<int>{H}, dev);
    bool nstate_bwd_done = false;
#ifdef USE_CUDA
    if (dev == Device::GPU && N <= cuda::mamba_nstate_max_n() &&
        !determinism::deterministic_reductions_enabled()) {
        cuda::launch_mamba_nstate_backward(
            g_yssd.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
            A.data.raw_data(), pp_B_.raw_data(), pp_C_.raw_data(),
            pp_state_hist_.raw_data(), gXc.raw_data(), gDt.raw_data(),
            gA.raw_data(), gB.raw_data(), gC.raw_data(), batch, seq, H, P, N);
        nstate_bwd_done = true;
    }
#endif
    if (!nstate_bwd_done) {
    Tensor gy_h = g_yssd.get_device() == Device::GPU ? g_yssd.cpu() : g_yssd;
    Tensor xc_h = pp_xc_.get_device() == Device::GPU ? pp_xc_.cpu() : pp_xc_;
    Tensor Bt_h = pp_B_.get_device() == Device::GPU ? pp_B_.cpu() : pp_B_;
    Tensor Ct_h = pp_C_.get_device() == Device::GPU ? pp_C_.cpu() : pp_C_;
    Tensor dt_h = pp_dt_.get_device() == Device::GPU ? pp_dt_.cpu() : pp_dt_;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    const float* gyp = gy_h.data();
    const float* xcp = xc_h.data();
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();
    const float* histp = pp_state_hist_.data();
    Tensor gB_h = Tensor::zeros({rows, H * N}, Device::CPU);
    Tensor gC_h = Tensor::zeros({rows, H * N}, Device::CPU);
    Tensor gDt_h = Tensor::zeros({rows, H}, Device::CPU);
    Tensor gXc_h = Tensor::zeros({rows, dim}, Device::CPU);
    Tensor gA_h = Tensor::zeros({H}, Device::CPU);
    float* gBp = gB_h.data();
    float* gCp = gC_h.data();
    float* gDtp = gDt_h.data();
    float* gXcp = gXc_h.data();
    float* gAp = gA_h.data();
    const size_t HPN = static_cast<size_t>(H) * P * N;
    std::vector<float> carry(HPN, 0.0f);
    for (int b = 0; b < batch; ++b) {
        std::fill(carry.begin(), carry.end(), 0.0f);
        for (int t = seq - 1; t >= 0; --t) {
            const int row = b * seq + t;
            const int prevrow = b * seq + (t - 1);
            for (int h = 0; h < H; ++h) {
                const float a_value = std::max(ap[h], 1e-3f);
                const float sp = softplus_stable(dtp[row * H + h]);
                const float decay = std::exp(-sp * a_value);
                float ddecay = 0.0f;
                for (int p = 0; p < P; ++p) {
                    const int chan = h * P + p;
                    const float xcv = xcp[static_cast<size_t>(row) * dim + chan];
                    const float gyv = gyp[static_cast<size_t>(row) * dim + chan];
                    for (int n = 0; n < N; ++n) {
                        const size_t sidx = (static_cast<size_t>(h) * P + p) * N + n;
                        const float h_t = histp[static_cast<size_t>(row) * HPN + sidx];
                        const float h_prev =
                            t == 0 ? 0.0f
                                   : histp[static_cast<size_t>(prevrow) * HPN + sidx];
                        const size_t bcidx = static_cast<size_t>(row) * (H * N) + h * N + n;
                        const float cval = cp[bcidx];
                        const float bval = bp[bcidx];
                        gCp[bcidx] += gyv * h_t;
                        const float grad_h = gyv * cval + carry[sidx];
                        gBp[bcidx] += grad_h * xcv;
                        gXcp[static_cast<size_t>(row) * dim + chan] += grad_h * bval;
                        ddecay += grad_h * h_prev;
                        carry[sidx] = grad_h * decay;
                    }
                }
                gDtp[row * H + h] =
                    ddecay * decay * (-a_value) * sigmoid_stable(dtp[row * H + h]);
                if (ap[h] > 1e-3f) {
                    gAp[h] += ddecay * decay * (-sp);
                }
            }
        }
    }

        gXc = dev == Device::GPU ? gXc_h.to(Device::GPU) : gXc_h;
        gB = dev == Device::GPU ? gB_h.to(Device::GPU) : gB_h;
        gC = dev == Device::GPU ? gC_h.to(Device::GPU) : gC_h;
        gDt = dev == Device::GPU ? gDt_h.to(Device::GPU) : gDt_h;
        gA = dev == Device::GPU ? gA_h.to(Device::GPU) : gA_h;
    }
    A.add_grad(gA);

    // conv SiLU backward + conv1d backward.
    Tensor grad_conv_pre = gXc.mul(dsilu(pp_conv_pre_));
    Tensor grad_xv = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor grad_conv_w = Tensor::zeros({dim, K}, dev);
#ifdef USE_CUDA
    if (dev == Device::GPU && !determinism::deterministic_reductions_enabled()) {
        cuda::launch_conv1d_causal_backward(
            grad_conv_pre.raw_data(), pp_xv_.raw_data(),
            conv_weight_.data.raw_data(), grad_xv.raw_data(),
            grad_conv_w.raw_data(), batch, seq, dim, K);
    } else
#endif
    {
        conv1d_causal_backward(grad_conv_pre.data(), pp_xv_.data(),
                               conv_weight_.data.data(), grad_xv.data(),
                               grad_conv_w.data(), batch, seq, dim, K);
    }
    conv_weight_.add_grad(grad_conv_w);

    // Projections backward into the layer input.
    g_u = g_u.add(x_proj_->backward(grad_xv).reshape({rows, dim}));
    g_u = g_u.add(B_proj_->backward(gB).reshape({rows, dim}));
    g_u = g_u.add(C_proj_->backward(gC).reshape({rows, dim}));
    g_u = g_u.add(dt_proj_->backward(gDt).reshape({rows, dim}));
    g_u = g_u.add(z_proj_->backward(g_z).reshape({rows, dim}));
    D.add_grad(grad_D);

    const bool rank_2 = grad_output.shape.size() == 2;
    return rank_2 ? g_u : g_u.reshape({batch, seq, dim});
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

Tensor Mamba2SSD::proper_stream_conv_(const Tensor& xv_host) {
    // xv_host: host [1,dim].  Builds the K-tap causal window (carried taps +
    // current x), runs the same host conv1d the full scan uses, returns
    // xc=silu(conv_pre) for the current token, then advances the window.
    const int dim = d_model;
    const int K = std::max(conv_kernel_, 1);
    Tensor cw_h = conv_weight_.data.get_device() == Device::GPU
                      ? conv_weight_.data.cpu()
                      : conv_weight_.data;
    Tensor win(std::vector<int>{1, K, dim}, Device::CPU);
    float* wptr = win.data();
    const bool have_ring =
        static_cast<int>(pp_stream_ring_.size()) == (K - 1) * dim;
    for (int s = 0; s < K - 1; ++s) {
        for (int c = 0; c < dim; ++c) {
            wptr[static_cast<size_t>(s) * dim + c] =
                have_ring ? pp_stream_ring_[static_cast<size_t>(s) * dim + c]
                          : 0.0f;
        }
    }
    const float* xvp = xv_host.data();
    for (int c = 0; c < dim; ++c) {
        wptr[static_cast<size_t>(K - 1) * dim + c] = xvp[c];
    }
    Tensor conv_full(std::vector<int>{1, K, dim}, Device::CPU);
    conv1d_causal_forward(win.data(), cw_h.data(), conv_full.data(), 1, K, dim, K);
    Tensor conv_pre(std::vector<int>{1, dim}, Device::CPU);
    std::memcpy(conv_pre.data(),
                conv_full.data() + static_cast<size_t>(K - 1) * dim,
                static_cast<size_t>(dim) * sizeof(float));
    Tensor xc = conv_pre.mul(conv_pre.sigmoid());
    // Advance the window: drop the oldest tap, append the current x.
    if (K - 1 > 0) {
        if (!have_ring) {
            pp_stream_ring_.assign(static_cast<size_t>(K - 1) * dim, 0.0f);
        }
        for (int s = 0; s + 1 < K - 1; ++s) {
            for (int c = 0; c < dim; ++c) {
                pp_stream_ring_[static_cast<size_t>(s) * dim + c] =
                    pp_stream_ring_[static_cast<size_t>(s + 1) * dim + c];
            }
        }
        for (int c = 0; c < dim; ++c) {
            pp_stream_ring_[static_cast<size_t>(K - 2) * dim + c] = xvp[c];
        }
    }
    return xc;
}

Tensor Mamba2SSD::forward_proper_step(const Tensor& u) {
    const int dim = d_model;
    const Device dev = u.get_device();
    Tensor u_flat = u.reshape({1, dim});

    // Projections (device-agnostic), pulled to host for the scalar recurrence.
    Tensor xv = x_proj_->forward(u_flat).reshape({1, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({1, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({1, dim});
    Tensor Ct = C_proj_->forward(u_flat).reshape({1, dim});
    Tensor dt = dt_proj_->forward(u_flat).reshape({1, dim});
    Tensor xv_h = dev == Device::GPU ? xv.cpu() : xv;
    Tensor Bt_h = dev == Device::GPU ? Bt.cpu() : Bt;
    Tensor Ct_h = dev == Device::GPU ? Ct.cpu() : Ct;
    Tensor dt_h = dev == Device::GPU ? dt.cpu() : dt;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;

    Tensor xc = proper_stream_conv_(xv_h);  // host [1,dim]
    const float* xcp = xc.data();

    if (static_cast<int>(pp_stream_state_.size()) != dim) {
        pp_stream_state_.assign(static_cast<size_t>(dim), 0.0f);
    }
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();
    Tensor y_ssd(std::vector<int>{1, dim}, Device::CPU);
    float* yp = y_ssd.data();
    for (int c = 0; c < dim; ++c) {
        const float a_value = std::max(ap[c], 1e-3f);
        const float decay = std::exp(-softplus_stable(dtp[c]) * a_value);
        const float st =
            decay * pp_stream_state_[static_cast<size_t>(c)] + bp[c] * xcp[c];
        pp_stream_state_[static_cast<size_t>(c)] = st;
        yp[c] = st * cp[c];
    }

    // Gate on host (parity), then project + skip on the input device.
    Tensor z_h = dev == Device::GPU ? z.cpu() : z;
    Tensor gated_h = y_ssd.mul(z_h.mul(z_h.sigmoid()));
    Tensor gated = dev == Device::GPU ? gated_h.to(Device::GPU) : gated_h;
    Tensor projected = out_proj.forward(gated).reshape({1, dim});
    Tensor skip = u_flat.mul(D.data);
    Tensor result = projected.add(skip);
    return u.shape.size() == 2 ? result : result.reshape({1, 1, dim});
}

Tensor Mamba2SSD::forward_proper_nstate_step(const Tensor& u) {
    const int dim = d_model;
    const int H = std::max(n_heads, 1);
    const int P = d_head;
    const int N = std::max(d_state, 1);
    const Device dev = u.get_device();
    Tensor u_flat = u.reshape({1, dim});

    Tensor xv = x_proj_->forward(u_flat).reshape({1, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({1, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({1, H * N});
    Tensor Ct = C_proj_->forward(u_flat).reshape({1, H * N});
    Tensor dt = dt_proj_->forward(u_flat).reshape({1, H});
    Tensor xv_h = dev == Device::GPU ? xv.cpu() : xv;
    Tensor Bt_h = dev == Device::GPU ? Bt.cpu() : Bt;
    Tensor Ct_h = dev == Device::GPU ? Ct.cpu() : Ct;
    Tensor dt_h = dev == Device::GPU ? dt.cpu() : dt;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;

    Tensor xc = proper_stream_conv_(xv_h);  // host [1,dim]
    const float* xcp = xc.data();

    const size_t HPN = static_cast<size_t>(H) * P * N;
    if (pp_stream_state_.size() != HPN) {
        pp_stream_state_.assign(HPN, 0.0f);
    }
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();
    Tensor y_ssd(std::vector<int>{1, dim}, Device::CPU);
    float* yp = y_ssd.data();
    for (int h = 0; h < H; ++h) {
        const float a_value = std::max(ap[h], 1e-3f);
        const float decay = std::exp(-softplus_stable(dtp[h]) * a_value);
        for (int p = 0; p < P; ++p) {
            const int chan = h * P + p;
            const float xcv = xcp[chan];
            float y_acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                const size_t sidx = (static_cast<size_t>(h) * P + p) * N + n;
                const float bval = bp[static_cast<size_t>(h) * N + n];
                const float cval = cp[static_cast<size_t>(h) * N + n];
                const float hv = decay * pp_stream_state_[sidx] + bval * xcv;
                pp_stream_state_[sidx] = hv;
                y_acc += hv * cval;
            }
            yp[chan] = y_acc;
        }
    }

    Tensor z_h = dev == Device::GPU ? z.cpu() : z;
    Tensor gated_h = y_ssd.mul(z_h.mul(z_h.sigmoid()));
    Tensor gated = dev == Device::GPU ? gated_h.to(Device::GPU) : gated_h;
    Tensor projected = out_proj.forward(gated).reshape({1, dim});
    Tensor skip = u_flat.mul(D.data);
    Tensor result = projected.add(skip);
    return u.shape.size() == 2 ? result : result.reshape({1, 1, dim});
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
        // Incremental single-token decode: once the prefill has primed the stream
        // (pp_stream_active_), each subsequent single token advances the carried
        // SSD state + conv window in O(1) instead of re-scanning the prefix.
        const bool single =
            (u.shape.size() == 2 && u.shape[0] == 1) ||
            (u.shape.size() == 3 && u.shape[0] == 1 && u.shape[1] == 1);
        if (streaming_inference_ && single && pp_stream_active_) {
            return config_.proper_state_expansion ? forward_proper_nstate_step(u)
                                                  : forward_proper_step(u);
        }
        return config_.proper_state_expansion ? forward_proper_nstate(u)
                                              : forward_proper(u);
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
        return config_.proper_state_expansion
                   ? backward_proper_nstate(grad_output)
                   : backward_proper(grad_output);
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
    pp_state_hist_ = Tensor();
    // Proper-path incremental decode cache.
    pp_stream_state_.clear();
    pp_stream_ring_.clear();
    pp_stream_active_ = false;
}

void Mamba2SSD::reset_runtime_telemetry() {
    gpu_fast_path_hits_ = 0;
    gpu_fast_path_fallbacks_ = 0;
    last_fallback_reason_.clear();
}

void Mamba2SSD::to(Device dev) {
    if (config_.proper_selective_ssm) {
        // GPU-first: move ALL proper-path components onto the device so the
        // forward/backward run fully on-device (conv1d + linear-readout scan
        // CUDA kernels; projections via BitLinear GPU path; gate/skip via Tensor
        // GPU ops).  No host round-trip.
        if (x_proj_) x_proj_->to(dev);
        if (z_proj_) z_proj_->to(dev);
        if (B_proj_) B_proj_->to(dev);
        if (C_proj_) C_proj_->to(dev);
        if (dt_proj_) dt_proj_->to(dev);
        out_proj.to(dev);
        if (conv_weight_.data.size > 0) {
            conv_weight_.data = conv_weight_.data.to(dev);
        }
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
    // Each streaming toggle starts a fresh proper-path stream: drop the carried
    // SSD state + conv window so the next prefill re-primes from zero.
    pp_stream_state_.clear();
    pp_stream_ring_.clear();
    pp_stream_active_ = false;
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
