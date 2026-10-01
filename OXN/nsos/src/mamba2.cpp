#include "../include/mamba2.h"
#include "../include/gpu_execution.h"
#include "../include/cuda/mamba_kernels.cuh"
#include "../include/jamba_utils.h"
#include "../include/nsos/determinism.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace nsos {

namespace {

#ifdef USE_CUDA
// Event-based, opt-in stage timer for faithful-Mamba diagnosis.  It adds no
// events, fences or output unless NSOS_MAMBA_STAGE_TIMING=1.  A single pair is
// reused between marks so diagnostic VRAM/resource use stays bounded.
class MambaStageTimer {
public:
    explicit MambaStageTimer(Device device)
        : enabled_(device == Device::GPU && [] {
              const char* value =
                  std::getenv("NSOS_MAMBA_STAGE_TIMING");
              return value != nullptr && value[0] == '1';
          }()) {
        if (!enabled_) return;
        const cudaError_t start_status = cudaEventCreate(&start_);
        const cudaError_t stop_status = cudaEventCreate(&stop_);
        if (start_status != cudaSuccess || stop_status != cudaSuccess) {
            if (start_ != nullptr) {
                gpu::report_cleanup_status(
                    cudaEventDestroy(start_),
                    "Mamba timer start-event rollback");
            }
            if (stop_ != nullptr) {
                gpu::report_cleanup_status(
                    cudaEventDestroy(stop_),
                    "Mamba timer stop-event rollback");
            }
            start_ = nullptr;
            stop_ = nullptr;
            throw std::runtime_error(
                "Mamba stage timing event creation failed");
        }
        if (cudaEventRecord(start_, nsos::gpu::current_stream()) != cudaSuccess) {
            gpu::report_cleanup_status(
                cudaEventDestroy(start_),
                "Mamba timer start-event rollback");
            gpu::report_cleanup_status(
                cudaEventDestroy(stop_),
                "Mamba timer stop-event rollback");
            start_ = nullptr;
            stop_ = nullptr;
            throw std::runtime_error(
                "Mamba stage timing start record failed");
        }
    }

    MambaStageTimer(const MambaStageTimer&) = delete;
    MambaStageTimer& operator=(const MambaStageTimer&) = delete;

    ~MambaStageTimer() noexcept {
        if (start_ != nullptr) {
            gpu::report_cleanup_status(
                cudaEventDestroy(start_), "Mamba timer start-event release");
        }
        if (stop_ != nullptr) {
            gpu::report_cleanup_status(
                cudaEventDestroy(stop_), "Mamba timer stop-event release");
        }
    }

    void mark(std::string_view label) {
        if (!enabled_) return;
        if (cudaEventRecord(stop_, nsos::gpu::current_stream()) != cudaSuccess ||
            cudaEventSynchronize(stop_) != cudaSuccess) {
            throw std::runtime_error(
                "Mamba stage timing fence failed");
        }
        float milliseconds = 0.0f;
        if (cudaEventElapsedTime(&milliseconds, start_, stop_) !=
            cudaSuccess) {
            throw std::runtime_error(
                "Mamba stage timing elapsed query failed");
        }
        milliseconds = std::max(milliseconds, 0.0f);
        std::fprintf(stderr, "[mtime] %.*s=%.3fms\n",
                     static_cast<int>(label.size()), label.data(),
                     static_cast<double>(milliseconds));
        if (cudaEventRecord(start_, nsos::gpu::current_stream()) != cudaSuccess) {
            throw std::runtime_error(
                "Mamba stage timing restart failed");
        }
    }

private:
    bool enabled_ = false;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};
#else
class MambaStageTimer {
public:
    explicit MambaStageTimer(Device) {}
    void mark(std::string_view) {}
};
#endif

// The chunked scan uses the same workspace sizes in every Mamba layer. Keeping
// one owner per host thread avoids sixteen deferred pool allocations before
// the default stream's release events become reusable. Thread-local ownership
// also prevents concurrent inference/training replicas from aliasing scratch.
struct FaithfulChunkWorkspace {
    Tensor carry;
    Tensor scale;
    Tensor lane_a;
    Tensor lane_d;

    void ensure(int carry_values, int scale_values,
                int lane_partial_values) {
        const auto ensure_tensor = [](Tensor& tensor, int values) {
            if (values <= 0) {
                throw std::invalid_argument(
                    "faithful chunk workspace requires positive size");
            }
            if (tensor.get_device() != Device::GPU ||
                tensor.size < values) {
                tensor = Tensor::uninitialized({values}, Device::GPU);
            }
        };
        ensure_tensor(carry, carry_values);
        ensure_tensor(scale, scale_values);
        ensure_tensor(lane_a, lane_partial_values);
        ensure_tensor(lane_d, lane_partial_values);
    }
};

FaithfulChunkWorkspace& faithful_chunk_workspace() {
    thread_local FaithfulChunkWorkspace workspace;
    return workspace;
}

// Scratch for the time-parallel forward scan: per-chunk end states, the
// per-chunk decay products and the carry entering each chunk.
struct FaithfulForwardChunkWorkspace {
    Tensor end_local;
    Tensor total_decay;
    Tensor carry;

    void ensure(int state_values, int decay_values) {
        const auto ensure_tensor = [](Tensor& tensor, int values) {
            if (values <= 0) {
                throw std::invalid_argument(
                    "faithful forward chunk workspace requires positive "
                    "size");
            }
            if (tensor.get_device() != Device::GPU ||
                tensor.size < values) {
                tensor = Tensor::uninitialized({values}, Device::GPU);
            }
        };
        ensure_tensor(end_local, state_values);
        ensure_tensor(total_decay, decay_values);
        ensure_tensor(carry, state_values);
    }
};

FaithfulForwardChunkWorkspace& faithful_forward_chunk_workspace() {
    thread_local FaithfulForwardChunkWorkspace workspace;
    return workspace;
}

// State-parallel backward needs one gh value for every row/channel/state.
// A transient Tensor per layer cannot be recycled until the asynchronous
// default-stream consumers finish, so a 16-layer backward can retain sixteen
// copies in the allocator.  One thread-local owner is safe because every
// producer/finalizer and the next layer are enqueued on the same ordered
// default stream; separate host threads never alias this storage.
struct FaithfulStateParallelWorkspace {
    Tensor gh_history;

    void ensure(int values) {
        if (values <= 0) {
            throw std::invalid_argument(
                "faithful state-parallel workspace requires positive size");
        }
        if (gh_history.get_device() != Device::GPU ||
            gh_history.size < values) {
            gh_history = Tensor::uninitialized({values}, Device::GPU);
        }
    }
};

FaithfulStateParallelWorkspace& faithful_state_parallel_workspace() {
    thread_local FaithfulStateParallelWorkspace workspace;
    return workspace;
}

#ifdef USE_CUDA
int ensure_faithful_runtime_warp_size(int& cached_warp_size) {
    if (cached_warp_size > 0) return cached_warp_size;
    int device_id = 0;
    cudaDeviceProp properties{};
    const cudaError_t device_status = cudaGetDevice(&device_id);
    const cudaError_t properties_status =
        device_status == cudaSuccess
            ? cudaGetDeviceProperties(&properties, device_id)
            : device_status;
    if (properties_status != cudaSuccess || properties.warpSize <= 0) {
        throw std::runtime_error(
            std::string("Mamba2 could not audit the active GPU warp size: ") +
            cudaGetErrorString(properties_status));
    }
    cached_warp_size = properties.warpSize;
    return cached_warp_size;
}
#endif

// GPU-first DEFAULT ON (NSOS_MAMBA_GPU_STEP=0 opts out): route the proper
// single-token decode (diagonal AND N-state) through the fused on-device step
// kernels (device-resident SSD state + conv ring, no per-token host
// round-trip).  Parity gates: test_gpu_parity_mamba_proper_stream (diagonal,
// T4-validated) and test_gpu_parity_mamba_nstate_stream.  The host step
// remains the =0 escape hatch and the CPU-device path.
bool mamba_gpu_step_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NSOS_MAMBA_GPU_STEP");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
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

int checked_int_product(int lhs, int rhs, const char* label) {
    if (lhs < 0 || rhs < 0) {
        throw std::invalid_argument(std::string(label) +
                                    " received a negative dimension");
    }
    const int64_t value =
        static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
    if (value > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string(label) +
                                  " exceeds the supported int index range");
    }
    return static_cast<int>(value);
}

int require_positive_mamba_dimension(int value, const char* label) {
    if (value <= 0) {
        throw std::invalid_argument(
            std::string("Mamba2SSD ") + label + " must be positive");
    }
    return value;
}

int checked_mamba_inner_width(
    int model_width, const MambaConfig& config) {
    if (!config.faithful_mamba2) {
        return model_width;
    }
    if (config.expand <= 0 || config.head_dim <= 0 ||
        config.n_groups <= 0 || config.conv_kernel <= 0) {
        throw std::invalid_argument(
            "Mamba2SSD faithful dimensions must be positive");
    }
    const int inner = checked_int_product(
        model_width, config.expand, "Mamba2 faithful inner width");
    if (inner % config.head_dim != 0) {
        throw std::invalid_argument(
            "Mamba2 faithful inner width must be divisible by head_dim");
    }
    return inner;
}

int checked_mamba_head_count(
    int model_width, int inner_width, int requested_heads,
    const MambaConfig& config) {
    if (!config.faithful_mamba2) {
        return require_positive_mamba_dimension(
            requested_heads, "head count");
    }
    const int heads = inner_width / config.head_dim;
    if (heads <= 0 || heads % config.n_groups != 0) {
        throw std::invalid_argument(
            "Mamba2 faithful head count must be positive and divisible by "
            "n_groups");
    }
    (void)model_width;
    return heads;
}

int checked_mamba_head_width(
    int model_width, int head_count, const MambaConfig& config) {
    if (config.faithful_mamba2) {
        return config.head_dim;
    }
    return std::max(model_width / head_count, 1);
}

int checked_mamba_group_count(const MambaConfig& config) {
    return config.faithful_mamba2
               ? require_positive_mamba_dimension(
                     config.n_groups, "group count")
               : 1;
}

int checked_mamba_conv_width(int inner_width, int group_count,
                             int state_width,
                             const MambaConfig& config) {
    if (!config.faithful_mamba2) {
        return inner_width;
    }
    const int group_state = checked_int_product(
        group_count, state_width, "Mamba2 faithful group-state width");
    const int64_t value =
        static_cast<int64_t>(inner_width) + 2LL * group_state;
    if (value > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "Mamba2 faithful convolution width exceeds int range");
    }
    return static_cast<int>(value);
}

void validate_faithful_conv_launch(int rows, int inner, int group_state,
                                   int kernel_width) {
    if (rows <= 0 || inner <= 0 || group_state <= 0 ||
        kernel_width <= 0) {
        throw std::invalid_argument(
            "Mamba2 faithful convolution dimensions must be positive");
    }
    const int64_t convdim =
        static_cast<int64_t>(inner) + 2LL * group_state;
    const int64_t elements = static_cast<int64_t>(rows) * convdim;
    if (convdim > std::numeric_limits<int>::max() ||
        elements > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "Mamba2 faithful convolution launch exceeds the supported "
            "32-bit kernel index range");
    }
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
        record_gpu_transfer(dst_device, src_device, bytes);
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
    record_gpu_transfer(dst_device, src_device, bytes);
}

// N1: A is stored in the LOG domain.  The effective decay rate is
// A_eff = exp(A_log) (always > 0, so decay = exp(-softplus(dt)·A_eff) ∈ (0,1)
// is unconditionally stable WITHOUT a hard clamp), and the gradient flows for
// every channel because ∂A_eff/∂A_log = A_eff ≠ 0 — no channel can get stuck at
// the old 1e-3 clamp with a masked (zero) gradient.  A_log = 0 ⇒ A_eff = 1,
// numerically identical to the historical A = ones init (the legacy code used
// max(A,1e-3) with A init to ones, i.e. A_eff = 1).  This is the standard
// Mamba/S4D parameterization (A = -exp(A_log)); we keep |A| = exp(A_log).
struct MambaDecayTerms {
    float delta;
    float delta_grad;
    float decay;
    float decay_dt_factor;
    float decay_alog_factor;
};

static inline double log_sigmoid_for_decay(double x) {
    return x >= 0.0 ? -std::log1p(std::exp(-x))
                    : x - std::log1p(std::exp(x));
}

static inline MambaDecayTerms mamba_decay_terms(float dt_raw, float a_log) {
    const float delta = dt_raw > 20.0f
                            ? dt_raw
                            : (dt_raw < -20.0f
                                   ? std::exp(dt_raw)
                                   : std::log1p(std::exp(dt_raw)));
    const float delta_grad = dt_raw >= 0.0f
                                 ? 1.0f / (1.0f + std::exp(-dt_raw))
                                 : std::exp(dt_raw) /
                                       (1.0f + std::exp(dt_raw));
    if (!std::isfinite(dt_raw) || !std::isfinite(a_log)) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return {delta, delta_grad, nan, nan, nan};
    }
    const double log_delta =
        dt_raw < -20.0f ? static_cast<double>(dt_raw)
                        : std::log(static_cast<double>(delta));
    const double log_q = log_delta + static_cast<double>(a_log);
    if (log_q > 50.0) {
        return {delta, delta_grad, 0.0f, 0.0f, 0.0f};
    }
    const double q = std::exp(log_q);
    const double decay = std::exp(-q);
    const double alog_factor = q * decay;
    const double dt_log = log_sigmoid_for_decay(dt_raw) +
                          static_cast<double>(a_log) - q;
    const double dt_factor = dt_log < -110.0 ? 0.0 : std::exp(dt_log);
    return {delta, delta_grad, static_cast<float>(decay),
            static_cast<float>(dt_factor),
            static_cast<float>(alog_factor)};
}

} // namespace

thread_local Mamba2SSD::ThreadBuffers Mamba2SSD::buffers_{};

Mamba2SSD::Mamba2SSD(int d_model_value, int d_state_value, int n_heads_value,
                     const MambaConfig& config)
    : d_model(require_positive_mamba_dimension(
          d_model_value, "model width")),
      d_state(require_positive_mamba_dimension(
          d_state_value, "state width")),
      d_inner(checked_mamba_inner_width(d_model, config)),
      n_heads(checked_mamba_head_count(
          d_model, d_inner, n_heads_value, config)),
      d_head(checked_mamba_head_width(
          d_model, n_heads, config)),
      n_groups(checked_mamba_group_count(config)),
      conv_dim(checked_mamba_conv_width(
          d_inner, n_groups, d_state, config)),
      config_(config),
      out_proj(d_inner, d_model, !config.faithful_mamba2),
      // N1: A holds A_log; A_log = 0 ⇒ A_eff = exp(0) = 1, matching the old
      // A = ones / max(A,1e-3) default exactly while removing the dead-channel
      // gradient mask.  D (the skip scale) stays ones.
      A(Tensor::zeros({config.faithful_mamba2 ? n_heads : d_model},
                      Device::CPU),
        "mamba.A"),
      D(Tensor::ones({config.faithful_mamba2 ? n_heads : d_model},
                     Device::CPU),
        "mamba.D") {
  if (config_.faithful_mamba2) {
    if (config_.expand <= 0 || config_.head_dim <= 0 || config_.n_groups <= 0 ||
        d_inner % config_.head_dim != 0 || n_heads <= 0 ||
        n_heads % n_groups != 0 ||
        !std::isfinite(config_.rms_norm_eps) ||
        config_.rms_norm_eps <= 0.0f) {
      throw std::invalid_argument("Mamba2SSD invalid faithful Mamba-2 configuration");
    }
    // In float mode these BitLinear instances must behave exactly as
    // nn.Linear.  QAT can still switch their weights to ternary later.
    out_proj.set_exact_linear_mode(true);
  }
  if (!config_.proper_selective_ssm && !config_.faithful_mamba2) {
    in_proj_robust =
        std::make_unique<BitLinear>(d_model, d_model, true);
    in_proj_sensitive =
        std::make_unique<BitLinear>(d_model, d_model, true);
    in_proj_sensitive->set_quantization_sensitive(true);
  }

  // N-state (full Mamba-2 SSD) uses a PER-HEAD decay A ∈ R^H instead of the
  // per-channel A ∈ R^{d_model}.  Resize BEFORE the optional log-spaced init so
  // that init fills the correct length.
  const bool proper_nstate_ctor =
      config_.faithful_mamba2 ||
      (config_.proper_selective_ssm && config_.proper_state_expansion);
  if (proper_nstate_ctor) {
    // Per-head A_log (N1): zeros ⇒ A_eff = 1 per head.
    A.data = Tensor::zeros({std::max(n_heads, 1)}, Device::CPU);
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
  // N1 + timescale hierarchy: initialise A_log on a log-spaced grid so channels
  // span tau ~ 1 .. 1/a_min tokens from step 0 (S4D-style) instead of a
  // degenerate single-rate spectrum.  Now DEFAULT ON (NSOS_MAMBA_A_LOGSPACED=0
  // disables).  Values are in the LOG domain: A_eff = exp(A_log) ranges
  // exp(0)=1 .. exp(log a_min)=a_min.  Deterministic (no RNG).
  if (!config_.faithful_mamba2) {
    const char* env = std::getenv("NSOS_MAMBA_A_LOGSPACED");
    if (env == nullptr || env[0] != '0') {
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
      a_ptr[d] = log_min * frac;  // A_log: 0 (A_eff=1) ... log(a_min) (A_eff=a_min)
    }
    }
  } else {
    // Official Mamba-2 default: A ~ Uniform(1,16), stored as A_log.
    A.data = Tensor::uniform({n_heads}, 1.0f, 16.0f, Device::CPU);
    float* a_ptr = A.data.data();
    for (int h = 0; h < n_heads; ++h) {
      a_ptr[h] = std::log(a_ptr[h]);
    }
  }

  // ── Proper selective-SSM components (opt-in) ───────────────────────────────
  // Allocated only when requested so the default path's parameter set / pack
  // layout is unchanged.  Independent projections give x, z(gate), B, C, dt
  // their own capacity (kills the delta==C degeneracy); the depthwise causal
  // conv is initialized to an identity pass-through (last tap = 1) so a freshly
  // enabled proper path starts numerically close to "no conv" and learns the
  // local mixing from there.
  if (config_.proper_selective_ssm || config_.faithful_mamba2) {
    conv_kernel_ = std::max(config_.conv_kernel, 1);
    if (config_.faithful_mamba2) {
      const int group_state = n_groups * std::max(d_state, 1);
      x_proj_ = std::make_unique<BitLinear>(d_model, d_inner, false);
      z_proj_ = std::make_unique<BitLinear>(d_model, d_inner, false);
      B_proj_ = std::make_unique<BitLinear>(d_model, group_state, false);
      C_proj_ = std::make_unique<BitLinear>(d_model, group_state, false);
      // A separate bias-bearing dt projection is algebraically identical to
      // the official bias-free combined in_proj plus standalone dt_bias.
      dt_proj_ = std::make_unique<BitLinear>(d_model, n_heads, true);
      for (BitLinear* projection :
           {x_proj_.get(), z_proj_.get(), B_proj_.get(), C_proj_.get(),
            dt_proj_.get()}) {
        projection->set_exact_linear_mode(true);
        const float bound =
            1.0f / std::sqrt(static_cast<float>(projection->input_features()));
        projection->weight.data =
            Tensor::uniform({projection->output_features(),
                             projection->input_features()},
                            -bound, bound, Device::CPU);
        projection->weight.mark_updated();
        if (projection->uses_bias()) {
          projection->bias.data =
              Tensor::zeros({projection->output_features()}, Device::CPU);
        }
      }
      {
        const float bound = 1.0f / std::sqrt(static_cast<float>(d_inner));
        out_proj.weight.data =
            Tensor::uniform({d_model, d_inner}, -bound, bound, Device::CPU);
        if (!std::isfinite(config_.out_proj_init_scale) ||
            config_.out_proj_init_scale <= 0.0f) {
          throw std::invalid_argument(
              "Mamba2SSD out_proj_init_scale must be finite and positive");
        }
        float* weight = out_proj.weight.data.data();
        for (int64_t i = 0; i < out_proj.weight.data.size; ++i) {
          weight[i] *= config_.out_proj_init_scale;
        }
        out_proj.weight.mark_updated();
      }
      const float conv_bound =
          1.0f / std::sqrt(static_cast<float>(conv_kernel_));
      conv_weight_ =
          Parameter(Tensor::uniform({conv_dim, conv_kernel_}, -conv_bound,
                                    conv_bound, Device::CPU),
                    "conv1d_weight");
      conv_bias_ = Parameter(
          Tensor::uniform({conv_dim}, -conv_bound, conv_bound, Device::CPU),
          "conv1d_bias");
      norm_weight_ =
          Parameter(Tensor::ones({d_inner}, Device::CPU), "norm_weight");
    } else {
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
      conv_weight_ = Parameter(Tensor::zeros({d_model, conv_kernel_}, Device::CPU),
                               "conv1d_weight");
      float* cw = conv_weight_.data.data();
      for (int c = 0; c < d_model; ++c) {
        cw[c * conv_kernel_ + (conv_kernel_ - 1)] = 1.0f;
      }
    }
    // dt/B/C carry the SSM's selectivity — keep them off the ternary path during
    // QAT (mixed precision), exactly as the legacy in_proj_sensitive does.
    dt_proj_->set_quantization_sensitive(true);
    B_proj_->set_quantization_sensitive(true);
    C_proj_->set_quantization_sensitive(true);
    // dt bias init — THE critical Mamba init (Gu & Dao, mamba-ssm reference).
    // Without it, dt_proj's bias starts ~0, so softplus(dt)≈softplus(0)=0.693
    // and the discrete decay exp(-dt·A) forgets ~50% of the SSM state PER
    // TOKEN at step 0 — the model literally cannot hold a key long enough to
    // learn associative recall (measured: our recall plateaus ~0.45 while the
    // reference Mamba-2, which does this init, reaches 1.0 / loss→0).  We set
    // dt_bias = softplus^{-1}(dt_init) so softplus(dt)≈dt_init at step 0 and
    // the discrete decay exp(-dt·A_eff) holds state over many tokens; the model
    // refines selectivity from there.
    //
    // dt is CONSTANT across the dt outputs (not per-head-varied): our A_log is
    // already initialised on a log-spaced grid (A_eff ~ 1 .. a_min), so the
    // timescale HIERARCHY comes from A.  Log-spacing dt over the same index
    // range as A — in the opposite sense — would make the product dt·A_eff
    // (which sets the decay) constant across heads and CANCEL A's hierarchy.
    //
    // The input discretisation is coupled to the same dt
    // (dB = softplus(dt)·B), so a small initial dt slows both forgetting and
    // writing instead of saturating the state.  Default ON; set
    // NSOS_MAMBA_DT_INIT=0 only to load/compare a historical checkpoint.
    // A numeric value in (1e-6, 1) overrides the default 0.01.
    if (config_.faithful_mamba2 && dt_proj_ &&
        dt_proj_->bias.data.size == n_heads) {
      // Official dt init: log-uniform in [1e-3, 1e-1], then inverse softplus.
      Tensor log_dt = Tensor::uniform({n_heads}, std::log(1e-3f),
                                      std::log(1e-1f), Device::CPU);
      float* db = dt_proj_->bias.data.data();
      const float* lp = log_dt.data();
      for (int h = 0; h < n_heads; ++h) {
        const float dt_value = std::max(std::exp(lp[h]), 1e-4f);
        db[h] = dt_value + std::log(-std::expm1(-dt_value));
      }
    } else if (dt_proj_ && dt_proj_->bias.data.size > 0) {
      const char* dtenv = std::getenv("NSOS_MAMBA_DT_INIT");
      const bool enabled =
          dtenv == nullptr || dtenv[0] == '\0' ||
          !(dtenv[0] == '0' && dtenv[1] == '\0');
      if (enabled) {
        float dt_init = 0.01f;  // NSOS_MAMBA_DT_INIT=1 -> default 0.01
        // A numeric value (anything other than "1") overrides dt_init.
        if (dtenv != nullptr && dtenv[0] != '\0' &&
            !(dtenv[0] == '1' && dtenv[1] == '\0')) {
          const float v = std::strtof(dtenv, nullptr);
          if (v > 1e-6f && v < 1.0f) dt_init = v;
        }
        // softplus^{-1}(y) = log(exp(y) - 1), numerically safe for small y.
        const float dt_bias = std::log(std::expm1(dt_init) + 1e-12f);
        float* db = dt_proj_->bias.data.data();
        const int nb = dt_proj_->bias.data.size;
        for (int i = 0; i < nb; ++i) db[i] = dt_bias;
      }
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
        if (strict_gpu_execution()) {
            throw std::runtime_error(
                "Strict GPU selective scan forward has no eligible device path");
        }
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
                const MambaDecayTerms decay_terms =
                    mamba_decay_terms(dt_t[d], a_ptr[d]);
                const float decay = decay_terms.decay;
                const float dt_scale = decay_terms.delta;
                state[d] =
                    state[d] * decay + dt_scale * b_t[d] * x_t[d];
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
                const float b_value = b_ptr[token_index * dim + d];
                const float c_value = c_ptr[token_index * dim + d];
                const float candidate = std::tanh(state_t);
                const MambaDecayTerms decay_terms = mamba_decay_terms(
                    dt_ptr[token_index * dim + d], a_ptr[d]);
                const float decay = decay_terms.decay;

                grad_c_ptr[token_index * dim + d] +=
                    grad_y_ptr[token_index * dim + d] * candidate;
                const float grad_candidate =
                    grad_y_ptr[token_index * dim + d] * c_value;
                const float grad_state =
                    grad_candidate * (1.0f - candidate * candidate) + grad_state_next[d];

                const float dt_scale = decay_terms.delta;
                grad_x_ptr[token_index * dim + d] +=
                    grad_state * dt_scale * b_value;
                grad_b_ptr[token_index * dim + d] +=
                    grad_state * dt_scale * x_ptr[token_index * dim + d];

                const float grad_decay = grad_state * prev_state;
                grad_state_next[d] = grad_state * decay;

                grad_delta_ptr[token_index * dim + d] +=
                    grad_state * b_value * x_ptr[token_index * dim + d] *
                        decay_terms.delta_grad -
                    grad_decay * decay_terms.decay_dt_factor;

                // N1: dL/dA_log = dL/dA_eff · A_eff; gradient flows for every
                // channel (no 1e-3 gate). a_value = A_eff = exp(A_log).
                grad_a_ptr[d] -=
                    grad_decay * decay_terms.decay_alog_factor;
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
    const int rows =
        checked_int_product(batch, seq, "Mamba2 proper forward rows");
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

    // xc = SiLU(conv_pre), fused into one elementwise device pass.
    Tensor xc = conv_pre.silu();

    // Diagonal selective recurrence:
    // h_t = decay_t*h_{t-1} + softplus(dt_t)*B_t*xc_t;
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
                    const MambaDecayTerms decay_terms =
                        mamba_decay_terms(dt_ptr[idx], a_ptr[c]);
                    const float dt_scale = decay_terms.delta;
                    const float decay = decay_terms.decay;
                    state[static_cast<size_t>(c)] =
                        decay * state[static_cast<size_t>(c)] +
                        dt_scale * b_ptr[idx] * xc_ptr[idx];
                    h_ptr[idx] = state[static_cast<size_t>(c)];
                    y_ptr[idx] = state[static_cast<size_t>(c)] * c_ptr[idx];
                }
            }
        }
    }

    // Separate SiLU gate, out projection and D skip — all device-agnostic.
    Tensor gated = y_ssd.mul(z.silu());
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
        const int taps = std::max(K - 1, 0);
        const int ring_elements = checked_int_product(
            taps, dim, "Mamba proper streaming ring");
#ifdef USE_CUDA
        if (dev == Device::GPU && batch == 1 &&
            mamba_gpu_step_enabled()) {
            pp_stream_h_dev_ =
                Tensor::uninitialized({dim}, Device::GPU);
            pp_stream_ring_dev_ = Tensor::uninitialized(
                {std::max(ring_elements, 1)}, Device::GPU);
            const bool prime_enqueued =
                cuda::launch_mamba_prime_stream_carry(
                h_hist.raw_data(), dim,
                xv.raw_data(), dim,
                nullptr, 0, nullptr, 0,
                pp_stream_h_dev_.raw_data(),
                pp_stream_ring_dev_.raw_data(),
                batch, seq, taps);
            if (!prime_enqueued) {
                throw std::runtime_error(
                    "Mamba proper GPU stream priming rejected invalid "
                    "arguments");
            }
            const cudaError_t prime_status = cudaGetLastError();
            if (prime_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Mamba proper GPU stream priming failed: ") +
                    cudaGetErrorString(prime_status));
            }
            pp_stream_state_.clear();
            pp_stream_ring_.clear();
            pp_stream_dev_live_ = true;
            ++stream_priming_gpu_calls_;
        } else
#endif
        {
#ifdef USE_CUDA
            if (dev == Device::GPU) {
                ++stream_priming_host_fallbacks_;
                ++gpu_fast_path_fallbacks_;
                last_fallback_reason_ =
                    "proper_stream_priming_host_fallback";
                if (strict_gpu_execution()) {
                    throw std::runtime_error(
                        "Strict GPU Mamba proper prefill cannot prime "
                        "incremental carry through host memory");
                }
            }
#endif
            Tensor h_h =
                dev == Device::GPU ? h_hist.cpu() : h_hist;
            Tensor xv_h =
                dev == Device::GPU ? xv.cpu() : xv;
            pp_stream_state_.assign(
                static_cast<size_t>(dim), 0.0f);
            const float* hp = h_h.data();
            for (int c = 0; c < dim; ++c) {
                pp_stream_state_[static_cast<size_t>(c)] =
                    hp[static_cast<size_t>(rows - 1) * dim + c];
            }
            pp_stream_ring_.assign(
                static_cast<size_t>(ring_elements), 0.0f);
            const float* xvp = xv_h.data();
            const int batch_base = (batch - 1) * seq;
            for (int s = 0; s < taps; ++s) {
                const int source_time = seq - taps + s;
                if (source_time < 0) continue;
                const int source_row = batch_base + source_time;
                for (int c = 0; c < dim; ++c) {
                    pp_stream_ring_[
                        static_cast<size_t>(s) * dim + c] =
                        xvp[
                            static_cast<size_t>(source_row) * dim + c];
                }
            }
            pp_stream_dev_live_ = false;
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
    const int rows =
        checked_int_product(batch, seq, "Mamba2 proper backward rows");
    const Device dev = grad_output.get_device();  // GPU-first: stay on device

    Tensor g = grad_output.shape.size() == 2 ? grad_output
                                             : grad_output.reshape({rows, dim});

    // Skip path: result = out_proj(...) + u*D.  grad_D = Σ_rows(g·u); the input
    // gradient via the skip is g·D (broadcast).  Device-agnostic Tensor ops.
    Tensor grad_D = g.mul(pp_u_).sum(0, false);
    Tensor g_u = g.mul(D.data);

    // out_proj backward -> grad wrt gated.
    Tensor g_gated = out_proj.backward(g).reshape({rows, dim});

    // Gate y = y_ssd · silu(z).
    Tensor silu_z = pp_z_.silu();
    Tensor g_yssd = g_gated.mul(silu_z);
    Tensor g_z = Tensor::silu_backward(
        g_gated.mul(pp_y_ssd_), pp_z_);

    // Scan backward (linear readout): GPU kernel on device, ordered host loop on
    // CPU.  Produces grads wrt the scan input xc, dt, A, B and C.
    Tensor g_xc = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor g_dt = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor g_B = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor g_C = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor grad_A = Tensor::zeros({dim}, dev);
    bool scan_backward_done = false;
#ifdef USE_CUDA
    // Deterministic mode: the GPU scan backward accumulates grad_A via atomicAdd
    // (order-nondeterministic); fall to the ordered host loop below.
    if (dev == Device::GPU && !determinism::deterministic_reductions_enabled()) {
        cuda::launch_mamba_proper_scan_backward(
            g_yssd.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
            A.data.raw_data(), pp_B_.raw_data(), pp_C_.raw_data(),
            pp_h_hist_.raw_data(), g_xc.raw_data(), g_dt.raw_data(),
            grad_A.raw_data(), g_B.raw_data(), g_C.raw_data(), batch, seq, dim);
        scan_backward_done = true;
    }
#endif
    if (!scan_backward_done) {
        auto host = [](const Tensor& value) {
            return value.get_device() == Device::GPU ? value.cpu() : value;
        };
        Tensor gy_h = host(g_yssd);
        Tensor hist_h = host(pp_h_hist_);
        Tensor C_h = host(pp_C_);
        Tensor B_h = host(pp_B_);
        Tensor dt_h = host(pp_dt_);
        Tensor xc_h = host(pp_xc_);
        Tensor A_h = host(A.data);
        Tensor gB_h = Tensor::zeros({rows, dim}, Device::CPU);
        Tensor gC_h = Tensor::zeros({rows, dim}, Device::CPU);
        Tensor gdt_h = Tensor::zeros({rows, dim}, Device::CPU);
        Tensor gxc_h = Tensor::zeros({rows, dim}, Device::CPU);
        Tensor gA_h = Tensor::zeros({dim}, Device::CPU);
        const float* gy = gy_h.data();
        const float* h_ptr = hist_h.data();
        const float* c_ptr = C_h.data();
        const float* b_ptr = B_h.data();
        const float* dt_ptr = dt_h.data();
        const float* xc_ptr = xc_h.data();
        const float* a_ptr = A_h.data();
        float* gBp = gB_h.data();
        float* gCp = gC_h.data();
        float* gDtp = gdt_h.data();
        float* gXcp = gxc_h.data();
        float* gA = gA_h.data();
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
                    const MambaDecayTerms decay_terms =
                        mamba_decay_terms(dt_ptr[idx], a_ptr[c]);
                    const float sp = decay_terms.delta;
                    const float decay = decay_terms.decay;
                    gCp[idx] = gy[idx] * h_t;
                    const float grad_h =
                        gy[idx] * c_ptr[idx] + carry[static_cast<size_t>(c)];
                    gBp[idx] = grad_h * sp * xc_ptr[idx];
                    gXcp[idx] = grad_h * sp * b_ptr[idx];
                    const float grad_decay = grad_h * h_prev;
                    gDtp[idx] =
                        grad_h * b_ptr[idx] * xc_ptr[idx] *
                            decay_terms.delta_grad -
                        grad_decay * decay_terms.decay_dt_factor;
                    // N1: gradient to A_log (= dL/dA_eff · A_eff), unconditional.
                    gA[c] -= grad_decay * decay_terms.decay_alog_factor;
                    carry[static_cast<size_t>(c)] = grad_h * decay;
                }
            }
        }
        g_B = dev == Device::GPU ? gB_h.to(Device::GPU) : gB_h;
        g_C = dev == Device::GPU ? gC_h.to(Device::GPU) : gC_h;
        g_dt = dev == Device::GPU ? gdt_h.to(Device::GPU) : gdt_h;
        g_xc = dev == Device::GPU ? gxc_h.to(Device::GPU) : gxc_h;
        grad_A = dev == Device::GPU ? gA_h.to(Device::GPU) : gA_h;
    }

    // xc = silu(conv_pre) -> grad_conv_pre (device-agnostic).
    Tensor grad_conv_pre =
        Tensor::silu_backward(g_xc, pp_conv_pre_);

    // conv1d backward: GPU kernel on device, ordered host loop on CPU.
    Tensor grad_xv = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor grad_conv_w = Tensor::zeros({dim, K}, dev);
    bool conv_backward_done = false;
#ifdef USE_CUDA
    if (dev == Device::GPU && !determinism::deterministic_reductions_enabled()) {
        cuda::launch_conv1d_causal_backward(
            grad_conv_pre.raw_data(), pp_xv_.raw_data(),
            conv_weight_.data.raw_data(), grad_xv.raw_data(),
            grad_conv_w.raw_data(), batch, seq, dim, K);
        conv_backward_done = true;
    }
#endif
    if (!conv_backward_done) {
        auto host = [](const Tensor& value) {
            return value.get_device() == Device::GPU ? value.cpu() : value;
        };
        Tensor gp_h = host(grad_conv_pre);
        Tensor xv_h = host(pp_xv_);
        Tensor cw_h = host(conv_weight_.data);
        Tensor gxv_h = Tensor::zeros({rows, dim}, Device::CPU);
        Tensor gcw_h = Tensor::zeros({dim, K}, Device::CPU);
        conv1d_causal_backward(gp_h.data(), xv_h.data(), cw_h.data(),
                               gxv_h.data(), gcw_h.data(), batch, seq, dim, K);
        grad_xv = dev == Device::GPU ? gxv_h.to(Device::GPU) : gxv_h;
        grad_conv_w = dev == Device::GPU ? gcw_h.to(Device::GPU) : gcw_h;
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
    const int rows =
        checked_int_product(batch, seq, "Mamba2 nstate forward rows");
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
    Tensor xc = conv_pre.silu();

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
                const MambaDecayTerms decay_terms =
                    mamba_decay_terms(dtp[row * H + h], ap[h]);
                const float dt_scale = decay_terms.delta;
                const float decay = decay_terms.decay;
                for (int p = 0; p < P; ++p) {
                    const int chan = h * P + p;
                    const float xcv = xcp[static_cast<size_t>(row) * dim + chan];
                    float y_acc = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        const size_t sidx = (static_cast<size_t>(h) * P + p) * N + n;
                        const float bval = bp[static_cast<size_t>(row) * (H * N) + h * N + n];
                        const float cval = cp[static_cast<size_t>(row) * (H * N) + h * N + n];
                        const float hv =
                            decay * state[sidx] + dt_scale * bval * xcv;
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
    Tensor gated = y_ssd.mul(z.silu());
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
        const size_t HPN = static_cast<size_t>(H) * P * N;
        if (HPN >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Mamba nstate streaming state exceeds int range");
        }
        const int taps = std::max(K - 1, 0);
        const int ring_elements = checked_int_product(
            taps, dim, "Mamba nstate streaming ring");
#ifdef USE_CUDA
        if (dev == Device::GPU && batch == 1 &&
            nstate_scan_done && mamba_gpu_step_enabled()) {
            pp_stream_h_dev_ = Tensor::uninitialized(
                {static_cast<int>(HPN)}, Device::GPU);
            pp_stream_ring_dev_ = Tensor::uninitialized(
                {std::max(ring_elements, 1)}, Device::GPU);
            const bool prime_enqueued =
                cuda::launch_mamba_prime_stream_carry(
                hist.raw_data(), static_cast<int>(HPN),
                xv.raw_data(), dim,
                nullptr, 0, nullptr, 0,
                pp_stream_h_dev_.raw_data(),
                pp_stream_ring_dev_.raw_data(),
                batch, seq, taps);
            if (!prime_enqueued) {
                throw std::runtime_error(
                    "Mamba nstate GPU stream priming rejected invalid "
                    "arguments");
            }
            const cudaError_t prime_status = cudaGetLastError();
            if (prime_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Mamba nstate GPU stream priming failed: ") +
                    cudaGetErrorString(prime_status));
            }
            pp_stream_state_.clear();
            pp_stream_ring_.clear();
            pp_stream_dev_live_ = true;
            ++stream_priming_gpu_calls_;
        } else
#endif
        {
#ifdef USE_CUDA
            if (dev == Device::GPU) {
                ++stream_priming_host_fallbacks_;
                ++gpu_fast_path_fallbacks_;
                last_fallback_reason_ =
                    "nstate_stream_priming_host_fallback";
                if (strict_gpu_execution()) {
                    throw std::runtime_error(
                        "Strict GPU Mamba nstate prefill cannot prime "
                        "incremental carry through host memory");
                }
            }
#endif
            Tensor st_h =
                dev == Device::GPU ? hist.cpu() : hist;
            Tensor xv_h =
                dev == Device::GPU ? xv.cpu() : xv;
            pp_stream_state_.assign(HPN, 0.0f);
            const float* hp = st_h.data();
            for (size_t k = 0; k < HPN; ++k) {
                pp_stream_state_[k] =
                    hp[static_cast<size_t>(rows - 1) * HPN + k];
            }
            pp_stream_ring_.assign(
                static_cast<size_t>(ring_elements), 0.0f);
            const float* xvp = xv_h.data();
            const int batch_base = (batch - 1) * seq;
            for (int s = 0; s < taps; ++s) {
                const int source_time = seq - taps + s;
                if (source_time < 0) continue;
                const int source_row = batch_base + source_time;
                for (int c = 0; c < dim; ++c) {
                    pp_stream_ring_[
                        static_cast<size_t>(s) * dim + c] =
                        xvp[
                            static_cast<size_t>(source_row) * dim + c];
                }
            }
            pp_stream_dev_live_ = false;
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
    const int rows =
        checked_int_product(batch, seq, "Mamba2 nstate backward rows");
    const int H = std::max(n_heads, 1);
    const int P = d_head;
    const int N = std::max(d_state, 1);
    const Device dev = grad_output.get_device();
    Tensor g = grad_output.shape.size() == 2 ? grad_output
                                             : grad_output.reshape({rows, dim});

    // Skip + out_proj + gate (device-agnostic).
    Tensor grad_D = g.mul(pp_u_).sum(0, false);
    Tensor g_u = g.mul(D.data);
    Tensor g_gated = out_proj.backward(g).reshape({rows, dim});
    Tensor silu_z = pp_z_.silu();
    Tensor g_yssd = g_gated.mul(silu_z);
    Tensor g_z = Tensor::silu_backward(
        g_gated.mul(pp_y_ssd_), pp_z_);

    // ── N-state SSD scan backward ── GPU-resident kernel on device; ordered
    // host loop on CPU or when N exceeds the kernel's MAX_N.
    Tensor gXc(std::vector<int>{rows, dim}, dev);
    Tensor gB(std::vector<int>{rows, H * N}, dev);
    Tensor gC(std::vector<int>{rows, H * N}, dev);
    Tensor gDt(std::vector<int>{rows, H}, dev);
    Tensor gA(std::vector<int>{H}, dev);
    bool nstate_bwd_done = false;
#ifdef USE_CUDA
    if (dev == Device::GPU && N <= cuda::mamba_nstate_max_n()) {
        if (determinism::deterministic_reductions_enabled() &&
            cuda::mamba_nstate_deterministic_backward_supported(P, N)) {
            const bool launch_enqueued =
                cuda::launch_mamba_nstate_backward_deterministic(
                g_yssd.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
                A.data.raw_data(), pp_B_.raw_data(), pp_C_.raw_data(),
                pp_state_hist_.raw_data(), gXc.raw_data(), gDt.raw_data(),
                gA.raw_data(), gB.raw_data(), gC.raw_data(), batch, seq, H, P,
                N);
            if (!launch_enqueued) {
                throw std::runtime_error(
                    "Mamba nstate deterministic backward rejected invalid "
                    "arguments");
            }
            nstate_bwd_done = true;
        } else if (!determinism::deterministic_reductions_enabled()) {
            cuda::launch_mamba_nstate_backward(
                g_yssd.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
                A.data.raw_data(), pp_B_.raw_data(), pp_C_.raw_data(),
                pp_state_hist_.raw_data(), gXc.raw_data(), gDt.raw_data(),
                gA.raw_data(), gB.raw_data(), gC.raw_data(), batch, seq, H, P,
                N);
            nstate_bwd_done = true;
        }
        if (nstate_bwd_done) {
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Mamba nstate backward kernel failed: ") +
                    cudaGetErrorString(launch_status));
            }
        }
    }
#endif
    if (!nstate_bwd_done) {
    Tensor gy_h = g_yssd.get_device() == Device::GPU ? g_yssd.cpu() : g_yssd;
    Tensor xc_h = pp_xc_.get_device() == Device::GPU ? pp_xc_.cpu() : pp_xc_;
    Tensor Bt_h = pp_B_.get_device() == Device::GPU ? pp_B_.cpu() : pp_B_;
    Tensor Ct_h = pp_C_.get_device() == Device::GPU ? pp_C_.cpu() : pp_C_;
    Tensor dt_h = pp_dt_.get_device() == Device::GPU ? pp_dt_.cpu() : pp_dt_;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    Tensor hist_h = pp_state_hist_.get_device() == Device::GPU
                        ? pp_state_hist_.cpu()
                        : pp_state_hist_;
    const float* gyp = gy_h.data();
    const float* xcp = xc_h.data();
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();
    const float* histp = hist_h.data();
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
                const MambaDecayTerms decay_terms =
                    mamba_decay_terms(dtp[row * H + h], ap[h]);
                const float sp = decay_terms.delta;
                const float decay = decay_terms.decay;
                float ddecay = 0.0f;
                float dinput_scale = 0.0f;
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
                        gBp[bcidx] += grad_h * sp * xcv;
                        gXcp[static_cast<size_t>(row) * dim + chan] +=
                            grad_h * sp * bval;
                        dinput_scale += grad_h * bval * xcv;
                        ddecay += grad_h * h_prev;
                        carry[sidx] = grad_h * decay;
                    }
                }
                gDtp[row * H + h] =
                    dinput_scale * decay_terms.delta_grad -
                    ddecay * decay_terms.decay_dt_factor;
                // N1: per-head gradient to A_log (= dL/dA_eff · A_eff), unconditional.
                gAp[h] -= ddecay * decay_terms.decay_alog_factor;
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
    Tensor grad_conv_pre =
        Tensor::silu_backward(gXc, pp_conv_pre_);
    Tensor grad_xv = Tensor::zeros(std::vector<int>{rows, dim}, dev);
    Tensor grad_conv_w = Tensor::zeros({dim, K}, dev);
    bool conv_backward_done = false;
#ifdef USE_CUDA
    if (dev == Device::GPU) {
        if (determinism::deterministic_reductions_enabled()) {
            conv_backward_done =
                cuda::launch_conv1d_causal_backward_deterministic(
                    grad_conv_pre.raw_data(), pp_xv_.raw_data(),
                    conv_weight_.data.raw_data(), grad_xv.raw_data(),
                    grad_conv_w.raw_data(), batch, seq, dim, K);
        } else {
            cuda::launch_conv1d_causal_backward(
                grad_conv_pre.raw_data(), pp_xv_.raw_data(),
                conv_weight_.data.raw_data(), grad_xv.raw_data(),
                grad_conv_w.raw_data(), batch, seq, dim, K);
            conv_backward_done = true;
        }
        if (conv_backward_done) {
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Mamba conv backward kernel failed: ") +
                    cudaGetErrorString(launch_status));
            }
        }
    }
#endif
    if (!conv_backward_done) {
        auto host = [](const Tensor& value) {
            return value.get_device() == Device::GPU ? value.cpu() : value;
        };
        Tensor gp_h = host(grad_conv_pre);
        Tensor xv_h = host(pp_xv_);
        Tensor cw_h = host(conv_weight_.data);
        Tensor gxv_h = Tensor::zeros({rows, dim}, Device::CPU);
        Tensor gcw_h = Tensor::zeros({dim, K}, Device::CPU);
        conv1d_causal_backward(gp_h.data(), xv_h.data(), cw_h.data(),
                               gxv_h.data(), gcw_h.data(), batch, seq, dim, K);
        grad_xv = dev == Device::GPU ? gxv_h.to(Device::GPU) : gxv_h;
        grad_conv_w = dev == Device::GPU ? gcw_h.to(Device::GPU) : gcw_h;
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

// ===========================================================================
// Faithful Mamba-2 block (state-spaces/mamba Mamba2 defaults).
// ===========================================================================
bool Mamba2SSD::faithful_grouped_projection_eligible(
    const Tensor& input) const {
#ifdef USE_CUDA
    if (!config_.faithful_mamba2 ||
        input.get_device() != Device::GPU ||
        input.shape.size() != 2 ||
        input.shape[1] != d_model) {
        return false;
    }
    const std::array<const BitLinear*, 5> projections = {
        z_proj_.get(), x_proj_.get(), B_proj_.get(), C_proj_.get(),
        dt_proj_.get()};
    const int group_state = checked_int_product(
        n_groups, d_state, "Mamba2 faithful grouped state width");
    const std::array<int, 5> widths = {
        d_inner, d_inner, group_state, group_state,
        n_heads};
    for (size_t index = 0; index < projections.size(); ++index) {
        const BitLinear* projection = projections[index];
        if (projection == nullptr ||
            !projection->exact_linear_mode() ||
            !projection->reference_path_enabled() ||
            projection->loqa.active ||
            projection->input_features() != d_model ||
            projection->output_features() != widths[index] ||
            (index < 4 && projection->uses_bias()) ||
            projection->weight.data.shape !=
                TensorShape({widths[index], d_model}) ||
            projection->weight.data.get_device() != Device::GPU) {
            return false;
        }
    }
    return !dt_proj_->uses_bias() ||
           (dt_proj_->bias.data.shape == TensorShape({n_heads}) &&
            dt_proj_->bias.data.get_device() == Device::GPU);
#else
    (void)input;
    return false;
#endif
}

bool Mamba2SSD::faithful_sensitive_grouped_projection_eligible(
    const Tensor& input) const {
#ifdef USE_CUDA
    if (!config_.faithful_mamba2 ||
        input.get_device() != Device::GPU ||
        input.shape.size() != 2 ||
        input.shape[1] != d_model) {
        return false;
    }
    const int group_state = checked_int_product(
        n_groups, d_state, "Mamba2 faithful grouped state width");
    const std::array<const BitLinear*, 5> projections = {
        z_proj_.get(), x_proj_.get(), B_proj_.get(), C_proj_.get(),
        dt_proj_.get()};
    const std::array<int, 5> widths = {
        d_inner, d_inner, group_state, group_state, n_heads};
    for (size_t index = 0; index < projections.size(); ++index) {
        const BitLinear* projection = projections[index];
        if (projection == nullptr ||
            projection->input_features() != d_model ||
            projection->output_features() != widths[index] ||
            (index < 4 && projection->uses_bias()) ||
            projection->weight.data.shape !=
                TensorShape({widths[index], d_model}) ||
            projection->weight.data.get_device() != Device::GPU) {
            return false;
        }
        // B, C and dt are the mixed-precision-sensitive projections. They
        // must remain exact plain linears for the grouped QAT path; z/x keep
        // their own BitLinear forward/backward and may therefore be ternary.
        if (index >= 2 &&
            (!projection->exact_linear_mode() ||
             !projection->reference_path_enabled() ||
             projection->loqa.active)) {
            return false;
        }
    }
    return !dt_proj_->uses_bias() ||
           (dt_proj_->bias.data.shape == TensorShape({n_heads}) &&
            dt_proj_->bias.data.get_device() == Device::GPU);
#else
    (void)input;
    return false;
#endif
}

void Mamba2SSD::refresh_faithful_grouped_projection_cache() {
    const int GS = checked_int_product(
        n_groups, d_state, "Mamba2 faithful grouped state width");
    const int64_t width64 =
        2LL * d_inner + 2LL * GS + n_heads;
    if (width64 <= 0 ||
        width64 > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "Mamba2 faithful grouped projection width exceeds int range");
    }
    const int width = static_cast<int>(width64);
    const std::array<BitLinear*, 5> projections = {
        z_proj_.get(), x_proj_.get(), B_proj_.get(), C_proj_.get(),
        dt_proj_.get()};
    std::array<uint64_t, 5> versions{};
    for (size_t index = 0; index < projections.size(); ++index) {
        if (projections[index] == nullptr) {
            throw std::logic_error(
                "Mamba2 faithful grouped projection is missing");
        }
        versions[index] = projections[index]->weight.version;
    }
    if (versions != faithful_grouped_projection_versions_) {
        if (faithful_grouped_projection_content_epoch_ ==
            std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error(
                "Mamba2 grouped projection content epoch is exhausted");
        }
        ++faithful_grouped_projection_content_epoch_;
    }
    const bool owner_valid =
        faithful_grouped_projection_weight_.shape ==
            TensorShape({width, d_model}) &&
        faithful_grouped_projection_weight_.get_device() == Device::GPU;
    size_t expected_offset = 0;
    bool storage_bound = owner_valid;
    for (BitLinear* projection : projections) {
        storage_bound =
            storage_bound &&
            projection->weight.data.raw_data() ==
                faithful_grouped_projection_weight_.raw_data() +
                    expected_offset;
        expected_offset +=
            static_cast<size_t>(projection->output_features()) *
            static_cast<size_t>(d_model);
    }
    if (storage_bound) {
        // Optimizer updates changed versions but wrote directly into the
        // canonical packed owner through these aliasing views. No repack/copy
        // is necessary.
        faithful_grouped_projection_versions_ = versions;
        return;
    }

    {
        Tensor next =
            Tensor::uninitialized({width, d_model}, Device::GPU);
        size_t offset = 0;
        for (BitLinear* projection : projections) {
            const size_t elements =
                static_cast<size_t>(projection->output_features()) *
                static_cast<size_t>(d_model);
            copy_tensor_bytes(
                next.raw_data() + offset, Device::GPU,
                projection->weight.data.raw_data(), Device::GPU,
                elements * sizeof(float), /*async_d2d=*/true);
            offset += elements;
        }
        if (offset != static_cast<size_t>(next.size)) {
            throw std::logic_error(
                "Mamba2 faithful grouped projection cache size mismatch");
        }
        faithful_grouped_projection_weight_ = std::move(next);
        faithful_grouped_projection_versions_ = versions;
        ++faithful_grouped_projection_cache_rebuilds_;
    }

    // Turn the independently named Parameter tensors into safe aliasing views
    // over the packed owner. Checkpoint names and optimizer states remain
    // unchanged, while every subsequent training step has zero packing copies.
    size_t offset = 0;
    for (BitLinear* projection : projections) {
        const int rows = projection->output_features();
        projection->weight.data =
            faithful_grouped_projection_weight_.storage_view(
                offset, {rows, d_model});
        offset +=
            static_cast<size_t>(rows) * static_cast<size_t>(d_model);
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
Mamba2SSD::forward_faithful_grouped_projections(
    const Tensor& input) {
#ifndef USE_CUDA
    (void)input;
    throw std::logic_error("Grouped GPU projections are unavailable in a CPU build");
#else
    if (!faithful_grouped_projection_eligible(input)) {
        throw std::logic_error(
            "Mamba2 faithful grouped projection path is ineligible");
    }
    refresh_faithful_grouped_projection_cache();
    const int GS = checked_int_product(
        n_groups, d_state, "Mamba2 faithful grouped state width");
    const int z_begin = 0;
    const int x_begin = z_begin + d_inner;
    const int b_begin = x_begin + d_inner;
    const int c_begin = b_begin + GS;
    const int dt_begin = c_begin + GS;
    const int width = dt_begin + n_heads;
    Tensor projected = matmul_nt_cached_weight(
        input, faithful_grouped_projection_weight_,
        faithful_grouped_projection_content_epoch_);
    if (projected.shape.size() != 2 ||
        projected.shape[1] != width) {
        throw std::logic_error(
            "Mamba2 faithful grouped projection output shape mismatch");
    }
    const int rows = input.shape[0];
    Tensor z = Tensor::uninitialized({rows, d_inner}, Device::GPU);
    Tensor xv = Tensor::uninitialized({rows, d_inner}, Device::GPU);
    Tensor Bv = Tensor::uninitialized({rows, GS}, Device::GPU);
    Tensor Cv = Tensor::uninitialized({rows, GS}, Device::GPU);
    Tensor dt = Tensor::uninitialized({rows, n_heads}, Device::GPU);
    const bool unpack_enqueued =
        cuda::launch_mamba2_unpack_projection(
        projected.raw_data(), z.raw_data(), xv.raw_data(), Bv.raw_data(),
        Cv.raw_data(), dt.raw_data(), rows, d_inner, GS, n_heads,
        false);
    if (!unpack_enqueued) {
        throw std::runtime_error(
            "Mamba2 grouped projection unpack rejected invalid arguments");
    }
    const cudaError_t unpack_status = cudaGetLastError();
    if (unpack_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Mamba2 grouped projection unpack failed: ") +
            cudaGetErrorString(unpack_status));
    }
    if (dt_proj_->uses_bias()) {
        dt = dt.add(dt_proj_->bias.data);
    }
    faithful_grouped_projection_backward_mode_ =
        FaithfulGroupedProjectionMode::Full;
    ++faithful_grouped_projection_forward_calls_;
    ++faithful_grouped_projection_full_forward_calls_;
    return {std::move(xv), std::move(z), std::move(Bv), std::move(Cv),
            std::move(dt)};
#endif
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
Mamba2SSD::forward_faithful_sensitive_grouped_projections(
    const Tensor& input) {
#ifndef USE_CUDA
    (void)input;
    throw std::logic_error("Grouped GPU projections are unavailable in a CPU build");
#else
    if (!faithful_sensitive_grouped_projection_eligible(input)) {
        throw std::logic_error(
            "Mamba2 faithful sensitive grouped projection path is "
            "ineligible");
    }
    refresh_faithful_grouped_projection_cache();
    const int GS = checked_int_product(
        n_groups, d_state, "Mamba2 faithful grouped state width");
    const int64_t sensitive_width64 = 2LL * GS + n_heads;
    if (sensitive_width64 <= 0 ||
        sensitive_width64 > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "Mamba2 sensitive projection width exceeds int range");
    }
    const int sensitive_width = static_cast<int>(sensitive_width64);
    const size_t sensitive_offset =
        static_cast<size_t>(2) * static_cast<size_t>(d_inner) *
        static_cast<size_t>(d_model);
    Tensor sensitive_weight =
        faithful_grouped_projection_weight_.storage_view(
            sensitive_offset, {sensitive_width, d_model});

    Tensor z = z_proj_->forward(input).reshape({input.shape[0], d_inner});
    Tensor xv = x_proj_->forward(input).reshape({input.shape[0], d_inner});
    Tensor projected = matmul_nt_cached_weight(
        input, sensitive_weight,
        faithful_grouped_projection_content_epoch_);
    if (projected.shape !=
        TensorShape({input.shape[0], sensitive_width})) {
        throw std::logic_error(
            "Mamba2 sensitive grouped projection output shape mismatch");
    }
    const int rows = input.shape[0];
    Tensor Bv = Tensor::uninitialized({rows, GS}, Device::GPU);
    Tensor Cv = Tensor::uninitialized({rows, GS}, Device::GPU);
    Tensor dt = Tensor::uninitialized({rows, n_heads}, Device::GPU);
    const bool unpack_enqueued =
        cuda::launch_mamba2_unpack_projection(
        projected.raw_data(), nullptr, nullptr, Bv.raw_data(),
        Cv.raw_data(), dt.raw_data(), rows, d_inner, GS, n_heads,
        true);
    if (!unpack_enqueued) {
        throw std::runtime_error(
            "Mamba2 sensitive grouped projection unpack rejected invalid "
            "arguments");
    }
    const cudaError_t unpack_status = cudaGetLastError();
    if (unpack_status != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "Mamba2 sensitive grouped projection unpack failed: ") +
            cudaGetErrorString(unpack_status));
    }
    if (dt_proj_->uses_bias()) {
        dt = dt.add(dt_proj_->bias.data);
    }
    faithful_grouped_projection_backward_mode_ =
        FaithfulGroupedProjectionMode::Sensitive;
    ++faithful_grouped_projection_forward_calls_;
    ++faithful_grouped_projection_sensitive_forward_calls_;
    return {std::move(xv), std::move(z), std::move(Bv), std::move(Cv),
            std::move(dt)};
#endif
}

Tensor Mamba2SSD::backward_faithful_grouped_projections(
    const Tensor& gx, const Tensor& gz, const Tensor& gB,
    const Tensor& gC, const Tensor& gdt) {
#ifdef USE_CUDA
    const FaithfulGroupedProjectionMode mode =
        faithful_grouped_projection_backward_mode_;
    if (mode == FaithfulGroupedProjectionMode::None ||
        pp_u_.get_device() != Device::GPU) {
        throw std::logic_error(
            "Mamba2 faithful grouped projection backward is inactive");
    }
    const std::array<BitLinear*, 5> projections = {
        z_proj_.get(), x_proj_.get(), B_proj_.get(), C_proj_.get(),
        dt_proj_.get()};
    for (size_t index = 0; index < projections.size(); ++index) {
        if (projections[index] == nullptr ||
            projections[index]->weight.version !=
                faithful_grouped_projection_versions_[index]) {
            throw std::logic_error(
                "Mamba2 projection weight changed between forward and "
                "backward");
        }
    }
    const int rows = pp_u_.shape[0];
    const int GS = checked_int_product(
        n_groups, d_state, "Mamba2 faithful grouped state width");
    const int64_t width64 =
        2LL * d_inner + 2LL * GS + n_heads;
    const int64_t sensitive_width64 = 2LL * GS + n_heads;
    const int64_t active_width64 =
        mode == FaithfulGroupedProjectionMode::Full
            ? width64
            : sensitive_width64;
    if (active_width64 <= 0 ||
        active_width64 > std::numeric_limits<int>::max() ||
        static_cast<int64_t>(rows) * active_width64 >
            std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "Mamba2 grouped projection backward exceeds kernel index range");
    }
    const auto require_shape =
        [rows](const Tensor& value, int columns, const char* label) {
            if (value.shape != TensorShape({rows, columns}) ||
                value.get_device() != Device::GPU) {
                throw std::invalid_argument(
                    std::string("Mamba2 grouped projection ") + label +
                    " gradient shape/device mismatch");
            }
        };
    require_shape(gx, d_inner, "x");
    require_shape(gz, d_inner, "z");
    require_shape(gB, GS, "B");
    require_shape(gC, GS, "C");
    require_shape(gdt, n_heads, "dt");

    const int active_width = static_cast<int>(active_width64);
    Tensor packed_grad =
        Tensor::uninitialized({rows, active_width}, Device::GPU);
    bool pack_enqueued = false;
    if (mode == FaithfulGroupedProjectionMode::Full) {
        pack_enqueued = cuda::launch_mamba2_pack_projection_grads(
            gx.raw_data(), gz.raw_data(), gB.raw_data(), gC.raw_data(),
            gdt.raw_data(), packed_grad.raw_data(), rows, d_inner, GS,
            n_heads);
    } else {
        pack_enqueued =
            cuda::launch_mamba2_pack_sensitive_projection_grads(
            gB.raw_data(), gC.raw_data(), gdt.raw_data(),
            packed_grad.raw_data(), rows, GS, n_heads);
    }
    if (!pack_enqueued) {
        throw std::runtime_error(
            "Mamba2 grouped projection gradient pack rejected invalid "
            "arguments");
    }
    const cudaError_t pack_status = cudaGetLastError();
    if (pack_status != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "Mamba2 grouped projection gradient pack failed: ") +
            cudaGetErrorString(pack_status));
    }

    // Consume the grouped forward exactly once. Any later exception leaves the
    // training step failed instead of allowing a duplicate partial backward.
    faithful_grouped_projection_backward_mode_ =
        FaithfulGroupedProjectionMode::None;
    if (dt_proj_->uses_bias()) {
        dt_proj_->bias.add_grad(gdt.sum(0));
    }
    Tensor all_weight_grad =
        matmul_tn(packed_grad, pp_u_);

    Tensor input_grad;
    if (mode == FaithfulGroupedProjectionMode::Full) {
        const int z_begin = 0;
        const int x_begin = z_begin + d_inner;
        const int b_begin = x_begin + d_inner;
        const int c_begin = b_begin + GS;
        const int dt_begin = c_begin + GS;
        z_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(z_begin) * d_model,
                {d_inner, d_model}));
        x_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(x_begin) * d_model,
                {d_inner, d_model}));
        B_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(b_begin) * d_model,
                {GS, d_model}));
        C_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(c_begin) * d_model,
                {GS, d_model}));
        dt_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(dt_begin) * d_model,
                {n_heads, d_model}));
        input_grad =
            packed_grad.matmul(faithful_grouped_projection_weight_);
        ++faithful_grouped_projection_full_backward_calls_;
    } else {
        const int c_begin = GS;
        const int dt_begin = 2 * GS;
        B_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                0, {GS, d_model}));
        C_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(c_begin) * d_model,
                {GS, d_model}));
        dt_proj_->weight.add_grad(
            all_weight_grad.storage_view(
                static_cast<size_t>(dt_begin) * d_model,
                {n_heads, d_model}));

        Tensor z_input_grad =
            z_proj_->backward(gz).reshape({rows, d_model});
        input_grad =
            x_proj_->backward(gx).reshape({rows, d_model});
        input_grad = input_grad.add(z_input_grad);
        const size_t sensitive_offset =
            static_cast<size_t>(2) * static_cast<size_t>(d_inner) *
            static_cast<size_t>(d_model);
        Tensor sensitive_weight =
            faithful_grouped_projection_weight_.storage_view(
                sensitive_offset, {active_width, d_model});
        input_grad =
            input_grad.add(packed_grad.matmul(sensitive_weight));
        ++faithful_grouped_projection_sensitive_backward_calls_;
    }
    ++faithful_grouped_projection_backward_calls_;
    return input_grad;
#else
    (void)gx;
    (void)gz;
    (void)gB;
    (void)gC;
    (void)gdt;
    throw std::logic_error(
        "Mamba2 grouped projection backward requires a GPU backend");
#endif
}

Tensor Mamba2SSD::forward_faithful(const Tensor& u) {
    const bool rank_2 = u.shape.size() == 2;
    if (!rank_2 && u.shape.size() != 3) {
        throw std::runtime_error("Mamba2 faithful path expects rank 2 or 3");
    }
    const int batch = rank_2 ? 1 : u.shape[0];
    const int seq = rank_2 ? u.shape[0] : u.shape[1];
    if (u.shape.back() != d_model || d_inner != n_heads * d_head) {
        throw std::runtime_error("Mamba2 faithful path shape mismatch");
    }
    const int rows =
        checked_int_product(batch, seq, "Mamba2 faithful forward rows");
    const int N = d_state;
    const int GS = n_groups * N;
    const int K = conv_kernel_;
    const Device dev = u.get_device();
    validate_faithful_conv_launch(rows, d_inner, GS, K);
    Tensor u_flat = rank_2 ? u : u.reshape({rows, d_model});

    faithful_grouped_projection_backward_mode_ =
        FaithfulGroupedProjectionMode::None;
    Tensor xv;
    Tensor z;
    Tensor Bv;
    Tensor Cv;
    Tensor dt;
    if (faithful_grouped_projection_eligible(u_flat)) {
        std::tie(xv, z, Bv, Cv, dt) =
            forward_faithful_grouped_projections(u_flat);
    } else if (faithful_sensitive_grouped_projection_eligible(u_flat)) {
        std::tie(xv, z, Bv, Cv, dt) =
            forward_faithful_sensitive_grouped_projections(u_flat);
    } else {
        xv = x_proj_->forward(u_flat).reshape({rows, d_inner});
        z = z_proj_->forward(u_flat).reshape({rows, d_inner});
        Bv = B_proj_->forward(u_flat).reshape({rows, GS});
        Cv = C_proj_->forward(u_flat).reshape({rows, GS});
        dt = dt_proj_->forward(u_flat).reshape({rows, n_heads});
    }

    // The causal-convolution producers cover every output element.
    Tensor x_pre = Tensor::uninitialized({rows, d_inner}, dev);
    Tensor B_pre = Tensor::uninitialized({rows, GS}, dev);
    Tensor C_pre = Tensor::uninitialized({rows, GS}, dev);
    Tensor x = Tensor::uninitialized({rows, d_inner}, dev);
    Tensor B = Tensor::uninitialized({rows, GS}, dev);
    Tensor C = Tensor::uninitialized({rows, GS}, dev);
#ifdef USE_CUDA
    if (dev == Device::GPU) {
        const bool conv_enqueued =
            cuda::launch_mamba2_faithful_conv_forward(
            xv.raw_data(), Bv.raw_data(), Cv.raw_data(),
            conv_weight_.data.raw_data(), conv_bias_.data.raw_data(),
            x_pre.raw_data(), B_pre.raw_data(), C_pre.raw_data(),
            x.raw_data(), B.raw_data(), C.raw_data(), batch, seq,
            d_inner, GS, K);
        if (!conv_enqueued) {
            throw std::runtime_error(
                "Mamba2 faithful convolution rejected invalid arguments");
        }
        const cudaError_t conv_status = cudaGetLastError();
        if (conv_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Mamba2 faithful convolution kernel failed: ") +
                cudaGetErrorString(conv_status));
        }
    } else
#endif
    {
        conv1d_causal_forward(xv.data(), conv_weight_.data.data(), x_pre.data(),
                              batch, seq, d_inner, K);
        conv1d_causal_forward(
            Bv.data(), conv_weight_.data.data() +
                           static_cast<size_t>(d_inner) * K,
            B_pre.data(), batch, seq, GS, K);
        conv1d_causal_forward(
            Cv.data(), conv_weight_.data.data() +
                           static_cast<size_t>(d_inner + GS) * K,
            C_pre.data(), batch, seq, GS, K);
        x_pre = x_pre.add(conv_bias_.data.slice(0, 0, d_inner));
        B_pre = B_pre.add(
            conv_bias_.data.slice(0, d_inner, d_inner + GS));
        C_pre = C_pre.add(
            conv_bias_.data.slice(0, d_inner + GS, d_inner + 2 * GS));
        x = x_pre.silu();
        B = B_pre.silu();
        C = C_pre.silu();
    }

    int history_chunk_size = 0;
#ifdef USE_CUDA
    if (dev == Device::GPU && training_mode_ && !streaming_inference_ &&
        cuda::faithful_boundary_history_enabled()) {
        const int wave = ensure_faithful_runtime_warp_size(faithful_runtime_warp_size_);
        if (!determinism::deterministic_reductions_enabled() ||
            !cuda::faithful_chunked_backward_enabled() ||
            cuda::faithful_backward_chunk_size() != 32 || N > cuda::mamba_nstate_max_n() ||
            !cuda::faithful_deterministic_head_wave_geometry_enabled(d_head, wave))
            throw std::invalid_argument("Boundary history requires deterministic head-wave backward with chunk size 32");
        history_chunk_size = 32;
    }
#endif
    const bool save_history = history_chunk_size > 0 ||
        streaming_inference_ ||
        (training_mode_ &&
         (!config_.recompute_ssd || faithful_recompute_active_));
    if (save_history && config_.max_seq_for_storage > 0 &&
        seq > config_.max_seq_for_storage) {
        throw std::runtime_error(
            "Mamba2 faithful history exceeds max_seq_for_storage; use "
            "sequence chunking or raise the explicit safety limit");
    }
    Tensor y = Tensor::uninitialized({rows, d_inner}, dev);
    Tensor hist;
    bool state_major_history = false;
#ifdef USE_CUDA
    state_major_history =
        save_history && dev == Device::GPU && training_mode_ &&
        !streaming_inference_ &&
        cuda::faithful_state_major_history_enabled();
#endif
    if (save_history) {
        const int history_rows = history_chunk_size > 0
            ? checked_int_product(batch, (seq + history_chunk_size - 1) / history_chunk_size,
                                  "Mamba boundary rows") : rows;
        hist = state_major_history
                   ? Tensor::uninitialized(
                         {history_rows, N, n_heads, d_head}, dev)
                   : Tensor::uninitialized(
                         {history_rows, n_heads, d_head, N}, dev);
        faithful_peak_state_history_bytes_ =
            std::max(
                faithful_peak_state_history_bytes_,
                static_cast<size_t>(hist.size) * sizeof(float));
    }
    bool scan_done = false;
    Tensor forward_decay_terms;
#ifdef USE_CUDA
    if (dev == Device::GPU && N <= cuda::mamba_nstate_max_n()) {
        if (cuda::faithful_precomputed_decay_enabled()) {
            forward_decay_terms = Tensor::uninitialized(
                {rows, n_heads, cuda::kFaithfulDecayTermsWidth},
                Device::GPU);
            const bool decay_enqueued =
                cuda::launch_mamba2_faithful_precompute_decay(
                    dt.raw_data(), A.data.raw_data(),
                    forward_decay_terms.raw_data(), batch, seq,
                    n_heads);
            if (!decay_enqueued) {
                throw std::runtime_error(
                    "Mamba2 faithful decay precompute rejected invalid "
                    "arguments");
            }
            const cudaError_t decay_status = cudaGetLastError();
            if (decay_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Mamba2 faithful decay precompute failed: ") +
                    cudaGetErrorString(decay_status));
            }
        }
        const int forward_warp_size =
            ensure_faithful_runtime_warp_size(faithful_runtime_warp_size_);
        const float* forward_decay_ptr =
            forward_decay_terms.size > 0
                ? forward_decay_terms.raw_data()
                : nullptr;
        bool launch_enqueued = false;
        bool forward_used_chunked = false;

        // Time-parallel scan: only attempted when explicitly enabled and the
        // shape is eligible.  The launcher itself re-validates and returns
        // false without touching any output, so the sequential path below
        // stays the guaranteed fallback.
        if (cuda::faithful_chunked_forward_enabled() &&
            d_head % forward_warp_size == 0) {
            const int forward_chunk_size =
                cuda::faithful_forward_chunk_size();
            const int forward_chunks =
                (seq + forward_chunk_size - 1) / forward_chunk_size;
            if (forward_chunks > 1) {
                const int forward_batch_heads = checked_int_product(
                    batch, n_heads, "Mamba2 forward chunk batch heads");
                const int forward_channel_values = checked_int_product(
                    forward_batch_heads, d_head,
                    "Mamba2 forward chunk channel values");
                const int forward_channel_states = checked_int_product(
                    forward_channel_values, N,
                    "Mamba2 forward chunk channel states");
                const int forward_state_values = checked_int_product(
                    forward_channel_states, forward_chunks,
                    "Mamba2 forward chunk state values");
                const int forward_decay_values = checked_int_product(
                    forward_batch_heads, forward_chunks,
                    "Mamba2 forward chunk decay values");
                FaithfulForwardChunkWorkspace& forward_workspace =
                    faithful_forward_chunk_workspace();
                forward_workspace.ensure(forward_state_values,
                                         forward_decay_values);
                int produced_chunks = 0;
                forward_used_chunked =
                    cuda::launch_mamba2_faithful_forward_chunked(
                        x.raw_data(), dt.raw_data(), A.data.raw_data(),
                        forward_decay_ptr, B.raw_data(), C.raw_data(),
                        D.data.raw_data(), y.raw_data(),
                        save_history ? hist.raw_data() : nullptr,
                        forward_workspace.end_local.raw_data(),
                        forward_workspace.total_decay.raw_data(),
                        forward_workspace.carry.raw_data(), batch, seq,
                        n_heads, d_head, N, n_groups, forward_warp_size,
                        state_major_history, &produced_chunks, history_chunk_size);
                launch_enqueued = forward_used_chunked;
            }
        }

        if (!forward_used_chunked) {
            launch_enqueued = cuda::launch_mamba2_faithful_forward(
                x.raw_data(), dt.raw_data(), A.data.raw_data(),
                forward_decay_ptr,
                B.raw_data(), C.raw_data(), D.data.raw_data(),
                y.raw_data(),
                save_history ? hist.raw_data() : nullptr,
                batch, seq, n_heads, d_head, N, n_groups,
                forward_warp_size,
                state_major_history, history_chunk_size);
        }
        if (!launch_enqueued) {
            throw std::runtime_error(
                "Mamba2 faithful forward rejected invalid arguments or "
                "failed to enqueue required device initialization");
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Mamba2 faithful forward kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        scan_done = true;
        ++gpu_fast_path_hits_;
        ++faithful_forward_gpu_calls_;
    }
#endif
    if (!scan_done) {
#ifdef USE_CUDA
        if (dev == Device::GPU) {
            ++gpu_fast_path_fallbacks_;
            ++faithful_forward_host_fallbacks_;
            last_fallback_reason_ =
                "faithful_forward_host_fallback";
        }
        if (dev == Device::GPU && strict_gpu_execution()) {
            throw std::runtime_error(
                "Strict GPU Mamba2 faithful forward has no eligible "
                "device scan");
        }
#endif
        Tensor xh = dev == Device::GPU ? x.cpu() : x;
        Tensor Bh = dev == Device::GPU ? B.cpu() : B;
        Tensor Ch = dev == Device::GPU ? C.cpu() : C;
        Tensor dth = dev == Device::GPU ? dt.cpu() : dt;
        Tensor Ah = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
        Tensor Dh = D.data.get_device() == Device::GPU ? D.data.cpu() : D.data;
        Tensor yh({rows, d_inner}, Device::CPU);
        Tensor hh;
        if (save_history) {
            hh = Tensor({rows, n_heads, d_head, N}, Device::CPU);
        }
        const float* xp = xh.data();
        const float* bp = Bh.data();
        const float* cp = Ch.data();
        const float* dp = dth.data();
        const float* ap = Ah.data();
        const float* Dp = Dh.data();
        float* yp = yh.data();
        float* hp = save_history ? hh.data() : nullptr;
        const size_t state_size = static_cast<size_t>(d_inner) * N;
        std::vector<float> state(state_size, 0.0f);
        for (int b = 0; b < batch; ++b) {
            std::fill(state.begin(), state.end(), 0.0f);
            for (int t = 0; t < seq; ++t) {
                const int row = b * seq + t;
                for (int h = 0; h < n_heads; ++h) {
                    const int group = (h * n_groups) / n_heads;
                    const MambaDecayTerms decay_terms =
                        mamba_decay_terms(dp[row * n_heads + h], ap[h]);
                    const float delta = decay_terms.delta;
                    const float decay = decay_terms.decay;
                    for (int p = 0; p < d_head; ++p) {
                        const int chan = h * d_head + p;
                        const float xv_local =
                            xp[static_cast<size_t>(row) * d_inner + chan];
                        float out = Dp[h] * xv_local;
                        for (int n = 0; n < N; ++n) {
                            const size_t si =
                                static_cast<size_t>(chan) * N + n;
                            const size_t bi =
                                static_cast<size_t>(row) * GS + group * N + n;
                            const float hv =
                                decay * state[si] +
                                delta * bp[bi] * xv_local;
                            state[si] = hv;
                            if (hp != nullptr) {
                                hp[static_cast<size_t>(row) * state_size + si] = hv;
                            }
                            out += hv * cp[bi];
                        }
                        yp[static_cast<size_t>(row) * d_inner + chan] = out;
                    }
                }
            }
        }
        y = dev == Device::GPU ? yh.to(Device::GPU) : yh;
        if (save_history) {
            hist = dev == Device::GPU ? hh.to(Device::GPU) : hh;
        }
    }

    // norm_before_gate=false: RMSNorm(y * SiLU(z)), then learned gamma.
    Tensor gated_input = Tensor::silu_gate(y, z);
    Tensor gated_norm = gated_input.rmsnorm(config_.rms_norm_eps);
    Tensor normalized = gated_norm.mul(norm_weight_.data);
    Tensor result = out_proj.forward(normalized).reshape({rows, d_model});

    const bool checkpoint_forward =
        training_mode_ && config_.recompute_ssd &&
        !faithful_recompute_active_ && !streaming_inference_;
    faithful_checkpoint_input_ = checkpoint_forward ? u : Tensor();
    // On a supported GPU, checkpoint only the O(B*S*d_inner*d_state) history.
    // The projections/convolution/gating intermediates are comparatively
    // small and are required by backward anyway. Retaining them avoids the
    // historical full-block recompute (all GEMMs, convolution, norms and
    // output projection) while preserving the dominant history saving.
    const bool selective_history_checkpoint =
        checkpoint_forward && dev == Device::GPU && scan_done && history_chunk_size == 0;
    if (checkpoint_forward && !selective_history_checkpoint && history_chunk_size == 0) {
        pp_u_ = Tensor();
        pp_xv_ = Tensor();
        pp_Bv_ = Tensor();
        pp_Cv_ = Tensor();
        pp_conv_pre_ = Tensor();
        pp_B_conv_pre_ = Tensor();
        pp_C_conv_pre_ = Tensor();
        pp_xc_ = Tensor();
        pp_B_ = Tensor();
        pp_C_ = Tensor();
        pp_dt_ = Tensor();
        pp_decay_terms_ = Tensor();
        pp_z_ = Tensor();
        pp_y_ssd_ = Tensor();
        pp_state_hist_ = Tensor();
        pp_state_hist_chunk_size_ = 0;
        pp_state_hist_state_major_ = false;
        pp_gated_input_ = Tensor();
        pp_gated_norm_ = Tensor();
        x_proj_->discard_backward_state();
        z_proj_->discard_backward_state();
        B_proj_->discard_backward_state();
        C_proj_->discard_backward_state();
        dt_proj_->discard_backward_state();
        out_proj.discard_backward_state();
        faithful_grouped_projection_backward_mode_ =
            FaithfulGroupedProjectionMode::None;
    } else {
        pp_u_ = u_flat;
        pp_xv_ = xv;
        pp_Bv_ = Bv;
        pp_Cv_ = Cv;
        pp_conv_pre_ = x_pre;
        pp_B_conv_pre_ = B_pre;
        pp_C_conv_pre_ = C_pre;
        pp_xc_ = x;
        pp_B_ = B;
        pp_C_ = C;
        pp_dt_ = dt;
        pp_decay_terms_ =
            training_mode_ && forward_decay_terms.size > 0
                ? forward_decay_terms
                : Tensor();
        pp_z_ = z;
        pp_y_ssd_ = y;
        pp_state_hist_ =
            selective_history_checkpoint ? Tensor() : hist;
        pp_state_hist_state_major_ =
            !selective_history_checkpoint && state_major_history;
        pp_state_hist_chunk_size_ = history_chunk_size;
        pp_gated_input_ = gated_input;
        pp_gated_norm_ = gated_norm;
    }
    pp_batch_ = batch;
    pp_seq_ = seq;
    proper_active_ = true;

    if (streaming_inference_) {
        const int taps = std::max(K - 1, 0);
        const int state_elements = checked_int_product(
            d_inner, N, "faithful Mamba streaming state");
        const int ring_elements = checked_int_product(
            taps, conv_dim, "faithful Mamba streaming ring");
#ifdef USE_CUDA
        if (dev == Device::GPU && batch == 1 && scan_done &&
            mamba_gpu_step_enabled()) {
            pp_stream_h_dev_ = Tensor::uninitialized(
                {state_elements}, Device::GPU);
            pp_stream_ring_dev_ = Tensor::uninitialized(
                {std::max(ring_elements, 1)}, Device::GPU);
            const bool prime_enqueued =
                cuda::launch_mamba_prime_stream_carry(
                hist.raw_data(), state_elements,
                xv.raw_data(), d_inner,
                Bv.raw_data(), GS,
                Cv.raw_data(), GS,
                pp_stream_h_dev_.raw_data(),
                pp_stream_ring_dev_.raw_data(),
                batch, seq, taps);
            if (!prime_enqueued) {
                throw std::runtime_error(
                    "Faithful Mamba-2 GPU stream priming rejected invalid "
                    "arguments");
            }
            const cudaError_t prime_status = cudaGetLastError();
            if (prime_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Faithful Mamba-2 GPU stream priming failed: ") +
                    cudaGetErrorString(prime_status));
            }
            pp_stream_state_.clear();
            pp_stream_ring_.clear();
            pp_stream_dev_live_ = true;
            ++stream_priming_gpu_calls_;
        } else
#endif
        {
#ifdef USE_CUDA
            if (dev == Device::GPU) {
                ++stream_priming_host_fallbacks_;
                ++gpu_fast_path_fallbacks_;
                last_fallback_reason_ =
                    "faithful_stream_priming_host_fallback";
                if (strict_gpu_execution()) {
                    throw std::runtime_error(
                        "Strict GPU faithful Mamba-2 prefill cannot prime "
                        "incremental carry through host memory");
                }
            }
#endif
            Tensor hh =
                dev == Device::GPU ? hist.cpu() : hist;
            Tensor xvh =
                dev == Device::GPU ? xv.cpu() : xv;
            Tensor Bvh =
                dev == Device::GPU ? Bv.cpu() : Bv;
            Tensor Cvh =
                dev == Device::GPU ? Cv.cpu() : Cv;
            pp_stream_state_.assign(
                hh.data() +
                    static_cast<size_t>(rows - 1) * state_elements,
                hh.data() +
                    static_cast<size_t>(rows) * state_elements);
            pp_stream_ring_.assign(
                static_cast<size_t>(ring_elements), 0.0f);
            const int batch_base = (batch - 1) * seq;
            for (int s = 0; s < taps; ++s) {
                const int source_time = seq - taps + s;
                if (source_time < 0) continue;
                const int source = batch_base + source_time;
                float* dst = pp_stream_ring_.data() +
                             static_cast<size_t>(s) * conv_dim;
                std::copy_n(
                    xvh.data() +
                        static_cast<size_t>(source) * d_inner,
                    d_inner, dst);
                std::copy_n(
                    Bvh.data() + static_cast<size_t>(source) * GS,
                    GS, dst + d_inner);
                std::copy_n(
                    Cvh.data() + static_cast<size_t>(source) * GS,
                    GS, dst + d_inner + GS);
            }
            pp_stream_dev_live_ = false;
        }
        pp_stream_active_ = true;
    }

    return rank_2 ? result : result.reshape({batch, seq, d_model});
}

Tensor Mamba2SSD::backward_faithful(const Tensor& grad_output) {
    MambaStageTimer stage_timer(grad_output.get_device());
    if (config_.recompute_ssd && pp_state_hist_.size == 0) {
        ++faithful_recompute_forwards_;
        bool history_restored = false;
#ifdef USE_CUDA
        if (pp_xc_.get_device() == Device::GPU &&
            pp_xc_.shape ==
                TensorShape({checked_int_product(
                                 pp_batch_, pp_seq_,
                                 "Mamba2 checkpoint rows"),
                             d_inner}) &&
            pp_dt_.shape ==
                TensorShape({checked_int_product(
                                 pp_batch_, pp_seq_,
                                 "Mamba2 checkpoint rows"),
                             n_heads}) &&
            pp_B_.shape ==
                TensorShape({checked_int_product(
                                 pp_batch_, pp_seq_,
                                 "Mamba2 checkpoint rows"),
                             n_groups * d_state}) &&
            pp_C_.shape == pp_B_.shape &&
            d_state <= cuda::mamba_nstate_max_n()) {
            const int rows = checked_int_product(
                pp_batch_, pp_seq_, "Mamba2 checkpoint rows");
            const bool state_major_history =
                cuda::faithful_state_major_history_enabled() &&
                !streaming_inference_;
            Tensor history =
                state_major_history
                    ? Tensor::uninitialized(
                          {rows, d_state, n_heads, d_head}, Device::GPU)
                    : Tensor::uninitialized(
                          {rows, n_heads, d_head, d_state}, Device::GPU);
            Tensor output_scratch =
                Tensor::uninitialized({rows, d_inner}, Device::GPU);
            Tensor decay_terms = pp_decay_terms_;
            if (cuda::faithful_precomputed_decay_enabled()) {
                const TensorShape expected_decay_shape(
                    {rows, n_heads, cuda::kFaithfulDecayTermsWidth});
                if (decay_terms.get_device() != Device::GPU ||
                    decay_terms.shape != expected_decay_shape) {
                    decay_terms = Tensor::uninitialized(
                        expected_decay_shape.dims, Device::GPU);
                    const bool decay_enqueued =
                        cuda::launch_mamba2_faithful_precompute_decay(
                            pp_dt_.raw_data(), A.data.raw_data(),
                            decay_terms.raw_data(), pp_batch_, pp_seq_,
                            n_heads);
                    if (!decay_enqueued) {
                        throw std::runtime_error(
                            "Mamba2 checkpoint decay precompute rejected "
                            "invalid arguments");
                    }
                    const cudaError_t decay_status = cudaGetLastError();
                    if (decay_status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string(
                                "Mamba2 checkpoint decay precompute failed: ") +
                            cudaGetErrorString(decay_status));
                    }
                    pp_decay_terms_ = decay_terms;
                }
            }
            const bool launch_enqueued =
                cuda::launch_mamba2_faithful_forward(
                pp_xc_.raw_data(), pp_dt_.raw_data(),
                A.data.raw_data(),
                decay_terms.size > 0 ? decay_terms.raw_data() : nullptr,
                pp_B_.raw_data(), pp_C_.raw_data(),
                D.data.raw_data(), output_scratch.raw_data(),
                history.raw_data(), pp_batch_, pp_seq_, n_heads, d_head,
                d_state, n_groups,
                ensure_faithful_runtime_warp_size(
                    faithful_runtime_warp_size_),
                state_major_history);
            if (!launch_enqueued) {
                throw std::runtime_error(
                    "Mamba2 selective checkpoint history recompute rejected "
                    "invalid arguments");
            }
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Mamba2 selective checkpoint history recompute "
                        "failed: ") +
                    cudaGetErrorString(launch_status));
            }
            faithful_peak_state_history_bytes_ =
                std::max(
                    faithful_peak_state_history_bytes_,
                    static_cast<size_t>(history.size) * sizeof(float));
            pp_state_hist_ = std::move(history);
            pp_state_hist_state_major_ = state_major_history;
            ++gpu_fast_path_hits_;
            ++faithful_forward_gpu_calls_;
            ++faithful_selective_history_recomputes_;
            history_restored = true;
        }
#endif
        if (!history_restored) {
            if (faithful_checkpoint_input_.size == 0) {
                throw std::runtime_error(
                    "Mamba2 faithful checkpoint input is unavailable");
            }
            Tensor checkpoint_input = faithful_checkpoint_input_;
            faithful_recompute_active_ = true;
            try {
                (void)forward_faithful(checkpoint_input);
            } catch (...) {
                faithful_recompute_active_ = false;
                throw;
            }
            faithful_recompute_active_ = false;
            ++faithful_full_block_recompute_forwards_;
        }
    }
    stage_timer.mark("history");
    const int batch = pp_batch_;
    const int seq = pp_seq_;
    const int rows =
        checked_int_product(batch, seq, "Mamba2 faithful backward rows");
    const int N = d_state;
    const int GS = n_groups * N;
    const int K = conv_kernel_;
    const Device dev = grad_output.get_device();
    validate_faithful_conv_launch(rows, d_inner, GS, K);
    Tensor go = grad_output.shape.size() == 2
                    ? grad_output
                    : grad_output.reshape({rows, d_model});

    Tensor gnormalized =
        out_proj.backward(go).reshape({rows, d_inner});
    norm_weight_.add_grad(gnormalized.mul(pp_gated_norm_).sum(0));
    Tensor ggated_norm = gnormalized.mul(norm_weight_.data);
    Tensor ggated = pp_gated_input_.rmsnorm_backward(
        ggated_norm, pp_gated_norm_, config_.rms_norm_eps);
    auto [gy, gz] = Tensor::silu_gate_backward(
        ggated, pp_y_ssd_, pp_z_);
    stage_timer.mark("outnorm_gate");

    // The GPU launcher initializes every reduction destination itself and
    // writes gX exhaustively. The CPU fallback replaces each tensor below.
    Tensor gx = Tensor::uninitialized({rows, d_inner}, dev);
    Tensor gdt = Tensor::uninitialized({rows, n_heads}, dev);
    Tensor gA = Tensor::uninitialized({n_heads}, dev);
    Tensor gB = Tensor::uninitialized({rows, GS}, dev);
    Tensor gC = Tensor::uninitialized({rows, GS}, dev);
    Tensor gD = Tensor::uninitialized({n_heads}, dev);
    // Keep deterministic two-stage workspaces alive through the complete
    // backward. The device allocator may recycle a destroyed Tensor before
    // asynchronous reduction consumers on the default stream have finished.
    Tensor deterministic_partial_B;
    Tensor deterministic_partial_C;
    Tensor deterministic_partial_dt;
    Tensor deterministic_partial_A;
    Tensor deterministic_partial_D;
    Tensor deterministic_gh_history;
    float* deterministic_chunk_carry = nullptr;
    float* deterministic_chunk_scale = nullptr;
    float* deterministic_chunk_lane_A = nullptr;
    float* deterministic_chunk_lane_D = nullptr;
    Tensor backward_decay_terms;
    bool scan_done = false;
#ifdef USE_CUDA
    if (dev == Device::GPU && N <= cuda::mamba_nstate_max_n()) {
        const bool chunked_requested =
            cuda::faithful_chunked_backward_enabled() &&
            (seq > cuda::faithful_backward_chunk_size() || pp_state_hist_chunk_size_ > 0);
        if (cuda::faithful_precomputed_decay_enabled() ||
            chunked_requested) {
            const TensorShape expected_decay_shape(
                {rows, n_heads, cuda::kFaithfulDecayTermsWidth});
            if (pp_decay_terms_.get_device() == Device::GPU &&
                pp_decay_terms_.shape == expected_decay_shape) {
                backward_decay_terms = pp_decay_terms_;
            } else {
                backward_decay_terms = Tensor::uninitialized(
                    expected_decay_shape.dims, Device::GPU);
                const bool decay_enqueued =
                    cuda::launch_mamba2_faithful_precompute_decay(
                        pp_dt_.raw_data(), A.data.raw_data(),
                        backward_decay_terms.raw_data(), batch, seq,
                        n_heads);
                if (!decay_enqueued) {
                    throw std::runtime_error(
                        "Mamba2 backward decay precompute rejected invalid "
                        "arguments");
                }
                const cudaError_t decay_status = cudaGetLastError();
                if (decay_status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string(
                            "Mamba2 backward decay precompute failed: ") +
                        cudaGetErrorString(decay_status));
                }
            }
        }
        ensure_faithful_runtime_warp_size(faithful_runtime_warp_size_);
        const bool deterministic =
            determinism::deterministic_reductions_enabled();
        const bool deterministic_gpu_eligible =
            deterministic &&
            d_head % faithful_runtime_warp_size_ == 0;
        if (deterministic_gpu_eligible) {
            const int waves_per_head =
                d_head / faithful_runtime_warp_size_;
            const bool head_wave_geometry =
                cuda::faithful_deterministic_head_wave_geometry_enabled(
                    d_head, faithful_runtime_warp_size_);
            const bool chunked =
                chunked_requested && head_wave_geometry;
            if (chunked &&
                (cuda::faithful_state_parallel_backward_enabled() ||
                 cuda::faithful_deterministic_shared_carry_enabled())) {
                throw std::invalid_argument(
                    "NSOS_MAMBA_CHUNKED_BACKWARD cannot be combined with "
                    "state-parallel or shared-carry experimental paths");
            }
            const int partials_per_group = checked_int_product(
                n_heads / n_groups, waves_per_head,
                "Mamba2 deterministic partials per group");
            const int bc_values = checked_int_product(
                rows, GS, "Mamba2 deterministic B/C values");
            const int bc_partials = checked_int_product(
                bc_values, partials_per_group,
                "Mamba2 deterministic B/C partials");
            const int dt_values = checked_int_product(
                rows, n_heads, "Mamba2 deterministic dt values");
            const int dt_partials = checked_int_product(
                dt_values, waves_per_head,
                "Mamba2 deterministic dt partials");
            const int ad_values = checked_int_product(
                batch, n_heads, "Mamba2 deterministic A/D values");
            const int ad_partials = checked_int_product(
                ad_values, waves_per_head,
                "Mamba2 deterministic A/D partials");
            deterministic_partial_B =
                Tensor::uninitialized({bc_partials}, Device::GPU);
            deterministic_partial_C =
                Tensor::uninitialized({bc_partials}, Device::GPU);
            deterministic_partial_dt =
                Tensor::uninitialized({dt_partials}, Device::GPU);
            deterministic_partial_A =
                Tensor::uninitialized({ad_partials}, Device::GPU);
            deterministic_partial_D =
                Tensor::uninitialized({ad_partials}, Device::GPU);
            if (chunked) {
                const int chunk_size =
                    cuda::faithful_backward_chunk_size();
                const int chunks =
                    (seq + chunk_size - 1) / chunk_size;
                const int batch_heads = checked_int_product(
                    batch, n_heads,
                    "Mamba2 chunked batch heads");
                const int channel_values = checked_int_product(
                    batch_heads, d_head,
                    "Mamba2 chunked channel values");
                const int channel_states = checked_int_product(
                    channel_values, N,
                    "Mamba2 chunked channel states");
                const int carry_values = checked_int_product(
                    channel_states, chunks,
                    "Mamba2 chunked carry values");
                const int scale_values = checked_int_product(
                    ad_values, chunks,
                    "Mamba2 chunked scale values");
                const int lane_partial_chunks = checked_int_product(
                    ad_partials, chunks,
                    "Mamba2 chunked lane partial chunks");
                const int lane_partial_values = checked_int_product(
                    lane_partial_chunks, faithful_runtime_warp_size_,
                    "Mamba2 chunked lane partial values");
                FaithfulChunkWorkspace& workspace =
                    faithful_chunk_workspace();
                workspace.ensure(carry_values, scale_values,
                                 lane_partial_values);
                deterministic_chunk_carry = workspace.carry.raw_data();
                deterministic_chunk_scale = workspace.scale.raw_data();
                deterministic_chunk_lane_A = workspace.lane_a.raw_data();
                deterministic_chunk_lane_D = workspace.lane_d.raw_data();
            } else if (
                cuda::faithful_state_parallel_backward_enabled() &&
                head_wave_geometry) {
                const int row_channels = checked_int_product(
                    rows, d_inner,
                    "Mamba2 state-parallel row channels");
                const int state_values = checked_int_product(
                    row_channels, N,
                    "Mamba2 state-parallel history values");
                FaithfulStateParallelWorkspace& workspace =
                    faithful_state_parallel_workspace();
                workspace.ensure(state_values);
                deterministic_gh_history = workspace.gh_history;
            }
            const bool launch_enqueued =
                chunked
                    ? cuda::launch_mamba2_faithful_backward_deterministic_chunked(
                          gy.raw_data(), pp_xc_.raw_data(),
                          pp_dt_.raw_data(), A.data.raw_data(),
                          backward_decay_terms.raw_data(),
                          pp_B_.raw_data(), pp_C_.raw_data(),
                          D.data.raw_data(), pp_state_hist_.raw_data(),
                          gx.raw_data(), gdt.raw_data(), gA.raw_data(),
                          gB.raw_data(), gC.raw_data(), gD.raw_data(),
                          deterministic_partial_B.raw_data(),
                          deterministic_partial_C.raw_data(),
                          deterministic_partial_dt.raw_data(),
                          deterministic_partial_A.raw_data(),
                          deterministic_partial_D.raw_data(),
                          deterministic_chunk_carry,
                          deterministic_chunk_scale,
                          deterministic_chunk_lane_A,
                          deterministic_chunk_lane_D, batch,
                          seq, n_heads, d_head, N, n_groups,
                          faithful_runtime_warp_size_,
                          pp_state_hist_state_major_, pp_state_hist_chunk_size_)
                    : cuda::launch_mamba2_faithful_backward_deterministic(
                gy.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
                A.data.raw_data(),
                backward_decay_terms.size > 0
                    ? backward_decay_terms.raw_data()
                    : nullptr,
                pp_B_.raw_data(), pp_C_.raw_data(),
                D.data.raw_data(), pp_state_hist_.raw_data(), gx.raw_data(),
                gdt.raw_data(), gA.raw_data(), gB.raw_data(), gC.raw_data(),
                gD.raw_data(), deterministic_partial_B.raw_data(),
                deterministic_partial_C.raw_data(),
                deterministic_partial_dt.raw_data(),
                deterministic_partial_A.raw_data(),
                deterministic_partial_D.raw_data(),
                deterministic_gh_history.size > 0
                    ? deterministic_gh_history.raw_data()
                    : nullptr,
                batch, seq, n_heads,
                d_head, N, n_groups, faithful_runtime_warp_size_,
                pp_state_hist_state_major_);
            if (!launch_enqueued) {
                throw std::runtime_error(
                    "Mamba2 deterministic faithful backward rejected "
                    "invalid arguments");
            }
            ++faithful_deterministic_backward_calls_;
            scan_done = true;
        } else if (!deterministic) {
            const bool launch_enqueued =
                cuda::launch_mamba2_faithful_backward(
                gy.raw_data(), pp_xc_.raw_data(), pp_dt_.raw_data(),
                A.data.raw_data(),
                backward_decay_terms.size > 0
                    ? backward_decay_terms.raw_data()
                    : nullptr,
                pp_B_.raw_data(), pp_C_.raw_data(),
                D.data.raw_data(), pp_state_hist_.raw_data(), gx.raw_data(),
                gdt.raw_data(), gA.raw_data(), gB.raw_data(), gC.raw_data(),
                gD.raw_data(), batch, seq, n_heads, d_head, N, n_groups,
                pp_state_hist_state_major_);

            if (!launch_enqueued) {
                throw std::runtime_error(
                    "Mamba2 faithful backward rejected invalid arguments or "
                    "failed to enqueue device initialization");
            }
            if (cuda::faithful_warp_aggregation_enabled() &&
                d_head % faithful_runtime_warp_size_ == 0) {
                ++faithful_warp_aggregated_backward_calls_;
            } else {
                ++faithful_scalar_atomic_backward_calls_;
            }
            scan_done = true;
        }
        if (scan_done) {
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Mamba2 faithful backward kernel failed: ") +
                    cudaGetErrorString(launch_status));
            }
            ++gpu_fast_path_hits_;
            ++faithful_backward_gpu_calls_;
        }
    }
#endif
    if (!scan_done) {
#ifdef USE_CUDA
        if (dev == Device::GPU) {
            ++gpu_fast_path_fallbacks_;
            ++faithful_backward_host_fallbacks_;
            last_fallback_reason_ =
                "faithful_backward_host_fallback";
        }
        if (dev == Device::GPU && strict_gpu_execution()) {
            throw std::runtime_error(
                "Strict GPU Mamba2 faithful backward requires the "
                "device reduction path; deterministic mode requires a "
                "warp-aligned head width");
        }
#endif
        auto host = [](const Tensor& value) {
            return value.get_device() == Device::GPU ? value.cpu() : value;
        };
        Tensor gyh = host(gy);
        Tensor xh = host(pp_xc_);
        Tensor dth = host(pp_dt_);
        Tensor Ah = host(A.data);
        Tensor Bh = host(pp_B_);
        Tensor Ch = host(pp_C_);
        Tensor Dh = host(D.data);
        Tensor hh = host(pp_state_hist_);
        Tensor gxh = Tensor::zeros({rows, d_inner}, Device::CPU);
        Tensor gdth = Tensor::zeros({rows, n_heads}, Device::CPU);
        Tensor gAh = Tensor::zeros({n_heads}, Device::CPU);
        Tensor gBh = Tensor::zeros({rows, GS}, Device::CPU);
        Tensor gCh = Tensor::zeros({rows, GS}, Device::CPU);
        Tensor gDh = Tensor::zeros({n_heads}, Device::CPU);
        const float* gyp = gyh.data();
        const float* xp = xh.data();
        const float* dtp = dth.data();
        const float* ap = Ah.data();
        const float* bp = Bh.data();
        const float* cp = Ch.data();
        const float* Dp = Dh.data();
        const float* hp = hh.data();
        float* gxp = gxh.data();
        float* gdtp = gdth.data();
        float* gAp = gAh.data();
        float* gBp = gBh.data();
        float* gCp = gCh.data();
        float* gDp = gDh.data();
        const size_t state_size = static_cast<size_t>(d_inner) * N;
        std::vector<float> carry(state_size, 0.0f);
        for (int b = 0; b < batch; ++b) {
            std::fill(carry.begin(), carry.end(), 0.0f);
            for (int t = seq - 1; t >= 0; --t) {
                const int row = b * seq + t;
                for (int h = 0; h < n_heads; ++h) {
                    const int group = (h * n_groups) / n_heads;
                    const float dt_raw = dtp[row * n_heads + h];
                    const MambaDecayTerms decay_terms =
                        mamba_decay_terms(dt_raw, ap[h]);
                    const float delta = decay_terms.delta;
                    const float decay = decay_terms.decay;
                    float ddecay = 0.0f;
                    float dinput_scale = 0.0f;
                    for (int p = 0; p < d_head; ++p) {
                        const int chan = h * d_head + p;
                        const size_t xi =
                            static_cast<size_t>(row) * d_inner + chan;
                        const float go_local = gyp[xi];
                        const float xv_local = xp[xi];
                        gxp[xi] += go_local * Dp[h];
                        gDp[h] += go_local * xv_local;
                        for (int n = 0; n < N; ++n) {
                            const size_t si =
                                static_cast<size_t>(chan) * N + n;
                            const size_t hi =
                                static_cast<size_t>(row) * state_size +
                                (pp_state_hist_state_major_
                                     ? static_cast<size_t>(n) * d_inner + chan
                                     : si);
                            const size_t bi =
                                static_cast<size_t>(row) * GS + group * N + n;
                            const float hprev =
                                t == 0
                                    ? 0.0f
                                    : hp[static_cast<size_t>(row - 1) *
                                             state_size +
                                         (pp_state_hist_state_major_
                                              ? static_cast<size_t>(n) *
                                                        d_inner +
                                                    chan
                                              : si)];
                            gCp[bi] += go_local * hp[hi];
                            const float gh =
                                go_local * cp[bi] + carry[si];
                            gBp[bi] += gh * delta * xv_local;
                            gxp[xi] += gh * delta * bp[bi];
                            dinput_scale += gh * bp[bi] * xv_local;
                            ddecay += gh * hprev;
                            carry[si] = gh * decay;
                        }
                    }
                    gdtp[row * n_heads + h] =
                        dinput_scale * decay_terms.delta_grad -
                        ddecay * decay_terms.decay_dt_factor;
                    gAp[h] -= ddecay * decay_terms.decay_alog_factor;
                }
            }
        }
        gx = dev == Device::GPU ? gxh.to(Device::GPU) : gxh;
        gdt = dev == Device::GPU ? gdth.to(Device::GPU) : gdth;
        gA = dev == Device::GPU ? gAh.to(Device::GPU) : gAh;
        gB = dev == Device::GPU ? gBh.to(Device::GPU) : gBh;
        gC = dev == Device::GPU ? gCh.to(Device::GPU) : gCh;
        gD = dev == Device::GPU ? gDh.to(Device::GPU) : gDh;
    }
    A.add_grad(gA);
    D.add_grad(gD);
    stage_timer.mark("scan");

    Tensor gxpre = Tensor::silu_backward(gx, pp_conv_pre_);
    Tensor gBpre = Tensor::silu_backward(gB, pp_B_conv_pre_);
    Tensor gCpre = Tensor::silu_backward(gC, pp_C_conv_pre_);
    Tensor gbias = Tensor::uninitialized({conv_dim}, dev);
    if (dev == Device::GPU) {
#ifdef USE_CUDA
        const bool bias_enqueued =
            cuda::launch_mamba2_faithful_bias_backward(
            gxpre.raw_data(), gBpre.raw_data(), gCpre.raw_data(),
            gbias.raw_data(), rows, d_inner, GS);
        if (!bias_enqueued) {
            throw std::runtime_error(
                "Mamba2 faithful bias backward rejected invalid arguments");
        }
        const cudaError_t bias_status = cudaGetLastError();
        if (bias_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "Mamba2 faithful bias backward kernel failed: ") +
                cudaGetErrorString(bias_status));
        }
#else
        throw std::logic_error(
            "GPU tensor reached a build without a GPU backend");
#endif
    } else {
        Tensor gxbh = gxpre.sum(0);
        Tensor gBbh = gBpre.sum(0);
        Tensor gCbh = gCpre.sum(0);
        std::copy_n(gxbh.data(), d_inner, gbias.data());
        std::copy_n(gBbh.data(), GS, gbias.data() + d_inner);
        std::copy_n(gCbh.data(), GS, gbias.data() + d_inner + GS);
    }
    conv_bias_.add_grad(gbias);
    stage_timer.mark("post_scan");

    // Each launcher clears its complete grad-input/grad-weight slice before
    // accumulating into it, so pre-zeroing these four device buffers would
    // duplicate bandwidth on every faithful backward pass.
    Tensor gxv = Tensor::uninitialized({rows, d_inner}, dev);
    Tensor gBv = Tensor::uninitialized({rows, GS}, dev);
    Tensor gCv = Tensor::uninitialized({rows, GS}, dev);
    Tensor gcw = Tensor::uninitialized({conv_dim, K}, dev);
    bool conv_done = false;
#ifdef USE_CUDA
    const bool deterministic_conv_gpu_eligible =
        determinism::deterministic_reductions_enabled() &&
        K <= cuda::kFaithfulReducedConvMaxKernel &&
        cuda::faithful_reduced_conv_enabled();
    if (dev == Device::GPU &&
        (!determinism::deterministic_reductions_enabled() ||
         deterministic_conv_gpu_eligible)) {
        const bool conv_enqueued =
            cuda::launch_mamba2_faithful_conv_backward(
            gxpre.raw_data(), gBpre.raw_data(), gCpre.raw_data(),
            pp_xv_.raw_data(), pp_Bv_.raw_data(), pp_Cv_.raw_data(),
            conv_weight_.data.raw_data(), gxv.raw_data(), gBv.raw_data(),
            gCv.raw_data(), gcw.raw_data(), batch, seq, d_inner, GS, K);
        if (!conv_enqueued) {
            throw std::runtime_error(
                "Mamba2 faithful conv backward rejected invalid arguments "
                "or failed to enqueue device initialization");
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "Mamba2 faithful conv backward kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        if (K <= cuda::kFaithfulReducedConvMaxKernel &&
            cuda::faithful_reduced_conv_enabled()) {
            ++faithful_reduced_conv_backward_calls_;
        } else {
            ++faithful_generic_atomic_conv_backward_calls_;
        }
        conv_done = true;
    }
#endif
    if (!conv_done) {
        auto host = [](const Tensor& value) {
            return value.get_device() == Device::GPU ? value.cpu() : value;
        };
        Tensor gxpreh = host(gxpre);
        Tensor gBpreh = host(gBpre);
        Tensor gCpreh = host(gCpre);
        Tensor xvh = host(pp_xv_);
        Tensor Bvh = host(pp_Bv_);
        Tensor Cvh = host(pp_Cv_);
        Tensor cwh = host(conv_weight_.data);
        Tensor gxvh = Tensor::zeros({rows, d_inner}, Device::CPU);
        Tensor gBvh = Tensor::zeros({rows, GS}, Device::CPU);
        Tensor gCvh = Tensor::zeros({rows, GS}, Device::CPU);
        Tensor gcwh = Tensor::zeros({conv_dim, K}, Device::CPU);
        conv1d_causal_backward(gxpreh.data(), xvh.data(), cwh.data(),
                               gxvh.data(), gcwh.data(), batch, seq, d_inner,
                               K);
        conv1d_causal_backward(
            gBpreh.data(), Bvh.data(),
            cwh.data() + static_cast<size_t>(d_inner) * K, gBvh.data(),
            gcwh.data() + static_cast<size_t>(d_inner) * K, batch, seq, GS,
            K);
        conv1d_causal_backward(
            gCpreh.data(), Cvh.data(),
            cwh.data() + static_cast<size_t>(d_inner + GS) * K, gCvh.data(),
            gcwh.data() + static_cast<size_t>(d_inner + GS) * K, batch, seq,
            GS, K);
        gxv = dev == Device::GPU ? gxvh.to(Device::GPU) : gxvh;
        gBv = dev == Device::GPU ? gBvh.to(Device::GPU) : gBvh;
        gCv = dev == Device::GPU ? gCvh.to(Device::GPU) : gCvh;
        gcw = dev == Device::GPU ? gcwh.to(Device::GPU) : gcwh;
    }
    conv_weight_.add_grad(gcw);
    stage_timer.mark("conv");

    Tensor gu;
    if (faithful_grouped_projection_backward_mode_ !=
        FaithfulGroupedProjectionMode::None) {
        gu = backward_faithful_grouped_projections(
            gxv, gz, gBv, gCv, gdt);
    } else {
        gu = x_proj_->backward(gxv).reshape({rows, d_model});
        gu = gu.add(B_proj_->backward(gBv).reshape({rows, d_model}));
        gu = gu.add(C_proj_->backward(gCv).reshape({rows, d_model}));
        gu = gu.add(dt_proj_->backward(gdt).reshape({rows, d_model}));
        gu = gu.add(z_proj_->backward(gz).reshape({rows, d_model}));
    }
    stage_timer.mark("projections");
    const bool rank_2 = grad_output.shape.size() == 2;
    Tensor input_gradient =
        rank_2 ? gu : gu.reshape({batch, seq, d_model});

    // A forward state is a one-shot backward ticket. Releasing it here keeps
    // checkpointed history residency to one layer during reverse traversal
    // instead of accumulating every recomputed history until the next step.
    faithful_checkpoint_input_ = Tensor();
    pp_u_ = Tensor();
    pp_xv_ = Tensor();
    pp_Bv_ = Tensor();
    pp_Cv_ = Tensor();
    pp_conv_pre_ = Tensor();
    pp_B_conv_pre_ = Tensor();
    pp_C_conv_pre_ = Tensor();
    pp_xc_ = Tensor();
    pp_B_ = Tensor();
    pp_C_ = Tensor();
    pp_dt_ = Tensor();
    pp_decay_terms_ = Tensor();
    pp_z_ = Tensor();
    pp_y_ssd_ = Tensor();
    pp_state_hist_ = Tensor();
    pp_state_hist_chunk_size_ = 0;
    pp_state_hist_state_major_ = false;
    pp_gated_input_ = Tensor();
    pp_gated_norm_ = Tensor();
    x_proj_->discard_backward_state();
    z_proj_->discard_backward_state();
    B_proj_->discard_backward_state();
    C_proj_->discard_backward_state();
    dt_proj_->discard_backward_state();
    out_proj.discard_backward_state();
    faithful_grouped_projection_backward_mode_ =
        FaithfulGroupedProjectionMode::None;
    proper_active_ = false;
    stage_timer.mark("release");
    return input_gradient;
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
    Tensor xc = conv_pre.silu();
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
    const int K = std::max(conv_kernel_, 1);
    const int taps = std::max(K - 1, 0);
    const Device dev = u.get_device();
    // Batched single-token decode: R sequences, each advancing its OWN carried
    // SSD state + conv ring (per-row slices of pp_stream_state_/pp_stream_ring_).
    // R == 1 is byte-identical to the original single-sequence step.
    const bool rank3 = u.shape.size() == 3;
    const int R = rank3 ? u.shape[0] : (u.shape.size() == 2 ? u.shape[0] : 1);
    Tensor u_flat = u.reshape({R, dim});

    // Projections (device-agnostic), pulled to host for the scalar recurrence.
    Tensor xv = x_proj_->forward(u_flat).reshape({R, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({R, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({R, dim});
    Tensor Ct = C_proj_->forward(u_flat).reshape({R, dim});
    Tensor dt = dt_proj_->forward(u_flat).reshape({R, dim});

#ifdef USE_CUDA
    // Device-resident state for every batch row; dispatch the fused recurrence
    // once per row without downloading or restoring state between tokens.
    if (R > 0 && dev == Device::GPU && mamba_gpu_step_enabled()) {
        if (pp_stream_dev_live_ &&
            (pp_stream_h_dev_.size != checked_int_product(R, dim, "proper batch state") ||
             pp_stream_ring_dev_.size != checked_int_product(R, std::max(taps * dim, 1), "proper batch ring"))) {
            throw std::invalid_argument("Mamba proper device state does not match the decode batch");
        }
        if (!pp_stream_dev_live_) {
            const int state_size = checked_int_product(R, dim, "proper batch state");
            Tensor h0(std::vector<int>{state_size}, Device::CPU);
            if (static_cast<int>(pp_stream_state_.size()) == state_size) {
                std::memcpy(h0.data(), pp_stream_state_.data(),
                            static_cast<size_t>(state_size) * sizeof(float));
            }
            pp_stream_h_dev_ = h0.to(Device::GPU);
            const int ringlen = checked_int_product(R, std::max(taps * dim, 1), "proper batch ring");
            Tensor r0(std::vector<int>{ringlen}, Device::CPU);
            std::memset(r0.data(), 0,
                        static_cast<size_t>(ringlen) * sizeof(float));
            if (taps > 0 &&
                static_cast<int>(pp_stream_ring_.size()) == R * taps * dim) {
                std::memcpy(r0.data(), pp_stream_ring_.data(),
                            static_cast<size_t>(R) * taps * dim * sizeof(float));
            }
            pp_stream_ring_dev_ = r0.to(Device::GPU);
            pp_stream_dev_live_ = true;
        }
        Tensor gated =
            Tensor::uninitialized(std::vector<int>{R, dim}, Device::GPU);
        for (int row = 0; row < R; ++row) {
            cuda::launch_mamba_proper_step(
                xv.raw_data() + row * dim, z.raw_data() + row * dim,
                Bt.raw_data() + row * dim, Ct.raw_data() + row * dim,
                dt.raw_data() + row * dim, A.data.raw_data(), conv_weight_.data.raw_data(),
                pp_stream_ring_dev_.raw_data() + row * std::max(taps * dim, 1),
                pp_stream_h_dev_.raw_data() + row * dim,
                gated.raw_data() + row * dim, dim, K);
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Mamba proper streaming kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        Tensor projected = out_proj.forward(gated).reshape({R, dim});
        Tensor skip = u_flat.mul(D.data);
        Tensor result = projected.add(skip);
        return u.shape.size() == 2 ? result : result.reshape({R, 1, dim});
    }
#endif

    if (dev == Device::GPU && strict_gpu_execution()) {
        throw std::runtime_error(
            "Strict GPU Mamba proper streaming step has no eligible "
            "device path; NSOS_MAMBA_GPU_STEP=0 cannot "
            "fall back to host execution");
    }

    // Host step (default): pull projections to host for the scalar recurrence.
    Tensor xv_h = dev == Device::GPU ? xv.cpu() : xv;
    Tensor Bt_h = dev == Device::GPU ? Bt.cpu() : Bt;
    Tensor Ct_h = dev == Device::GPU ? Ct.cpu() : Ct;
    Tensor dt_h = dev == Device::GPU ? dt.cpu() : dt;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    Tensor cw_h = conv_weight_.data.get_device() == Device::GPU
                      ? conv_weight_.data.cpu()
                      : conv_weight_.data;

    if (static_cast<int>(pp_stream_state_.size()) != R * dim) {
        pp_stream_state_.assign(static_cast<size_t>(R) * dim, 0.0f);
    }
    if (taps > 0 &&
        static_cast<int>(pp_stream_ring_.size()) != R * taps * dim) {
        pp_stream_ring_.assign(static_cast<size_t>(R) * taps * dim, 0.0f);
    }
    const float* xvp = xv_h.data();
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();
    const float* cwp = cw_h.data();
    Tensor y_ssd =
        Tensor::uninitialized(std::vector<int>{R, dim}, Device::CPU);
    float* yp = y_ssd.data();
    for (int r = 0; r < R; ++r) {
        const size_t base = static_cast<size_t>(r) * dim;
        float* state_r = pp_stream_state_.data() + base;
        float* ring_r = taps > 0 ? pp_stream_ring_.data() +
                                       static_cast<size_t>(r) * taps * dim
                                 : nullptr;
        for (int c = 0; c < dim; ++c) {
            // Causal depthwise conv over the K-tap window (ring taps + current x),
            // then SiLU — matches conv1d_causal_forward + silu on the full scan.
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                const float tapv = (j < taps) ? ring_r[static_cast<size_t>(j) * dim + c]
                                              : xvp[base + c];
                acc += cwp[static_cast<size_t>(c) * K + j] * tapv;
            }
            const float xc = acc * (1.0f / (1.0f + std::exp(-acc)));
            // A is a per-channel parameter [d_model], SHARED across batch rows —
            // index by channel only.  dt/B/C/x are [R,dim] so they use base+c,
            // but ap[base+c] (base=r*dim) read A out of bounds for any row r>0
            // (batched decode), corrupting rows>=1.  Mirrors the nstate step's
            // ap[h] (also shared across rows).
            const MambaDecayTerms decay_terms =
                mamba_decay_terms(dtp[base + c], ap[c]);
            const float dt_scale = decay_terms.delta;
            const float decay = decay_terms.decay;
            const float st =
                decay * state_r[c] + dt_scale * bp[base + c] * xc;
            state_r[c] = st;
            yp[base + c] = st * cp[base + c];
        }
        // Advance this row's conv ring: drop oldest tap, append current x.
        if (taps > 0) {
            for (int s = 0; s + 1 < taps; ++s) {
                for (int c = 0; c < dim; ++c) {
                    ring_r[static_cast<size_t>(s) * dim + c] =
                        ring_r[static_cast<size_t>(s + 1) * dim + c];
                }
            }
            for (int c = 0; c < dim; ++c) {
                ring_r[static_cast<size_t>(taps - 1) * dim + c] = xvp[base + c];
            }
        }
    }

    // Gate on host (parity), then project + skip on the input device.
    Tensor z_h = dev == Device::GPU ? z.cpu() : z;
    Tensor gated_h = y_ssd.mul(z_h.silu());
    Tensor gated = dev == Device::GPU ? gated_h.to(Device::GPU) : gated_h;
    Tensor projected = out_proj.forward(gated).reshape({R, dim});
    Tensor skip = u_flat.mul(D.data);
    Tensor result = projected.add(skip);
    return rank3 ? result.reshape({R, 1, dim})
                 : (u.shape.size() == 2 ? result : result.reshape({1, 1, dim}));
}

Tensor Mamba2SSD::forward_proper_nstate_step(const Tensor& u) {
    const int dim = d_model;
    const int H = std::max(n_heads, 1);
    const int P = d_head;
    const int N = std::max(d_state, 1);
    const int K = std::max(conv_kernel_, 1);
    const int taps = std::max(K - 1, 0);
    const Device dev = u.get_device();
    // Batched single-token decode (R sequences), each with its own H×P×N state +
    // conv ring slice.  R == 1 is byte-identical to the original single-seq step.
    const bool rank3 = u.shape.size() == 3;
    const int R = rank3 ? u.shape[0] : (u.shape.size() == 2 ? u.shape[0] : 1);
    Tensor u_flat = u.reshape({R, dim});

    Tensor xv = x_proj_->forward(u_flat).reshape({R, dim});
    Tensor z = z_proj_->forward(u_flat).reshape({R, dim});
    Tensor Bt = B_proj_->forward(u_flat).reshape({R, H * N});
    Tensor Ct = C_proj_->forward(u_flat).reshape({R, H * N});
    Tensor dt = dt_proj_->forward(u_flat).reshape({R, H});

#ifdef USE_CUDA
    // Fully on-device N-state step (GPU-first default; NSOS_MAMBA_GPU_STEP=0
    // opts out) — all batch rows. The H×P×N state + conv ring stay resident
    // on the GPU across tokens; one fused kernel does conv+N-state
    // recurrence+readout+gate.  Mirrors the host loop below 1:1
    // (test_gpu_parity_mamba_nstate_stream) and is CUDA-graph capturable.
    if (R > 0 && dev == Device::GPU && mamba_gpu_step_enabled()) {
        const size_t HPN_sz = static_cast<size_t>(H) * P * N;  // == dim * N
        const int HPN_i = checked_int_product(R, static_cast<int>(HPN_sz), "nstate batch state");
        if (pp_stream_dev_live_ &&
            (pp_stream_h_dev_.size != HPN_i ||
             pp_stream_ring_dev_.size != checked_int_product(R, std::max(taps * dim, 1), "nstate batch ring"))) {
            throw std::invalid_argument("Mamba N-state device state does not match the decode batch");
        }
        if (!pp_stream_dev_live_) {
            Tensor h0(std::vector<int>{HPN_i}, Device::CPU);
            std::memset(h0.data(), 0,
                        static_cast<size_t>(HPN_i) * sizeof(float));
            if (pp_stream_state_.size() == static_cast<size_t>(HPN_i)) {
                std::memcpy(h0.data(), pp_stream_state_.data(),
                            static_cast<size_t>(HPN_i) * sizeof(float));
            }
            pp_stream_h_dev_ = h0.to(Device::GPU);
            const int ringlen = checked_int_product(R, std::max(taps * dim, 1), "nstate batch ring");
            Tensor r0(std::vector<int>{ringlen}, Device::CPU);
            std::memset(r0.data(), 0,
                        static_cast<size_t>(ringlen) * sizeof(float));
            if (taps > 0 &&
                static_cast<int>(pp_stream_ring_.size()) == R * taps * dim) {
                std::memcpy(r0.data(), pp_stream_ring_.data(),
                            static_cast<size_t>(R) * taps * dim * sizeof(float));
            }
            pp_stream_ring_dev_ = r0.to(Device::GPU);
            pp_stream_dev_live_ = true;
        }
        Tensor gated =
            Tensor::uninitialized(std::vector<int>{R, dim}, Device::GPU);
        for (int row = 0; row < R; ++row) {
            cuda::launch_mamba_nstate_step(
                xv.raw_data() + row * dim, z.raw_data() + row * dim,
                Bt.raw_data() + row * H * N, Ct.raw_data() + row * H * N,
                dt.raw_data() + row * H, A.data.raw_data(), conv_weight_.data.raw_data(),
                pp_stream_ring_dev_.raw_data() + row * std::max(taps * dim, 1),
                pp_stream_h_dev_.raw_data() + row * HPN_sz,
                gated.raw_data() + row * dim, dim, K, P, N);
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Mamba N-state streaming kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        Tensor projected = out_proj.forward(gated).reshape({R, dim});
        Tensor skip = u_flat.mul(D.data);
        Tensor result = projected.add(skip);
        return rank3 ? result.reshape({R, 1, dim})
                     : (u.shape.size() == 2 ? result
                                            : result.reshape({1, 1, dim}));
    }
#endif

    if (dev == Device::GPU && strict_gpu_execution()) {
        throw std::runtime_error(
            "Strict GPU Mamba N-state streaming step has no eligible "
            "device path; NSOS_MAMBA_GPU_STEP=0 cannot "
            "fall back to host execution");
    }

    Tensor xv_h = dev == Device::GPU ? xv.cpu() : xv;
    Tensor Bt_h = dev == Device::GPU ? Bt.cpu() : Bt;
    Tensor Ct_h = dev == Device::GPU ? Ct.cpu() : Ct;
    Tensor dt_h = dev == Device::GPU ? dt.cpu() : dt;
    Tensor A_h = A.data.get_device() == Device::GPU ? A.data.cpu() : A.data;
    Tensor cw_h = conv_weight_.data.get_device() == Device::GPU
                      ? conv_weight_.data.cpu()
                      : conv_weight_.data;

    const size_t HPN = static_cast<size_t>(H) * P * N;
    if (pp_stream_state_.size() != static_cast<size_t>(R) * HPN) {
        pp_stream_state_.assign(static_cast<size_t>(R) * HPN, 0.0f);
    }
    if (taps > 0 &&
        static_cast<int>(pp_stream_ring_.size()) != R * taps * dim) {
        pp_stream_ring_.assign(static_cast<size_t>(R) * taps * dim, 0.0f);
    }
    const float* xvp = xv_h.data();
    const float* bp = Bt_h.data();
    const float* cp = Ct_h.data();
    const float* dtp = dt_h.data();
    const float* ap = A_h.data();   // per-head A_log [H], shared across rows
    const float* cwp = cw_h.data(); // conv weight [dim,K], shared across rows
    Tensor y_ssd =
        Tensor::uninitialized(std::vector<int>{R, dim}, Device::CPU);
    float* yp = y_ssd.data();
    std::vector<float> xc(static_cast<size_t>(dim), 0.0f);
    for (int r = 0; r < R; ++r) {
        const size_t xbase = static_cast<size_t>(r) * dim;
        const size_t bcbase = static_cast<size_t>(r) * H * N;
        const size_t dtbase = static_cast<size_t>(r) * H;
        float* state_r = pp_stream_state_.data() + static_cast<size_t>(r) * HPN;
        float* ring_r = taps > 0 ? pp_stream_ring_.data() +
                                       static_cast<size_t>(r) * taps * dim
                                 : nullptr;
        // Causal conv + SiLU per channel (shared conv with the diagonal path).
        for (int c = 0; c < dim; ++c) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                const float tapv = (j < taps) ? ring_r[static_cast<size_t>(j) * dim + c]
                                              : xvp[xbase + c];
                acc += cwp[static_cast<size_t>(c) * K + j] * tapv;
            }
            xc[static_cast<size_t>(c)] = acc * (1.0f / (1.0f + std::exp(-acc)));
        }
        for (int h = 0; h < H; ++h) {
            const MambaDecayTerms decay_terms =
                mamba_decay_terms(dtp[dtbase + h], ap[h]);
            const float dt_scale = decay_terms.delta;
            const float decay = decay_terms.decay;
            for (int p = 0; p < P; ++p) {
                const int chan = h * P + p;
                const float xcv = xc[static_cast<size_t>(chan)];
                float y_acc = 0.0f;
                for (int n = 0; n < N; ++n) {
                    const size_t sidx = (static_cast<size_t>(h) * P + p) * N + n;
                    const float bval = bp[bcbase + static_cast<size_t>(h) * N + n];
                    const float cval = cp[bcbase + static_cast<size_t>(h) * N + n];
                    const float hv =
                        decay * state_r[sidx] + dt_scale * bval * xcv;
                    state_r[sidx] = hv;
                    y_acc += hv * cval;
                }
                yp[xbase + chan] = y_acc;
            }
        }
        if (taps > 0) {
            for (int s = 0; s + 1 < taps; ++s) {
                for (int c = 0; c < dim; ++c) {
                    ring_r[static_cast<size_t>(s) * dim + c] =
                        ring_r[static_cast<size_t>(s + 1) * dim + c];
                }
            }
            for (int c = 0; c < dim; ++c) {
                ring_r[static_cast<size_t>(taps - 1) * dim + c] = xvp[xbase + c];
            }
        }
    }

    Tensor z_h = dev == Device::GPU ? z.cpu() : z;
    Tensor gated_h = y_ssd.mul(z_h.silu());
    Tensor gated = dev == Device::GPU ? gated_h.to(Device::GPU) : gated_h;
    Tensor projected = out_proj.forward(gated).reshape({R, dim});
    Tensor skip = u_flat.mul(D.data);
    Tensor result = projected.add(skip);
    return rank3 ? result.reshape({R, 1, dim})
                 : (u.shape.size() == 2 ? result : result.reshape({1, 1, dim}));
}

Tensor Mamba2SSD::forward_faithful_step(const Tensor& u) {
    const bool rank3 = u.shape.size() == 3;
    const int R = rank3 ? u.shape[0] : 1;
    const int N = d_state;
    const int GS = n_groups * N;
    const int K = conv_kernel_;
    const int taps = std::max(K - 1, 0);
    const Device dev = u.get_device();
    Tensor u_flat = u.reshape({R, d_model});
    auto grouped = !training_mode_ && R == 1
        ? decode_projections_.forward(u_flat, {x_proj_.get(), z_proj_.get(), B_proj_.get(), C_proj_.get(), dt_proj_.get()})
        : std::vector<Tensor>{};
    Tensor xv = (grouped.empty() ? x_proj_->forward(u_flat) : grouped[0]).reshape({R, d_inner});
    Tensor z = (grouped.empty() ? z_proj_->forward(u_flat) : grouped[1]).reshape({R, d_inner});
    Tensor Bv = (grouped.empty() ? B_proj_->forward(u_flat) : grouped[2]).reshape({R, GS});
    Tensor Cv = (grouped.empty() ? C_proj_->forward(u_flat) : grouped[3]).reshape({R, GS});
    Tensor dt = (grouped.empty() ? dt_proj_->forward(u_flat) : grouped[4]).reshape({R, n_heads});

#ifdef USE_CUDA
    if (R > 0 && dev == Device::GPU && mamba_gpu_step_enabled() &&
        N <= cuda::mamba_nstate_max_n()) {
        const int row_state_size = checked_int_product(d_inner, N, "faithful decode row state");
        const int row_ring_size = std::max(taps * conv_dim, 1);
        const int state_size = checked_int_product(R, row_state_size, "faithful decode batch state");
        const int ring_size = checked_int_product(R, row_ring_size, "faithful decode batch ring");
        if (!pp_stream_dev_live_ ||
            pp_stream_h_dev_.size != state_size ||
            pp_stream_ring_dev_.size != ring_size) {
            Tensor hs({state_size}, Device::CPU);
            if (static_cast<int>(pp_stream_state_.size()) == state_size) {
                std::copy(pp_stream_state_.begin(), pp_stream_state_.end(),
                          hs.data());
            }
            Tensor ring({ring_size}, Device::CPU);
            if (taps > 0 &&
                static_cast<int>(pp_stream_ring_.size()) ==
                    R * taps * conv_dim) {
                std::copy(pp_stream_ring_.begin(), pp_stream_ring_.end(),
                          ring.data());
            }
            pp_stream_h_dev_ = hs.to(Device::GPU);
            pp_stream_ring_dev_ = ring.to(Device::GPU);
            pp_stream_dev_live_ = true;
        }
        Tensor xBC =
            Tensor::uninitialized({R, conv_dim}, Device::GPU);
        Tensor y =
            Tensor::uninitialized({R, d_inner}, Device::GPU);
        cuda::launch_mamba2_faithful_conv_step(
            xv.raw_data(), Bv.raw_data(), Cv.raw_data(),
            conv_weight_.data.raw_data(), conv_bias_.data.raw_data(),
            pp_stream_ring_dev_.raw_data(), xBC.raw_data(), d_inner, GS, K, R);
        cuda::launch_mamba2_faithful_step(
            xBC.raw_data(), dt.raw_data(), A.data.raw_data(), D.data.raw_data(),
            pp_stream_h_dev_.raw_data(), y.raw_data(), n_heads, d_head, N, n_groups, R);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Faithful Mamba-2 streaming kernels failed: ") +
                cudaGetErrorString(launch_status));
        }
        Tensor normalized;
        const char* fused = std::getenv("NSOS_GPU_MAMBA_FUSED_EPILOGUE");
        if (fused && fused[0] == '0') {
            normalized = y.mul(z.silu()).rmsnorm(config_.rms_norm_eps).mul(norm_weight_.data);
        } else {
            normalized = Tensor::uninitialized({R, d_inner}, Device::GPU);
            cuda::launch_mamba_gated_rmsnorm(y.raw_data(), z.raw_data(), norm_weight_.data.raw_data(),
                normalized.raw_data(), R, d_inner, config_.rms_norm_eps);
            gpu::record_dispatch(gpu::DispatchPath::MambaEpilogue);
        }
        Tensor result = out_proj.forward(normalized).reshape({R, d_model});
        ++gpu_fast_path_hits_;
        ++faithful_streaming_gpu_calls_;
        return rank3 ? result.reshape({R, 1, d_model}) : result;
    }
#endif

    if (dev == Device::GPU) {
        ++gpu_fast_path_fallbacks_;
        ++faithful_streaming_host_fallbacks_;
        last_fallback_reason_ =
            "faithful_streaming_host_fallback";
    }
    if (dev == Device::GPU && strict_gpu_execution()) {
        throw std::runtime_error(
            "Strict GPU faithful Mamba-2 streaming step has no eligible "
            "device path; unsupported state width and "
            "NSOS_MAMBA_GPU_STEP=0 cannot fall back to host execution");
    }

    auto host = [](const Tensor& value) {
        return value.get_device() == Device::GPU ? value.cpu() : value;
    };
    Tensor xvh = host(xv);
    Tensor zh = host(z);
    Tensor Bvh = host(Bv);
    Tensor Cvh = host(Cv);
    Tensor dth = host(dt);
    Tensor Ah = host(A.data);
    Tensor Dh = host(D.data);
    Tensor cwh = host(conv_weight_.data);
    Tensor cbh = host(conv_bias_.data);
    const size_t one_state = static_cast<size_t>(d_inner) * N;
    if (pp_stream_state_.size() != static_cast<size_t>(R) * one_state) {
        pp_stream_state_.assign(static_cast<size_t>(R) * one_state, 0.0f);
    }
    if (pp_stream_ring_.size() !=
        static_cast<size_t>(R) * taps * conv_dim) {
        pp_stream_ring_.assign(
            static_cast<size_t>(R) * taps * conv_dim, 0.0f);
    }
    Tensor yh =
        Tensor::uninitialized({R, d_inner}, Device::CPU);
    std::vector<float> xBC(static_cast<size_t>(conv_dim), 0.0f);
    for (int r = 0; r < R; ++r) {
        float* state =
            pp_stream_state_.data() + static_cast<size_t>(r) * one_state;
        float* ring =
            taps == 0
                ? nullptr
                : pp_stream_ring_.data() +
                      static_cast<size_t>(r) * taps * conv_dim;
        const float* xrow =
            xvh.data() + static_cast<size_t>(r) * d_inner;
        const float* Brow = Bvh.data() + static_cast<size_t>(r) * GS;
        const float* Crow = Cvh.data() + static_cast<size_t>(r) * GS;
        for (int c = 0; c < conv_dim; ++c) {
            const float current =
                c < d_inner
                    ? xrow[c]
                    : (c < d_inner + GS ? Brow[c - d_inner]
                                        : Crow[c - d_inner - GS]);
            float value = cbh.data()[c];
            for (int j = 0; j < K; ++j) {
                value += cwh.data()[static_cast<size_t>(c) * K + j] *
                         (j < taps
                              ? ring[static_cast<size_t>(j) * conv_dim + c]
                              : current);
            }
            xBC[static_cast<size_t>(c)] = silu_stable(value);
            if (taps > 0) {
                for (int s = 0; s + 1 < taps; ++s) {
                    ring[static_cast<size_t>(s) * conv_dim + c] =
                        ring[static_cast<size_t>(s + 1) * conv_dim + c];
                }
                ring[static_cast<size_t>(taps - 1) * conv_dim + c] = current;
            }
        }
        for (int h = 0; h < n_heads; ++h) {
            const int group = (h * n_groups) / n_heads;
            const MambaDecayTerms decay_terms = mamba_decay_terms(
                dth.data()[r * n_heads + h], Ah.data()[h]);
            const float delta = decay_terms.delta;
            const float decay = decay_terms.decay;
            for (int p = 0; p < d_head; ++p) {
                const int chan = h * d_head + p;
                const float input = xBC[static_cast<size_t>(chan)];
                float out = Dh.data()[h] * input;
                for (int n = 0; n < N; ++n) {
                    const size_t si = static_cast<size_t>(chan) * N + n;
                    const float Bvalue =
                        xBC[static_cast<size_t>(d_inner + group * N + n)];
                    const float Cvalue =
                        xBC[static_cast<size_t>(d_inner + GS + group * N + n)];
                    state[si] =
                        decay * state[si] + delta * Bvalue * input;
                    out += state[si] * Cvalue;
                }
                yh.data()[static_cast<size_t>(r) * d_inner + chan] = out;
            }
        }
    }
    Tensor y = dev == Device::GPU ? yh.to(Device::GPU) : yh;
    Tensor gated = y.mul(z.silu());
    Tensor normalized =
        gated.rmsnorm(config_.rms_norm_eps).mul(norm_weight_.data);
    Tensor result = out_proj.forward(normalized).reshape({R, d_model});
    return rank3 ? result.reshape({R, 1, d_model}) : result;
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
    if (config_.faithful_mamba2) {
        (void)ctx;
        const bool token_step =
            (u.shape.size() == 3 && u.shape[1] == 1) ||
            (u.shape.size() == 2 && u.shape[0] == 1);
        if (streaming_inference_ && token_step && pp_stream_active_) {
            return forward_faithful_step(u);
        }
        return forward_faithful(u);
    }
    if (config_.proper_selective_ssm) {
        (void)ctx;
        // Incremental single-token decode: once the prefill has primed the stream
        // (pp_stream_active_), each subsequent single token advances the carried
        // SSD state + conv window in O(1) instead of re-scanning the prefix.
        // A single-token-per-sequence decode step: rank-3 [B,1,D] (B sequences,
        // one new token each — incl. B>1 batched decode) or rank-2 [1,D] (one
        // sequence).  rank-2 [S,D] with S>1 is a prefill (multi-token, single
        // sequence), NOT a step.
        const bool token_step =
            (u.shape.size() == 3 && u.shape[1] == 1) ||
            (u.shape.size() == 2 && u.shape[0] == 1);
        if (streaming_inference_ && token_step && pp_stream_active_) {
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

        Tensor x_proj = in_proj_robust->forward(input);
        Tensor gate = in_proj_sensitive->forward(input);
        Tensor delta = gate.sigmoid();
        Tensor selective_b = x_proj.sigmoid();
        Tensor selective_c = gate.sigmoid();
        Tensor x_scaled = x_proj.mul(selective_b);
        Tensor y_ssd =
            Tensor::uninitialized(state_shape, input.get_device());

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
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Mamba single-token update kernel failed: ") +
                    cudaGetErrorString(launch_status));
            }
            y_ssd = y_ssd.mul(selective_c.reshape(state_shape));
        } else
#endif
        {
            ++gpu_fast_path_fallbacks_;
#ifdef USE_CUDA
            last_fallback_reason_ = "selective_bc_cpu_fallback";
            if (input.get_device() == Device::GPU &&
                strict_gpu_execution()) {
                throw std::runtime_error(
                    "Strict GPU Mamba single-token execution has no "
                    "eligible device kernel");
            }
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
                    const MambaDecayTerms decay_terms =
                        mamba_decay_terms(delta_ptr[index], a_ptr[d]);
                    const float dt_scale = decay_terms.delta;
                    const float decay = decay_terms.decay;
                    const float next_state =
                        state_ptr[index] * decay +
                        dt_scale * b_ptr[index] * x_ptr[index];
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
        // y_ssd already contains the selective C readout.  Applying C again
        // here was the historical double-C bug.
        Tensor projected = out_proj.forward(y_ssd);
        Tensor skip = input.mul(D.data);
        Tensor result = projected.add(skip);
        if (single_token_rank1) {
            return result.reshape({d_model});
        }
        return result;
    }

    saved_input_ = u;
    saved_x_proj_ = in_proj_robust->forward(u);
    saved_gate_ = in_proj_sensitive->forward(u);

    saved_delta_ = saved_gate_.sigmoid();
    saved_B_ = saved_x_proj_.sigmoid();
    saved_C_ = saved_gate_.sigmoid();
    Tensor y_ssd =
        ssd_forward(saved_x_proj_, saved_delta_, A.data, saved_B_, saved_C_, ctx, true);
    saved_ssd_ = y_ssd;
    update_streaming_state_from_history(u);
    Tensor projected = out_proj.forward(y_ssd);
    Tensor skip = saved_input_.mul(D.data);
    return projected.add(skip);
}

Tensor Mamba2SSD::backward(const Tensor& grad_output, Context& ctx) {
    (void)ctx;
    if (config_.faithful_mamba2) {
        if (!proper_active_) {
            throw std::runtime_error(
                "Mamba2SSD faithful backward called before forward");
        }
        return backward_faithful(grad_output);
    }
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
    Tensor grad_y_ssd = grad_gated;
    Tensor grad_gate_raw =
        Tensor::zeros(saved_gate_.shape.dims, saved_gate_.get_device());

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

    Tensor grad_from_xproj =
        in_proj_robust->backward(grad_x_proj_ssd.add(grad_b_raw));
    Tensor grad_from_gate =
        in_proj_sensitive->backward(grad_gate_raw.add(grad_c_raw));
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
    // A failed/cancelled faithful training attempt reaches reset() through the
    // Trainer rollback guard before backward can consume this recompute input.
    // Releasing it here prevents exception paths from retaining a full input
    // activation until the next successful forward overwrites the handle.
    faithful_checkpoint_input_ = Tensor();
    faithful_recompute_active_ = false;
    streaming_state_.reset();
    // Proper-path forward caches.
    proper_active_ = false;
    pp_u_ = Tensor();
    pp_xv_ = Tensor();
    pp_Bv_ = Tensor();
    pp_Cv_ = Tensor();
    pp_conv_pre_ = Tensor();
    pp_B_conv_pre_ = Tensor();
    pp_C_conv_pre_ = Tensor();
    pp_xc_ = Tensor();
    pp_z_ = Tensor();
    pp_B_ = Tensor();
    pp_C_ = Tensor();
    pp_dt_ = Tensor();
    pp_decay_terms_ = Tensor();
    pp_h_hist_ = Tensor();
    pp_y_ssd_ = Tensor();
    pp_gated_input_ = Tensor();
    pp_gated_norm_ = Tensor();
    pp_state_hist_ = Tensor();
    pp_state_hist_chunk_size_ = 0;
    pp_state_hist_state_major_ = false;
    faithful_grouped_projection_backward_mode_ =
        FaithfulGroupedProjectionMode::None;
    // Proper-path incremental decode cache.
    pp_stream_state_.clear();
    pp_stream_ring_.clear();
    pp_stream_active_ = false;
    pp_stream_h_dev_ = Tensor();
    pp_stream_ring_dev_ = Tensor();
    pp_stream_dev_live_ = false;
}

void Mamba2SSD::reset_runtime_telemetry() {
    gpu_fast_path_hits_ = 0;
    gpu_fast_path_fallbacks_ = 0;
    last_fallback_reason_.clear();
    faithful_forward_gpu_calls_ = 0;
    faithful_forward_host_fallbacks_ = 0;
    faithful_backward_gpu_calls_ = 0;
    faithful_backward_host_fallbacks_ = 0;
    faithful_streaming_gpu_calls_ = 0;
    faithful_streaming_host_fallbacks_ = 0;
    stream_priming_gpu_calls_ = 0;
    stream_priming_host_fallbacks_ = 0;
    faithful_recompute_forwards_ = 0;
    faithful_selective_history_recomputes_ = 0;
    faithful_full_block_recompute_forwards_ = 0;
    faithful_warp_aggregated_backward_calls_ = 0;
    faithful_deterministic_backward_calls_ = 0;
    faithful_scalar_atomic_backward_calls_ = 0;
    faithful_reduced_conv_backward_calls_ = 0;
    faithful_generic_atomic_conv_backward_calls_ = 0;
    faithful_peak_state_history_bytes_ = 0;
    faithful_grouped_projection_forward_calls_ = 0;
    faithful_grouped_projection_backward_calls_ = 0;
    faithful_grouped_projection_cache_rebuilds_ = 0;
    faithful_grouped_projection_full_forward_calls_ = 0;
    faithful_grouped_projection_sensitive_forward_calls_ = 0;
    faithful_grouped_projection_full_backward_calls_ = 0;
    faithful_grouped_projection_sensitive_backward_calls_ = 0;
}

void Mamba2SSD::to(Device dev) {
    auto move_parameter = [dev](Parameter& parameter) {
        if (parameter.data.size > 0 &&
            parameter.data.get_device() != dev) {
            parameter.data = parameter.data.to(dev);
        }
        if (parameter.grad.size > 0 &&
            parameter.grad.get_device() != dev) {
            parameter.grad = parameter.grad.to(dev);
        }
    };
    if (config_.proper_selective_ssm || config_.faithful_mamba2) {
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
        move_parameter(conv_weight_);
        move_parameter(conv_bias_);
        move_parameter(norm_weight_);
        move_parameter(A);
        move_parameter(D);
        faithful_grouped_projection_weight_ = Tensor();
        faithful_grouped_projection_versions_.fill(0);
        faithful_grouped_projection_backward_mode_ =
            FaithfulGroupedProjectionMode::None;
        faithful_runtime_warp_size_ = 0;
        pp_decay_terms_ = Tensor();
        pp_stream_h_dev_ = Tensor();
        pp_stream_ring_dev_ = Tensor();
        pp_stream_dev_live_ = false;
        return;
    }
    in_proj_robust->to(dev);
    in_proj_sensitive->to(dev);
    out_proj.to(dev);
    move_parameter(A);
    move_parameter(D);
}

std::vector<Parameter*> Mamba2SSD::parameters() {
    std::vector<Parameter*> params;
    auto add_proj = [&](BitLinear& bl, const char* prefix) {
        auto p = bl.parameters();
        prefix_parameter_names(p, prefix);
        params.insert(params.end(), p.begin(), p.end());
    };
    if (config_.proper_selective_ssm || config_.faithful_mamba2) {
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
        conv_weight_.assign_relative_name("conv1d_weight");
        params.push_back(&conv_weight_);
        if (config_.faithful_mamba2) {
            conv_bias_.assign_relative_name("conv1d_bias");
            params.push_back(&conv_bias_);
            norm_weight_.assign_relative_name("norm.weight");
            params.push_back(&norm_weight_);
        }
    } else {
        add_proj(*in_proj_robust, "in_proj_robust.");
        add_proj(*in_proj_sensitive, "in_proj_sensitive.");
        add_proj(out_proj, "out_proj.");
    }
    A.assign_relative_name("A");
    params.push_back(&A);
    D.assign_relative_name("D");
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
    pp_stream_h_dev_ = Tensor();
    pp_stream_ring_dev_ = Tensor();
    pp_stream_dev_live_ = false;
    if (!enabled) {
        streaming_state_.reset();
    } else if (!streaming_state_ || streaming_state_->shape.back() != d_model) {
        streaming_state_ =
            std::make_shared<Tensor>(Tensor::zeros({1, d_model}, A.data.get_device()));
    }
}

MambaStreamSnapshot Mamba2SSD::snapshot_streaming_state(bool device_resident) const {
    MambaStreamSnapshot snapshot;
    snapshot.enabled = streaming_inference_;
    snapshot.state = streaming_state_;
    if (device_resident && pp_stream_dev_live_ && pp_stream_h_dev_.size > 0) {
        snapshot.proper_state_device = pp_stream_h_dev_.clone();
        snapshot.proper_ring_device = pp_stream_ring_dev_.clone();
        snapshot.proper_active = pp_stream_active_;
        return snapshot;
    }
    // Proper-path incremental decode state: carry per-sequence so fork/restore
    // does not bleed the SSD state + conv window across sequences.
    // If the device buffers hold the live state (GPU step path after >=1 token),
    // the host vectors are stale -> download the current device state instead.
    if (pp_stream_dev_live_ && pp_stream_h_dev_.size > 0) {
#ifdef USE_CUDA
        Tensor hc = pp_stream_h_dev_.cpu();
        snapshot.proper_state.assign(hc.data(), hc.data() + hc.size);
        if (pp_stream_ring_dev_.size > 0) {
            Tensor rc = pp_stream_ring_dev_.cpu();
            snapshot.proper_ring.assign(rc.data(), rc.data() + rc.size);
        } else {
            snapshot.proper_ring = pp_stream_ring_;
        }
#else
        snapshot.proper_state = pp_stream_state_;
        snapshot.proper_ring = pp_stream_ring_;
#endif
    } else {
        snapshot.proper_state = pp_stream_state_;
        snapshot.proper_ring = pp_stream_ring_;
    }
    snapshot.proper_active = pp_stream_active_;
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
    pp_stream_state_ = snapshot.proper_state;
    pp_stream_ring_ = snapshot.proper_ring;
    pp_stream_active_ = snapshot.proper_active;
    // Force the device buffers to re-sync from the restored host vectors on the
    // next GPU step (the previous device state belonged to another sequence).
    pp_stream_dev_live_ = false;
    if (snapshot.proper_state_device.size > 0) {
        if (A.data.get_device() == Device::GPU) {
            pp_stream_h_dev_ = snapshot.proper_state_device.to(Device::GPU).clone();
            pp_stream_ring_dev_ = snapshot.proper_ring_device.to(Device::GPU).clone();
            pp_stream_dev_live_ = true;
        } else {
            Tensor state = snapshot.proper_state_device.cpu();
            Tensor ring = snapshot.proper_ring_device.cpu();
            pp_stream_state_.assign(state.data(), state.data() + state.size);
            pp_stream_ring_.assign(ring.data(), ring.data() + ring.size);
        }
    }
}

std::vector<MambaStreamSnapshot> Mamba2SSD::snapshot_streaming_state_batch(bool device_resident) const {
    std::vector<MambaStreamSnapshot> snapshots;
    // ── Proper path: split the per-row SSD state + conv ring back into one
    // snapshot per sequence (the batched decode keeps R sequences contiguous in
    // pp_stream_state_/pp_stream_ring_). ──
    if (config_.proper_selective_ssm || config_.faithful_mamba2) {
        const int H = std::max(n_heads, 1);
        const int P = d_head;
        const int N = std::max(d_state, 1);
        const size_t statelen = (config_.proper_state_expansion ||
                                 config_.faithful_mamba2)
                                    ? static_cast<size_t>(H) * P * N
                                    : static_cast<size_t>(d_model);
        const int taps = std::max(conv_kernel_ - 1, 0);
        const size_t ringlen =
            static_cast<size_t>(taps) *
            (config_.faithful_mamba2 ? conv_dim : d_model);
        if (device_resident && pp_stream_dev_live_ && pp_stream_h_dev_.size > 0) {
            const int rows = static_cast<int>(pp_stream_h_dev_.size / statelen);
            snapshots.resize(static_cast<size_t>(rows));
            for (int row = 0; row < rows; ++row) {
                auto& s = snapshots[static_cast<size_t>(row)];
                s.enabled = streaming_inference_;
                s.proper_active = pp_stream_active_;
                s.proper_state_device = pp_stream_h_dev_.reshape({rows, static_cast<int>(statelen)})
                    .slice(0, row, row + 1).reshape({static_cast<int>(statelen)});
                const int ring_stride = static_cast<int>(std::max<size_t>(ringlen, 1));
                s.proper_ring_device = pp_stream_ring_dev_.reshape({rows, ring_stride})
                    .slice(0, row, row + 1).reshape({ring_stride});
            }
            return snapshots;
        }
        std::vector<float> st = pp_stream_state_;
        std::vector<float> rg = pp_stream_ring_;
#ifdef USE_CUDA
        if (pp_stream_dev_live_ && pp_stream_h_dev_.size > 0) {
            Tensor hc = pp_stream_h_dev_.cpu();
            st.assign(hc.data(), hc.data() + hc.size);
            if (pp_stream_ring_dev_.size > 0) {
                Tensor rc = pp_stream_ring_dev_.cpu();
                rg.assign(rc.data(), rc.data() + rc.size);
            }
        }
#endif
        const int rows = (statelen > 0) ? static_cast<int>(st.size() / statelen) : 0;
        snapshots.resize(static_cast<size_t>(std::max(rows, 0)));
        for (int r = 0; r < rows; ++r) {
            auto& s = snapshots[static_cast<size_t>(r)];
            s.enabled = streaming_inference_;
            s.proper_active = pp_stream_active_;
            s.proper_state.assign(st.begin() + static_cast<size_t>(r) * statelen,
                                  st.begin() + static_cast<size_t>(r + 1) * statelen);
            if (ringlen > 0 && rg.size() >= static_cast<size_t>(r + 1) * ringlen) {
                s.proper_ring.assign(rg.begin() + static_cast<size_t>(r) * ringlen,
                                     rg.begin() + static_cast<size_t>(r + 1) * ringlen);
            }
        }
        return snapshots;
    }

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

    // ── Proper path: concatenate each sequence's SSD state + conv ring into the
    // contiguous per-row buffers the batched decode step reads. ──
    if (config_.proper_selective_ssm || config_.faithful_mamba2) {
        streaming_inference_ = snapshots.front().enabled;
        const bool device_snapshots = std::all_of(snapshots.begin(), snapshots.end(),
            [](const auto& s) { return s.proper_state_device.size > 0; });
        if (device_snapshots && A.data.get_device() == Device::GPU) {
            if (snapshots.size() > size_t(std::numeric_limits<int>::max()) ||
                snapshots.front().proper_state_device.size <= 0 ||
                snapshots.front().proper_state_device.size > std::numeric_limits<int>::max() ||
                snapshots.front().proper_ring_device.size <= 0 ||
                snapshots.front().proper_ring_device.size > std::numeric_limits<int>::max())
                throw std::overflow_error("Mamba device snapshot size exceeds supported range");
            const int rows = static_cast<int>(snapshots.size());
            const int state_width = static_cast<int>(snapshots.front().proper_state_device.size);
            const int ring_width = static_cast<int>(snapshots.front().proper_ring_device.size);
            const int state_elements = checked_int_product(rows, state_width, "Mamba restored batch state");
            const int ring_elements = checked_int_product(rows, ring_width, "Mamba restored batch ring");
            Tensor states = Tensor::uninitialized({rows, state_width}, Device::GPU);
            Tensor rings = Tensor::uninitialized({rows, ring_width}, Device::GPU);
            for (int row = 0; row < rows; ++row) {
                const auto& s = snapshots[static_cast<size_t>(row)];
                if (s.proper_state_device.size != state_width || s.proper_ring_device.size != ring_width) {
                    throw std::invalid_argument("Mamba device snapshot geometry differs between rows");
                }
                copy_float_bytes_device_safe(states.raw_data() + row * state_width, Device::GPU,
                    s.proper_state_device.raw_data(), s.proper_state_device.get_device(),
                    static_cast<size_t>(state_width) * sizeof(float));
                copy_float_bytes_device_safe(rings.raw_data() + row * ring_width, Device::GPU,
                    s.proper_ring_device.raw_data(), s.proper_ring_device.get_device(),
                    static_cast<size_t>(ring_width) * sizeof(float));
            }
            pp_stream_h_dev_ = states.reshape({state_elements});
            pp_stream_ring_dev_ = rings.reshape({ring_elements});
            pp_stream_dev_live_ = true;
            pp_stream_active_ = std::any_of(snapshots.begin(), snapshots.end(),
                [](const auto& s) { return s.proper_active; });
            pp_stream_state_.clear();
            pp_stream_ring_.clear();
            return;
        }
        pp_stream_state_.clear();
        pp_stream_ring_.clear();
        bool any_active = false;
        for (const auto& s : snapshots) {
            if (s.proper_state_device.size > 0) {
                Tensor state = s.proper_state_device.cpu();
                Tensor ring = s.proper_ring_device.cpu();
                pp_stream_state_.insert(pp_stream_state_.end(), state.data(), state.data() + state.size);
                pp_stream_ring_.insert(pp_stream_ring_.end(), ring.data(), ring.data() + ring.size);
            } else {
                pp_stream_state_.insert(pp_stream_state_.end(), s.proper_state.begin(), s.proper_state.end());
                pp_stream_ring_.insert(pp_stream_ring_.end(), s.proper_ring.begin(), s.proper_ring.end());
            }
            any_active = any_active || s.proper_active;
        }
        pp_stream_active_ = any_active;
        pp_stream_dev_live_ = false;  // re-sync device carry on next GPU step
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
    if (config_.proper_selective_ssm || config_.faithful_mamba2) {
        out.push_back(x_proj_.get());
        out.push_back(z_proj_.get());
        out.push_back(B_proj_.get());
        out.push_back(C_proj_.get());
        out.push_back(dt_proj_.get());
        out.push_back(&out_proj);
        return;
    }
    out.push_back(in_proj_robust.get());
    out.push_back(in_proj_sensitive.get());
    out.push_back(&out_proj);
}

} // namespace nsos
