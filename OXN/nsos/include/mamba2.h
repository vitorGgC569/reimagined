#ifndef MAMBA2_H
#define MAMBA2_H

#include "autograd.h"
#include "bitlinear.h"
#include "tensor.h"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace nsos {

// Configuration for checkpointing and recomputation tradeoff
struct MambaConfig {
  bool recompute_ssd = true;      // true: recompute forward during backward
  bool save_intermediates = true; // save z, x, dt, B, C to context
  int max_seq_for_storage = 2048; // seq_len limit for history storage

  // ── Proper selective SSM (Mamba diagonal form) — OPT-IN ──────────────────
  // The DEFAULT (false) path is the historical one: a per-channel scalar gated
  // recurrence whose dt, C and output gate all collapse onto a SINGLE sensitive
  // projection (delta == C, and C is applied twice), with no input convolution.
  // That path is preserved byte-for-byte so existing checkpoints keep loading.
  //
  // When true, the layer instead runs the corrected selective recurrence with
  // INDEPENDENT projections for x, the gate z, B, C and dt, a short causal
  // depthwise conv1d on x (Mamba's local token mixing), a LINEAR readout
  // y = h * C (C applied exactly once), and a separate SiLU gate y * silu(z):
  //   xc_t   = silu(conv1d_causal(x_proj(u)))
  //   h_t    = exp(-softplus(dt_t) * A) * h_{t-1} + B_t * xc_t
  //   y_t    = h_t * C_t
  //   out    = out_proj(y * silu(z)) + u * D
  // This removes the delta==C / double-C degeneracy and gives the time axis its
  // own selectivity.  By itself the state is diagonal (per-channel scalar); set
  // proper_state_expansion below for the full N-dimensional SSD state.  All
  // gradients are hand-derived and covered by tests/test_gradcheck.cpp.
  bool proper_selective_ssm = false;
  int conv_kernel = 4; // causal depthwise conv width for the proper path

  // ── N-dimensional state expansion (full Mamba-2 SSD) — OPT-IN ──────────────
  // Requires proper_selective_ssm.  The diagonal proper path above keeps a
  // per-channel SCALAR state; this expands it to the true SSD state h ∈ R^{H×P×N}
  // (H heads, P=d_head channels/head, N=d_state).  B and C become per-head
  // N-dimensional, dt and A become per-head:
  //   decay_{t,h} = exp(-softplus(dt_{t,h}) * A_h)
  //   h_{t,h,p,n} = decay_{t,h} * h_{t-1,h,p,n} + B_{t,h,n} * xc_{t,h,p}
  //   y_{t,h,p}   = Σ_n h_{t,h,p,n} * C_{t,h,n}
  // This is the state-space duality form (Gu & Dao 2024).  Default false keeps
  // the diagonal proper path (and its parameter set) unchanged.  All gradients
  // hand-derived and covered by tests/test_gradcheck.cpp (check_mamba2_nstate).
  bool proper_state_expansion = false;
};

struct MambaStreamSnapshot {
  bool enabled = false;
  std::shared_ptr<Tensor> state;
  // Proper-path incremental decode state (see Mamba2SSD::pp_stream_*).  Carried
  // through fork/restore so multi-sequence streaming does not bleed SSD state /
  // conv window across sequences (single-session snapshot path).
  std::vector<float> proper_state;
  std::vector<float> proper_ring;
  bool proper_active = false;
};

class Mamba2SSD {
public:
  Mamba2SSD(int d_model, int d_state, int n_heads,
            const MambaConfig &config = {});

  Tensor forward(const Tensor &u, Context *ctx = nullptr);
  Tensor backward(const Tensor &grad_output, Context &ctx);
  void reset();
  void to(Device dev);
  std::vector<Parameter *> parameters();
  void set_streaming_mode(bool enabled);
  bool streaming_mode() const { return streaming_inference_; }
  // True when the corrected selective-SSM path is active.  Used to gate batched
  // fork/restore streaming (that path snapshots only the legacy [1,d_model]
  // state, not the proper SSD state + conv window).
  bool proper_selective_ssm_enabled() const {
    return config_.proper_selective_ssm;
  }
  MambaStreamSnapshot snapshot_streaming_state() const;
  void restore_streaming_state(const MambaStreamSnapshot& snapshot);
  std::vector<MambaStreamSnapshot> snapshot_streaming_state_batch() const;
  void restore_streaming_state_batch(const std::vector<MambaStreamSnapshot>& snapshots);
  int streaming_batch_size() const;
  void collect_bitlinear_layers(std::vector<BitLinear*>& out);
  void reset_runtime_telemetry();
  size_t gpu_fast_path_hits() const { return gpu_fast_path_hits_; }
  size_t gpu_fast_path_fallbacks() const { return gpu_fast_path_fallbacks_; }
  const std::string& last_fallback_reason() const { return last_fallback_reason_; }

  // Accessors
  const std::string &get_layer_name() const { return layer_name; }
  void set_layer_name(const std::string &name) { layer_name = name; }

private:
  // Forward SSM core logic
  Tensor ssd_forward(const Tensor &x, const Tensor &delta, const Tensor &A,
                     const Tensor &B, const Tensor &C, Context *ctx,
                     bool save_history);

  // Backward SSM with BPTT Through Time
  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
  ssd_backward(const Tensor &grad_y, const Tensor &x, const Tensor &delta,
               const Tensor &A, const Tensor &B, const Tensor &C);

  // ── Proper selective SSM path (config_.proper_selective_ssm) ──────────────
  // Full forward/backward for the corrected recurrence with independent
  // projections + causal conv1d.  Operates on CPU (host) tensors; GPU inputs are
  // moved host-side and the result restored to the input device, mirroring the
  // historical CPU scan fallback.  All gradients hand-derived; gradchecked.
  Tensor forward_proper(const Tensor &u);
  Tensor backward_proper(const Tensor &grad_output);
  // N-state (full Mamba-2 SSD) variant of the proper path — config_.
  // proper_state_expansion.  Same conv1d + SiLU gate, but the recurrence carries
  // an H×P×N state and B/C/dt/A are per-head.  Scan + its BPTT run on host in v1
  // (GPU kernels wired in Phase B); projections/gate via device-agnostic ops.
  Tensor forward_proper_nstate(const Tensor &u);
  Tensor backward_proper_nstate(const Tensor &grad_output);
  // Single-token incremental decode for the proper path.  Carries the SSD state
  // and a (K-1)-tap conv window (pp_stream_*) across calls so each generated
  // token is O(1) instead of re-scanning the whole prefix.  Numerically mirrors
  // the corresponding forward_proper* one step at a time.
  Tensor forward_proper_step(const Tensor &u);
  Tensor forward_proper_nstate_step(const Tensor &u);
  // Shared conv step: xv_host [1,dim] -> xc=silu(causal_conv) [1,dim] using the
  // carried window, then advances the window.  Host math (parity with the scan).
  Tensor proper_stream_conv_(const Tensor &xv_host);
  // Causal depthwise conv1d over [rows, dim] laid out as `batch` sequences of
  // `seq` steps.  weight is [dim, K]; out[t,c] = sum_j weight[c,j]*in[t-(K-1)+j,c]
  // with left zero-padding.  Pure host math.
  static void conv1d_causal_forward(const float *in, const float *weight,
                                    float *out, int batch, int seq, int dim,
                                    int K);
  static void conv1d_causal_backward(const float *grad_out, const float *in,
                                     const float *weight, float *grad_in,
                                     float *grad_weight, int batch, int seq,
                                     int dim, int K);

  // Numeric stable helpers
  static float softplus_stable(float x);
  static float sigmoid_stable(float x);
  static float silu_stable(float x);
  static float d_silu_stable(float x, float sigmoid_x);

  // Gating application helper
  Tensor apply_gating(const Tensor &y_ssd, const Tensor &x, const Tensor &z,
                      Tensor *grad_buffer = nullptr);
  void update_streaming_state_from_history(const Tensor &input);

  int d_model;
  int d_state;
  int n_heads;
  int d_head;

  MambaConfig config_;

  // Mixed Precision Components
  BitLinear in_proj_robust;    // 1.58-bit (z, x)
  BitLinear in_proj_sensitive; // FP32 (dt, B, C)
  BitLinear out_proj;          // 1.58-bit (y)

  Parameter A;
  Parameter D; // Skip connection D

  // ── Proper-path components (allocated only when proper_selective_ssm) ──────
  // unique_ptr so the DEFAULT path constructs and serializes exactly the same
  // parameter set as before (these stay null → not exposed via parameters()).
  std::unique_ptr<BitLinear> x_proj_;  // x stream
  std::unique_ptr<BitLinear> z_proj_;  // gate stream (silu)
  std::unique_ptr<BitLinear> B_proj_;  // selective B
  std::unique_ptr<BitLinear> C_proj_;  // selective C
  std::unique_ptr<BitLinear> dt_proj_; // timestep stream
  Parameter conv_weight_;              // [d_model, conv_kernel] depthwise causal
  int conv_kernel_ = 0;                // 0 until proper path constructed

  std::string layer_name = "mamba";
  Tensor saved_input_;
  Tensor saved_x_proj_;
  Tensor saved_gate_;
  Tensor saved_delta_;
  Tensor saved_B_;
  Tensor saved_C_;
  Tensor saved_ssd_;
  Tensor saved_state_history_;

  // Proper-path forward caches (host tensors) for backward_proper.
  bool proper_active_ = false; // true after a forward_proper ran
  Tensor pp_u_;                // layer input [rows, dim]
  Tensor pp_xv_;              // x_proj output (pre-conv)
  Tensor pp_conv_pre_;       // conv output (pre-silu)
  Tensor pp_xc_;             // silu(conv) — the SSM input
  Tensor pp_z_;              // gate projection raw
  Tensor pp_B_;              // B projection
  Tensor pp_C_;              // C projection
  Tensor pp_dt_;             // dt projection raw
  Tensor pp_h_hist_;         // diagonal state history h_t [rows, dim]
  Tensor pp_y_ssd_;          // h_t * C_t (pre-gate)
  Tensor pp_state_hist_;     // N-state SSD history [rows, H, P, N] (BPTT)
  int pp_batch_ = 0;
  int pp_seq_ = 0;

  // ── Incremental proper-path streaming decode (opt-in) ──────────────────────
  // Primed by the prefill forward_proper*/nstate when streaming_inference_ is on,
  // then advanced one token per call by forward_proper_step*.  pp_stream_state_
  // is the carried SSD state ([dim] diagonal / [H*P*N] nstate); pp_stream_ring_
  // is the (K-1)-tap causal-conv window over x_proj ([slot*dim + c], slot 0 =
  // oldest).  Cleared on reset()/set_streaming_mode().
  std::vector<float> pp_stream_state_;
  std::vector<float> pp_stream_ring_;
  bool pp_stream_active_ = false;

  // Reusable buffers for scan operation
  struct ThreadBuffers {
    std::vector<float> ssm_state;
    std::vector<float> ssm_history;
    std::vector<float> grad_state;
  };
  static thread_local ThreadBuffers buffers_;
  bool streaming_inference_ = false;
  std::shared_ptr<Tensor> streaming_state_;
  size_t gpu_fast_path_hits_ = 0;
  size_t gpu_fast_path_fallbacks_ = 0;
  std::string last_fallback_reason_;
};

} // namespace nsos

#endif
