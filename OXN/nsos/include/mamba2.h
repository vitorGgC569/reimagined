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
};

struct MambaStreamSnapshot {
  bool enabled = false;
  std::shared_ptr<Tensor> state;
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

  std::string layer_name = "mamba";
  Tensor saved_input_;
  Tensor saved_x_proj_;
  Tensor saved_gate_;
  Tensor saved_delta_;
  Tensor saved_B_;
  Tensor saved_C_;
  Tensor saved_ssd_;
  Tensor saved_state_history_;

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
