#ifndef JAMBA_H
#define JAMBA_H

#include "autograd.h"
#include "bitlinear.h"
#include "embedding.h"
#include "mamba2.h"
#include "mcts_reasoning.h"
#include "memory_system.h"
#include "module.h"
#include "ttt_layer.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nsos {

class Attention {
public:
  Attention(int d_model, int n_heads, int n_latents = 512);
  Tensor forward(const Tensor &input, Context *ctx);
  Tensor backward(const Tensor &dy, Context *ctx);
  void to(Device dev);
  std::vector<Parameter *> parameters();

private:
  int d_model;
  int n_heads;
  int head_dim;
  int n_latents;
  std::unique_ptr<BitLinear> q_down_proj, q_up_proj, kv_down_proj, kv_up_proj, q_rope_proj, k_rope_proj, out_proj;
  std::vector<float> cos_cached, sin_cached;
  int max_seq_len;
  float theta;
  void precompute_freqs_cis();
  std::pair<Tensor, Tensor> apply_rope(const Tensor &q, const Tensor &k, int start_pos = 0);
};

class MoERouter {
public:
  MoERouter(int d_model, int n = 256, int k = 8);
  std::pair<Tensor, Tensor> forward(const Tensor &x);
  Tensor backward(const Tensor &grad_logits);
  void to(Device dev);
  std::vector<Parameter *> parameters();
  int num_experts, top_k;
  float aux_loss_coef;
  std::vector<float> expert_loads;
  float compute_aux_loss();
  std::unique_ptr<BitLinear> gate, shared_expert_gate, shared_expert_up, shared_expert_down;
};

class JambaBlock {
public:
  JambaBlock(int d_model, bool is_attn, bool is_moe_flag, bool is_ttt_layer, int layer_idx, int total_layers);
  ~JambaBlock();
  Tensor forward(const Tensor &x, Context *ctx);
  Tensor backward(const Tensor &dy, Context *ctx);
  void reset();
  void to(Device dev);
  std::vector<Parameter *> parameters();

private:
  Tensor forward_moe(const Tensor &x, Context *ctx, const std::string &ln);
  Tensor backward_moe(const Tensor &dy, Context *ctx, const std::string &ln, const Tensor &x);
  std::unique_ptr<Parameter> norm_gamma1, norm_gamma2;
  bool is_attention, is_moe, is_ttt;
  int layer_idx, total_layers, d_model, num_experts;
public:
  std::unique_ptr<Attention> attn_layer;
  std::unique_ptr<Mamba2SSD> mamba_layer;
  std::unique_ptr<TTTLayer> ttt_layer;
  std::unique_ptr<MoERouter> router;
  std::vector<std::unique_ptr<BitLinear>> expert_gate_up, expert_down;
  std::unique_ptr<BitLinear> ffn_gate_up, ffn_down;
};

class JambaModel : public Module {
public:
  std::vector<std::unique_ptr<JambaBlock>> layers;
  std::unique_ptr<Embedding> embedding;
  std::unique_ptr<BitLinear> value_head;
  std::unique_ptr<MCTSReasoning> mcts;

  JambaModel(int num_layers, int d_model, int vocab_size = 128000, Device device = Device::CPU);
  Tensor forward(const Tensor &x) override;
  Tensor forward(const Tensor &x, Context *ctx);
  void to(Device device) override;
  Tensor forward_ids(const std::vector<int> &ids, Context *ctx = nullptr);
  Tensor forward_trunk(const std::vector<int> &ids, Context *ctx = nullptr);
  Tensor forward_embedding(const Tensor &x, Context *ctx = nullptr);
  Tensor reason(const Tensor &x, int num_simulations = 100);
  std::vector<Parameter *> parameters() override;
  void save(const std::string &filename);
  void load(const std::string &filename);
  Tensor forward_thought(const Tensor &x, int steps);
  void run_reasoning_loop(int iterations);
  void backward_external(const Tensor &grad, Context &ctx);
  void backward_embedding(const Tensor &grad, Context &ctx);
  void backward(const Tensor &grad, Context &ctx);
  void reset_session();
  void set_hamiltonian_mode(bool enabled);
  Tensor run_simd_inference(const Tensor &x, int steps);
  void session_adapt(const Tensor &x, const Tensor &y);
private:
  int num_layers, d_model, vocab_size;
  Device device;
};

class D2FDecoder {
public:
    JambaModel* model;
    D2FDecoder(JambaModel* m) : model(m) {}
    std::vector<int> generate(const std::vector<int>& prompt, int length) { return {}; }
};

} // namespace nsos
#endif
