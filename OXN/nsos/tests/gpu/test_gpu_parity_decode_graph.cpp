// =====================================================================
// GPU parity — CUDA-graph decode (NSOS_CUDA_GRAPH_DECODE, NSOS_CUDA_PTDS).
//
// Contract: greedy decode driven through JambaModel::forward_ids_decode_graph
// (warm eager step -> one-step capture -> per-token graph replays) must
// reproduce the eager forward_ids decode TOKEN-FOR-TOKEN, and the final-step
// logits must match.  The graph replays the exact same kernels on the same
// buffers, so any divergence means a frozen host argument leaked into the
// capture (position, shared-memory sizing, staging) — precisely the failure
// class this gate exists to catch.
//
// Model: the FULL architecture — Mamba-2 N-state (default) + GQA attention +
// MoE (dense single-row device decode path).  Requires an NSOS_CUDA_PTDS
// build — on a legacy-stream build the test SKIPs (exit 0) because
// default-stream kernel launches cannot be captured.
// =====================================================================
#include "gpu_parity_common.h"

#include "../../include/jamba.h"
#include "../../include/nsos/determinism.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

namespace {

int greedy_from(const Tensor& logits) {
  Tensor host = logits.cpu();
  const int vocab = host.shape.back();
  const float* row = host.data() + (host.size - vocab);
  int best = 0;
  for (int t = 1; t < vocab; ++t) {
    if (row[t] > row[best]) best = t;
  }
  return best;
}

}  // namespace

int main() {
#ifndef NSOS_CUDA_PTDS
  std::cout << "[GPUParity:decode_graph] skipped: build without NSOS_CUDA_PTDS "
               "(legacy default stream cannot be captured; configure with "
               "-DNSOS_CUDA_PTDS=ON)."
            << std::endl;
  return 0;
#else
  // Both envs must be set BEFORE any layer/model use (the gates cache on
  // first read).  GPU-resident Mamba step is a graph-decode prerequisite.
#ifdef _WIN32
  _putenv_s("NSOS_MAMBA_GPU_STEP", "1");
  _putenv_s("NSOS_CUDA_GRAPH_DECODE", "1");
#else
  setenv("NSOS_MAMBA_GPU_STEP", "1", 1);
  setenv("NSOS_CUDA_GRAPH_DECODE", "1", 1);
#endif

  return run_parity("decode_graph", [] {
    determinism::DeterminismManager::instance().set_global_seed(1234);

    ModelConfig mc;
    mc.num_layers = 4;
    mc.d_model = 64;
    mc.vocab_size = 96;
    mc.n_heads = 4;
    mc.n_kv_heads = 2;
    mc.attention_period = 2;  // layers 2 and 4 are attention
    mc.attention_slot = 1;
    // FULL architecture under the graph: N-state Mamba (config default) via
    // the fused device step, and MoE via the dense single-row device decode
    // path (top-k-masked weights read on the device).
    mc.use_moe = true;
    mc.num_experts = 4;
    mc.num_experts_per_token = 2;
    mc.moe_period = 2;   // MoE FFN on layers 1 and 3
    mc.moe_slot = 0;
    mc.use_ttt = false;
    mc.max_context_tokens = 256;
    mc.use_cuda = true;

    JambaModel model(mc, Device::GPU);
    model.to(Device::GPU);
    model.set_training_mode(false);

    const std::vector<int> prompt = {5, 17, 3, 42, 9, 88, 21, 60};
    const int decode_steps = 24;
    const int reserve_total = static_cast<int>(prompt.size()) + decode_steps + 4;

    // ── Eager reference decode ──────────────────────────────────────
    model.reset_session();
    model.set_streaming_inference(true);
    model.reserve_kv_cache(reserve_total, Device::GPU, 1);
    Tensor logits = model.forward_ids(prompt, nullptr);
    std::vector<int> eager_tokens;
    int tok = greedy_from(logits);
    for (int s = 0; s < decode_steps; ++s) {
      eager_tokens.push_back(tok);
      logits = model.forward_ids({tok}, nullptr);
      tok = greedy_from(logits);
    }
    Tensor eager_last = logits.cpu();
    model.set_streaming_inference(false);
    // Localize faults precisely across the two failed T4 rounds: prove the
    // GPU-first eager decode path (N-state step + MoE dense + attention
    // decode) is clean BEFORE any graph capture.  If this throws, the bug is
    // in the decode integration, NOT the capture.
    cuda_sync_or_throw("eager-reference-decode");

    // ── Graph decode (warm -> capture -> replays) ───────────────────
    model.reset_session();
    model.set_streaming_inference(true);
    model.reserve_kv_cache(reserve_total, Device::GPU, 1);
    logits = model.forward_ids(prompt, nullptr);
    cuda_sync_or_throw("graph-prefill");
    std::vector<int> graph_tokens;
    bool graph_engaged = false;
    tok = greedy_from(logits);
    for (int s = 0; s < decode_steps; ++s) {
      graph_tokens.push_back(tok);
      Tensor graphed = model.forward_ids_decode_graph(tok);
      if (graphed.size > 0) {
        logits = graphed;
      } else {
        logits = model.forward_ids({tok}, nullptr);
      }
      // s==0 warm (eager), s==1 capture + first replay, s>=2 pure replays —
      // a labeled sync per step names the exact failing phase.
      cuda_sync_or_throw(
          (std::string("graph-step-") + std::to_string(s) + "-" +
           model.decode_graph_status()).c_str());
      graph_engaged = graph_engaged || model.decode_graph_active();
      tok = greedy_from(logits);
    }
    Tensor graph_last = logits.cpu();
    const std::string status = model.decode_graph_status();
    model.set_streaming_inference(false);

    if (!graph_engaged) {
      throw std::runtime_error(
          "decode graph never activated on this GPU — status: " + status);
    }
    for (size_t i = 0; i < eager_tokens.size(); ++i) {
      if (eager_tokens[i] != graph_tokens[i]) {
        throw std::runtime_error(
            "token divergence at step " + std::to_string(i) + ": eager=" +
            std::to_string(eager_tokens[i]) + " graph=" +
            std::to_string(graph_tokens[i]) + " (status: " + status + ")");
      }
    }
    assert_close(eager_last, graph_last, 1e-5f, "final-step logits");
    std::cout << "[GPUParity:decode_graph] " << decode_steps
              << " tokens byte-identical; status=" << status << std::endl;
  });
#endif  // NSOS_CUDA_PTDS
}
