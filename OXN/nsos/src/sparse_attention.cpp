#include <cstdio>
#include "sparse_attention.h"
#include "nsos/determinism.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include "cuda/sparse_attention_kernels.cuh"
#endif

namespace nsos {

namespace {
void select_sparse_positions(
    int query_index, int sequence_length, int head_dim, int block_size,
    const SparseAttentionConfig& cfg, const float* route,
    const float* block_means, float scale, std::vector<char>& selected,
    std::vector<std::pair<float, int>>& candidates, std::vector<int>& positions,
    long long* block_score_pairs);
}

SparseAttentionBackwardResult sparse_selective_attention_backward(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const SparseAttentionConfig& cfg, const Tensor& dOut,
    const Tensor* Wsel) {
  if (Q.shape.size() != 2 || K.shape != Q.shape || V.shape != Q.shape ||
      dOut.shape != Q.shape) {
    throw std::invalid_argument(
        "sparse_selective_attention_backward expects aligned rank-2 tensors");
  }
#ifdef USE_CUDA
  if (Q.get_device() == Device::GPU && K.get_device() == Device::GPU &&
      V.get_device() == Device::GPU && dOut.get_device() == Device::GPU &&
      Q.shape[1] <= 256 && cfg.top_k_blocks >= 0 &&
      cfg.top_k_blocks <= 64) {
    const int n = Q.shape[0];
    const int d = Q.shape[1];
    const int B = std::max(cfg.block_size, 1);
    const int nb = (n + B - 1) / B;
    const float scale = cfg.scale > 0.0f
                            ? cfg.scale
                            : 1.0f / std::sqrt(static_cast<float>(d));
    Tensor route_owned;
    const Tensor* route = &Q;
    if (Wsel != nullptr) {
      if (Wsel->get_device() != Device::GPU) {
        throw std::invalid_argument(
            "GPU SSA backward requires Wsel on the GPU");
      }
      if (Wsel->shape.size() == 2 && Wsel->shape[0] == d &&
          Wsel->shape[1] == d) {
        route_owned = Q.matmul(Wsel->transpose());
        route = &route_owned;
      }
    }
    Tensor block_means({nb, d}, Device::GPU);
    Tensor dQ = Tensor::zeros({n, d}, Device::GPU);
    Tensor dK = Tensor::zeros({n, d}, Device::GPU);
    Tensor dV = Tensor::zeros({n, d}, Device::GPU);
    cuda::launch_sparse_block_means(
        K.raw_data(), block_means.raw_data(), n, d, B);
    cuda::launch_sparse_selective_attention_backward(
        Q.raw_data(), K.raw_data(), V.raw_data(), route->raw_data(),
        block_means.raw_data(), dOut.raw_data(), dQ.raw_data(), dK.raw_data(),
        dV.raw_data(), n, d, B, cfg.top_k_blocks, cfg.local_blocks,
        cfg.sink_blocks, scale,
        determinism::deterministic_reductions_enabled());
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("SSA backward CUDA launch failed: ") +
          cudaGetErrorString(status));
    }
    return {dQ, dK, dV};
  }
#endif
  if (Q.get_device() == Device::GPU && strict_gpu_execution()) {
    throw std::runtime_error(
        "Strict GPU execution forbids SSA backward host fallback");
  }

  const Device output_device = Q.get_device();
  Tensor q_host = output_device == Device::GPU ? Q.cpu() : Q;
  Tensor k_host = K.get_device() == Device::GPU ? K.cpu() : K;
  Tensor v_host = V.get_device() == Device::GPU ? V.cpu() : V;
  Tensor go_host = dOut.get_device() == Device::GPU ? dOut.cpu() : dOut;
  Tensor w_host;
  const Tensor* w_ptr = nullptr;
  if (Wsel != nullptr) {
    w_host = Wsel->get_device() == Device::GPU ? Wsel->cpu() : *Wsel;
    w_ptr = &w_host;
  }

  const int n = q_host.shape[0];
  const int d = q_host.shape[1];
  const int B = std::max(cfg.block_size, 1);
  const int nb = (n + B - 1) / B;
  const float scale =
      cfg.scale > 0.0f ? cfg.scale : 1.0f / std::sqrt(static_cast<float>(d));

  Tensor block_mean({nb, d}, Device::CPU);
  float* bm = block_mean.data();
  const float* k = k_host.data();
  for (int block = 0; block < nb; ++block) {
    const int begin = block * B;
    const int end = std::min((block + 1) * B, n);
    for (int row = begin; row < end; ++row) {
      for (int dim = 0; dim < d; ++dim) {
        bm[static_cast<size_t>(block) * d + dim] +=
            k[static_cast<size_t>(row) * d + dim];
      }
    }
    const float inv = 1.0f / static_cast<float>(std::max(end - begin, 1));
    for (int dim = 0; dim < d; ++dim) {
      bm[static_cast<size_t>(block) * d + dim] *= inv;
    }
  }

  const float* q = q_host.data();
  const float* route = q;
  Tensor route_owned;
  if (w_ptr != nullptr && w_ptr->shape.size() == 2 &&
      w_ptr->shape[0] == d && w_ptr->shape[1] == d) {
    route_owned = q_host.matmul(w_ptr->transpose());
    route = route_owned.data();
  }

  Tensor dQ({n, d}, Device::CPU);
  Tensor dK({n, d}, Device::CPU);
  Tensor dV({n, d}, Device::CPU);
  float* dq = dQ.data();
  float* dk = dK.data();
  float* dv = dV.data();
  const float* v = v_host.data();
  const float* go = go_host.data();

  std::vector<char> selected;
  std::vector<std::pair<float, int>> candidates;
  std::vector<int> positions;
  std::vector<float> probabilities;
  std::vector<float> dprobabilities;
  for (int i = 0; i < n; ++i) {
    select_sparse_positions(i, n, d, B, cfg, route, bm, scale, selected,
                            candidates, positions, nullptr);
    if (positions.empty()) continue;

    probabilities.assign(positions.size(), 0.0f);
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t slot = 0; slot < positions.size(); ++slot) {
      const int j = positions[slot];
      float score = 0.0f;
      for (int dim = 0; dim < d; ++dim) {
        score += q[static_cast<size_t>(i) * d + dim] *
                 k[static_cast<size_t>(j) * d + dim];
      }
      probabilities[slot] = score * scale;
      max_score = std::max(max_score, probabilities[slot]);
    }
    float sum = 0.0f;
    for (float& value : probabilities) {
      value = std::exp(value - max_score);
      sum += value;
    }
    const float inv_sum = 1.0f / (sum + 1e-20f);
    for (float& value : probabilities) value *= inv_sum;

    dprobabilities.assign(positions.size(), 0.0f);
    float softmax_dot = 0.0f;
    for (size_t slot = 0; slot < positions.size(); ++slot) {
      const int j = positions[slot];
      float upstream = 0.0f;
      for (int dim = 0; dim < d; ++dim) {
        upstream += go[static_cast<size_t>(i) * d + dim] *
                    v[static_cast<size_t>(j) * d + dim];
        dv[static_cast<size_t>(j) * d + dim] +=
            probabilities[slot] * go[static_cast<size_t>(i) * d + dim];
      }
      dprobabilities[slot] = upstream;
      softmax_dot += probabilities[slot] * upstream;
    }

    for (size_t slot = 0; slot < positions.size(); ++slot) {
      const int j = positions[slot];
      const float dscore = probabilities[slot] *
                           (dprobabilities[slot] - softmax_dot) * scale;
      for (int dim = 0; dim < d; ++dim) {
        dq[static_cast<size_t>(i) * d + dim] +=
            dscore * k[static_cast<size_t>(j) * d + dim];
        dk[static_cast<size_t>(j) * d + dim] +=
            dscore * q[static_cast<size_t>(i) * d + dim];
      }
    }
  }

  if (output_device == Device::GPU) {
    return {dQ.to(Device::GPU), dK.to(Device::GPU), dV.to(Device::GPU)};
  }
  return {dQ, dK, dV};
}

namespace {
void ssa_warn_gpu_once(const Tensor& t) {
    if (t.get_device() != Device::GPU) return;
    if (strict_gpu_execution()) {
        throw std::runtime_error(
            "Strict GPU execution forbids SSA host fallback");
    }
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[ssa] WARN: sparse attention executa em CPU (sem kernel "
                     "CUDA ainda) — entrada GPU sera copiada; use só em "
                     "fine-tune/inferencia CPU\n");
    }
}

void select_sparse_positions(
    int query_index, int sequence_length, int head_dim, int block_size,
    const SparseAttentionConfig& cfg, const float* route,
    const float* block_means, float scale, std::vector<char>& selected,
    std::vector<std::pair<float, int>>& candidates, std::vector<int>& positions,
    long long* block_score_pairs = nullptr) {
  const int current_block = query_index / block_size;
  const int candidate_blocks = current_block + 1;
  selected.assign(static_cast<size_t>(candidate_blocks), 0);

  const int sinks =
      std::min(std::max(cfg.sink_blocks, 0), candidate_blocks);
  for (int block = 0; block < sinks; ++block) {
    selected[static_cast<size_t>(block)] = 1;
  }

  const int local = std::max(cfg.local_blocks, 0);
  const int local_begin = std::max(0, current_block - local + 1);
  for (int block = local_begin; block <= current_block; ++block) {
    selected[static_cast<size_t>(block)] = 1;
  }

  if (cfg.top_k_blocks > 0) {
    candidates.clear();
    for (int block = 0; block < candidate_blocks; ++block) {
      if (selected[static_cast<size_t>(block)]) continue;
      float score = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += route[static_cast<size_t>(query_index) * head_dim + dim] *
                 block_means[static_cast<size_t>(block) * head_dim + dim];
      }
      candidates.emplace_back(score * scale, block);
    }
    if (block_score_pairs) {
      *block_score_pairs += static_cast<long long>(candidates.size());
    }
    const int keep =
        std::min(cfg.top_k_blocks, static_cast<int>(candidates.size()));
    std::partial_sort(
        candidates.begin(), candidates.begin() + keep, candidates.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
    for (int rank = 0; rank < keep; ++rank) {
      selected[static_cast<size_t>(
          candidates[static_cast<size_t>(rank)].second)] = 1;
    }
  }

  positions.clear();
  for (int block = 0; block < candidate_blocks; ++block) {
    if (!selected[static_cast<size_t>(block)]) continue;
    const int begin = block * block_size;
    const int end = std::min((block + 1) * block_size, query_index + 1);
    for (int key_index = begin;
         key_index < end && key_index < sequence_length; ++key_index) {
      positions.push_back(key_index);
    }
  }
}
}  // namespace

Tensor dense_causal_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                              float scale) {
  const int n = Q.shape[0];
  const int d = Q.shape[1];
  if (scale <= 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(d));
  }
  Tensor out({n, d}, Device::CPU);  // zero-filled
  const float* q = Q.data();
  const float* k = K.data();
  const float* v = V.data();
  float* o = out.data();

  std::vector<float> sc;
  for (int i = 0; i < n; ++i) {
    sc.assign(static_cast<size_t>(i) + 1, 0.0f);
    float maxs = -std::numeric_limits<float>::infinity();
    for (int j = 0; j <= i; ++j) {
      float s = 0.0f;
      for (int c = 0; c < d; ++c) {
        s += q[static_cast<size_t>(i) * d + c] * k[static_cast<size_t>(j) * d + c];
      }
      sc[static_cast<size_t>(j)] = s * scale;
      maxs = std::max(maxs, sc[static_cast<size_t>(j)]);
    }
    float sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
      sc[static_cast<size_t>(j)] = std::exp(sc[static_cast<size_t>(j)] - maxs);
      sum += sc[static_cast<size_t>(j)];
    }
    const float inv = 1.0f / (sum + 1e-20f);
    for (int j = 0; j <= i; ++j) {
      const float w = sc[static_cast<size_t>(j)] * inv;
      for (int c = 0; c < d; ++c) {
        o[static_cast<size_t>(i) * d + c] += w * v[static_cast<size_t>(j) * d + c];
      }
    }
  }
  return out;
}

Tensor sparse_selective_attention(const Tensor& Q, const Tensor& K,
                                  const Tensor& V,
                                  const SparseAttentionConfig& cfg,
                                  SparseAttentionStats* stats,
                                  const Tensor* Wsel) {
  const int n = Q.shape[0];
  const int d = Q.shape[1];
  const int B = std::max(cfg.block_size, 1);
  const int nb = (n + B - 1) / B;
  float scale = cfg.scale;
  if (scale <= 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(d));
  }

#ifdef USE_CUDA
  // GPU fast path: all operands on device, no stats requested (stats need
  // host-side counters), and within the kernel caps (head dim <= 256, top_k
  // <= 64).  Matches the CPU result within float tolerance (parity-tested).
  if (Q.get_device() == Device::GPU && K.get_device() == Device::GPU &&
      V.get_device() == Device::GPU && stats == nullptr && d <= 256 &&
      cfg.top_k_blocks <= 64) {
    Tensor route_owned;
    const Tensor* route_t = &Q;  // default: raw query routes block selection
    if (Wsel != nullptr && Wsel->shape.size() == 2 && Wsel->shape[0] == d &&
        Wsel->shape[1] == d) {
      route_owned = Q.matmul(Wsel->transpose());  // route[i] = Wsel @ q[i]
      route_t = &route_owned;
    }
    Tensor bm({nb, d}, Device::GPU);
    cuda::launch_sparse_block_means(K.raw_data(), bm.raw_data(), n, d, B);
    Tensor out_gpu({n, d}, Device::GPU);
    cuda::launch_sparse_selective_attention(
        Q.raw_data(), K.raw_data(), V.raw_data(), route_t->raw_data(),
        bm.raw_data(), out_gpu.raw_data(), n, d, B, cfg.top_k_blocks,
        cfg.local_blocks, cfg.sink_blocks, scale);
    return out_gpu;
  }
#endif

  ssa_warn_gpu_once(Q);

  const float* q = Q.data();
  const float* k = K.data();
  const float* v = V.data();

  // Per-block key summaries (mean key over the block) -- the cheap routing
  // signal scored against the query (O(n/B) per query instead of O(n)).
  Tensor block_mean({nb, d}, Device::CPU);  // zero-filled
  float* bm = block_mean.data();
  for (int b = 0; b < nb; ++b) {
    const int start = b * B;
    const int end = std::min((b + 1) * B, n);
    const int cnt = end - start;
    for (int j = start; j < end; ++j) {
      for (int c = 0; c < d; ++c) {
        bm[static_cast<size_t>(b) * d + c] += k[static_cast<size_t>(j) * d + c];
      }
    }
    if (cnt > 0) {
      const float inv = 1.0f / static_cast<float>(cnt);
      for (int c = 0; c < d; ++c) {
        bm[static_cast<size_t>(b) * d + c] *= inv;
      }
    }
  }

  // Optional LEARNED routing query for block scoring: route = Wsel @ q.  The
  // exact attention below still uses the raw q; only the block SELECTION uses
  // the learned routing.  Wsel == identity reproduces the mean-key heuristic.
  const float* route = q;
  Tensor qroute;
  if (Wsel != nullptr && Wsel->shape.size() == 2 && Wsel->shape[0] == d &&
      Wsel->shape[1] == d) {
    qroute = Tensor({n, d}, Device::CPU);
    const float* w = Wsel->data();
    float* r = qroute.data();
    for (int i = 0; i < n; ++i)
      for (int a = 0; a < d; ++a) {
        float acc = 0.0f;
        for (int b = 0; b < d; ++b)
          acc += w[static_cast<size_t>(a) * d + b] * q[static_cast<size_t>(i) * d + b];
        r[static_cast<size_t>(i) * d + a] = acc;
      }
    route = qroute.data();
  }

  Tensor out({n, d}, Device::CPU);  // zero-filled
  float* o = out.data();

  long long dense_pairs = 0;
  long long sparse_pairs = 0;
  long long block_score_pairs = 0;

  std::vector<char> selected;
  std::vector<std::pair<float, int>> cand;
  std::vector<int> js;
  std::vector<float> sc;

  for (int i = 0; i < n; ++i) {
    select_sparse_positions(i, n, d, B, cfg, route, bm, scale, selected,
                            cand, js, &block_score_pairs);

    // Exact softmax attention over the selected positions only.
    sc.assign(js.size(), 0.0f);
    float maxs = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < js.size(); ++t) {
      const int j = js[t];
      float s = 0.0f;
      for (int c = 0; c < d; ++c) {
        s += q[static_cast<size_t>(i) * d + c] * k[static_cast<size_t>(j) * d + c];
      }
      sc[t] = s * scale;
      maxs = std::max(maxs, sc[t]);
    }
    float sum = 0.0f;
    for (size_t t = 0; t < js.size(); ++t) {
      sc[t] = std::exp(sc[t] - maxs);
      sum += sc[t];
    }
    const float inv = 1.0f / (sum + 1e-20f);
    for (size_t t = 0; t < js.size(); ++t) {
      const int j = js[t];
      const float w = sc[t] * inv;
      for (int c = 0; c < d; ++c) {
        o[static_cast<size_t>(i) * d + c] += w * v[static_cast<size_t>(j) * d + c];
      }
    }

    dense_pairs += static_cast<long long>(i) + 1;
    sparse_pairs += static_cast<long long>(js.size());
  }

  if (stats != nullptr) {
    stats->dense_attended_pairs = dense_pairs;
    stats->sparse_attended_pairs = sparse_pairs;
    // 2*d FLOPs per scored pair (multiply + add); block scoring adds its own.
    const long long two_d = 2LL * d;
    stats->dense_score_flops = dense_pairs * two_d;
    stats->sparse_score_flops = (sparse_pairs + block_score_pairs) * two_d;
  }
  return out;
}

// ── Learned block selection: shared helpers ──
namespace {

// Per-block mean key: [nb, d].
Tensor compute_block_means(const Tensor& K, int B, int n, int d) {
  const int nb = (n + B - 1) / B;
  Tensor bm({nb, d}, Device::CPU);  // zero-filled
  float* m = bm.data();
  const float* k = K.data();
  for (int b = 0; b < nb; ++b) {
    const int start = b * B;
    const int end = std::min((b + 1) * B, n);
    const int cnt = end - start;
    for (int j = start; j < end; ++j)
      for (int c = 0; c < d; ++c)
        m[static_cast<size_t>(b) * d + c] += k[static_cast<size_t>(j) * d + c];
    if (cnt > 0) {
      const float inv = 1.0f / static_cast<float>(cnt);
      for (int c = 0; c < d; ++c) m[static_cast<size_t>(b) * d + c] *= inv;
    }
  }
  return bm;
}

// q_sel = Q @ Wsel^T : q_sel[i][a] = sum_b Wsel[a][b] * Q[i][b].  [n, d].
Tensor project_qsel(const Tensor& Q, const Tensor& Wsel, int n, int d) {
  Tensor qs({n, d}, Device::CPU);  // zero-filled
  float* qsd = qs.data();
  const float* q = Q.data();
  const float* w = Wsel.data();
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < d; ++a) {
      float acc = 0.0f;
      for (int b = 0; b < d; ++b)
        acc += w[static_cast<size_t>(a) * d + b] * q[static_cast<size_t>(i) * d + b];
      qsd[static_cast<size_t>(i) * d + a] = acc;
    }
  return qs;
}

}  // namespace

Tensor learned_block_bias_attention(const Tensor& Q, const Tensor& K,
                                    const Tensor& V, const Tensor& Wsel,
                                    const SparseAttentionConfig& cfg) {
  const int n = Q.shape[0];
  const int d = Q.shape[1];
  const int B = std::max(cfg.block_size, 1);
  float scale = cfg.scale > 0.0f ? cfg.scale : 1.0f / std::sqrt(static_cast<float>(d));

  const Tensor bm = compute_block_means(K, B, n, d);
  const Tensor qsel = project_qsel(Q, Wsel, n, d);
  const float* q = Q.data();
  const float* k = K.data();
  const float* v = V.data();
  const float* bmd = bm.data();
  const float* qs = qsel.data();

  Tensor out({n, d}, Device::CPU);  // zero-filled
  float* o = out.data();

  std::vector<float> lg;
  for (int i = 0; i < n; ++i) {
    lg.assign(static_cast<size_t>(i) + 1, 0.0f);
    float maxs = -std::numeric_limits<float>::infinity();
    for (int j = 0; j <= i; ++j) {
      float qk = 0.0f;
      for (int c = 0; c < d; ++c)
        qk += q[static_cast<size_t>(i) * d + c] * k[static_cast<size_t>(j) * d + c];
      const int bj = j / B;
      float bb = 0.0f;
      for (int c = 0; c < d; ++c)
        bb += qs[static_cast<size_t>(i) * d + c] * bmd[static_cast<size_t>(bj) * d + c];
      lg[static_cast<size_t>(j)] = (qk + bb) * scale;  // learned block-prior bias
      maxs = std::max(maxs, lg[static_cast<size_t>(j)]);
    }
    float sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
      lg[static_cast<size_t>(j)] = std::exp(lg[static_cast<size_t>(j)] - maxs);
      sum += lg[static_cast<size_t>(j)];
    }
    const float inv = 1.0f / (sum + 1e-20f);
    for (int j = 0; j <= i; ++j) {
      const float w = lg[static_cast<size_t>(j)] * inv;
      for (int c = 0; c < d; ++c)
        o[static_cast<size_t>(i) * d + c] += w * v[static_cast<size_t>(j) * d + c];
    }
  }
  return out;
}

Tensor learned_block_bias_attention_backward(const Tensor& Q, const Tensor& K,
                                             const Tensor& V, const Tensor& Wsel,
                                             const SparseAttentionConfig& cfg,
                                             const Tensor& dOut) {
  const int n = Q.shape[0];
  const int d = Q.shape[1];
  const int B = std::max(cfg.block_size, 1);
  float scale = cfg.scale > 0.0f ? cfg.scale : 1.0f / std::sqrt(static_cast<float>(d));

  const Tensor bm = compute_block_means(K, B, n, d);
  const Tensor qsel = project_qsel(Q, Wsel, n, d);
  const float* q = Q.data();
  const float* k = K.data();
  const float* v = V.data();
  const float* bmd = bm.data();
  const float* qs = qsel.data();
  const float* go = dOut.data();

  // dq_sel accumulates the gradient flowing to the projected query.
  Tensor dqsel({n, d}, Device::CPU);  // zero-filled
  float* dqs = dqsel.data();

  std::vector<float> lg, attn, dattn;
  for (int i = 0; i < n; ++i) {
    const int m = i + 1;
    lg.assign(static_cast<size_t>(m), 0.0f);
    float maxs = -std::numeric_limits<float>::infinity();
    for (int j = 0; j <= i; ++j) {
      float qk = 0.0f;
      for (int c = 0; c < d; ++c)
        qk += q[static_cast<size_t>(i) * d + c] * k[static_cast<size_t>(j) * d + c];
      const int bj = j / B;
      float bb = 0.0f;
      for (int c = 0; c < d; ++c)
        bb += qs[static_cast<size_t>(i) * d + c] * bmd[static_cast<size_t>(bj) * d + c];
      lg[static_cast<size_t>(j)] = (qk + bb) * scale;
      maxs = std::max(maxs, lg[static_cast<size_t>(j)]);
    }
    attn.assign(static_cast<size_t>(m), 0.0f);
    float sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
      attn[static_cast<size_t>(j)] = std::exp(lg[static_cast<size_t>(j)] - maxs);
      sum += attn[static_cast<size_t>(j)];
    }
    const float inv = 1.0f / (sum + 1e-20f);
    for (int j = 0; j <= i; ++j) attn[static_cast<size_t>(j)] *= inv;

    // d_attn[j] = dOut[i] . V[j]
    dattn.assign(static_cast<size_t>(m), 0.0f);
    float dot = 0.0f;
    for (int j = 0; j <= i; ++j) {
      float a = 0.0f;
      for (int c = 0; c < d; ++c)
        a += go[static_cast<size_t>(i) * d + c] * v[static_cast<size_t>(j) * d + c];
      dattn[static_cast<size_t>(j)] = a;
      dot += attn[static_cast<size_t>(j)] * a;
    }
    // softmax backward -> d_logit[j], then flow through the (q_sel . bm)*scale
    // bias into dq_sel[i].
    for (int j = 0; j <= i; ++j) {
      const float dlogit =
          attn[static_cast<size_t>(j)] * (dattn[static_cast<size_t>(j)] - dot);
      const float s = dlogit * scale;
      const int bj = j / B;
      for (int c = 0; c < d; ++c)
        dqs[static_cast<size_t>(i) * d + c] += s * bmd[static_cast<size_t>(bj) * d + c];
    }
  }

  // q_sel[i][a] = sum_b Wsel[a][b] * Q[i][b]  =>  dWsel[a][b] = sum_i dq_sel[i][a] * Q[i][b]
  Tensor dWsel({d, d}, Device::CPU);  // zero-filled
  float* dw = dWsel.data();
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < d; ++a) {
      const float g = dqs[static_cast<size_t>(i) * d + a];
      if (g == 0.0f) continue;
      for (int b = 0; b < d; ++b)
        dw[static_cast<size_t>(a) * d + b] += g * q[static_cast<size_t>(i) * d + b];
    }
  return dWsel;
}

float block_selector_distill_step(const Tensor& Q, const Tensor& K, Tensor& Wsel,
                                  int block_size, float lr) {
  const int n = Q.shape[0];
  const int d = Q.shape[1];
  const int B = std::max(block_size, 1);
  const int nb = (n + B - 1) / B;
  if (nb < 2) return 0.0f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  const float* q = Q.data();
  const float* k = K.data();
  float* W = Wsel.data();
  const Tensor bm = compute_block_means(K, B, n, d);
  const float* bmd = bm.data();

  Tensor dW({d, d}, Device::CPU);  // zero-filled
  float* dw = dW.data();
  double loss = 0.0;
  long count = 0;

  std::vector<float> attn(static_cast<size_t>(n), 0.0f);
  std::vector<float> mass(static_cast<size_t>(nb), 0.0f);
  std::vector<float> score(static_cast<size_t>(nb), 0.0f);
  std::vector<float> pred(static_cast<size_t>(nb), 0.0f);
  std::vector<float> qsel(static_cast<size_t>(d), 0.0f);

  for (int i = 0; i < n; ++i) {
    const int ncand = i / B + 1;
    if (ncand < 2) continue;
    // TARGET: dense attention's per-block mass over j <= i.
    float mx = -std::numeric_limits<float>::infinity();
    for (int j = 0; j <= i; ++j) {
      float dv = 0.0f;
      for (int c = 0; c < d; ++c)
        dv += q[static_cast<size_t>(i) * d + c] * k[static_cast<size_t>(j) * d + c];
      attn[static_cast<size_t>(j)] = dv * scale;
      mx = std::max(mx, attn[static_cast<size_t>(j)]);
    }
    float sm = 0.0f;
    for (int j = 0; j <= i; ++j) {
      attn[static_cast<size_t>(j)] = std::exp(attn[static_cast<size_t>(j)] - mx);
      sm += attn[static_cast<size_t>(j)];
    }
    const float invsm = 1.0f / (sm + 1e-20f);
    for (int b = 0; b < ncand; ++b) mass[static_cast<size_t>(b)] = 0.0f;
    for (int j = 0; j <= i; ++j)
      mass[static_cast<size_t>(j / B)] += attn[static_cast<size_t>(j)] * invsm;
    // PREDICTED: softmax over candidate blocks of (Wsel @ q) . block_mean.
    for (int a = 0; a < d; ++a) {
      float acc = 0.0f;
      for (int c = 0; c < d; ++c)
        acc += W[static_cast<size_t>(a) * d + c] * q[static_cast<size_t>(i) * d + c];
      qsel[static_cast<size_t>(a)] = acc;
    }
    float smx = -std::numeric_limits<float>::infinity();
    for (int b = 0; b < ncand; ++b) {
      float sc = 0.0f;
      for (int c = 0; c < d; ++c)
        sc += qsel[static_cast<size_t>(c)] * bmd[static_cast<size_t>(b) * d + c];
      score[static_cast<size_t>(b)] = sc;
      smx = std::max(smx, sc);
    }
    float ssm = 0.0f;
    for (int b = 0; b < ncand; ++b) {
      pred[static_cast<size_t>(b)] = std::exp(score[static_cast<size_t>(b)] - smx);
      ssm += pred[static_cast<size_t>(b)];
    }
    const float invssm = 1.0f / (ssm + 1e-20f);
    for (int b = 0; b < ncand; ++b) {
      pred[static_cast<size_t>(b)] *= invssm;
      loss += -static_cast<double>(mass[static_cast<size_t>(b)]) *
              std::log(pred[static_cast<size_t>(b)] + 1e-20f);
    }
    ++count;
    // CE+softmax grad: dL/dscore[b] = pred[b] - mass[b].
    for (int a = 0; a < d; ++a) {
      float dqa = 0.0f;
      for (int b = 0; b < ncand; ++b)
        dqa += (pred[static_cast<size_t>(b)] - mass[static_cast<size_t>(b)]) *
               bmd[static_cast<size_t>(b) * d + a];
      for (int c = 0; c < d; ++c)
        dw[static_cast<size_t>(a) * d + c] += dqa * q[static_cast<size_t>(i) * d + c];
    }
  }
  if (count == 0) return 0.0f;
  const float invc = 1.0f / static_cast<float>(count);
  for (int i = 0; i < d * d; ++i) W[i] -= lr * dw[i] * invc;
  return static_cast<float>(loss / count);
}

}  // namespace nsos
