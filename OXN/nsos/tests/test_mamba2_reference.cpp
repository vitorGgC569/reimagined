#include "../include/mamba2.h"
#include "../include/jamba.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nsos;

namespace {

Parameter* parameter(std::vector<Parameter*>& params, const std::string& suffix) {
  for (Parameter* p : params) {
    if (p && p->name.size() >= suffix.size() &&
        p->name.compare(p->name.size() - suffix.size(), suffix.size(), suffix) ==
            0) {
      return p;
    }
  }
  throw std::runtime_error("missing parameter: " + suffix);
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float softplus(float x) {
  return x > 20.0f ? x : (x < -20.0f ? std::exp(x)
                                      : std::log1p(std::exp(x)));
}

std::vector<float> linear(const std::vector<float>& input, int rows, int in,
                          const Tensor& weight, const Tensor* bias = nullptr) {
  const int out = weight.shape[0];
  std::vector<float> result(static_cast<size_t>(rows) * out, 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int o = 0; o < out; ++o) {
      float value = bias ? bias->data()[o] : 0.0f;
      for (int i = 0; i < in; ++i) {
        value += input[static_cast<size_t>(r) * in + i] *
                 weight.data()[static_cast<size_t>(o) * in + i];
      }
      result[static_cast<size_t>(r) * out + o] = value;
    }
  }
  return result;
}

}  // namespace

int main() {
  constexpr int D = 4;
  constexpr int N = 3;
  constexpr int I = D * 2;
  constexpr int P = 4;
  constexpr int H = I / P;
  constexpr int G = 1;
  constexpr int L = 5;
  constexpr int K = 3;
  constexpr int GS = G * N;
  constexpr int CD = I + 2 * GS;
  MambaConfig cfg;
  cfg.faithful_mamba2 = true;
  cfg.expand = 2;
  cfg.head_dim = P;
  cfg.n_groups = G;
  cfg.conv_kernel = K;
  cfg.rms_norm_eps = 1e-5f;
  Mamba2SSD layer(D, N, 1, cfg);
  auto params = layer.parameters();

  for (Parameter* p : params) {
    if (!p) continue;
    for (int i = 0; i < p->data.size; ++i) {
      p->data.data()[i] =
          0.08f * std::sin(0.19f * static_cast<float>(i + 1));
    }
  }
  Parameter* A = parameter(params, "A");
  Parameter* Dskip = parameter(params, "D");
  Parameter* norm = parameter(params, "norm.weight");
  Parameter* conv_b = parameter(params, "conv1d_bias");
  for (int h = 0; h < H; ++h) {
    A->data.data()[h] = std::log(1.0f + h);
    Dskip->data.data()[h] = 1.0f + 0.1f * h;
  }
  for (int i = 0; i < I; ++i) norm->data.data()[i] = 0.9f + 0.01f * i;
  for (int i = 0; i < CD; ++i) conv_b->data.data()[i] *= 0.2f;

  Tensor input({L, D});
  std::vector<float> u(static_cast<size_t>(L) * D);
  for (int i = 0; i < L * D; ++i) {
    u[static_cast<size_t>(i)] =
        0.35f * std::cos(0.23f * static_cast<float>(i + 1));
    input.data()[i] = u[static_cast<size_t>(i)];
  }

  const Tensor& wx = parameter(params, "x_proj.weight")->data;
  const Tensor& wz = parameter(params, "z_proj.weight")->data;
  const Tensor& wB = parameter(params, "B_proj.weight")->data;
  const Tensor& wC = parameter(params, "C_proj.weight")->data;
  const Tensor& wdt = parameter(params, "dt_proj.weight")->data;
  const Tensor& bdt = parameter(params, "dt_proj.bias")->data;
  const Tensor& wo = parameter(params, "out_proj.weight")->data;
  const Tensor& cw = parameter(params, "conv1d_weight")->data;

  std::vector<float> xv = linear(u, L, D, wx);
  std::vector<float> z = linear(u, L, D, wz);
  std::vector<float> Bv = linear(u, L, D, wB);
  std::vector<float> Cv = linear(u, L, D, wC);
  std::vector<float> dt = linear(u, L, D, wdt, &bdt);
  std::vector<float> xBC(static_cast<size_t>(L) * CD, 0.0f);
  for (int t = 0; t < L; ++t) {
    for (int c = 0; c < CD; ++c) {
      float value = conv_b->data.data()[c];
      for (int j = 0; j < K; ++j) {
        const int source_t = t - (K - 1) + j;
        if (source_t < 0) continue;
        const float source =
            c < I ? xv[static_cast<size_t>(source_t) * I + c]
                  : (c < I + GS
                         ? Bv[static_cast<size_t>(source_t) * GS + c - I]
                         : Cv[static_cast<size_t>(source_t) * GS + c - I -
                              GS]);
        value += cw.data()[static_cast<size_t>(c) * K + j] * source;
      }
      xBC[static_cast<size_t>(t) * CD + c] = silu(value);
    }
  }

  std::vector<float> state(static_cast<size_t>(I) * N, 0.0f);
  std::vector<float> normalized(static_cast<size_t>(L) * I, 0.0f);
  for (int t = 0; t < L; ++t) {
    std::vector<float> gated(I, 0.0f);
    for (int h = 0; h < H; ++h) {
      const int group = (h * G) / H;
      const float delta = softplus(dt[static_cast<size_t>(t) * H + h]);
      const float decay = std::exp(-delta * std::exp(A->data.data()[h]));
      for (int p = 0; p < P; ++p) {
        const int chan = h * P + p;
        const float x = xBC[static_cast<size_t>(t) * CD + chan];
        float y = Dskip->data.data()[h] * x;
        for (int n = 0; n < N; ++n) {
          const size_t si = static_cast<size_t>(chan) * N + n;
          const float B = xBC[static_cast<size_t>(t) * CD + I + group * N + n];
          const float C =
              xBC[static_cast<size_t>(t) * CD + I + GS + group * N + n];
          state[si] = decay * state[si] + delta * B * x;
          y += state[si] * C;
        }
        gated[chan] = y * silu(z[static_cast<size_t>(t) * I + chan]);
      }
    }
    float mean_sq = 0.0f;
    for (float value : gated) mean_sq += value * value;
    const float inv =
        1.0f / std::sqrt(mean_sq / static_cast<float>(I) + cfg.rms_norm_eps);
    for (int i = 0; i < I; ++i) {
      normalized[static_cast<size_t>(t) * I + i] =
          gated[i] * inv * norm->data.data()[i];
    }
  }
  const std::vector<float> expected = linear(normalized, L, I, wo);
  Tensor actual = layer.forward(input);
  float max_abs = 0.0f;
  for (int i = 0; i < actual.size; ++i) {
    max_abs =
        std::max(max_abs,
                 std::fabs(actual.data()[i] - expected[static_cast<size_t>(i)]));
  }
  if (max_abs > 2e-5f) {
    std::cerr << "Mamba2 reference parity failed: max_abs=" << max_abs
              << std::endl;
    return 1;
  }
  ModelConfig model_cfg;
  model_cfg.num_layers = 1;
  model_cfg.d_model = 8;
  model_cfg.vocab_size = 16;
  model_cfg.n_heads = 2;
  model_cfg.n_kv_heads = 1;
  model_cfg.attention_period = 99;
  model_cfg.use_moe = false;
  model_cfg.use_kan = false;
  model_cfg.use_chrass = false;
  model_cfg.mamba2_faithful = true;
  model_cfg.tie_word_embeddings = false;
  JambaModel faithful_stack(model_cfg, Device::CPU);
  bool found_layer_norm = false;
  bool found_final_norm = false;
  for (Parameter* p : faithful_stack.parameters()) {
    if (!p) continue;
    if (p->name.find("ffn_") != std::string::npos) {
      std::cerr << "Faithful Mamba stack unexpectedly contains FFN parameter: "
                << p->name << '\n';
      return 1;
    }
    found_layer_norm = found_layer_norm || p->name == "layers.0.norm.weight";
    found_final_norm = found_final_norm || p->name == "norm_f.weight";
  }
  if (!found_layer_norm || !found_final_norm) {
    std::cerr << "Faithful stack is missing learnable RMSNorm weights"
              << " (layer=" << found_layer_norm
              << ", final=" << found_final_norm << ")\n";
    return 1;
  }
  std::cout << "Mamba2 official-formula parity OK (max_abs=" << max_abs << ")\n";
  return 0;
}
