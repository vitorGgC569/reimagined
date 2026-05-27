#include "chrass_layer_v2.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace nsos;

ChrassLayer::ChrassLayer(int dimension, const std::vector<float> &adjacency)
    : dim(dimension), t(0), bias(Tensor::zeros({dimension}), "bias") {

  // Two-pass construction: first pass counts nnz + builds col_indices + row_ptr
  // and a temporary vector of weights.  Second pass copies into values_param
  // backing Tensor.  This avoids std::vector resize semantics on Parameter.
  std::vector<float> tmp_values;
  row_ptr.push_back(0);

  for (int r = 0; r < dim; ++r) {
    long double row_sum = 0.0;
    for (int c = 0; c < dim; ++c) {
      float val = adjacency[r * dim + c];
      if (std::abs(val) > 1e-6)
        row_sum += (long double)std::abs(val);
    }

    long double scale = (row_sum > 1e-12) ? (1.0 / row_sum) : 1.0;

    for (int c = 0; c < dim; ++c) {
      float val = adjacency[r * dim + c];
      if (std::abs(val) > 1e-6) {
        float safe_weight = (float)((long double)val * scale);
        tmp_values.push_back(safe_weight);
        col_indices.push_back(c);
        m.push_back(0.0f);
        v.push_back(0.0f);
      }
    }
    row_ptr.push_back(static_cast<int>(tmp_values.size()));
  }

  // Allocate Parameter-backed sparse weight Tensor [nnz].
  const int nnz_count = static_cast<int>(tmp_values.size());
  values_param = Parameter(Tensor::zeros({nnz_count}), "chrass_values");
  if (nnz_count > 0) {
    std::memcpy(values_param.data.data(), tmp_values.data(),
                static_cast<size_t>(nnz_count) * sizeof(float));
  }
}

// Compatibility helpers
float*       ChrassLayer::values_data()       { return values_param.data.data(); }
const float* ChrassLayer::values_data() const { return values_param.data.data(); }
int          ChrassLayer::nnz()         const { return values_param.data.size; }

std::vector<Parameter*> ChrassLayer::parameters() {
    return {&values_param, &bias};
}

std::vector<float> ChrassLayer::random_adjacency(int dim,
                                                 float density,
                                                 uint32_t seed) {
    std::vector<float> A(static_cast<size_t>(dim) * dim, 0.0f);
    if (dim <= 0 || density <= 0.0f) return A;
    if (density > 1.0f) density = 1.0f;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight(-1.0f, 1.0f);

    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            if (r == c) continue;  // exclude self-loops
            if (coin(rng) < density) {
                A[r * dim + c] = weight(rng);
            }
        }
    }
    return A;
}

Tensor ChrassLayer::forward(const Tensor &x) {
  int batch = x.shape[0];
  Tensor output = Tensor::zeros({batch, dim}, x.get_device());

  const float *in_ptr = x.data();
  float *out_ptr = output.data();
  const float *b_ptr = bias.data.data();
  const float *w_ptr = values_param.data.data();

// Parallelize batch
#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    const float *in_row = in_ptr + b * dim;
    float *out_row = out_ptr + b * dim;

    for (int r = 0; r < dim; ++r) {
      float sum = 0;
      int start = row_ptr[r];
      int end = row_ptr[r + 1];

      // CSR Loop (Linear Memory Access for Weights)
      for (int i = start; i < end; ++i) {
        sum += w_ptr[i] * in_row[col_indices[i]];
      }
      sum += b_ptr[r];

      // Aggressive sanitize for stability
      if (std::isnan(sum) || std::isinf(sum))
        sum = 0.0f;
      if (sum > 100.0f)
        sum = 100.0f;
      if (sum < -100.0f)
        sum = -100.0f;
      out_row[r] = sum;
    }
  }
  return output;
}

#include "chrass_layer_backward_v2.cpp"

void ChrassLayer::step(float lr) {
  // Standalone AdamW update.  When used inside JambaBlock, the central
  // Trainer drives updates via parameters() instead -- DO NOT call this.
  t++;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;

  float *w_ptr = values_param.data.data();
  const float *g_ptr = values_param.grad.size > 0
                          ? values_param.grad.data() : nullptr;
  int size = values_param.data.size;
  if (g_ptr == nullptr) {
    // No gradient buffer allocated yet -- nothing to apply.
    return;
  }

  // Bias correction
  float bc1 = 1.0f - std::pow(beta1, (float)t);
  float bc2 = 1.0f - std::pow(beta2, (float)t);

  for (int i = 0; i < size; ++i) {
    float g = g_ptr[i];
    m[i] = beta1 * m[i] + (1 - beta1) * g;
    v[i] = beta2 * v[i] + (1 - beta2) * g * g;

    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;

    w_ptr[i] -= lr * weight_decay * w_ptr[i];
    w_ptr[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

    if (std::isnan(w_ptr[i])) w_ptr[i] = 0.0f;
    if (w_ptr[i] > 10.0f)     w_ptr[i] = 10.0f;
    if (w_ptr[i] < -10.0f)    w_ptr[i] = -10.0f;
  }
}
