#include "chrass_layer_v2.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nsos;

ChrassLayer::ChrassLayer(int dimension, const std::vector<float> &adjacency)
    : dim(dimension), t(0), bias(Tensor::zeros({dimension}), "bias") {

  // CSR Construction
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
        values.push_back(safe_weight);
        col_indices.push_back(c);
        m.push_back(0.0f);
        v.push_back(0.0f);
        grad_values.push_back(0.0f);
      }
    }
    row_ptr.push_back(values.size());
  }
}

Tensor ChrassLayer::forward(const Tensor &x) {
  int batch = x.shape[0];
  Tensor output = Tensor::zeros({batch, dim}, x.get_device());

  const float *in_ptr = x.data();
  float *out_ptr = output.data();
  const float *b_ptr = bias.data.data();

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
        sum += values[i] * in_row[col_indices[i]];
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
  t++;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;

  // Sparse AdamW
  // Iterate flat sparse array
  int size = values.size();

  // Bias correction
  float bc1 = 1.0f - std::pow(beta1, t);
  float bc2 = 1.0f - std::pow(beta2, t);

  for (int i = 0; i < size; ++i) {
    float g = grad_values[i];

    // Update moments
    m[i] = beta1 * m[i] + (1 - beta1) * g;
    v[i] = beta2 * v[i] + (1 - beta2) * g * g;

    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;

    // Apply Weight Decay
    values[i] -= lr * weight_decay * values[i];

    // Update Weight
    values[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

    // Sanitize weight
    if (std::isnan(values[i]))
      values[i] = 0.0f;
    if (values[i] > 10.0f)
      values[i] = 10.0f;
    if (values[i] < -10.0f)
      values[i] = -10.0f;
  }
}
