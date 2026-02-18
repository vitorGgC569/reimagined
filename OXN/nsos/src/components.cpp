#include "components.h"
#include <algorithm> // Fix: Needed for std::max
#include <cmath>
#include <iostream>

namespace nsos {

// -- SGDOptimizer --
SGDOptimizer::SGDOptimizer(float learning_rate) : lr(learning_rate) {}

void SGDOptimizer::step(Tensor &param, const Tensor &grad) {
  // p = p - lr * g
  // Using basic Tensor ops
  int sz = param.size;
  float *p_data = param.data();
  const float *g_data = grad.data();

  for (int i = 0; i < sz; ++i) {
    p_data[i] -= lr * g_data[i];
  }
}

// -- MuonOptimizer --
MuonOptimizer::MuonOptimizer(std::vector<int> shape, float learning_rate)
    : lr(learning_rate), momentum_decay(0.9f), momentum(shape, Device::CPU) {}

void MuonOptimizer::step(Tensor &param, const Tensor &grad) {
  // 1. Momentum Update: m = 0.9*m + 0.1*g
  float *m_ptr = momentum.data();
  const float *g_ptr = grad.data();
  int sz = param.size;

  for (int i = 0; i < sz; ++i) {
    m_ptr[i] = momentum_decay * m_ptr[i] + (1.0f - momentum_decay) * g_ptr[i];
  }

  // 2. Newton-Schulz on Momentum (Real Implementation)
  // If shape is 2D and square-ish.
  if (param.shape.size() == 2 && param.shape[0] > 1 && param.shape[1] > 1) {
    Tensor update = newton_schulz(momentum);

    // 3. Apply Update
    float *p_ptr = param.data();
    const float *u_ptr = update.data();
    for (int i = 0; i < sz; ++i) {
      p_ptr[i] -= lr * u_ptr[i];
    }
  } else {
    // Fallback: simple SGD with momentum for non-matrix params
    float *p_ptr = param.data();
    for (int i = 0; i < sz; ++i) {
      p_ptr[i] -= lr * m_ptr[i];
    }
  }
}

Tensor MuonOptimizer::newton_schulz(const Tensor &G) {
  // Real Newton-Schulz Iteration
  // X0 = G / (norm(G) + eps)
  // For k=1..Steps:
  //   X_{k+1} = 0.5 * X_k * (3I - X_k^T * X_k)
  // This orthogonalizes the matrix.

  int R = G.shape[0];
  int C = G.shape[1];

  // 1. Normalize (Spectral Norm approximation via Frobenius or just max?)
  // Standard Muon uses Frobenius or similar. Let's use Frobenius.
  float sum_sq = 0;
  const float *g_data = G.data();
  for (int i = 0; i < G.size; ++i)
    sum_sq += g_data[i] * g_data[i];
  float frob_norm = std::sqrt(sum_sq);
  if (frob_norm < 1e-6f)
    return G; // Zero gradient

  Tensor X = G; // Copy
  float *x_data = X.data();
  for (int i = 0; i < X.size; ++i)
    x_data[i] /= frob_norm;

  // 5 Iterations is standard for Muon
  int steps = 5;

  // Identity Matrix
  Tensor I = Tensor::eye(
      C, Device::CPU); // Assuming square or handling rectangular logic?
  // Muon logic usually assumes 2D weight matrices (R x C).
  // X^T * X results in C x C.
  // 3I is 3 * Identity(C).

  // If R != C, logic:
  // X_{k+1} = 0.5 * X * (3I - X^T X)
  // Dimensions: (R x C) * [ (C x C) - (C x R)*(R x C) ]
  //           = (R x C) * [ (C x C) - (C x C) ]
  //           = (R x C) * (C x C) -> (R x C). Correct.

  for (int k = 0; k < steps; ++k) {
    // A = X^T * X
    Tensor XT = X.transpose();
    Tensor A = XT.matmul(X); // [C, R] @ [R, C] -> [C, C]

    // B = 3I - A
    // A and I are C x C
    // We can do this element-wise
    Tensor B({C, C}, Device::CPU);
    float *b_ptr = B.data();
    const float *a_ptr = A.data();

    for (int i = 0; i < C * C; ++i) {
      float eye_val = (i % (C + 1) == 0) ? 1.0f : 0.0f; // Diagonal
      b_ptr[i] = 3.0f * eye_val - a_ptr[i];
    }

    // New X = 0.5 * X * B
    Tensor XB = X.matmul(B); // [R, C] @ [C, C] -> [R, C]

    float *xb_ptr = XB.data();
    for (int i = 0; i < XB.size; ++i) {
      x_data[i] = 0.5f * xb_ptr[i];
    }
  }

  // Scale back? Muon usually keeps it orthogonal (spectral radius 1).
  // So the update is this orthogonalized direction.
  return X;
}

void MuonOptimizer::step_and_quantize(Tensor &param, const Tensor &grad,
                                      Tensor &quantized_param) {
  step(param, grad);

  // Real Quantization (BitNet b1.58 style: Round to -1, 0, 1)
  // Also usually involves scaling. For now, simple ternary round.
  // q = clamp(round(w), -1, 1)

  int sz = param.size;
  const float *p_ptr = param.data();
  float *q_ptr = quantized_param.data();

  for (int i = 0; i < sz; ++i) {
    float val = p_ptr[i];
    float r = std::round(val);
    if (r > 1.0f)
      r = 1.0f;
    if (r < -1.0f)
      r = -1.0f;
    q_ptr[i] = r;
  }
}

// -- SophiaOptimizer --
SophiaOptimizer::SophiaOptimizer(std::vector<int> shape, float learning_rate)
    : lr(learning_rate), momentum(shape, Device::CPU),
      hessian_diag(shape, Device::CPU), beta1(0.9f), beta2(0.99f), rho(0.04f) {}

void SophiaOptimizer::step(Tensor &param, const Tensor &grad,
                           const Tensor &hessian_est) {
  // Sophia-G:
  int sz = param.size;
  float *m_ptr = momentum.data();
  float *h_ptr = hessian_diag.data();
  float *p_ptr = param.data();

  const float *g_ptr = grad.data();
  const float *he_ptr = hessian_est.data();

  for (int i = 0; i < sz; ++i) {
    // Momentum
    m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g_ptr[i];

    // Hessian (using absolute value of diagonal estimate)
    h_ptr[i] = beta2 * h_ptr[i] + (1.0f - beta2) * std::abs(he_ptr[i]);

    // Update
    float denom = std::max(h_ptr[i], rho);
    p_ptr[i] -= lr * m_ptr[i] / denom;
  }
}

// -- FOGZOOptimizer --
FOGZOOptimizer::FOGZOOptimizer(float mixing_factor, float perturbation)
    : alpha(mixing_factor), sigma(perturbation) {}

Tensor FOGZOOptimizer::calculate_update(const Tensor &param,
                                        const Tensor &grad_ste,
                                        const Tensor &perturbation,
                                        float loss_curr, float loss_plus,
                                        float loss_minus) {
  // Estimator: (loss_plus - loss_minus) / (2*sigma) * z

  // Gradient Estimate from ZO
  float g_zo_scalar = (loss_plus - loss_minus) / (2.0f * sigma);

  Tensor update = grad_ste; // Copy to init with STE part

  float *u_ptr = update.data();
  const float *ste_ptr = grad_ste.data();
  const float *z_ptr = perturbation.data();
  int sz = update.size;

  for (int i = 0; i < sz; ++i) {
    // g_combined = alpha * g_ste + (1 - alpha) * g_zo
    // g_zo = g_zo_scalar * z[i]

    float g_zo = g_zo_scalar * z_ptr[i];
    u_ptr[i] = alpha * ste_ptr[i] + (1.0f - alpha) * g_zo;
  }
  return update;
}

// -- MCTS Implementation Moved to mcts_reasoning.cpp --

} // namespace nsos
