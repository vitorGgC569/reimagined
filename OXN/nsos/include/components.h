#ifndef COMPONENT_H
#define COMPONENT_H

#include "tensor.h"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace nsos {

class JambaModel; // Forward declaration

// -- Optimizer: SGD (For Phase 0 Stability) --
class SGDOptimizer {
public:
  float lr;
  SGDOptimizer(float learning_rate = 0.01f);
  void step(Tensor &param, const Tensor &grad);
};

// -- Optimizer: Muon (2D Matrix / Spectral) --
class MuonOptimizer {
public:
  Tensor momentum;
  float lr;
  float momentum_decay;

  MuonOptimizer(std::vector<int> shape, float learning_rate = 0.01f);
  void step(Tensor &param, const Tensor &grad);

  Tensor newton_schulz(const Tensor &G);

  // Fused Update + Quantize
  // Simulates "BitLinear-Muon" kernel fusion
  void step_and_quantize(Tensor &param, const Tensor &grad,
                         Tensor &quantized_param);
};

// -- Optimizer: Sophia (1D Vector / 2nd Order Clipped) --
class SophiaOptimizer {
public:
  Tensor momentum;     // 1st moment
  Tensor hessian_diag; // 2nd moment (diagonal estimator)
  float lr;
  float beta1; // Momentum decay
  float beta2; // Hessian decay
  float rho;   // Hessian constant

  SophiaOptimizer(std::vector<int> shape, float learning_rate = 0.001f);

  // Needs 'hessian_est' which can be roughly grad^2 in a simplified setup
  // or provided externally.
  void step(Tensor &param, const Tensor &grad, const Tensor &hessian_est);
};

// -- Optimizer: FOGZO (First-Order-Guided Zeroth-Order) --
// Used for QAT where gradients are ill-defined (STE).
class FOGZOOptimizer {
public:
  float alpha; // Mixing factor (0.5 = equal mix)
  float sigma; // Perturbation scale

  FOGZOOptimizer(float mixing_factor = 0.5f, float perturbation = 0.001f);

  // This is a simplified "update rule" logic.
  // In a real ZO optimizer, you run the model twice.
  // Here we assume the user calculated 'loss_plus' and 'loss_minus' by
  // perturbing the weight. Requires the perturbation vector used (z).
  Tensor calculate_update(const Tensor &param, const Tensor &grad_ste,
                          const Tensor &perturbation, float loss_curr,
                          float loss_plus, float loss_minus);
};

// -- Inference: MCTS (Moved to mcts_reasoning.h) --

} // namespace nsos

#endif
