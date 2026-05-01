#ifndef SPRECHER_KAN_SOTA_H
#define SPRECHER_KAN_SOTA_H

#include "autograd.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace nsos {

constexpr float kSprecherPi = 3.14159265358979323846f;

/**
 * Grid Configuration para Grid Extension (EfficientKAN 2025)
 * Permite refinamento adaptativo de splines durante treino
 */
struct GridConfig {
  int grid_size = 5;      // Tamanho inicial (SOTA: começar pequeno)
  int max_grid_size = 20; // Limite superior para extension
  float grid_range = 1.0f;
  float epsilon = 0.01f;

  std::vector<float> compute_chebyshev_grid() const {
    std::vector<float> grid(grid_size);
    for (int i = 0; i < grid_size; ++i) {
      float theta = kSprecherPi * (2.0f * i + 1.0f) / (2.0f * grid_size);
      grid[i] = -grid_range * std::cos(theta);
    }
    return grid;
  }

  std::vector<float> compute_uniform_grid() const {
    std::vector<float> grid(grid_size);
    for (int i = 0; i < grid_size; ++i) {
      grid[i] = 2.0f * grid_range * i / (grid_size - 1) - grid_range;
    }
    return grid;
  }
};

enum class ActivationType { RBF, B_SPLINE, CHEBYSHEV, LEGENDRE, SPLINE_CONV };

/**
 * IMPLEMENTAÇÃO CORRIGIDA DO TEOREMA DE SPRECHER SOTA 2025-2026
 */
class SprecherKAN {
public:
  SprecherKAN(int in_features, int out_features, int hidden_dim,
              ActivationType act_type = ActivationType::B_SPLINE,
              const GridConfig &grid_config = GridConfig());

  Tensor forward(const Tensor &x);
  Tensor backward(const Tensor &grad_output);

  void extend_grid();
  void to(Device dev);
  std::vector<Parameter *> parameters();

  float get_grid_resolution() const { return (float)current_grid_size_; }
  ActivationType get_activation_type() const { return act_type_; }

private:
  const int in_features_;
  const int out_features_;
  const int hidden_dim_;

  ActivationType act_type_;
  GridConfig grid_config_;
  int current_grid_size_;

  // Parâmetros treináveis (SOTA)
  Parameter lambda_;      // [in_features, hidden_dim]
  Parameter eta_;         // [hidden_dim]
  Parameter psi_coefs_;   // [grid_size] (SHARED!)
  Parameter phi_weights_; // [hidden_dim, out_features]

  // Cache para backward
  Tensor saved_input_;
  Tensor saved_psi_u_; // Cache da ativação inner
  std::vector<float> saved_act_scales_;

  // Grids
  std::vector<float> grid_points_;
  std::vector<float> grid_deltas_;

  void initialize_parameters();
  void update_grids();
  Tensor eval_psi_shared(const Tensor &u);
  Tensor eval_b_spline(const Tensor &u);
  Tensor eval_chebyshev(const Tensor &u);
  Tensor eval_rbf(const Tensor &u);
};

} // namespace nsos

#endif
