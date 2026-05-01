#include "../include/sprecher_kan.h"

#include <cmath>
#include <stdexcept>

namespace nsos {

namespace {

float clamp_unit(float x) { return std::max(-1.0f, std::min(1.0f, x)); }

float safe_denominator(float value) {
  return std::fabs(value) < 1e-6f ? 1e-6f : value;
}

float basis_rbf(float x, float center, float width) {
  const float scaled = (x - center) / safe_denominator(width);
  return std::exp(-0.5f * scaled * scaled);
}

float basis_rbf_derivative(float x, float center, float width) {
  const float safe_width = safe_denominator(width);
  const float scaled = (x - center) / safe_width;
  return -scaled / safe_width * std::exp(-0.5f * scaled * scaled);
}

float basis_triangle(float x, float center, float width) {
  const float distance = std::fabs(x - center);
  if (distance >= width) {
    return 0.0f;
  }
  return 1.0f - distance / safe_denominator(width);
}

float basis_triangle_derivative(float x, float center, float width) {
  const float distance = std::fabs(x - center);
  if (distance >= width || distance < 1e-6f) {
    return 0.0f;
  }
  const float sign = x > center ? -1.0f : 1.0f;
  return sign / safe_denominator(width);
}

float chebyshev_t(int order, float x) {
  if (order == 0) {
    return 1.0f;
  }
  if (order == 1) {
    return x;
  }

  float t0 = 1.0f;
  float t1 = x;
  for (int i = 2; i <= order; ++i) {
    const float next = 2.0f * x * t1 - t0;
    t0 = t1;
    t1 = next;
  }
  return t1;
}

float chebyshev_u(int order, float x) {
  if (order <= 0) {
    return 1.0f;
  }

  float u0 = 1.0f;
  float u1 = 2.0f * x;
  if (order == 1) {
    return u1;
  }

  for (int i = 2; i <= order; ++i) {
    const float next = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = next;
  }
  return u1;
}

} // namespace

SprecherKAN::SprecherKAN(int in_features, int out_features, int hidden_dim,
                         ActivationType act_type,
                         const GridConfig &grid_config)
    : in_features_(in_features), out_features_(out_features),
      hidden_dim_(hidden_dim), act_type_(act_type), grid_config_(grid_config),
      current_grid_size_(std::max(2, grid_config.grid_size)),
      lambda_(Tensor::kaiming_uniform({in_features, hidden_dim}, Device::CPU),
              "sprecher_lambda"),
      eta_(Tensor::zeros({hidden_dim}, Device::CPU), "sprecher_eta"),
      psi_coefs_(Tensor::random({current_grid_size_}, Device::CPU),
                 "sprecher_psi_coefs"),
      phi_weights_(
          Tensor::xavier_uniform({hidden_dim, out_features}, Device::CPU),
          "sprecher_phi_weights") {
  if (in_features_ <= 0 || out_features_ <= 0 || hidden_dim_ <= 0) {
    throw std::invalid_argument(
        "SprecherKAN dimensions must all be strictly positive");
  }

  initialize_parameters();
  update_grids();
}

void SprecherKAN::initialize_parameters() {
  const float psi_scale = 1.0f / std::sqrt(static_cast<float>(current_grid_size_));
  for (int i = 0; i < psi_coefs_.data.size; ++i) {
    psi_coefs_.data.data()[i] *= psi_scale;
  }

  lambda_.grad = Tensor::zeros(lambda_.data.shape, lambda_.data.device);
  eta_.grad = Tensor::zeros(eta_.data.shape, eta_.data.device);
  psi_coefs_.grad = Tensor::zeros(psi_coefs_.data.shape, psi_coefs_.data.device);
  phi_weights_.grad =
      Tensor::zeros(phi_weights_.data.shape, phi_weights_.data.device);
}

void SprecherKAN::update_grids() {
  grid_config_.grid_size = current_grid_size_;

  switch (act_type_) {
  case ActivationType::CHEBYSHEV:
    grid_points_ = grid_config_.compute_chebyshev_grid();
    break;
  case ActivationType::RBF:
  case ActivationType::B_SPLINE:
  case ActivationType::LEGENDRE:
  case ActivationType::SPLINE_CONV:
  default:
    grid_points_ = grid_config_.compute_uniform_grid();
    break;
  }

  grid_deltas_.assign(current_grid_size_, 0.0f);
  for (int i = 0; i < current_grid_size_; ++i) {
    if (current_grid_size_ == 1) {
      grid_deltas_[i] = grid_config_.grid_range;
    } else if (i == 0) {
      grid_deltas_[i] = std::fabs(grid_points_[1] - grid_points_[0]);
    } else if (i == current_grid_size_ - 1) {
      grid_deltas_[i] =
          std::fabs(grid_points_[current_grid_size_ - 1] -
                    grid_points_[current_grid_size_ - 2]);
    } else {
      grid_deltas_[i] =
          0.5f * std::fabs(grid_points_[i + 1] - grid_points_[i - 1]);
    }
  }
}

Tensor SprecherKAN::eval_b_spline(const Tensor &u) {
  Tensor out = Tensor::zeros(u.shape.dims, u.device);

  for (int idx = 0; idx < u.size; ++idx) {
    float value = 0.0f;
    for (int g = 0; g < current_grid_size_; ++g) {
      value += psi_coefs_.data.data()[g] *
               basis_triangle(u.data()[idx], grid_points_[g],
                              std::max(grid_deltas_[g], grid_config_.epsilon));
    }
    out.data()[idx] = value;
  }

  return out;
}

Tensor SprecherKAN::eval_chebyshev(const Tensor &u) {
  Tensor out = Tensor::zeros(u.shape.dims, u.device);

  for (int idx = 0; idx < u.size; ++idx) {
    const float normalized = clamp_unit(
        u.data()[idx] / safe_denominator(grid_config_.grid_range));
    float value = 0.0f;
    for (int g = 0; g < current_grid_size_; ++g) {
      value += psi_coefs_.data.data()[g] * chebyshev_t(g, normalized);
    }
    out.data()[idx] = value;
  }

  return out;
}

Tensor SprecherKAN::eval_rbf(const Tensor &u) {
  Tensor out = Tensor::zeros(u.shape.dims, u.device);

  for (int idx = 0; idx < u.size; ++idx) {
    float value = 0.0f;
    for (int g = 0; g < current_grid_size_; ++g) {
      value += psi_coefs_.data.data()[g] *
               basis_rbf(u.data()[idx], grid_points_[g],
                         std::max(grid_deltas_[g], grid_config_.epsilon));
    }
    out.data()[idx] = value;
  }

  return out;
}

Tensor SprecherKAN::eval_psi_shared(const Tensor &u) {
  switch (act_type_) {
  case ActivationType::CHEBYSHEV:
    return eval_chebyshev(u);
  case ActivationType::RBF:
    return eval_rbf(u);
  case ActivationType::LEGENDRE:
  case ActivationType::SPLINE_CONV:
  case ActivationType::B_SPLINE:
  default:
    return eval_b_spline(u);
  }
}

Tensor SprecherKAN::forward(const Tensor &x) {
  if (x.shape.size() != 2 || x.shape[1] != in_features_) {
    throw std::invalid_argument(
        "SprecherKAN::forward expects a [batch, in_features] tensor");
  }

  saved_input_ = x.clone();
  saved_act_scales_.assign(x.shape[0] * hidden_dim_, 0.0f);

  Tensor u = Tensor::zeros({x.shape[0], hidden_dim_}, x.device);
  for (int batch = 0; batch < x.shape[0]; ++batch) {
    for (int hidden = 0; hidden < hidden_dim_; ++hidden) {
      float value = eta_.data.data()[hidden];
      for (int feature = 0; feature < in_features_; ++feature) {
        value += x.data()[batch * in_features_ + feature] *
                 lambda_.data.data()[feature * hidden_dim_ + hidden];
      }
      u.data()[batch * hidden_dim_ + hidden] = value;
      saved_act_scales_[batch * hidden_dim_ + hidden] = value;
    }
  }

  saved_psi_u_ = eval_psi_shared(u);

  Tensor output = Tensor::zeros({x.shape[0], out_features_}, x.device);
  for (int batch = 0; batch < x.shape[0]; ++batch) {
    for (int out_feature = 0; out_feature < out_features_; ++out_feature) {
      float value = 0.0f;
      for (int hidden = 0; hidden < hidden_dim_; ++hidden) {
        value += saved_psi_u_.data()[batch * hidden_dim_ + hidden] *
                 phi_weights_.data.data()[hidden * out_features_ + out_feature];
      }
      output.data()[batch * out_features_ + out_feature] = value;
    }
  }

  return output;
}

Tensor SprecherKAN::backward(const Tensor &grad_output) {
  if (saved_input_.size == 0 || saved_psi_u_.size == 0) {
    throw std::runtime_error(
        "SprecherKAN::backward called before a valid forward pass");
  }
  if (grad_output.shape.size() != 2 || grad_output.shape[1] != out_features_) {
    throw std::invalid_argument(
        "SprecherKAN::backward expects a [batch, out_features] gradient tensor");
  }

  Tensor grad_input =
      Tensor::zeros({grad_output.shape[0], in_features_}, grad_output.device);

  for (int batch = 0; batch < grad_output.shape[0]; ++batch) {
    for (int hidden = 0; hidden < hidden_dim_; ++hidden) {
      float grad_hidden = 0.0f;

      for (int out_feature = 0; out_feature < out_features_; ++out_feature) {
        const float upstream =
            grad_output.data()[batch * out_features_ + out_feature];
        grad_hidden +=
            upstream *
            phi_weights_.data.data()[hidden * out_features_ + out_feature];
        phi_weights_.grad.data()[hidden * out_features_ + out_feature] +=
            saved_psi_u_.data()[batch * hidden_dim_ + hidden] * upstream;
      }

      const float u_value = saved_act_scales_[batch * hidden_dim_ + hidden];
      float dpsi_du = 0.0f;

      for (int g = 0; g < current_grid_size_; ++g) {
        float basis = 0.0f;
        float dbasis_du = 0.0f;

        switch (act_type_) {
        case ActivationType::CHEBYSHEV: {
          const float normalized = clamp_unit(
              u_value / safe_denominator(grid_config_.grid_range));
          const float scale = 1.0f / safe_denominator(grid_config_.grid_range);
          basis = chebyshev_t(g, normalized);
          dbasis_du = g == 0 ? 0.0f : g * chebyshev_u(g - 1, normalized) * scale;
          break;
        }
        case ActivationType::RBF:
          basis = basis_rbf(u_value, grid_points_[g],
                            std::max(grid_deltas_[g], grid_config_.epsilon));
          dbasis_du =
              basis_rbf_derivative(u_value, grid_points_[g],
                                   std::max(grid_deltas_[g], grid_config_.epsilon));
          break;
        case ActivationType::LEGENDRE:
        case ActivationType::SPLINE_CONV:
        case ActivationType::B_SPLINE:
        default:
          basis = basis_triangle(u_value, grid_points_[g],
                                 std::max(grid_deltas_[g], grid_config_.epsilon));
          dbasis_du = basis_triangle_derivative(
              u_value, grid_points_[g],
              std::max(grid_deltas_[g], grid_config_.epsilon));
          break;
        }

        psi_coefs_.grad.data()[g] += grad_hidden * basis;
        dpsi_du += psi_coefs_.data.data()[g] * dbasis_du;
      }

      const float grad_u = grad_hidden * dpsi_du;
      eta_.grad.data()[hidden] += grad_u;

      for (int feature = 0; feature < in_features_; ++feature) {
        lambda_.grad.data()[feature * hidden_dim_ + hidden] +=
            saved_input_.data()[batch * in_features_ + feature] * grad_u;
        grad_input.data()[batch * in_features_ + feature] +=
            lambda_.data.data()[feature * hidden_dim_ + hidden] * grad_u;
      }
    }
  }

  return grad_input;
}

void SprecherKAN::extend_grid() {
  if (current_grid_size_ >= grid_config_.max_grid_size) {
    return;
  }

  const int previous_grid_size = current_grid_size_;
  const Tensor previous_coefs = psi_coefs_.data.clone();

  current_grid_size_ = std::min(grid_config_.max_grid_size, current_grid_size_ + 1);
  update_grids();

  psi_coefs_.data = Tensor::zeros({current_grid_size_}, previous_coefs.device);
  psi_coefs_.grad = Tensor::zeros({current_grid_size_}, previous_coefs.device);

  for (int i = 0; i < current_grid_size_; ++i) {
    const float mapped_position =
        previous_grid_size == 1
            ? 0.0f
            : static_cast<float>(i) * static_cast<float>(previous_grid_size - 1) /
                  static_cast<float>(current_grid_size_ - 1);
    const int left = static_cast<int>(std::floor(mapped_position));
    const int right = std::min(previous_grid_size - 1, left + 1);
    const float alpha = mapped_position - static_cast<float>(left);
    const float left_value = previous_coefs.data()[left];
    const float right_value = previous_coefs.data()[right];
    psi_coefs_.data.data()[i] = left_value * (1.0f - alpha) + right_value * alpha;
  }
}

void SprecherKAN::to(Device dev) {
  lambda_.data = lambda_.data.to(dev);
  eta_.data = eta_.data.to(dev);
  psi_coefs_.data = psi_coefs_.data.to(dev);
  phi_weights_.data = phi_weights_.data.to(dev);

  if (lambda_.grad.size > 0) {
    lambda_.grad = lambda_.grad.to(dev);
  }
  if (eta_.grad.size > 0) {
    eta_.grad = eta_.grad.to(dev);
  }
  if (psi_coefs_.grad.size > 0) {
    psi_coefs_.grad = psi_coefs_.grad.to(dev);
  }
  if (phi_weights_.grad.size > 0) {
    phi_weights_.grad = phi_weights_.grad.to(dev);
  }

  if (saved_input_.size > 0) {
    saved_input_ = saved_input_.to(dev);
  }
  if (saved_psi_u_.size > 0) {
    saved_psi_u_ = saved_psi_u_.to(dev);
  }
}

std::vector<Parameter *> SprecherKAN::parameters() {
  return {&lambda_, &eta_, &psi_coefs_, &phi_weights_};
}

} // namespace nsos
