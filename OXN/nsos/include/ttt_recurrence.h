#pragma once

#include "tensor.h"

namespace nsos::ttt {

// Full differentiation within one sequence. Initial session state is a
// detached input; neither samples nor separate forward calls share an adjoint.
struct RecurrenceConfig {
    int hidden;
    int dim;
    int chunk = 32;
    float decay;
    float step;
    float max_norm;
    bool hamiltonian;
};

void forward_sequence(const Tensor& input, const Tensor& keys, const Tensor& base,
    Tensor& adaptation, Tensor& momentum, Tensor& boundaries, Tensor& momentum_boundaries,
    Tensor& errors, Tensor& output, int offset, int rows, int boundary_offset,
    const RecurrenceConfig& config);

void backward_sequence(const Tensor& keys, const Tensor& errors, const Tensor& boundaries,
    const Tensor& momentum_boundaries, const Tensor& grad, Tensor& grad_keys,
    Tensor& grad_base, Tensor& grad_direct, int offset, int rows, int boundary_offset,
    const RecurrenceConfig& config);

} // namespace nsos::ttt
