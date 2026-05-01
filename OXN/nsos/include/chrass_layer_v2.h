#ifndef CHRASS_LAYER_V2_H
#define CHRASS_LAYER_V2_H

#include "autograd.h"
#include "components.h"
#include "tensor.h"
#include <vector>


namespace nsos {

// -- ChrassLayer: Isomorphic Neural Layer --
// Concept: Weights are initialized and constrained by the Graph Laplacian.
// This forces the neural signal to flow only through valid physical
// connections.

class ChrassLayer {
public:
  int dim;
  // CSR Storage (V2)
  std::vector<float> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;

  // AdamW State
  std::vector<float> m;
  std::vector<float> v;
  int t; // Timestep for bias correction

  // Gradients for sparse weights
  std::vector<float> grad_values;

  Parameter bias; // [dim]

  // Initialize with a known topology (adjacency matrix)
  ChrassLayer(int dimension, const std::vector<float> &adjacency);

  // Sparse Forward: O(E) instead of O(N^2) with Output Clamping
  Tensor forward(const Tensor &x);

  // Manual Backward: Computes dL/dx AND dL/dValues
  Tensor backward(const Tensor &grad_output, const Tensor &input);

  // Sparse AdamW Update with Gradient Clipping
  void step(float lr);
};

} // namespace nsos

#endif
