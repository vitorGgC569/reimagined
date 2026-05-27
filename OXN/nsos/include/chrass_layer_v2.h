#ifndef CHRASS_LAYER_V2_H
#define CHRASS_LAYER_V2_H

#include "autograd.h"
#include "components.h"
#include "tensor.h"
#include <cstdint>
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
  //
  // 2026-05-25 refactor: weights are now owned by `values_param`
  // (Parameter-backed Tensor) so the central Trainer can update them.
  // Backward writes gradients to `values_param.grad` directly (no more
  // private `grad_values` vector).  Standalone AdamW in step() still
  // works -- it operates on the same Parameter using internal m/v.
  Parameter values_param;          // [nnz] sparse weights as 1D Tensor
  std::vector<int> col_indices;
  std::vector<int> row_ptr;

  // AdamW State (used by standalone step(); ignored when external Trainer drives)
  std::vector<float> m;
  std::vector<float> v;
  int t; // Timestep for bias correction

  Parameter bias; // [dim]

  // Initialize with a known topology (adjacency matrix)
  ChrassLayer(int dimension, const std::vector<float> &adjacency);

  // Sparse Forward: O(E) instead of O(N^2) with Output Clamping
  Tensor forward(const Tensor &x);

  // Manual Backward: Computes dL/dx AND dL/dValues
  Tensor backward(const Tensor &grad_output, const Tensor &input);

  // Sparse AdamW Update with Gradient Clipping (standalone use only)
  void step(float lr);

  // Trainer integration: expose values_param + bias to central optimizer.
  // When this is used, do NOT call step() -- the Trainer handles updates.
  std::vector<Parameter*> parameters();

  // Helpers for compatibility with older code that read .values directly.
  // Returns a pointer to the underlying float buffer of values_param.data.
  float* values_data();
  const float* values_data() const;
  int nnz() const;

  // ────────────────────────────────────────────────────────────────────
  //  Static factory: build a random sparse adjacency matrix [dim*dim]
  //  with target `density` fraction of nonzero entries, deterministically
  //  seeded.  Self-loops excluded.  Weights sampled uniform from [-1,1].
  //
  //  Used by JambaBlock when ModelConfig.use_chrass is true:
  //     adj = ChrassLayer::random_adjacency(d_model,
  //                                         cfg.chrass_density,
  //                                         cfg.chrass_seed + layer_idx);
  //     chrass = std::make_unique<ChrassLayer>(d_model, adj);
  // ────────────────────────────────────────────────────────────────────
  static std::vector<float> random_adjacency(int dim,
                                             float density,
                                             uint32_t seed);
};

} // namespace nsos

#endif
