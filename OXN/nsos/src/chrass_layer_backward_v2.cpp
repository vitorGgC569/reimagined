#include "chrass_layer_v2.h"
#include <algorithm>
#include <cmath>
#include <vector>

using namespace nsos;

Tensor ChrassLayer::backward(const Tensor &grad_output, const Tensor &input) {
  // Contract Checks
  if (grad_output.shape.size() != 2) {
    throw std::runtime_error(
        "ChrassLayer::backward requires 2D grad_output [B, D]");
  }

  int batch = grad_output.shape[0];
  if (grad_output.shape[1] != dim) {
    throw std::runtime_error("ChrassLayer::backward grad dimension mismatch");
  }

  Tensor grad_input = Tensor::zeros({batch, dim}, grad_output.get_device());

  const float *gy_ptr = grad_output.data();
  const float *x_ptr = input.data();
  float *gx_ptr = grad_input.data();
  const float *w_ptr = values_param.data.data();
  const int nnz_count = values_param.data.size;

  // Prepare Bias Gradients (Accumulate over batch)
  Tensor d_bias = Tensor::zeros({dim}, bias.data.get_device());
  float *db_ptr = d_bias.data();

  // KNOWN FOLLOW-UP (CHRASS is off by default): unlike every other layer here,
  // this backward RESETS values_param.grad instead of accumulating, so a
  // multi-bucket optimizer step (Trainer runs several backwards between two
  // zero_grad calls) keeps only the LAST bucket's weight gradient.  Switching
  // to the codebase-wide accumulate-then-zero contract requires a coordinated
  // change (relocate the per-call grad clip below, make the standalone step()
  // zero the grad, and update the two contract tests in test_chrass_validation)
  // and is tracked separately.  The reset is preserved here for now to keep the
  // standalone step() loop and its gate tests correct.
  if (values_param.grad.size != nnz_count) {
    values_param.grad = Tensor::zeros({nnz_count}, bias.data.get_device());
  } else {
    std::memset(values_param.grad.data(), 0,
                static_cast<size_t>(nnz_count) * sizeof(float));
  }
  float *gw_ptr = values_param.grad.data();

  // Saturation mask: the forward clamps |out| > 100 and zeroes NaN/Inf, both of
  // which have zero local derivative.  Recompute the pre-clamp pre-activation so
  // the backward routes NO gradient through saturated/sanitized outputs (else
  // dL/dx, dL/dW and dL/db are wrong at clamped elements).
  const float *bias_ptr = bias.data.data();
  std::vector<float> active(static_cast<size_t>(batch) * static_cast<size_t>(dim),
                            1.0f);
#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    const float *in_row = x_ptr + b * dim;
    for (int r = 0; r < dim; ++r) {
      float sum = 0.0f;
      const int start = row_ptr[r];
      const int end = row_ptr[r + 1];
      for (int i = start; i < end; ++i) {
        sum += w_ptr[i] * in_row[col_indices[i]];
      }
      sum += bias_ptr[r];
      const bool saturated = std::isnan(sum) || std::isinf(sum) ||
                             sum > 100.0f || sum < -100.0f;
      active[static_cast<size_t>(b) * dim + r] = saturated ? 0.0f : 1.0f;
    }
  }

// Backward Pass:
// y = Wx + b
// dL/dx = W^T * dL/dy
// dL/dW = dL/dy * x^T
// dL/db = sum(dL/dy, axis=0)

// The matrix W is sparse CSR. W_{rc} exists if adjacency[r][c] != 0
// forward: out[r] += W_{rc} * in[c]

// 1. Compute dL/db and dL/dx and dL/dW in one pass?
// dL/dx requires transposing W. Since W is CSR (row-major),
// iterating by row allows computing dL/dW easily, but dL/dx requires
// scattering.

// Parallel over Batch
// Need atomic adds for dW and db if parallel? Yes.
// Or thread-local accumulation.
// Given the constraints, let's do serial accumulation for weights/bias to
// ensure determinism, or use OpenMP reduction if possible (complex with
// vectors). Let's stick to simple deterministic serial loop over batch for now,
// or parallel with thread-local buffers.

// Parallelizing over batches for dL/dx is safe (writes to distinct rows of gx).
// But dL/dW and dL/db conflict.

// Phase 1: Compute dL/dx (Parallelizable per batch)
#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    const float *gy_row = gy_ptr + b * dim;
    float *gx_row = gx_ptr + b * dim;

    // W^T * gy
    // W is CSR. W_{rc} connects input c to output r.
    // gx[c] += W_{rc} * gy[r]
    // We iterate rows r of W. For each non-zero col c:
    // gx[c] += value * gy[r]

    for (int r = 0; r < dim; ++r) {
      float g_val = gy_row[r] * active[static_cast<size_t>(b) * dim + r];
      // Skip if gradient is effectively zero (Sparse Backprop)
      if (std::abs(g_val) < 1e-9)
        continue;

      int start = row_ptr[r];
      int end = row_ptr[r + 1];

      for (int i = start; i < end; ++i) {
        int c = col_indices[i];
        float w = w_ptr[i];
        gx_row[c] += w * g_val;
      }
    }
  }

  // Phase 2: Compute dL/dW and dL/db (Accumulate across batch)
  // Serial or Reduction
  for (int b = 0; b < batch; ++b) {
    const float *gy_row = gy_ptr + b * dim;
    const float *x_row = x_ptr + b * dim;

    for (int r = 0; r < dim; ++r) {
      float grad = gy_row[r] * active[static_cast<size_t>(b) * dim + r];

      // Bias Grad
      db_ptr[r] += grad;

      // Weight Grad -> values_param.grad
      int start = row_ptr[r];
      int end = row_ptr[r + 1];
      for (int i = start; i < end; ++i) {
        int c = col_indices[i];
        // dL/dW_{rc} = gy[r] * x[c]
        gw_ptr[i] += grad * x_row[c];
      }
    }
  }

  // Gradient Clipping for Weights (Stability)
  // Value-clip in-place on values_param.grad
  for (int i = 0; i < nnz_count; ++i) {
    if (gw_ptr[i] > 1.0f) gw_ptr[i] = 1.0f;
    if (gw_ptr[i] < -1.0f) gw_ptr[i] = -1.0f;
  }

  // Update Bias Grad
  bias.add_grad(d_bias);

  return grad_input;
}
