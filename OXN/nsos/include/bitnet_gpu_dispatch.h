#ifndef NSOS_BITNET_GPU_DISPATCH_H
#define NSOS_BITNET_GPU_DISPATCH_H

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace nsos {

// =====================================================================
// BitNet 1.58-bit / INT-N GPU end-to-end dispatch (Phase 5a).
//
// Pure utility: takes float activations + already-packed weights and
// returns the GEMM output, performing on the device:
//   1. Per-row activation quantization (float → int8 + scales)
//   2. __dp4a-accelerated 1.58-bit GEMM (existing launch_bitnet_gemm)
//   3. Per-row dequantization scaling
//
// This function does not own any state and does not modify the
// BitLinear public surface.  It is intended for inference-time use by
// callers that already maintain packed weights — typically the
// inference engine when serving an edge pack on GPU.  The default
// BitLinear::forward GPU path remains float matmul (Phase 5b will fold
// this dispatch into BitLinear::forward behind a feature flag).
//
// Inputs:
//   x_gpu          : float [M, K]  (device, contiguous)
//   packed_weights_gpu : uint32 [(N*K + 15)/16] (device, contiguous;
//                         layout matches BitNetAdapter::pack_weights_microsoft_style)
//   weight_scale   : float scalar
//   M, K, N        : matrix dimensions
//   precision_bits : selects q_max for activation quantization
//                    (2 → ternary, 8 → INT8)
//
// Returns:
//   y_gpu : float [M, N]  (device)
//
// Throws std::runtime_error when CUDA is not enabled at build time, when
// any input does not live on the GPU, or when shapes are inconsistent.
//
// Threading: launches on the default CUDA stream.  Caller must
// synchronize before reading results from the host.
// =====================================================================
Tensor bitnet_gemm_158bit_gpu(const Tensor& x_gpu,
                              const Tensor& packed_weights_gpu,
                              float weight_scale, int M, int K, int N,
                              int precision_bits);

}  // namespace nsos

#endif  // NSOS_BITNET_GPU_DISPATCH_H
