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

// =====================================================================
// K3: device-aware QAT fake-quant (straight-through estimator) helpers.
// These let BitLinear run TRUE quantization-aware training on the GPU
// (the dp4a inference path is non-differentiable; the float GPU path did
// no quantization at all).  Each helper dispatches to a CUDA kernel when
// the input is on the GPU and to an identical-math host loop on the CPU,
// so the QAT forward/backward is device-agnostic at the call site and
// CPU↔GPU numerics match.
// =====================================================================

// Ternary weight fake-quant: returns clamp(round(w/scale), -1, +1) * scale
// (same rule as BitLinear::quantize_weights, scaled).  Same shape/device as w.
Tensor qat_fake_quant_ternary(const Tensor& w, float scale);

// Per-row activation fake-quant (quant→dequant): returns the dequantized
// float activations the matmul should multiply.  precision_bits selects q_max
// (2 → ±1 ternary, else 2^(b-1)-1).  x is [M, K]; output matches shape/device.
Tensor qat_fake_quant_activations(const Tensor& x, int precision_bits);

// STE clip applied in place to the weight gradient: zeros entries whose latent
// weight already saturated past the ternary band (|w/scale| > 1).
void qat_ste_clip_weight_grad(Tensor& dW, const Tensor& w, float scale);

}  // namespace nsos

#endif  // NSOS_BITNET_GPU_DISPATCH_H
