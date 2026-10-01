#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace nsos::attention_rdna {

// Standalone, opt-in arithmetic ABI. Q/K/V FP32 master operands are rounded
// to the selected lowp type. QK uses wave32 rocWMMA with FP32 accumulation;
// probabilities, PV, dP and dQ/dK/dV contractions remain FP32 (no P/dS/dO
// rounding). The VJP treats the Q/K/V casts as straight-through operations,
// evaluated at rounded operands. This is NOT the derivative of a quantizer,
// nor parity with the FP32 attention policy. No dropout, bias, RoPE or cache.
inline constexpr const char* identity =
    "rdna3_wave32_qk_wmma_lowp_ste_fp32_pv_vjp_exp_owner_v1";
enum class Precision : int { BF16 = 1, FP16 = 2 };
enum class DeviceStatus : int { Ok = 0, InvalidPrefix = 1, InvalidOperand = 2, NonFiniteResult = 3 };

struct Shape {
    int batch = 0, sequence = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    int window = 0; // positive; includes the current token; >= sequence = full causal
    float scale = 0;
    Precision precision = Precision::BF16;
};

// Arithmetic eligibility is deliberately bounded. The device preflight also
// requires |Q|,|K|,|V|,|dO| <= 64 and finite, in valid prefixes only. Padded
// payloads can contain arbitrary values and are never consumed. A failed
// batch's tensor outputs for the current call are zeroed and status[batch]
// is marked without atomics; backward does not rewrite the forward tape/O.
inline bool geometry_eligible(const Shape& s) {
    if (s.batch <= 0 || s.batch > 65535 || s.sequence <= 0 ||
        s.sequence > 65535 * 16 || s.query_heads <= 0 || s.query_heads > 1024 ||
        s.kv_heads <= 0 || s.kv_heads > s.query_heads ||
        s.query_heads % s.kv_heads != 0 || s.head_dim <= 0 || s.head_dim > 256 ||
        s.window <= 0 || !std::isfinite(s.scale) || s.scale <= 0 || s.scale > 16 ||
        (s.precision != Precision::BF16 && s.precision != Precision::FP16)) return false;
    const std::uint64_t elements = static_cast<std::uint64_t>(s.batch) * s.sequence *
        s.query_heads * s.head_dim;
    return elements <= std::numeric_limits<std::size_t>::max() / sizeof(float);
}

inline std::size_t query_elements(const Shape& s) {
    return geometry_eligible(s) ? static_cast<std::size_t>(s.batch) * s.sequence *
        s.query_heads * s.head_dim : 0;
}
inline std::size_t kv_elements(const Shape& s) {
    return geometry_eligible(s) ? static_cast<std::size_t>(s.batch) * s.sequence *
        s.kv_heads * s.head_dim : 0;
}
inline std::size_t row_elements(const Shape& s) {
    return geometry_eligible(s) ? query_elements(s) / s.head_dim : 0;
}

// Checks the current device, this binary's architecture coverage, wave size,
// LDS capacity, and the actual instantiated function attributes. No device
// selection, scalar fallback, launches, allocation, or synchronization.
bool supported(const Shape& shape);

// All tensors contiguous: Q/O/dO/dQ [B,S,H,D], K/V/dK/dV [B,S,KV,D].
// Stats/delta [B,S,H]; valid/status [B]. Tape owns max/inverse_sum/status,
// rounded-operand source lifetimes, precision/shape and stream/device identity.
// valid is immutable device metadata with values in [0,S]. Forward writes all
// outputs, both stats, and status. Backward preserves forward status failures,
// additionally validates dO, writes delta and overwrites all three gradients.
// Caller owns allocations/capacities; no global S*S storage or hidden scratch.
// Written ranges must not overlap any read range or each other. Backward's
// status is the one explicitly permitted in/out range. No output accumulation.
// true means enqueued, NOT numeric success: asynchronously inspect status via
// the trainer's finite gate before publishing outputs/gradients. On false,
// launch failure may have queued work; discard tape/results and drain its lane.
// Inputs/stats must stay immutable through backward completion. Run both calls
// in the same current_stream() lane, or establish explicit event dependencies.
bool forward(const Shape& shape, const float* q, const float* k, const float* v,
             const int* valid, float* out, float* row_max, float* row_inverse_sum,
             int* status);
bool backward(const Shape& shape, const float* q, const float* k, const float* v,
              const float* out, const float* grad_out, const int* valid,
              const float* row_max, const float* row_inverse_sum, int* status,
              float* delta, float* dq, float* dk, float* dv);

} // namespace nsos::attention_rdna

// ATTENTION OWNED TRAINING INTEGRATION v1
// Included by cuda/attention_rdna_training.cuh; internal integration launchers.
namespace nsos::attention_rdna {
bool prepare_projected(const Shape&, int start, int table_rows,
    const float* q, const float* kv, const float* cosine, const float* sine,
    const int* valid, float* qr, float* kr, float* v);
bool join_projected_gradient(const Shape&, int start, int table_rows,
    const float* dq, const float* dk, const float* dv,
    const float* cosine, const float* sine, const int* valid, const int* status,
    float* q, float* kv);
bool mask_prefix(const Shape&, int features, const int* valid, float* tensor);
bool inspect_result(const Shape&, int features, const int* valid,
    const float* tensor, int* status);
bool merge_device_status(const int* status, int count, int* issue);
}
