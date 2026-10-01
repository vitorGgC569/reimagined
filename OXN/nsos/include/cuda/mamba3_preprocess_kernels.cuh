#pragma once
#include "mamba3_siso_kernels.cuh"

namespace nsos::mamba3_preprocess_gpu {
using Shape=mamba3_siso_gpu::Shape;
inline constexpr const char* identity="rdna3_fp32_bcnorm_fp64_radial_vjp_heavy_dt_broadcast_v2";
struct Config {float norm_eps=1e-5f,a_floor=1e-4f;};
inline bool eligible(const Shape& s,Config c) {
    return mamba3_siso_gpu::eligible(s)&&std::isfinite(c.norm_eps)&&c.norm_eps>=1e-12f&&c.norm_eps<=1&&
        std::isfinite(c.a_floor)&&c.a_floor>0&&c.a_floor<=64;
}
inline std::size_t partitions(const Shape& s) {
    return mamba3_siso_gpu::eligible(s)?(static_cast<std::size_t>(s.batch)*s.sequence*s.groups+127)/128:0;
}
inline std::size_t partial_elements(const Shape& s) {
    // Norm weight reductions partition over rows; dt_bias over token rows.
    return mamba3_siso_gpu::eligible(s)?partitions(s)*2*s.state_dim+
        ((static_cast<std::size_t>(s.batch)*s.sequence+127)/128)*s.heads:0;
}
struct Input {
    const float *q=nullptr,*k=nullptr; // [B,S,G,N], raw BC
    const float *raw_a=nullptr,*raw_dt=nullptr; // [B,S,H]
    const float *angles=nullptr; // [B,S,R], BEFORE pi*tanh
    const float *q_norm=nullptr,*k_norm=nullptr; // [N]
    const float *dt_bias=nullptr; // [H]
    const int* valid=nullptr; // [B] immutable prefixes
    const float* inverse=nullptr; // backward only: [B,S,G,2], saved Q/K inverse
};
struct Prepared {float *q=nullptr,*k=nullptr,*adt=nullptr,*dt=nullptr,*angles=nullptr,*inverse=nullptr;};
struct Adjoint {const float *q=nullptr,*k=nullptr,*adt=nullptr,*dt=nullptr,*angles=nullptr;};
struct Gradient {
    float *q=nullptr,*k=nullptr,*raw_a=nullptr,*raw_dt=nullptr,*angles=nullptr,
        *q_norm=nullptr,*k_norm=nullptr,*dt_bias=nullptr;
};
// Contiguous FP32 selected-device storage, exact capacities per shapes.
// All written ranges disjoint from reads and one another; caller owns lifetime.
// No device allocation/readback/fence. true means enqueue, not numerical success.
// Required status[B]: forward initializes; backward preserves sticky failure.
// Raw operands/weights/biases |x|<=64; derived Q/K<=64, DT<=16, ADT<=0.
// Invalid prefix=1, invalid range=2, nonfinite computation=3. No silent clamps
// except the declared A floor with zero subgradient at equality. Padding unread.
bool forward(Shape shape,Config config,Input input,Prepared output,int* status);
bool backward(Shape shape,Config config,Input input,Adjoint adjoint,
    Gradient output,float* partials,int* status);
// After preprocessing VJP, reject any whole-op shared gradient publication if
// one batch failed. Recurrence inputs/initial adjoints are additionally cleared
// on failed batches. Passing raw Gradient pointers requires the same validated
// capacities/alias/lifetime contract as backward; no buffers are created here.
bool gate_chain(Shape shape,const int* status,Gradient raw,
    mamba3_siso_gpu::Gradient recurrent,mamba3_siso_gpu::State initial_gradient);
}
