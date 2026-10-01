#pragma once
#include <cmath>
#include <cstddef>
#include <limits>

namespace nsos::mamba3_siso_gpu {
inline constexpr const char* identity="rdna3_fp32_siso_boundary_replay_owner_v2";
struct Shape {
    int batch=0,sequence=0,heads=0,groups=0,head_dim=0,state_dim=0,rotary_pairs=0,chunk=32;
};
inline bool eligible(const Shape& s) {
    if (s.batch<=0||s.batch>65535||s.sequence<=0||s.sequence>std::numeric_limits<int>::max()-32||
        s.heads<=0||s.heads>1024||s.groups<=0||s.groups>s.heads||s.heads%s.groups||
        s.head_dim<=0||s.head_dim>64||s.state_dim<=0||s.state_dim>64||s.state_dim%2||
        s.rotary_pairs<=0||s.rotary_pairs>s.state_dim/2||(s.chunk!=16&&s.chunk!=32)) return false;
    const auto bh=static_cast<std::size_t>(s.batch)*s.heads;
    const auto limit=std::numeric_limits<std::size_t>::max()/sizeof(float);
    const auto count=static_cast<std::size_t>(1+(s.sequence-1)/s.chunk);
    const auto width=static_cast<std::size_t>(s.head_dim*s.state_dim+s.state_dim+s.head_dim+s.rotary_pairs);
    return bh<=limit/static_cast<std::size_t>(s.sequence)/(2*s.state_dim+s.head_dim+s.rotary_pairs+3)&&
        bh<=limit/count/width&&bh<=limit/((s.chunk+1)*s.head_dim*s.state_dim+2*s.chunk*s.state_dim+s.chunk*s.rotary_pairs);
}
inline int chunks(const Shape& s) {return eligible(s)?1+(s.sequence-1)/s.chunk:0;}
inline std::size_t state_elements(const Shape& s) {
    return eligible(s)?static_cast<std::size_t>(s.batch)*s.heads*(s.head_dim*s.state_dim+s.state_dim+s.head_dim+s.rotary_pairs):0;
}
inline std::size_t boundary_elements(const Shape& s) {return state_elements(s)*chunks(s);}
inline std::size_t replay_elements(const Shape& s) {
    // Only one chunk per batch/head, reused by its persistent reverse owner.
    return eligible(s)?static_cast<std::size_t>(s.batch)*s.heads*
        ((s.chunk+1)*s.head_dim*s.state_dim+2*s.chunk*s.state_dim+s.chunk*s.rotary_pairs):0;
}
inline std::size_t gradient_partial_elements(const Shape& s) {
    // Expanded dQ/dK for deterministic GQA reduction + per-B head bias/D sums.
    return eligible(s)?static_cast<std::size_t>(s.batch)*s.heads*
        (2*static_cast<std::size_t>(s.sequence)*s.state_dim+2*s.state_dim+1):0;
}
// All storage is FP32 and contiguous, all pointers refer to the selected device.
// B/S/H/G/P/N/R correspond to batch/time/heads/groups/head_dim/state_dim/rotary.
// This is the POST-BCNorm SISO operator, not a complete Mamba3 projection block.
struct Input {
    const float *q=nullptr,*k=nullptr; // [B,S,G,N], pre-bias/pre-rotation
    const float *v=nullptr,*z=nullptr; // [B,S,H,P], optional Z
    const float *adt=nullptr,*dt=nullptr,*trap=nullptr; // [B,S,H], Trap logits
    const float *angles=nullptr; // [B,S,H,R], raw before pi*tanh
    const float *q_bias=nullptr,*k_bias=nullptr; // [H,N]
    const float *d=nullptr; // optional [H]
    const int* valid=nullptr; // required device prefixes [B], immutable through VJP
};
struct State {float *phase=nullptr,*ssm=nullptr,*k=nullptr,*v=nullptr;};
struct ConstState {const float *phase=nullptr,*ssm=nullptr,*k=nullptr,*v=nullptr;};
struct Gradient {
    float *q=nullptr,*k=nullptr,*v=nullptr,*z=nullptr,*adt=nullptr,*dt=nullptr,*trap=nullptr,
        *angles=nullptr,*q_bias=nullptr,*k_bias=nullptr,*d=nullptr;
};
// Entering boundaries are packed per (B,H): phase,SSM,K,V, each with C slots,
// then the next (B,H). The whole allocation has boundary_elements(s) floats.
// Replay and partial buffers have exactly the sizes declared above.
// No storage may overlap written ranges. Initial/final seed must be either
// completely null (=zero) or complete; final destinations always complete.
// Each tape/session owns its own buffers. Inputs/initial/boundaries must remain
// immutable until backward finishes; forward and backward use current_stream().
// No hidden allocation, H2D, D2H, fence, fallback, or shared/global host state.
// true=enqueued. Inspect status[B] at the trainer finite/publication boundary.
// Status: 0=ok,1=bad prefix,2=operand/seed range,3=numerical overflow.
// Valid projected operands: finite, |Q/K/V/Z/bias/D/angles|<=64,
// 0<=DT<=16, ADT<=0. Initial SSM/K/V and all output/final-state adjoints
// are arbitrary finite FP32; limiting these like projections would reject
// the operator's own streaming outputs/adjoints. Initial phase |x|<=2pi.
// Finite inputs/adjoints can overflow during computation: status3 rejects
// publication, never silently clamps/truncates the derivative.
// Invalid batches produce zero outputs/gradients/state, and must be discarded.
// Backward preserves sticky forward status. Padding is never read as data.
bool supported(const Shape& shape);
bool forward(const Shape& shape,Input input,ConstState initial,State final,
    float* output,float* boundaries,int* status,const int* upstream_status=nullptr);
bool backward(const Shape& shape,Input input,ConstState initial,
    const float* output_gradient,ConstState final_gradient,Gradient gradient,
    State initial_gradient,const float* boundaries,float* replay,float* partials,int* status);
}
