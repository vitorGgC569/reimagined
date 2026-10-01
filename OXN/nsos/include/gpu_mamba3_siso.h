#pragma once
#include "tensor.h"
#include "cuda/mamba3_siso_kernels.cuh"
#include <memory>

namespace nsos::mamba3_siso {
using Shape=mamba3_siso_gpu::Shape;
struct Operands {
    Tensor q,k,v,z,adt,dt,trap,angles,q_bias,k_bias,d;
};
struct State {Tensor phase,ssm,k,v;};
struct Gradients {Operands input;State initial_state;};

// One explicit, owning forward/backward tape per invocation/session. This
// operator is GPU-only and post-BCNorm; the block's projections/preprocessing,
// MIMO, parameter registry and checkpoint are separate integration work.
// Inputs and initial state are deep-cloned into immutable private storage.
// All scratch and metadata are private, allocated before forward; no D2H or
// explicit fence occurs in forward/backward. Host prefix upload uses synchronous
// H2D for source lifetime; raw device ABI avoids that metadata boundary.
// Destruction of DeviceBuffer
// metadata can synchronize driver releases, so keep tapes alive in their lane.
// Optional upstream status[B] is cloned on the same stream and its failure
// code remains sticky through the recurrence. Caller owns its source until the
// clone completes; raw ABI requires immutable borrowed upstream metadata.
class Tape {
public:
    static std::unique_ptr<Tape> forward(const Shape& shape,const Operands& operands,
        const State& initial={},const std::vector<int>& valid_lengths={},
        const int* upstream_device_status=nullptr);
    ~Tape();
    Tape(const Tape&)=delete;Tape& operator=(const Tape&)=delete;
    Tensor output() const;
    State snapshot_final_state() const; // owning copies, safe to fork sessions
    Gradients backward(const Tensor& output_gradient,const State& final_state_gradient={});
    void cancel(); // marks consumed; retains queued storage until destruction
    bool consumed() const;
    const int* device_status() const; // [B], caller finite gate, never write
    std::vector<int> audit_status() const; // EXPLICIT audit D2H+stream fence
    std::size_t boundary_bytes() const;
    std::size_t replay_bytes() const;
    std::size_t partial_bytes() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
    explicit Tape(std::unique_ptr<Impl> impl);
};
}
