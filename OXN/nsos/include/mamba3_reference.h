#pragma once
#include <vector>
#include "cuda/mamba3_layer_kernels.cuh"

namespace nsos::mamba3_reference {
// Functional FP64 reference, not a production Tensor/GPU backend. Its input
// contract is the original post-BCNorm SISO operator. All time-varying fields
// use [batch,sequence,...] (ADT/DT/Trap must be transposed from upstream B,H,S).
struct Geometry {
    int batch=0, sequence=0, heads=0, groups=0, head_dim=0, state_dim=0, rotary_pairs=0;
};
struct Inputs {
    Geometry geometry;
    std::vector<double> q,k; // [B,S,G,N], BEFORE head-specific biases and rotation
    std::vector<double> v,z; // [B,S,H,P]; z empty means ungated
    std::vector<double> adt,dt,trap; // [B,S,H], trap contains logits
    std::vector<double> angles; // [B,S,H,R], raw angles before pi*tanh
    std::vector<double> q_bias,k_bias; // [H,N], AFTER BCNorm
    std::vector<double> d; // [H], empty means no skip
    std::vector<int> valid_lengths; // empty means S; otherwise valid prefixes [B]
};
struct State {
    std::vector<double> phase; // [B,H,R], cumulative phase modulo 2pi
    std::vector<double> ssm; // [B,H,P,N]
    std::vector<double> k; // [B,H,N], previous rotated key
    std::vector<double> v; // [B,H,P], previous value
};
struct Forward { std::vector<double> output; State final_state; };
struct Gradients { Inputs input; State initial_state; };
void validate_geometry(const Geometry& geometry);
State zero_state(const Geometry& geometry);
Forward forward(const Inputs& input,const State& initial_state={});
// Exact VJP includes both output and optional FINAL state seeds. The returned
// initial-state adjoint is explicit: caller chooses detach vs cross-call BPTT.
// Phase modulo has derivative one away from its discontinuity. DT and ADT are
// independent here; block preprocessing must chain ADT=A*DT separately.
Gradients backward(const Inputs& input,const State& initial_state,
    const std::vector<double>& output_gradient,const State& final_state_gradient={});

// Integral post-linear FP64 oracle, including rank-summed MIMO recurrence,
// rank projections, BCNorm/heavy-tail/DT, optional pregate headwise output norm
// and output gating. Projection/core layouts use mamba3_block::Shape/Layout.
// Initial V is RAW, not rank-projected; K uses [B,H,rank,N].
struct BlockInputs {
    mamba3_block::Shape geometry;
    std::vector<double> projection,core;
    std::vector<int> valid_lengths;
};
struct BlockGradients {std::vector<double> projection,core;State initial_state;};
State block_zero_state(const mamba3_block::Shape& shape);
Forward block_forward(const BlockInputs& input,const State& initial={});
BlockGradients block_backward(const BlockInputs& input,const State& initial,
    const std::vector<double>& gradient,const State& final_seed={});
} // namespace nsos::mamba3_reference
