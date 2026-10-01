#pragma once
#include "mamba3_reference.h"

namespace nsos::mamba3_preprocessing {
// FP64 oracle for the original SISO block's post-projection preprocessing.
// Projections, output linear/norm, model ownership and HIP are NOT provided.
// B/K and C/Q are group-shared raw projections, normalized before head biases.
struct Inputs {
    mamba3_reference::Geometry geometry;
    std::vector<double> q,k; // [B,S,G,N], BEFORE RMSNorm
    std::vector<double> v,z; // [B,S,H,P], z optional
    std::vector<double> raw_a,raw_dt,trap; // [B,S,H]
    std::vector<double> angles; // [B,S,R], shared over heads by original projection
    std::vector<double> q_norm,k_norm; // [N], shared RMSNorm weights
    std::vector<double> dt_bias; // [H]
    std::vector<double> q_bias,k_bias; // [H,N]
    std::vector<double> d; // [H], optional
    std::vector<int> valid_lengths;
    double norm_eps=1e-5,a_floor=1e-4; // fixed, NOT trainable
};
mamba3_reference::Inputs forward(const Inputs& input);
// VJP through RMSNorm, heavy-tail/floor, softplus and head broadcast.
// Floor uses a zero subgradient on equality. All padded gradients are zero.
// Temporal/state VJP is a separate call to mamba3_reference::backward.
Inputs backward(const Inputs& input,const mamba3_reference::Inputs& gradient);
}
