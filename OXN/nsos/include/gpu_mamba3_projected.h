#pragma once
#include "gpu_mamba3_siso.h"
#include "cuda/mamba3_preprocess_kernels.cuh"

namespace nsos::mamba3_projected {
using Shape=mamba3_siso::Shape;
using Config=mamba3_preprocess_gpu::Config;
using State=mamba3_siso::State;
struct Operands {
    Tensor q,k,v,z,raw_a,raw_dt,trap,angles,q_norm,k_norm,dt_bias,q_bias,k_bias,d;
};
struct Gradients {Operands input;State initial_state;};
// Owning GPU composition of original post-projection SISO preprocessing and
// recurrence. It is NOT an input/output projection module, MIMO or Jamba path.
// All tensors are contiguous FP32. Shared projected angles [B,S,R]; BC weights
// [N]; head bias [H,N], dt_bias/D [H]. Complete state adjoints are returned.
// Status/publication remains device-side and sticky across both stages.
// Forward/backward keep private immutable clones and retain every queued output;
// same lane only; no hidden CPU numerical fallback. Host prefix upload is sync.
class Tape {
public:
    static std::unique_ptr<Tape> forward(const Shape& shape,const Operands& input,
        const State& initial={},const std::vector<int>& valid={},Config config={});
    ~Tape();
    Tape(const Tape&)=delete;Tape& operator=(const Tape&)=delete;
    Tensor output() const;
    State snapshot_final_state() const;
    Gradients backward(const Tensor& dy,const State& final_seed={});
    void cancel();
    bool consumed() const;
    const int* device_status() const;
    std::vector<int> audit_status() const; // explicit D2H+fence
    std::size_t preprocessing_workspace_bytes() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
    explicit Tape(std::unique_ptr<Impl> impl);
};
}
