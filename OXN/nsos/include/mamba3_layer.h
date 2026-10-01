#pragma once
#include "autograd.h"
#include "cuda/mamba3_layer_kernels.cuh"
#include <memory>
#include <string>
#include <vector>

namespace nsos {
// Independent architecture opt-in. Dense FP32 projections have no BitLinear
// normalization, magnitude, QAT or ternary reinterpretation. Schema is mandatory
// in every pack/checkpoint: Mamba2 shapes are never accepted as Mamba3 weights.
struct Mamba3Config {
    int schema_version=1,expand=2,head_dim=64,state_dim=128,n_groups=1,mimo_rank=4;
    bool mimo=false,outproj_norm=false;
    float rope_fraction=0.5f,norm_eps=1e-5f,a_floor=1e-4f;
    float dt_min=0.001f,dt_max=0.1f,dt_init_floor=1e-4f;
    std::uint64_t seed=1;
};
struct Mamba3State {
    Tensor phase,ssm,k,v; // [B,H,A], [B,H,P,N], [B,H,R,N], [B,H,P] RAW V
};
struct Mamba3SessionSnapshot {
    int schema_version=1;
    std::string configuration;
    bool enabled=false;
    Mamba3State state;
};
struct Mamba3Backward;
class Mamba3Tape : public std::enable_shared_from_this<Mamba3Tape> {
public:
    ~Mamba3Tape();
    Mamba3Tape(const Mamba3Tape&)=delete;
    Mamba3Tape& operator=(const Mamba3Tape&)=delete;
    Tensor output() const; // owning copy; caller writes cannot invalidate tape
    Mamba3State snapshot_final_state() const;
    Mamba3Backward backward(const Tensor& dy,const Mamba3State& final_seed={});
    void cancel(); // single use; retains every queued allocation until destruction
    bool consumed() const;
    std::vector<int> audit_status() const; // explicit readback + fence
    const Tensor& status_tensor() const; // [B] FP32 integer codes 0/1/2/3
    std::size_t workspace_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit Mamba3Tape(std::unique_ptr<Impl>);
    friend class Mamba3Layer;
};
struct Mamba3Backward {
    Tensor input;
    Mamba3State initial_state;
    std::vector<Tensor> parameters; // canonical enumeration order
    std::shared_ptr<Mamba3Tape> owner; // publication binds lifetime/identity/status
};
struct Mamba3Telemetry {
    std::uint64_t cpu_forward=0,gpu_forward=0,cpu_backward=0,gpu_backward=0,cancelled=0;
    std::size_t peak_workspace_bytes=0;
};
class Mamba3Layer {
public:
    Mamba3Layer(int d_model,const Mamba3Config& config={});
    ~Mamba3Layer();
    Mamba3Layer(const Mamba3Layer&)=delete;
    Mamba3Layer& operator=(const Mamba3Layer&)=delete;
    // Explicit owning API for concurrent independent tapes and cross-call BPTT.
    // Layer/session mutation is serialized by Jamba's existing lane/lock.
    std::shared_ptr<Mamba3Tape> forward_owned(const Tensor& input,
        const Mamba3State& initial={},const std::vector<int>& valid_lengths={});
    Mamba3Backward backward_owned(const std::shared_ptr<Mamba3Tape>& tape,
        const Tensor& dy,const Mamba3State& final_seed={});
    // Baseline integration publication boundary: audit ALL batches first,
    // reject failed op, then add gradients once. No parameters published by VJP.
    void publish(Mamba3Backward& result);
    Tensor forward(const Tensor& input,Context* context=nullptr,
        const std::vector<int>& valid_lengths={});
    Tensor backward(const Tensor& dy,Context& context);
    void cancel_pending();
    void reset();
    void to(Device device);
    std::vector<Parameter*> parameters();
    std::vector<Parameter*> no_weight_decay_parameters(); // dt_bias and D
    void set_training_mode(bool enabled);
    void set_streaming_mode(bool enabled);
    bool streaming_mode() const;
    Mamba3SessionSnapshot snapshot_streaming_state(bool device_resident=false) const;
    void restore_streaming_state(const Mamba3SessionSnapshot& snapshot);
    std::vector<Mamba3SessionSnapshot> snapshot_streaming_state_batch(bool device_resident=false) const;
    void restore_streaming_state_batch(const std::vector<Mamba3SessionSnapshot>& snapshots);
    int streaming_batch_size() const;
    const Mamba3Config& config() const;
    std::string configuration_identity() const;
    const std::string& get_layer_name() const;
    void set_layer_name(const std::string& name);
    Mamba3Telemetry telemetry() const;
    void reset_runtime_telemetry();
    // Checked standalone versioned binary artifact (weights + optional session).
    // Model packs should instead use canonical registry plus architecture metadata.
    void save_checkpoint(const std::string& path,bool include_session=false) const;
    void load_checkpoint(const std::string& path,bool restore_session=false);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
