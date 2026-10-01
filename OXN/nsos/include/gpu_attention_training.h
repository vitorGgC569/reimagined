#pragma once
#include "tensor.h"
#include "cuda/attention_rdna_training.cuh"
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace nsos {
class BitLinear;
class Parameter;
namespace attention_training {
enum class Policy { FP32, RdnaBF16, RdnaFP16 };
// Single parser shared by dispatch and checkpoint identity. No fallback after opt-in.
Policy policy();
Policy parse_policy(const std::string& value);
const char* policy_identity(Policy value);
std::array<std::uint64_t,3> dispatch_counters(); // forward, backward, finite merge
inline constexpr const char* integration_identity =
    "attention_training_owned_rope_split_half_bitlinear_replay_prefix_sticky_v1";

struct Config {
    attention_rdna::Shape shape;
    int start_position = 0;
};
struct ProjectionGradient { Tensor q, kv; };
// Post-projection tape. Copies all inputs, including caller-supplied RoPE tables.
// Tables [positions,D/2] use the established split-half NSOS rotation. Odd D
// preserves the unpaired last coordinate. Each tape is single-consumption.
class Tape {
public:
    static std::unique_ptr<Tape> forward(const Config&, const Tensor& q,
        const Tensor& kv, const Tensor& rope_cos, const Tensor& rope_sin,
        const std::vector<int>& valid = {}, std::unique_ptr<Tape> reuse = {});
    ~Tape();
    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;
    Tensor output() const; // private O is never exposed for mutation
    ProjectionGradient backward(const Tensor& grad_output);
    const int* device_status() const;
    std::vector<int> audit_status() const; // explicit fence and D2H
    bool consumed() const;
    void cancel(); // retains in-flight storage through its completion boundary
    std::size_t workspace_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit Tape(std::unique_ptr<Impl>);
};

// Composes the existing BitLinear projections; never substitutes a plain GEMM.
// References must outlive this provider. One outstanding forward. Projection
// parameters/policy must remain unchanged until backward; ordinary Parameter
// add_grad handles accumulation. Replays restore BitLinear's private tape.
class Provider {
public:
    Provider(BitLinear& q, BitLinear& kv, BitLinear& out);
    ~Provider();
    Provider(const Provider&) = delete;
    Provider& operator=(const Provider&) = delete;
    Tensor forward(const Tensor& input, const Config&, const Tensor& rope_cos,
        const Tensor& rope_sin, const std::vector<int>& valid = {});
    Tensor backward(const Tensor& dy);
    void cancel_pending();
    bool pending() const;
    std::size_t workspace_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Weak registry of provider-owned ledgers, filtered by actual Parameter identity.
// Called once at group-open/abort, never between accumulation microbatches.
void reset_status(const std::vector<Parameter*>& parameters);
// ORs sticky status into the EXISTING trainer finite issue on its owning lane.
// Does not clear it, allocate, download or sync. Supports deferred Adam gates.
void merge_status(const std::vector<Parameter*>& parameters, int* device_issue);
} // namespace attention_training
} // namespace nsos
