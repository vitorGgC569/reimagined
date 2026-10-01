#pragma once

#include "nsos_config.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

namespace nsos {

class JambaModel;

// Training/checkpoint operations hold a shared lease. Runtime-policy setters
// take the mutation guard, so a precision/determinism/scan switch cannot land
// halfway through an operation. Nested leases on the same thread are safe.
class RuntimeExecutionPolicyLease {
public:
    RuntimeExecutionPolicyLease();
    ~RuntimeExecutionPolicyLease();
    RuntimeExecutionPolicyLease(
        const RuntimeExecutionPolicyLease&) = delete;
    RuntimeExecutionPolicyLease& operator=(
        const RuntimeExecutionPolicyLease&) = delete;

private:
    std::optional<std::shared_lock<std::shared_mutex>> lock_;
};

class RuntimeExecutionPolicyMutationGuard {
public:
    RuntimeExecutionPolicyMutationGuard();
    ~RuntimeExecutionPolicyMutationGuard();
    RuntimeExecutionPolicyMutationGuard(
        const RuntimeExecutionPolicyMutationGuard&) = delete;
    RuntimeExecutionPolicyMutationGuard& operator=(
        const RuntimeExecutionPolicyMutationGuard&) = delete;

private:
    std::optional<std::unique_lock<std::shared_mutex>> lock_;
};

// Canonical, versioned description of every runtime choice that may alter a
// training continuation.  Model weights deliberately do not embed this
// identity: a model checkpoint remains portable for inference, while the
// matching Trainer sidecar and run manifest make training resume fail closed.
struct RuntimeExecutionIdentity {
    using Field = std::pair<std::string, std::string>;

    std::vector<Field> fields;

    std::string canonical_text() const;
    std::string digest() const;
};

// Captures effective values, not merely environment requests.  The optimizer
// metadata is supplied by Trainer because it is configured after construction
// in some public APIs.
// `gradient_accumulation_steps` e `token_scheduler` descrevem o contrato de
// treinamento, não a implementação: A=1 e A=64 podem partir dos mesmos pesos,
// otimizador, LR e dados e ainda assim percorrer trajetórias diferentes, de
// modo que um checkpoint de um não pode retomar como o outro.  Ambos são
// emitidos apenas quando saem do padrão, preservando o digest — e portanto a
// retomada — de todos os checkpoints já escritos.
RuntimeExecutionIdentity capture_runtime_execution_identity(
    JambaModel& model,
    int optimizer_state_bits,
    bool dynamic_loss_scaling_enabled,
    int gradient_accumulation_steps = 1,
    bool token_scheduler = false);

// Validates ordering, uniqueness, limits, schema and required fields.  This is
// also used on untrusted sidecar input before any optimizer state is staged.
RuntimeExecutionIdentity validated_runtime_execution_identity(
    std::vector<RuntimeExecutionIdentity::Field> fields);

bool runtime_execution_identity_equal(
    const RuntimeExecutionIdentity& expected,
    const RuntimeExecutionIdentity& actual) noexcept;

// Actionable, bounded field-level report suitable for resume errors.
std::string runtime_execution_identity_mismatch(
    const RuntimeExecutionIdentity& expected,
    const RuntimeExecutionIdentity& actual);

}  // namespace nsos
