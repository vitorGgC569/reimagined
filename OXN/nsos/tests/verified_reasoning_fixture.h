#pragma once
#include "mcts_reasoning.h"
#include <cmath>

// Executable structured-task oracle for plumbing tests ONLY. It does not
// represent a trained language verifier: candidate[0] must encode integer 4.
inline nsos::ReasoningPolicy arithmetic_reasoning_fixture() {
    nsos::ReasoningPolicy policy;
    policy.policy_id = "test:structured-candidates:v1";
    policy.verifier_id = "test:2-plus-2-exact:v1";
    policy.propose = [](const nsos::Tensor& state, int, int limit) {
        std::vector<nsos::ReasoningProposal> candidates;
        for (int value : {99, 4}) {
            if (static_cast<int>(candidates.size()) >= limit) break;
            nsos::Tensor host = state.get_device() == nsos::Device::GPU ? state.cpu() : state.clone();
            host.data()[0] = static_cast<float>(value);
            nsos::Tensor candidate = state.get_device() == nsos::Device::GPU ? host.to(nsos::Device::GPU) : host;
            candidates.push_back({candidate, value == 99 ? 0.999f : 0.001f});
        }
        return candidates;
    };
    policy.verify = nsos::exact_token_verifier({4}, [](const nsos::Tensor& state) {
        nsos::Tensor host = state.get_device() == nsos::Device::GPU ? state.cpu() : state;
        const float value = host.data()[0];
        return std::vector<int>{value == 4.0f ? 4 : -1};
    });
    return policy;
}
