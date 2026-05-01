#include "../include/nsos_mpi.h"

#include <iostream>

namespace nsos {

MultiNodeOrchestrator::MultiNodeOrchestrator() {
    std::cout << "[NSOS] MPI disabled, running in single-node orchestrator mode."
              << std::endl;
}

MultiNodeOrchestrator::~MultiNodeOrchestrator() = default;

void MultiNodeOrchestrator::sync_gradients(Tensor& grads) {
    if (grads.size <= 0) {
        return;
    }

    buffer.assign(grads.data(), grads.data() + grads.size);
    if (world_size > 1) {
        const float inv_world = 1.0f / static_cast<float>(world_size);
        float* grad_ptr = grads.data();
        for (int i = 0; i < grads.size; ++i) {
            grad_ptr[i] = buffer[i] * inv_world;
        }
    }
}

void MultiNodeOrchestrator::broadcast_params(Tensor& params) {
    if (params.size <= 0) {
        return;
    }
    buffer.assign(params.data(), params.data() + params.size);
}

int MultiNodeOrchestrator::get_rank() const { return rank; }

int MultiNodeOrchestrator::get_world_size() const { return world_size; }

} // namespace nsos
