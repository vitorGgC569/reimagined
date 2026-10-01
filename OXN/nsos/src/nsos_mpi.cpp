#include "../include/nsos_mpi.h"

#include <stdexcept>

namespace nsos {

MultiNodeOrchestrator::MultiNodeOrchestrator() = default;

MultiNodeOrchestrator::~MultiNodeOrchestrator() noexcept = default;

void MultiNodeOrchestrator::sync_gradients(Tensor& grads) {
    fabric_.all_reduce(grads);
    if (fabric_.world_size > 1 && grads.size > 0) {
        const Tensor averaged =
            grads.mul(1.0f / static_cast<float>(fabric_.world_size));
        grads.copy_from(averaged);
    }
}

void MultiNodeOrchestrator::broadcast_params(Tensor& params) {
    fabric_.broadcast(params, 0);
}

int MultiNodeOrchestrator::get_rank() const { return fabric_.rank; }

int MultiNodeOrchestrator::get_world_size() const { return fabric_.world_size; }

} // namespace nsos
