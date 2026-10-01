#ifndef NSOS_MPI_H
#define NSOS_MPI_H

#include "fabric.h"

namespace nsos {

class MultiNodeOrchestrator {
  Fabric fabric_;

public:
  MultiNodeOrchestrator();
  ~MultiNodeOrchestrator() noexcept;

  // Distributed Training Interface
  // Synchronize Gradients (AllReduce + Average)
  void sync_gradients(Tensor &grads);

  // Broadcast Initial Params (Master to Workers)
  void broadcast_params(Tensor &params);

  int get_rank() const;
  int get_world_size() const;
};

} // namespace nsos

#endif
