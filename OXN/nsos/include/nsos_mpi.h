#ifndef NSOS_MPI_H
#define NSOS_MPI_H

#include "tensor.h"
#include <vector>

namespace nsos {

class MultiNodeOrchestrator {
  int rank = 0;
  int world_size = 1;
  std::vector<float> buffer; // Persistent buffer for sync

public:
  MultiNodeOrchestrator();
  ~MultiNodeOrchestrator();

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
