#ifndef FABRIC_H
#define FABRIC_H

#include "tensor.h"
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

// Fabric for distributed training through an explicitly available MPI runtime.
class Fabric {
public:
  int rank = 0;
  int world_size = 1;

  Fabric();
  ~Fabric() noexcept;
  Fabric(const Fabric&) = delete;
  Fabric& operator=(const Fabric&) = delete;
  Fabric(Fabric&&) = delete;
  Fabric& operator=(Fabric&&) = delete;

  void barrier();
  void all_reduce(nsos::Tensor &t);
  void broadcast(nsos::Tensor &t, int source_rank);

  // P2P (for Ring Algorithms if needed manually, though MPI_Allreduce handles
  // it)
  void send(const nsos::Tensor &t, int dest_rank);
  nsos::Tensor recv(int src_rank, std::vector<int> shape);

private:
#ifdef USE_MPI
  bool registered_mpi_ = false;
  MPI_Comm communicator_ = MPI_COMM_NULL;
#endif
};

#endif
