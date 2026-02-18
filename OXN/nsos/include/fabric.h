#ifndef FABRIC_H
#define FABRIC_H

#include "tensor.h"
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

// Fabric for Distributed Training (MPI/NCCL)
class Fabric {
public:
  int rank;
  int world_size;

  Fabric();
  ~Fabric();

  void barrier();
  void all_reduce(nsos::Tensor &t);

  // P2P (for Ring Algorithms if needed manually, though MPI_Allreduce handles
  // it)
  void send(const nsos::Tensor &t, int dest_rank);
  nsos::Tensor recv(int src_rank, std::vector<int> shape);
};

#endif
