#ifndef MPI_MOCK_H
#define MPI_MOCK_H

#include <cstring>

// MPI Constants
#define MPI_COMM_WORLD 0
#define MPI_FLOAT 1
#define MPI_SUM 2

// MPI Types
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

// MPI Mock Functions
inline int MPI_Initialized(int *initialized) {
  if (initialized)
    *initialized = 1;
  return 0;
}

inline int MPI_Init(int *argc, char ***argv) { return 0; }

inline int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  if (rank)
    *rank = 0;
  return 0;
}

inline int MPI_Comm_size(MPI_Comm comm, int *size) {
  if (size)
    *size = 1;
  return 0;
}

inline int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (sendbuf != recvbuf) {
    std::memcpy(recvbuf, sendbuf, count * sizeof(float));
  }
  return 0;
}

inline int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
                     MPI_Comm comm) {
  return 0;
}

#endif // MPI_MOCK_H
