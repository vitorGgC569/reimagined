#include "fabric.h"
#include <cstring>
#include <iostream>
#include <stdexcept>


using namespace nsos;

#ifdef USE_MPI

Fabric::Fabric() {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

Fabric::~Fabric() {
  int flag;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Finalize();
  }
}

void Fabric::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

void Fabric::all_reduce(nsos::Tensor &t) {
  if (world_size == 1)
    return;
  if (t.get_device() == Device::GPU) {
    nsos::Tensor cpu_t = t.to(Device::CPU);
    MPI_Allreduce(MPI_IN_PLACE, cpu_t.data(), cpu_t.size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
    t.copy_from(cpu_t);
  } else {
    MPI_Allreduce(MPI_IN_PLACE, t.data(), t.size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
  }
}

void Fabric::send(const nsos::Tensor &t, int dest_rank) {
  MPI_Send(t.data(), t.size, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD);
}

nsos::Tensor Fabric::recv(int src_rank, std::vector<int> shape) {
  nsos::Tensor t(shape, Device::CPU);
  MPI_Recv(t.data(), t.size, MPI_FLOAT, src_rank, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  return t;
}

#else

#include <condition_variable>
#include <mutex>
#include <thread>


static std::vector<std::vector<float>> shared_buffers(8);
static std::vector<std::mutex> rank_mutexes(8);
static std::vector<std::condition_variable> rank_cvs(8);
static std::vector<bool> data_ready(8, false);

Fabric::Fabric() : rank(0), world_size(1) {
  if (const char *env_rank = std::getenv("RANK"))
    rank = std::atoi(env_rank);
  if (const char *env_ws = std::getenv("WORLD_SIZE"))
    world_size = std::atoi(env_ws);
}

Fabric::~Fabric() {}
void Fabric::barrier() {}

void Fabric::send(const nsos::Tensor &t, int dest_rank) {
  if (dest_rank >= 8)
    return;
  std::unique_lock<std::mutex> lock(rank_mutexes[dest_rank]);
  shared_buffers[dest_rank].resize(t.size);
  std::memcpy(shared_buffers[dest_rank].data(), t.data(),
              t.size * sizeof(float));
  data_ready[dest_rank] = true;
  rank_cvs[dest_rank].notify_one();
}

nsos::Tensor Fabric::recv(int src_rank, std::vector<int> shape) {
  if (rank >= 8)
    return nsos::Tensor::zeros(shape, Device::CPU);
  std::unique_lock<std::mutex> lock(rank_mutexes[rank]);
  rank_cvs[rank].wait(lock, [this]() { return data_ready[rank]; });
  nsos::Tensor t(shape, Device::CPU);
  std::memcpy(t.data(), shared_buffers[rank].data(), t.size * sizeof(float));
  data_ready[rank] = false;
  return t;
}

void Fabric::all_reduce(nsos::Tensor &t) {
  if (world_size <= 1)
    return;
  int right = (rank + 1) % world_size;
  int left = (rank - 1 + world_size) % world_size;
  send(t, right);
  nsos::Tensor r = recv(left, t.shape);

  // Safety check size
  if (t.size != r.size)
    return;

  for (int i = 0; i < t.size; ++i) {
    t.data()[i] += r.data()[i];
  }
}

#endif
