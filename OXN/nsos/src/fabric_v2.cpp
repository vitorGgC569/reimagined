#include "fabric.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace nsos;

namespace {

#ifdef USE_MPI

std::mutex mpi_lifecycle_mutex;
std::mutex mpi_call_mutex;
size_t mpi_fabric_instances = 0;
bool mpi_initialized_by_fabric = false;

int checked_mpi_count(std::int64_t count, const char* operation) {
  if (count < 0 ||
      count > static_cast<std::int64_t>((std::numeric_limits<int>::max)())) {
    throw std::overflow_error(std::string(operation) +
                              " tensor size exceeds MPI int count");
  }
  return static_cast<int>(count);
}

void validate_rank(int candidate, int world_size, const char* operation) {
  if (candidate < 0 || candidate >= world_size) {
    throw std::out_of_range(std::string(operation) +
                            " rank is outside the communicator");
  }
}

std::string mpi_error_message(int status) {
  char buffer[MPI_MAX_ERROR_STRING]{};
  int length = 0;
  if (MPI_Error_string(status, buffer, &length) != MPI_SUCCESS || length <= 0) {
    return "MPI error code " + std::to_string(status);
  }
  return std::string(buffer, static_cast<size_t>(length));
}

void mpi_check(int status, const char* operation) {
  if (status != MPI_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             mpi_error_message(status));
  }
}

#else

int parse_process_environment(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  const std::string_view text(raw);
  int value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    throw std::runtime_error(std::string("Invalid ") + name +
                             " for non-MPI NSOS build");
  }
  return value;
}

#endif

}  // namespace

#ifdef USE_MPI

Fabric::Fabric() {
  std::lock_guard<std::mutex> lifecycle_lock(mpi_lifecycle_mutex);
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  bool started_here = false;
  try {
    int initialized = 0;
    mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    int finalized = 0;
    mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized != 0) {
      throw std::runtime_error(
          "Fabric cannot initialize after MPI has been finalized");
    }
    int provided = 0;
    if (initialized == 0) {
      int argc = 0;
      char** argv = nullptr;
      mpi_check(
          MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided),
          "MPI_Init_thread");
      started_here = true;
    } else {
      mpi_check(MPI_Query_thread(&provided), "MPI_Query_thread");
    }
    if (provided < MPI_THREAD_SERIALIZED) {
      throw std::runtime_error(
          "MPI provider does not support MPI_THREAD_SERIALIZED");
    }

    mpi_check(MPI_Comm_dup(MPI_COMM_WORLD, &communicator_), "MPI_Comm_dup");
    mpi_check(MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler");
    mpi_check(MPI_Comm_rank(communicator_, &rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(communicator_, &world_size), "MPI_Comm_size");
    if (world_size <= 0) {
      throw std::runtime_error("MPI communicator has an invalid world size");
    }
    validate_rank(rank, world_size, "MPI local");
    if (started_here) {
      mpi_initialized_by_fabric = true;
    }
    ++mpi_fabric_instances;
    registered_mpi_ = true;
  } catch (...) {
    if (communicator_ != MPI_COMM_NULL) {
      const int free_status = MPI_Comm_free(&communicator_);
      if (free_status != MPI_SUCCESS) {
        std::cerr << "Fabric constructor communicator cleanup failed: "
                  << mpi_error_message(free_status) << std::endl;
      }
      communicator_ = MPI_COMM_NULL;
    }
    if (started_here) {
      const int status = MPI_Finalize();
      if (status != MPI_SUCCESS) {
        std::cerr << "Fabric constructor cleanup failed: "
                  << mpi_error_message(status) << std::endl;
      }
    }
    throw;
  }
}

Fabric::~Fabric() noexcept {
  std::lock_guard<std::mutex> lifecycle_lock(mpi_lifecycle_mutex);
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  if (!registered_mpi_) {
    return;
  }
  registered_mpi_ = false;
  int finalized = 0;
  const int query_status = MPI_Finalized(&finalized);
  if (query_status != MPI_SUCCESS) {
    std::cerr << "MPI_Finalized failed during Fabric destruction: "
              << mpi_error_message(query_status) << std::endl;
  } else if (finalized == 0 && communicator_ != MPI_COMM_NULL) {
    const int free_status = MPI_Comm_free(&communicator_);
    if (free_status != MPI_SUCCESS) {
      std::cerr << "MPI_Comm_free failed during Fabric destruction: "
                << mpi_error_message(free_status) << std::endl;
    }
  }
  communicator_ = MPI_COMM_NULL;
  if (mpi_fabric_instances == 0) {
    std::cerr << "Fabric MPI lifecycle registry underflow" << std::endl;
    std::terminate();
  }
  --mpi_fabric_instances;
  if (mpi_fabric_instances != 0 || !mpi_initialized_by_fabric) {
    return;
  }
  if (query_status != MPI_SUCCESS) {
    return;
  }
  if (finalized == 0) {
    const int finalize_status = MPI_Finalize();
    if (finalize_status != MPI_SUCCESS) {
      std::cerr << "MPI_Finalize failed during Fabric destruction: "
                << mpi_error_message(finalize_status) << std::endl;
      return;
    }
  }
  mpi_initialized_by_fabric = false;
}

void Fabric::barrier() {
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  mpi_check(MPI_Barrier(communicator_), "MPI_Barrier");
}

void Fabric::all_reduce(nsos::Tensor& tensor) {
  if (world_size == 1 || tensor.size == 0) {
    return;
  }
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  const int count = checked_mpi_count(tensor.size, "MPI_Allreduce");
  if (tensor.get_device() == Device::GPU) {
    nsos::Tensor host = tensor.to(Device::CPU);
    mpi_check(MPI_Allreduce(MPI_IN_PLACE, host.data(), count, MPI_FLOAT,
                            MPI_SUM, communicator_),
              "MPI_Allreduce");
    tensor.copy_from(host);
    return;
  }
  mpi_check(MPI_Allreduce(MPI_IN_PLACE, tensor.data(), count, MPI_FLOAT,
                          MPI_SUM, communicator_),
            "MPI_Allreduce");
}

void Fabric::broadcast(nsos::Tensor& tensor, int source_rank) {
  validate_rank(source_rank, world_size, "MPI_Bcast source");
  if (world_size == 1 || tensor.size == 0) {
    return;
  }
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  const int count = checked_mpi_count(tensor.size, "MPI_Bcast");
  if (tensor.get_device() == Device::GPU) {
    nsos::Tensor host = rank == source_rank
                            ? tensor.to(Device::CPU)
                            : nsos::Tensor(tensor.shape.dims, Device::CPU);
    mpi_check(MPI_Bcast(host.data(), count, MPI_FLOAT, source_rank,
                        communicator_),
              "MPI_Bcast");
    tensor.copy_from(host);
    return;
  }
  mpi_check(MPI_Bcast(tensor.data(), count, MPI_FLOAT, source_rank,
                      communicator_),
            "MPI_Bcast");
}

void Fabric::send(const nsos::Tensor& tensor, int destination_rank) {
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  validate_rank(destination_rank, world_size, "MPI_Send destination");
  const int count = checked_mpi_count(tensor.size, "MPI_Send");
  const nsos::Tensor host = tensor.get_device() == Device::GPU
                                ? tensor.to(Device::CPU)
                                : tensor;
  mpi_check(MPI_Send(host.data(), count, MPI_FLOAT, destination_rank, 0,
                     communicator_),
            "MPI_Send");
}

nsos::Tensor Fabric::recv(int source_rank, std::vector<int> shape) {
  std::lock_guard<std::mutex> call_lock(mpi_call_mutex);
  validate_rank(source_rank, world_size, "MPI_Recv source");
  nsos::Tensor tensor(shape, Device::CPU);
  const int count = checked_mpi_count(tensor.size, "MPI_Recv");
  MPI_Status status{};
  mpi_check(MPI_Recv(tensor.data(), count, MPI_FLOAT, source_rank, 0,
                     communicator_, &status),
            "MPI_Recv");
  int received = 0;
  mpi_check(MPI_Get_count(&status, MPI_FLOAT, &received), "MPI_Get_count");
  if (received != count) {
    throw std::runtime_error(
        "MPI_Recv returned a payload with an unexpected element count");
  }
  return tensor;
}

#else

Fabric::Fabric() {
  const int requested_rank = parse_process_environment("RANK", 0);
  const int requested_world_size =
      parse_process_environment("WORLD_SIZE", 1);
  if (requested_rank != 0 || requested_world_size != 1) {
    throw std::runtime_error(
        "Distributed execution requested, but NSOS was built without MPI");
  }
}

Fabric::~Fabric() noexcept = default;

void Fabric::barrier() {}

void Fabric::all_reduce(nsos::Tensor& tensor) {
  if (rank != 0 || world_size != 1) {
    throw std::logic_error("Non-MPI Fabric entered an invalid process state");
  }
  static_cast<void>(tensor);
}

void Fabric::broadcast(nsos::Tensor& tensor, int source_rank) {
  if (rank != 0 || world_size != 1) {
    throw std::logic_error("Non-MPI Fabric entered an invalid process state");
  }
  if (source_rank != 0) {
    throw std::out_of_range(
        "single-process Fabric broadcast source must be rank zero");
  }
  static_cast<void>(tensor);
}

void Fabric::send(const nsos::Tensor& tensor, int destination_rank) {
  static_cast<void>(tensor);
  static_cast<void>(destination_rank);
  throw std::logic_error("Fabric::send requires an MPI-enabled NSOS build");
}

nsos::Tensor Fabric::recv(int source_rank, std::vector<int> shape) {
  static_cast<void>(source_rank);
  static_cast<void>(shape);
  throw std::logic_error("Fabric::recv requires an MPI-enabled NSOS build");
}

#endif
