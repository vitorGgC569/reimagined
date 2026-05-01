#include "../include/fabric.h"
#include "../include/nsos_config.h"
#include <iostream>
#include <vector>
#include <cstring>

namespace nsos {

Fabric::Fabric(int rank, int world_size) 
    : rank_(rank), world_size_(world_size) {
    // Initialize buffers
    send_buffer_.resize(1024 * 1024); // 1MB default
    recv_buffer_.resize(1024 * 1024);
}

void Fabric::send(const Tensor& data, int dest_rank) {
    if (dest_rank == rank_) return;
    
    // Simulate send (or use MPI if configured)
    #ifdef NSOS_USE_MPI
    // MPI_Send implementation placeholder
    #else
    // Local loopback not implemented for distributed simulation in single process
    // Just a placeholder to ensure compilation
    #endif
}

Tensor Fabric::recv(int src_rank, const std::vector<int>& shape) {
    if (src_rank == rank_) return Tensor::zeros(shape, Device::CPU);
    
    // Simulate recv
    return Tensor::zeros(shape, Device::CPU);
}

void Fabric::broadcast(Tensor& data, int root) {
    // In single node, broadcast is a no-op
    if (world_size_ <= 1) return;
    
    #ifdef NSOS_USE_MPI
    // MPI_Bcast implementation placeholder
    #endif
}

void Fabric::all_reduce(Tensor& data, ReduceOp op) {
    if (world_size_ <= 1) return;
    
    // In a real distributed setting, this would sum gradients across nodes
    // For now, it's an identity operation in single-node mode
}

void Fabric::barrier() {
    // No-op for single node
}

} // namespace nsos
