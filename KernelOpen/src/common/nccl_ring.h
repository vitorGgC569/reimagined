#ifndef UHK_COMMON_NCCL_RING_H
#define UHK_COMMON_NCCL_RING_H

#include <vector>
#include <cstdint>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

// =============================================================================
// NCCL RING-ALLREDUCE (Cluster Training Logic)
// =============================================================================
// Implements Ring-AllReduce.
// If compiled with USE_MPI, executes real MPI calls.
// Otherwise, simulates logic for single-node debugging.
// =============================================================================

namespace uhk {
namespace comm {

    class RingNode {
    public:
        int rank;
        int size;
        int next_rank;
        int prev_rank;

        RingNode() {
            rank = 0;
            size = 1;

#ifdef USE_MPI
            int initialized;
            MPI_Initialized(&initialized);
            if (!initialized) {
                // Initialize if not already done (though usually done in main)
                // MPI_Init(NULL, NULL);
                // Warning: MPI_Init should be called once. Assuming external init.
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
            next_rank = (rank + 1) % size;
            prev_rank = (rank - 1 + size) % size;
        }

        void perform_allreduce(float* data, size_t count) {
            if (size == 1) return; // No-op for single node

            size_t chunk_size = count / size;

            // Temporary buffers for MPI
            std::vector<float> recv_buffer(chunk_size);
            std::vector<float> send_buffer(chunk_size);

#ifdef USE_MPI
            MPI_Status status;

            // Phase 1: Scatter-Reduce
            for (int step = 0; step < size - 1; ++step) {
                int send_chunk_idx = (rank - step + size) % size;
                int recv_chunk_idx = (rank - step - 1 + size) % size;

                size_t send_offset = send_chunk_idx * chunk_size;
                size_t recv_offset = recv_chunk_idx * chunk_size;

                // Copy data to send buffer (for safety/simplicity)
                std::copy(data + send_offset, data + send_offset + chunk_size, send_buffer.begin());

                // SendRecv in ring
                // Send to Next, Recv from Prev
                MPI_Sendrecv(send_buffer.data(), chunk_size, MPI_FLOAT, next_rank, 0,
                             recv_buffer.data(), chunk_size, MPI_FLOAT, prev_rank, 0,
                             MPI_COMM_WORLD, &status);

                // Reduce
                for(size_t i=0; i<chunk_size; ++i) {
                    data[recv_offset + i] += recv_buffer[i];
                }
            }

            // Phase 2: All-Gather
            for (int step = 0; step < size - 1; ++step) {
                int send_chunk_idx = (rank - step + 1 + size) % size;
                int recv_chunk_idx = (rank - step + size) % size;

                size_t send_offset = send_chunk_idx * chunk_size;
                size_t recv_offset = recv_chunk_idx * chunk_size;

                std::copy(data + send_offset, data + send_offset + chunk_size, send_buffer.begin());

                MPI_Sendrecv(send_buffer.data(), chunk_size, MPI_FLOAT, next_rank, 1,
                             recv_buffer.data(), chunk_size, MPI_FLOAT, prev_rank, 1,
                             MPI_COMM_WORLD, &status);

                // Overwrite (Gather)
                for(size_t i=0; i<chunk_size; ++i) {
                    data[recv_offset + i] = recv_buffer[i];
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
#else
            // Simulation Mode (Print Trace)
            // std::cout << "[Ring] Rank " << rank << " simulating AllReduce on " << count << " elements." << std::endl;
#endif
        }
    };

}
}

#endif
