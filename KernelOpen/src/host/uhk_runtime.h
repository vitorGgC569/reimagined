#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>
#include <cstdint>
#include <cmath>
#include <limits>

#include "ring_buffer_manager.h"
#include "../common/uhk_types.h"

// Check for CUDA Compiler
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#else
// Mock definitions for Host compilation without CUDA
typedef void* cudaStream_t;
typedef int cudaError_t;
struct dim3 { unsigned int x, y, z; dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1) : x(x), y(y), z(z) {} };
#define cudaSuccess 0
#define cudaHostAllocDefault 0
#define cudaHostAllocMapped 0
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size);
extern "C" cudaError_t cudaFree(void* devPtr);
extern "C" cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
extern "C" cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern "C" cudaError_t cudaDeviceSynchronize();
#endif

namespace uhk {
namespace runtime {

// Forward declaration of the kernel entry point
extern "C" void universal_persistent_kernel(RingBufferControl* control_ptr);

class UniversalKernelRuntime {
public:
    UniversalKernelRuntime() {
        std::cout << "[Runtime] Initializing Universal Heterogeneous Kernel Runtime..." << std::endl;

        // 1. Allocate Ring Buffer Control in Unified/Pinned Memory
        // In a real scenario, we use cudaHostAlloc with Mapped flag to allow Zero-Copy
        #ifdef __CUDACC__
            cudaHostAlloc((void**)&control_ptr_, sizeof(RingBufferControl), cudaHostAllocMapped);
        #else
            // Fallback for non-CUDA env (Simulation/Test)
            control_ptr_ = new RingBufferControl();
        #endif

        // Initialize logic wrapper
        rb_manager_ = std::make_unique<uhk::host::RingBufferManager>(control_ptr_);

        // 2. Launch Persistent Kernel
        launch_kernel();
    }

    ~UniversalKernelRuntime() {
        shutdown();

        #ifdef __CUDACC__
            cudaFreeHost(control_ptr_);
        #else
            delete control_ptr_;
        #endif
    }

    // Métricas
    float get_throughput() const { return control_ptr_->throughput; }
    float get_latency() const { return control_ptr_->latency; }
    uint32_t get_active_sms() const { return control_ptr_->active_sms; }
    uint64_t get_tasks_processed() const { return control_ptr_->tasks_processed; }
    uint32_t get_ring_buffer_usage() const {
        // Calcula a diferença entre head e tail
        uint32_t t = control_ptr_->tail.load();
        uint32_t h = control_ptr_->head.load();
        if (t >= h) return t - h;
        return (RING_BUFFER_SIZE - h) + t;
    }

    void launch_kernel() {
        // Configuration for H100 (Hopper)
        // 132 SMs, expecting 1 block per SM for maximum persistent occupancy
        int num_sms = 132; // This would be queried via cudaGetDeviceProperties
        int num_threads = 128; // Warp specialized groups (4 warps)

        void* args[] = { &control_ptr_ };

        std::cout << "[Runtime] Launching Persistent Kernel on " << num_sms << " SMs..." << std::endl;

        // Launch Cooperative Kernel to ensure all blocks are resident (Grid Synchronization capable)
        #ifdef __CUDACC__
            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)universal_persistent_kernel,
                dim3(num_sms),
                dim3(num_threads),
                args,
                0, // Shared Mem bytes
                0  // Stream
            );

            if (err != cudaSuccess) {
                std::cerr << "[Runtime] FATAL: Kernel Launch Failed: " << err << std::endl;
            }
        #else
             std::cout << "[Runtime] (Simulation) Starting Host-Side Ghost Consumer..." << std::endl;
             simulation_thread_ = std::thread(&UniversalKernelRuntime::run_simulation, this);
        #endif

        is_running_ = true;
    }

    void run_simulation() {
        // "Ghost Kernel" - Simula o consumo de tarefas pela GPU
        while (is_running_) {
            // Verifica se há comandos
            uint32_t h = control_ptr_->head.load();
            uint32_t t = control_ptr_->tail.load();

            if (h != t) {
                // Consome comando
                CommandPacket& pkt = control_ptr_->commands[h];

                // Simula latência de processamento (Kernel Execution)
                // 1.5 microsegundos (target)
                std::this_thread::sleep_for(std::chrono::microseconds(1));

                // Atualiza métricas
                control_ptr_->tasks_processed++;
                control_ptr_->latency = 1.1f + ((float)(rand() % 100) / 1000.0f); // ~1.1 - 1.2 us
                control_ptr_->throughput = 4000.0f + ((float)(rand() % 200)); // ~4000 TOPS
                control_ptr_->active_sms = 132;

                // Avança Head
                control_ptr_->head.store((h + 1) % RING_BUFFER_SIZE);
            } else {
                // Idle
                control_ptr_->active_sms = 0;
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }

    void submit_bitnet_gemm(int M, int N, int K, const void* A, const void* B, void* C) {
        if (!is_running_) return;

        // 0. Safety Checks (NaN/Inf/Null)
        if (!A || !B || !C) {
            std::cerr << "[Runtime] Error: Null pointer in submission." << std::endl;
            return;
        }

        // Although alpha/beta are hardcoded to 1.0/0.0 in the previous snippet,
        // if they were arguments, we would check them here.
        float alpha = 1.0f;
        float beta = 0.0f;

        if (std::isnan(alpha) || std::isinf(alpha) || std::isnan(beta) || std::isinf(beta)) {
             std::cerr << "[Runtime] Error: NaN/Inf detected in scalar parameters." << std::endl;
             return;
        }

        // 1. Prepare Descriptor
        // Ideally this memory is also Unified/Pinned so GPU can read it via Zero-Copy pointer
        BitNetGEMMParams* params;
        #ifdef __CUDACC__
            cudaMallocManaged((void**)&params, sizeof(BitNetGEMMParams));
        #else
            params = new BitNetGEMMParams;
        #endif

        params->M = M;
        params->N = N;
        params->K = K;
        params->A_ptr = A;
        params->B_ptr = B;
        params->C_ptr = C;
        params->alpha = 1.0f;
        params->beta = 0.0f;

        // 2. Push Command
        bool success = false;
        int retries = 0;
        while (!success && retries < 1000000) {
            success = rb_manager_->push_command(OpType::GEMM_BITNET, (uint64_t)params, task_counter_);
            if (!success) {
                // Backoff
                std::this_thread::yield();
                retries++;
            }
        }

        if (success) {
            task_counter_++;
        } else {
             std::cerr << "[Runtime] Error: Ring Buffer Full, dropped command." << std::endl;
        }
    }

    void shutdown() {
        if (!is_running_) return;

        // Enviar sinal de kill
        rb_manager_->push_command(OpType::SHUTDOWN, 0, task_counter_++);
        is_running_ = false;

        // Join simulation thread if active
        #ifndef __CUDACC__
        if (simulation_thread_.joinable()) {
            simulation_thread_.join();
        }
        #endif

        // Wait for GPU to finish
        #ifdef __CUDACC__
            cudaDeviceSynchronize();
        #endif
        std::cout << "[Runtime] Shutdown Complete." << std::endl;
    }

    // Helper para testes
    uhk::host::RingBufferManager* get_rb_manager() { return rb_manager_.get(); }

private:
    RingBufferControl* control_ptr_; // Shared Memory
    std::unique_ptr<uhk::host::RingBufferManager> rb_manager_;
    bool is_running_ = false;
    uint64_t task_counter_ = 0;
    std::thread simulation_thread_;
};

} // namespace runtime
} // namespace uhk
