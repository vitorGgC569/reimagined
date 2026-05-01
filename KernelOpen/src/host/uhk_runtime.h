#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ring_buffer_manager.h"
#include "../common/uhk_types.h"

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
typedef int cudaError_t;
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_value = 1, unsigned int y_value = 1,
         unsigned int z_value = 1)
        : x(x_value), y(y_value), z(z_value) {}
};
#define cudaSuccess 0
#define cudaHostAllocDefault 0
#define cudaHostAllocMapped 0
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size);
extern "C" cudaError_t cudaFree(void* devPtr);
extern "C" cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
extern "C" cudaError_t cudaLaunchCooperativeKernel(
    const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem,
    cudaStream_t stream);
extern "C" cudaError_t cudaDeviceSynchronize();
#endif

namespace uhk {
namespace runtime {

extern "C" void universal_persistent_kernel(RingBufferControl* control_ptr);

class UniversalKernelRuntime {
public:
    UniversalKernelRuntime() {
        std::cout << "[Runtime] Initializing Universal Heterogeneous Kernel Runtime..."
                  << std::endl;

#ifdef __CUDACC__
        cudaHostAlloc((void**)&control_ptr_, sizeof(RingBufferControl),
                      cudaHostAllocMapped);
#else
        control_ptr_ = new RingBufferControl();
#endif

        rb_manager_ = std::make_unique<uhk::host::RingBufferManager>(control_ptr_);
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

    float get_throughput() const { return control_ptr_->throughput; }
    float get_latency() const { return control_ptr_->latency; }
    uint32_t get_active_sms() const { return control_ptr_->active_sms; }
    uint64_t get_tasks_processed() const { return control_ptr_->tasks_processed; }

    uint32_t get_ring_buffer_usage() const {
        uint32_t tail = control_ptr_->tail.load();
        uint32_t head = control_ptr_->head.load();
        if (tail >= head) {
            return tail - head;
        }
        return (RING_BUFFER_SIZE - head) + tail;
    }

    void launch_kernel() {
        int num_sms = 132;
        int num_threads = 128;
        void* args[] = {&control_ptr_};

        std::cout << "[Runtime] Launching Persistent Kernel on " << num_sms
                  << " SMs..." << std::endl;

#ifdef __CUDACC__
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)universal_persistent_kernel, dim3(num_sms), dim3(num_threads),
            args, 0, 0);

        if (err != cudaSuccess) {
            std::cerr << "[Runtime] FATAL: Kernel Launch Failed: " << err
                      << std::endl;
        }
#else
        std::cout << "[Runtime] Starting host fallback backend..." << std::endl;
        simulation_thread_ =
            std::thread(&UniversalKernelRuntime::run_simulation, this);
#endif

        is_running_ = true;
    }

    void run_simulation() {
        while (is_running_) {
            uint32_t head = control_ptr_->head.load();
            uint32_t tail = control_ptr_->tail.load();

            if (head == tail) {
                control_ptr_->active_sms = 0;
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }

            CommandPacket& packet = control_ptr_->commands[head];
            if (packet.op_type == OpType::SHUTDOWN) {
                control_ptr_->head.store((head + 1) % RING_BUFFER_SIZE);
                break;
            }

            if (packet.op_type == OpType::GEMM_BITNET) {
                execute_host_gemm(packet);
            }

            control_ptr_->head.store((head + 1) % RING_BUFFER_SIZE);
        }
    }

    void submit_bitnet_gemm(int M, int N, int K, const void* A, const void* B,
                            void* C) {
        if (!is_running_) {
            return;
        }

        if (!A || !B || !C) {
            std::cerr << "[Runtime] Error: Null pointer in submission."
                      << std::endl;
            return;
        }

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        if (std::isnan(alpha) || std::isinf(alpha) || std::isnan(beta) ||
            std::isinf(beta)) {
            std::cerr << "[Runtime] Error: NaN/Inf detected in scalar parameters."
                      << std::endl;
            return;
        }

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
        params->alpha = alpha;
        params->beta = beta;

        bool success = false;
        int retries = 0;
        while (!success && retries < 1000000) {
            success = rb_manager_->push_command(OpType::GEMM_BITNET,
                                                reinterpret_cast<uint64_t>(params),
                                                task_counter_);
            if (!success) {
                std::this_thread::yield();
                ++retries;
            }
        }

        if (success) {
            ++task_counter_;
        } else {
            std::cerr << "[Runtime] Error: Ring Buffer Full, dropped command."
                      << std::endl;
#ifndef __CUDACC__
            delete params;
#endif
        }
    }

    void shutdown() {
        if (!is_running_) {
            return;
        }

        rb_manager_->push_command(OpType::SHUTDOWN, 0, task_counter_++);
        is_running_ = false;

#ifndef __CUDACC__
        if (simulation_thread_.joinable()) {
            simulation_thread_.join();
        }
#endif

#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif
        std::cout << "[Runtime] Shutdown Complete." << std::endl;
    }

    uhk::host::RingBufferManager* get_rb_manager() { return rb_manager_.get(); }

private:
    void execute_host_gemm(const CommandPacket& packet) {
        auto* params = reinterpret_cast<BitNetGEMMParams*>(packet.param_ptr);
        if (!params || !params->A_ptr || !params->B_ptr || !params->C_ptr) {
            throw std::runtime_error("Invalid host GEMM submission");
        }

        const auto start = std::chrono::high_resolution_clock::now();

        const float* a_ptr = static_cast<const float*>(params->A_ptr);
        const float* b_ptr = static_cast<const float*>(params->B_ptr);
        float* c_ptr = static_cast<float*>(params->C_ptr);

        for (int row = 0; row < params->M; ++row) {
            for (int col = 0; col < params->N; ++col) {
                float acc = 0.0f;
                for (int k = 0; k < params->K; ++k) {
                    acc += a_ptr[row * params->K + k] *
                           b_ptr[k * params->N + col];
                }
                c_ptr[row * params->N + col] =
                    params->alpha * acc + params->beta * c_ptr[row * params->N + col];
            }
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration<double, std::micro>(end - start).count();
        const double elapsed_s = std::max(elapsed_us / 1'000'000.0, 1e-9);
        const double operations =
            static_cast<double>(2LL) * params->M * params->N * params->K;

        control_ptr_->tasks_processed++;
        control_ptr_->latency = static_cast<float>(elapsed_us);
        control_ptr_->throughput =
            static_cast<float>(operations / elapsed_s / 1.0e12);
        control_ptr_->active_sms = 1;

#ifndef __CUDACC__
        delete params;
#endif
    }

    RingBufferControl* control_ptr_;
    std::unique_ptr<uhk::host::RingBufferManager> rb_manager_;
    bool is_running_ = false;
    uint64_t task_counter_ = 0;
    std::thread simulation_thread_;
};

} // namespace runtime
} // namespace uhk
