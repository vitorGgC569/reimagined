#pragma once
#include <memory>
#include <stdexcept>
#include <cstdint>
#ifdef USE_CUDA
#include "gpu_execution.h"
#include "cuda/device_buffer.h"
#include "cuda/sparse_optimizer_activity.cuh"
#endif

namespace nsos {
class GpuMoeTraining;
// Runtime-only ownership, never serialized pointers. One domain per MoE block
// and accumulation group. Parameters keep it alive until explicitly unbound.
class GpuGradientActivity {
public:
    explicit GpuGradientActivity(int experts) : experts_(experts) {
        if (experts <= 0 || experts > 1024) throw std::invalid_argument("Invalid device activity domain");
#ifdef USE_CUDA
        check(cudaGetDevice(&device_)); stream_ = gpu::current_stream();
        if (!accumulated_.ensure(experts) || !current_.ensure(experts) ||
            !first_.ensure(experts) || !issue_.ensure(1)) throw std::bad_alloc();
        reset();
#else
        throw std::runtime_error("Device gradient activity requires GPU build");
#endif
    }
    float gradient_scale = 1.0f; // immutable during a group, inverse loss scale
    int experts() const noexcept { return experts_; }
    void set_producer(const std::shared_ptr<GpuMoeTraining>& producer) { producer_ = producer; }
    std::shared_ptr<GpuMoeTraining> producer() const { return producer_.lock(); }
#ifdef USE_CUDA
    NsosSparseOptimizerActivity view() {
        assert_lane();
        return {accumulated_.get(), current_.get(), first_.get(), issue_.get(), experts_};
    }
    const unsigned char* predicate(int expert) const {
        if (expert < 0 || expert >= experts_) throw std::out_of_range("Device activity expert");
        assert_lane(); return accumulated_.get() + expert;
    }
    void assert_lane() const {
        int selected = -1; check(cudaGetDevice(&selected));
        if (selected != device_ || gpu::current_stream() != stream_)
            throw std::logic_error("Device activity requires its owning device/stream");
    }
    void reset() {
        assert_lane();
        if (!launch_sparse_optimizer_activity_reset(view())) throw std::runtime_error("Activity reset launch failed");
    }
    void abort() {
        assert_lane();
        if (!launch_sparse_optimizer_activity_abort(view())) throw std::runtime_error("Activity abort launch failed");
    }
    const int* issue() const { assert_lane(); return issue_.get(); }
private:
    static void check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }
    cuda_detail::DeviceBuffer<unsigned char> accumulated_, current_, first_;
    cuda_detail::DeviceBuffer<int> issue_;
    cudaStream_t stream_ = nullptr;
    int device_ = -1;
#endif
    std::weak_ptr<GpuMoeTraining> producer_;
    int experts_;
};
struct DeviceGradientBinding {
    std::shared_ptr<GpuGradientActivity> owner;
    int expert = -1;
};
} // namespace nsos
