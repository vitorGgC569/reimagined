#pragma once
#include "tensor.h"
#include "gpu_linear_view.h"
#include "gpu_gradient_activity.h"
#include <cstdint>
#include <vector>
#include <utility>
#ifdef USE_CUDA
#include "cuda/device_buffer.h"
#endif

namespace nsos {
class BitLinear;
class Parameter;

// Borrowed parameter/effective-weight metadata, never a serialized format.
// The grouped owner retains QAT effective weights/scales until backward.
// Exact sparse membership with fixed-capacity storage, not dense all-expert
// evaluation. Device offsets delimit every GEMM; no forward counts download.
// Legacy callers keep the host registry; device groups bind GPU predicates.
class GpuMoeTraining : public std::enable_shared_from_this<GpuMoeTraining> {
public:
    ~GpuMoeTraining();
    // Begin once before any microbatch; call finish only after the transaction
    // committed/rolled back on this lane. Tape discard is NOT a group reset.
    void begin_device_accumulation(const std::vector<BitLinear*>& up,
        const std::vector<BitLinear*>& down);
    void finish_device_accumulation(bool abort, bool materialize_for_audit = false);
    // Active-only QAT objective is explicit and separately versioned. Invoke
    // once per group after ALL task microbatches, before optimizer scaling.
    Tensor add_device_qat_regularization(float base_coefficient, int accumulation_steps);
    std::shared_ptr<GpuGradientActivity> device_activity() const { return activity_; }
    Tensor forward(const Tensor& input, const Tensor& routing,
        const std::vector<BitLinear*>& up, const std::vector<BitLinear*>& down,
        int top_k);
    std::pair<Tensor, Tensor> backward(const Tensor& grad, bool router_gradient);
    std::vector<int> materialize_counts(); // explicit audit/registry boundary
    void discard_backward_state();
private:
    struct LinearState {
        std::vector<BitLinear*> layers;
        std::vector<GpuMoeTrainingLinearView> views;
        std::vector<Tensor> effective_weights, qat_scales;
        std::vector<Tensor> regularization_weights, regularization_scales;
        std::vector<GpuMoeTrainingLinearView> regularization_views;
        std::vector<uint64_t> weight_versions, magnitude_versions, bias_versions;
        std::vector<GpuMoeTrainingLinearView> uploaded_views;
        std::vector<GpuMoeGradientView> gradient_views, uploaded_gradient_views;
#ifdef USE_CUDA
        cuda_detail::DeviceBuffer<GpuMoeTrainingLinearView> device_views;
        cuda_detail::DeviceBuffer<float> qat_partials;
        cuda_detail::DeviceBuffer<double> qat_loss_partials;
        cuda_detail::DeviceBuffer<GpuMoeTrainingLinearView> regularization_device_views;
        GpuMoeTrainingLinearView* uploaded_device = nullptr;
        cuda_detail::DeviceBuffer<GpuMoeGradientView> device_gradient_views;
        GpuMoeGradientView* uploaded_gradient_device = nullptr;
#endif
        Tensor normalized, prepared, inverse_rms, pre, post;
        Tensor grad_out, grad_pre, grad_input, grad_weight, grad_bias, grad_magnitude;
    };
    void prepare_linear(LinearState& state, const std::vector<BitLinear*>& layers);
    void allocate_gradients(LinearState& state);
    void verify_parameters(const LinearState& state) const;
    void prepare_gradient_commit(LinearState& state, const std::vector<int>& counts);
    void enqueue_gradient_commit(LinearState& state);
    void publish_gradient_commit(LinearState& state, const std::vector<int>& counts);
    void prepare_device_gradient_commit(LinearState& state);
    std::shared_ptr<GpuGradientActivity> activity_;
    std::vector<Parameter*> bound_parameters_;
    std::vector<BitLinear*> bound_up_, bound_down_;
    LinearState up_, down_;
    std::vector<float*> gradient_destinations_; // reusable alias-check scratch
    Tensor input_, hidden_, scales_;
#ifdef USE_CUDA
    struct RetiredTape { cudaEvent_t ready; std::vector<Tensor> storage; };
    std::vector<RetiredTape> retired_tapes_;
    cuda_detail::DeviceBuffer<int> counts_, offsets_, permutation_, inverse_;
    Tensor qat_group_loss_;
    cuda_detail::DeviceBuffer<int> qat_union_offsets_;
#endif
    std::vector<int> input_shape_;
    int rows_ = 0, dim_ = 0, hidden_dim_ = 0, experts_ = 0, capacity_ = 0;
    int matmul_mode_ = 0;
    bool wmma_enabled_ = false;
    bool pending_ = false;
};
} // namespace nsos
