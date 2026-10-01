#pragma once
#include "gpu_linear_view.h"
#include "cuda/sparse_optimizer_activity.cuh"

extern "C" {
// New API: both destination banks must pass host preflight before activity
// union update. Device current/first_write control this write; no host add_mask.
bool launch_moe_training_accumulate_gradients_activity(
    const nsos::GpuMoeGradientView* views, NsosSparseOptimizerActivity activity,
    const float* weight, const float* bias, const float* magnitude,
    int weight_elements, int outputs, float scale = 1.0f);
// Active-only QAT penalty: recompute absmean+ternary on the UNION rather
// than using a possibly stale last-microbatch workspace. Adds A*base*diff to
// already-contributed gradients; returns 0.5*base*sum(diff^2). FP64 ordered
// reduction, one scalar loss per bank. partials[experts*blocks] is reused.
bool launch_moe_training_active_qat_regularization(
    const nsos::GpuMoeTrainingLinearView* views,
    const nsos::GpuMoeGradientView* gradients, NsosSparseOptimizerActivity activity,
    int* union_offsets, float* partials, double* loss_partials, float* loss, int elements, float base_coefficient, int accumulation_steps);
// One bank-wide launch, no per-expert slices/copies or atomics. Offsets guard
// inactive scratch before any source or destination is fetched.
bool launch_moe_training_accumulate_gradients(const nsos::GpuMoeGradientView* views,
    const int* offsets, const float* weight, const float* bias,
    const float* magnitude, int experts, int weight_elements, int outputs);
// Effective-weight and scale storage referenced by QAT descriptors is writable
// and owned by GpuMoeTraining. Reference descriptors are never modified.
bool launch_moe_training_prepare_qat(const nsos::GpuMoeTrainingLinearView* views,
    const int* offsets, float* partials, int experts, int elements);
bool launch_moe_training_route(const float* routing, int* counts, int* offsets,
    int* permutation, int* inverse, float* scales, int rows, int experts, int capacity);
bool launch_moe_training_linear_forward(const nsos::GpuMoeTrainingLinearView* views,
    const int* offsets, const int* permutation, const float* input, bool gather,
    float* normalized, float* prepared, float* inverse_rms, float* pre, float* post,
    float* squared_output, int rows, int capacity, int experts, int inputs, int outputs,
    int matmul_mode, bool use_wmma = false);
bool launch_moe_training_linear_backward(const nsos::GpuMoeTrainingLinearView* views,
    const int* offsets, const int* permutation, const float* scales,
    const float* upstream, bool gather, const float* squared_pre,
    const float* normalized, const float* prepared, const float* inverse_rms,
    const float* pre, float* grad_out, float* grad_pre, float* grad_input,
    float* grad_weight, float* grad_bias, float* grad_magnitude,
    int rows, int capacity, int experts, int inputs, int outputs, int matmul_mode, bool use_wmma = false);
bool launch_moe_training_combine(const float* values, const int* inverse,
    const int* offsets, const float* scales, float* output,
    int rows, int dim, int experts);
bool launch_moe_training_router_grad(const float* grad, const float* expert_out,
    const int* inverse, const int* offsets, float* router_grad,
    int rows, int dim, int experts);
}
