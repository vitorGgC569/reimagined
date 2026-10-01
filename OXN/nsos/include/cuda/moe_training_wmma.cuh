#pragma once
#include "gpu_linear_view.h"

// Selection is independent of routing payloads: all segment sizes remain on
// device. Small/FP32 shapes intentionally retain the scalar arithmetic policy.
inline bool moe_training_wmma_geometry(int rows, int inputs, int outputs, int mode) {
    return rows >= 16 && inputs >= 32 && outputs >= 32 && (mode == 1 || mode == 2);
}
bool moe_training_wmma_supported();
// Kind 0: FWD; 1: dX; 2: dW. Returns false, never vendor/scalar fallback,
// if a selected WMMA launch cannot run on this binary/current device.
bool launch_moe_training_wmma_gemm(int kind,
    const nsos::GpuMoeTrainingLinearView* views, const int* offsets,
    const float* a, const float* b, float* output, float* post, float* squared,
    int rows, int experts, int inputs, int outputs, int mode);
