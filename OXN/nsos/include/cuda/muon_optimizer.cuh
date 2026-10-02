#pragma once
#include "cuda/sparse_optimizer_activity.cuh"

// Scratch is caller-owned and reused serially on the descriptor's owning lane:
// x,y,direction each rows*cols; gram,poly each min(rows,cols)^2. Norm scalar one
// double. Inactive/aborted matrices never read weights/gradients/momentum/scratch.
// This modifies candidate momentum under an already armed bank rollback; it
// never publishes weight/cache/version/lazy metadata. Gradients are already
// averaged+clipped in deterministic mode; folded mode supplies sqsum+scale.
extern "C" bool launch_activity_muon_direction(NsosActivityAwareOptimizerDesc desc,
    int tensor, int rows, int cols, float* x, float* y, float* gram, float* poly,
    float* direction, double* norm, const float* sqsum, float accumulation_scale,
    float max_norm, int* output_issue);

// Atomicity belongs to the surrounding owner: all backward producers complete,
// global input gate/norm/snapshot precede this fused epilogue, and ALL candidate
// weights/moments are checked before version/cache/step publication. This fuses
// update+optional gradient clearing; it does not claim universal backward fusion
// or elimination of gradient allocation under accumulation/global clipping.
extern "C" bool launch_activity_hybrid_muon_adam_epilogue(
    NsosActivityAwareOptimizerDesc desc, const unsigned char* muon_mask,
    float* const* directions, const float* sqsum, float accumulation_scale,
    float max_norm, float beta1, float beta2, float bc1, float bc2, float eps,
    float weight_decay, bool clear_gradients, int* output_issue);
