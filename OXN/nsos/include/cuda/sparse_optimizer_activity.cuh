#pragma once

#include "cuda/kernels.cuh"

// Caller owns three disjoint device byte arrays [experts] and a device int.
// One activity object per accumulation group/device/stream; never global.
// Reset only at begin-group or after an ordered abort/finished step. Preserve
// accumulated across microbatches; current and first_write describe the latest
// successful update. first_write[e] => overwrite; current[e] && !first_write[e]
// => add. A contributed numerical zero is active. Buffer presence is irrelevant.
// All producers/consumers/reset must be ordered on current_stream(), or joined
// by explicit events. Caller must retain offsets/status/masks/descriptors and
// gradient/moment buffers until all queued work is finished; no hidden allocation.
// Bool results report host-argument/launch acceptance only. Device status must
// govern all later work. Any false return requires the caller to abort; it does
// not promise a device abort was enqueued. Launch faults/overflow need the
// existing transaction/snapshot rollback, including logical lazy-state metadata.
typedef struct NsosSparseOptimizerActivity {
  unsigned char* accumulated;
  unsigned char* current;
  unsigned char* first_write;
  int* abort_issue;  // sticky until explicit reset; never clear mid-group
  int experts;
} NsosSparseOptimizerActivity;

extern "C" {
bool launch_activity_criticality_metrics(float* const* weights,
    const unsigned long long* offsets, const int* fan_in, int count,
    const unsigned char* const* predicates, const int* const* issues,
    float* gammas, float* gains);
bool launch_activity_criticality_gradient(float* const* weights, float* const* gradients,
    const unsigned long long* offsets, const float* coefficients, int count,
    const unsigned char* const* predicates, const int* const* issues);
// Transaction helpers. State/version/cache publication follows the completed
// post-update finite gate; versions increment only for active tensors, even if
// their contribution was numerically zero. cache_valid may be null.
bool launch_activity_merge_abort(int* issue, const int* const* sources, int count);
bool launch_activity_preflight_versions(NsosActivityAwareOptimizerDesc desc, const uint64_t* versions, int* issue);
bool launch_activity_publish_versions(NsosActivityAwareOptimizerDesc desc,
    uint64_t* versions, unsigned char* cache_valid, const int* output_issue);
// Ordered snapshot/restore of existing active storage, before scaling/init.
// scratch has 4*total floats (w,g,m,v); previously absent moments are not read.
// backup_initialized has n_tensors bytes. Restore is deliberately ungated by
// abort_issue; failed groups must restore the values they changed.
bool launch_activity_snapshot(NsosActivityAwareOptimizerDesc desc,
    const unsigned char* initialized, unsigned char* backup_initialized,
    float* scratch);
bool launch_activity_restore(NsosActivityAwareOptimizerDesc desc,
    unsigned char* initialized, const unsigned char* backup_initialized,
    const float* scratch);
bool launch_sparse_optimizer_activity_reset(NsosSparseOptimizerActivity activity);
// Offsets must start at zero and be monotonic/nonnegative, with the terminal
// offset <= capacity. The actual assignment count is offsets[experts], read
// only on device, and may be smaller than capacity (masked/empty routing).
// routing_issue, when supplied, is a stable upstream device status; the current
// MoE routing layout supplies offsets + experts + 1 as this pointer.
// Whole-offset preflight precedes any union publication. Invalid routing clears
// current/first_write, leaves accumulated untouched, and latches abort_issue.
// Call only after BOTH gradient-bank destination preflights succeed; a failed
// downstream gradient commit must abort the whole group before optimization.
bool launch_sparse_optimizer_activity_update(NsosSparseOptimizerActivity activity,
    const int* offsets, int capacity, const int* routing_issue = nullptr);
bool launch_sparse_optimizer_activity_abort(NsosSparseOptimizerActivity activity);

// New entry points leave every existing kernels.cuh launch signature unchanged.
// Null contribution predicates preserve dense behavior; inactive must not read
// gradients/weights/moments, clip, update, or decay. abort_issue gates dense too.
// Preflight must precede state initialization and Adam; found_issue must be the
// stable desc.contribution.abort_issue for their later launches. Validates the
// entire bank (including complete ordered chunk coverage when supplied) before
// any writes. Active storage must already be reserved, but need not be logically
// initialized. Restored moments need a separate activity-aware finite scan.
// found_issue must equal desc.contribution.abort_issue (enforced by the API).
bool launch_activity_multi_tensor_preflight(int* found_issue,
    NsosActivityAwareOptimizerDesc desc,
    const NsosMultiTensorChunk* chunks = nullptr, int chunk_count = 0);
// In-place accumulation scaling must be activity-gated too. Call after the
// pre-update finite gate; clipping alone does not apply accumulation scaling.
bool launch_activity_multi_tensor_scale_gradients(
    NsosActivityAwareOptimizerDesc desc, float scale);
bool launch_activity_multi_tensor_sqsum(float* accum,
    NsosActivityAwareOptimizerDesc desc);
bool launch_activity_multi_tensor_check_finite(int* found_issue,
    const float* const* values, const unsigned char* require_nonnegative,
    NsosActivityAwareOptimizerDesc desc,
    const NsosMultiTensorChunk* chunks = nullptr, int chunk_count = 0,
    const unsigned char* initialized = nullptr);
// Supply initialized only for EXISTING moment scans: active, logically absent
// moments are skipped before fetching values. Weight/gradient scans pass null.
// Scan existing active moments before lazy initialization, so a later gate
// failure cannot publish new moment presence in an aborted group.
// Deterministic chunk tree256 FP64 partials, ordered finalize, then in-place
// FP32 clip. partials[chunk_count], total[1], coefficient[1], found_issue[1]
// are caller-owned. found_issue is sticky and must be zeroed at begin-group;
// invalid/aborted/nonfinite input prevents every gradient write.
bool launch_activity_chunked_norm_device_clip(double* total, double* partials,
    float* coefficient, int* found_issue, NsosActivityAwareOptimizerDesc desc,
    const NsosMultiTensorChunk* chunks, int chunk_count, float max_norm);
// Initialize ONLY active, previously uninitialized moment entries. Storage is
// already reserved by the caller; initialized[n_tensors] records logical lazy
// presence, independent of reserved storage. Restore/checkpoint this metadata.
// Order pre-update finite gates before initialization; no rollback is provided.
bool launch_activity_multi_tensor_initialize_moments(
    NsosActivityAwareOptimizerDesc desc, unsigned char* initialized);
// Gradients already scaled/clipped. learning_rates are absolute per-tensor lr,
// matching the legacy deterministic lane; bias correction uses caller scalars.
// desc.contribution.abort_issue must include the stable pre-update finite gate.
// found_nonfinite is an OUTPUT status; keep it distinct from abort_issue so an
// update-time overflow cannot cause scheduling-dependent partial skipping.
bool launch_activity_multi_tensor_adamw_update_deterministic(
    NsosActivityAwareOptimizerDesc desc,
    const NsosMultiTensorChunk* chunks, int chunk_count,
    float beta1, float beta2, float bc1, float bc2, float eps,
    float weight_decay, int* found_nonfinite);
// Atomic FP32 norm/folded accumulation+clip legacy lane, also predicate-aware.
// lr multiplies optional learning_rates (scales in this lane).
bool launch_activity_multi_tensor_adamw(NsosActivityAwareOptimizerDesc desc,
    const float* gradient_sq_sum, float accumulation_scale, float max_grad_norm,
    float beta1, float beta2, float bc1, float bc2, float lr, float eps,
    float weight_decay, int* found_nonfinite);
}
