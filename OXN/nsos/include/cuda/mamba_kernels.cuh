#ifndef MAMBA_KERNELS_CUH
#define MAMBA_KERNELS_CUH

#ifdef USE_CUDA
#include "../gpu_backend.h"
#endif
#include <cstddef>

namespace nsos {
namespace cuda {

inline constexpr int kFaithfulReducedConvMaxKernel = 16;
inline constexpr int kFaithfulDecayTermsWidth = 5;

#ifdef USE_CUDA
// (launch_mamba_ssd_forward removido — kernel chunked morto, zero callers.)

// =====================================================================
// Selective scan with B/C gating, matched 1:1 against the CPU
// implementation in src/mamba2.cpp::Mamba2SSD::ssd_forward.
//
// Recurrence (per (batch, dim) channel, sequential over time):
//   decay     = exp(-softplus(dt[t]) * exp(A_log[d]))            (N1)
//   state[d]  = state[d] * decay
//               + softplus(dt[t,d]) * B_in[t,d] * x[t,d]
//   y[t,d]    = tanh(state[d]) * C_in[t,d]
//
// Tensors (all device pointers):
//   x, dt, B_in, C_in, y :  [Batch, Seq, D]
//   A                    :  [D]
//   state_history (opt)  :  [Batch, Seq, D]   nullptr to skip recording
//
// state_history records the SSM state AFTER the update at each timestep
// so that the matching backward kernel can recompute gradients without
// re-running the forward pass.  Pass nullptr for inference paths that
// do not need backward.
//
// This launcher does NOT call cudaDeviceSynchronize().  Callers that
// must serialize with the host should sync explicitly (e.g. before
// reading the output on the CPU).
// =====================================================================
void launch_mamba_selective_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D);

// Runtime toggle for the OPT-IN parallel-prefix (associative) selective scan
// (default from NSOS_MAMBA_PARALLEL_SCAN env).  OFF -> the validated channel-
// parallel sequential kernel.  Exposed so the parity test / Python A/B can
// switch paths in a single process.  Must clear the 1e-4 parity gate before
// being promoted to default.
void set_mamba_parallel_scan(bool enabled);
bool mamba_parallel_scan_enabled();

// =====================================================================
// Selective scan backward, paired with the forward kernel above.
//
// Inputs (all device pointers):
//   grad_y         : [Batch, Seq, D]   upstream gradient
//   x, dt, B_in, C_in : [Batch, Seq, D]   forward inputs
//   A              : [D]                  state-decay weights
//   state_history  : [Batch, Seq, D]      from forward(state_history != null)
//
// Outputs (all device pointers, must be pre-allocated and pre-zeroed
// where appropriate — grad_A is accumulated with atomicAdd so it MUST
// be cudaMemset to 0 before this launcher is called):
//   grad_x   : [Batch, Seq, D]
//   grad_dt  : [Batch, Seq, D]
//   grad_A   : [D]
//   grad_B   : [Batch, Seq, D]
//   grad_C   : [Batch, Seq, D]
//
// Same no-sync contract as the forward launcher.
// =====================================================================
void launch_mamba_selective_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D);

// (launch_mamba_simple_scan_forward/backward removidos — mortos.)

void launch_mamba_single_token_update(const float *x, const float *dt,
                                      const float *A, float *state, float *y,
                                      int Batch, int D);
// The launcher is asynchronous on the current/default stream, like every
// other Mamba launcher in this header. Device consumers are ordered by the
// stream; host consumers must establish their own synchronization boundary.

// =====================================================================
// Proper diagonal SSM (linear readout y = h*C, no tanh) — GPU-resident
// path for Mamba2SSD::forward_proper/backward_proper.  Same affine
// recurrence + state_history contract as the legacy selective scan.
// Tensors: x, dt, B_in, C_in, y : [Batch, Seq, D]; A : [D];
// state_history : [Batch, Seq, D] (h_t after each update; nullptr to skip).
// No internal cudaDeviceSynchronize.
// =====================================================================
void launch_mamba_proper_scan_forward(
    const float *x, const float *dt, const float *A, const float *B_in,
    const float *C_in, float *y, float *state_history, int Batch, int Seq,
    int D);

void launch_mamba_proper_scan_backward(
    const float *grad_y, const float *x, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *grad_x, float *grad_dt, float *grad_A, float *grad_B,
    float *grad_C, int Batch, int Seq, int D);

// Fused single-token incremental decode step for the proper diagonal path.
// xv/z/B_in/C_in/dt/A are [dim]; convw is [dim*K]; ring is [(K-1)*dim] (in/out,
// the carried conv window); h is [dim] (in/out, the carried SSD state); gated is
// [dim] (out).  Device-resident carry -> no per-token host round-trip.
void launch_mamba_proper_step(const float *xv, const float *z,
                              const float *B_in, const float *C_in,
                              const float *dt, const float *A,
                              const float *convw, float *ring, float *h,
                              float *gated, int dim, int K);

// Fused single-token incremental decode step for the N-STATE path (full
// Mamba-2).  xv/z are [dim]; B_in/C_in are [H*N] (per-head N-dim); dt/A are
// [H] (per-head, A in log-domain); convw is [dim*K]; ring is [(K-1)*dim]
// (in/out); h is [dim*N] == [H*P*N] (in/out, the carried N-state); gated is
// [dim] (out).  Mirrors Mamba2SSD::forward_proper_nstate_step's host loop
// 1:1; device-resident carry -> no per-token host round-trip; CUDA-graph
// capturable.
void launch_mamba_nstate_step(const float *xv, const float *z,
                              const float *B_in, const float *C_in,
                              const float *dt, const float *A,
                              const float *convw, float *ring, float *h,
                              float *gated, int dim, int K, int P, int N);

// Prime the device-resident incremental carry directly from a completed
// full-sequence prefill. `history` is [batch*seq,state_width]. The convolution
// ring concatenates x/B/C planes per timestep; B/C may be null with width 0.
// Every output element, including left padding and the K=1 sentinel, is
// written by the kernel. This avoids downloading the full history/projections
// to host and uploading their tails again before the first decode token.
bool launch_mamba_prime_stream_carry(
    const float *history, int state_width,
    const float *x_values, int x_width,
    const float *b_values, int b_width,
    const float *c_values, int c_width,
    float *state, float *ring,
    int batch, int seq, int taps);

// =====================================================================
// Causal depthwise conv1d (proper-path local mixing).
//   in/out : [Batch, Seq, D]   weight : [D, K]
//   out[b,t,c] = sum_{j} weight[c,j] * in[b, t-(K-1)+j, c]   (in[<0]=0)
// backward: grad_in and grad_weight accumulate via atomicAdd; the
// launcher zeroes both before the kernel.  No internal sync.
// =====================================================================
void launch_conv1d_causal_forward(const float *in, const float *weight,
                                  float *out, int batch, int seq, int dim,
                                  int K);
// Faithful-path fused producer: computes the causal depthwise convolution,
// adds the per-channel bias, stores the exact pre-activation required by
// backward, and materializes SiLU(pre) in one pass.
bool launch_conv1d_causal_bias_silu_forward(
    const float *in, const float *weight, const float *bias,
    float *pre_activation, float *activated, int batch, int seq, int dim,
    int K);
// Faithful Mamba uses three independently projected channel groups (x/B/C)
// backed by one contiguous convolution weight/bias tensor. This launcher
// routes those three inputs/outputs inside one grid, eliminating two launches
// without materializing a concatenated activation.
bool launch_mamba2_faithful_conv_forward(
    const float *x_in, const float *b_in, const float *c_in,
    const float *weight, const float *bias, float *x_pre, float *b_pre,
    float *c_pre, float *x_out, float *b_out, float *c_out, int batch,
    int seq, int inner, int group_state, int K);
void launch_conv1d_causal_backward(const float *grad_out, const float *in,
                                   const float *weight, float *grad_in,
                                   float *grad_weight, int batch, int seq,
                                   int dim, int K);
// Fixed-order one-block-per-channel backward for K<=16. Returns false when
// the shape is outside that deterministic kernel's contract.
bool launch_conv1d_causal_backward_deterministic(
    const float *grad_out, const float *in, const float *weight,
    float *grad_in, float *grad_weight, int batch, int seq, int dim, int K);
bool launch_mamba2_faithful_conv_backward(
    const float *gx_out, const float *gb_out, const float *gc_out,
    const float *x_in, const float *b_in, const float *c_in,
    const float *weight, float *gx_in, float *gb_in, float *gc_in,
    float *grad_weight, int batch, int seq, int inner, int group_state,
    int K);
// ModelConfig's K<=16 path uses one deterministic block reduction per
// depthwise channel: grad_input is written exclusively and grad_weight is
// reduced without global atomics or pre-zero memsets. Larger direct-API
// kernels retain a correct generic atomic fallback.
// Deterministic one-block-per-channel bias reduction over the three faithful
// activation groups, writing directly into the contiguous bias-gradient
// layout. No temporary reductions or D2D concatenation are required.
bool launch_mamba2_faithful_bias_backward(
    const float *gx_pre, const float *gb_pre, const float *gc_pre,
    float *grad_bias, int rows, int inner, int group_state);

// Packs the five independent faithful input-projection gradients into the
// official combined projection layout [z,x,B,C,dt]. This enables one dX GEMM
// and one dW GEMM while keeping every canonical Parameter/checkpoint name.
bool launch_mamba2_pack_projection_grads(
    const float *gx, const float *gz, const float *gb, const float *gc,
    const float *gdt, float *packed, int rows, int inner, int group_state,
    int heads);
// One-pass inverse of the canonical [z,x,B,C,dt] column packing. For the
// sensitive [B,C,dt] layout, z/x are omitted and may be null.
bool launch_mamba2_unpack_projection(
    const float *packed, float *z, float *x, float *b, float *c, float *dt,
    int rows, int inner, int group_state, int heads, bool sensitive_only);
// QAT keeps z/x on their ternary BitLinear paths but B/C/dt in exact
// precision. Pack only the sensitive gradients in [B,C,dt] order so those
// three projections still share one dW GEMM and one dX GEMM.
bool launch_mamba2_pack_sensitive_projection_grads(
    const float *gb, const float *gc, const float *gdt, float *packed,
    int rows, int group_state, int heads);

// =====================================================================
// Full Mamba-2 SSD with N-dimensional state expansion (GPU-resident).
// One thread per (batch, head, p-channel), N-vector state in registers.
//   xc,y : [B,Seq,dim] (dim=H*P, channel=h*P+p)
//   dt   : [B,Seq,H]   A : [H]   B_in,C_in : [B,Seq,H,N]
//   state_history : [B,Seq,dim,N]  (h_t after update; nullptr to skip)
// N must be <= mamba_nstate_max_n(); the launcher no-ops otherwise so the
// caller falls back to the host scan. The fast backward accumulates shared
// gradients with atomicAdd. The deterministic backward assigns one block to
// each head and traverses batch/sequence/channel/state in a fixed order.
// =====================================================================
int mamba_nstate_max_n();
bool mamba_nstate_deterministic_backward_supported(int P, int N);
void launch_mamba_nstate_forward(const float *xc, const float *dt,
                                 const float *A, const float *B_in,
                                 const float *C_in, float *y,
                                 float *state_history, int Batch, int Seq, int H,
                                 int P, int N);
void launch_mamba_nstate_backward(const float *gy, const float *xc,
                                  const float *dt, const float *A,
                                  const float *B_in, const float *C_in,
                                  const float *state_history, float *gXc,
                                  float *gDt, float *gA, float *gB, float *gC,
                                  int Batch, int Seq, int H, int P, int N);
bool launch_mamba_nstate_backward_deterministic(
    const float *gy, const float *xc, const float *dt, const float *A,
    const float *B_in, const float *C_in, const float *state_history,
    float *gXc, float *gDt, float *gA, float *gB, float *gC, int Batch,
    int Seq, int H, int P, int N);

// Faithful Mamba-2 SSD.  B/C are [B,Seq,G,N] and shared by the heads in
// each group; D is [H] and contributes D[h]*x inside the normalized SSM
// branch.  This matches state-spaces/mamba Mamba2 with D_has_hdim=false.
// Backward aggregates channel-shared gradients within each hardware
// warp/wave before global atomic publication when P is warp-aligned; arbitrary
// head widths retain the generic atomic path with identical semantics.
bool launch_mamba2_faithful_forward(
    const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in,
    const float *C_in, const float *D, float *y, float *state_history,
    int Batch, int Seq, int H, int P, int N, int G,
    int runtime_warp_size, bool state_major_history, int history_chunk_size = 0);
// Time-parallel forward scan: summary -> prefix -> apply, mirroring the
// decomposition the deterministic backward already uses.  Raises the resident
// wavefront count by the chunk factor at the cost of replaying each chunk
// twice.  Returns false (leaving every output untouched) when the shape is not
// eligible, so the caller must fall back to the sequential launcher.
// `chunk_end_local` and `carry_entering` each need Batch*H*P*N*chunks floats
// and `chunk_total_decay` needs Batch*H*chunks; query the chunk count with
// faithful_forward_chunk_count().
bool launch_mamba2_faithful_forward_chunked(
    const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D, float *y, float *state_history,
    float *chunk_end_local, float *chunk_total_decay,
    float *carry_entering, int Batch, int Seq, int H, int P, int N,
    int G, int runtime_warp_size, bool state_major_history,
    int *out_chunks, int history_chunk_size = 0);
// Opt-in training policy: retain entering states for 32-token chunks and
// reconstruct one state's local history in LDS during backward, never S*P*N.
bool faithful_boundary_history_enabled();
bool faithful_chunked_forward_enabled();
int faithful_forward_chunk_size();
// Computes the five stable decay scalars once per (batch,time,head), replacing
// 64 identical transcendental evaluations in production P=64 scan blocks.
bool launch_mamba2_faithful_precompute_decay(
    const float *dt, const float *A, float *packed_terms,
    int Batch, int Seq, int H);
bool launch_mamba2_faithful_backward(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D,
    const float *state_history, float *gX, float *gDt, float *gA,
    float *gB, float *gC, float *gD, int Batch, int Seq, int H, int P,
    int N, int G, bool state_major_history);
// Deterministic GPU backward for warp-aligned head widths. The first kernel
// publishes one fixed-order partial per wave; three second-stage kernels fold
// those partials without atomics. Workspace element counts are:
//   B/C: Batch*Seq*G*N*((H/G)*(P/runtime_warp_size)) each
//   dt : Batch*Seq*H*(P/runtime_warp_size)
//   A/D: Batch*H*(P/runtime_warp_size) each
//   gh : Batch*Seq*H*P*N when the exact-order state-parallel path is enabled.
bool launch_mamba2_faithful_backward_deterministic(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D,
    const float *state_history, float *gX, float *gDt, float *gA,
    float *gB, float *gC, float *gD, float *partial_B,
    float *partial_C, float *partial_dt, float *partial_A,
    float *partial_D, float *gh_history, int Batch, int Seq, int H,
    int P, int N, int G, int runtime_warp_size,
    bool state_major_history);
// Time-chunked deterministic reverse scan. Each chunk preserves the native
// reverse arithmetic internally; a compact affine carry summary connects
// chunks before the final gradient pass. The additional workspaces contain:
//   carry: Batch*H*P*N*ceil(Seq/chunk_size)
//   scale: Batch*H*ceil(Seq/chunk_size)
//   lane A/D: Batch*H*(P/warp)*ceil(Seq/chunk_size)*warp each.
// B/C, dt and final A/D partial workspaces use the same layouts as the
// reference deterministic launcher.
bool launch_mamba2_faithful_backward_deterministic_chunked(
    const float *gy, const float *x, const float *dt, const float *A,
    const float *decay_terms, const float *B_in, const float *C_in,
    const float *D, const float *state_history, float *gX,
    float *gDt, float *gA, float *gB, float *gC, float *gD,
    float *partial_B, float *partial_C, float *partial_dt,
    float *partial_A, float *partial_D, float *chunk_carry,
    float *chunk_scale, float *chunk_lane_A, float *chunk_lane_D,
    int Batch, int Seq, int H, int P, int N, int G,
    int runtime_warp_size, bool state_major_history, int history_chunk_size = 0);
// Diagnostic A/B controls. Both optimizations default to enabled; setting the
// corresponding environment variable to 0 restores the scalar-atomic kernels
// used by the preserved pre-optimization HIP build.
bool faithful_warp_aggregation_enabled();
bool faithful_reduced_conv_enabled();
bool faithful_k4_conv_enabled();
// Geometry-only dispatch: one block per (batch, head), one thread per channel
// for P<=256. NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY=1 restores the prior linear
// 256-thread grid without changing layouts or arithmetic. In deterministic
// backward, the default schedules one native wave per block so P=64 exposes
// two independent blocks on wave32 hardware while preserving each wave's
// arithmetic and fixed partial slot. Set
// NSOS_MAMBA_DETERMINISTIC_HEAD_WAVE_GEOMETRY=0 to restore one block per head.
bool faithful_head_channel_geometry_enabled(int channels_per_head);
bool faithful_deterministic_head_wave_geometry_enabled(
    int channels_per_head, int runtime_warp_size);
// The head-wave kernel keeps each lane's N-state reverse-scan carry in a
// disjoint shared-memory slice. This removes compiler-generated global scratch
// without changing arithmetic. It remains an opt-in A/B path because gfx1102
// profiling showed a throughput regression; set
// NSOS_MAMBA_DETERMINISTIC_SHARED_CARRY=1 to exercise it.
bool faithful_deterministic_shared_carry_enabled();
// Production default. The five stable terms are computed once per row/head
// and retained from forward through backward. Set
// NSOS_MAMBA_PRECOMPUTE_DECAY=0 to restore per-channel evaluation.
bool faithful_precomputed_decay_enabled();
// Exact-order deterministic backward split: the recurrence runs independently
// per state, then a second kernel folds states in the original ascending order.
// Retained as an opt-in A/B path after gfx1102 showed a large memory-traffic
// regression; NSOS_MAMBA_STATE_PARALLEL_BACKWARD=1 enables it.
bool faithful_state_parallel_backward_enabled();
// Production bounded-memory temporal parallelism. It is used only when the
// sequence spans more than one tile; NSOS_MAMBA_CHUNKED_BACKWARD=0 restores
// the fused reverse reference kernel.
// NSOS_MAMBA_BACKWARD_CHUNK_SIZE selects a power-of-two temporal tile in
// [8,256]; the measured gfx1102 production default is 128.
bool faithful_chunked_backward_enabled();
int faithful_backward_chunk_size();
// Production chunked-backward LDS layout. [state,lane] maps adjacent wave
// lanes to adjacent banks without changing ownership or arithmetic order.
// NSOS_MAMBA_CHUNK_LDS_STATE_MAJOR=0 restores [lane,state].
bool faithful_chunk_lds_state_major_enabled();
// Production training-only GPU history layout. State-major makes adjacent
// wave lanes contiguous for each n. NSOS_MAMBA_STATE_MAJOR_HISTORY=0 restores
// the row-channel-state reference layout.
bool faithful_state_major_history_enabled();

// Incremental faithful path split in two kernels so all x/B/C convolution
// channels are materialized before the SSD consumes shared B/C values.
void launch_mamba2_faithful_conv_step(
    const float *xv, const float *Bv, const float *Cv,
    const float *conv_weight, const float *conv_bias, float *ring,
    float *xBC, int inner, int group_state, int K, int rows = 1);
void launch_mamba2_faithful_step(
    const float *xBC, const float *dt, const float *A, const float *D,
    float *state, float *y, int H, int P, int N, int G, int rows = 1);
void launch_mamba_gated_rmsnorm(const float* y, const float* z,
    const float* weight, float* output, int rows, int dim, float epsilon);
#endif

}  // namespace cuda
}  // namespace nsos

#endif  // MAMBA_KERNELS_CUH
