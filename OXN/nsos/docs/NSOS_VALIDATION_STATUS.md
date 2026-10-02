# NSOS Validation Status

## Current Position

### Mamba-3 integrated parallel/Flash backward checkpoint — 2026-10-02

The newer integrated block supports SISO/MIMO, tile-parallel forward/backward,
Flash LDS replay v2, BF16/FP16 projections with FP32 controltail, native fused
Adam and Muon paths. This supersedes the standalone-only status of the earlier
September 30 entries below. It does not promote GPU to a supported product.

Rank-sized Flash backward LDS capacities1/2/4/8 preserve arithmetic order and
shrink SISO shared scratch from45608 to16936 bytes. Three private paired fresh
processes on RX7600 measured1.89467x whole-step speedup [1.84969,1.94074] at
D768/L16/B1/S512/N128/P64/R1, BF16 projections, fused AdamLR.002, synthetic
vocab257 data. Shared no-override execution measured1.456938steps/s745.952tok/s;
separate timing1 diagnostic backward488.033ms. All524296912 authoritative
state snapshot bytes match the private baseline after13 commits.

Latest shared HIP162/162, CPU72/72 and strengthened active/empty odd-rank GPU
regression pass. These are correctness and scoped performance results, not
general language quality, cross-hardware certification or SOTA evidence.
Four original lowp-vs-FP32 GOLD comparisons still fail; GOLD576 remains blocked.
Hierarchical interchunk transport remains experimental opt-in because measured
target probes regressed. The archived ABI64 profiler cannot link to this ABI80
build; its rejected negative HIP event is not an accepted timing trace.

See [rank-LDS implementation and evidence](GPU_MAMBA3_BACKWARD_RANK_LDS_2026-10-02.md),
[portable acceptance summary](GPU_MAMBA3_BACKWARD_ACCEPTANCE_2026-10-02.json), and
[product closure assessment](PRODUCT_ASSESSMENT_2026-10-02.md).

### Mamba-3 SISO RDNA boundary replay and owning tape — 2026-09-30

Post-BCNorm SISO FP32 forward/backward now runs on gfx1102 with explicit
phase/SSM/previous-K/previous-V states and complete state adjoints across
chunks/calls. Private tape storage and lane ownership are checked. No hidden
CPU numerical fallback. This is NOT the complete Mamba-3 block or MIMO, and
it is not registered as an active Jamba/Trainer path.

Final CPU61/61 (17.41s), HIP127/127 (139.14s), directed SISO1/1 (0.77s) and
the same executable with NSOS_CUDA_SYNC=0 passed on RX7600/gfx1102. Final
review corrected a potential nonuniform barrier-loop bound on shared failure
status; HIP was rebuilt and fully rerun. Tests include FP64/finite differences,
streaming VJP, masks/NaN padding, modulo phase crossings, explicit stream,
ownership/cancel guards, invalid metadata and canaries. A general regression
does not certify those providers' future model integration or language quality.

Latest module SHA: HIP d6602a6b..., CPU c1eb3904...; full hashes, commands,
earlier superseded runs and logs in
[Mamba-3 SISO evidence](GPU_MAMBA3_SISO_2026-09-30.md).
KAN stays held and defaults unchanged. Complete block/MIMO, integrated
Attention/MoE/Adam/CCE, actual speed/quality and release gates remain open.

### Parallel foundations — 2026-09-30

Two native Codex subagents inherited the same model/effort: Attention RDNA and
device sparse optimizer. Lead implemented Mamba3 SISO FP64 recurrence,
preprocessing, complete VJP/output+state seeds and cross-chunk gradient tests.
Standalone prerequisites are now compiled and validated, NOT integrated into
the model/Trainer. No production/default/SOTA/throughput promotion.

Final CPU61/61 (16.55s), HIP126/126 (146.94s), new directed tests3/3 (0.98s),
RX7600/gfx1102. Initial CPU60/61 failed the stale expected kernel manifest;
fixed and fully rerun. That lot's HIP module SHA2bf2aedd...; CPU0b021097...;
full hashes, limits and logs in
[Parallel foundations](GPU_PARALLEL_FOUNDATIONS_2026-09-30.md).
KAN remains held; preserved numerical/checkpoint tests passed in the general
regression, without any new KAN performance/quality campaign.

### User-directed KAN hold and architecture priority — 2026-09-30

Further KAN development is suspended. Keep candidates opt-in and preserve
tests/negative D768 results. Resume only after a quality/cost ablation beats
SwiGLU/MLP. Active order: Mamba-3 -> FA4-inspired RDNA attention -> device-side
MoE/Adam -> head/loss/CCE. No GPU/TTT/product/default promotion.

The scalar KAN candidate passed CPU60/60, HIP122/122 and Python17/17 on
HIP SHA961f8e3b...; its paired D768 probe was26–28% slower despite D2H8->4.
A later WMMA increment was written before the hold; its already-running builds
completed and Python policy tests19/19 passed, but its numerical/checkpoint/
performance campaign did NOT run at that point. Those earlier suites do not
certify later binaries; the subsequent general regression is recorded above.
Original SHAs/logs/limitations:
[KAN evidence and hold](GPU_KAN_RECOMPUTE_2026-09-30.md).
[Active roadmap](GPU_ARCHITECTURE_ROADMAP_2026-09-30.md).

### Grouped device gradient accumulation — 2026-09-30

Grouped MoE now commits weight/bias/magnitude gradients with two bank-wide
device launches instead of per-parameter slices and copies. First/add policy
is independent per destination; inactive experts, accumulation union, stable
buffers and lazy Adam state remain preserved. Destination metadata is cached;
invalid shape/device and exact-address alias are refused before writes.
Final suites CPU60/60 (18.08 s), HIP121/121 (148.52 s), Python15/15 passed.
The late offsets D2H/host activity boundary is NOT removed. GPU/TTT product
promotion, Mamba-3 and language-quality/release gates remain open.
Details: [gradient commit evidence](GPU_MOE_GRADIENT_COMMIT_2026-09-30.md).

### Explicit sparse-gradient contribution correction — 2026-09-30

Latest correction separates retained expert gradient storage from contribution
to the current accumulation group across finite checks, clipping, all Trainer
Adam paths, criticality selection and parameter audit. Inactive-after-active
experts preserve weights/moments/versions; contributed zero gradients remain
active. CPU60/60, HIP121/121 and 15 Python policy tests passed after implementation.
Real preserved old HIP binary running CPU reproduced unwanted updates with zero
gradients; the new binary preserves those inactive parameters exactly. The Python
guard rejects that old binary before samples, even for legacy MoE compute.
Policy `optimizer.sparse_gradient_policy=explicit_group_contribution_host_v1`
prevents silent checkpoint trajectory substitution. A current five-step RX7600
WMMA BF16 probe executed finite. This is still HOST activity: device-only
activity/Adam, late-offset removal, language-quality and release gates are open.
Details: [sparse activity correction](GPU_SPARSE_GRADIENT_ACTIVITY_2026-09-30.md).

### RDNA3 WMMA and subsequent device auxiliary/QAT evidence — 2026-09-30

Latest lot: opt-in grouped MoE BF16/FP16 WMMA forward/dX/dW with FP32
accumulation, geometry selection, active-only QAT/STE, immutable tape and
versioned checkpoint policy. Native capability failures do not silently use
a different provider; stale binary is rejected before training samples.
Final CPU60/60 and HIP121/121 include independent double matrix oracle,
per-expert parity and deterministic checkpoint continuation/policy rejection.
Paired reversed-order wide probe: WMMA36.64/37.78 ms, grouped scalar41.92/43.84,
ordered37.74/40.06; BF16 training trajectories differ between providers, so
language quality/convergence is not certified. WMMA remains default OFF.
Details: [WMMA evidence](GPU_MOE_WMMA_2026-09-30.md).

The prior auxiliary/QAT lot moved Switch loss and load telemetry device-side,
prepares QAT only for active experts and caches descriptor uploads. One total
auxiliary scalar still crosses the logging boundary; grouped late counts read
and CPU active-gradient registration remain. Details:
[auxiliary/QAT evidence](GPU_MOE_AUX_QAT_2026-09-30.md).
GPU/TTT product status and the integral Mamba-3/layer/release/quality scope
remain unchanged. Sections below describe historical campaigns, not the latest
performance baseline.

### Device-segmented MoE training — 2026-09-30

Opt-in grouped sparse MoE now schedules forward/backward expert GEMMs from
device offsets, preserving QAT/STE, RMSNorm/magnitude/bias, squared-ReLU,
padding, router task gradients and absent-gradient semantics for empty experts.
A late counts/status read still registers active parameter gradients; auxiliary
balance was CPU-dependent in this initial lot (subsequent device implementation
is documented above). Final suites: HIP 119/119, CPU 60/60. Independent
per-expert numerical parity and deterministic checkpoint continuation/policy
rejection passed. A real stale native binary was refused before its first step.

The paired repeat reduced D2D calls 50→18 per B1 step, but did not eliminate
D2H or stream fences. D128/S256/B2 BF16 median latency was 14.475→13.540 ms;
D768/S512/B1 regressed 39.475→43.100 ms. Default remains OFF, with the old
ordered/hipBLAS path preserved. No quality, SOTA or release promotion is claimed.
Exact evidence and remaining engineering gaps: [grouped MoE report](GPU_MOE_GROUPED_TRAINING_2026-09-30.md).

### Full-sequence TTT evidence — 2026-09-29, subsequent implementation

Opt-in full TTT sequence BPTT now differentiates adaptation, clipping and
momentum, isolates rank-3 training samples and respects valid-prefix padding.
It uses boundary history/recomputation and distributed-column q reduction for
large states. Independent double-primal finite differences cover dInput and
all trainable parameters. Final native suites: HIP 117/117, CPU 60/60; the
subsequent Python policy checks passed 3/3 on each build. Checkpoint continuation
and incompatible-policy rejection passed. The Python training entry point
also refuses a stale native binary before the first step.

The large synthetic D768/S512 probe measured 2562.01→1672.53/1742.43 ms per
step for the same full derivative, with identical final-weight hashes in
those bounded runs. Compact-history peak was 1064.25 MiB, compared to 5156.5
MiB for the legacy dense/truncated contract. These are not quality or official
throughput certifications; small full-BPTT cases cost more than truncation.
GPU/TTT product status is unchanged. Full-project implementation/release and
held-out language-quality gaps remain open. Exact SHAs, contracts, logs and
limits: [full TTT report](GPU_TTT_FULL_SEQUENCE_2026-09-29.md).

### Training-engineering evidence — 2026-09-29

The opt-in `redesign-v1` GPU training profile now has native boundary-history
Mamba recomputation, device clipping/TTT recurrence, ordered MoE and exact
tiled training attention. Final RX 7600 CTest: 114/114; CPU: 59/59. This does
not promote GPU from Experimental or TTT from Research in `PRODUCT.md`.

The paired Mamba-only BF16 probe measured 789.78→1732.45 median tok/s and
5374.375→2398.375 MiB runtime peak live allocation under the measured Windows
conditions. Short-context integration cases can be slower; opt-in defaults
remain. No new held-out intelligence result or production training run is
claimed. Full TTT sequence derivatives and training batch isolation were
subsequently closed opt-in above; GPU memory checker and release/quality gaps
remain open. Commands, binary and
weight hashes, results and limits: [training redesign report](GPU_TRAINING_REDESIGN_2026-09-29.md).

### Product/profile position

NSOS is already validated as:

- a buildable neural runtime
- a trainable small-model stack
- a pack/load product surface
- an HTTP/CLI/Python serving surface
- a CPU-authoritative packed ternary inference path
- a memory-aware system with native causal store plus OxtaMem FFI
- an AMD HIP/ROCm GPU training runtime validated on RX 7600 (`gfx1102`)

Official profile truth:

- `mamba_small` is the production small-profile baseline
- `small` is a legacy alias for `mamba_small`
- `hybrid_pilot` validates attention + Mamba without TTT
- `hybrid_small` validates attention + sparse MoE + Mamba
- `TTT` is research-only until snapshot/replay and serving parity are formally closed

NSOS is not yet fully validated as:

- a small LLM with strong held-out generalization
- a GPU runtime across every supported AMD/NVIDIA architecture
- a benchmarked model family with stable external scorecards

## What Is Solid

- `InferenceEngine`, model packs, tokenizer packs, CLI, HTTP API
- packed linear export and load
- streaming decode for the Mamba-only / no-attention profiles
- curriculum build + training + per-run artifacts
- fast supervised probe loop for quick iteration
- external benchmark harness with held-out curriculum + Wikitext-2 + fixed NSOS eval suite

## Main Open Gaps

### 1. Real Generalization

Blocked by:

- low held-out exact match on `pilot`
- weak teacher-token accuracy on external held-out evaluation
- high perplexity on Wikitext-2

What needs to happen:

- replay-aware curriculum must materially improve held-out answer metrics
- `pilot` needs to show sustained growth in masked answer accuracy before `small`
- `small` must be run only after fast probes show clear upward signal

### 2. Packed Runtime Story

Current truth:

- packed ternary runtime is validated on CPU
- AMD HIP float/training runtime passed full parity and end-to-end gates on
  RX 7600
- NVIDIA CUDA keeps the original NVCC, CUDA runtime/cuBLAS and shared-kernel
  build graph under an automated static guard; it still needs a target NVIDIA
  compile/runtime validation before binary promotion
- packed/GPU lifecycle and DP4A parity are covered, but edge-product promotion
  still requires its external throughput/quality gate

Official support matrix:

- CPU float: validated
- CPU packed: validated
- AMD HIP float/training: validated on RX 7600
- NVIDIA CUDA float/training: pending target-hardware rerun
- GPU packed: parity/lifecycle validated; external product gate pending
- TTT: research-only

What needs to happen:

- repeat the complete GPU gate on each promoted AMD/NVIDIA architecture
- close the packed edge throughput and quality thresholds

### 3. External Benchmarks

Current truth:

- we have a first honest external harness
- we do not yet have a strong report card

What needs to happen:

- repeated runs with the same suite
- `pilot` and `small` reports stored side-by-side
- fixed acceptance thresholds for exact match, teacher-token accuracy, perplexity, and edge throughput

## Validation Gates

NSOS should be considered validated when all of these are true:

1. `pilot` shows clear held-out improvement on the fixed suite and curriculum-held-out probes
2. `small` finishes training and beats `pilot` on the same scorecard
3. packed edge decode remains materially faster than baseline decode
4. model-pack and edge-pack load/reload stay deterministic
5. the official support matrix is explicit:
   - CPU float
   - CPU packed
   - GPU float
   - GPU packed

Release gates are stricter than research gates:

- research champion can be selected by composite score
- release candidate must pass fixed thresholds for exact accuracy, teacher-token accuracy,
  held-out loss and generation behavior on the global suite

## Operational Rule

Use fast probes to choose direction.

Do not spend long runs unless:

- masked answer loss is going down
- first-token accuracy is moving
- teacher-token accuracy is moving

The fast probe loop is the decision layer.
The long `pilot` / `small` runs are confirmation layers.
