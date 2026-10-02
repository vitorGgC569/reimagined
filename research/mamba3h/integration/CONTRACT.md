# P0 common contract v1 — frozen before learned trials

Baseline: commit `4db1250e7435318e1c2522b27649a5d5433f2f48`. Native CPU
provider: `OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd` (SHA in freeze.json).
No production edits, shared builds, GPU, commit or push. Each worker sets
OMP/MKL/OPENBLAS/NUMEXPR threads <=2 before imports and torch threads <=2.

## Small differentiable interface

Use PyTorch CPU tensors, batch first. `step(x, state, *, control=None)` returns
`(y, next_state, stats)`; x/y `[B,D]`, state is explicit and independently reset
per sample. No hidden cross-split caches. Autograd must reach x and learned
parameters. `control` contains current/past operation metadata only, never
targets, logits, future token contents or answer values. Modules may expose
their existing API; Root2 writes adapters in integration/ after handoff.

Lyra: minimum adapter hook `transition(h, u, alpha, diagonal=None) -> h_next`,
with h `[B,D]`, u `[B,rank,D]`; order is chronological. Exact Householder uses
normalized u and alpha=2, diagonal=1. Learned rank 1/2/4 and commuting control
must identify composition/closure approximations and count every parameter.
Provide a learnable nn.Module with `step` if feasible; no forced peer rewrites.

Neris: minimum `step(x, state, control)` with fixed causal capacity and explicit
write/read gates. keys/values are projections of x BEFORE retrieval. top-k is
exact; no-read skips similarity work. Expose slot occupancy, similarity/read/
write counts, parameter bytes, persistent state bytes and cache bytes. Oracle
W/WR/WRA may change routing only, never stored values or readout. Revoke/overwrite
must remove stale evidence and isolate entities.

Orin: `generate(task, seed, split, count, **difficulty)` returns examples with
tokens/events, targets, query_mask and separate causal control metadata. Target
and solver/oracle objects must not enter the model interface. Supply a vocabulary
or numeric encoding, an independent solver and reproducible split fingerprints.
Train/validation/test seeds and ordered-pair partitions must be disjoint.

## Integration scope and predeclared bounded trials

Root2 may train frozen actual-native Mamba3 features plus differentiable optional
adapters/readout. This is `actual_native_frozen_backbone_adapter_P0`, not native
end-to-end Mamba3H. A port/toy must carry its own label and cannot replace M0.
Native bridge parity includes forward, all boundary states, input/parameter/
initial-state VJP, empty/masked/chunked calls. A port is barred from architecture
claims until differential tests pass. A disabled adapter must reproduce M0.

Paired seeds: [11,23,37,53,71]. Arms M0/M1/M2/M3/MA plus MC commuting/capacity.
Initial pilot: D=8, length<=24, 64 train/32 validation/64 test examples, <=60
optimizer updates per arm/seed, one fixed hyperparameter setting, <=900 seconds
aggregate Root2 learned CPU wall time; per-job timeout 90 seconds. No selection
using test accuracy. Stop nonfinite immediately, preserve failed/censored/
unattempted cells. If generator/module incompatibility requires a protocol
revision, record it BEFORE training. Additional task grids remain unattempted.

Report accuracy and five paired deltas with t interval (df=4), all seed outcomes,
parameters/state/cache bytes, retrieval operations, latency, updates, gradient
and state norm tails. State/count matching uses equal reserved workspace where
possible; residual active-parameter/FLOP differences must be explicit. No claim
of complete compute matching without measurements. Gap recovery undefined if
ceiling-baseline <=0.01. No synthetic SOTA, language or multimodal claims.

Ownership: Root2 integration/, review/, manifests/ only. Completed peer handoffs
are read/imported, never edited. Write HANDOFF.md with executed/prepared status.
