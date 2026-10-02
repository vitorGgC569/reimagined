# Mamba-3H P0: optional, falsifiable experiments

User authorized 2026-10-01: keep Vela/Atria/Silex on GPU completion; add Lyra,
Neris and Orin for Mamba-3H experiments, with Root2 coordinating and independently
reviewing them. Codex/root integrates production GPU changes and controls the
single RX7600 execution lane. No automatic push is authorized for new work.

## Frozen baseline and boundaries

Baseline commit: `4db1250e7435318e1c2522b27649a5d5433f2f48` on main.
CPU72/72 and HIP162/162 passed; corrected Adam/fused GPU replay passed bitwise
for BF16/FP16/FP32 across 24 commits each. Current HIP extension SHA256:
`8212a02b9e1486e84352aaeba2c42b02072d3b775d2f4cc2b6b98f5dbb8f5060`.
Existing receipts live in artifacts/maestri_integral_20261001_gpuopt.

Mamba-3H is experimental and disabled by default. Its modules, tests and runners
belong under research/mamba3h. No edits to OXN/nsos, shared build directories,
baseline sidecar/checkpoint identity, or GPU runs without a scheduled root lease.
Private CPU builds are allowed. Use CPU PyTorch/NumPy with at most two threads
per worker; set process-local thread flags before imports. No background GPU
use, large unattended sweeps, or external communication beyond the named team.

## Hypothesis and arms

Transition expressivity is not equivalent to addressable retrieval capacity.
M0 = frozen Mamba-3; M1 = M0 + structured noncommutative dynamics;
M2 = M0 + controlled explicit memory; M3 = M0 + both;
MA = M0 + sparse retrieval attention. Compare matched compute/parameters/state
bytes where possible and report all residual budget differences. Include a
matched commuting/extra-capacity control. Microtests of isolated components are
not evidence that the integrated Mamba-3H beats M0.

Root2 freezes a common interface and bounded experiment contract before training.
Prefer an adapter to the actual frozen native Mamba3 layer; any PyTorch recurrence
port must first pass forward/state/VJP differential tests against it. A simplified
toy recurrence must be labeled as such, never as an official Mamba-3 baseline.
No pretrained checkpoint, corpus-quality or multimodal result is implied by P0.

## Ownership and first deliverables

- Lyra owns algebra/: exact S5 Householder/permutation controls, learned rank
  1/2/4 transitions, commuting controls, closure and stability. Start with
  D=I, alpha=2, u=(ei-ej)/sqrt(2); distinguish representability, learning and
  runtime. Compare sequential/chunk/tree composition; fit/evaluation probes
  must be independent. Report relative norm and squared error separately.
- Neris owns memory/: bounded causal slots, exact top-k, WRITE/RETAIN/READ,
  oracle W/WR/WRA and store-all controls, sparse-attention baseline. Keys/values
  come from model states before retrieval; oracles cannot change values,
  logits/readout or expose future contents/answers. Test overwrite/revocation,
  entity isolation and budgets. Read gating must actually avoid retrieval work.
- Orin owns benchmarks/: MQAR, group tracking and INST generators/oracles,
  independent solver, anti-leakage and matched seeded splits. Vary length,
  entities, operations, distractors, group and overwrite independently. Start
  within-group generalization; unseen ordered pairs then cross-group separately.
- Root2 owns integration/, review/ and experiment manifests: interfaces,
  integration of handed-off candidates, independent negative tests, paired
  seeded experiments and conclusions. Communicate with Codex by Maestri and
  file handoffs. Do not block component workers on synchronous replies.

## Required scientific controls

At least five paired seeds for claims in small-model trials; predeclare actual
CPU/time budgets, hyperparameter budgets and stopping rules. Use cheap correctness
smokes before learned trials. Preserve failures, censored runs and unattempted
jobs; never replace them by a successful subset or silently relax tolerances.
Report absolute accuracy, paired uncertainty, parameter/state/cache bytes,
actual retrieval operations, latency and training steps. Gap recovery is
undefined when its ceiling-minus-baseline denominator is too small; no arbitrary
epsilon division. Raw-accuracy superadditivity is optional evidence, not a
necessary condition for useful complementarity or a proof of mechanism.

For closure: small exact FP64 matrices, held-out random operator probes at larger
size, repeated recompression and tree-association error. For stability: state
norms/tails, gradient spikes, perturbation gain and product growth. Local operator
norm or eigenvalues alone do not certify switching stability; unit Householder
and bounded diagonal factors can provide a common homogeneous contraction bound.

Write runnable code plus CPU tests, SHA-pinned manifests and HANDOFF.md with
executed versus prepared results explicit. Conclusions must distinguish
component correctness, representational feasibility, learned feasibility,
integrated factorial evidence and SOTA/real-language claims. Multimodal,
quantization and large-scale pretraining are later stages.

## Primary related work

- Mamba-3: https://arxiv.org/abs/2603.15569
- Retrieval-aware distillation: https://proceedings.mlr.press/v306/bick26a.html
- Diagonal limits: https://arxiv.org/abs/2603.01959
- HDLA: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8bb3540613dafc26d08d7fec325cf458-Abstract-Conference.html
- DeltaProduct: https://github.com/automl/DeltaProduct
- Titans: https://research.google/pubs/titans-learning-to-memorize-at-test-time/
- MIRAS: https://arxiv.org/abs/2504.13173
- Engram: https://arxiv.org/abs/2601.07372
- Prefix-scannable models: https://arxiv.org/abs/2506.10918
- Stability: https://proceedings.mlr.press/v306/cao26l.html
