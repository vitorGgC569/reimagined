# Lyra algebra P0 handoff

All files live in `research/mamba3h/algebra` in the absolute repository root.
Baseline HEAD observed: `4db1250e7435318e1c2522b27649a5d5433f2f48`.
Owned source/build changes: algebra only. CPU only, process-local numerical
thread limits 2 before imports, torch threads 2. No shared builds or GPU executed.
Uses the Maestri communication skill at
`C:/Users/vitor/.agents/skills/maestri/SKILL.md` for authorized team messages.

## Import and interface

`operators.py` exports `transition`, `StructuredTransition`, `compose`, `scan`,
`s5_matrices`, `permutation_matrix`, `recompress_identity`, and `S5_PAIRS`.

```python
import torch
from research.mamba3h.algebra.operators import StructuredTransition, transition
model = StructuredTransition(dim=8, rank=2, operations=10,
                             commuting=False, seed=11, dtype=torch.float32)
x = torch.zeros(8, 8)
y, next_state, stats = model.step(x, None,
    control={'op_id': torch.zeros(8, dtype=torch.long)})
```

Common-contract shapes are `x/state/y [B,D]`, `u [B,R,D]`, `alpha [B,R]`.
`transition` applies diagonal D first, then normalized direction factors in
chronological order: `A=H_R ... H_1 D`; exact Householder alpha=2, D=1.
Directions must have positive norm; no silent zero-vector normalization.
An adapter step has `h_next=A(op_id) h+x`, with explicit independently reset state.
`state=None` starts zeros. Disabled `control={'enabled':False}` returns x and
unchanged state, enabling an identity optional branch. Autograd reaches x,
initial state and all active learned parameters. Input and module dtype must
match; default FP64 is for tests, pass FP32 for the Root2 D8 pilot.
Only `op_id` (current/past int64 [B]) and optional boolean `enabled` are accepted;
other metadata must be filtered by Root2's adapter. Targets/future/logits/answers
are rejected. There are no hidden cross-sample or cross-split caches.

S5 vocabulary is the ten lexicographic pairs `(0,1),(0,2),(0,3),(0,4),(1,2),
(1,3),(1,4),(2,3),(2,4),(3,4)`. Identity is tuple(range(5)). For scatter matrix
`P[perm[i],i]=1`, left action `P_new=P_op@P_state` equals
`p_new[i]=op[p[i]]`. Exact controls use `(e_i-e_j)/sqrt(2)` and alpha=2.
Z5^3 benchmark labels/solver remain Orin's independent separate control;
algebra's commuting arm uses a single fixed orthonormal eigenbasis.

`compose(late,early)` implements affine `(A2@A1,A2@b1+b2)`; `scan` consumes
`A[T,B,D,D],b[T,B,D],h0[B,D]`. Sequential/chunk/tree yield all inclusive states.
Dense tree is a correctness reference with repeated prefix composition,
not an efficient parallel scan. `recompress_identity` is explicitly lossy SVD
of the residual from I and is NOT used in exact scan or learned trials.

Rank 1/2/4 in NC means a product of that many rank-one factors, not a claim
that their temporal product stays rank 1/2/4. Learned alpha=2 sigmoid(raw),
diagonal=.98+.02 sigmoid(raw), giving homogeneous contraction. The commuting
arm uses the same raw direction/alpha/diagonal tensors, with softmax direction
weights to produce common-basis eigenvalues. Every scalar parameter is counted.
Both arms have `K*(R*D+R+D)` parameters; at D8 K10: 170/260/440 for ranks 1/2/4.
Both reserve D*D basis buffer elements, unused by NC. Commuting may have full
residual rank and redundant effective parameters: equal parameter counts do
not establish equal effective capacity, FLOPs or runtime. No compute matching
claim is made. Stats report parameter/state/buffer bytes, zero persistent cache,
live matrix bytes and mark total workspace as unmeasured (autograd and temporary
allocation overhead excluded). Batch1 step latency is measured separately.

## Executed results

`python test_algebra.py`: **9/9 PASS**. Exhaustive 120 permutations x 10 actions
with an independent pointwise action solver; involutions and order negatives;
dense differential and gradcheck; seq/chunk/tree VJP against independent reverse
adjoint; ranks 1/2/4 parameter equality, commutators, autograd and parameter
directional finite differences; reset/isolation/disabled and leakage rejection;
chunk boundary negative; low-rank closure negative; switched-shear negative.
`tests.log` is the executed final runner evidence.

`protocol.json` was written before learned trials. One fixed Adam lr=.08,
60 updates/cell, seeds [11,23,37,53,71], D5, ranks [1,2,4], NC/CC. Fit probes:
32/op; independent evaluation probes:64/op; independent sequence evaluation:
64 sequences of length8. Thirty cells executed, no failed/censored/unattempted
learned cells. Final full runner wall time 4.40 seconds, within job15s and
aggregate180s bounds. No test-based tuning/selection occurred. Probe and
sequence fingerprints, every loss/update, parameter budgets, gradient/state
tails, latency, absolute squared and relative norm errors are in `results.json`.

Mean whole-permutation sequence accuracy over five paired seeds:

| Rank | NC | commuting | NC-CC | conventional paired t95 (df4) |
|---|---:|---:|---:|---|
|1|0.89375|0.003125|0.890625|[0.59768,1.18357]|
|2|0.909375|0.003125|0.90625|[0.67710,1.13540]|
|4|0.703125|0|0.703125|[0.30152,1.10473]|

Intervals are ordinary small-sample t intervals and can exceed the bounded
accuracy range; they are not clipped. All five deltas, including worse seeds,
are preserved. This supervised isolated operator-fit experiment supplies exact
operation metadata, not retrieval or language supervision. It supports learned
feasibility on these known transpositions, not integrated architecture evidence.

Closure diagnostics: small FP64 S5 affine chunk/tree relative norm errors
2.19e-16/2.22e-16. D64 24-factor products with 32 held-out random probes:
uncompressed dense relative error1.85e-15; repeated rank4 recompression errors
sequential1.24566/chunk1.24473/tree1.23981; tree-vs-sequential association error
0.63024. **Fixed residual rank4 is not closed**, and these approximate products
do not retain the exact scan guarantee. Compression failure is preserved.
Homogeneous exact product perturbation gain .785678 matches the common bound
.99^24. Additive inputs are not bounded by this homogeneous certificate.
Negative shear example: both local spectral radii .9 yet switched 24-step
product norm26891.62. Local spectral radius alone is not a stability proof.

`python native_probe.py`: existing read-only native CPU validation **9/9 PASS**,
no skips, 16 full-layer variants, 252 directional derivatives including input,
every parameter and all initial-state fields; shared-group chunk boundary VJPs,
masked/empty prefix and failure tests. Independent NumPy oracle versus native,
maximum output absolute error4.21315e-6. This reuses the pinned existing test
source rather than claiming Lyra implemented a new Mamba3 port. Hashes of the
test/oracle/provider are recorded, and the provider was unchanged before/after.
Native provider SHA256:
`050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f`.
Root2 owns source-to-binary provenance freeze for its M0. This gate does not test
the algebra adapter attached to native Mamba3 and does not establish end-to-end
trainability of the hybrid.

## Reproduction and handoff freeze

From the algebra directory run:

```text
python test_algebra.py
python run_diagnostics.py
python native_probe.py
python freeze_manifest.py --verify
```

`manifest.json` pins all delivered code, protocol, this handoff and executed
evidence, plus read-only PROGRAM/CONTRACT/native-test/oracle/provider dependencies.
Its own SHA256 is sent to Root2 by Maestri. Verify before import/snapshot.
Re-running diagnostics/probe replaces local evidence, so snapshot the frozen
directory first; do not expect a prior manifest to verify after rerun.

Prepared/unattempted: Root2's actual-native frozen-backbone D8 pilot, disabled
M0 integration parity, factorial M0/M1/M2/M3/MA/MC, cross-group generalization,
memory coupling, GPU, real-language/multimodal/pretraining/SOTA. No peer or
production files were edited. No current component blocker; production
integration and scheduler remain Codex/Root2 responsibilities.
