# Neris memory P0 handoff, complete candidate v1

Owned area: `research/mamba3h/memory` only. Read-only baseline remains
`4db1250e7435318e1c2522b27649a5d5433f2f48` (HEAD verified after execution).
No native source, OXN, shared build, GPU, production default, commit or push changed.
This is an **isolated addressable-memory component**, not integrated Mamba3H.
Root2 owns all integration and native-backbone evidence.

## Import and interface

```python
from research.mamba3h.memory import CausalSlotMemory, Control
memory = CausalSlotMemory(dim=8, capacity=4, top_k=1)
state = memory.initial_state(batch=8)
# All tensors CPU. bool[B] gates; int64[B] entity and revoke_entity.
control = Control(write, read, entity, revoke_entity)
y, next_state, stats = memory.step(x, state, control)  # x/y [B,D]
```

The package import is lazy: the caller sets OMP/MKL/OPENBLAS/NUMEXPR <=2 before
importing Torch; our executable runners set all four to 2 and Torch intra/inter-op
threads to 2. Model float dtype must match x and state. `.double()` works.
`forward(x,state,control=...)` delegates to step. State is explicit, batched,
resettable, and has no hidden cross-split cache. `SlotState.detach()` makes a cloned
truncated-BPTT boundary; Root2 chooses when to detach.

Optional Control fields: `retain: bool[B,C]` (false removes and clears that slot),
`address: int64[B,k]` (-1 pads unused positions). Default entity=-1 is anonymous
global retrieval; a nonnegative entity restricts retrieval to that entity.
`revoke_entity=-1` means no revocation. Control rejects dictionaries, wrong
dtypes/shapes and unrecognized content/answer/logit fields. Targets and benchmark
solver objects never enter the model interface. Visible token values belong in
x, never in Control. Caller-provided state and oracle metadata are trusted prefix
objects: no component can prove that an external caller did not fabricate them.

Processing order is **project Q/K/V from x -> invalidate revoke/overwrite/retain
-> read previous valid slots -> commit current projected K/V**. Current writes
are invisible to same-step reads. Overwrite and revoke clear keys, values, IDs
and timestamps before reading, so same-step stale evidence is unavailable.
On WRITE/SET/OP, pass entity/write_id as `entity` and write=true; QUERY uses
entity and read=true; REVOKE passes revoke_id. NOOP has both gates false.
OP writes the supplied model state: this component does not implement an INST
group solver or update entity values using an answer oracle.

## Routing and comparator

`OracleRoutes` has only write/read/retain/address fields. `route_oracle` returns
new Control metadata without access to x, slots, parameters or readout:

* W permits write override only.
* WR permits write, read and retention overrides.
* WRA additionally permits discrete existing slot indices.
* store_all forces write at every visible token, while retaining fixed capacity.

The extra retention permission under WR is explicit; Root2 can use only the
write/read subset for its contract. Oracles never inject Q/K logits, attention
weights, contents or output values. WRA uses the same model-derived selected Q/K
dot products and softmax as ordinary attention. Invalid, duplicate, inactive,
future and cross-entity addresses are rejected. No READ means no candidate
construction, similarity, sorting/top-k, softmax or value gather.

Default selection is exact descending QK score with stable lower-slot-index ties.
FIFO eviction uses oldest timestamp with lower-index ties. Capacity/top_k are
strict positive integer budgets; top_k cannot exceed capacity. Revocation and
overwrite protect identity semantics, not perfect recall beyond capacity.

`SparseRetrievalAttention` uses the identical Q/K/V/output and sparse top-k
attention path as a bounded prefix K/V cache. The MA convention is anonymous
entity=-1 plus store_all routing; its chronological token eviction differs from
semantic WRITE-only storage. It is a control policy, not a second recurrence or
a native Mamba implementation. Controls exposing entity identity are reported
as routing metadata and must be matched across arms if causal isolation is used.

## Differentiation, budgets and counters

Autograd reaches x, written prior x and Q/K/V/output parameters when selection
contains multiple slots. Top-k and gate decisions are discrete with no STE.
**k=1 softmax is constant: Q/K receive no gradient through hard selection.**
Entity-isolated one-slot reads have the same limitation. Thresholded learned
gates exist but receive no gradient from explicit routing and are counted as
inactive parameters in receipts. P0 does not claim learned routing feasibility.

Stats report actual per-step read requests, executed read branches, QK dot
evaluations/madds, exact top-k calls/candidates, address candidates, gathered
values/attention madds, writes, evictions, overwrite/revoke/retention removals,
Q/K/V projection madds, output madds and retrieval candidate-slot checks.
`state.counters[B,n]` accumulates these independently per sample. These count
implemented primitives, not complete machine instruction counts/FLOPs: sorting
comparison counts, validation scans, allocation/copy work and learned-gate ops
are not included. Read=false has zero actual search counters even on full cache.

`parameter_bytes` counts all model parameters. `state_bytes`/`tensor_bytes()`
count reserved K/V, occupancy, entity, timestamp, clock and counter tensors;
fixed capacity means fixed state bytes independent of occupancy. This is logical
persistent tensor size, excluding Python object overhead and autograd history.
`cache_bytes_estimate` is an explicit projection plus row search tensor workspace
estimate, **not allocator peak memory**: functional state clones, autograd saved
tensors and optimizer memory are not comprehensively measured. Training graph
retention can exceed logical fixed slot storage until a detach/sequence boundary.
No complete compute/cache/parameter matching is claimed.

## Executed correctness

Command from project root:

```powershell
python research/mamba3h/memory/verify.py test
```

**20/20 CPU tests passed**, 0.157 s unittest body, captured subprocess wall time
in `test_receipt.json`; exact log in `test_results.log`. Cases cover preread
projections/current-write invisibility; deterministic and randomized exact top-k;
READ-gated absence of sort/softmax/search; overwrite/revoke clearing; batch/entity
isolation; eviction/store-all fixed budgets; retain clearing; routing-only
content/parameter invariance; selected model score/softmax reference; malicious
future/inactive/duplicate/cross-entity addresses; suffix independence; reset and
caller-state immutability; autograd to earlier writes/current queries/parameters;
matched sparse attention; malformed controls/nonfinite input; mixed-batch gating
counters/gradient suppression; evicted-entity exclusion; and content perturbation
that cannot change Q/K addressing. Rejected inputs are expected negative tests,
not failed jobs. No correctness failure was discarded.

## Executed bounded learned feasibility

Protocol was written before training: `trial_protocol.json` (one fixed Adam
setting, lr .03; CPU 2 threads, per job 25 s, aggregate 180 s; no tuning/selection).
Command: `python -m research.mamba3h.memory.learned_trial`.
Five paired seeds 11/23/37/53/71, D8, C4, k2, length6, train32/validation16/test32,
30 full-batch updates per arm. Explicit operation gates; anonymous retrieval
must learn query-key addressing. Targets remain separate from the model.
The input task has three distinct visible entity/value writes, two id-only NOOP
distractors, then an id-only query. This independent small smoke is not the Orin
MQAR/INST benchmark, and its numeric encoding is not a Mamba-3 backbone.

All **10/10 jobs complete**, 30 updates each; no failed, censored or unattempted
cells within these two arms. Final receipt `learned_results.json` records source
hashes, protocol SHA, split generator seeds/input/target fingerprints, all
accuracy/loss results, gradients, state norms, byte/operation counts and latency.
Final wall time **17.056 s**, plus preserved preliminary run **14.359 s**; total
learned-run wall time **31.415 s**, below 180 s. Each final arm has 319 parameters
(1276 bytes), including 27 inactive gate parameters; 14976 persistent state bytes
per batch32 (468 per sample). Actual test searches are 96 QK evaluations/96 writes
for memory_WR versus 128 QK evaluations/192 writes/64 evictions for sparse_store_all;
both gather 64 values. This residual budget/routing difference is material.

| Seed | memory_WR test accuracy | sparse_store_all test accuracy | Paired delta |
|---|---:|---:|---:|
| 11 | .71875 | .53125 | .18750 |
| 23 | .78125 | .43750 | .34375 |
| 37 | .96875 | .25000 | .71875 |
| 53 | .90625 | .46875 | .43750 |
| 71 | .78125 | .68750 | .09375 |

Means .83125 and .47500; paired mean .35625, t95 interval df4
[.05494054,.65755946]. The comparator can evict an early write because it also
stores distractors; these results establish bounded projection/readout learning
on this smoke, not an advantage over a fully matched attention baseline. Max
training gradient norm .559832 and max slot norm 3.393824 across all jobs; these
small empirical maxima do not certify switching or long-context stability.
Gap recovery is undefined: no native baseline/ceiling was measured here.

The preliminary receipt is retained as `learned_results_initial.json` with
`initial_run_note.json`: an early eager package import could precede the runner's
thread environment settings. Lazy import corrects that ordering, and final runs
record preimport caps explicitly. The rerun also includes expanded operation
counters. Outcomes are identical; the original is not substituted or hidden.

## Prepared/unattempted and freeze

Root2 native-backbone integration, full M0/M1/M2/M3/MA/MC factorial, native VJP
parity, learned discrete routing, Orin task grids, longer contexts, GPU, language,
multimodal and production enablement are **not executed by Neris**. The common
Torch interface is ready for Root2's adapter and independent review.

`manifest.json` pins every owned top-level source/protocol/receipt/log/handoff
file with SHA256 (excluding itself and generated __pycache__). Freeze command:
`python research/mamba3h/memory/verify.py freeze`. Root2 should snapshot/import
these hashes; no peer files are edited. Further changes need a new handoff/hash.
