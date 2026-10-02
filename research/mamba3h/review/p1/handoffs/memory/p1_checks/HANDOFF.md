# Neris P1 checks: C6/top-k2, no learned trials

All new files are under `research/mamba3h/memory/p1_checks/`. Frozen P0 files
were SHA-verified before and after both executions and were not edited.
Frozen parent manifest SHA256:
`376ba6fb332a77235950c9a3ad0fb41daa2f488b60c7360d60e76848417b6149`.
Baseline commit remains a read-only reference, not a baseline run in these checks.
No GPU, OXN, shared source/build, optimizer, checkpoint or native feature run.

Executed from project root:

```powershell
python -B research/mamba3h/memory/p1_checks/run_checks.py
```

`-B` plus sys.dont_write_bytecode prevents bytecode writes outside the new area.
OMP/MKL/OPENBLAS/NUMEXPR are set to 2 before Torch imports; Torch intra/inter-op
threads are 2. Final expanded suite: **12/12 PASS**, 1.002 s test body,
1.286393 s complete checks/receipt body. `attempt-1.*` retains the earlier
10/10-pass suite; `attempt-2.*` and `results.json` contain the final 12/12 suite
and expanded positive witnesses. No failed/censored attempt was discarded.

## Fixture and matched initialization

Capacity C6, exact top-k2, D8, float64, B3, L6. Four possible entities/four value
classes. Visible writes `(entity,value)=(0,2),(1,0),(2,3)`; two id-only NOOPs
with entities 3 and 1; id-only QUERY of each written entity in the three samples.
Three write IDs are distinct; all three queries have valid previous evidence.
Value one-hot channels are zero on both NOOPs and QUERY. Targets [2,0,3] are
separate and enter only the external loss/positive assertion, never model or
Control. `Operation(kind,entity)` accepts no value/answer/target/logit field.

Paired fixed seeds [11,23,37,53,71]. Q/K/V/output, gate and readout initial
weights are copied identically across all three arms per seed; their full hashes
are recorded. One cross-entropy backward at initialization counts gradients;
**zero optimizer updates** and initial/final parameter hashes are identical.
These are correctness/gradient checks, not learned trials or seed accuracy claims.

| Arm | Stored prior to query | QK candidates/query | Values read/query | Writes/sample, L6 | Evictions |
|---|---:|---:|---:|---:|---:|
| C6 anonymous WRITE-only | 3 | 3 | 2 | 3 | 0 |
| C6 symbolic ADDRESS | 3 | 1 | 1 | 3 | 0 |
| Sparse C6 anonymous store_all | 5 | 5 | 2 | 6 | 0 |

Sparse commits its sixth/query token only AFTER retrieval, so it sees five past
candidates and never the current query slot. C6 removes P0's forced-eviction
confound for this L6 fixture. Sparse still stores additional NOOP/query tokens:
actual searches, writes, occupancy and active content differ despite matched
reserved state/parameters; complete FLOP/behavior matching is not claimed.

## Review: rename the existing entity-filter arm

The frozen module's nonnegative `Control.entity` masks eligible slots using stored
entity IDs. With distinct writes and overwrite semantics, that yields the correct
single slot before QK scores are evaluated. **This is symbolic ADDRESS**, even
when `Control.address is None`. Call it `WR + symbolic ADDRESS` or
`symbolic_ADDRESS`, not anonymous WR or learned addressing. No P0 API/source was
renamed or changed: this is a corrected reporting label and a P1 adapter.

`routing.py:causal_control` gives literal W/WR/WRA routing semantics using only
current opcode/ID and prior occupancy/entity/timestamp metadata:

* W controls WRITE only; default READ remains false.
* WR controls WRITE and READ only; no retention override is used in P1.
* WRA adds optional literal existing slot indices, with -1 padding.
* store_all writes each visible token; READ uses the current QUERY opcode.

`symbolic_address=True` is an explicit, independent ADDRESS policy flag.
The symbolic metadata does not alter Q/K projections, contents, attention logits,
softmax or model readout. The adapter derives literal indices from existing slot
IDs, not projected values, answers, targets or future contents. A truly anonymous
cache has no stored entity directory; literal entity addressing there would need
a separately tracked causal directory with additional state/operations. P1 does
not silently provide that extra directory. The REVOKE metadata path is included,
but revocation semantics for a truly anonymous multi-write cache are not exercised
by this distinct-write fixture and would likewise need identity bookkeeping.

Positive WRA tests compare implicit symbolic ADDRESS against literal indices
`[[0,-1],[1,-1],[2,-1]]`. Outputs and K/V contents match **bitwise**. Literal
indices are also sufficient with read entity=-1 when the previously stored IDs
remain available to the causal address adapter. Literal WRA does not add a new
capability beyond the symbolic single-slot policy on this fixture:

| Symbolic query routing, batch3 | QK dot evaluations | top-k calls | literal address candidates | values read |
|---|---:|---:|---:|---:|
| Implicit entity filter | 3 | 3 | 0 | 3 |
| Explicit WRA indices | 3 | 0 | 3 | 3 |

Both paths use the same one-element model-score softmax. Literal metadata lookup
additionally scans C6 prior slot IDs per queried sample in our check adapter;
that adapter work is **not** in frozen module counters. Frozen model
`candidate_slot_checks` is 6/sample for every READ arm, including symbolic/literal,
so a single QK candidate does not imply O(1) total address lookup cost.

## Q/K gradients and budgets

The gradient counts below are exact `torch.count_nonzero` across query/key weights
after one finite float64 backward, no optimizer or threshold-based selection:

| Seed | Anonymous Q+K nonzero elements | Symbolic ADDRESS Q+K | Sparse anonymous Q+K |
|---|---:|---:|---:|
| 11 | 72 | 0 | 72 |
| 23 | 72 | 0 | 64 |
| 37 | 72 | 0 | 80 |
| 53 | 72 | 0 | 72 |
| 71 | 72 | 0 | 72 |

Q alone has 24 nonzero elements in all anonymous/sparse rows; K has 48 for anonymous
and 40/48/56 for sparse. Symbolic Q and K gradients are exactly zero because
softmax over one selected slot is constant even though configured top_k=2.
The three stored IDs remove competition. Top-k2 with at least two selected slots
restores ordinary differentiable QK-score weighting in the anonymous arms; the
discrete membership of top-k itself remains nondifferentiable. This demonstrates
a gradient path at initialization, not convergence or a native Mamba3H mechanism.
Full per-parameter nonzero counts, norms, maxima and finiteness are in results.

Every arm has 319 parameters including readout, **2552 bytes FP64** (283 memory
parameters/2264 bytes plus 36 readout parameters/288 bytes). Gate parameters are
included but receive no gradient under these explicit controls. Reserved logical
state is **1014 bytes/sample, 3042 bytes/batch3** in every arm: 768 K/V, 6 occupancy,
48 entity IDs, 48 timestamps, 8 clock and 136 operation-counter bytes per sample.
For Root2's FP32 use these same shapes imply 630 bytes/sample and 1276 parameter
bytes including readout; that is a dtype calculation, not an executed native trial.

Total L6 operations per batch3 (unchanged across seeds):

| Counter | Anonymous | Symbolic ADDRESS | Sparse store_all |
|---|---:|---:|---:|
| QK evaluations | 9 | 3 | 15 |
| QK multiply-adds | 72 | 24 | 120 |
| top-k candidates | 9 | 3 | 15 |
| top-k calls | 3 | 3 | 3 |
| values read | 6 | 3 | 6 |
| attention multiply-adds | 48 | 24 | 48 |
| writes | 9 | 9 | 18 |
| evictions/overwrites/revocations | 0 | 0 | 0 |
| candidate slot checks | 18 | 18 | 18 |
| Q/K/V projection multiply-adds | 3456 | 3456 | 3456 |
| output multiply-adds | 1152 | 1152 | 1152 |

Query-step tensor workspace estimates are 1184/936/1344 bytes respectively.
These are the frozen module's explicit workspace estimates, not measured allocator
peak; state clones, autograd saved tensors, readout and Python overhead remain
outside that estimate. No optimizer cache exists in these checks. Logical fixed
slot state bytes do not bound untruncated autograd graph storage.

## Positive/causal tests and scope

The final suite checks frozen hashes, fixture validity and common complete weight
initialization; expected candidate/gradient counts; zero eviction and model-derived
K/V; all arm outputs against an independent direct prefix-attention calculation;
literal WRA/implicit symbolic equality; literal addressing without a read entity
filter; W/WR/WRA restrictions and no-answer fields; inactive future-content
perturbations and future suffixes leaving prior reads unchanged; current query
write excluded from sparse retrieval; and fabricated active future timestamps or
inactive literal indices rejected. The direct reference sorts candidate scores
using Python, with 1e-12 FP64 tolerances, separately from module top-k.

A constructed identity Q/K/V/output plus value-channel readout decodes all three
visible query answers correctly in every arm with identical parameters and no
training. This is a positive **representational witness**; it is not an observed
learned accuracy or native benchmark result, and the readout is never changed by
an oracle. Main gradient rows retain the paired random initializations.

Executed: these isolated correctness/backward checks only. Unattempted by Neris:
all new learned trials, native feature generation/fresh MQAR, Root2's <=300-second
experiment, integrated factorial/native architecture claims, GPU and production.
`protocol.json` was written before checks. `results.json` and immutable attempt
logs/receipts preserve the actual results; `manifest.json` pins all P1 top-level
files and the unchanged frozen-parent manifest. Root2 may snapshot/import these
read-only checks; any later candidate needs another version/hash.
