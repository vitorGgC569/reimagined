# Mamba3H synthetic benchmarks P0

Run from `C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined`:

```powershell
python -m research.mamba3h.benchmarks test
python -m research.mamba3h.benchmarks prepare
python -m research.mamba3h.benchmarks smoke
python -m research.mamba3h.benchmarks audit
python -m research.mamba3h.benchmarks verify-sha
```

These commands use the Python standard library on CPU, no numerical dependency,
GPU, model build, background worker or production edit. The package sets process
OMP/MKL/OPENBLAS/NUMEXPR flags to 2 before optional numerical imports. A caller
that uses PyTorch must also set `torch.set_num_threads(2)` and its own interop cap
before work. No model or frozen Mamba implementation is supplied here: Root2 owns
the actual native baseline and learned integration.

## Import and integration

```python
from research.mamba3h.benchmarks import (
    generate, solve, model_view, encode_numeric, compact_targets, label_vocabulary,
)

examples = generate("inst", 11, "train", 64,
    length=16, entities=3, operations=4, queries=4,
    distractors=2, overwrite=1, revocations=1, group="s5")
ep = examples[0]
assert solve(ep) == ep["targets"]  # Offline correctness check only.
view = model_view(ep)             # Whitelist: no labels, solver, seed or split.
x = encode_numeric(ep, dim=8)     # Python floats [L,8], event-local and causal.
labels = compact_targets(ep)      # Keep outside step/control.
```

`generate(task, seed, split, count, **difficulty)` returns a list of episodes.
Tasks are `mqar`, `group`, `inst`; splits are `train`, `validation`, `test`.
Each episode has `task`, `group`, `events`, categorical `tokens[L,4]`, `targets[L]`,
`query_mask[L]`, `control[L]`, and separate provenance `metadata`.

Targets are -100 off-query. Raw undefined is 125, counted as a supervised answer,
including after REVOKE and for an uninitialized entity. `compact_targets` maps
the raw vocabulary to contiguous readout classes. INST S5 uses `[0,1,2,3,4,125]`
and therefore six readout classes; undefined maps to class 5. Group S5 uses
120 permutation ranks plus undefined; MQAR and Z5^3 use 125 values plus undefined.
Do not use raw class 125 in the six-class loss. `label_vocabulary` is the inverse
mapping from compact predictions to raw symbolic answers for `protocol.accuracy`.

Numeric encoding is fixed, not fitted: kind/6, entity/63, (operator+1)/10,
(value+1)/125, task/2, group/2, query flag, revoke flag. Absent operator/value is
encoded 0. Larger dimensions add deterministic current-event sine features.
The `tokens` vocabulary uses kind ids from `schema.KINDS`, entity [0,63],
operator ids 1..10 (0 absent), and value ids 1..125 (0 absent).

For the common contract, form x[B,D] from the current numeric row. Pass only
current-event routing control to `step(x,state,control=...)`; reset state per
episode. `schema.validate_model_view` rejects extra fields or poisoned controls.
Episode targets and metadata must never be forwarded through `**episode`.

Control keys are `kind`, `entity`, `write_id`, `revoke_id`, `write`, `read`.
QUERY reads, WRITE/SET/OP write, REVOKE supplies the entity to delete. Neutral
events have both gates false. There is no value, answer or default slot address
in control. For Neris's `Control`, convert write/read to bool[B], entity to
long[B], and `revoke_id` to `revoke_entity` long[B]. Omit optional retain/address
until an adapter explicitly constructs causal capacity-valid routing. Entity
identity is not a slot index. QKV/stored values must be projections of preread
model x, even for oracle routing.

## Tasks, axes and splits

MQAR initializes one binding per entity, interleaves READs (QUERY), extra WRITE
bindings, REVOKE and neutral distractors. Each query asks its current entity's
latest unrevoked binding. Values are sampled independently of identities.

Group tracks one exact group element. S5 operators are the lexicographic ten
transpositions `(0,1)..(3,4)`, identity `tuple(range(5))`, permutation ranks are
lexicographic, and chronological action is `p_new[i]=op[p[i]]`. Z5^3 is a separate
commuting task with lexicographic ternary-base-5 encoding and operators listed
in `generation.ABELIAN_OPS`. SET resets the element; OP on a revoked state stays
undefined until a SET. INST maintains independently reset/revoked per-entity
states: one point in {0..4} acted on by S5, or one vector in Z5^3.

The online generator and query-local independent oracle share only the schema.
The MQAR oracle scans backwards. INST finds the last SET/REVOKE for the queried
entity and replays only that suffix. The S5 group oracle uses an inverse map and
independent Lehmer rank/unrank; the abelian oracle sums suffix moves then reduces.
Neither oracle reads labels, provenance or future events.

Length, entities, operations, distractors, overwrite and revocations are explicit
integer counts. For MQAR `operations` means READ count; for group/INST it means
OP count, with `queries` a separate count. `overwrite` is the number of extra
WRITE/SET events, which may overwrite live state or reinitialize after revoke;
the audit reports those actual counts separately. It never pretends these are
all live overwrites. DISTRACTOR is an observed random-value neutral event, not a
hidden write. NOOP pads exact L. Impossible budgets fail, never adjust another
axis. `entities` is the universe, not the memory slot budget. Single-stream group
requires entities=1; entity extrapolation belongs to MQAR/INST.

`manifests/p0.json` lists 26 bounded profiles. One-axis within-group profiles
change only one field from their base. Increasing L preserves the chronological
non-neutral stream and query answers. Within-group evaluation comes first.
Ordered pairs and cross-group generalization are separate exploratory profiles,
never pooled into within-group accuracy.

For unseen ordered pairs, pair partitions are train offsets 0..5, validation
offsets 6..7 and test offsets 8..9 modulo 10: 60/20/20 disjoint ordered edges,
all 100 covered, every operator has successors in each split. Pairs are successive
same-entity OPs, reset by SET/REVOKE. S5 has commuting and noncommuting pairs, so
this partition alone does not prove a noncommutativity mechanism. Cross-group
uses S5 train/validation and Z5^3 test, with different state semantics/vocabularies;
Root2 must predeclare a compatible shared readout before attempting that trial.

SHA-derived namespaces include task, paired seed, split, sample and regime.
Train/validation/test derived seeds and inputs are disjoint; paired arms reuse
identical episode bytes and sample seeds. Difficulty profiles deliberately reuse
the underlying sample RNG for matching. No arm id enters generator randomness.

## Bounded plans and evidence

Paired seeds: 11,23,37,53,71. Arms: M0,M1,M2,M3,MA plus MC commuting/capacity
control. Baseline commit is frozen `4db1250e7435318e1c2522b27649a5d5433f2f48`.
`p0.json` prepares 90 base-grid cells, all unattempted and unscheduled.
`pilot-v1-corpus.json` snapshots Root2's predeclared INST L16 protocol and pins its
15 datasets/800 examples and 30 initial cells. The corpus protocol is 64 train,
32 validation,64 test per seed, 20 updates, minibatch8, D8 and one hyperparameter
setting. Root2 owns execution/statuses; this directory executes zero learned jobs.

The outer P0 envelope is CPU<=2 threads, <=60 updates/job,90s/job and900s aggregate
learned wall time. Root2's pilot narrows updates to20. `check_budget(cells,
limits={"max_updates":20})` checks that narrower schedule. Stop nonfinite or
timeout and preserve failed/censored/unattempted cells. The reporter flags unknown
measurements, records parameter/persistent-state/cache/reserved bytes, actual
retrieval operations, latency, updates and wall time. It cannot enforce a runner's
wall clock or assert compute matching from missing counters.

`protocol.paired_summary` gives every seed's absolute accuracy/status and paired
delta. A mean and t95 interval (df4) require all five complete pairs. With missing
or failed pairs, CI/mean are undefined and all-seed worst-case accuracy-delta
bounds are retained. Partial censored accuracy is not a completed observation.
`factorial_summary` preserves the six-arm matrix and optional raw interaction
M3-M1-M2+M0. This interaction is descriptive, not a required complementarity test.
Gap recovery is undefined for ceiling-minus-baseline <=.01, including negative
gaps; no epsilon division or dropping of undefined cases.

```powershell
python -m research.mamba3h.benchmarks report --results research/mamba3h/benchmarks/manifests/results-template.json --task inst --profile pilot-v1
```

The report command reads only and never trains. Do not pool profiles/tasks.
`prepare` is deterministic for the frozen code/protocol; after Root2 snapshots,
use its pinned bytes rather than regenerating following a protocol revision.
`manifests/SHA256.json` pins source, corpora, protocols, documentation and receipts
(excluding pycache and itself). `sha` refreshes after intentional owned changes.
Test receipts preserve runs separately; exact wall time is observational, not a
reproducibility fingerprint for corpus generation.
