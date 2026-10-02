# Lyra P1 exact per-entity checks

Scope: **isolated exact known operator control NOT actual-native/NCE2E**.
New files only under `research/mamba3h/algebra/p1_checks/`. All P0 algebra
artifact bytes, including HANDOFF and manifest, verified unchanged before/after.
No learned runs, optimizer updates, native calls, GPU, shared build, peer edits,
commit or push. CPU numerical environment flags and torch thread count are 2.

## Executable API and causal information

`exact_entity.py` defines `ExactEntityControl(entities=3,per_entity=True)` and
`step(events,state=None) -> logits[B,6],next_state[B,slots,5],stats`. One current
observed event per independent batch sample is accepted, with exactly the four
fields `{kind,entity,operator,value}`. No episode, label, answer, solver, metadata,
logits or future object is accepted by step. `run_events(events,model)` loops
causally and resets state explicitly on each call.

SET overwrites the addressed five-dimensional state with a one-hot decoded
from that **observed SET token** value0..4. Current OP uses the known lexical
S5 transposition Householder direction `(ei-ej)/sqrt(2)`, alpha2, identity
diagonal. No accumulated target permutation or query answer is supplied to
the model. QUERY is nonmutating and applies a fixed linear model readout:
first five logits are the state coordinates; sixth is `1-sum(state)`.
An uninitialized/revoked all-zero state yields compact undefined class5.
REVOKE writes zero, and subsequent OP cannot restore the entity; a later SET
reinitializes it. NOOP/DISTRACTOR tokens leave state unchanged.

The readout has fixed weight `[I5;-ones]` and bias `[0,0,0,0,0,1]`, registered
as model buffers. No answer oracle enters QUERY. Routing, SET decoding,
operators and readout are hand-specified representational structure, not
learned from native features or learned task embeddings. There are no keys,
values, targets-based caches or similarity retrievals. Although the model
is a torch nn.Module and readout is differentiable, no optimization was run.

Global uses one shared five-vector for all entity events; per-entity uses E
independently addressed five-vectors. Both have zero learned parameters and
688 fixed buffer bytes in FP64. Active persistent state per sample is
40 bytes global vs `40*E` bytes per-entity; the E3 advantage is **not state
capacity matched**. State sizes, operation counters and timing are reported.
No complete FLOP/temporary workspace matching claim is made.

Labels are computed offline by Orin's unchanged
`research/mamba3h/benchmarks/oracle.py:solve`. Its query-local entity-suffix
replay does not use this Householder implementation. Model execution receives
only `ep['events']`, never `ep['targets']`. Orin's raw INST undefined=125,
ignore=-100, and raw vocabulary `[0,1,2,3,4,125]` are mapped to compact labels
`[0,1,2,3,4,5]` only for offline comparison. Undefined queries are included.
Orin's fresh MQAR P1 corpus is a separate control and is not consumed here.

## Predeclared and executed

`protocol.json` was written before tests. Bounded grid: seeds11/23/37/53/71,
train/validation/test, eight episodes per seed/split and configuration:
E1 L12 OP4 QUERY4 distractor1 overwrite1 revoke1; E3 L16 OP4 QUERY4
distractor2 overwrite1 revoke1. No hyperparameter search or learned arm.
240 generated episodes, 960 queries per arm, 30 paired cells. Wall budget60s;
executed test/grid time0.634s excluding interpreter/library import. No failed,
censored or unattempted cells. Every global mismatch is preserved in results.

`test_exact_entity.py`: **9/9 PASS**. Tests cover all50 observed point/operator
combinations in both controls; reversed noncommutative OP-order negative;
directed global entity interference; overwrite, revoke, OP-after-revoke and
reinitialization; neutral/query nonmutation; altered future/labels/metadata;
label leakage rejection; batch isolation/reset/state bytes; fixed readout
and derivative; illegal event and state rejection. FP32 and FP64 smoke pass.

| Grid | Global | Per-entity | Active FP64 state/sample |
|---|---:|---:|---|
|E1|480/480 (100%)|480/480 (100%)|40 vs40 bytes|
|E3|161/480 (33.5417%)|480/480 (100%)|40 vs120 bytes|

All fifteen E3 per-entity cells individually reach1.0. Global E3 test
accuracies by seeds11/23/37/53/71 are .25/.34375/.21875/.375/.1875;
the matching per-entity test accuracies are all1.0. No equivalence or numerical
exactness result is promoted to a learned or native architecture claim.
Single-entity equivalence and multi-entity isolation establish a positive
control of this explicit representation under known operations and routing.

## Reproduction, freeze and boundaries

```text
cd research/mamba3h/algebra/p1_checks
python -B test_exact_entity.py
python -B run_checks.py
python -B freeze.py --verify
```

`results.json`, `tests.log` and separate `manifest.json` pin the executed
evidence, protocol, local code and unchanged read-only benchmark dependencies.
P0 owned-file hashes before/after are in results; the old P0 manifest SHA256
remains `7a389aac8f7d54fef77344b6b8c9ef1704235586b40799d56e2f58a1c80ed43d`.
Snapshot this new directory before re-execution, which replaces P1 evidence;
the separate manifest itself is generated once and verifies pinned bytes.
Interpreter bytecode writing is disabled when executing the new runner/tests.

Prepared/unattempted: learned routing/operator/readout, actual-native adapter,
NCE2E, native factorial trial, GPU, language/SOTA. Root2 owns integration and
independent review. No blocker remains for importing this isolated control.
