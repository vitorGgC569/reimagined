# Orin handoff: benchmark contract v1

Owner: Orin, only `research/mamba3h/benchmarks/`. Date:2026-10-01.
Baseline: `4db1250e7435318e1c2522b27649a5d5433f2f48`, unchanged.
No OXN/nsos edit, source/build/GPU work, model training, commit or push.

## Executed

- 50/50 CPU tests PASS, 0 failures,0 errors,0 skipped. Receipt `receipts/tests.json`
  and `receipts/tests-run-001.json`. A preceding 46-test pass was followed by four
  additional correctness/reporting tests; no test tolerance was relaxed.
- Exact S5 labels checked against an independent oracle over all120 permutations
  and10 transpositions; long random chains, abelian inverse/order, overwrite,
  revoke, missing entities and entity isolation pass.
- Target/provenance poisoning, future-suffix mutation, event/control target
  injection and stale serialized control negatives pass. No labels/values enter
  control; numeric input depends only on the current observed event.
- Split and pair tests pass: disjoint train/validation/test seeds and input hashes,
  60/20/20 ordered-pair partitions, correct per-entity reset boundaries, separate
  cross-group semantics and 26 valid one-axis/separate-regime profiles.
- `prepare` produced60 JSONL files/3200 unique model inputs with independent solver
  agreement:2400 base examples plus800 Root2 pilot-v1 examples.
- Full audit PASS over all3200 examples/60files,0 SHA/count/input-fingerprint/
  duplicate/causal-target failures. Receipt `receipts/audit.json` includes actual
  live overwrites, reinitializations, undefined queries and order-sensitive query
  counts by task/seed/split, without dropping undefined observations.
- Component smoke:400 examples,5 paired seeds across MQAR,group S5/abelian,INST
  S5/abelian. Solver accuracy1.0 everywhere. Negative stale-memory MQAR accuracy
  0.5625..0.796875 and constant-undefined0.09375..0.140625 in MQAR. Reversed order
  preserves abelian answers1.0; S5 group accuracy0.875..0.979167. Receipt
  `receipts/smoke.json`. These are oracle/diagnostic accuracies, not learned arms.

## Prepared, not executed here

- `manifests/p0.json`:90 base cells M0/M1/M2/M3/MA/MC x5seeds x3tasks, all
  unattempted/unscheduled;26 additional bounded profiles are listed, not trained.
- `manifests/pilot-v1-corpus.json`:Root2's exact predeclared INST S5 L16,
  entities3,operations4,queries4,distractors2,overwrite1,revocations1;64train/
  32validation/64test per seed. Protocol bytes/SHA snapshot,15files/800examples,
  30initial cells. Training belongs to Root2 and remains unattempted in this
  prepared handoff, regardless of later external execution.
- `manifests/results-template.json`:all120 prepared result cells, explicit
  unattempted statuses and null measured metrics. No success-only placeholder.
- CI/gap/budget/factorial reporting implemented and tested with deliberate failed,
  censored and absent cells. No actual learned outcome or compute matching claim.

## Exact API for Root2

Import from `research.mamba3h.benchmarks`:

```python
generate(task, seed, split, count, **difficulty)  # list[dict]
solve(ep)                                     # raw targets[L], offline only
model_view(ep)                                # label-free whitelist dict
encode_numeric(ep, dim=8)                     # current-event floats[L,D]
compact_targets(ep)                           # readout targets, never control
label_vocabulary(task, group)                 # compact->raw symbolic ids
```

Use `model_view` plus current `encode_numeric` row to adapt the common
`step(x[B,D],state,control)->(y,next_state,stats)` interface. Reset per episode.
Six-class INST S5 readout requires raw undefined125 -> compact5; non-query -100
remains ignored. Keep solver, targets, metadata, split and seed outside model.
Neris control conversion: write/read bool[B], entity long[B], `revoke_id` ->
`revoke_entity` long[B]. Optional retain/address are absent by default. Never use
entity id as a capacity slot index. Values only enter visible tokens/model x.

Read exact pinned JSONL from:

```text
benchmarks/datasets/pilot-v1/inst/seed-{11,23,37,53,71}/{train,validation,test}.jsonl
benchmarks/datasets/{mqar,group,inst}/seed-{11,23,37,53,71}/{train,validation,test}.jsonl
```

Paths in corpus manifests are relative to `benchmarks/`. SHA pins every corpus
file and model-input fingerprint. `manifests/SHA256.json` is the final owned
source/data/receipt hash inventory; its aggregate fingerprint is sent via Maestri
after the final refresh (not embedded here to avoid a self-referential hash).
`README.md` provides full vocabularies, semantic counts, controls and commands.

## Limits affecting interpretation

INST's cheap base smoke has reversed-order accuracy0.958333..1.0. Many queries
have fewer than two relevant operations or order-insensitive outcomes. The
adversarial fixture demonstrates order sensitivity exists; this small pilot is
weak evidence for a noncommutativity learning mechanism. Use the audit's measured
order-sensitive counts and predeclare a harder profile before a later trial.

The overwrite axis is an extra-WRITE/SET event count, not a guarantee every event
hits live state; post-revoke reinitialization is separately reported. Group is
single-stream; entity extrapolation belongs to INST/MQAR. Cross-group changes
state semantics/vocabulary and needs a separately predeclared compatible readout.
Known budget checks do not enforce an external runner's timer. Record actual
parameter/state/cache/reserved bytes and work residuals; missing measurements
cannot establish matching. Gap recovery <=.01 is undefined. Incomplete five-seed
pairs have no successful-subset mean/CI. Failures and censored jobs stay explicit.

Evidence supports synthetic component/contract correctness. It does not establish
learned feasibility, integrated Mamba3H superiority, mechanism identification,
SOTA, language quality, multimodal ability or native end-to-end training.

## Reproduce

```powershell
python -m research.mamba3h.benchmarks test
python -m research.mamba3h.benchmarks prepare
python -m research.mamba3h.benchmarks smoke
python -m research.mamba3h.benchmarks audit
python -m research.mamba3h.benchmarks verify-sha
```

Run at the absolute project root. CPU standard-library execution,<=2 thread
flags,no build/GPU. Root2 should snapshot before training; no ask-back is needed.
