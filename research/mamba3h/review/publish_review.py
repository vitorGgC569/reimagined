"""Publish evidence-based verdict and handoff; no new training or selection."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[3];A=ROOT/'research/mamba3h'
def read(path):return json.loads((A/path).read_text())
def put(path,text):(A/path).write_text(text)
def main():
    old=read('review/pilot/summary.json');new=read('review/pilot-v2/summary.json');d=read('review/pilot-v2/diagnostics.json')
    assert all(j['status']=='completed' for j in old['ledger']+new['ledger'])
    peers=read('review/handoffs/receipt.json');tests=read('review/final-peer-tests.json');assert tests['passed']
    audit=read('review/handoffs/research/mamba3h/benchmarks/receipts/audit.json')
    rows=[r for r in audit['rows'] if r['profile']=='pilot-v1' and r['split']=='test']
    ordered=sum(r['counts'].get('order_sensitive_query',0) for r in rows);queries=sum(r['counts']['QUERY'] for r in rows)
    original_wall=sum(j['wall_seconds'] for j in old['ledger']);corrected_wall=sum(j['wall_seconds'] for j in new['ledger'])
    header='''# Root2 independent P0 verdict

Verdict: component and causal-adapter implementation is viable for further
optional research; integrated learned advantage is **not demonstrated**.
Native end-to-end Mamba3H and production enablement remain unattempted. Fixed
rank4 residual recompression is rejected as an exact closure strategy.

This is `actual_native_frozen_backbone_adapter_P0`: the real CPU Mamba3 layer
generates frozen features, a common visible-input residual is added, and a
six-class readout/optional Torch adapters are trained. The optional algebra
state is auxiliary, global per episode, and does not replace the native SSM.
There is no recurrence port or toy Mamba baseline. The datasets are synthetic.
These observations establish neither SOTA nor language/multimodal quality.

## Frozen identity and independent correctness

M0 source commit: `4db1250e7435318e1c2522b27649a5d5433f2f48`.
CPU provider SHA256: `050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f`.
Freeze receipt: `../manifests/freeze.json`. Binary hash alone is not a build
provenance certificate. Executed parity supplies the behavioral gate:

- Pinned upstream reference: eight MIMO/rotary/norm configurations; forward,
  input and every parameter VJP passed, max output abs error 2.69596604e-7.
- Root2 native rank1/4 forward, all boundary states, chunk composition and
  cross-chunk input/parameter/initial-state VJP passed; chunk output bitwise
  identical. Smooth-branch initial-state/input finite differences passed.
- Padded NaN tails and zero valid-prefix states passed. Physical zero-token
  calls are unsupported and explicitly recorded, not silently counted as PASS.
- Root2 five independent integration/negative tests and two v2 wiring tests
  passed: causal prefix, reset/isolation, label/metadata poisoning, routing-only
  oracles, no-read work avoidance, no stale entity evidence, disabled algebra
  identity and the memory-value gradient path into algebra.
- Completed copied peer suites were rerun in separate capped processes:
  algebra9 + memory20 + benchmarks50 =79 PASS. All800 executed pilot examples
  exactly matched Orin's final SHA-pinned JSONL corpus and compact label map.

Lyra's independently delivered native oracle also passed16 full-layer variants
and252 directional derivatives. It is corroboration, not an additional hybrid
architecture claim. Native and Torch run in separate CPU processes to avoid
mixed OpenMP-runtime interference.

## Declared paired experiment and causal correction

Seeds11/23/37/53/71; INST S5 within-group; D8, length16, entities3, OP4,
QUERY4, distractors2, extra SET1, REVOKE1; train64/validation32/test64 per seed.
Rank1, capacity4, top-k1. Twenty Adam updates, lr.01, minibatch8, gradient
clip1, one fixed hyperparameter setting. No validation/test model selection.

v1 executed all30 M0/M1/M2/M3/MA/MC cells. Its M3 had parallel branches:
memory consumed native `z`, while algebra produced `ast`. This is preserved in
`pilot/`; original M3 mean16.5625%, delta+1.09375pp, t95[-4.84063,+7.02813]pp.
Coordinator requested review of this wiring. Before rerunning, protocolv2
declared M3 memory input=`ast` BEFORE retrieval; Q/K/V and written contents now
derive from structured model state, without an answer/future oracle. Two
independent tests verify the input equality and value-storage gradient into
algebra. Hyperparameters, data, RNG/minibatch schedule and all other arms were
unchanged. Five new M3 cells ran;25 other cells were reused with exact SHA
provenance. v1 is not replaced by a successful subset. v2 is a requested causal
wiring correction, not a test-guided hyperparameter search.

Corrected v2 accuracy and paired conventional t95 intervals (df4):

| Arm | Mean test accuracy | Delta vs M0 (pp) | t95 delta (pp) |
|---|---:|---:|---:|
'''
    for r in d['rows']:
        arm=r['arm'];cmp=new['paired_vs_M0'].get(arm)
        ci=cmp['interval_95_t_df4'] if cmp else None
        header+=f"|{arm}|{100*r['accuracy_mean']:.5f}%|{100*cmp['mean_delta']:+.5f}" if cmp else f"|{arm}|{100*r['accuracy_mean']:.5f}%|—"
        header+=f"|[{100*ci[0]:+.5f}, {100*ci[1]:+.5f}]|\n" if ci else '|—|\n'
    header+='''
All gain intervals include zero. Five-seed raw-accuracy interaction is optional
descriptive evidence, not a necessary criterion for useful complementarity.
'''
    inter=d['interaction'];header+=f"Corrected interaction mean{100*inter['mean']:+.5f}pp, t95[{100*inter['interval_95_t_df4'][0]:+.5f},{100*inter['interval_95_t_df4'][1]:+.5f}]pp.\n\n"
    header+='''Every seed, absolute correct/query counts, losses, updates, actual retrieval
counts, gradients and timings remain in summary/diagnostic JSONs. No measured
admissible memory-oracle ceiling exists, so gap recovery is undefined. The
reporting code also tests that ceiling-baseline<=.01 stays undefined, including
the floating boundary1-.99. A replay solver's perfect accuracy is a label
correctness check, not a learned-arm ceiling.

## Budgets, conditioning and mechanism limits

All arms share3340 frozen native parameters and5408 native state bytes/example.
The following are **additional** trainable/auxiliary budgets. Measured latency
covers cached-feature adapters/readout on64 test episodes; native feature
extraction is excluded and separately timed in each seed's native.json.

| Arm | Trainable params | Extra state bytes/example | Mean cached test latency (s) |
|---|---:|---:|---:|
'''
    for r in d['rows']:header+=f"|{r['arm']}|{r['trainable_parameters']}|{r['aux_state_bytes_per_example']}|{r['test_latency_seconds_mean']:.5f}|\n"
    header+='''
M1/MC have equal raw parameter counts and reserved basis buffers; effective
capacity and FLOPs differ. M2/MA share initial modules/capacity; semantic WRITE
versus anonymous store-all changes occupancy/searches/eviction. M3 has more
parameters/state than M0. Full compute/parameter/state matching is **not**
established. Persistent sizes include logical tensor metadata/counters, not
Python overhead. Cache numbers are explicit tensor-workspace estimates, not
allocator peaks, and exclude comprehensive autograd/optimizer retention.
Frozen input+native feature cache is65536/32768/65536 bytes per seed for
train/validation/test. Read/search/gather counts are measured primitives;
complete sorting/validation/allocation work is unmeasured.

Hard top-k1 softmax is constant and discrete selection has no STE. Q/K cannot
learn via that path; explicit-routing gates are inactive. Effective parameters
with nonzero gradients are M0=54,M1=224,M2=182,M3=352,MA=182,MC=224.
Entity metadata restricts memory reads; this pilot does not establish learned
addressing or learned discrete routing. OP writes supplied model state; there
is no oracle group-state update inside memory. Global auxiliary algebra state
also mixes entities and is not an entity-scoped composition mechanism.

Native features were not normalized: seed11 feature-norm p99 was~123.
Auxiliary state maxima reached~697 and preclip gradient maxima~248. All values
remained finite and clipping applied. These additive-state measurements do not
certify long-context/switching stability. Conditioning and the short fixed
schedule constrain interpretation; no post-test scaling or update retuning ran.
The train-selected constant classifier averaged15.625% test accuracy, versus
15.46875% for M0; uniform six-class reference is16.66667%. Performance is weak.

'''
    header+=f"The final corpus audit found only{ordered}/{queries} ({100*ordered/queries:.3f}%) pilot test queries sensitive to reversing the relevant operation order. This is a weak test of a noncommutativity learning mechanism. The extra-SET quota is an event count:277 live overwrites and43 post-revoke reinitializations were observed across the1280 test queries; they are not silently conflated.\n\n"
    header+='''## Closure, component evidence and failures

Exact dense affine sequential/chunk/tree composition passed. Small FP64 S5
relative errors~2.2e-16; held-out D64 dense probes~1.85e-15. Repeated rank4
residual recompression failed: relative norm error~1.24 and tree-association
error~.63. Rank-r factors over time do not stay rank-r. No approximate scan is
enabled. Bounded diagonal/unit-factor operators supply a common homogeneous
contraction bound; Lyra perturbation gain.785678 matched.99^24. The switched
shear negative has local spectral radius.9 but product norm26891.62; local
spectra alone do not certify switching stability.

Lyra isolated operator-fit learned feasibility was demonstrated on known S5
transpositions across five seeds/ranks1/2/4; Neris isolated k2 anonymous-memory
smoke averaged83.125% versus47.5% sparse-store-all. Neris's delta35.625pp has
t95[5.49405,65.75595]pp, but attention stores distractors and may evict useful
writes: residual work/occupancy differs. Neither isolated component result
substitutes for the weak actual-native-backbone factorial.

Preserved issues: finite-difference phase-wrap crossing, unavailable mutable
weight binding, unsupported physical empty input, combined-thread test harness
import conflict, initial Neris thread-import-order limitation, and recompression
closure failure. See FAILURES.md and original logs/receipts. No tolerance was
relaxed. All35 learned jobs completed; zero failed/censored jobs in these grids.
'''
    header+=f"Actual Root2 learned-process wall time{original_wall:.3f}+{corrected_wall:.3f}={original_wall+corrected_wall:.3f}s, below900s; each job<=90s, each process<=2 threads. Native preparation and correctness receipts are separate.\n\n"
    header+='''## Remaining cases and decision

Orin's90 base learned cells and26 additional profiles remain unattempted;
its30 prepared pilot cells map to Root2's executed v1 grid. Updated statuses
live in Root2 manifests, leaving peer prepared receipts unchanged. MQAR/group
learned grids, unseen ordered pairs, cross-group, integrated ranks2/4, full
compute matching, entity-scoped structured memory update, differentiable
addressing/gates, native end-to-end learning, GPU/pretraining/language/multimodal
are unattempted. Multimodal/quantization are later stages.

Production recommendation: keep P0 optional and disabled; no production/GPU
integration handoff is recommended by this evidence. Next bounded protocol
would need independent conditioning controls, k>=2/appropriate addressing
gradient design, entity-scoped composition, an order-sensitive dataset and
matched capacity/compute controls declared before new learned trials. That
protocol is not executed here. Codex's separate GPU quality results are not
used for P0 tuning, pooling or SOTA claims.
'''
    put('review/VERDICT.md',header)
    machine=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_commit=old['protocol'].get('baseline_commit','4db1250e7435318e1c2522b27649a5d5433f2f48'),
        label=old['label'],native_equivalence='passed',peer_correctness_tests=79,root_independent_tests=7,
        corpus_exact_episodes=800,original_learned_jobs=30,causal_correction_jobs=5,failed_jobs=0,censored_jobs=0,
        learned_wall_seconds=original_wall+corrected_wall,integrated_advantage='not_demonstrated',
        component_feasibility='supported_on_isolated_controls',fixed_rank4_closure='rejected',
        whole_grid_budget_matching='not_established',native_end_to_end='unattempted',production_enablement='not_recommended',
        order_sensitive_test_queries=ordered,total_test_queries=queries,full_5_seed_pairs=True,
        peer_manifests={p['owner']:p['manifest_sha256'] for p in peers['owners']},
        corrected_paired_comparisons=new['paired_vs_M0'],unattempted=old['protocol']['unattempted_axes'])
    put('review/verdict.json',json.dumps(machine,indent=2)+'\n')
    handoff='''# Root2 completed P0 handoff

Evidence-based verdict: optional component/adapters implemented and independently
reviewed; learned superiority is not demonstrated. Native Mamba3H end-to-end and
production enablement are unattempted. No OXN/nsos/default/GPU/shared-build/role
edit, commit or push was made. Maestri team messages were explicitly authorized.

Read `review/VERDICT.md` and `review/verdict.json` for results, limitations and
failures. Corrected five-seed M3 mean17.03125%, M0 mean15.46875%, delta+1.5625pp,
t95[-6.27332,+9.39832]pp. Original30-case v1 remains intact; only five causal
M3 wiring corrections ran under declared v2, with25 unchanged cells SHA-reused.
Root2 learned CPU wall~160.823s/900s, max2threads, fixed20 updates.

Delivered:

- `integration/CONTRACT.md`: small differentiable explicit-state interface.
- `manifests/freeze.json`: commit/native/source hashes; CPU provider behavioral
  parity checked against actual native and pinned upstream references.
- `integration/native_cpu.py`: isolated real-native CPU feature/parity bridge.
- `integration/vendor/`: immutable released APIs used in both pilot versions.
- `review/handoffs/`: completed peer source/data/receipt snapshots with all
  owner SHA manifests independently verified (103 files).
- `manifests/pilot-v1.json`, `pilot-v2.json`: predeclared schedules and causal
  revision; `review/pilot*/`: all seed records, raw datasets/features, statuses,
  exact reuse provenance, paired intervals, work/bytes/latency/gradient diagnostics.
- `review/independent_tests.py`, `wiring_tests.py`, `final_peer_tests.py`:
  executed7 Root2 tests +79 copied peer tests; all800 corpus examples matched.
- `review/FAILURES.md`: preserved failed probes and scientific limitations.
- `manifests/root2-manifest.json`: final SHA inventory (created after this file).

Reproduce from absolute repository root, CPU only:

```powershell
python research/mamba3h/integration/native_cpu.py smoke --out research/mamba3h/review/repro-native
python -B research/mamba3h/review/independent_tests.py
python -B research/mamba3h/review/wiring_tests.py
python -B research/mamba3h/review/final_peer_tests.py
python research/mamba3h/review/freeze_final.py verify
```

Runner `paired_pilot.py` accepts prepare/train/run/summarize and --protocol.
To reproduce learned trials, choose a new optional --out directory, preserve
existing evidence and copy the five pinned datasets. Do not overwrite frozen
receipts or silently schedule the90 unattempted base jobs. v2 reuse preparation
and ledger are exact, not a successful-seed subset.

Remaining native GPU/production work belongs to Codex; no GPU request is pending
from Root2. Subsequent research requires a fresh declared protocol; current P0
execution and independent handoff are complete.
'''
    put('integration/HANDOFF.md',handoff);print('Published VERDICT.md, verdict.json, integration/HANDOFF.md')
if __name__=='__main__':main()
