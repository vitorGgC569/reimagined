"""Publish all P1 outcomes without pooling phases or selecting successful seeds."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h';OUT=AREA/'review/p1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((OUT/'positive-r2/summary.json').read_text());mechanism=json.loads((OUT/'mechanism/results.json').read_text())
preflight=json.loads((OUT/'preflight-r2/receipt.json').read_text());handoffs=json.loads((OUT/'handoffs/receipt.json').read_text())
assert sha(AREA/'integration/p1/positive_runner.py')==preflight['runner_sha256']
assert len(s['ledger'])==25 and all(r['status']=='completed' for r in s['ledger'])
assert len(mechanism['ledger'])==45 and all(r['status']=='completed' for r in mechanism['ledger'])
assert len(s['initialization_checks'])==5
summary=[]
for arm in s['protocol']['arms']:
    r=[x for x in s['records'] if x['arm']==arm];assert len(r)==5
    summary.append(dict(arm=arm,test_accuracy_mean=statistics.mean(x['evaluations']['test']['accuracy'] for x in r),
        seed_accuracies={x['seed']:x['evaluations']['test']['accuracy'] for x in r},
        parameters=r[0]['parameters'],active_parameter_tensor_sizes=[x['gradient_active_parameter_tensor_sizes'] for x in r],
        active_aux_bytes_per_example=r[0]['evaluations']['test']['active_aux_bytes_per_example'],reserved_aux_bytes_per_example=630,
        cached_adapter_latency_ms_per_example_mean=statistics.mean(x['evaluations']['test']['latency_seconds']*1000/64 for x in r),
        qk_nonzero_gradient_updates=[sum(v['query']>0 and v['key']>0 for v in x['qk_gradient_norms']) for x in r],
        test_counters_per_seed=r[0]['evaluations']['test']['actual_counters']))
with (OUT/'paired-results.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['seed','arm','accuracy','parameters','active_parameter_tensor_sizes','updates','active_aux_bytes','similarity_evaluations','value_reads','writes','evictions'])
    writer.writeheader()
    for r in s['records']:
        e=r['evaluations']['test'];c=e['actual_counters']
        writer.writerow(dict(seed=r['seed'],arm=r['arm'],accuracy=e['accuracy'],parameters=r['parameters'],active_parameter_tensor_sizes=r['gradient_active_parameter_tensor_sizes'],updates=r['updates'],active_aux_bytes=e['active_aux_bytes_per_example'],**{k:c[k] for k in ('similarity_evaluations','value_reads','writes','evictions')}))
verdict=dict(scope=s['protocol']['claim_scope'],status='positive_control_learned_memory_feasibility_supported',
    full_Mamba3H_advantage='not_demonstrated',native_end_to_end='unattempted',NC_complementarity='not_demonstrated',
    attention_superiority='not_demonstrated',learned_QK_specific_benefit='exploratory_positive_mean_but_anonymous_interval_includes_zero',
    summaries=summary,paired_comparisons=s['paired_comparisons'],
    learned_completed=25,learned_failed=0,learned_censored=0,learned_unattempted=0,
    prior_setup_failed=6,prior_setup_unattempted=19,setup_updates=0,
    learned_wall_seconds=s['learned_wall_seconds'],setup_conservative_charge_seconds=s['setup_failure_conservative_charge_seconds'],
    cumulative_root2_charged_wall_seconds=s['cumulative_root2_learned_wall_seconds'],
    tests_passed=43,model_numeric_solver_examples_verified=800,recovered_native_all5seed_bitwise_replay=True,
    mechanism_updates=0,mechanism_completed=45,mechanism_results_sha256=sha(OUT/'mechanism/results.json'),
    protocol_sha256=sha(AREA/'manifests/p1a-positive-r2.json'),runner_sha256=sha(AREA/'integration/p1/positive_runner.py'),
    old_P0_322_files_unchanged=True,old_P0_manifest_sha256=handoffs['old_manifest_sha256'],
    shared_current_baseline_drift=handoffs['shared_baseline_drift'],private_provider_sha256=preflight['provider_sha256'],
    GPU=False,shared_build=False,OXN_edits=False,commit_push=False)
(OUT/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
table='\n'.join(f"| {r['arm']} | {r['test_accuracy_mean']*100:.4f}% | {r['parameters']} | {r['active_parameter_tensor_sizes'][0]} | {r['active_aux_bytes_per_example']} |" for r in summary)
text=f'''# Root2 P1 independent research verdict

The fresh positive control supports learned explicit-memory feasibility on this
small retrieval task. It strengthens the case for continuing optional research.
Full Mamba3H superiority and noncommutative-memory complementarity remain
unproved; the prior P0 INST null is preserved, not replaced by this easier task.

## Executed evidence

Actual native frozen CPU Mamba3 features, no recurrence port: D8, MIMO rank1,
N128/P4, 3340 frozen parameters and 5408 native state bytes/example.
Seeds11/23/37/53/71, fresh MQAR L6, three distinct WRITE tokens plus two shuffled
entity-only NOOPs and one final QUERY; four entities/values,64train/32validation/
64test each. Thirty fixed Adam updates,lr.03,batch8,clip1; no test tuning or
best-checkpoint selection. Native/raw channels receive training-only RMS scaling.

All25 declared cells completed,0learned failures/censors/unattempted.
Native features, all model inputs, split labels, weights and raw cell results
are saved. All memory Q/K/V/output/head initial tensors were bitwise matched
for all five seeds.800unique inputs, independent solver/numeric encoding checked.
Root5 + Lyra9 + Neris12 + Orin17 =43tests passed in isolated CPU processes.

| Arm | Mean test accuracy | Trainable scalars | Gradient-active tensor sizes | Active auxiliary bytes/example |
|---|---:|---:|---:|---:|
{table}

Anonymous k2 minus M0: +23.4375pp, five paired unadjusted t95
[+6.69184,+40.18316]pp; all five deltas positive.
Anonymous k2 minus MC with the same319 trainable scalars: +25.625pp,
t95[+8.40286,+42.84714]pp. Extra feedforward parameters alone did not reproduce
the memory gain in this schedule; this does not establish every capacity control.
Symbolic ADDRESS minus M0: +62.1875pp,t95[+52.84274,+71.53226]pp.
Anonymous memory minus anonymous sparse attention: +12.5pp,
t95[-1.14981,+26.14981]pp; superiority over attention is not demonstrated.
These are small five-seed synthetic estimates with multiple exploratory
comparisons and no multiplicity correction. Dataset shifts are not pooled.

## Routing, budgets and runtime

Both anonymous arms have nonzero Q/K gradients on all30 updates in all5seeds,
292gradient-active tensor sizes. Symbolic ADDRESS has one candidate and zero
Q/K gradients;164active tensor sizes. Gates27 are unused in all memory arms.
Routing uses current observed WRITE/QUERY kinds, not learned write/read gates.
Symbolic routing additionally uses visible entity IDs; it never supplies values,
answers/logits or future contents. Anonymous control sends entity=-1.

C6/k2: query candidates symbolic1/anonymous3/store-all5 per episode;
similarity evaluations per64test examples64/192/320, value reads64/128/128,
writes192/192/384. MA includes a harmless query write after retrieval.
No eviction/overwrite/revocation in this positive task. Every query scans6slots;
no similarity/topk work on nonqueries. All17 counters are saved per cell.

Memory319scalars=1276parameter bytes including readout; active state630bytes.
All arms reserve630logical auxiliary bytes, but M0/MC use dead padding with zero
active memory. This is byte reservation, not equal semantic capacity, FLOPs or
allocator/autograd peak. Frozen native parameters/state are additional and common.
Cached-adapter mean latency per episode is in verdict.json; it excludes native
feature extraction, which was separately measured and pinned per seed.
Python routing overhead and occupancy/search work differ. No speed claim follows.

## Zero-update mechanism audit

Declared after primary test scores, before these interventions: explicitly posthoc
exploratory, not a fresh confirmation.45cells completed,0optimizer updates;
all15full-checkpoint test scores reproduced exactly. Each intervention retains
final value/output/readout parameters except the named tensor change.

Anonymous memory accuracy55.9375% fell to33.4375% when its retrieval output was
zeroed: paired +22.5pp,t95[+11.47381,+33.52619]pp. This supports dependence on the
retrieval residual of those trained checkpoints, not a retrained baseline claim.
Restoring only Q/K to their initial values gave42.5%: full-minus-initialQK
+13.4375pp,t95[-0.45828,+27.33328]pp. Anonymous Q/K improvement remains uncertain
despite nonzero gradients. Original relevant-WRITE slot inclusion increased
from79.375% to87.8125%; this is a proxy, since other native token states may carry
related information. Symbolic restore-QK is exactly unchanged, as predicted.
Sparse attention full-minus-initialQK +6.875pp,t95[+2.20262,+11.54738]pp.
All variants/seeds and interventions are retained; no tuning follows them.

## Other positive controls and interpretation

Lyra's fixed known S5 operator control: single entity global/perentity480/480;
three entities global161/480 versus perentity480/480. This shows the importance
of entity-scoped state on INST. It is zero-update known-operator correctness,
not learned native algebra;40vs120FP64state bytes are not capacity matched.
Neris independently verified12positive/negative memory checks, identical module
initialization, no future access and literal-address redundancy with symbolic
entity matching. Orin's exact observed-token reference solves800/800, not a
learned model or admissible learned-addressing ceiling. Gap recovery is undefined.

P0 corrected M3 minus M0 remains+1.5625pp with interval crossing zero. Its global
auxiliary algebra and sparse order-sensitive queries do not constitute a strong
test of perentity NC dynamics. No learned NC synergy is claimed from P1 MQAR.
Fixed-rank recompression closure/stability failures remain part of the evidence.

Next decisive study needs fresh order-sensitive multi-entity INST, explicit
perentity state, overwrite/revoke cases, a commuting/control factorial with
active capacity/work differences disclosed, and eventually native end-to-end
gradients and learned routing. These are unattempted here, not implied successes.
No language, long-context, multimodal, production enablement or SOTA result exists.

## Preservation and execution limits

M0 commit4db1250; original CPU SHA050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f.
Shared CPU provider/source/executables changed during production work. Old
Root2 frozen evidence322files remains byte-identical. Exact original .pyd was
recovered from root-baseline-4db1250 and privately copied under integration/p1.
P0 code/hash guard preserved through a location-only bridge; five-seed original
P0 feature replay was bitwise identical in every split. Current shared source
is not used as the P1 identity; its drift is explicitly inventoried.

The initial guard rejected preparation. A harness launch error still attempted
five failed setup children and one further failed setup before logging;19jobs
unattempted,0updates. The sixth duration was not captured and is conservatively
charged90seconds. Complete setup charge101.58506s is retained. The revision adds
readiness/no-overwrite gates and exact private provider; data/hparams unchanged.
One partial collection guard and one copied-test import setup failure are also
retained; both occurred before optimization. See FAILURES.md.

New25learned jobs wall73.77516s; cumulative Root2 charged wall336.18304s/900s,
including P0 and conservative setup charge. CPU max2threads/process. No GPU,
shared build, OXN/default/role/peer-active-file edit, commit or push by Root2.
SHA manifests keep P0 and P1 identities separate; no failed record is overwritten.
'''
(OUT/'VERDICT.md').write_text(text)
(OUT/'FAILURES.md').write_text('''# P1 failures retained

- Initial native preparation rejected shared-provider SHA drift before feature
  generation or optimization. Exact original provider privately recovered.
- Original PowerShell launch used a semicolon and ran even after prepare failed.
  Five seed11 children failed missing native arrays; seed23M0 also attempted
  missing preparation, then parent failed writing a log to absent directory.
  run-ledger.json retains five measured failures; setup-failure-preserved/
  failure-audit.json records the sixth plus19unattempted,0optimizer updates.
  Unmeasured sixth duration is conservatively charged90s. Original driver and
  protocol are copied unchanged; r2 readiness/no-overwrite guard added before
  any optimization. No hyperparameter/data change followed test feedback.
- First handoff collector verified copied peer hashes and322oldRoot2 files,
  then rejected broader shared source/executable drift. collection-attempt-1.json
  preserves this failure. Subsequent receipt inventories shared changes and
  verifies exact recovered compiled provider, without rewriting old freeze.
- First preflight copied algebra test failed importing research from relocated
  filename; root5tests had passed. preflight/ retains logs and attempt.json.
  PYTHONPATH was fixed to absolute repo root and fresh preflight-r2 ran all43
  unchanged assertions. Numerical tolerances were not relaxed.
- Revised learned run:25completed,0failed,0censored,0unattempted. Mechanism audit:
  45completed,0updates. Stronger tasks/full native hybrid remain unattempted.
- Anonymous vs attention uncertainty and anonymous learned-QK intervention
  interval crossing zero are retained scientific limitations, not discarded.
''')
(AREA/'integration/p1/HANDOFF.md').write_text('''# Root2 P1 completed handoff

Read research/mamba3h/review/p1/VERDICT.md and verdict.json. Fresh native-frozen
MQARpositive:25cells,5pairedseeds,30fixedupdates,CPU2threads. M0mean32.5%,
anonymousk2memory55.9375%,delta+23.4375pp,t95[6.69184,40.18316]pp;
symbolicADDRESS94.6875%;MC319params30.3125%;MAanonymousk243.4375%.
Anonymous vs MA+12.5pp interval crosses zero: no attention superiority claim.
Full Mamba3H/NC synergy/native end-to-end remain unattempted or undemonstrated.

43independent tests PASS,800numeric/solver checks,5seed exact P0 native replay.
Provider privately restored SHA050f71f..., no shared rebuild. Old322Root2 files
unchanged. All peer P1 snapshots/hash checks under review/p1/handoffs.
Initial six zero-update setup failures and19unattempted preserved separately;
new25cells allcompleted. Charged cumulative336.18304/900seconds includes
conservative101.58506setup charge. No GPU/OXN/default/peer/role edit/commit/push.

Posthoc45zero-update interventions: removing anonymous retrieval drops22.5pp;
restoring initialQK drops13.4375pp but interval includes zero. Addresses gradient
activity vs learned-addressing benefit distinction. No tuning follows test.

Files: manifests/p1a-positive-r2.json, p1a-mechanism.json; integration/p1/
positive_runner.py,native_provider.py,preflight.py,mechanism_audit.py;
review/p1/positive-r2/{run-ledger.json,summary.json,seed-*/};
review/p1/preflight-r2/receipt.json,mechanism/results.json,paired-results.csv,
FAILURES.md,VERDICT.md,verdict.json; separate final P1 SHA inventory.

Optional CPU reproduction in NEW output paths from absolute repo root:
python -B research/mamba3h/integration/p1/test_positive.py
python -B research/mamba3h/integration/p1/native_provider.py smoke --out <new-dir>
python -B research/mamba3h/integration/p1/positive_runner.py prepare --out <new-dir> --corpus research/mamba3h/review/p1/handoffs/benchmarks/p1_positive/datasets
Only if prepare returns success:
python -B research/mamba3h/integration/p1/positive_runner.py run --out <same-new-dir>
python -B research/mamba3h/integration/p1/positive_runner.py summarize --out <same-new-dir>
Further learned runs consume additional budget; none are implied by reproduction.
''')
print(json.dumps(dict(published=True,accuracy_means={r['arm']:r['test_accuracy_mean'] for r in summary},charged_wall_seconds=verdict['cumulative_root2_charged_wall_seconds'])))
