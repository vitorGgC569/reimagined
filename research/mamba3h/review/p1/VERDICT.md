# Root2 P1 independent research verdict

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
| M0 | 32.5000% | 36 | 36 | 0 |
| MC_capacity | 30.3125% | 319 | 319 | 0 |
| M2_symbolic_address | 94.6875% | 319 | 164 | 630 |
| M2_anonymous_k2 | 55.9375% | 319 | 292 | 630 |
| MA_anonymous_k2 | 43.4375% | 319 | 292 | 630 |

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
