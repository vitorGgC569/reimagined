# P0 correctness contracts — 2026-09-20

Scope: evaluation integrity, consumed auxiliary training, correctness-driven
reasoning. No new trained checkpoint, quality gain or above-SOTA claim.

## Evaluation

- PT-BR explicitly maps the scorer's `ppl` to its primary `perplexity` metric.
  Missing fields never become zero. PPL >= 1, positive scored-token count,
  nonnegative finite NLL and `log(PPL) == NLL/tokens` are required.
- Both the orchestrator and CLI use the same metric-domain checks. Empty input,
  invalid context/stride and non-finite/positive log probabilities fail closed.
  Window accounting must score each token after the first exactly once.
- Reports identify synthetic smoke, model quick/full and unverified adapters.
  A model artifact is NOT automatically an attestation that it was trained.
  Smoke/quick runs cannot overwrite the authored product scorecard.
- Evidence includes evaluator-source hash, loaded-artifact/binary hashes,
  context/device/runtime options, benchmark options and actual adapter-input
  transcript hash. PT-BR also hashes the evaluated text. Transcript hashes are
  explicitly NOT full dataset/gold-label hashes. For packs, the strictly checked
  manifest transitively binds weights/config/tokenizer hashes; raw checkpoints
  record the three separately. Retain those artifacts to reproduce a run.
- Historical reports are retained untouched. The August dummy reports with
  primary PT-BR PPL=0 are INVALID measurements, not model-quality evidence.

## Auxiliary training

An enabled auxiliary stack must have `auxiliary_session_adapt_enabled=true`,
`ModelConfig.use_ttt=true`, and at least one actual TTT layer. Memory blending
must be positive when enabled. Reasoning additionally requires an explicit
model policy/verifier. Invalid combinations are rejected at trainer preflight,
before optimizer updates or auxiliary memory work.

This is prompt-only session adaptation, not an extra supervised loss. The
refined prompt state is consumed by `session_adapt`; answer tokens remain the
target of the existing masked CE. The old answer-trunk forward only computed a
diagnostic norm and has been removed. `answer_tokens=0` now truthfully reports
auxiliary compute; `target_state_norm` describes the state actually consumed.
The legacy serialized answer-token-limit field is retained for compatibility.

Old memory/reasoning-only profiles are deliberately rejected rather than
silently wasting compute or silently enabling TTT. Disabling the stack preserves
the existing ordinary training path. Curriculum scheduling now retains valid
positive limits even when that stack is disabled.

## Verified reasoning

`JambaModel::reason` no longer evaluates margin, entropy or scaled RMSNorm
probes, and does not implicitly propose random latent perturbations. It requires
a `ReasoningPolicy` with nonempty versioned `policy_id` and `verifier_id`:

- `propose(state, depth, limit)` returns at most `limit` `ReasoningProposal`
  objects, each with a same-shape/device state and finite nonnegative prior.
- `verify(states)` returns exactly one `ReasoningVerification` per state:
  finite task correctness/utility score in [0,1] and nonempty evidence.
- Callbacks must be task-specific and side-effect-free with respect to the
  model/session. Callback tensors are cloned to protect search-owned state.
  Errors propagate; there is no confidence-based fallback.
- A bounded PUCT tree verifies the root and unique candidates once each.
  `num_simulations` bounds both search iterations and total verified states,
  including root; `max_nodes` provides an additional bound. Time limits are
  cooperative between callbacks (cannot interrupt arbitrary callback code).
- Return the highest directly verified score, including the original root;
  ties preserve the earlier state. Stop at score 1. No trunk projection occurs
  after acceptance. Whole-state hashes plus exact comparison prevent the old
  first-128-float equivalence problem in the verified path.
- The report records baseline/best scores, evidence, identities, verified-state
  count, proposal calls and elapsed time (including verifier work).
- Python exposes policy/proposal/verification classes, registration, report and
  `exact_token_verifier(expected, decode)`. This executable exact-answer helper
  is useful for structured tasks. It is NOT a general-language correctness model.

Policies are runtime callbacks, not checkpoint payloads; explicitly register
them for each loaded model. Normal SDK generation is unchanged and does not
silently activate reasoning. The low-level `MCTSReasoning`/Cauchy API remains a
research/synthetic-test primitive, not the default product reasoning policy.
`forward_thought` remains an experimental repeated trunk, not a trained critic.

The offline arithmetic oracle checks wiring/correctness selection under a
budget; it is not evidence of learned arithmetic or language generalization.
General-language reasoning still requires a validated task verifier and matched
budget comparisons against direct generation and candidate sampling.

## Final validation

Regression sources cover PPL oracles/window boundaries, malformed metrics,
report identity, absent/invalid verifiers, priors versus correctness, budgets,
cycles/tail differences, model integration, auxiliary consumers and Python
callbacks. CPU and HIP validation is run only after implementation is complete.
Results (Release, Clang, Windows; HIP gfx1102 / RX 7600):

| Validation | Result | Log under `artifacts/p0-corrections-20260920/` |
|---|---|---|
| Offline evaluation regressions | 6/6 passed | `evaluation.log` |
| Existing runtime/evaluation contracts | 8/8 passed | `core-runtime.log` |
| Full CPU CTest suite, after final correction | 56/56 passed | `ctest-cpu-final.log` |
| Full HIP CTest suite, serialized | 106/106 passed | `ctest-hip-final.log` |

Python regressions are also included in CTest; these counts are not additive
independent coverage. Both native builds and `git diff --check` succeeded.

The first targeted HIP run exposed host-pointer access through `Tensor::data()`
in auxiliary gather/reasoning/recall/store copies. Those device transfers now
use `raw_data()` with explicit source/destination devices. The new GPU test and
the complete HIP suite pass after correction. The initial failing log is kept
as `ctest-hip-targeted.log`, not presented as a final result.

Validated binary SHA-256 identities:

- CPU `nsos_ext.cp312-win_amd64.pyd`:
  `50b25ff81098aa151b5010ec0c8a168cdde6e1913f96da7768d2211dfcd0ab2f`
- HIP `nsos_ext.cp312-win_amd64.pyd`:
  `3de2df4510db61d9cba7e7170b11609246bcfc02b14087c061c9837d06de1388`
- CPU `test_verified_reasoning.exe`:
  `148ffeedff5c7c064ff1f2f2bd23b5af796a985b38ee5427a33ade67e4f5ab6f`
- HIP `test_verified_reasoning.exe`:
  `1c0981612fcc7766867d40ec2c50e61b1b9ac5a0d52de916dc6b29194b5ee065`

No trained-model benchmark, throughput improvement or general-language verifier
quality is inferred from these contract tests. Linux/NVIDIA were not executed.
