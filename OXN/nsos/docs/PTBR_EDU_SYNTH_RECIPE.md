# PT-BR EDU + Synth + complete-answer SFT

**Current status: blocked by the manual synthetic-data quality review. No model
training was started.** The automatic pipeline was stopped during educational
data collection; both partial corpora and their logs were retained. Do not rerun
it by weakening the review gate. See
`scripts/data_recipes/ptbr_edu_synth_review.json` for sample IDs and findings.

User-requested run, 2026-09-20. Separate corpus, tokenizer, weights and checkpoints.
Entrypoint: `scripts/train_ptbr_edu_synth.py`. Detached Windows launcher:
`powershell -File scripts/start_ptbr_edu_synth.ps1 -Action run`.

## Scope

- Keep the existing 16-layer, d_model=768, ~71M native Mamba-only pilot and FP32
  GEMMs. No PyTorch, teacher API, paid service, implicit MCTS or architecture change.
- Fresh 80M-token base: 70% GigaVerbo-v2 educational, 30% GigaVerbo-v2 Synth.
  This ratio is a hypothesis, not a demonstrated optimum. Source-family cap:
  60% of the educational slice. Actual packed-token quotas, not upstream counts.
- 12M conversational continuation tokens, followed by 4M assistant target tokens
  for two SFT epochs. SFT task mix: general 60%, math 15%, retrieval 15%,
  structured 10%. Never truncate a prompt or answer and fabricate an EOS.
- Educational admission: edu_int_score >=4, toxic_int_score <=1, existing
  commercial-strict allowlist only. Do not relax conditional/unknown licenses
  to fill the budget. Synth and SFT declare Apache-2.0; original-source rights
  still need review before commercialization.
- No legacy shards/checkpoints. Headroom is collected for dedup and tokenization;
  underfilled task/source buckets block training rather than silently rebalance.

## Audit and reproducibility

Pinned source revisions, stable source/content identities, explicit recipe hash,
all-language langid 1.1.6 (Portuguese probability >=0.90), boilerplate/contact
filters, finite SFT score >=4, conversation-role checks, repetition checks and
cross-phase lexical dedup. Holdout anchors take precedence; identical user prompts
share the same train/eval split. Sampling previews and all rejected counts are
written alongside the SQLite corpus. Source revision and generator/seed metadata
must remain traceable. The native trainer hashes binary, tokenizer, corpus and
packed artifacts and checkpoints optimizer/RNG/progress for exact resumption.

`quality_audit.json` certifies these structural checks, NOT factual correctness
of synthetic text. Near-dedup is bounded lexical matching, not a guarantee of
semantic benchmark decontamination. Language ID does not reliably distinguish
Brazilian Portuguese from European Portuguese. This is a quality pilot, not a
claim of general reasoning, SOTA or production readiness.

Runtime state: `artifacts/ptbr_edu_synth_20260920_v2/job_status.json`.
Training logs: `job-*.out.log` / `job-*.err.log` in the same directory.
Checkpoints: `runs/main/checkpoints`. Old runs remain untouched.

Install the isolated preparation dependency when absent:
`python -m pip install --target artifacts/ptbr_edu_synth_20260920_v2/dependencies langid==1.1.6 --no-deps`.

The first collection (`ptbr_edu_synth_20260920`) was stopped before training:
manual sample inspection found worksheet answers with factual errors, truncated
pages and residual footer/license notices despite EDU score 4. Its partial corpus
is retained as diagnostic evidence. Version 2 rejects worksheet headers, trailing
ellipses and explicit NC/SA original-content notices; additional navigation patterns
are removed. EDU score predicates are pushed into Arrow; ingest counters for EDU
count prefiltered rows, not all upstream documents. The audit still does not imply
that all accepted facts are correct.

Regression coverage: `tests/test_ptbr_edu_synth.py`, plus the existing preparation,
checkpoint, evaluation and native HIP contracts. Validation runs after code
implementation, before starting the long training job.

## Validation and subsequent quality finding

- Full CPU suite: 57/57 passed (`validation-cpu-full.log`).
- Full HIP suite: 107/107 passed (`validation-hip-full.log`).
- Dataset regression tests rerun after each final data-filter correction.
- Initial concurrent CPU/HIP validation had transient native test failures;
  the sequential HIP rerun passed. No native code was changed for this recipe.
- A separate inspection of four Synth candidates passing the v2 filters found
  serious flaws. This was not a representative estimate of dataset-wide error
  rate, but it invalidates the claim that the proposed selection is already
  quality-approved. Known failures include an internal formula/prose contradiction,
  mixed-language degradation, a context-inconsistent case summary and web footers.
- Structural tests passing does not override a failed data-quality review.
  Launcher and Python entrypoint now both refuse a run until the source review
  is resolved. Collected educational candidates are not final packed tokens.

Next decision: a smaller independently verifiable synthetic subset versus an
additional semantic-review stage. Neither was silently substituted for the
user-requested recipe, and no paid review service was invoked.
