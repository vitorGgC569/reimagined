# Root2 P1 completed handoff

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
