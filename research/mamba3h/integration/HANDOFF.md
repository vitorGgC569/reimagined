# Root2 completed P0 handoff

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
