# Optional Mamba3H research — paused

This tree is research/incubation, outside the supported NSOS product. The user
paused these studies to prioritize the Mamba-3 GPU backward. Publishing the
existing sources and reports does not resume training or promote an architecture.

Start with `integration/CONTRACT.md`, `review/VERDICT.md` when present, and the
completed P1 report [`review/p1/VERDICT.md`](review/p1/VERDICT.md). The P0/P1
protocols, manifests, paired seeds, datasets, failures and subsequent corrections
are retained rather than replacing an earlier outcome with a later experiment.

P1 uses actual frozen native CPU Mamba-3 features with small differentiable
adapters; it is not native end-to-end Mamba3H. Five paired seeds completed25
fresh MQAR cells: M0 32.5%, parameter control30.3125%, anonymous memory55.9375%,
anonymous sparse attention43.4375%, symbolic ADDRESS94.6875%. Anonymous memory
improved over M0 and the parameter control in this experiment; its comparison
with sparse attention is uncertain. Symbolic addressing, unused gates, byte
reservation and posthoc interventions are explicitly described in the report.
Full NC/memory complementarity, learned WRITE/RETAIN/READ, language quality and
SOTA superiority remain unproved. Earlier null results remain intact.

## Reproduction boundary

The exact frozen CPU provider is SHA-256
`050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f`.
The native binary, feature arrays, checkpoint arrays and worker console logs
remain local artifacts and are excluded by `.gitignore`. Some archived manifests
refer to these omitted files and to the original Windows workspace paths.
Manifests are historical provenance, not a promise of self-contained reproduction
from a Git checkout. Obtain the matching provider/data artifacts or perform a
new explicitly labeled source build with fresh parity gates before replay.
Never substitute a newer binary and silently repin the archived experiment.

`test/Circuit.py` and `tests/texteMedicaoGeral.py` are separate experimental
scripts shipped as source. Publication and syntax checks do not assert that
their long training plans were executed, validate physical circuit/GPU design,
or certify a product model. No new research grid is authorized by this checkpoint.
