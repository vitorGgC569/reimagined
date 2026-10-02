# Orin P1 positive freshholdout handoff

COMPLETED 2026-10-02. Ownership: new `benchmarks/p1_positive/` only.
Frozen P0 source, corpora, receipts and manifests are unchanged; all79oldpinned
files verify against aggregate
`4802349ce71b88f6caa2bc9661b5b754aa753a4e83ba8879b16a506af4a29b7a`.
No native import, model training, GPU, shared build, OXN edit, commit or push.

## Executed

- 17/17 executable tests PASS; no failures/errors/skips. Standard-library CPU,
  numerical-library thread flags<=2, no background worker. Tests cover all800
  generated labels against independent backward replay, profile counts, global
  input/seed disjointness, deterministic subset generation, forced collision
  rejection/exhaustion, no target/control/future leakage, capacity failure,
  no-read work avoidance and read-only P0 freeze verification.
- 800examples/15JSONL files prepared and audited:64train/32validation/64test per
  model seed11,23,37,53,71. Namespace `P1fresh-MQAR-positive-v1`, fresh data seeds,
  no old P0 data-seed reuse.800unique model-visible inputs/800distinct data seeds.
  Sampling needed800attempts,0rejections, hard bound16attempts/example.
- Independent solver agrees with every label. The exact positive reference
  retrieves800/800 answers from preread observed x alone,3slots/3writes/1read/
  3similarity comparisons per episode;0retrieval work before final QUERY.
  This is fixed-routing component correctness, not a learned/native arm.
- Full corpus/file/split numeric/input SHA audit and P0 preservation checks PASS.
  Receipts preserve test/prepare/audit outcomes under this new directory only.

## Exact schema/API

Fresh MQAR L6: three distinct-entity WRITE events plus two entity-id NOOPs in
shuffled first five positions; final QUERY selects a written entity. Entities4,
values4. Targets are separate length6, non-query-100, final class0..3. Four classes,
no undefined class; do not apply the old P0 INST125->5 convention to this data.

Import from `research.mamba3h.benchmarks.p1_positive`:

```python
generate("mqar", seed, split, count)  # Registered5seeds,64/32/64max.
solve(ep)                           # Offline independent replay labels[L].
model_view(ep)                      # Whitelist, no labels/provenance.
model_numeric(ep)                   # floats[6,8].
encode_numeric(ep, dim=8)            # Same fixed encoding.
```

Each numeric row is onehot(entity4)+onehot(observedvalue4) on WRITE; NOOP/QUERY
have only onehot(entity4) and zero value block. Current query does not contain
the answer. Kind/write/read identity routing is separate. Control contains
kind/entity/write_id/revoke_id/write/read only; no values/targets/logits/future/
default slot address. Root2/Neris converts write/read bool[B], entity long[B],
revoke_id=-1 to revoke_entity long[B]. Root2 adapts x[B,D] and preread projections.

Independent solver is `oracle.py`, with no generator/routing imports. On arbitrary
unbound query it raises `UnboundQuery`, never scans future or inserts an extra
class. Canonical globally bounded generation is implemented in `generation.py`;
subset API calls return byte-equivalent episodes to published full splits.

## Prepared paths for Root2

Relative to `research/mamba3h/benchmarks/p1_positive/`:

```text
datasets/seed-{11,23,37,53,71}/{train,validation,test}.jsonl
manifests/corpus.json
manifests/SHA256.json
receipts/tests.json
receipts/prepare.json
receipts/audit.json
README.md
```

The corpus manifest pins file and model-visible split hashes, exact counts and
sampling budget. The SHA inventory pins new source/data/docs/receipts and its
aggregate is sent asynchronously to Root2 after the final refresh. No old
manifest is rewritten. Root2 freezes learned experiment protocols separately;
zero training jobs or learned results are supplied here, and no pooling with
P0v1/v2 or later learned P1 results is authorized by this handoff.

## Reproduce (project absolute root)

```powershell
python -m research.mamba3h.benchmarks.p1_positive test
python -m research.mamba3h.benchmarks.p1_positive prepare
python -m research.mamba3h.benchmarks.p1_positive audit
python -m research.mamba3h.benchmarks.p1_positive verify-sha
```

Use `p1_positive` explicitly, never parent P0 write commands. This positive
reference proves addressable retrieval is feasible on this bounded visible
onehot task; it does not establish learned performance, M0 comparison, mechanism,
native end-to-end correctness or language/SOTA capability. Logical reference
reserved bytes216 are not Python RSS or a matched native budget claim.
