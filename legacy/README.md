# `legacy/` — Frozen Historical Artifacts

Files under this tree are **frozen historical artifacts**. They are:

- **Not built** — no entry in `CMakeLists.txt`, no compileall in CI.
- **Not tested** — no CTest target, no `pytest` invocation.
- **Not supported** — no product claim depends on them; they may use removed APIs, broken imports, or stale assumptions.
- **Not referenced from `PRODUCT.md`** — only from `INCUBATION.md` historical notes if useful.

## Why keep them at all?

This tree exists to preserve research and prototyping work without leaving it in the active repository. The MVP cleanup removed mocks and demos from the root tree; archiving (instead of pure deletion) lets a future engineer recover patterns or context without spelunking through `git log`.

## Rules

1. **`legacy/` is append-only.** New PRs may add files (more archiving). They must not modify existing files. To "fix" something in legacy, copy it into the active product tree, fix it there, and treat the legacy copy as immutable history.
2. **`legacy/` is not on any execution path.** No script, build, test, doc, or workflow may depend on a file under `legacy/`.
3. **Whole-tree removal at major-version boundaries only.** If `legacy/` becomes burdensome, it can be deleted wholesale at a `vN.0` release — not piece by piece.

## Structure

- `legacy/scripts/` — Python scripts removed from repo root (training prototypes, debug/verify, diagnostics).
- `legacy/docs/` — historical reports, audits, and superseded roadmaps.
- `legacy/src/` — C++ source files superseded by `_v2` rewrites or refactored away.
- `legacy/tests/cpp/` — C++ tests not wired into CTest.

Each subdirectory may carry its own short `README.md` describing what was archived and when.
