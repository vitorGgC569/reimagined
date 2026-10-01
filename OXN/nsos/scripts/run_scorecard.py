"""CLI: run the 5-benchmark scorecard against an NSOS pack (or DummyAdapter).

Examples
--------

# Smoke test the eval framework with no real model (quick):
python OXN/nsos/scripts/run_scorecard.py --adapter dummy --quick

# Score a trained NSOS pack:
python OXN/nsos/scripts/run_scorecard.py \\
    --pack /content/drive/MyDrive/nsos_v11/runs/.../final_model.bin \\
    --build-dir /content/reimagined/OXN/nsos/build-colab \\
    --quick

# Multi-seed eval for MMLU stability (1 cheap, 3 honest):
python OXN/nsos/scripts/run_scorecard.py \\
    --pack ... --seeds 1 2 3

The output lands in `<out>/scorecard/<timestamp>/{scorecard.json,scorecard.md,per_benchmark/}`.
Use --copy-to-docs to explicitly publish a complete scorecard to docs/SCORECARD.md.
"""
from __future__ import annotations

# Eager datasets import before torch — Windows segfault workaround.
try:
    import datasets  # noqa: F401
except ImportError:
    pass  # Each benchmark reports its missing dataset dependency explicitly.

import argparse
import math
import shutil
import sys
from pathlib import Path

# Add OXN/nsos/ to sys.path so `eval` package imports work both as a
# module and a script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from eval.orchestrator import run_scorecard  # noqa: E402
from eval.adapters.dummy_adapter import DummyAdapter  # noqa: E402
from eval.benchmarks.base import measurement_errors  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="NSOS scorecard runner.")
    p.add_argument("--adapter", choices=["nsos", "dummy"], default="nsos",
                   help="Which model to score.  'dummy' for framework smoke test.")
    p.add_argument("--pack", type=str, default=None,
                   help="Model-pack directory or raw .bin checkpoint (required for nsos).")
    p.add_argument("--build-dir", type=str, default=None,
                   help="NSOS build directory containing nsos_ext.")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Override tokenizer path.")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--out", type=str,
                   default=str(HERE.parent / "artifacts" / "scorecard"),
                   help="Output root.  A <timestamp>/ subdir gets created.")
    p.add_argument("--quick", action="store_true", default=True,
                   help="Quick caps (default).  --full overrides.")
    p.add_argument("--full", action="store_true",
                   help="Full benchmark sizes (much slower).")
    p.add_argument("--suite", choices=["default", "english", "ptbr", "all"],
                   default="default",
                   help="Which benchmark suite to run. 'default'/'english' = "
                        "the canonical 5; 'ptbr' = Portuguese only; 'all' = "
                        "both. The English suite says nothing about a PT-BR "
                        "model, so use 'ptbr' or 'all' for this product.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these benchmark ids.  e.g. --only hellaswag arc_easy")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Seeds for seed-sensitive benchmarks (MMLU subset).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--copy-to-docs", action="store_true", default=False,
                   help="Explicitly publish a complete scorecard to docs/SCORECARD.md.")
    p.add_argument("--no-copy-to-docs", dest="copy_to_docs", action="store_false")
    args = p.parse_args(argv)
    if args.copy_to_docs and (args.adapter == "dummy" or not args.full):
        p.error("Publishing requires a full model evaluation; smoke/quick runs remain artifacts")

    # Build adapter
    if args.adapter == "dummy":
        adapter = DummyAdapter()
        print(f"[scorecard] using DummyAdapter (framework smoke test)", flush=True)
    else:
        if not args.pack or not args.build_dir:
            print("ERROR: --pack and --build-dir are required for --adapter nsos",
                  file=sys.stderr)
            return 2
        from eval.adapters.nsos_adapter import NsosAdapter
        adapter = NsosAdapter(
            pack_path=args.pack,
            build_dir=args.build_dir,
            device=args.device,
            tokenizer_path=args.tokenizer,
        )
        print(f"[scorecard] loaded NSOS pack: {adapter.capability.notes}", flush=True)

    # Filter benchmarks
    from eval.orchestrator import resolve_suite
    suite = resolve_suite(args.suite)
    if args.only:
        benchmarks = [b for b in suite if b[0] in args.only]
        missing = set(args.only) - {b[0] for b in benchmarks}
        if missing:
            print(f"ERROR: unknown benchmark ids for suite "
                  f"{args.suite!r}: {missing}", file=sys.stderr)
            return 2
    else:
        benchmarks = suite
    print(f"[scorecard] suite={args.suite} "
          f"benchmarks={[b[0] for b in benchmarks]}", flush=True)

    quick = not args.full

    result = run_scorecard(
        adapter,
        out_dir=Path(args.out),
        benchmarks=benchmarks,
        seeds=args.seeds,
        quick=quick,
        verbose=args.verbose,
    )

    failures = scorecard_failures(result)
    # Never replace the authored scorecard with a failed or incomplete run.
    if args.copy_to_docs and not failures:
        src = Path(args.out) / result.timestamp / "scorecard.md"
        dst = HERE.parent / "docs" / "SCORECARD.md"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[scorecard] copied to {dst}", flush=True)
        except Exception as e:
            print(f"[scorecard] WARN: could not copy to docs: {e}", file=sys.stderr)

    # Print short summary at the end
    print()
    print(f"=== SUMMARY ({result.adapter_model_name}) ===")
    for b in result.benchmarks:
        v = b.metrics.get(b.primary_metric, float("nan"))
        print(f"  {b.name:<22} {b.primary_metric:<10} = {v:.4f}  "
              f"[{b.status}, n={b.n_examples}, {b.wall_time_s:.1f}s]")
    print(f"  total wall: {result.wall_time_s:.1f}s")
    failures = scorecard_failures(result)
    for failure in failures:
        print(f"[scorecard:error] {failure}", file=sys.stderr)
    return 1 if failures else 0


def scorecard_failures(result) -> list[str]:
    """An incomplete or non-finite scorecard must not look green to CI."""
    failures = []
    if not result.benchmarks:
        failures.append("no benchmarks were evaluated")
    for benchmark in result.benchmarks:
        errors = measurement_errors(benchmark)
        if errors:
            failures.append(
                f"{benchmark.name}: " + "; ".join(errors)
            )
    return failures


if __name__ == "__main__":
    sys.exit(main())
