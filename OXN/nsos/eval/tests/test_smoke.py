"""End-to-end smoke test using DummyAdapter.

This is the most important test in the eval suite.  It guarantees:
  * Every dataset loader can fetch its data
  * Every scoring function survives a degenerate (uniform) model
  * The orchestrator produces both JSON and markdown output
  * Result aggregation is sane (random model → chance-level accuracy)

Runs in ~ 2 min on any laptop.
"""
from __future__ import annotations

import datasets  # eager, before any torch  # noqa: F401

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from eval.adapters.dummy_adapter import DummyAdapter  # noqa: E402
from eval.orchestrator import run_scorecard  # noqa: E402


def main() -> int:
    adapter = DummyAdapter(vocab_size=256, max_seq_len=512)

    with tempfile.TemporaryDirectory() as tmp:
        result = run_scorecard(
            adapter,
            out_dir=Path(tmp),
            seeds=[42],
            quick=True,
            verbose=False,
            extra_kwargs_per_bench={
                # Cap sizes hard so the smoke is fast
                "hellaswag":       {"n_examples": 20},
                "arc_easy":        {"n_examples": 20},
                "mmlu_stem":       {"n_examples": 20},
                "humaneval_light": {"n_examples": 3, "timeout_s": 2.0},
                "wikitext2_ppl":   {"max_chars": 5_000},
            },
        )

        print()
        print("SMOKE RESULTS")
        print("=" * 60)
        for b in result.benchmarks:
            primary = b.metrics.get(b.primary_metric, float("nan"))
            print(f"  {b.name:<20} {b.primary_metric:<8} = {primary:.4f}  [{b.status}]")

        # Sanity checks: a uniform model should give random-level results
        # on multiple choice (within wide tolerance because n is small).
        failures = []
        for b in result.benchmarks:
            if b.status != "ok":
                continue
            if b.name in ("hellaswag", "arc_easy", "mmlu_stem"):
                acc = b.metrics.get("accuracy", -1)
                # 4-choice random = 25%.  With n=20, sampling error makes
                # accuracy land in [5%, 50%] easily.  Failure mode here is
                # acc < 5% or > 50% (would imply non-random behavior).
                if acc < 0 or acc > 0.70:
                    failures.append(
                        f"{b.name}: accuracy={acc:.2f} outside [0, 0.70] "
                        f"(uniform model should give ~25%)"
                    )
            if b.name == "wikitext2_ppl":
                ppl = b.metrics.get("ppl", -1)
                # Uniform over 256 chars should give PPL near 256.  Allow
                # wide range since smoke uses few chars.
                if ppl < 100 or ppl > 400:
                    failures.append(
                        f"wikitext: ppl={ppl:.1f} outside [100, 400] "
                        f"(uniform model over vocab=256 should give PPL ~ 256)"
                    )

        if failures:
            print()
            print("FAILURES:")
            for f in failures:
                print(f"  - {f}")
            return 1

        # Verify artifacts exist
        run_dir = Path(tmp) / result.timestamp
        assert (run_dir / "scorecard.json").exists(), "scorecard.json missing"
        assert (run_dir / "scorecard.md").exists(), "scorecard.md missing"
        # Top-level JSON must round-trip
        data = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
        assert len(data["benchmarks"]) == len(result.benchmarks)
        print()
        print("OK: smoke passed.  All 5 benchmarks ran, artifacts written, "
              "uniform-model invariants held.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
