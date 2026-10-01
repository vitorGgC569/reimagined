"""Benchmark base type — uniform return so the orchestrator can summarize."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Optional, Protocol


@dataclass
class BenchmarkResult:
    """Output every benchmark must produce."""
    name: str                              # short id, e.g. "hellaswag"
    primary_metric: str                    # which key in `metrics` is THE number
    metrics: Dict[str, float]              # all measured numbers
    n_examples: int                        # how many test items were scored
    wall_time_s: float                     # seconds spent
    notes: str = ""                        # human-readable caveats
    extra: Dict[str, Any] = field(default_factory=dict)
    # If a benchmark was skipped (missing dataset, unsupported capability,
    # etc.), it returns BenchmarkResult with status='skipped' and the reason.
    status: str = "ok"                     # 'ok' / 'skipped' / 'partial' / 'error'
    error: str = ""


class Benchmark(Protocol):
    """A benchmark is just a module exposing `run(adapter, **opts)`."""
    def run(self, adapter, **opts) -> BenchmarkResult: ...


def make_skipped(name: str, primary_metric: str, reason: str) -> BenchmarkResult:
    """Convenience for benchmarks that bail out."""
    return BenchmarkResult(
        name=name, primary_metric=primary_metric, metrics={},
        n_examples=0, wall_time_s=0.0,
        status="skipped", error=reason,
        notes=f"skipped: {reason}",
    )


def make_error(name: str, primary_metric: str, exc: BaseException) -> BenchmarkResult:
    return BenchmarkResult(
        name=name, primary_metric=primary_metric, metrics={},
        n_examples=0, wall_time_s=0.0,
        status="error", error=f"{type(exc).__name__}: {exc}",
        notes=f"error: {type(exc).__name__}",
    )


def measurement_errors(result: BenchmarkResult) -> list[str]:
    """The same domain contract is used by orchestration and release gates."""
    errors = []
    value = result.metrics.get(result.primary_metric)
    if result.status != "ok":
        errors.append(f"status={result.status}")
    if (isinstance(result.n_examples, bool) or
            not isinstance(result.n_examples, int) or result.n_examples <= 0):
        errors.append("no evaluated examples")
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value)):
        errors.append("missing/non-finite primary metric")
    elif result.primary_metric in {"ppl", "perplexity"}:
        from ..scoring.perplexity import validate_perplexity_stats
        try:
            validate_perplexity_stats({**result.metrics, "ppl": value})
        except ValueError as exc:
            errors.append(str(exc))
    elif (result.primary_metric in {"accuracy", "exact_match", "pass@1"} and
          not 0 <= value <= 1):
        errors.append("accuracy must be in [0, 1]")
    return errors
