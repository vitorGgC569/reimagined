"""Benchmark base type — uniform return so the orchestrator can summarize."""
from __future__ import annotations

from dataclasses import dataclass, field
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
