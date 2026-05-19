"""Benchmarks — each isolates one published evaluation.

Naming convention: `bench_<dataset>.run(adapter, **opts) -> BenchmarkResult`.
All benchmarks expose a `.run` function so the orchestrator can dispatch
uniformly.
"""
from .base import Benchmark, BenchmarkResult
from . import hellaswag, arc, mmlu, humaneval, wikitext

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "hellaswag",
    "arc",
    "mmlu",
    "humaneval",
    "wikitext",
]
