"""NSOS eval suite — standard benchmarks producing publishable scorecard numbers.

Layout:
  adapters/       — ModelAdapter abstraction over NSOS pack, dummy (for testing),
                    and optional HuggingFace baseline.
  benchmarks/     — One module per benchmark (HellaSwag, ARC-Easy, MMLU,
                    HumanEval-light, WikiText2-PPL).  Each is independent.
  scoring/        — Pure-function helpers: log-prob of token sequence,
                    multiple-choice scoring, perplexity computation.
  orchestrator.py — Single entry point: run a scorecard against a pack.
  tests/          — Smoke tests with DummyAdapter so the framework verifies
                    even before a trained NSOS pack exists.
"""

from .adapters.base import ModelAdapter, AdapterCapability
from .adapters.dummy_adapter import DummyAdapter
from .benchmarks.base import Benchmark, BenchmarkResult
from .orchestrator import run_scorecard, ScorecardResult

__all__ = [
    "ModelAdapter",
    "AdapterCapability",
    "DummyAdapter",
    "Benchmark",
    "BenchmarkResult",
    "run_scorecard",
    "ScorecardResult",
]
