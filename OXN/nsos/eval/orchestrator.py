"""Scorecard orchestrator — runs the 5 benchmarks against an adapter,
optionally multi-seed (item #17), and emits both JSON + markdown summary.

Public entry: `run_scorecard(adapter, ...) -> ScorecardResult`.

Output layout:
  <out_dir>/<timestamp>/
    scorecard.json    — machine-readable
    scorecard.md      — human-readable, ready to drop into SCORECARD.md
    per_benchmark/    — full benchmark dumps (per_problem for HumanEval,
                        per_subject for MMLU, etc.)

Multi-seed behavior:
  When `seeds=[s1, s2, ...]` has > 1 entry, each benchmark that respects
  seeds (currently MMLU subset sampling — others are deterministic given
  the dataset split) is run once per seed.  We report mean + stdev.
  Other benchmarks are run once because their inputs are seed-independent.
"""
from __future__ import annotations

import datetime as dt
import json
import hashlib
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .adapters.base import ModelAdapter
from .benchmarks import hellaswag, arc, mmlu, humaneval, wikitext
from .benchmarks import ptbr_assin2, ptbr_enem, ptbr_perplexity
from .benchmarks.base import BenchmarkResult, measurement_errors


def _eval_code_sha256():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class _RecordedAdapter(ModelAdapter):
    """Fingerprint actual benchmark inputs, without storing private text."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.digest = hashlib.sha256()
        self.calls = 0

    @property
    def capability(self):
        return self.adapter.capability

    def _record(self, operation, values):
        payload = json.dumps([operation, values], sort_keys=True,
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.digest.update(len(payload).to_bytes(8, "little"))
        self.digest.update(payload)
        self.calls += 1

    def tokenize(self, text):
        self._record("tokenize", text)
        return self.adapter.tokenize(text)

    def detokenize(self, ids):
        self._record("detokenize", ids)
        return self.adapter.detokenize(ids)

    def score_tokens(self, prompt_ids, target_ids):
        self._record("score_tokens", [prompt_ids, target_ids])
        return self.adapter.score_tokens(prompt_ids, target_ids)

    def batch_score_tokens(self, items):
        self._record("batch_score_tokens", items)
        return self.adapter.batch_score_tokens(items)

    def generate(self, prompt_text, **kwargs):
        self._record("generate", [prompt_text, kwargs])
        return self.adapter.generate(prompt_text, **kwargs)


# The canonical 5 — order matters because we present them in this order
# in the markdown table.  Each entry: (id, runner, default_kwargs,
# is_seed_sensitive).
DEFAULT_BENCHMARKS: List[tuple] = [
    ("wikitext2_ppl", wikitext.run, {}, False),
    ("hellaswag",     hellaswag.run, {}, False),
    ("arc_easy",      arc.run,       {"config": "ARC-Easy"}, False),
    ("mmlu_stem",     mmlu.run,      {}, True),
    ("humaneval_light", humaneval.run, {}, False),
]

# Portuguese suite.  The canonical 5 are English-only, so they say nothing
# about a PT-BR product: a model can score at chance on all of them and still
# be a competent Brazilian-Portuguese chatbot, or vice versa.  Kept as a
# separate list so an English-targeted run is unchanged.
PTBR_BENCHMARKS: List[tuple] = [
    ("ptbr_perplexity", ptbr_perplexity.run, {}, False),
    ("ptbr_assin2",     ptbr_assin2.run,     {}, False),
    ("ptbr_enem",       ptbr_enem.run,       {}, False),
]

# Everything, for a release scorecard.
ALL_BENCHMARKS: List[tuple] = DEFAULT_BENCHMARKS + PTBR_BENCHMARKS


def resolve_suite(name: str) -> List[tuple]:
    """Map a suite name to its benchmark list."""
    suites = {
        "default": DEFAULT_BENCHMARKS,
        "english": DEFAULT_BENCHMARKS,
        "ptbr": PTBR_BENCHMARKS,
        "all": ALL_BENCHMARKS,
    }
    if name not in suites:
        raise ValueError(
            f"suite desconhecida: {name!r}. Opções: {sorted(suites)}"
        )
    return suites[name]


@dataclass
class ScorecardResult:
    timestamp: str
    adapter_model_name: str
    adapter_params: int
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    seeds_used: List[int] = field(default_factory=list)
    wall_time_s: float = 0.0
    measurement_kind: str = "unverified"
    evidence_identity: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "adapter_model_name": self.adapter_model_name,
            "adapter_params": self.adapter_params,
            "seeds_used": self.seeds_used,
            "wall_time_s": self.wall_time_s,
            "measurement_kind": self.measurement_kind,
            "evidence_identity": self.evidence_identity,
            "benchmarks": [asdict(b) for b in self.benchmarks],
        }


def run_scorecard(
    adapter: ModelAdapter,
    *,
    out_dir: Optional[Path | str] = None,
    benchmarks: Optional[List[tuple]] = None,
    seeds: Optional[List[int]] = None,
    quick: bool = True,
    verbose: bool = False,
    extra_kwargs_per_bench: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ScorecardResult:
    """Run the full scorecard against `adapter`.

    Args:
      adapter: any ModelAdapter (NsosAdapter, DummyAdapter, etc.).
      out_dir: where to land artifacts.  None = no files written (in-memory).
      benchmarks: which to run.  Default = all 5.
      seeds: list of seeds for seed-sensitive benchmarks.  Default = [42].
        Multi-seed = aggregates mean ± stdev for seed-sensitive ones.
      quick: if True (default), use 'quick' n_examples in each benchmark.
        False = full splits, takes much longer.
      verbose: pass through to benchmarks.
      extra_kwargs_per_bench: override defaults per benchmark, e.g.
        {"hellaswag": {"n_examples": 1000}}.
    """
    benchmarks = DEFAULT_BENCHMARKS if benchmarks is None else benchmarks
    seeds = [42] if seeds is None else seeds
    if not benchmarks or not seeds:
        raise ValueError("Scorecards require non-empty benchmarks and seeds")
    extra_kwargs_per_bench = extra_kwargs_per_bench or {}

    quick_caps = {
        "wikitext2_ppl":   {"max_chars": 100_000},
        "hellaswag":       {"n_examples": 200},
        "arc_easy":        {"n_examples": 200},
        "mmlu_stem":       {"n_examples": 80},
        "humaneval_light": {"n_examples": 15},
        "ptbr_perplexity": {"n_documents": 20, "max_length": 100_000},
        "ptbr_assin2":     {"n_examples": 200},
        "ptbr_enem":       {"n_examples": 150},
    }
    full_caps = {
        "wikitext2_ppl":   {"max_chars": None},
        "hellaswag":       {"n_examples": 2000},
        "arc_easy":        {"n_examples": 570},  # validation split size
        "mmlu_stem":       {"n_examples": 300},
        "humaneval_light": {"n_examples": 60},
        "ptbr_perplexity": {"n_documents": 200, "max_length": None},
        "ptbr_assin2":     {"n_examples": 2448},  # validation split size
        "ptbr_enem":       {"n_examples": 1000},
    }
    caps = quick_caps if quick else full_caps

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    identity = adapter.evaluation_identity()
    kind = ("synthetic_smoke" if identity.get("kind") == "synthetic" else
            ("model_quick" if quick else "model_full") if identity.get("kind") == "model"
            else "unverified")
    result = ScorecardResult(
        timestamp=ts,
        adapter_model_name=adapter.capability.model_name,
        adapter_params=adapter.capability.model_params,
        seeds_used=list(seeds),
        measurement_kind=kind,
        evidence_identity={"adapter": identity,
                           "suite": [item[0] for item in benchmarks],
                           "quick": quick, "eval_code_sha256": _eval_code_sha256()},
    )

    t_global = time.time()
    for bench_id, runner, default_kw, seed_sensitive in benchmarks:
        merged_kw = {**default_kw, **caps.get(bench_id, {}),
                     **extra_kwargs_per_bench.get(bench_id, {})}
        if "verbose" in runner.__code__.co_varnames:
            merged_kw["verbose"] = verbose

        seed_list = seeds if seed_sensitive else [seeds[0]]
        seed_results: List[BenchmarkResult] = []
        for s in seed_list:
            local_kw = dict(merged_kw)
            if seed_sensitive and "seed" in runner.__code__.co_varnames:
                local_kw["seed"] = s
            t0 = time.time()
            print(f"[scorecard] running {bench_id} (seed={s if seed_sensitive else 'n/a'})...",
                  flush=True)
            recorded = _RecordedAdapter(adapter)
            res = runner(recorded, **local_kw)
            res.extra["evaluation_inputs"] = {
                "sha256": recorded.digest.hexdigest(), "calls": recorded.calls,
                "options": local_kw, "seed": s,
                "scope": "adapter input transcript; not a dataset/gold-label digest",
            }
            errors = measurement_errors(res)
            if res.status == "ok" and errors:
                res.status = "error"
                res.error = "; ".join(errors)
            elapsed = time.time() - t0
            print(f"[scorecard]   {bench_id}: {res.status} "
                  f"{res.primary_metric}={res.metrics.get(res.primary_metric, float('nan')):.4f} "
                  f"({elapsed:.1f}s, n={res.n_examples})", flush=True)
            seed_results.append(res)

        if len(seed_results) == 1:
            result.benchmarks.append(seed_results[0])
        else:
            # Aggregate multi-seed: mean primary, stdev primary, append all
            primaries = [
                r.metrics.get(r.primary_metric, float("nan"))
                for r in seed_results
                if r.status == "ok" and r.n_examples > 0
                and isinstance(r.metrics.get(r.primary_metric), (int, float))
                and not isinstance(r.metrics.get(r.primary_metric), bool)
                and math.isfinite(r.metrics[r.primary_metric])
            ]
            ok_count = len(primaries)
            agg_metrics = {}
            primary_name = seed_results[0].primary_metric
            if ok_count >= 1:
                agg_metrics[primary_name] = statistics.mean(primaries)
                agg_metrics[f"{primary_name}_stdev"] = (
                    statistics.pstdev(primaries) if ok_count > 1 else 0.0
                )
                agg_metrics["n_seeds_ok"] = float(ok_count)
            agg = BenchmarkResult(
                name=seed_results[0].name,
                primary_metric=seed_results[0].primary_metric,
                metrics=agg_metrics,
                n_examples=seed_results[0].n_examples,
                wall_time_s=sum(r.wall_time_s for r in seed_results),
                status="ok" if ok_count == len(seed_results) else "error",
                notes=(seed_results[0].notes
                       + f" | aggregated across {len(seed_list)} seeds={seed_list}"),
                extra={"per_seed": [asdict(r) for r in seed_results]},
            )
            result.benchmarks.append(agg)

    result.wall_time_s = time.time() - t_global

    if out_dir is not None:
        _persist(result, Path(out_dir))

    return result


def _persist(result: ScorecardResult, out_dir: Path) -> None:
    """Write JSON + markdown to disk."""
    run_dir = out_dir / result.timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    per_bench_dir = run_dir / "per_benchmark"
    per_bench_dir.mkdir(exist_ok=True)

    # Stripped result for top-level scorecard.json
    top_level = result.to_dict()
    # Move per-problem / per-subject blobs into separate files
    for entry, bench_dict in zip(result.benchmarks, top_level["benchmarks"]):
        if entry.extra:
            (per_bench_dir / f"{entry.name}.json").write_text(
                json.dumps(_json_safe(entry.extra), indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            # Strip from top-level (keep summary metrics there)
            bench_dict["extra"] = {"see": f"per_benchmark/{entry.name}.json"}

    (run_dir / "scorecard.json").write_text(
        json.dumps(_json_safe(top_level), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    (run_dir / "scorecard.md").write_text(
        _to_markdown(result), encoding="utf-8",
    )
    print(f"[scorecard] artifacts -> {run_dir}", flush=True)


def _json_safe(value):
    # Preserve failed measurements as null, never invented zeros or invalid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _to_markdown(result: ScorecardResult) -> str:
    lines = [
        f"# NSOS Scorecard — {result.adapter_model_name}",
        "",
        f"- **Generated:** {result.timestamp} UTC",
        f"- **Measurement kind:** {result.measurement_kind} (not a training attestation)",
        f"- **Model params:** {result.adapter_params:,}" if result.adapter_params else "- **Model params:** unknown",
        f"- **Seeds:** {result.seeds_used}",
        f"- **Total wall time:** {result.wall_time_s:.1f} s",
        "",
        "## Results",
        "",
        "| Benchmark | Primary metric | Value | n | Wall (s) | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in result.benchmarks:
        primary = b.metrics.get(b.primary_metric, float("nan"))
        stdev_key = f"{b.primary_metric}_stdev"
        stdev = b.metrics.get(stdev_key)
        value_cell = f"**{primary:.4f}**"
        if stdev is not None and stdev > 0:
            value_cell += f" ± {stdev:.4f}"
        notes = b.notes.replace("|", "\\|")[:80]
        lines.append(
            f"| {b.name} | {b.primary_metric} | {value_cell} | {b.n_examples} | "
            f"{b.wall_time_s:.1f} | {b.status} | {notes} |"
        )
    lines.append("")
    lines.append("## Reference benchmarks (for context)")
    lines.append("")
    lines.append("| Model | HellaSwag | ARC-Easy | MMLU-STEM | HumanEval | WikiText2 PPL |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| Random | 25.0% | 25.0% | 25.0% | 0% | — |")
    lines.append("| GPT-2 small (124M) | 31.1% | ~44% | ~25% | ~0% | 29.4 |")
    lines.append("| TinyLlama 1.1B | 59.2% | 55.5% | ~26% | ~10% | ~12 |")
    lines.append("| Llama-3-8B | 79.0% | 92% | ~50% | ~33% | ~6.5 |")
    lines.append("")
    lines.append("## Per-benchmark dumps")
    lines.append("")
    for b in result.benchmarks:
        if b.extra:
            lines.append(f"- `{b.name}` → `per_benchmark/{b.name}.json`")
    return "\n".join(lines) + "\n"
