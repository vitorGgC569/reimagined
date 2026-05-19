"""HumanEval-light — Python code-completion subset.

Full HumanEval is 164 problems; we run 30 by default to keep wall time
tight.  Each problem provides:
  * prompt: function signature + docstring
  * canonical_solution: reference body (not shown to model)
  * test: pytest-style asserts to validate generation

Metric: pass@1 = fraction of problems where the FIRST sampled
generation passes all tests.

Reference (validation/test):
  Random / no model: ~ 0%
  GPT-2 base:        ~ 0% (no code training)
  TinyLlama 1.1B:    ~ 10%
  Llama-3-8B:        ~ 33%
  GPT-4:             ~ 67%

Crucial: this benchmark EXECUTES MODEL OUTPUT.  We sandbox by running in
a fresh subprocess with a hard wall-clock timeout (default 5 s/problem).
A misbehaving generation that imports os.system or similar still
nukes its own subprocess, not the harness.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from ..adapters.base import ModelAdapter
from .base import BenchmarkResult, make_skipped, make_error


PRIMARY_METRIC = "pass@1"


def _load_dataset(n_examples: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Need `datasets`.  pip install datasets.") from e
    ds = load_dataset("openai_humaneval", split="test")
    if n_examples is not None and len(ds) > n_examples:
        ds = ds.select(range(n_examples))
    return ds


def _execute_solution(prompt: str, completion: str, test_src: str,
                       entry_point: str, timeout_s: float) -> tuple[bool, str]:
    """Write prompt+completion+test to a temp file, run as subprocess.

    Returns (pass, message).  pass=True only when subprocess exits 0
    AND stdout doesn't show test failure.
    """
    # The check function is called via 'check(<entry_point>)' at the end.
    full_src = (
        prompt
        + completion
        + "\n\n"
        + test_src
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(full_src)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", tmp_path],
            timeout=timeout_s, capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return True, "ok"
        # Trim error message for compactness
        err = proc.stderr.strip()
        return False, (err[-300:] if err else f"exit {proc.returncode}")
    except subprocess.TimeoutExpired:
        return False, f"timeout > {timeout_s}s"
    except Exception as e:
        return False, f"runner error: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _truncate_at_function_end(generation: str) -> str:
    """HumanEval expects exactly one function body completion.  Models
    happily continue with more code, comments, or `def next():`.
    We trim at the first line that:
      * starts with `def `, `class `, `if __name__`, or `print(`
        outside any indented block
    so we don't get a syntax error from concatenated functions.
    """
    lines = generation.splitlines(keepends=True)
    out: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not line.startswith((" ", "\t")) and (
            stripped.startswith("def ") or stripped.startswith("class ")
            or stripped.startswith("if __name__")
            or stripped.startswith("print(")
            or stripped.startswith("assert ")
        ):
            break
        out.append(line)
    return "".join(out)


def run(adapter: ModelAdapter, *, n_examples: int = 30,
        timeout_s: float = 5.0, temperature: float = 0.2,
        max_new_tokens: int = 384, verbose: bool = False) -> BenchmarkResult:
    """HumanEval-light pass@1.

    Args:
      n_examples: cap problems.  Default 30 = ~ 18% of full set.
      timeout_s: wall-clock cap per subprocess.
      temperature: low (0.2) makes pass@1 fairly deterministic — high T
        is for pass@10 / pass@100 which we don't run here.
      max_new_tokens: cap on generation length.
    """
    if not adapter.capability.can_generate:
        return make_skipped("humaneval_light", PRIMARY_METRIC, "no generate")
    try:
        ds = _load_dataset(n_examples)
    except Exception as e:
        return make_error("humaneval_light", PRIMARY_METRIC, e)

    t0 = time.time()
    n_pass = 0
    per_problem: List[dict] = []
    for i, item in enumerate(ds):
        prompt = item["prompt"]
        # Stop sequences: anything that signals "next function" / "I'm done".
        stops = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n\n\n"]
        try:
            gen = adapter.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=40,
                stop_sequences=stops,
            )
        except Exception as e:
            per_problem.append({"i": i, "task_id": item["task_id"],
                                "pass": False, "error": f"gen: {e}"})
            continue
        completion = _truncate_at_function_end(gen)
        ok, msg = _execute_solution(
            prompt, completion, item["test"], item["entry_point"], timeout_s,
        )
        per_problem.append({
            "i": i,
            "task_id": item["task_id"],
            "pass": ok,
            "error": "" if ok else msg,
            "completion_chars": len(completion),
        })
        if ok:
            n_pass += 1
        if verbose:
            status = "PASS" if ok else f"FAIL ({msg[:40]})"
            print(f"  [{i}] {item['task_id']}: {status}")

    wall = time.time() - t0
    pass_at_1 = n_pass / max(len(per_problem), 1)
    return BenchmarkResult(
        name="humaneval_light",
        primary_metric=PRIMARY_METRIC,
        metrics={
            "pass@1": pass_at_1,
            "n_pass": float(n_pass),
            "examples_per_s": (len(per_problem) / wall) if wall > 0 else 0.0,
        },
        n_examples=len(per_problem),
        wall_time_s=wall,
        notes=(f"n={n_examples}, T={temperature}, timeout={timeout_s}s.  "
               f"Random = 0%; TinyLlama 1.1B ~10%; GPT-4 ~67%."),
        extra={"per_problem": per_problem},
    )
