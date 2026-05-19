"""WikiText-2 raw perplexity.

Standard small-model LM benchmark.  Concatenates all eval split text,
slides a window over it, computes token-level PPL.

Reference (test split, BPE-token PPL, our methodology):
  GPT-2 small (124M):  29.4
  GPT-2 large (774M):  19.9
  Llama-3-8B:           ~6.5
  TinyLlama 1.1B:       ~12

Char-level PPL is on a different scale (smaller vocab) — NSOS uses
BPE so numbers should be roughly comparable to the above when measured
on the BPE tokenizer.  Our `local_train_8h.py` used char-level
intentionally for the 1050 Ti baseline; this benchmark uses the
adapter's native tokenizer.
"""
from __future__ import annotations

import time
from typing import Optional

from ..adapters.base import ModelAdapter
from ..scoring.perplexity import sliding_window_perplexity
from .base import BenchmarkResult, make_skipped, make_error


PRIMARY_METRIC = "ppl"


def _load_text(split: str, max_chars: Optional[int]) -> str:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Need `datasets`.  pip install datasets.") from e
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    lines = [line for line in ds["text"] if line.strip()]
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def run(adapter: ModelAdapter, *, split: str = "test",
        max_chars: Optional[int] = 300_000, stride: Optional[int] = None,
        verbose: bool = False) -> BenchmarkResult:
    """WikiText-2 sliding-window perplexity.

    Args:
      split: 'test' (default, what published numbers report) or
        'validation'.
      max_chars: cap text size (saves time).  Default 300K chars covers
        a substantial portion of the test split.  None = full split.
      stride: sliding window step.  Default = max_seq_len // 2.
    """
    if not adapter.capability.can_score_tokens:
        return make_skipped("wikitext2_ppl", PRIMARY_METRIC, "no score_tokens")
    try:
        text = _load_text(split, max_chars)
    except Exception as e:
        return make_error("wikitext2_ppl", PRIMARY_METRIC, e)
    if not text:
        return make_skipped("wikitext2_ppl", PRIMARY_METRIC, "empty text")

    t0 = time.time()
    try:
        result = sliding_window_perplexity(adapter, text, stride=stride)
    except Exception as e:
        return make_error("wikitext2_ppl", PRIMARY_METRIC, e)
    wall = time.time() - t0

    if verbose:
        print(f"  ppl={result['ppl']:.2f}  n_tokens={result['n_tokens']:,} "
              f"n_windows={result['n_windows']}  wall={wall:.1f}s")

    return BenchmarkResult(
        name="wikitext2_ppl",
        primary_metric=PRIMARY_METRIC,
        metrics={
            "ppl": result["ppl"],
            "nll_per_token": result["nll_sum"] / max(result["n_tokens"], 1),
            "n_tokens": float(result["n_tokens"]),
            "n_windows": float(result["n_windows"]),
            "tokens_per_s": (result["n_tokens"] / wall) if wall > 0 else 0.0,
        },
        n_examples=int(result["n_tokens"]),
        wall_time_s=wall,
        notes=(f"split={split}, max_chars={max_chars}, stride={stride or 'auto'}.  "
               f"GPT-2 small ≈ 29 PPL (BPE), Llama-3 ≈ 6.5."),
    )
