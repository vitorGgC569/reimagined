"""HellaSwag — commonsense sentence-completion benchmark.

Dataset: `Rowan/hellaswag` validation split (≈ 10K items).  Each item has
a context (concatenation of `activity_label` + `ctx_a` + `ctx_b`) plus 4
candidate endings.  Score each ending and pick argmax.

Official metric: accuracy with CHAR-LENGTH-NORMALIZED logprob.  This
reduces the bias toward shorter options that raw logprob exhibits.

Reference numbers (validation set, char-length normalized):
  Random:      25.0%
  GPT-2-small: 31.1%
  GPT-2-large: 40.5%
  Llama-3-8B:  79.0%

A 40-80M parameter NSOS model on a few-billion-token budget would do
well to land 30-35%.  Below random (< 25%) means something is wrong
with tokenization or scoring direction.
"""
from __future__ import annotations

import time
from typing import List, Optional

from ..adapters.base import ModelAdapter
from ..scoring.multiple_choice import score_multiple_choice, mc_accuracy
from .base import BenchmarkResult, make_skipped, make_error


PRIMARY_METRIC = "accuracy"


def _load_dataset(split: str, n_examples: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Need `datasets` to load HellaSwag.  pip install datasets."
        ) from e
    ds = load_dataset("Rowan/hellaswag", split=split)
    if n_examples is not None and len(ds) > n_examples:
        ds = ds.select(range(n_examples))
    return ds


def _preprocess(text: str) -> str:
    """lm-eval-harness HellaSwag preprocessing, reproduced exactly.

    The raw dataset carries WikiHow artifacts like ``[title]`` and
    ``[step]`` inside contexts and endings; published numbers strip
    them.  Skipping this step changes accuracy by a few points and
    breaks comparability.
    """
    import re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _format_context(item: dict) -> str:
    """Standard HellaSwag context: activity label + sentence opener.

    Matches lm-eval-harness: ``ctx = ctx_a + " " + ctx_b.capitalize()``
    then ``preprocess(activity_label + ": " + ctx)``.
    """
    label = item.get("activity_label", "").strip()
    ctx_a = item.get("ctx_a", "").strip()
    ctx_b = item.get("ctx_b", "").strip().capitalize()
    ctx = f"{ctx_a} {ctx_b}".strip()
    if label:
        return _preprocess(f"{label}: {ctx}")
    return _preprocess(ctx)


def run(adapter: ModelAdapter, *, n_examples: Optional[int] = 500,
        split: str = "validation", normalization: str = "char_length",
        verbose: bool = False) -> BenchmarkResult:
    """HellaSwag accuracy on the validation split.

    Args:
      n_examples: cap on items.  None = full split.  Default 500 keeps
        the smoke run < 5 min on a small model; full = 10K items.
      normalization: 'char_length' (HellaSwag official), 'token_length',
        or 'none'.
      verbose: print per-example trace.
    """
    if not adapter.capability.can_score_tokens:
        return make_skipped("hellaswag", PRIMARY_METRIC,
                              "adapter has no score_tokens capability")
    try:
        ds = _load_dataset(split, n_examples)
    except Exception as e:
        return make_error("hellaswag", PRIMARY_METRIC, e)

    t0 = time.time()
    predictions: List[int] = []
    gold: List[int] = []
    for i, item in enumerate(ds):
        context = _format_context(item)
        endings = [_preprocess(e) for e in item["endings"]]
        try:
            scores = score_multiple_choice(adapter, context, endings,
                                            normalization=normalization)
            pred = int(max(range(len(scores)), key=lambda k: scores[k]))
            predictions.append(pred)
            gold.append(int(item["label"]))
            if verbose and i < 5:
                print(f"  [{i}] gold={item['label']} pred={pred} "
                      f"scores={[f'{s:.2f}' for s in scores]}")
        except Exception as e:
            return make_error("hellaswag", PRIMARY_METRIC, e)

    acc = mc_accuracy(predictions, gold)
    wall = time.time() - t0
    return BenchmarkResult(
        name="hellaswag",
        primary_metric=PRIMARY_METRIC,
        metrics={
            "accuracy": acc,
            "examples_per_s": (len(predictions) / wall) if wall > 0 else 0.0,
        },
        n_examples=len(predictions),
        wall_time_s=wall,
        notes=(f"split={split}, n={n_examples or 'full'}, "
               f"normalization={normalization}.  Random = 25%."),
    )
