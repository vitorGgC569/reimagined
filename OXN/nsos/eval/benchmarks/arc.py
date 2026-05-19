"""ARC-Easy — AI2 Reasoning Challenge, Easy split.

Dataset: `allenai/ai2_arc` config `ARC-Easy`, validation split (~570 items).
Multiple-choice science QA at elementary-school level.  Each question
has 3-5 choices; we score each by token-length-normalized logprob.

Reference numbers (validation):
  Random:        25.0% (4-choice) or 20% (5-choice)
  GPT-2-small:   ~44%
  Llama-3-8B:    ~92%

Sub-30% means scoring is broken; 40%+ is competitive for a small model.
"""
from __future__ import annotations

import time
from typing import List, Optional

from ..adapters.base import ModelAdapter
from ..scoring.multiple_choice import score_multiple_choice, mc_accuracy
from .base import BenchmarkResult, make_skipped, make_error


PRIMARY_METRIC = "accuracy"


def _load_dataset(split: str, n_examples: Optional[int], config: str = "ARC-Easy"):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Need `datasets`.  pip install datasets.") from e
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    if n_examples is not None and len(ds) > n_examples:
        ds = ds.select(range(n_examples))
    return ds


def _label_to_index(label: str, choice_labels: List[str]) -> int:
    """Some items use letters (A-E), some use digits (1-5).  Map to 0-based."""
    s = str(label).strip()
    if s in choice_labels:
        return choice_labels.index(s)
    # Common fallback: A/B/C/D → 0/1/2/3
    if s.isalpha() and len(s) == 1:
        return ord(s.upper()) - ord("A")
    if s.isdigit():
        return int(s) - 1
    return -1


def run(adapter: ModelAdapter, *, n_examples: Optional[int] = 500,
        split: str = "validation", config: str = "ARC-Easy",
        normalization: str = "token_length",
        verbose: bool = False) -> BenchmarkResult:
    """ARC-{Easy,Challenge} accuracy.

    Args:
      n_examples: None for full.  Default 500.
      split: 'validation' (default) or 'test'.
      config: 'ARC-Easy' or 'ARC-Challenge'.
      normalization: 'token_length' (lm-eval-harness default for ARC).
    """
    if not adapter.capability.can_score_tokens:
        return make_skipped("arc_easy", PRIMARY_METRIC, "no score_tokens")
    try:
        ds = _load_dataset(split, n_examples, config=config)
    except Exception as e:
        return make_error("arc_easy", PRIMARY_METRIC, e)

    t0 = time.time()
    predictions: List[int] = []
    gold: List[int] = []
    invalid = 0
    for i, item in enumerate(ds):
        # ARC uses item['question'] as context; item['choices'] is a dict
        # {'text': [...], 'label': [...]}.
        question = item["question"].strip()
        choices = item["choices"]
        texts = list(choices["text"])
        labels = list(choices["label"])
        gold_idx = _label_to_index(item["answerKey"], labels)
        if gold_idx < 0 or gold_idx >= len(texts):
            invalid += 1
            continue
        # Format question as a prompt that expects an answer continuation.
        # The lm-eval-harness pattern: "Question: ...\nAnswer:" but we keep
        # it simpler so it works on a base model with no chat templating.
        context = f"Question: {question}\nAnswer:"
        try:
            scores = score_multiple_choice(adapter, context, texts,
                                            normalization=normalization)
            pred = int(max(range(len(scores)), key=lambda k: scores[k]))
            predictions.append(pred)
            gold.append(gold_idx)
            if verbose and i < 5:
                print(f"  [{i}] q={question[:60]}... gold={gold_idx} pred={pred}")
        except Exception as e:
            return make_error("arc_easy", PRIMARY_METRIC, e)

    acc = mc_accuracy(predictions, gold)
    wall = time.time() - t0
    return BenchmarkResult(
        name=f"arc_{config.lower().replace('arc-', '')}",
        primary_metric=PRIMARY_METRIC,
        metrics={
            "accuracy": acc,
            "examples_per_s": (len(predictions) / wall) if wall > 0 else 0.0,
            "invalid_skipped": float(invalid),
        },
        n_examples=len(predictions),
        wall_time_s=wall,
        notes=(f"split={split}, config={config}, "
               f"normalization={normalization}.  Random = {1/4:.0%}."),
    )
