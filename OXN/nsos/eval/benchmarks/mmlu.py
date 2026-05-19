"""MMLU — Massive Multitask Language Understanding (subset).

The full MMLU is 57 subjects × ~270 questions = 15,908 test items.  Even
in 'quick' eval mode that's prohibitive for an edge model.  We sample
100 questions stratified across STEM subjects (which is what VISION.md
Eixo 2 promises) so a smoke run completes in minutes.

Reference numbers (test split, 100-Q STEM stratified):
  Random:        25.0% (always 4-choice)
  GPT-2 small:   ~25% (chance level — no real signal at this size)
  Llama-3-8B:    ~50%

A 40-80M model is unlikely to exceed 27-30%; sitting above random for
this subset would be a small but real signal.
"""
from __future__ import annotations

import random
import time
from typing import List, Optional

from ..adapters.base import ModelAdapter
from ..scoring.multiple_choice import score_multiple_choice, mc_accuracy
from .base import BenchmarkResult, make_skipped, make_error


PRIMARY_METRIC = "accuracy"


# STEM subjects in MMLU per the official Hendrycks et al. categorization.
# We pick this subset because VISION.md Eixo 2 specifically targets
# "mmlu_stem" and these are the subjects most aligned with the
# instruction/algorithm-heavy curriculum NSOS already trains on.
STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]


def _load_subset(n_total: int, subjects: List[str], seed: int = 42):
    """Sample `n_total` questions stratified roughly evenly across `subjects`."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Need `datasets`.  pip install datasets.") from e

    per = max(1, n_total // len(subjects))
    rng = random.Random(seed)
    items = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            continue
        if len(ds) == 0:
            continue
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:per]:
            row = ds[i]
            items.append({
                "subject": subj,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            })
        if len(items) >= n_total:
            break
    return items[:n_total]


def _format_question(item: dict) -> str:
    """Standard MMLU prompt: 'The following is a multiple choice question...'.

    Matches lm-eval-harness MMLU prompt for comparability.
    """
    return (
        f"The following is a multiple choice question (with answers) "
        f"about {item['subject'].replace('_', ' ')}.\n\n"
        f"{item['question']}\n"
        f"A. {item['choices'][0]}\n"
        f"B. {item['choices'][1]}\n"
        f"C. {item['choices'][2]}\n"
        f"D. {item['choices'][3]}\n"
        f"Answer:"
    )


# Letters used as continuations.  Each is a single space-prefixed letter
# so it's typically one BPE token in any reasonable tokenizer.
LETTERS = [" A", " B", " C", " D"]


def run(adapter: ModelAdapter, *, n_examples: int = 100,
        subjects: Optional[List[str]] = None, seed: int = 42,
        verbose: bool = False) -> BenchmarkResult:
    """MMLU STEM subset accuracy.

    Default n=100 stratified across STEM subjects.  Quick (~ 5 min on
    a small model) but stable enough that random vs trained is
    statistically distinguishable at p < 0.05.
    """
    if not adapter.capability.can_score_tokens:
        return make_skipped("mmlu_stem", PRIMARY_METRIC, "no score_tokens")
    if subjects is None:
        subjects = STEM_SUBJECTS
    try:
        items = _load_subset(n_examples, subjects, seed=seed)
    except Exception as e:
        return make_error("mmlu_stem", PRIMARY_METRIC, e)
    if not items:
        return make_skipped("mmlu_stem", PRIMARY_METRIC, "empty subject load")

    t0 = time.time()
    predictions: List[int] = []
    gold: List[int] = []
    per_subject_correct: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        prompt = _format_question(item)
        try:
            # Score each letter as a 1-2 token continuation.
            # We use 'none' normalization since all options are the same length
            # (each is a single letter).
            scores = score_multiple_choice(
                adapter, prompt, LETTERS, normalization="none",
                add_space_before_option=False,  # letters already have a space
            )
            pred = int(max(range(len(scores)), key=lambda k: scores[k]))
            predictions.append(pred)
            gold.append(item["answer"])
            if verbose and i < 5:
                print(f"  [{i}] {item['subject']}: gold={item['answer']} pred={pred}")
            subj = item["subject"]
            per_subject_correct.setdefault(subj, [0, 0])
            per_subject_correct[subj][1] += 1
            if pred == item["answer"]:
                per_subject_correct[subj][0] += 1
        except Exception as e:
            return make_error("mmlu_stem", PRIMARY_METRIC, e)

    acc = mc_accuracy(predictions, gold)
    wall = time.time() - t0
    return BenchmarkResult(
        name="mmlu_stem",
        primary_metric=PRIMARY_METRIC,
        metrics={
            "accuracy": acc,
            "n_subjects": float(len(per_subject_correct)),
            "examples_per_s": (len(predictions) / wall) if wall > 0 else 0.0,
        },
        n_examples=len(predictions),
        wall_time_s=wall,
        notes=(f"STEM stratified, n={n_examples}, {len(per_subject_correct)} "
               f"subjects.  Random = 25%."),
        extra={
            "per_subject": {
                k: {"correct": v[0], "total": v[1], "accuracy": v[0] / max(v[1], 1)}
                for k, v in per_subject_correct.items()
            }
        },
    )
