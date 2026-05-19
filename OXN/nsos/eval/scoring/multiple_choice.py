"""Multiple-choice scoring used by HellaSwag, ARC, MMLU.

Standard approach: for each option, compute Σ log P(option_i | context).
Pick the option with highest score.  Three normalization variants are
supported because each benchmark publishes numbers with a specific choice:

  * 'none'         — raw sum of token logprobs.  GPT-3 / Llama default.
  * 'token_length' — average logprob per option token.  Reduces bias
                     toward shorter options.  ARC-Easy uses this in
                     the EleutherAI lm-eval-harness.
  * 'char_length'  — divide by number of characters.  HellaSwag's
                     official scoring; further reduces length bias.
"""
from __future__ import annotations

from typing import List, Tuple

from ..adapters.base import ModelAdapter


def score_multiple_choice(
    adapter: ModelAdapter,
    context: str,
    options: List[str],
    *,
    normalization: str = "none",
    add_space_before_option: bool = True,
) -> List[float]:
    """Return one score per option.  Higher = more likely.

    `context` is tokenized once.  Each option is appended (with a leading
    space if `add_space_before_option`) and scored conditional on the
    context.  We add the leading space so BPE tokenizers don't merge
    the last context character with the option's first character.

    Returns scores in same order as `options`.
    """
    ctx_ids = adapter.tokenize(context)
    pairs: List[Tuple[List[int], List[int]]] = []
    raw_opt_lens: List[int] = []
    char_lens: List[int] = []
    for opt in options:
        opt_text = (" " + opt) if (add_space_before_option and not opt.startswith(" ")) else opt
        opt_ids = adapter.tokenize(opt_text)
        pairs.append((ctx_ids, opt_ids))
        raw_opt_lens.append(len(opt_ids))
        char_lens.append(max(len(opt_text), 1))

    if adapter.capability.can_batch_score:
        raw_scores = adapter.batch_score_tokens(pairs)
    else:
        raw_scores = [adapter.score_tokens(p, t) for p, t in pairs]

    if normalization == "none":
        return raw_scores
    if normalization == "token_length":
        return [s / max(n, 1) for s, n in zip(raw_scores, raw_opt_lens)]
    if normalization == "char_length":
        return [s / c for s, c in zip(raw_scores, char_lens)]
    raise ValueError(f"unknown normalization {normalization!r}")


def mc_accuracy(predictions: List[int], gold: List[int]) -> float:
    """Plain top-1 accuracy.  Both lists must have the same length."""
    if not predictions:
        return 0.0
    assert len(predictions) == len(gold), \
        f"length mismatch: {len(predictions)} predictions vs {len(gold)} gold"
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    return correct / len(predictions)
