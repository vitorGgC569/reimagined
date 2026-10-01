"""Perplexity scoring for WikiText-2 and similar language-modeling benchmarks.

Convention: token-level PPL = exp( (Σ -logprob(token_i)) / N_tokens ).
We use natural log throughout (matches log_softmax from the model).

For long documents, we slide a window of `max_seq_len` over the text
with `stride` step; tokens past the first window are scored conditional
on the previous window.  This is the standard 'sliding window' PPL used
by EleutherAI and the WikiText-2 leaderboard.
"""
from __future__ import annotations

import math
from typing import List

from ..adapters.base import ModelAdapter


def _context_limit(adapter: ModelAdapter) -> int:
    limit = adapter.capability.max_seq_len
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("Perplexity requires max_seq_len >= 2")
    if not adapter.capability.can_score_tokens:
        raise ValueError("Perplexity requires token scoring")
    return limit


def _negative_log_probability(adapter, prompt, target) -> float:
    value = adapter.score_tokens(prompt, target)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Token scorer must return a numeric log probability")
    if not math.isfinite(value) or value > 0:
        raise ValueError("Token log probability must be finite and <= 0")
    return -float(value)


def validate_perplexity_stats(stats: dict) -> None:
    """Reject absent/empty measurements, including the former zero fallback."""
    for key in ("ppl", "nll_sum", "n_tokens"):
        if key not in stats or isinstance(stats[key], bool):
            raise ValueError(f"Missing or invalid perplexity field: {key}")
        if not isinstance(stats[key], (int, float)) or not math.isfinite(stats[key]):
            raise ValueError(f"Non-finite perplexity field: {key}")
    if stats["n_tokens"] <= 0 or int(stats["n_tokens"]) != stats["n_tokens"]:
        raise ValueError("Perplexity requires a positive scored-token count")
    if stats["ppl"] < 1 or stats["nll_sum"] < 0:
        raise ValueError("Perplexity must be >= 1 and NLL must be nonnegative")
    if not math.isclose(math.log(stats["ppl"]),
                        stats["nll_sum"] / stats["n_tokens"],
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Perplexity disagrees with NLL/token count")


def token_perplexity(adapter: ModelAdapter, text: str) -> dict:
    """Plain (no-window) PPL.  Cheap but truncates long inputs."""
    max_len = _context_limit(adapter)
    ids = adapter.tokenize(text)
    if len(ids) < 2:
        raise ValueError("Perplexity requires at least two tokens")

    truncated = 0
    if len(ids) > max_len:
        truncated = len(ids) - max_len
        ids = ids[:max_len]

    # Score every token except the first (no context for it)
    nll_sum = _negative_log_probability(adapter, [ids[0]], ids[1:])
    n = len(ids) - 1
    result = {
        "ppl": math.exp(nll_sum / max(n, 1)),
        "nll_sum": nll_sum,
        "n_tokens": n,
        "n_truncated": truncated,
    }
    validate_perplexity_stats(result)
    return result


def sliding_window_perplexity(adapter: ModelAdapter, text: str, *,
                                stride: int | None = None,
                                max_tokens: int | None = None) -> dict:
    """Sliding-window PPL — the right number to report for WikiText.

    For each window of size `max_seq_len`, we score the LAST `stride`
    tokens conditional on the rest.  This avoids re-scoring tokens that
    have less context than the model can use.

    `max_tokens`: cap total scored tokens (None = whole document).  Use
    this for quick smoke tests against very long docs.
    """
    L = _context_limit(adapter)
    if max_tokens is not None and (isinstance(max_tokens, bool) or
            not isinstance(max_tokens, int) or max_tokens < 2):
        raise ValueError("max_tokens must be >= 2")
    if stride is not None and (isinstance(stride, bool) or
            not isinstance(stride, int) or not 1 <= stride < L):
        raise ValueError("stride must be in [1, max_seq_len)")
    full = adapter.tokenize(text)
    if len(full) < 2:
        raise ValueError("Perplexity requires at least two tokens")

    if max_tokens is not None and len(full) > max_tokens:
        full = full[:max_tokens]

    if stride is None:
        stride = max(L // 2, 1)
    stride = min(stride, L - 1)

    nll_sum = 0.0
    n_tokens = 0
    n_windows = 0

    pos = 0
    first_window = True
    while pos < len(full):
        end = min(pos + L, len(full))
        window = full[pos:end]
        if len(window) < 2:
            break
        n_windows += 1
        # Which tokens get scored this iteration?
        if first_window:
            # First window: score everything except the first token
            scored_start = 1
            first_window = False
        else:
            # Subsequent windows: only score the new `stride` tokens
            scored_start = max(L - stride, 1)
        prompt_ids = window[:scored_start]
        target_ids = window[scored_start:]
        if not target_ids:
            break
        nll_sum += _negative_log_probability(adapter, prompt_ids, target_ids)
        n_tokens += len(target_ids)
        if end == len(full):
            break
        pos += stride

    result = {
        "ppl": math.exp(nll_sum / max(n_tokens, 1)),
        "nll_sum": nll_sum,
        "n_tokens": n_tokens,
        "n_windows": n_windows,
    }
    validate_perplexity_stats(result)
    if n_tokens != len(full) - 1:
        raise ValueError("Sliding-window scoring omitted or repeated tokens")
    return result
