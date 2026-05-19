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


def token_perplexity(adapter: ModelAdapter, text: str) -> dict:
    """Plain (no-window) PPL.  Cheap but truncates long inputs."""
    ids = adapter.tokenize(text)
    if len(ids) < 2:
        return {"ppl": float("nan"), "nll_sum": 0.0, "n_tokens": 0,
                "n_truncated": 0}

    max_len = adapter.capability.max_seq_len
    truncated = 0
    if len(ids) > max_len:
        truncated = len(ids) - max_len
        ids = ids[:max_len]

    # Score every token except the first (no context for it)
    nll_sum = -adapter.score_tokens([ids[0]], ids[1:])
    n = len(ids) - 1
    return {
        "ppl": math.exp(nll_sum / max(n, 1)),
        "nll_sum": nll_sum,
        "n_tokens": n,
        "n_truncated": truncated,
    }


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
    full = adapter.tokenize(text)
    if len(full) < 2:
        return {"ppl": float("nan"), "nll_sum": 0.0, "n_tokens": 0,
                "n_windows": 0}

    if max_tokens is not None and len(full) > max_tokens:
        full = full[:max_tokens]

    L = adapter.capability.max_seq_len
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
        logprob_sum = adapter.score_tokens(prompt_ids, target_ids)
        nll_sum += -logprob_sum
        n_tokens += len(target_ids)
        if end == len(full):
            break
        pos += stride

    return {
        "ppl": math.exp(nll_sum / max(n_tokens, 1)),
        "nll_sum": nll_sum,
        "n_tokens": n_tokens,
        "n_windows": n_windows,
    }
