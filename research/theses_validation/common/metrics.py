"""Stand-alone metric functions for ad-hoc analysis outside the training loop."""
from __future__ import annotations

import math
import statistics
from typing import List


def perplexity(cross_entropy_nats: float) -> float:
    """Convert a cross-entropy (nats/token) to perplexity."""
    return math.exp(cross_entropy_nats)


def count_loss_spikes(losses: List[float], window: int = 50, k: float = 0.5) -> int:
    """Count training steps where loss > rolling_mean + k * rolling_std.

    Same definition as training._running_spike_count but exposed for
    use on saved curves.
    """
    if len(losses) < window + 1:
        return 0
    n_spikes = 0
    for i in range(window, len(losses)):
        window_vals = losses[i - window:i]
        m = sum(window_vals) / window
        sd = statistics.pstdev(window_vals) if len(window_vals) > 1 else 0.0
        if sd > 0 and losses[i] > m + k * sd:
            n_spikes += 1
    return n_spikes


def throughput_tok_s(total_tokens: int, wall_time_s: float) -> float:
    """Simple tokens-per-second from totals."""
    if wall_time_s <= 0:
        return 0.0
    return total_tokens / wall_time_s


def relative_delta(baseline: float, variant: float) -> float:
    """Return the variant's improvement over baseline as a fraction
    (negative = improvement when LOWER is better, e.g. loss/PPL)."""
    if baseline == 0:
        return 0.0
    return (variant - baseline) / baseline
