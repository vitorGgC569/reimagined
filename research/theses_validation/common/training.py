"""Training loop + RunResult dataclass shared across all tests.

The loop is intentionally simple — no LR schedule beyond constant, no
gradient accumulation, no mixed precision.  Tests that want those
behaviors can wrap the model in a module that opts in, but the harness
itself stays minimal so that A/B comparisons isolate exactly one
variable.

Key features for honest comparison:
  * Same data-order RNG (`data_rng`) so two variants see identical batches.
  * Same model init seed at the call site (responsibility of the test).
  * Loss-spike detection during training (running mean + σ window).
  * Held-out eval at fixed step intervals; held-out PPL is the primary
    quality metric reported.
  * Tokens-per-second from wall clock, not just step time.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CharDataset, get_batch


@dataclass
class RunResult:
    label: str
    final_train_loss: float
    eval_ppl: float
    loss_spikes: int          # count of steps where loss > running_mean + 0.5*sigma
    train_loss_curve: List[float] = field(default_factory=list)
    eval_loss_curve: List[float] = field(default_factory=list)
    eval_steps: List[int] = field(default_factory=list)
    wall_time_s: float = 0.0
    tokens_per_s: float = 0.0
    total_params: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "final_train_loss": self.final_train_loss,
            "eval_ppl": self.eval_ppl,
            "loss_spikes": self.loss_spikes,
            "wall_time_s": self.wall_time_s,
            "tokens_per_s": self.tokens_per_s,
            "total_params": self.total_params,
            "notes": self.notes,
            "train_loss_curve": self.train_loss_curve,
            "eval_loss_curve": self.eval_loss_curve,
            "eval_steps": self.eval_steps,
        }


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _running_spike_count(losses: List[float], window: int = 50, k: float = 0.5) -> int:
    """A step is a 'spike' when its loss exceeds the rolling mean of the
    previous `window` steps by more than `k * rolling_std`.

    The threshold (0.5σ) is intentionally loose so we catch the
    pendulum-style oscillations described in the thesis (ternary STE
    instability), not just hard divergences.  Calibrated empirically:
    a healthy run on this harness gets ~3-8 spikes; a STE-unstable run
    gets 20+.
    """
    if len(losses) < window + 1:
        return 0
    n_spikes = 0
    for i in range(window, len(losses)):
        window_vals = losses[i - window:i]
        m = sum(window_vals) / window
        var = sum((v - m) ** 2 for v in window_vals) / window
        sd = math.sqrt(max(var, 1e-12))
        if losses[i] > m + k * sd:
            n_spikes += 1
    return n_spikes


def train_and_eval(
    model: nn.Module,
    dataset: CharDataset,
    *,
    label: str,
    device: torch.device,
    steps: int = 2000,
    batch_size: int = 32,
    seq_len: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    eval_every: int = 200,
    eval_batches: int = 16,
    data_seed: int = 12345,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    grad_clip: Optional[float] = 1.0,
    notes: str = "",
) -> RunResult:
    """One training run with periodic held-out eval.

    Returns a RunResult with curves + summary metrics.  Tokens-per-second
    counts tokens processed by the loss (batch * seq_len * steps), not
    parameter updates.

    `data_seed` is the SEED FOR DATA ORDER ONLY.  Two variants passed
    the same data_seed see identical batches (in the same order) even if
    their model init touches the global RNG differently.  This is the
    critical isolation knob.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Dedicated RNG for sample selection — independent of model RNG.
    # Pin to CPU since torch.randint with a generator needs the device
    # of the generator to match.  Indices get moved to device by get_batch.
    data_rng = torch.Generator(device="cpu")
    data_rng.manual_seed(data_seed)

    train_curve: List[float] = []
    eval_curve: List[float] = []
    eval_step_marks: List[int] = []

    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(dataset.train_tensor, seq_len, batch_size, device, rng=data_rng)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, dataset.vocab_size),
            y.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_val = float(loss.detach().item())
        train_curve.append(loss_val)

        if progress_callback and step % 50 == 0:
            progress_callback(step, loss_val)

        if step % eval_every == 0 or step == steps:
            eval_loss = _eval(model, dataset, seq_len, batch_size, device, eval_batches)
            eval_curve.append(eval_loss)
            eval_step_marks.append(step)
            model.train()

    wall_time = time.time() - start
    total_tokens = steps * batch_size * seq_len

    final_train = sum(train_curve[-50:]) / max(len(train_curve[-50:]), 1)
    final_eval_loss = eval_curve[-1] if eval_curve else float("nan")

    return RunResult(
        label=label,
        final_train_loss=final_train,
        eval_ppl=float(math.exp(final_eval_loss)) if not math.isnan(final_eval_loss) else float("nan"),
        loss_spikes=_running_spike_count(train_curve),
        train_loss_curve=train_curve,
        eval_loss_curve=eval_curve,
        eval_steps=eval_step_marks,
        wall_time_s=wall_time,
        tokens_per_s=total_tokens / wall_time if wall_time > 0 else 0.0,
        total_params=_count_params(model),
        notes=notes,
    )


@torch.no_grad()
def _eval(
    model: nn.Module,
    dataset: CharDataset,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    n_batches: int,
) -> float:
    """Held-out cross-entropy averaged across `n_batches` random eval windows.

    Uses a fixed RNG seed per call so eval noise is constant across
    A/B runs at the same step — eliminates one source of jitter when
    comparing two variants' eval curves side by side.
    """
    model.eval()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(7777)  # constant, deliberately
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(dataset.eval_tensor, seq_len, batch_size, device, rng=rng)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, dataset.vocab_size),
            y.reshape(-1),
        )
        total += float(loss.item())
    return total / n_batches
