"""NCA pretraining transfer test (Tese 1, section 'Morfogênese Cognitiva').

Claim under test: a small amount of Neural Cellular Automaton
pretraining (the thesis says ~164M tokens, claims 1.6× convergence
acceleration + ~6% perplexity gain) makes downstream language
modeling converge faster.

Honest scope of this test:
  * We use ~50K NCA "tokens" (cell states), not 164M.  Scaling to
    164M takes hours on T4 alone — out of budget for a single-test
    file.  This means our test detects DIRECTION, not magnitude.
  * The downstream task is the same char-LM (WikiText-2 small) we
    use elsewhere.  Same vocab, same data slice.
  * Compared models: identical Transformer LM init.  Variant A
    trains from random init on language.  Variant B does NCA
    pretraining first, then trains on language with the same
    remaining step budget.

What we measure:
  * Loss curve and held-out PPL of the language phase.
  * If NCA-pretrained converges faster (same loss reached in fewer
    steps) AND/OR plateaus lower → claim partially validated.
  * If they converge identically or NCA hurts → claim doesn't replicate
    at this scale.

Setup of the NCA itself (intentionally simple):
  * 32×32 grid, 8-channel state (matches Mordvintsev's original).
  * Diffusion + per-cell MLP update for K steps per "rollout".
  * "Tokens" are produced by flattening the grid and quantizing each
    cell's 8-channel state into a single bucket via argmax + bias.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Eager datasets import before torch — Windows segfault workaround;
# see common/data.py docstring for details.
import datasets  # eager, before torch  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_wikitext_chars, set_global_seed, get_batch
from common.training import train_and_eval, RunResult


# ──────────────────────────────────────────────────────────────────────
#  Simple NCA — Mordvintsev-style
# ──────────────────────────────────────────────────────────────────────

class SimpleNCA(nn.Module):
    """8-channel NCA on a 32x32 grid.  Sees a 3x3 neighborhood (via
    depthwise conv with fixed Sobel-like kernels) and produces a delta
    per channel via a small per-cell MLP.

    State evolves via x_{t+1} = x_t + alpha * delta.  We run K=16 steps
    per rollout and use the trajectory as our "token stream" by
    quantizing the per-cell state.
    """

    def __init__(self, channels: int = 8, grid: int = 32):
        super().__init__()
        self.channels = channels
        self.grid = grid
        # Three fixed perception filters: identity, sobel_x, sobel_y.
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        sobel_y = sobel_x.t()
        self.register_buffer("filters", torch.stack([identity, sobel_x, sobel_y]).unsqueeze(1))
        # Per-cell MLP: 3*C → 64 → C
        self.update = nn.Sequential(
            nn.Linear(3 * channels, 64),
            nn.GELU(),
            nn.Linear(64, channels),
        )

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W).  Apply 3 depthwise filters → (B, 3*C, H, W).
        b, c, h, w = x.shape
        # Repeat filters per channel to do depthwise conv.
        f = self.filters.repeat(c, 1, 1, 1)  # (C, 1, 3, 3) → repeated 3× along output dim
        # Reshape to (3*C, 1, 3, 3) so each output channel uses one filter on one input channel.
        # Easier: do groups=c conv with filters tiled appropriately.
        filt = torch.cat([self.filters] * c, dim=0)  # (3*c, 1, 3, 3)
        # Pad reflect so edges don't blow up
        x_pad = F.pad(x, (1, 1, 1, 1), mode="circular")
        perc = F.conv2d(x_pad, filt, groups=c)
        return perc  # (B, 3*c, H, W)

    def step(self, x: torch.Tensor, alpha: float = 1.0, fire_p: float = 0.5) -> torch.Tensor:
        perc = self.perceive(x)  # (B, 3c, H, W)
        # Per-cell MLP on the channel dim
        b, ch3, h, w = perc.shape
        flat = perc.permute(0, 2, 3, 1).reshape(b * h * w, ch3)
        delta = self.update(flat).reshape(b, h, w, self.channels).permute(0, 3, 1, 2)
        # Stochastic firing: random mask of cells to update (NCA "asynchrony")
        mask = (torch.rand(b, 1, h, w, device=x.device) < fire_p).float()
        return x + alpha * delta * mask

    def rollout(self, batch: int, steps: int = 16, device: torch.device | None = None) -> torch.Tensor:
        """Return all intermediate states as (B, steps, C, H, W)."""
        if device is None:
            device = next(self.parameters()).device
        x = torch.zeros(batch, self.channels, self.grid, self.grid, device=device)
        # Seed: alive cell at center
        x[:, :, self.grid // 2, self.grid // 2] = 1.0
        states = []
        for _ in range(steps):
            x = self.step(x)
            states.append(x.clone())
        return torch.stack(states, dim=1)


# ──────────────────────────────────────────────────────────────────────
#  Tokenize NCA trajectories into a vocab the LM can ingest
# ──────────────────────────────────────────────────────────────────────

def nca_to_tokens(states: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map (B, T, C, H, W) states to a token sequence (B*H*W*T,).

    Per-cell argmax of the channels → integer in [0, C); we hash with
    H, W positions modulo vocab_size to get a richer token stream
    bounded by the LM vocab.
    """
    b, t, c, h, w = states.shape
    per_cell = states.argmax(dim=2)  # (B, T, H, W)
    # Mix with positional hash to lift integer to vocab_size range
    pos_h = torch.arange(h, device=states.device).view(1, 1, h, 1)
    pos_w = torch.arange(w, device=states.device).view(1, 1, 1, w)
    tokens = (per_cell * 31 + pos_h * 7 + pos_w * 13) % vocab_size
    return tokens.reshape(-1)


# ──────────────────────────────────────────────────────────────────────
#  LM backbone (same as elsewhere, kept local for self-containment)
# ──────────────────────────────────────────────────────────────────────

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_seq: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.max_seq = max_seq
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                            dropout=0.0, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(self.ln(h))


# ──────────────────────────────────────────────────────────────────────
#  Pretrain on NCA tokens
# ──────────────────────────────────────────────────────────────────────

def pretrain_on_nca(model: nn.Module, vocab_size: int, *, device: torch.device,
                     n_rollouts: int = 50, batch: int = 8, steps_per_rollout: int = 16,
                     lr: float = 3e-4) -> List[float]:
    # The caller passes a freshly-constructed model that lives on CPU.
    # We move it to device here so tokens (also on device) match.
    # train_and_eval downstream re-moves it, which is a no-op.
    model.to(device)
    nca = SimpleNCA(channels=8, grid=32).to(device)
    # We DO NOT train the NCA — we use a randomly-init'd NCA as a
    # synthetic data generator.  Mordvintsev's original NCA paper
    # trains the NCA to reach a target image; for transfer pretraining
    # what matters is the temporal-spatial token distribution, not the
    # NCA reaching any particular target.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: List[float] = []
    seq_len = 64
    model.train()
    for r in range(n_rollouts):
        with torch.no_grad():
            states = nca.rollout(batch=batch, steps=steps_per_rollout, device=device)
            tokens = nca_to_tokens(states, vocab_size).to(device)
        # Chunk into seq_len windows
        n_full = (tokens.numel() // (seq_len + 1)) - 1
        for i in range(max(n_full, 1)):
            start = i * seq_len
            x = tokens[start:start + seq_len].unsqueeze(0)
            y = tokens[start + 1:start + 1 + seq_len].unsqueeze(0)
            if x.size(1) < seq_len or y.size(1) < seq_len:
                continue
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
    return losses


# ──────────────────────────────────────────────────────────────────────
#  Test entry point
# ──────────────────────────────────────────────────────────────────────

def test_nca_pretrain(*, device: torch.device, dataset, steps: int = 1500,
                       seed: int = 42, nca_rollouts: int = 50) -> dict:
    out = {}

    # Variant A: cold-start LM on language
    set_global_seed(seed)
    cold = TinyTransformerLM(dataset.vocab_size)
    out["Cold-start"] = train_and_eval(
        cold, dataset, label="Cold-start LM", device=device,
        steps=steps, batch_size=32, seq_len=64, lr=3e-4,
        notes="random init, language directly",
    )

    # Variant B: NCA pretrain → then language
    set_global_seed(seed)
    pretr = TinyTransformerLM(dataset.vocab_size)
    nca_losses = pretrain_on_nca(pretr, dataset.vocab_size,
                                  device=device, n_rollouts=nca_rollouts)
    nca_final = sum(nca_losses[-20:]) / max(len(nca_losses[-20:]), 1) if nca_losses else float("nan")
    out["NCA-pretrained"] = train_and_eval(
        pretr, dataset, label="NCA-pretrained LM", device=device,
        steps=steps, batch_size=32, seq_len=64, lr=3e-4,
        notes=f"after {nca_rollouts} NCA rollouts (~{len(nca_losses)} pretrain steps, final NCA loss ~{nca_final:.3f})",
    )
    return out


ALL_TESTS = {"nca_pretrain": test_nca_pretrain}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="NCA pretrain transfer test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--nca_rollouts", type=int, default=50)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nca_pretrain] device={device.type}")
    dataset = load_wikitext_chars(seed=args.seed)
    results = test_nca_pretrain(device=device, dataset=dataset, steps=args.steps,
                                 seed=args.seed, nca_rollouts=args.nca_rollouts)
    print(f"\n=== test_nca_pretrain ===")
    print(f"{'Variant':<35} {'TrainLoss':>10} {'EvalPPL':>10} {'Spikes':>8}")
    for res in results.values():
        print(f"{res.label:<35} {res.final_train_loss:>10.4f} {res.eval_ppl:>10.2f} {res.loss_spikes:>8d}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
