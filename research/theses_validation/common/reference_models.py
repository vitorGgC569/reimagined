"""Reference baseline models used by multiple tests.

These are deliberately small and standard.  We do NOT hand-roll
exotic versions of standard architectures — when comparing X against
"a Transformer", we want "a Transformer" to be a fair, vanilla
implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleTransformerLM(nn.Module):
    """Plain decoder-only Transformer language model.

    Used as a sanity baseline.  Default config (d_model=128, layers=2)
    gives ~300K parameters, matches roughly what the test harness'
    other small models look like.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_seq: int = 256, dropout: float = 0.0):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq = max_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq)
        b, t = x.shape
        assert t <= self.max_seq, "input seq exceeds positional table"
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.tok_embed(x) + self.pos_embed(pos)
        # Causal mask
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.head(h)


def build_simple_transformer_lm(
    vocab_size: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    max_seq: int = 256,
) -> SimpleTransformerLM:
    """Factory matching the build_* convention used by the test files."""
    return SimpleTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq=max_seq,
    )
