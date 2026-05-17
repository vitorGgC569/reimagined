"""Common infrastructure for the theses validation harness.

Public API (everything else is internal):
  - data.load_wikitext_chars(...) -> (data_tensor, vocab_size, char_to_ix, ix_to_char)
  - data.get_batch(data_tensor, seq_len, batch_size, device)
  - training.train_and_eval(...) -> RunResult
  - training.RunResult dataclass
  - metrics.perplexity, metrics.count_loss_spikes, metrics.throughput_tok_s
  - reference_models.build_baseline_lm(...)
"""
from .data import load_wikitext_chars, get_batch, set_global_seed
from .training import train_and_eval, RunResult
from .metrics import perplexity, count_loss_spikes, throughput_tok_s
from .reference_models import build_simple_transformer_lm

__all__ = [
    "load_wikitext_chars",
    "get_batch",
    "set_global_seed",
    "train_and_eval",
    "RunResult",
    "perplexity",
    "count_loss_spikes",
    "throughput_tok_s",
    "build_simple_transformer_lm",
]
