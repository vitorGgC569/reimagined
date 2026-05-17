"""WikiText-2 character-level loader shared across all tests.

We use character-level tokenization (not BPE) because:
  1. Every test trains a small model in minutes; char-level eliminates
     tokenizer setup overhead and keeps vocab tiny (~70 characters).
  2. Char-level makes loss-spike detection cleaner (per-token loss has
     less variance than per-BPE-piece loss).
  3. Matches the user's original POC so the harness is directly comparable.

For tests that benefit from larger vocab (e.g. KAN MoE), wrap this with a
BPE pre-tokenizer at the call site — the loader is intentionally minimal.

CRITICAL IMPORT ORDER (Windows + torch 2.6 + datasets 4.8 known issue):
Importing `datasets` AFTER `torch` segfaults with ACCESS_VIOLATION in
the pyarrow runtime on some Windows + Python 3.11 + torch 2.6 combos
(reproduced on the user's GTX 1050 Ti dev box).  Importing `datasets`
first works.  We do the eager import at module load below so anyone
who does `from common.data import ...` from a script that already
imported torch picks up our prior `datasets` import in sys.modules.

If you're maintaining this and remove the eager import, make sure the
TOP of every test file does `import datasets` before any torch-touching
import.
"""
from __future__ import annotations

# Eager datasets import BEFORE torch — see docstring above.
import datasets as _datasets_eager  # noqa: F401

import random
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class CharDataset:
    """Loaded, tokenized, ready-to-batch char dataset."""
    train_tensor: torch.Tensor   # 1D long tensor, train split
    eval_tensor: torch.Tensor    # 1D long tensor, held-out
    vocab_size: int
    char_to_ix: dict
    ix_to_char: dict


def set_global_seed(seed: int) -> None:
    """Set every RNG that affects test outcomes.  CUDA non-determinism
    in cuBLAS/cuDNN still allows small numerical drift across runs;
    we accept that and report it in test headers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wikitext_chars(
    split_rows_train: int = 8000,
    split_rows_eval: int = 1500,
    max_train_chars: int = 400_000,
    max_eval_chars: int = 80_000,
    seed: int = 42,
) -> CharDataset:
    """Load WikiText-2-raw, build a char-level vocab from train, return tensors.

    We slice train_rows + eval_rows from the start of the WikiText train
    split.  The eval slice comes AFTER the train slice (no overlap) so
    held-out PPL is a fair signal.  Both slices share the vocab built
    from train — eval-only chars are mapped to a single 'unk' bucket.
    """
    set_global_seed(seed)

    # Lazy import so the loader file is cheap to import even without
    # `datasets` installed (e.g. when the user just wants the dataclass).
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rows = ds["text"]

    def _select(start: int, count: int, max_chars: int) -> str:
        chunk = " ".join(line for line in rows[start:start + count] if len(line.strip()) > 10)
        chunk = chunk.lower()
        chunk = re.sub(r"[^a-z0-9 \.,;!?]", "", chunk)
        chunk = re.sub(r"\s+", " ", chunk)
        return chunk[:max_chars]

    train_text = _select(0, split_rows_train, max_train_chars)
    eval_text = _select(split_rows_train, split_rows_eval, max_eval_chars)

    vocab_chars = sorted(set(train_text))
    char_to_ix = {ch: i for i, ch in enumerate(vocab_chars)}
    # Reserve last slot for unknown characters that show up only in eval.
    unk_ix = len(vocab_chars)
    ix_to_char = {i: ch for ch, i in char_to_ix.items()}
    ix_to_char[unk_ix] = "?"

    train_ids = [char_to_ix[c] for c in train_text]
    eval_ids = [char_to_ix.get(c, unk_ix) for c in eval_text]

    return CharDataset(
        train_tensor=torch.tensor(train_ids, dtype=torch.long),
        eval_tensor=torch.tensor(eval_ids, dtype=torch.long),
        vocab_size=len(vocab_chars) + 1,  # +1 for unk
        char_to_ix=char_to_ix,
        ix_to_char=ix_to_char,
    )


def get_batch(
    data_tensor: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random (x, y) windows from a 1D long tensor.

    `rng` lets a test pin the data-order RNG separately from the model
    init RNG, so two variants see the EXACT same batches even when their
    model init touches the global RNG differently.  Without this, "same
    seed" wouldn't actually give same data order.
    """
    n = data_tensor.size(0)
    if rng is None:
        ix = torch.randint(0, n - seq_len - 1, (batch_size,))
    else:
        ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=rng)
    x = torch.stack([data_tensor[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data_tensor[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y
