"""DummyAdapter — deterministic random-uniform "model" for testing the harness.

Why we need it: the eval suite must work BEFORE an NSOS checkpoint exists.
DummyAdapter returns uniform logprobs over a fixed vocab, so every
benchmark runs cleanly and yields a chance-level result.  This lets us
catch dataset-loading bugs, scoring bugs, and orchestration bugs
without burning training time first.

A multiple-choice benchmark scored against DummyAdapter should give
~ 1/N_options accuracy (HellaSwag = 25%, ARC = 25%, MMLU = 25%).
A PPL benchmark should give PPL ≈ vocab_size.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

from .base import ModelAdapter, AdapterCapability


class DummyAdapter(ModelAdapter):
    """Random-uniform pseudo-model.  All sequences cost equal logprob."""

    def __init__(self, vocab_size: int = 256, max_seq_len: int = 2048,
                 seed: int = 42, generation_seed: int = 7):
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        # Two independent RNGs: one for tokenize/score determinism,
        # one for generate.  Scoring is deterministic given inputs;
        # generation is reproducible per call only when temperature=0,
        # which we approximate via the generation_seed.
        self._tok_rng = random.Random(seed)
        self._gen_rng = random.Random(generation_seed)

    @property
    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            can_score_tokens=True,
            can_generate=True,
            can_batch_score=False,
            max_seq_len=self._max_seq_len,
            vocab_size=self._vocab_size,
            model_name="dummy/uniform-random",
            model_params=0,
            notes=("Returns log(1/vocab) per token.  Used to smoke-test the eval "
                   "framework before any real model exists."),
        )

    def tokenize(self, text: str) -> List[int]:
        # Char-level toy tokenizer modulo vocab.  Deterministic.
        return [ord(c) % self._vocab_size for c in text]

    def detokenize(self, ids: List[int]) -> str:
        # Best-effort inverse — only safe for ids in printable ASCII range.
        return "".join(chr((i % 95) + 32) for i in ids)

    def score_tokens(self, prompt_ids: List[int], target_ids: List[int]) -> float:
        # Uniform distribution over vocab → each token has log(1/V) logprob.
        return len(target_ids) * math.log(1.0 / max(self._vocab_size, 1))

    def generate(self, prompt_text: str, *, max_new_tokens: int = 256,
                  temperature: float = 0.7, top_k: int = 40,
                  stop_sequences: Optional[List[str]] = None) -> str:
        # Emit random printable chars.  This is genuinely useless output,
        # but lets HumanEval-style benchmarks complete and report 0% pass.
        out_chars: List[str] = []
        for _ in range(max_new_tokens):
            c = chr(self._gen_rng.randint(32, 126))
            out_chars.append(c)
            current = "".join(out_chars)
            if stop_sequences and any(s in current for s in stop_sequences):
                break
        return "".join(out_chars)
