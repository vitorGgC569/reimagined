"""ModelAdapter — the uniform interface every benchmark talks to.

We deliberately keep the surface minimal: scoring, generation, and
tokenization.  Anything benchmark-specific (multiple-choice formatting,
chain-of-thought prompts, etc.) lives in the benchmark module, not here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AdapterCapability:
    """Tells the orchestrator what this adapter actually supports.

    Benchmarks that require a missing capability are skipped with a clear
    'unsupported' status, not failed.  This lets the harness run on
    cheap adapters (e.g., DummyAdapter doesn't really generate) without
    cascading failures.
    """
    can_score_tokens: bool = True
    can_generate: bool = True
    can_batch_score: bool = False        # batched score_tokens speeds up MC benchmarks
    max_seq_len: int = 2048
    vocab_size: int = 0                  # 0 = unknown / runtime
    model_name: str = "adapter"
    model_params: int = 0                # 0 = unknown
    notes: str = ""


class ModelAdapter(ABC):
    """Abstract base every benchmark consumes.

    Implementations:
      * Must keep `tokenize`/`detokenize` deterministic and reproducible.
      * Must accept arbitrary token sequences up to `capability.max_seq_len`.
      * May raise `RuntimeError` on capability misuse — benchmarks check
        capabilities before calling.
    """

    def evaluation_identity(self) -> dict:
        """Unknown adapters are never silently classified as trained models."""
        return {"kind": "unverified", "model_name": self.capability.model_name}

    @property
    @abstractmethod
    def capability(self) -> AdapterCapability:
        """Static description of what this adapter can do."""

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Return integer token ids for `text`.  No special-token wrapping."""

    @abstractmethod
    def detokenize(self, ids: List[int]) -> str:
        """Inverse of `tokenize`.  Round-trip is not guaranteed exact
        (BPE adds subtokens) but `detokenize(tokenize(x))` should be
        a faithful textual reproduction."""

    @abstractmethod
    def score_tokens(self, prompt_ids: List[int], target_ids: List[int]) -> float:
        """Return sum of natural-log probabilities of `target_ids`
        conditioned on `prompt_ids`.

        Mathematically:
            log P(t_0..t_{n-1} | prompt) = Σ log softmax(logits[prompt + t_0..t_{i-1}])[t_i]

        Implementations should NOT add or subtract any normalization
        (length, temperature, etc.) — benchmarks apply those themselves
        if needed.
        """

    @abstractmethod
    def generate(self, prompt_text: str, *, max_new_tokens: int = 256,
                  temperature: float = 0.7, top_k: int = 40,
                  stop_sequences: Optional[List[str]] = None) -> str:
        """Greedy or sampled text completion.  Returns the GENERATED
        portion only (does not echo the prompt).

        `stop_sequences` may be `None` (no stopping) or a list of strings;
        generation halts the first time any stop sequence appears in the
        decoded output (substring match).
        """

    def batch_score_tokens(self, items: List[tuple]) -> List[float]:
        """Optional fast path for multiple-choice benchmarks.  Default
        implementation falls back to a serial loop; subclasses that
        support batched forward passes should override.

        `items` is a list of `(prompt_ids, target_ids)` pairs.
        """
        return [self.score_tokens(p, t) for p, t in items]
