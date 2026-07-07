"""Standalone NSOS Mamba Python wrapper.

This package is intentionally thin: it does not implement a new architecture.
It builds a pure faithful-Mamba NSOS ``JambaModel`` on top of ``nsos_ext`` and
standardizes device selection, streaming incremental decoding and small
supervised datasets.
"""

from .core import (
    DecodeBenchmark,
    GenerationResult,
    MambaModuleConfig,
    NSOSMamba,
    detect_device,
)
from .datasets import (
    CharTokenizer,
    TextPair,
    load_text_pairs,
    load_text_pairs_json,
    load_text_pairs_jsonl,
    pairs_to_token_ids,
)

__all__ = [
    "CharTokenizer",
    "DecodeBenchmark",
    "GenerationResult",
    "MambaModuleConfig",
    "NSOSMamba",
    "TextPair",
    "detect_device",
    "load_text_pairs",
    "load_text_pairs_json",
    "load_text_pairs_jsonl",
    "pairs_to_token_ids",
]
