"""
Experimental Python blueprint for NSOS inference ideas.

This module is intentionally non-production. The shipping runtime lives in the
native C++ InferenceEngine and NSOS SDK.
"""

EXPERIMENTAL_BLUEPRINT_MESSAGE = (
    "OXN/nsos/inference.py is an experimental blueprint, not the shipping "
    "inference runtime. Use the C++ InferenceEngine/NSOS SDK instead."
)


class MambaKVCache:
    def __init__(self, **kwargs):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)


class ContinuousBatchScheduler:
    def __init__(self, **kwargs):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)


class TTTSpeculator:
    def __init__(self, **kwargs):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)


class NSOSInferenceEngine:
    """Blueprint-only placeholder kept to avoid silent fake inference."""

    from typing import Any, Optional

    kv_cache: Optional[Any]
    batch_scheduler: Optional[Any]
    speculator: Optional[Any]

    def __init__(self, model_path: str, config: dict):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def generate(self, prompt: str, max_new_tokens: int = 100):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def run_needle_test(self):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)
