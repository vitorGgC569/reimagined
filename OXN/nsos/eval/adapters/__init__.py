"""Model adapters that present a uniform interface for benchmarks.

Every adapter exposes:
  * score_tokens(prompt_ids, target_ids) -> sum of logprobs of target
    conditioned on prompt.  This is the workhorse for multiple-choice
    benchmarks (HellaSwag, ARC, MMLU) and perplexity (WikiText).
  * generate(prompt_text, max_new_tokens, temperature) -> str.  For
    HumanEval-style generative benchmarks.
  * tokenize(text) -> list[int] and detokenize(ids) -> str.

The base class is in `base.py`.  Concrete adapters:
  * NsosAdapter — loads an NSOS .bin pack via `nsos_ext`.
  * DummyAdapter — random/identity model for testing the framework
    without a trained checkpoint.
  * HfAdapter — optional, wraps a HuggingFace `transformers.AutoModelForCausalLM`
    for baseline comparison (e.g. against TinyLlama or GPT-2-small).
"""
from .base import ModelAdapter, AdapterCapability
from .dummy_adapter import DummyAdapter

__all__ = ["ModelAdapter", "AdapterCapability", "DummyAdapter"]
