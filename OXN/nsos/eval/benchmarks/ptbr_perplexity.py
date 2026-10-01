"""Perplexidade em português do Brasil sobre texto held-out.

Contraparte PT-BR do `wikitext.py`: sem ela, a única medida de modelagem de
linguagem do harness é em inglês, o que não diz nada sobre o objetivo do
produto.

Fonte padrão: `wikimedia/wikipedia`, configuração `20231101.pt`, amostrada de
forma determinística. Wikipedia-PT é CC BY-SA — adequada para avaliação, e a
avaliação não redistribui o texto.

Importante: perplexidade só é comparável entre modelos que compartilham o
mesmo tokenizador. Serve para acompanhar o próprio modelo ao longo do treino,
não para ranquear contra modelos de terceiros.
"""
from __future__ import annotations

import time
import hashlib
from typing import Optional

from ..adapters.base import ModelAdapter
from ..scoring.perplexity import sliding_window_perplexity, validate_perplexity_stats
from .base import BenchmarkResult, make_skipped, make_error

PRIMARY_METRIC = "perplexity"


def _load_text(
    n_documents: int, seed: int, config: str
) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Requer `datasets`. pip install datasets.") from exc
    ds = load_dataset("wikimedia/wikipedia", config, split="train",
                      streaming=True)
    # Deterministic prefix skip keeps the sample stable across runs without
    # materialising the dataset.
    shuffled = ds.shuffle(seed=seed, buffer_size=2_000)
    chunks = []
    for index, row in enumerate(shuffled):
        if index >= n_documents:
            break
        text = (row.get("text") or "").strip()
        if len(text) > 200:
            chunks.append(text)
    if not chunks:
        raise RuntimeError("nenhum documento utilizável recuperado")
    return "\n\n".join(chunks)


def run(
    adapter: ModelAdapter,
    n_documents: int = 60,
    seed: int = 20260806,
    config: str = "20231101.pt",
    max_length: Optional[int] = None,
    **_: object,
) -> BenchmarkResult:
    started = time.perf_counter()
    if not adapter.capability.can_score_tokens:
        return make_skipped("ptbr_perplexity", PRIMARY_METRIC, "no score_tokens")
    if n_documents <= 0 or (max_length is not None and max_length < 2):
        return make_error("ptbr_perplexity", PRIMARY_METRIC,
                          ValueError("Invalid document/character limit"))
    try:
        text = _load_text(n_documents, seed, config)
    except Exception as exc:  # noqa: BLE001 - reported as a skip, never silent
        return make_skipped("ptbr_perplexity", PRIMARY_METRIC, str(exc))

    try:
        if max_length is not None:
            text = text[:max_length]
        stats = sliding_window_perplexity(adapter, text)
        validate_perplexity_stats(stats)
        metrics = {
            key: float(value)
            for key, value in stats.items()
            if isinstance(value, (int, float))
        }
        metrics[PRIMARY_METRIC] = stats["ppl"]
        return BenchmarkResult(
            name="ptbr_perplexity",
            primary_metric=PRIMARY_METRIC,
            metrics=metrics,
            n_examples=int(stats["n_tokens"]),
            wall_time_s=time.perf_counter() - started,
            notes=(
                f"Wikipedia-PT ({config}), amostra determinística seed={seed}. "
                "Comparável apenas entre modelos com o mesmo tokenizador."
            ),
            extra={"dataset": "wikimedia/wikipedia", "config": config,
                   "split": "train", "seed": seed,
                   "requested_documents": n_documents,
                   "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                   "count_unit": "scored_tokens"},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        return make_error("ptbr_perplexity", PRIMARY_METRIC, exc)
