"""ENEM — questões objetivas do Exame Nacional do Ensino Médio.

Dataset: `eduagarcia/enem_challenge` (com `maritaca-ai/enem` como alternativa).
Questões reais de múltipla escolha do exame brasileiro, cobrindo linguagens,
humanas, natureza e matemática.

Por que este benchmark em vez de FaQuAD-NLI: o FaQuAD no Hub é baseado em
script de carregamento, formato que o `datasets` moderno não executa mais.
O ENEM é parquet nativo, mais difícil, e mede conhecimento e raciocínio em
português — não apenas concordância textual.

Questões marcadas como anuladas (`nullified`) são descartadas: não têm
gabarito válido.

Calibração (5 alternativas):
  Aleatório:            20,0%
  Modelo pequeno:       ~20% (indistinguível do acaso é o esperado nesta escala)
  GPT-4 classe:         90%+

Para um modelo de 71M parâmetros, o resultado esperado É o acaso. A métrica
serve para detectar (a) regressão abaixo do acaso, que indica bug de scoring,
e (b) o momento em que a escala começa a produzir sinal real.
"""
from __future__ import annotations

import time
from typing import List, Optional

from ..adapters.base import ModelAdapter
from ..scoring.multiple_choice import score_multiple_choice, mc_accuracy
from .base import BenchmarkResult, make_skipped, make_error

PRIMARY_METRIC = "accuracy"

DATASET_CANDIDATES = ("eduagarcia/enem_challenge", "maritaca-ai/enem")
LETTERS = ("A", "B", "C", "D", "E")


def _load_dataset(split: str, n_examples: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Requer `datasets`. pip install datasets.") from exc
    errors = []
    for dataset_id in DATASET_CANDIDATES:
        try:
            ds = load_dataset(dataset_id, split=split)
        except Exception as exc:  # noqa: BLE001 - tenta o próximo candidato
            errors.append(f"{dataset_id}: {exc}")
            continue
        if n_examples is not None and len(ds) > n_examples:
            ds = ds.select(range(n_examples))
        return ds, dataset_id
    raise RuntimeError("nenhum dataset ENEM disponível — " + " | ".join(errors))


def _choices(item) -> List[str]:
    """Aceita `choices.text` (enem_challenge) ou `alternatives` (maritaca)."""
    raw = item.get("choices")
    if isinstance(raw, dict) and raw.get("text"):
        return [str(c) for c in raw["text"]]
    if isinstance(raw, list) and raw:
        return [str(c) for c in raw]
    alternatives = item.get("alternatives")
    if isinstance(alternatives, list) and alternatives:
        return [str(c) for c in alternatives]
    return []


def _gold_index(item, n_choices: int) -> Optional[int]:
    key = item.get("answerKey") or item.get("label")
    if key is None:
        return None
    text = str(key).strip().upper()
    if text in LETTERS:
        index = LETTERS.index(text)
        return index if index < n_choices else None
    if text.isdigit():
        index = int(text)
        return index if 0 <= index < n_choices else None
    return None


def run(
    adapter: ModelAdapter,
    split: str = "train",
    n_examples: Optional[int] = None,
    **_: object,
) -> BenchmarkResult:
    started = time.perf_counter()
    try:
        dataset, dataset_id = _load_dataset(split, n_examples)
    except Exception as exc:  # noqa: BLE001 - reported as a skip, never silent
        return make_skipped("ptbr_enem", PRIMARY_METRIC, str(exc))

    try:
        predictions: List[int] = []
        gold: List[int] = []
        skipped_nullified = 0
        skipped_malformed = 0

        for item in dataset:
            if item.get("nullified"):
                skipped_nullified += 1
                continue
            question = (item.get("question") or "").strip()
            choices = _choices(item)
            if not question or len(choices) < 2:
                skipped_malformed += 1
                continue
            answer = _gold_index(item, len(choices))
            if answer is None:
                skipped_malformed += 1
                continue

            context = f"Pergunta: {question}\nResposta:"
            scores = score_multiple_choice(
                adapter, context, [f" {choice}" for choice in choices]
            )
            predictions.append(
                int(max(range(len(scores)), key=lambda i: scores[i]))
            )
            gold.append(answer)

        if not predictions:
            return make_skipped(
                "ptbr_enem", PRIMARY_METRIC,
                "nenhuma questão utilizável após descartar anuladas/malformadas",
            )

        chance = 1.0 / (len(LETTERS))
        return BenchmarkResult(
            name="ptbr_enem",
            primary_metric=PRIMARY_METRIC,
            metrics={
                "accuracy": mc_accuracy(predictions, gold),
                "chance": chance,
                "skipped_nullified": float(skipped_nullified),
                "skipped_malformed": float(skipped_malformed),
            },
            n_examples=len(predictions),
            wall_time_s=time.perf_counter() - started,
            notes=(
                f"dataset={dataset_id}, split={split}. Questões anuladas "
                f"descartadas ({skipped_nullified}). Nesta escala de modelo o "
                "resultado esperado é o acaso (~20%); abaixo disso indica bug "
                "de scoring, não incompetência do modelo."
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        return make_error("ptbr_enem", PRIMARY_METRIC, exc)
