"""ASSIN 2 — inferência textual (entailment) em português do Brasil.

Dataset: `assin2`, split `validation`. Cada item traz um par
(premissa, hipótese) e um rótulo de implicação. É um dos poucos benchmarks
PT-BR com anotação humana, licença permissiva e uso consolidado na literatura
brasileira de PLN.

Formatamos como escolha múltipla de duas opções ("Sim" / "Não") e pontuamos
por logprob normalizado pelo comprimento, exatamente como os benchmarks de
múltipla escolha em inglês deste harness — assim os números são comparáveis
em metodologia.

Referências de calibração (validation, 2 opções):
  Aleatório:        50,0%
  Modelo pequeno mal treinado: ~50% (indistinguível do acaso)
  Encoders PT-BR ajustados:    85%+

Abaixo de ~52% significa que o modelo não distingue nada; a métrica só passa
a ser informativa acima disso.
"""
from __future__ import annotations

import time
from typing import Optional

from ..adapters.base import ModelAdapter
from ..scoring.multiple_choice import score_multiple_choice, mc_accuracy
from .base import BenchmarkResult, make_skipped, make_error

PRIMARY_METRIC = "accuracy"

CHOICES = ("Não", "Sim")


# O id sem namespace (`assin2`) deixou de ser aceito pelo Hub; o canônico é
# `nilc-nlp/assin2`. A lista permite migração futura sem quebrar o gate.
DATASET_CANDIDATES = ("nilc-nlp/assin2", "assin2")


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
    raise RuntimeError("nenhum dataset ASSIN 2 disponível — " + " | ".join(errors))


def _prompt(premise: str, hypothesis: str) -> str:
    return (
        "Leia o texto e responda se a afirmação decorre dele.\n"
        f"Texto: {premise.strip()}\n"
        f"Afirmação: {hypothesis.strip()}\n"
        "A afirmação decorre do texto?"
    )


def run(
    adapter: ModelAdapter,
    split: str = "validation",
    n_examples: Optional[int] = None,
    **_: object,
) -> BenchmarkResult:
    started = time.perf_counter()
    try:
        dataset, dataset_id = _load_dataset(split, n_examples)
    except Exception as exc:  # noqa: BLE001 - reported as a skip, never silent
        return make_skipped("ptbr_assin2", PRIMARY_METRIC, str(exc))

    try:
        predictions = []
        gold = []
        for item in dataset:
            premise = item.get("premise") or item.get("text") or ""
            hypothesis = item.get("hypothesis") or ""
            label = item.get("entailment_judgment")
            if label is None or not premise or not hypothesis:
                continue
            # ASSIN 2 codifica 0 = None (não implica), 1 = Entailment.
            label = int(label)
            context = _prompt(premise, hypothesis)
            continuations = [f" {choice}" for choice in CHOICES]
            scores = score_multiple_choice(adapter, context, continuations)
            predictions.append(int(max(range(len(scores)),
                                       key=lambda i: scores[i])))
            gold.append(int(label))

        if not predictions:
            return make_skipped(
                "ptbr_assin2", PRIMARY_METRIC,
                "nenhum item utilizável no split solicitado",
            )

        accuracy = mc_accuracy(predictions, gold)
        positive_rate = sum(predictions) / len(predictions)
        return BenchmarkResult(
            name="ptbr_assin2",
            primary_metric=PRIMARY_METRIC,
            metrics={
                "accuracy": accuracy,
                "predicted_positive_rate": positive_rate,
                "chance": 0.5,
            },
            n_examples=len(predictions),
            wall_time_s=time.perf_counter() - started,
            notes=(
                f"dataset={dataset_id}. Entailment PT-BR como escolha "
                "binária, logprob normalizado por comprimento. "
                "predicted_positive_rate próximo de 0 ou 1 indica colapso "
                "de classe, não competência."
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        return make_error("ptbr_assin2", PRIMARY_METRIC, exc)
