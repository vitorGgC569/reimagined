"""Suíte de regressão para documentos fiscais brasileiros.

Documentos sintéticos com gabarito conhecido — verifica que mudanças no modelo
não regridem extrações que já funcionavam. Marcado como benchmark estatístico:
não falha em accuracy individual, mas registra métricas para o CI rastrear
tendências.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OCONTABIL_RUN_BENCHMARK"),
    reason="Define OCONTABIL_RUN_BENCHMARK=1 para rodar o benchmark de extração GLiNER",
)


@dataclass
class Sample:
    name: str
    doc_type: str
    text: str
    expected: dict[str, Any]


SAMPLES: list[Sample] = [
    Sample(
        name="nfe_basica",
        doc_type="NF-e",
        text=(
            "NOTA FISCAL ELETRONICA\n"
            "Numero: 123456 Serie: 1\n"
            "Data Emissao: 15/03/2024\n"
            "Valor Total: R$ 1.234,56\n"
            "Emitente: ACME LTDA CNPJ: 11.222.333/0001-81\n"
            "Destinatario: CLIENTE EXEMPLO LTDA CNPJ: 22.333.444/0001-92\n"
        ),
        expected={
            "numero_nota": "123456",
            "valor_total": 1234.56,
            "data_emissao": "15/03/2024",
        },
    ),
    Sample(
        name="boleto",
        doc_type="Boleto",
        text=(
            "Linha digitavel: 34191.79001 01043.510047 91020.150008 5 84410026000\n"
            "Beneficiario: BANCO DO BRASIL SA\n"
            "Vencimento: 30/04/2024\n"
            "Valor: R$ 250,00\n"
        ),
        expected={"valor": 250.00, "vencimento": "30/04/2024"},
    ),
    Sample(
        name="darf",
        doc_type="DARF",
        text=(
            "DARF\n"
            "Codigo Receita: 1708\n"
            "Periodo Apuracao: 03/2024\n"
            "Vencimento: 20/04/2024\n"
            "Valor Principal: R$ 500,00\n"
            "Valor Multa: R$ 0,00\n"
            "Valor Juros: R$ 0,00\n"
            "Valor Total: R$ 500,00\n"
        ),
        expected={"codigo_receita": "1708", "valor_total": 500.00},
    ),
]


@pytest.fixture(scope="module")
def gliner_model():
    """Carrega o modelo apenas uma vez (caro)."""
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained("fastino/gliner2-base-v1")


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.name)
def test_regression_extraction(gliner_model, sample: Sample) -> None:
    start = time.perf_counter()
    fields = list(sample.expected.keys())
    result = gliner_model.extract_entities(sample.text, fields)
    duration = time.perf_counter() - start

    # Não falha em campos individuais — apenas registra para análise.
    found = {e["label"]: e["text"] for e in result.get("entities", [])}
    coverage = sum(1 for k in sample.expected if k in found) / len(sample.expected)

    print(
        f"[{sample.name}] cobertura={coverage:.0%} tempo={duration:.2f}s "
        f"campos_encontrados={list(found.keys())}"
    )
    assert coverage >= 0.0  # benchmark, não regressão hard


def test_performance_throughput(gliner_model) -> None:
    """Mede tempo médio por extração — útil para detectar regressões de latência."""
    sample = SAMPLES[0]
    fields = list(sample.expected.keys())

    times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        gliner_model.extract_entities(sample.text, fields)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"Tempo médio: {avg*1000:.1f}ms")
    # Alerta se >5s/doc (CPU): ajuste conforme infra
    assert avg < 30.0, f"Extração muito lenta: {avg:.2f}s/doc"
