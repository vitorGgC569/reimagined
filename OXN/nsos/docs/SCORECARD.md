# NSOS Scorecard

> Referenciado por `PRODUCT.md` como "External scorecard". Define **como** o
> produto é pontuado e registra o estado atual de cada métrica.
>
> Regra de honestidade: uma linha só recebe número quando existe artefato
> reproduzível. Enquanto não existir, a linha diz `sem medição` — nunca uma
> estimativa.

---

## 1. Métricas do MVP

Quatro métricas decidem não-regressão no degrau `pilot`.

| # | Métrica | Definição | Fonte |
|---|---|---|---|
| M1 | Exact match held-out | Fração de respostas exatamente corretas na suíte fixa | `benchmarks/nsos_eval_suite.jsonl` |
| M2 | Teacher-token accuracy | Fração de tokens de resposta previstos corretamente sob teacher forcing | mesma suíte |
| M3 | Perplexidade held-out | Perplexidade em texto não visto | `eval/benchmarks/wikitext.py` (EN); `ptbr_perplexity.py` (PT-BR) |
| M4 | Throughput de decode | tokens/s de geração no dispositivo alvo | `scripts/benchmark_gate.py` |

Uma release candidate precisa de limiar fixo nas quatro. Um champion de
pesquisa pode ser escolhido por score composto.

---

## 2. Estado atual

### 2.1 Runtime — medido

| Item | Valor | Artefato |
|---|---|---|
| Prompt throughput (CPU, `mamba_small`) | **8.117 tok/s** | `scripts/benchmark_gate.py`, mínimo imposto 1,0 |
| Decode throughput (CPU, `mamba_small`) | **306,6 tok/s** | idem, mínimo imposto 0,1 |
| Determinismo save/reload | **aprovado** | `reload_deterministic: true` no relatório do gate |
| Treino GPU (RX 7600, pilot 71M, fp32) | **1.579,8 tok/s p50** | `docs/benchmarks/RX7600_PERF_CAMPAIGN_2026-08-06.md` |
| Treino GPU (RX 7600, pilot 71M, bf16) | **2.485,6 tok/s p50** | idem, +57,3% |
| Treino GPU (RX 7600, pilot 71M, fp16) | **2.624,6 tok/s p50** | idem, +66,1% |
| Estabilidade em soak de 10.000 passos | **aprovado** | 0 loss não-finita, 0 checkpoint inválido, 0 falha de pool |

### 2.2 Corpus — medido

| Item | Valor | Artefato |
|---|---|---|
| Vazamento train/eval | **0,0000%** nas três fases | `scripts/corpus_quality_report.py` |
| Duplicatas exatas | 0,00% | idem |
| Quase-duplicatas | 0,00% / 1,10% / 1,38% (base/cont/sft) | idem |
| Boilerplate não filtrado (base) | **13,45% dos documentos** | idem |
| E-mails residuais (base) | **4,20% dos documentos** | idem — questão de PII |
| Diversidade de fontes | **1 fonte (100% HPLT 2.0)** | idem |

Detalhe em `docs/DATASETS.md`.

### 2.3 Qualidade do modelo — sem medição

| Métrica | Estado | Bloqueador |
|---|---|---|
| M1 exact match (PT-BR) | **sem medição** | Falta campanha reproduzível de modelo treinado na suíte fixa |
| M2 teacher-token accuracy (PT-BR) | **sem medição** | idem |
| M3 perplexidade (PT-BR) | **sem medição** | Harness PT-BR implementado; falta avaliação válida de checkpoint treinado |
| M4 decode throughput | medido acima | — |
| Modelo treinado até o fim | **não existe** | Execução mais longa até hoje: 10.000 passos ≈ 5% da receita `pilot` |

**Nenhuma alegação de qualidade do chatbot PT-BR pode ser feita hoje.** Não é
uma questão de o número ser ruim; é que ele não existe.

Nota de integridade (2026-09-20): os relatórios históricos `dummy/uniform-random`
com perplexidade PT-BR primária igual a zero são inválidos. O contrato `ppl` /
`perplexity` foi corrigido; esses artefatos não são medidas de qualidade de
modelo treinado. Ver `P0_CORRECTIONS.md`. ASSIN2, ENEM e perplexidade PT-BR já
possuem módulos; existência de módulo não equivale a campanha de qualidade.

---

## 3. Limiares de release

Preenchidos quando houver a primeira execução completa de `pilot` com suíte
PT-BR. Até lá permanecem deliberadamente vazios para não legitimar um número
inventado.

| Métrica | Limiar `pilot` | Limiar `small` |
|---|---|---|
| M1 | a definir | a definir |
| M2 | a definir | a definir |
| M3 | a definir | a definir |
| M4 | ≥ 0,1 tok/s decode (já imposto) | idem |

---

## 4. Como reproduzir

```bash
# runtime
python scripts/benchmark_gate.py --build-dir <build> --device cpu \
  --profile mamba_small --max-tokens 8 --report-path gate.json

# corpus
python scripts/corpus_quality_report.py --preset pilot --sample 12000 \
  --json corpus_quality_report.json

# treino GPU (ver docs/benchmarks/RX7600_PERF_CAMPAIGN_2026-08-06.md)
NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD=1 NSOS_MAMBA_BACKWARD_CHUNK_SIZE=32 \
  python scripts/train_ptbr_conversational.py train --preset pilot \
    --device gpu --matmul-precision bf16 --max-train-steps 1000
```

---

## 5. Caminho até um scorecard completo

1. Executar a suíte PT-BR existente contra checkpoint treinado, mantendo os
   contratos de métricas e a identidade dos artefatos (`P0_CORRECTIONS.md`).
2. Corrigir e re-preparar o corpus (PII de e-mail e URLs — ver `DATASETS.md`).
3. Executar `pilot` completo (~11 h em bf16 na RX 7600).
4. Preencher M1–M3 com números reais e fixar limiares.
5. Repetir para estabelecer variância entre seeds antes de chamar de release.
