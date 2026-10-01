# Oxta Contábil — benchmark AMD RX 7600 (2026-07-28)

## Escopo

O benchmark valida o piloto de produto com apenas Mamba, Attention, OxtaMem e
quantização ternária de 2 bits. MoE, KAN, TTT, CHRASS e Slender foram
explicitamente desligados.

Hardware principal: AMD Radeon RX 7600, `gfx1102`, 8 GB; backend HIP estrito,
matmul FP32. O ambiente também expôs a iGPU `gfx1103`, mas o processo usou o
dispositivo discreto de índice 0.

## Dados

- bAbI QA1 `en-10k-qa1`, revisão
  `1d86ad39d1c3ea2ff4b77eeb85f7c6ebd622a95f`.
- 10.000 exemplos de treino e 1.000 de teste.
- Vocabulário criado somente com treino: 25 tokens; sequência máxima: 83.
- Baseline majoritário: 18,7%.
- BR-TaxQA-R completo (185,1 MB) e 64 MB/12.968 documentos de FineWeb-2
  `por_Latn` foram baixados para a fase de produto. Eles não foram usados para
  inflar esta ablação sintética.

## Resultado equilibrado — 400 passos, seed 11

| Braço | Parâmetros | Acurácia | Exemplos/s observados |
|---|---:|---:|---:|
| Mamba-only | 480.816 | 51,4% | 30,6* |
| Attention-only | 745.472 | 48,9% | 273,2* |
| Mamba+Attention | 613.144 | 50,5% | 46,8* |
| Mamba+Attention+QAT | 613.144 | 46,5% | 46,5* |
| Híbrido QAT + OxtaMem | 613.144 | 81,8% | avaliação sistêmica |

\* As velocidades desta campanha sofreram concorrência de um processo antigo
que continuou após o encerramento do wrapper. As acurácias são válidas, mas
essas velocidades não devem ser usadas como medição isolada.

O QAT preservou 92,1% da acurácia do híbrido FP32 (queda absoluta de 4,0 pontos).
OxtaMem obteve retrieval@1 de 100% com a chave exata e com ruído gaussiano
normalizado de sigma 0,05. A acurácia sistêmica foi 81,8%. Esse braço usa
vetores determinísticos por entidade e não usa resposta nem `supporting_ids`;
é evidência de integração, não prova model-only.

## Velocidade isolada — 100 passos, batch 32

| Braço | Passos/s | Exemplos/s |
|---|---:|---:|
| Attention-only | 14,62 | 467,95 |
| Mamba+Attention | 2,45 | 78,54 |
| Mamba+Attention+QAT | 2,33 | 74,59 |

O custo do QAT sobre o híbrido foi aproximadamente 5%. O gargalo é o caminho
Mamba/HIP atual: o híbrido ficou cerca de 6 vezes mais lento que Attention no
mesmo formato.

## Execuções de 1.500 passos

Uma campanha longa foi interrompida depois de persistir resultados parciais:

| Braço | Acurácia | Observação |
|---|---:|---|
| Mamba-only | 90,9% | velocidade isolada: 48,44 exemplos/s, 990,99 s |
| Attention-only | 49,7% | velocidade contaminada por concorrência |
| Mamba+Attention | 68,8% | velocidade contaminada por concorrência |

O resultado Mamba-only mostra que 400 passos eram subtreino (51,4% para 90,9%
ao chegar a 1.500). No protocolo histórico T4, Jamba dense obteve 49,3%,
Jamba-MoE 27,1%, e o NSOS híbrido teve 68,5% ± 7,1% em três seeds (76,7% no
seed 11). As comparações históricas são úteis, mas não substituem uma nova
campanha isolada e multisseed.

## Veredito

A RX 7600 treina corretamente o piloto NSOS com HIP e QAT, e OxtaMem funciona
como biblioteca nativa de produção. Porém, o benchmark não sustenta ainda a
alegação de SOTA nem prova que o híbrido é melhor que Mamba-only: no treino
longo disponível, Mamba-only venceu o híbrido por ampla margem.

Decisão recomendada: manter Mamba+Attention+OxtaMem+QAT como hipótese de produto,
mas não congelá-la como arquitetura vencedora. Antes de produção, executar:

1. três ou mais seeds isolados com 1.500 passos por braço;
2. holdout de BR-TaxQA-R separado por hash de conteúdo;
3. tarefas reais do OContábil (extração de campos, normalização e exportação);
4. retrieval semântico não estruturado no OxtaMem;
5. otimização/perfil do kernel Mamba HIP em `gfx1102`.

## Reprodutibilidade

- Script: `scripts/oxta_contabil/benchmark_product_architecture.py`
- Notebook: `notebooks/oxta_contabil_amd_product.ipynb`
- Resultado: `artifacts/oxta_contabil_amd/benchmark/rx7600_product_architecture_ablation400.json`
- Velocidade isolada: `artifacts/oxta_contabil_amd/benchmark/rx7600_isolated_speed100.json`
- Parcial 1.500 passos: `artifacts/oxta_contabil_amd/benchmark/rx7600_product_architecture_babi.partial.json`
- Manifesto dos dados: `artifacts/oxta_contabil_amd/data/oxta_contabil/manifest.json`

