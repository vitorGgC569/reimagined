# NSOS Mamba-only + OxtaMem na RX 7600 (2026-07-28)

## Resposta executiva

O novo treino Mamba-only atingiu 99,9% no holdout bAbI QA1. Com o contexto
recuperado pelo OxtaMem, o sistema atingiu 100,0%. O OxtaMem corrigiu apenas
um exemplo adicional porque o modelo sozinho já havia acertado 999/1.000.

Esse resultado valida o sistema NSOS Mamba-only + OxtaMem nesta tarefa. Ele não
prova superioridade sobre um Mamba industrial pré-treinado.

## Protocolo

- Dataset: `facebook/babi_qa`, configuração `en-10k-qa1`.
- Revisão: `1d86ad39d1c3ea2ff4b77eeb85f7c6ebd622a95f`.
- 10.000 exemplos de treino e 1.000 de teste.
- Hash de treino:
  `f1d67aa230d7aba0ed310df0d696a3ba9a07270e1670fe64c6901c24e5016f3f`.
- Hash de teste:
  `9875dfad271cbfa5c49adb5809dd67be3826394a5d4d66dc74cab0c81483e2f8`.
- Seed do modelo: 11; seed de amostragem: 200011.
- 1.500 passos, batch 32, learning rate `0,002`.
- Quatro camadas Mamba-2 faithful, `d_model=128`, 480.816 parâmetros.
- Attention, MoE, KAN, TTT, CHRASS, Slender e QAT desligados.
- AMD Radeon RX 7600 `gfx1102`, HIP estrito e matmul FP32.
- OxtaMem ABI v2, avaliado depois do treino como memória externa.

## Curva de treino

| Passo | Perda média aproximada |
|---:|---:|
| 300 | 3,6498 |
| 600 | 2,8300 |
| 900 | 2,3683 |
| 1.200 | 0,4143 |
| 1.500 | 0,0030 |

Objetivo final registrado: `0,00224456`.

## Resultado

| Braço | Acurácia | Acertos |
|---|---:|---:|
| Mamba-only, execução anterior | 90,9% | 909/1.000 |
| Mamba-only, nova execução | **99,9%** | 999/1.000 |
| Mamba-only + OxtaMem | **100,0%** | 1.000/1.000 |

O novo treino levou 902,74 s e processou 53,17 exemplos/s. A execução anterior
levou 990,99 s e processou 48,44 exemplos/s. A nova execução foi 8,9% mais
curta e teve throughput 9,8% maior.

A diferença de acurácia entre as duas execuções Mamba-only não é efeito da
proteção da última camada: esse guard só atua quando a camada terminal seria
Attention, e Mamba-only não possui Attention. O benchmark configura reduções
GPU não determinísticas; além disso, a curva entra abruptamente no regime de
solução depois de 900 passos. O resultado demonstra alta treinabilidade, mas
requer múltiplas seeds para estimar a média real.

## OxtaMem

| Métrica | Resultado |
|---|---:|
| Retrieval@1 com chave exata | 100,0% |
| Retrieval@1 com ruído gaussiano normalizado σ=0,05 | 100,0% |
| Acurácia sistêmica com recuperação exata | 100,0% |

A memória armazena somente a sentença de contexto mais recente por entidade.
Ela não armazena a resposta nem os `supporting_ids`. Os vetores de consulta são
determinísticos por `example_id + entidade`, portanto este é um teste de
integração estruturada e robustez do armazenamento vetorial, não um benchmark
de recuperação semântica real.

O tempo `eval_seconds` do artefato cobre a inferência do modelo depois de os
prompts terem sido preparados; ele não mede separadamente latência de escrita
e busca do OxtaMem. Logo, não deve ser usado como latência end-to-end da memória.

## Comparação com “Mamba industrial”

Não existe neste protocolo um checkpoint Mamba industrial avaliado de forma
equivalente. Em `experimental/large_scale_multi_model_benchmark.py`, a classe
rotulada como `Mamba-1 / SSM` é uma recorrência NumPy simplificada:

```python
ssm_state = 0.85 * ssm_state + 0.15 * (h_in @ self.W_ssm)
```

Ela não implementa o Mamba oficial, não carrega pesos pré-treinados e não
executa o pipeline industrial de treino do Mamba. Os resultados desse script
não autorizam declarar o NSOS superior a Mamba-1/Mamba-2 de produção.

O que os dados autorizam afirmar:

- o NSOS Mamba-only é o melhor braço treinado do zero no protocolo bAbI atual;
- ele superou os baselines Jamba e híbridos usados na ablação interna;
- Mamba-only + OxtaMem resolveu 1.000/1.000 exemplos no teste estruturado;
- ainda falta um A/B com implementação oficial Mamba-2, mesmo dataset, tokens,
  parâmetros, seeds e hardware para sustentar superioridade industrial.

## Artefatos

- Resultado:
  `artifacts/oxta_contabil_amd/benchmark/rx7600_mamba_oxtamem_seed11_1500.json`.
- SHA-256:
  `2130dee5dfcff892005dd38528fb7879ec9ea64c76c4454cf033549da3d2cf78`.
- Script:
  `scripts/oxta_contabil/benchmark_product_architecture.py`.
- O script agora aceita `--oxtamem-arm mamba`, preservando
  `hybrid_qat` como padrão.

