# NSOS — impacto da última camada Mamba (2026-07-28)

## Pergunta

Forçar a última camada do NSOS híbrido faithful para Mamba ainda melhora o
treino depois da correção de raiz do Attention por LayerScale treinável com
gamma inicial `0,01`?

## Alteração validada

A proteção histórica foi restaurada em `src/jamba.cpp`:

```cpp
if (use_attention && model_config_.mamba2_faithful &&
    layer_one_based == num_layers) {
    const char* g = std::getenv("NSOS_ATTN_LAST_GUARD");
    if (g == nullptr || g[0] != '0') {
        use_attention = false;
    }
}
```

Com quatro camadas, `attention_period=2` e `attention_slot=1`, a pilha efetiva
muda de:

- proteção desativada: `Mamba → Attention → Mamba → Attention`;
- proteção ativa: `Mamba → Attention → Mamba → Mamba`.

O override `NSOS_ATTN_LAST_GUARD=0` continua disponível para A/B.

Foi acrescentado um teste de regressão em `tests/test_jamba.cpp`, cobrindo:

1. proteção faithful ativa por padrão;
2. override `NSOS_ATTN_LAST_GUARD=0`;
3. ausência da proteção no modo não-faithful.

Validação direcionada: 5/5 testes passaram via CTest:
`test_python_binding_smoke`, `test_jamba`,
`test_gpu_parity_decode_incremental`, `test_gpu_parity_jamba` e
`test_gpu_parity_jamba_sparse`.

## Protocolo controlado

- Somente o NSOS híbrido corrigido foi treinado novamente.
- Dataset: `facebook/babi_qa`, `en-10k-qa1`.
- Revisão: `1d86ad39d1c3ea2ff4b77eeb85f7c6ebd622a95f`.
- Hash de treino:
  `f1d67aa230d7aba0ed310df0d696a3ba9a07270e1670fe64c6901c24e5016f3f`.
- Hash de teste:
  `9875dfad271cbfa5c49adb5809dd67be3826394a5d4d66dc74cab0c81483e2f8`.
- Treino/teste: 10.000/1.000 exemplos.
- Seed do modelo: 11; seed de amostragem: 200011.
- 1.500 passos, batch 32, learning rate `0,002`.
- Quatro camadas, `d_model=128`, Mamba-2 faithful e Attention exata.
- MoE, KAN, TTT, CHRASS, Slender, QAT e OxtaMem desligados.
- AMD Radeon RX 7600 `gfx1102`, HIP estrito, matmul FP32.

## Resultado A/B

| Configuração | Pilha efetiva | Parâmetros | Acurácia | Objetivo final |
|---|---|---:|---:|---:|
| Sem proteção, resultado anterior | `M-A-M-A` | 613.144 | **68,8%** | 2,0556 |
| Proteção restaurada | `M-A-M-M` | 546.980 | **50,3%** | 2,8218 |
| Diferença | — | -66.164 (-10,8%) | **-18,5 p.p.** | +0,7662 |

A curva de perda média aproximada do modelo corrigido foi:

| Passo | Perda |
|---:|---:|
| 300 | 3,8180 |
| 600 | 2,8835 |
| 900 | 2,6563 |
| 1.200 | 2,5863 |
| 1.500 | 2,5044 |

O treino terminou normalmente, sem NaN, fallback ou erro de runtime. A proteção
não causou stall; ela reduziu a qualidade no holdout.

## Velocidade observada

O modelo corrigido treinou em 671,93 s, ou 71,44 exemplos/s. O resultado
anterior registrou 994,86 s e 48,25 exemplos/s, mas aquela campanha sofreu
concorrência de outro processo. Portanto, a nova medição prova que a execução
corrigida é funcional e rápida, mas não permite atribuir com rigor uma melhora
de 48% à troca da camada.

A redução de 10,8% nos parâmetros é real e esperada porque uma camada Attention
foi substituída por Mamba nesta configuração.

## Conclusão

Neste A/B de mesma seed, a última camada Attention **não estava prejudicando**
o modelo após a introdução do LayerScale `0,01`. Ao contrário: substituí-la por
Mamba reduziu a acurácia de 68,8% para 50,3%.

A antiga proteção tratava o stall observado antes da correção de raiz do
Attention. Com o LayerScale atual, mantê-la ativa por padrão parece
conservador demais e prejudicial nesta tarefa. O NSOS Mamba-only continua sendo
o melhor braço treinado do zero. Uma execução posterior atingiu 99,9% model-only
e 100,0% com OxtaMem; consulte
`docs/benchmarks/nsos_mamba_oxtamem_rx7600_2026-07-28.md`. O híbrido corrigido
não o superou.

Limitação: este é um A/B determinístico de uma seed. Antes de alterar novamente
o padrão de produção, a confirmação recomendada é executar pelo menos seeds
11, 12 e 13 para as duas pilhas. O resultado atual já é evidência forte de que
a proteção deve permanecer como flag de diagnóstico, e não ser presumida como
melhoria de qualidade.

## Artefatos

- Resultado:
  `artifacts/oxta_contabil_amd/benchmark/rx7600_hybrid_last_guard_seed11_1500.json`.
- SHA-256 do resultado:
  `11e2d015e3e36e8d3e0136e8fa4264fcf4623004d6b14bf1d8cb0f84bad02aa3`.
- Baseline anterior:
  `artifacts/oxta_contabil_amd/benchmark/rx7600_product_architecture_babi.partial.json`.
- Script:
  `scripts/oxta_contabil/benchmark_product_architecture.py`.
