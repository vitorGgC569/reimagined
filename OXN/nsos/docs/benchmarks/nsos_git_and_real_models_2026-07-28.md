# NSOS — auditoria Mamba+Attention e modelos reais (2026-07-28)

## Resumo executivo

- O branch local `codex-gpu-gold-validation` e o branch remoto homônimo estão
  no mesmo commit: `98068282...` (`0` commits à frente e `0` atrás após
  `git fetch origin --prune --tags`).
- Este clone não expõe outros branches locais ou remotos. Logo, não existe
  outro branch disponível aqui do qual trazer uma correção.
- A correção principal do híbrido está no `HEAD`: LayerScale treinável nos
  blocos de Attention faithful, habilitado por padrão e iniciado em `0,01`.
- A proteção adicional que forçava a última camada a ser Mamba foi adicionada
  em `33c7376`, mas desapareceu em `5b4d39d`. Ela foi restaurada no worktree
  após esta auditoria, com teste de regressão e novo treino controlado.
- O diretório de trabalho não está limpo: há muitas alterações não commitadas.
  “Sincronizado com o remoto” descreve o commit do branch, não significa que
  os arquivos locais sejam idênticos ao `HEAD`.

## Histórico da correção Mamba+Attention

Os commits relevantes são ancestrais do `HEAD` atual:

| Commit | Papel |
|---|---|
| `c164007` | alinhamento do bloco/wrapper com Mamba-2 oficial |
| `33c7376` | proteção experimental: última camada sempre Mamba |
| `343679b` | identificação do residual scale como causa do stall |
| `a6601c4` | LayerScale treinável como correção de raiz |
| `1ef0f9f` | inicialização de gamma alterada de `0,1` para `0,01` |
| `5b4d39d` | sincronização ampla que removeu a proteção adicional |

O código atual ainda contém a correção de raiz:

```cpp
use_attn_layerscale_ =
    mamba2_faithful && is_attn && (e == nullptr || e[0] != '0');
float ls_init = 0.01f;
```

Agora há um teste dedicado em `tests/test_jamba.cpp` para a proteção ativa, o
override `NSOS_ATTN_LAST_GUARD=0` e o modo não-faithful. O novo treino mostrou
que, com LayerScale `0,01`, forçar a última camada para Mamba reduziu a
acurácia de 68,8% para 50,3% na seed 11. Consulte
`docs/benchmarks/nsos_last_layer_guard_ab_2026-07-28.md`.

## Pesos aleatórios: o que realmente venceu

“Pesos aleatórios” descreve somente a inicialização. Os números abaixo são
medidos **depois do treino do zero**, não antes dele.

| Arquitetura treinada do zero | Acurácia |
|---|---:|
| NSOS Mamba-only, 1.500 passos | **90,9% na primeira execução; 99,9% na repetição** |
| NSOS Mamba+Attention, 1.500 passos | **68,8%** |
| NSOS Attention-only, 1.500 passos | 49,7% |
| Jamba dense, protocolo histórico T4 | 49,3% |
| Jamba-MoE, protocolo histórico T4 | 27,1% |

Conclusão restrita ao experimento: o NSOS Mamba-only foi o vencedor, e o
híbrido NSOS superou as variantes Jamba iniciadas aleatoriamente. O híbrido
não superou o próprio Mamba-only. Antes do treino, todos estavam em nível de
acaso. Como os resultados Jamba vêm do protocolo histórico T4, o comparativo
deve ser repetido numa única campanha isolada, com os mesmos dados, número de
passos, orçamento de parâmetros e múltiplas seeds.

A repetição Mamba-only atingiu 99,9% model-only e 100,0% com OxtaMem. Como as
reduções HIP deste benchmark não são determinísticas, a diferença para 90,9%
deve ser tratada como variação de execução até existir uma campanha multisseed.

Isso mede treinabilidade da arquitetura em uma tarefa sintética. Não demonstra
que um NSOS sem pré-treino seja melhor que um modelo público pré-treinado.

## Comparativo com checkpoints públicos reais

### Protocolo

- Dataset: 60 perguntas determinísticas do BR-TaxQA-R.
- Para cada pergunta, a resposta correta foi comparada com três distratores
  difíceis, e cada par foi avaliado nas duas ordens.
- Total: 360 decisões por modelo; posição correta exatamente balanceada.
- Decodificação: temperatura `0`, seed fixa, resposta exata `1` ou `2`, sem
  LLM avaliador.
- Checkpoints públicos quantizados em `Q4_K_M`, executados pelo Ollama local.
- Baseline aleatório: 50%. Baseline lexical Jaccard: 83,61%.

O primeiro ensaio de múltipla escolha com quatro alternativas foi rejeitado:
os modelos exibiram forte viés pela alternativa A. O protocolo pareado
balanceado abaixo controla esse viés.

### Resultado

| Checkpoint real | Parâmetros | Acurácia | IC Wilson 95% | Escolheu posição 1 | Pares vencidos nas duas ordens | VRAM carregada | Latência média |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2 Instruct Q4 | 134,52M | 36,94% | 32,12–42,04% | 43,06% | 10,0% | 205,3 MiB | 129 ms |
| Qwen2.5 Instruct Q4 | 494,03M | 50,00% | 44,86–55,14% | 100,00% | 0,0% | 459,5 MiB | 228 ms |
| Falcon-H1 Instruct Q4 | 521M | 48,61% | 43,49–53,76% | 80,28% | 15,0% | 407,2 MiB | 329 ms |
| Baseline lexical | — | **83,61%** | 79,44–87,08% | 50,83% | — | — | — |

O Falcon-H1 é especialmente relevante porque é um checkpoint híbrido
Transformer+Mamba real. Neste teste zero-shot em português tributário, ele não
ficou significativamente acima do acaso. Isso não invalida a família híbrida:
o modelo é pequeno, voltado principalmente a inglês, quantizado e não recebeu
adaptação ao domínio contábil brasileiro.

O Qwen2.5 marcou exatamente 50%, mas escolheu sempre a primeira posição; logo,
o número é viés posicional puro, não conhecimento. O SmolLM2 teve 81 respostas
inválidas em 360. Nenhum dos três pequenos checkpoints atingiu o baseline
lexical, o que evidencia que esta seleção de respostas do BR-TaxQA-R é
adversa para modelos gerais pequenos sem adaptação.

### Referência local de maior porte

O checkpoint local `qwen3.6:latest` é um modelo real `qwen35moe`, 36B,
`Q4_K_M`. Como seu arquivo tem 23,79 GB e somente cerca de 6,49 GB ficam na
VRAM da RX 7600, grande parte é descarregada para RAM/CPU. Ele é uma referência
de capacidade, não uma comparação de velocidade justa com modelos que cabem
inteiros na GPU.

Em uma amostra de controle menor, de cinco perguntas e 30 decisões, ele obteve
30/30 (100%; IC Wilson 95% de 88,65–100%), escolheu cada posição exatamente 50%
das vezes e venceu 100% dos pares nas duas ordens. O baseline lexical nessa
mesma subamostra fez 26/30 (86,67%). A latência média foi 8,22 s por decisão,
com aproximadamente 49,05 tokens de prompt/s e 11,30 tokens gerados/s.

O resultado mostra que o protocolo consegue distinguir um modelo com
capacidade muito maior. Não deve ser lido como placar definitivo: cinco
perguntas é uma amostra pequena, enquanto os três checkpoints pequenos foram
avaliados em 60 perguntas/360 decisões. Também não é uma opção de velocidade
comparável na RX 7600 por causa do offload.

## O que ainda falta para comparar o NSOS diretamente

O NSOS não aparece no placar dos checkpoints reais porque ainda não existe um
checkpoint NSOS pré-treinado em português/contabilidade. Colocar a acurácia
bAbI treinada do zero ao lado do BR-TaxQA-R zero-shot seria misturar tarefas e
orçamentos diferentes.

O comparativo honesto de produção requer:

1. pré-treinar ou continuar o treino do NSOS no corpus português;
2. separar treino, validação e teste do BR-TaxQA-R por hash/conteúdo;
3. adaptar NSOS e os baselines públicos com o mesmo número de tokens;
4. avaliar todos no mesmo holdout, com múltiplas seeds e intervalos de
   confiança;
5. comparar qualidade, VRAM, tokens/s, energia e tempo total de adaptação;
6. incluir Mamba-only, Mamba+Attention e Mamba+Attention+QAT para decidir se o
   custo da Attention realmente se paga.

## Reprodutibilidade

- Auditoria: branch `codex-gpu-gold-validation`, `HEAD 98068282...`.
- Script principal:
  `scripts/oxta_contabil/benchmark_real_models_pairwise.py`.
- Resultado dos três checkpoints pequenos:
  `artifacts/oxta_contabil_amd/benchmark/real_models_br_taxqa_pairwise.json`.
- Smoke test do checkpoint 36B:
  `artifacts/oxta_contabil_amd/benchmark/qwen36_pairwise_smoke2.json`.
- Controle ampliado do checkpoint 36B:
  `artifacts/oxta_contabil_amd/benchmark/qwen36_pairwise_5.json`.
- Ablation NSOS:
  `artifacts/oxta_contabil_amd/benchmark/rx7600_product_architecture_babi.partial.json`.
- Relatório AMD anterior:
  `docs/benchmarks/oxta_contabil_amd_rx7600_2026-07-28.md`.

Model cards: [Falcon-H1 0.5B Instruct](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct),
[Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) e
[SmolLM2 135M Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
