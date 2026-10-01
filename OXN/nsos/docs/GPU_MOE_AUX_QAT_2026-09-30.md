# MoE: objetivo auxiliar no dispositivo e QAT de experts ativos

## Implementação

O objetivo Switch é preservado: `coef * E * sum(f_e * P_e)`, sobre todas as
linhas válidas do step. Não foi substituído por balanceamento heurístico ou
por médias de objetivos calculados separadamente por microbatch.

- `MoERouter::accumulate_switch_aux_grad_device` retorna a loss como tensor
  escalar no dispositivo. O trainer agrega os escalares e materializa somente
  a loss auxiliar total. Wrappers float continuam disponíveis para callers
  que deliberadamente precisam de um valor no host.
- Cargas acumuladas são retidas na GPU, tanto no roteamento ordered quanto
  no caminho atomic. `materialize_expert_loads` é a fronteira explícita de
  telemetry. `finalize_aux_accumulation()` sem argumento mantém o contrato
  público de materialização imediata.
- A redução determinística Switch tem um owner por expert, preservando a ordem
  das adições por linha. Estatísticas/gradientes completamente escritos não
  recebem zeramento redundante. Agregações têm checks de limite de indexing.
- A antiga ramificação de imbalance foi removida do trainer: era código morto,
  não uma transferência ativa. Não apresentar sua remoção como ganho medido.
- QAT grouped é preparado após o roteamento; offsets no dispositivo excluem
  experts vazios. Partials FP32 tree256, até 4096 blocos por expert, são somados
  em ordem fixa; quantização usa a regra ternária/epsilon existente.
- Destinos de pesos efetivos e escalas são workspaces reutilizáveis, nunca
  pesos mestres. A troca referência -> QAT aloca destino distinto. Expert ativo
  é re-preparado a cada forward, sem cache aritmético de versão antiga.
- Descritores só são enviados quando ponteiros ou metadados mudam. Um envio
  necessário ainda é bloqueante: este lote não certifica HIP Graph de treino.
- Política nativa grouped é `device_segmented_tile16_active_qat_v2`; a ordem
  de redução QAT mudou, então checkpoints/binários v1 não são intercambiáveis.
  Python verifica a identidade do código nativo antes do primeiro step.

## Validação final, depois da implementação

Builds Release CPU e HIP/gfx1102 completos. Suítes:

- CPU: **60/60**, `cpu-full-final.log`, 67,60 s.
- HIP: **119/119**, `hip-full-corrected-final.log`, 233,37 s.
- Python de políticas/probe: **13 casos**; também integra o CTest.
- Novos testes exercitam loss/VJP sem D2H antes da agregação, acumulação de
  máscaras heterogêneas, cargas finais sob demanda, masks vazias, cancelamento,
  coeficiente NaN e 65 experts com empates estáveis nos dois modos de redução.
- QAT: fixture independente com 1041 elementos, referência ativa e expert
  vazio com pesos NaN; destinos inativos/referência permanecem intactos;
  repetição produz resultado idêntico. Troca referência -> QAT não faz alias
  do peso mestre. A paridade grouped cobre FP32/BF16/FP16, referência/QAT,
  RMS/magnitude versus linear exato, bias, padding e caudas.
- Checkpoint grouped continua comparando loss, pesos e momentos bitwise em
  quatro steps, incluindo retomada e rejeição de política incompatível sem
  mutação. Esse braço de checkpoint tem QAT desligado; não alegar uma campanha
  integral de checkpoint QAT por causa dele.
- O binário v1 real preservado foi rejeitado antes de samples de treino,
  com `native_verified=false`, em `stale-native-rejected.json`.

A primeira execução de checkpoint exigia o contrato antigo de transferências.
O probe confirmou exatamente seis chamadas/48 bytes, antes de atualizar a
asserção de sete chamadas/64 bytes. O teste continua exigindo igualdade exata,
não um intervalo permissivo. Logs iniciais e a falha de compilação do helper
de diagnóstico foram preservados; a suíte completa foi repetida após corrigir.

## Medição e decisão

RX 7600, gfx1102, wave32, TheRock/Windows. Mesma seed, entradas, pesos iniciais,
arquitetura e política de precisão em cada trio. Duas camadas Mamba faithful,
uma MoE E4/k2/H=D, vocab257; IDs sintéticos, sem tokenizer, QAT desligado.
29 steps/processo, primeiro excluído da mediana. Segunda série inverte a ordem
dos braços. Não são medidas de qualidade linguística ou do modelo completo 71M.

| Geometria | Ordered antes, série 1 / repetição (ms) | Ordered depois (ms) | Grouped depois (ms) |
|---|---:|---:|---:|
| D128/S32/B1 FP32 | 12,09 / 12,32 | 12,15 / 11,17 | 11,77 / 11,42 |
| D128/S256/B2 BF16 | 18,95 / 20,11 | 17,74 / 18,41 | 18,11 / 17,90 |
| D768/S512/B1 BF16 | 52,93 / 53,69 | 51,04 / 52,09 | 54,71 / 54,77 |

Ordered antes/depois: hashes dos pesos finais iguais em todas as três
geometrias, em ambas as séries; na repetição, diferença de loss zero em todos
os 29 steps. Isso não é equivalência bitwise entre ordered e grouped, que têm
algoritmos GEMM diferentes.

Contadores sustentados por step nas duas séries:

- Ordered: D2H **7 -> 6 calls**, **60 -> 44 bytes**. Stream sync B1 **8 -> 7**,
  B2 **6 -> 5**. O vetor de quatro cargas saiu do caminho obrigatório.
- Grouped v2: **6 D2H calls / 48 bytes**, stream sync B1 **8**, B2 **6**.
  Inclui a leitura tardia de `(E+2)` offsets inteiros para registrar apenas
  gradientes ativos; isso NÃO foi eliminado ou escondido.
- Grouped mantém D2D reduzido, mas o caso largo ainda perde para ordered.
  Permanece opt-in/default desligado, sem promoção automática por estas sondas.
- Há variabilidade Windows; o caso curto não mostra ganho consistente.
  As duas séries sugerem melhora ordered no B2 e D768, mas não são certificação
  de throughput. O ganho QAT não pode ser inferido de probes com QAT desligado.

Não executar outro workload GPU durante a medição. A primeira tentativa de
carregar o baseline preservado não tinha a árvore Tensile junto ao DLL e
abortou antes de gravar resultado. A referência agora tem junction `rocblas`
para a biblioteca do build, compartilhada nos dois braços. Isso não é um
baseline autocontido: preservar também a árvore ROCm ao exportar os artefatos.

## Identidade e reprodução

Artefatos: `OXN/nsos/artifacts/gpu_moe_aux_qat_20260930`.

- Baseline HIP SHA256: `7410fa8d18f0007d97b799ebf8d13b850ae8f5c970804c875bfb2e8c2688a473`.
- HIP atual SHA256: `996d66cb7b4b6d35fb11342d215d3c66acc55cba092ec8d2e6a9eccec4eb3329`.
- CPU atual SHA256: `a90f2f81a15c7db944fcc9084c1a8582ef89ac66886aa73e20942eaa9cff412d`.

Cada JSON registra identidade nativa, configuração, hashes de scripts/binário,
pesos iniciais/finais, samples e contadores. Não sobrescrever arquivos existentes.

```powershell
python OXN/nsos/scripts/audit_training_integrations.py --build-dir OXN/nsos/build-gm-hip --variant moe --redesign --grouped-moe --d-model 768 --seq-len 512 --batch-size 1 --precision bf16 --steps 29 --output <novo-arquivo.json>
```

Omitir `--grouped-moe` para ordered; usar o diretório `reference-before` para
o controle anterior ordered. Todos os processos devem ser independentes.

## Pendências do objetivo integral

GEMM largo/WMMA, seleção por geometria e registro de atividade/Adam no dispositivo
continuam abertos. O inteiro `(E+2)` tardio ainda é necessário para preservar
ausência de gradiente/momentos/decay em experts vazios. Não é permitido substituí-lo
por registro de gradientes zero para todos.

Mamba-3, demais camadas, isolamento/concorrência, qualidade PT-BR, checker de
memória GPU, promoção de produto e superioridade ao SOTA não foram concluídos
neste lote. O inventário integral está em `GPU_RELEASE_SCOPE_2026-09-30.md`.
