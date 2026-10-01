# TTT: derivada completa por sequência, isolamento e memória compacta

## Resultado e escopo

O TTT agora possui um contrato opt-in que diferencia a recorrência interna
através de todos os tokens válidos de **um forward**, incluindo reconstrução,
clipping e momentum. Isso fecha a lacuna entre paridade do backward truncado
e uma derivada da adaptação. Não certifica qualidade linguística, superioridade
ao SOTA nem BPTT ilimitado entre chamadas. GPU continua Experimental e TTT
continua Research em `PRODUCT.md`.

O lote anterior de Mamba/otimizador/MoE/attention e suas medições permanecem em
[GPU_TRAINING_REDESIGN_2026-09-29.md](GPU_TRAINING_REDESIGN_2026-09-29.md).
Os artefatos deste lote ficam em `../artifacts/gpu_ttt_full_20260929/`.

## Contratos e integração

| Entrada no modo completo de treino | Estado inicial | Commit | Padding |
|---|---|---|---|
| Rank-2 `[S,D]` | Clone do estado de sessão; adjunto inicial destacado | Commit do estado final da sequência | Prefixo válido, se informado |
| Rank-3 `[B,S,D]`, inclusive B=1 | Zero independente por amostra | Não lê nem altera estado de serving | Prefixos individuais; saída e gradiente zero no padding |

Rank-3 não achata mais amostras em uma única recorrência no **novo contrato**.
O contrato legado permanece disponível e não foi reescrito silenciosamente.
Isolamento de treino não equivale a certificação de serving concorrente.

`JambaBlock::set_batch_valid_lengths` propaga os comprimentos ao TTT. Os
parâmetros e o contrato selecionado são capturados no forward: alterações
posteriores de flags, clipping, friction ou Hamiltonian não trocam seu VJP.
O backward completo consome/libera seus caches; um segundo backward sem novo
forward é rejeitado. Reset, restauração de sessão e mudanças de modo invalidam
o backward pendente. Snapshots exigem ambas as matrizes com geometria correta;
geometria inválida é rejeitada antes de substituir o estado.

Dimensões, multiplicações de tamanho, coeficientes não finitos e overflow do
passo de adaptação são verificados. Isso não é uma varredura universal de
finitude de todos os snapshots externos nem um memory checker de GPU.

## Derivada implementada

Com key `k`, base `w_out(w_v(x))`, adaptação `A` e momentum `M`:

```text
y = base + k A
e = y - x
s = max_norm / (norm(e) + 1e-6) quando clipping ativo; senão 1
G = (k outer e) s

ordinário: M' = decay M + G; A' = decay A - step G
Hamiltonian: M' = decay M + (1-decay) G
             A' = A - step (G + 0.25 M')
```

O reverse passa os adjuntos futuros de A/M entre tokens e chunks. Propaga
ambos os fatores de `k outer e`, o coeficiente de clipping, o termo direto
`-x` da reconstrução e a sensibilidade de `k A`. As três projeções recebem
seus gradientes uma vez por tensor, sem backward separado por token.
CPU usa adjuntos double como referência; GPU usa adjuntos FP32 e reduções
FP64 da norma/dot. Não se declara equivalência bitwise geral entre backends.
QAT/BitLinear STE continua um contrato próprio, não uma derivada exata da
quantização discreta. A borda de clipping é uma função por partes.

## Memória e desenho GPU

São guardadas fronteiras de **A e M** a cada 32 tokens e os erros originais.
O backward reconstrói A/M dentro do chunk usando keys/erros salvos, não pesos
atualizados ou erros recalculados. Um cache local de até 32 estados de A evita
guardar uma matriz por token. Adjuntos sobrevivem entre chunks, não entre
amostras. Não existe download da norma por token no novo caminho.

Com `cells=H*D`, o histórico retido usa
`2 * B * ceil(S/32) * cells * sizeof(float)`.
Em S=512, são 32 estados em vez dos 512 do legado: **16× menos histórico
retido**, não 16× menos VRAM total. Há também cache local e scratch.
Para H=3072/D=768/B=1/S=512: 288 MiB retidos, até 288 MiB de cache local e
36 MiB nas quatro matrizes de scratch/adjuntos. Essa conta exclui projeções,
keys, erros, parâmetros, gradientes e otimizador.

O primeiro backward completo ainda calculava q em um único CTA. A implementação
final distribui colunas em CTAs de 32 threads quando **H>=256 e D>=256**;
cada dono soma hidden na mesma ordem. Um kernel seguinte aplica a mesma árvore
FP64 da norma/dot. Não há atomics, nova árvore no eixo hidden ou novo buffer
de partials. Estados menores conservam a geometria de um único launch.
Essa seleção não resolve o forward single-CTA nem o custo de replay e launches.

## Seleção e identidade

`NSOS_TTT_FULL_BPTT=1` seleciona o contrato completo; somente `0`/`1` são
aceitos. Default off. O CLI semântico é
`--ttt-gradient-policy full-sequence-v1`, separadamente de
`--gpu-training-profile redesign-v1`; exige `use_ttt` explicitamente ativo,
não altera a arquitetura por conta própria. `truncated` desliga o contrato.

A identidade nativa final é
`ttt.training_policy=full_sequence_bptt_isolated_boundary32_column_q_v2`.
O seletor CLI versiona a semântica; a identidade também versiona o desenho
concreto do kernel. Resume sob outra identidade é recusado. `redesign-v1`
sozinho não promove a nova derivada, nem transforma TTT em capacidade oficial.

O Python agora verifica as políticas solicitadas contra a identidade do
Trainer antes do primeiro step. O arquivo de política começa com
`native_verified=false` e só passa a verdadeiro depois dessa conferência.
Um binário antigo que ignore flags ou implemente outra versão é rejeitado;
ambiente solicitado não é evidência de execução.

## Arquivos responsáveis

| Arquivo | Responsabilidade |
|---|---|
| `include/ttt_recurrence.h`, `src/ttt_recurrence.cpp` | Contrato, referência CPU, replay e workspaces GPU |
| `include/ttt_layer.h`, `src/ttt_layer.cpp` | Seleção, caches, sessões, comprimentos válidos, validações e projeções |
| `src/cuda/kernels.cu`, `include/cuda/kernels.cuh` | Primal GPU, fronteiras, replay, clipping VJP, q e adjuntos de estado |
| `src/jamba.cpp` | Encaminhamento dos comprimentos de batch |
| `include/training_runtime_policy.h`, `src/runtime_execution_identity.cpp` | Parsing e compatibilidade de continuação |
| `scripts/train_ptbr_conversational.py` | Seleção explícita e conferência do binário nativo |
| `scripts/audit_training_integrations.py` | Probes por largura/batch, identidade e hashes dos pesos |
| `tests/test_ttt_full_bptt.cpp` | Primal independente double e finite differences |
| `tests/gpu/test_gpu_parity_ttt.cpp`, `test_gpu_checkpoint_continuation.cpp` | Paridade, caudas, padding e continuação/rejeição sem commit |

## Validação após implementação

Builds Release clang/Ninja, OxtaMem ON; Windows/TheRock, RX 7600 gfx1102,
wave32. NVIDIA e gfx1103 não foram executados.

- HIP final SHA: `01afcd8d02085a7180787c2d35179d3ea38b9b66fc5fcce7b0ff9354540145c1`.
- CPU final SHA: `459324bd6b866def9337e21c031ff5e42bb9178b60a3e9816cdf8b956d0472a4`.
- Primeiro completo/reference-q HIP, preservado em `reference-full-v1/`:
  `18b0805b8189d2c559db4213327367b5dfd6089e8cdab876b32327646fbbf4ed`.

| Gate | Resultado / evidência |
|---|---|
| CPU nativo final | 60/60, `validation-cpu-column-q-full.log` |
| HIP nativo final | 117/117, `validation-hip-column-q-full.log` |
| Contratos Python após conferência nativa | 3/3 em cada build, `validation-native-policy-python-*.log`; probe unitário contém 12 casos |
| Finite differences | 24 combinações: exact-linear/RMS-magnitude, ordinário/Hamiltonian, clip/sem clip, S=3/33/65; todos os inputs e parâmetros treináveis |
| Tolerância FD | atol 3e-4 + rtol 2e-3; primal double independente com tolerância 2e-5 + 1e-5 |
| Isolamento | B=3, comprimentos 33/29/0, estado de serving não zero, permutação e soma dos gradientes por amostra |
| GPU completo | Paridade output/dInput/parâmetros; caudas 257×257, S=33; batch 65/61/0; consumo do histórico |
| Checkpoint TTT | Loss/pesos/momentos/continuação determinística exatos no teste; gradiente de tarefa não zero |
| Política incompatível | Full→truncado recusado sem commit de pesos, momentos, LR, passo ou RNG |
| Binário obsoleto real | `stale-native-rejected.json`: falha **esperada**, antes de qualquer step; não é falha de um treino lançado |
| Integrações finais | 4 probes D=128/S=256 (FP32/BF16, B=1/2) e 2 probes D=768/S=512/BF16; todos finite, ramos com gradientes não zero |

O teste FD usa um primal próprio, sem Tensor ops, helper de recorrência ou VJP
da implementação. Também distingue explicitamente a derivada nova da truncada;
uma referência que repetisse o erro antigo não fecharia o gate.

## Medições e ressalvas

Tokens sintéticos, sem tokenizer/dataset linguístico. QAT off, pesos mestres,
gradientes e estado FP32; GEMMs conforme a coluna precisão. Dois blocos: um
TTT e um Mamba faithful, vocab=257, deterministic/strict GPU. Os JSONs finais
registram configurações, SHA do binário/caller, pesos iniciais/finais e caminhos
ativos. Tempo exclui o primeiro step; contadores são capturados antes da
auditoria CPU de gradientes/pesos.

| Probe final | ms/step mediana | Pico live runtime MiB | Reservado MiB |
|---|---:|---:|---:|
| D128/S256/B1 FP32 | 50,67 | 41,5 | 51,25 |
| D128/S256/B1 BF16 | 49,79 | 41,5 | 51,25 |
| D128/S256/B2 FP32, SFT heterogêneo | 90,27 | 64,25 | 77,5 |
| D128/S256/B2 BF16, SFT heterogêneo | 88,67 | 64,25 | 77,5 |
| D768/S512/B1 BF16 | 1672,53 | 1064,25 | 1195,4375 |
| Repetição D768/S512/B1 BF16 | 1742,43 | 1064,25 | 1195,4375 |

Os pequenos probes têm 9 steps (8 medidos), os grandes 5 (4 medidos).
Arquivos `ttt-full-column-q-*.json`. D128 registra 486.668 elementos; D768,
15.711.688. Não são modelos 40M/150M nem a topologia de 16 camadas do relatório
anterior. B1 exercita causal training; B2, SFT com respostas mascaradas.

O completo com q referência mediu 2562,01 ms em D768/S512. Distribuir colunas
mediu ganho curto de **1,47–1,53×**, com mesma memória. Pesos iniciais:
`db71d6d075783ca50fb55f82131ea21d2c45128395e7afac7ab0240e47414a74`.
Pesos finais do reference-q e dos dois column-q:
`4cf79d6699784010e287b3454c436fcbd5f51a99929f960c037051b9bae357c5`.
As cinco losses também coincidem exatamente nesse probe, não em todos os casos
possíveis. A primeira medição column-q teve validação Python concorrente em
parte do intervalo; a repetição não. Não há intervalo de confiança, isolamento
completo da máquina nem contadores físicos de cache/occupancy.

O legado truncado device-only mediu pico de 5156,5 MiB na mesma geometria,
contra 1064,25 MiB no completo compacto (~79,4% menos pico **do runtime**).
Sua mediana de 10510,23 ms não é tratada como speedup causal: muda a derivada,
retém muito mais memória e pode sofrer pressão WDDM. Pesos finais diferem.
Em D128/S256, o truncado mediu 30,80 ms contra ~50 ms no completo: derivada
mais completa **custa mais** nesse caso. Não esconder esse trade-off.

Todos os probes finais ainda têm 4 D2H/20 bytes e 4 stream-syncs por step,
pelos gates escalares existentes. Zero device-wide sync não significa zero
sincronização. Pico do allocator não inclui todo o consumo do driver/BLAS/WDDM.

## Lacunas que continuam abertas

- Não há eval linguística que demonstre benefício do TTT completo por token,
  segundo ou VRAM. Manter opt-in e comparar ablações com mesmo orçamento.
- O forward TTT ainda concentra a projeção de estado num CTA; replay e reverse
  continuam com múltiplos launches por token. Não foi implementado monokernel,
  captura de treino completo ou concorrência de sessões certificada.
- Estado inicial é destacado; não há BPTT contínuo entre forwards.
- Grouped GEMM MoE device-side, downloads residuais KAN, seleção por geometria
  dos outros redesenhos e validação multi-GPU/checker continuam pendentes.
- Piloto real continua parado no gate SFT: structured=18.275 no último estado
  do workspace. Não se baixou a quota nem se reutilizou holdout para iniciar treino.
- Docker/CI Linux/build limpo independente e gates de qualidade da v1 não foram
  fechados neste lote. A implementação integral do projeto não está concluída.
