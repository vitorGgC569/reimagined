# Objetivo integral e evidência de conclusão

Este inventário preserva o objetivo em
`C:/Users/vitor/.codex/attachments/b5f73246-e92d-405f-b1d3-c44215e38702/goal-objective.md`.
Lotes intermediários não encerram o objetivo. Testes de kernels não provam
qualidade do modelo, promoção GPU ou superioridade ao SOTA.

## Diretriz posterior do usuário — prevalece sobre a ordem histórica

Desenvolvimento adicional KAN suspenso em2026-09-30. Preservar o novo caminho
opt-in, testes e regressões D768; não promover default. Retomar somente se
ablação demonstrar qualidade/custo melhor que SwiGLU/MLP.
Ordem ativa: **Mamba-3 -> Attention/FA4-inspired RDNA -> MoE/Adam device-side
-> head/loss/CCE**. Demais requisitos permanecem no inventário, sem adicionar
automaticamente as alternativas de pesquisa do texto colado ao escopo.
Ver `GPU_ARCHITECTURE_ROADMAP_2026-09-30.md` e
`GPU_KAN_RECOMPUTE_2026-09-30.md`. Builds posteriores compilados não herdam
certificação de suites de binários anteriores.

## Inventário atual — implementação publicada não encerra os gates

Atualização documental de2026-10-01: as quatro frentes ativas têm código
integrado em root, opt-in e com identidades versionadas. Os lotes detalhados
adiante são históricos e mantêm seus resultados/limites originais. A situação
atual não deve descrevê-las como código inexistente, nem converter publicação
em certificação final, qualidade, superioridade ou mudança de default.

| Requisito | Evidência necessária para fechamento | Situação |
|---|---|---|
| BitLinear/projeções: treino/prefill versus decode, WMMA, GEMV compacto, QAT e conversões | Paridade/VJP/STE; paths realmente executados; medidas por geometria e modelo completo | WMMA BF16/FP16 MoE opt-in validado; demais projeções/GEMV/preparação integral ainda incompletos |
| Mamba: geometria, SSD matricial, fusões e estabilidade | Referência independente, máscaras/estado/retomada, erro numérico e perfil RX 7600 | Mamba-2 preservado; Mamba3Layer SISO/MIMO N128 e integração publicados opt-in; otimização e gates finais de estabilidade/retomada/perfil permanecem abertos |
| Attention: reutilização, precisão mista, geometria e contexto | Forward/backward exatos e máscara/janela; throughput/memória; qualidade de eventuais aproximações | Tiled FP32 preservado; provider de treino RDNA BF16/FP16 com backward/ownership/status integrado e testes provider/consumer passando; perfil, matriz ampla e híbrido final pendentes |
| MoE: GEMM largo, seleção por forma, atividade/Adam GPU, aux GPU, QAT ativo | Experts vazios e microbatch accumulation sem alterar Adam/decay; checkpoint; medidas largas/curtas | Aux/QAT/WMMA e banco preservados; domínio de atividade device e GpuSparseAdam publicados; union/lazy/finite/clipping/decay/versões implementados; checkpoint v11, GPU final, seleção/perfil/qualidade pendentes |
| TTT: launches, isolamento e benefício por segundo | BPTT local + sessões/reset/replay; ablação de qualidade com orçamento fixo | BPTT local existe; restante aberto |
| KAN: escala GPU, RBF/projeção fundida, backward tiled/recompute | Gate de ablação qualidade/custo contra SwiGLU/MLP antes de retomar | Candidato escalar validado opt-in, D76826–28% mais lento; incremento WMMA escrito/compilado mas não certificado; desenvolvimento suspenso por decisão do usuário |
| Normas/ativações/resíduos | Auditoria dos caminhos completos, máscaras/precisão/gradientes e medidas de fusões | Aberto |
| Embedding/cabeça/loss | Gathers/scatters, vocabulário, masking de resposta/padding; paridade e profile | Head/loss CCE exato por tiles com recompute VJP e integração Jamba/Trainer publicados; testes oracle/masks/tying e tiled GPU existem; memória/perfil/qualidade e matriz final pendentes |
| Otimizador: atividade esparsa, descritores e atualização GPU | Estado por parâmetro, acumulação, decay, checkpoint e overflow sem corrupção | GpuSparseAdam e predicados device publicados, com moments FP32/lazy, clipping deterministic/folded, rollback e versões; interfaces host históricas e auditorias explícitas preservadas; checkpoint v11 e execução final pendentes |
| Ownership/contratos/lifecycle | Pesos, estado, tape, KV e scratch; cancel/reset/retomada; sessões e workspaces | Owners de tape/status/atividade e contratos de versão/abort publicados nas quatro frentes; fechamento integrado de lifecycle/checkpoint e matriz de concorrência pendentes |
| Seleção e observabilidade | Políticas versionadas, rejeição de incompatibilidades, execução real e perfil por caminho | Parcial |
| Decomposição arquitetural gradual | Responsabilidades separadas, sem reescrita que invalide os contratos | Pendente |
| Mamba-3: bloco e integração alinhados à referência fixada | Equações e parametrização originais; CPU/HIP fwd/bwd; SISO/MIMO, estado, integração, checkpoint, referência independente | Mamba3Layer SISO/MIMO N128/projeções densas FP32/backward/estado/Jamba/Trainer/packs/checkpoint/bindings publicados; testes isolados e Mamba3 combinado passaram; resume de trajetória v11, GPU final, perf/qualidade/hardware/sanitizer pendentes |
| Referências FlashAttention-4/AITER e tentativa de superação | Aplicabilidade RDNA explicitada; benchmark equivalente e ganho end-to-end medido | Provider RDNA local publicado; não é port Blackwell/AITER nem evidência de superar FA4; comparação equivalente/perfil end-to-end permanece aberta |

## Evidência atual e gates abertos — snapshot de 2026-10-01

- **Mamba-3:** implementação independente SISO/MIMO N128, config schema3/bloco1,
  identidade `mamba3_dense_fp32_siso_mimo_n128_v1`; dt_bias/D excluídos do decay
  por registro canônico, sem excluir Adam. CPU6/6 e build HIP7/7 isolados
  (inclui testes CPU e um nativo GPU);8 casos upstream pinados CPU e8 native
  HIP contra referência Torch CPU. Fonte primária SHA256
  `2259f1f32b4c58ef3ed2976076bd4ab6a50382a9902a16fbfbac404a1fde3618`
  verificado pelo harness. Não executa kernels upstream NVIDIA nem certifica
  superioridade/performance. Contrato e limites em
  [mamba3-integration](../artifacts/maestri_integral_20260930/mamba3-integration.md)
  e [mamba3-evidence](../artifacts/maestri_integral_20260930/mamba3-evidence.md).
- **Attention e optimizer:** provider/consumer e sparse transaction passaram;
  status Attention é mesclado diretamente no owner antes de preflight/update.
  Domínio device mantém union/first-write, experts inativos e contribuição zero
  legítima. Grouped/WMMA passaram no rerun de correção de tape; readbacks de
  controle/auditoria e mirrors de versões/checkpoint continuam explícitos.
  [Review](../artifacts/maestri_integral_20260930/atria-resume-20261001.md) e
  [rerun2/2](../artifacts/maestri_integral_20260930/final-hip-moe-tape-ctest-20261001.log).
- **CCE:** head/loss exato, sem filtering, tiles default128×256, recompute VJP,
  máscaras/denominadores e head tied nos caminhos suportados. O teste tiled GPU
  passou e a sonda híbrida5×17 reportou workspace85. Memória total inclui outros
  buffers e BLAS workspace; reduzir logits não prova throughput ou pico global.
  [Contrato](HEAD_CCE_2026-09-30.md).
- **Checkpoint v11:** root integrou correção do clone para preservar
  `Parameter.trainable`, com controle negativo CPU reproduzindo o erro e
  positivo corrigido. Snapshot frozen passou no executável final em0,16s.
  O híbrido passou loss/pesos/moments/presença lazy e versões locais
  (39 deltas+1/41 deltas+0 em cada réplica) na lane determinística, mas falhou
  `resume scheduler/token trajectory differs`. Clone/sidecar v10 omitem campos
  de trajetória; schema11/regressão CPU estão com Silex e precisam de rerun root.
  [Log final dirigido](../artifacts/maestri_integral_20260930/final-hip-snapshot-repair-ctest-20261001.log)
  e [cobertura restante](../artifacts/maestri_integral_20260930/vela-hybrid-remaining-20261001.md).
- **GPU final:** o log histórico HIP136/140 inclui quatro falhas; grouped/WMMA
  passaram depois em2/2, enquanto os consumidores checkpoint seguem abertos.
  CPU65/65 anterior e passes de revisões distintas não compõem um novo total
  final. Após schema11 é necessário rebuild/rerun combinado com identidade do
  binário, inventário e logs da mesma revisão. Fault Attention no híbrido,
  recovery e lane ordinary ainda não foram alcançados no último log.
- **Perf/qualidade:** não há fechamento de tempo até qualidade held-out,
  throughput/memória/custo do modelo completo ou ganho equivalente sobre
  baseline. Mamba3 GPU denso é baseline de correção; provider RDNA não transporta
  resultados FA4 Blackwell. Não promover default, SOTA ou objetivo integral.
- **Hardware/sanitizer:** execução GPU publicada é AMD RX7600/gfx1102/HIP.
  Matriz em hardware adicional/CUDA real e sanitizer dos novos consumidores
  na revisão final permanecem pendentes; compilação/capability condicional
  não substituem essas evidências. GPU permanece Experimental, TTT Research.
- **Docker/Linux:** root substituiu runtime `python3-minimal` por `python3`;
  build Docker e smoke Linux passaram antes de schema11. Smoke reportou
  `global_step=1`, `tokens_committed=4` e loss nativa finita. Isso comprova o
  smoke CPU/container dessa revisão, não GPU ou gates de qualidade.
  [Build aprovado](../artifacts/maestri_integral_20260930/final-docker-build-20261001-attempt2.log)
  e [smoke aprovado](../artifacts/maestri_integral_20260930/final-docker-smoke-20261001-attempt3.log).
  Falhas anteriores foram preservadas; rebuild/smoke pós-schema11 ainda serão
  executados por root.

## Histórico preservado dos lotes anteriores

Os parágrafos seguintes descrevem o código/evidência daquele lote específico.
Frases sobre leitura tardia host ou ausência de Adam device não descrevem a
lane opt-in posterior publicada. As contagens e medições históricas permanecem
válidas apenas para seus binários, flags e geometrias; não fecham gates atuais.

## Lote histórico: objetivo auxiliar e QAT MoE

- Objetivo Switch e VJP continuam exatos, agregados sobre todas as linhas válidas
  do step, não uma média de objetivos independentes por chunk.
- API device retorna a loss sem D2H; trainer soma as losses e materializa um
  escalar por step para logging/controle. Wrappers float continuam explícitos.
- Cargas finais são conservadas no dispositivo e materializadas sob demanda.
  Finalização pública padrão conserva a compatibilidade de telemetry imediata.
- Código morto de imbalance heuristic foi removido: ele não era um round-trip
  ativo. Redução determinística Switch usa um owner por expert, mantendo ordem
  de soma por linha. Buffers totalmente escritos não recebem memset redundante.
- QAT agrupado consulta offsets no dispositivo e prepara somente experts ativos;
  partials FP32 tree256 são somados em ordem fixa. Armazenamento efetivo é
  workspace reutilizável, separado dos pesos mestres, não um resultado em cache.
- Descritores só são reenviados se ponteiro/metadados mudarem. A cópia de um
  descritor novo permanece bloqueante; isso NÃO promete captura de treino.
- Identidade grouped passa a `device_segmented_tile16_active_qat_v2`. Mudança de
  ordem de redução QAT exige rejeição de checkpoints/binários da política antiga.
- A leitura tardia de offsets para registrar gradientes ainda existe. Não foi
  substituída por gradientes zero de todos os experts.

Validação deste lote: **CPU 60/60, HIP 119/119**, após implementação.
Medição paired com baseline preservado e repetição em ordem invertida:
ordered 7 -> 6 D2H calls, 60 -> 44 bytes; grouped v2 6 calls/48 bytes.
Hashes finais ordered antes/depois iguais nos probes. Variabilidade de tempo
impede promover ganho universal; grouped largo permanece mais lento.
Detalhes e limitações em `GPU_MOE_AUX_QAT_2026-09-30.md`.
Sem afirmação de WMMA, zero CPU, qualidade PT-BR, Mamba-3 ou release concluída.

## Lote histórico posterior: WMMA MoE RDNA3

Provider BF16/FP16 FP32acc para forward/dX/dW implementado com rocWMMA;
seleção por forma versionada, flag OFF, capability fail-closed, tape imutável,
checkpoint e rejeição de binário antigo. CPU60/60, HIP121/121 após implementação.
ISA confirma instruções WMMA, wave32/LDS8KiB, sem private segment no metadata.
Campanha paired e invertida: largo WMMA36.64/37.78 ms versus grouped
41.92/43.84 e ordered37.74/40.06 ms. FP32 não usa WMMA; B2 tem ganho incremental
pequeno; seis D2H e registro ativo CPU permanecem. Loss/weights BF16 entre
providers não são bitwise iguais; não é evidência de qualidade/convergência.
Não encerra o objetivo integral. Contratos, falhas preservadas, SHAs e limites
em `GPU_MOE_WMMA_2026-09-30.md`.

## Lote histórico posterior: contribuição esparsa explícita

`GPU_SPARSE_GRADIENT_ACTIVITY_2026-09-30.md` registra a correção do predicado
de atividade no Trainer. Buffers retidos não são contribuição do grupo;
expert inativo após um step ativo preserva pesos/momentos/versões. Zero
contribuído continua ativo. Acumulação, abort e snapshots integram o contrato.
Identidade host v1 rejeita retomada silenciosa da trajetória anterior MoE.
Isso não implementa a atividade/Adam device nem elimina a leitura tardia.

## Lote histórico posterior: acumulação de gradientes MoE por banco

`GPU_MOE_GRADIENT_COMMIT_2026-09-30.md` registra dois launches device por
backward com contribuições, sem slices e chamadas D2D por parâmetro. Destinos
e first/add são cacheados e conservam a união de microbatches, buffers estáveis
e experts inativos. Preflight valida os dois bancos antes de writes/publicação.
CPU60/60, HIP121/121 e Python15/15 passaram após implementação.
Leitura tardia/atividade host permanecem: não declarar atividade/Adam device
nem promoção de qualidade, SOTA, release ou conclusão do objetivo integral.
