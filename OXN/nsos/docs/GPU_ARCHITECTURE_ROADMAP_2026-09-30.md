# Roadmap ativo: inteligência/custo e treinamento AMD

## Diretriz do usuário

Este documento registra a mudança explícita de prioridade de2026-09-30:

1. **Mamba-3**, mantendo Mamba-2 como baseline versionado.
2. **Attention inspirada em FA4 para RDNA**, não um port literal Blackwell.
3. **MoE/Adam device-side**, preservando contribuição esparsa e experts inativos.
4. **Head/loss/CCE**, preservando masks, precisão e weight tying.

KAN não participa do desenvolvimento ativo. Preservar código, testes e resultados
negativos, todos os novos caminhos opt-in e sem promoção de default. Só retomar
se uma ablação demonstrar qualidade/custo superior a SwiGLU/MLP. Evidência e
limites de certificação em `GPU_KAN_RECOMPUTE_2026-09-30.md`.

A lista histórica integral continua em `GPU_RELEASE_SCOPE_2026-09-30.md`; a
diretriz posterior acima prevalece sobre sua ordem e sobre aprofundamento KAN.
Muon/SOAP, DyT, sparse N:M e alternativas de memória citadas no texto colado
continuam referências de experimentos, não substituições ou defaults autorizados.

## Estado publicado das quatro frentes — atualização de 2026-10-01

Há código integrado em root para as quatro frentes abaixo. Publicação de código
não significa fechamento do objetivo, promoção de default ou evidência de SOTA.
Os caminhos novos permanecem opt-in/versionados; Mamba-2 continua como baseline,
KAN permanece congelado e GPU permanece Experimental.

| Frente | Implementação publicada | Evidência e limite atual |
|---|---|---|
| Mamba-3 | `Mamba3Layer` SISO/MIMO, N128, projeções densas FP32, normalização/gating, backward completo, estado/tapes, Jamba/Trainer/bindings/config/packs/checkpoint | Suites isoladas CPU6/6 e build HIP7/7; referência upstream pinada em8 casos CPU e8 native HIP; teste Mamba3 no core combinado passou. Não é execução dos kernels upstream NVIDIA nem evidência de desempenho/qualidade |
| Attention RDNA | Provider de treino BF16/FP16, forward/backward, RoPE/GQA/prefix, replay das projeções BitLinear, ownership e ledger sticky; opt-in `NSOS_ATTN_TRAINING_PROVIDER` | Provider/consumer passaram no binário HIP; merge de status publicado no finite gate. A rejeição Attention no grafo híbrido ainda não foi alcançada no último log |
| MoE/Adam device | Domínio de contribuição device, união de microbatches, grouped MoE, clipping e Adam FP32 presence-aware, moments lazy, decay, rollback e versões locais | Transaction passou; grouped/WMMA passaram no rerun de root. Checkpoint de trajetória v11 e rerun combinado pendentes; auditorias/controle ainda têm fronteiras host explícitas |
| Head/loss/CCE | `NSOS_HEAD_CCE=1`, LSE/logits por tiles com recompute VJP, gradientes exatos sem filtering, headless Jamba/Trainer e identidade versionada | Tiled-head test passou; sonda no híbrido executou tiles5×17/workspace85. Ainda faltam gates finais de memória/throughput/qualidade do modelo |

Contratos/evidências:
[Mamba3](../artifacts/maestri_integral_20260930/mamba3-integration.md),
[resultados Mamba3](../artifacts/maestri_integral_20260930/mamba3-evidence.md),
[review Attention](../artifacts/maestri_integral_20260930/atria-resume-20261001.md),
[MoE/Adam](../artifacts/maestri_integral_20260930/silex-resume-20261001.md),
[CCE](HEAD_CCE_2026-09-30.md) e
[cobertura restante do híbrido](../artifacts/maestri_integral_20260930/vela-hybrid-remaining-20261001.md).
Os números históricos abaixo pertencem aos respectivos lotes e não certificam
revisões posteriores.

## 1. Mamba-3: fidelidade antes de otimização

Referência original verificada: [paper v1](https://arxiv.org/html/2603.15569v1)
e [repositório oficial](https://github.com/state-spaces/mamba).
Revisão fixada para o port:
`e9594ce1c732d97440f0332fdc43170a2294dbfa` (2026-07-22).
[Módulo original nessa revisão](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py).

O código oficial tem projeções z/x/B/C/dt/A/trap/ângulos, A dependente do dado
com heavy-tail activation/floor, BC RMSNorm seguida de biases por head e estado
de inferência composto por fase, SSM, K anterior e V anterior. Ele não é o
Mamba-2 atual com uma flag ou uma convolution renomeada. [Código original](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py).

Sequência de entregas do plano original, preservada como histórico; o estado
publicado está descrito acima e abaixo. SISO parcial não era um bloco integral:

- Derivar referência CPU FP64 da recorrência e VJP, incluindo exponential-
  trapezoidal, fase complexa/RoPE, normalização, biases, gating e skip.
- Implementar módulo SISO novo, sem reinterpretar pesos/checkpoints Mamba-2.
  Versionar configuração, parametrização, precisão e tape. Projeções ternárias
  NSOS são uma ablação explícita, não equivalência automática ao bloco original.
- Implementar forward/backward HIP exatos e streaming com todo o estado; usar
  fronteiras de chunks/recompute somente com adjunto correto entre fronteiras.
- Completar MIMO, incluindo projeções, layouts/rotação e todas as derivadas;
  preservar SISO como baseline e não trocar seu contrato silenciosamente.
- Integrar Jamba, registry de parâmetros, Trainer, bindings, packs, checkpoint,
  sessões/reset/cancelamento e telemetry. Resolver ownership de estado/tape.
- Só ao final do lote: referência independente, diferenças finitas, paridade
  upstream fixada, equivalência sequência/streaming, tails, masks, batches,
  retomada, overflow e perf RX7600. Build sozinho não fecha a entrega.

Pontos de ancoragem upstream:
`mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py`,
`mamba3_siso_fwd.py`, `mamba3_siso_bwd.py`, `mamba3_siso_step.py`, `angle_dt.py`;
MIMO em `mamba_ssm/ops/tilelang/mamba3/` e step em
`mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py`; referências em
`tests/ops/triton/test_mamba3_siso.py`,
`tests/ops/tilelang/test_mamba3_mimo.py` e `tests/modules/test_mamba3_varlen.py`.
O step CuTe upstream declara teste em H100; isso NÃO certifica AMD.

Registro histórico do lote SISO: referência FP64 e pré-processamento/VJP,
operador HIP com fronteiras/replay e tape proprietário, CPU61/61 e HIP127/127,
incluindo stream explícito/modo assíncrono. Naquele lote o bloco completo e MIMO
ainda não estavam incluídos. Ver `GPU_MAMBA3_SISO_2026-09-30.md` e
`GPU_PARALLEL_FOUNDATIONS_2026-09-30.md`; esses resultados foram preservados.

Estado publicado posterior: `Mamba3Layer` SISO/MIMO N128 possui projeções
z/x/B/C/dt/A/trap/ângulos, RMS BC/biases, output norm opcional, gating/skip,
VJP dos parâmetros/entrada/estado e sessões/reset/cancelamento. ModelConfig
schema3/bloco schema1 e identidade `mamba3_dense_fp32_siso_mimo_n128_v1`
distinguem esse caminho de Mamba-2. dt_bias/D continuam recebendo Adam, com
exclusão explícita de weight decay pelo registro canônico do modelo.

A referência primária fixada usa fonte SHA256
`2259f1f32b4c58ef3ed2976076bd4ab6a50382a9902a16fbfbac404a1fde3618`,
verificado pelo harness antes da referência Torch CPU. O build HIP isolado7/7
inclui os testes CPU reutilizados e o teste GPU nativo; não são7 kernels GPU
independentes. A implementação GPU é baseline de correção com histórico denso
e fronteiras de auditoria explícitas; otimização/perfil/qualidade permanecem
abertos. `Mamba2SSD`/`MambaConfig` preservam seus contratos históricos;
Mamba-3 não foi adicionado como alias de Mamba-2.

## 2. Attention/FA4-inspired RDNA

Implementado/publicado: provider RDNA de treino com políticas BF16/FP16,
backward, replay de BitLinear, tape proprietário, RoPE e status sticky. Há
geometrias/capabilities explícitas e rejeição de opt-in não suportado, sem
fallback silencioso. O status é mesclado diretamente no `GpuSparseAdam`
antes do preflight; o callback genérico permanece e Trainer não duplica merge.
Isso não implementa FA4 Blackwell nem comprova seu ganho em AMD.

As metas de otimização e validação ampla abaixo permanecem abertas:

Estudar scheduler causal/varlen, reutilização de tiles, redução de tráfego LDS e
overlap entre operações. FA4 explora recursos Blackwell específicos; os números
B200 não se transferem à RX7600. [Publicação dos autores](https://tridao.me/blog/2026/flash4/).

Entregas locais: QK/PV e backward adaptados a wave32/WMMA/LDS; geometria por
head/seq/batch; seleção de precisão versionada; softmax estável e máscaras
exatas. Não inserir exponencial aproximada sob uma política chamada exata.
Validar dQ/dK/dV, GQA, padding, janelas, recompute e determinismo; medir caminho
inteiro com contexto curto/longo. Tiling não elimina custo quadrático global.

## 3. MoE/Adam device-side

Implementado/publicado: `GpuGradientActivity`, `GpuMoeTraining` e
`GpuSparseAdam`, integrados ao Trainer com domínio por grupo de acumulação.
Predicados device controlam união/current/first-write, finite gate, clipping,
momentos lazy, Adam/decay e publicação de versões; rejeição mantém a trajetória
e abort fecha o domínio. Isso substitui o registro tardio host na lane opt-in;
os caminhos históricos permanecem disponíveis. Readbacks de auditoria,
controle/telemetry e mirrors de checkpoint/versões continuam explícitos;
não se afirma zero CPU ou captura de treino.

O checkpoint v10 perdeu campos de trajetória no clone/sidecar. Schema11 e
regressão CPU estão com Silex; a certificação do resume e do grafo híbrido
depende do rebuild/rerun final de root. Requisitos de fechamento/perfil:

Remover a fronteira tardia offsets->CPU sem transformar experts inativos em
gradientes zero ativos. Device activity deve controlar finite gate, clipping,
coorte Adam, momentos lazy, decay, versão/cache, acumulação e abort. Descritores
devem ter ownership e lifetime explícitos. Preservar objetivos auxiliares exatos,
QAT somente ativo e checkpoint. Validar experts vazios/inativos-após-ativos,
contribuição zero legítima, união de microbatches, status inválido e concorrência.

## 4. Head/loss/CCE

Implementado/publicado: política
`exact_unfiltered_bitlinear_row128_vocab256_recompute_v1`, selecionada por
`NSOS_HEAD_CCE=1`. Logits/LSE por tiles e recompute VJP evitam salvar o plano
global tokens*vocab. Integra BitLinear, QAT/RMS/magnitude/bias e gradientes no
Parameter canônico, incluindo head tied nos caminhos suportados; Trainer usa
o contrato headless e grava sua identidade. Inference mantém logits explícitos.
Ver `HEAD_CCE_2026-09-30.md` para restrições e matriz de testes.

O contrato inicial de implementação foi:
Primeiro contrato sem gradient filtering: a referência CCE distingue caminhos
filtrados de uma opção exata. [Implementação original CCE](https://github.com/apple-aiml-research/ml-cross-entropy).

Preservar supervisão da resposta, padding/ignore, denominador da redução,
weight tying e acumulação embedding+head no mesmo Parameter. QAT/RMS/magnitude/
bias da cabeça existente não podem ser omitidos para chamar uma linear simples
de equivalente. Inference/generation continua podendo requerer logits explícitos.
Validar loss e gradientes independentes, parâmetros compartilhados, máscaras
vazias/heterogêneas, precisão e checkpoint; medir pico de memória e throughput.

## Gates ainda abertos — evidência não equivale a promoção

- Checkpoint de trajetória: schema11 deve preservar counters de tokens,
  scheduler/A e validar migração/compatibilidade, sem contornar guards. O patch
  de trainability frozen já foi integrado; a regressão host passou no executável
  final ligado ao core HIP. No híbrido, pesos/moments/presença lazy e versões
  locais passaram na lane determinística antes da falha de counters.
- GPU final: rerun do binário combinado após schema11, incluindo Attention
  inválido impedindo commit MoE/Adam, recovery e lane ordinary. O histórico
  HIP136/140 e reruns dirigidos de revisões diferentes não constituem uma suite
  final integral aprovada; CPU65/65 anterior também não certifica código novo.
- Desempenho/qualidade: medidas por geometria e modelo completo, VRAM real,
  custo/tempo até qualidade held-out equivalente. Sem afirmação SOTA, default
  ou objetivo integral concluído.
- Hardware/sanitizer: evidência GPU atual é RX7600/gfx1102/HIP; falta validação
  dos novos consumidores em outros hardwares/CUDA real e sanitizer de memória
  GPU da revisão final. Um build ou capability conditional não fecha esse gate.
- Linux/container: root corrigiu runtime `python3-minimal` para `python3`;
  Docker build e smoke Linux passaram antes de schema11. Logs
  [build](../artifacts/maestri_integral_20260930/final-docker-build-20261001-attempt2.log)
  e [smoke](../artifacts/maestri_integral_20260930/final-docker-smoke-20261001-attempt3.log).
  Rebuild/smoke da revisão pós-schema11 ainda pendentes; não é evidência GPU.

## Gate transversal: qualidade/custo, não quantidade de técnicas

Dataset/tokenizer/holdout versionados e deduplicados; identidade dos pesos,
binário, configurações, flags e caminhos realmente executados. Uma variável
arquitetural por experimento. Comparar seeds, parâmetros treináveis/ativos,
tokens supervisionados, wall-clock, VRAM e custo da avaliação. Reportar qualidade
held-out e tempo até qualidade-alvo, não só loss de treino ou tok/s.
Resultados negativos permanecem; ganho de kernel não promove default/SOTA.

Testes de validação somente ao final da implementação de cada lote; resultados
de uma revisão não certificam uma posterior. GPU Experimental e TTT Research
permanecem conforme `PRODUCT.md`. O objetivo integral permanece aberto.
