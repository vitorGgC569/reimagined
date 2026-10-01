# Treinamento MoE segmentado na GPU — 2026-09-30

> Registro histórico do primeiro lote v1. O lote subsequente
> `GPU_MOE_AUX_QAT_2026-09-30.md` implementa aux no dispositivo, QAT apenas de
> experts ativos e cache de descritores, com identidade grouped v2. As medidas
> abaixo não devem ser atribuídas ao binário posterior.

## Escopo e contrato

Este lote implementa o candidato `NSOS_MOE_GROUPED_TRAINING=1`, com
`NSOS_MOE_ORDERED_DEVICE=1`. Não promove automaticamente um kernel experimental
e não certifica qualidade linguística, treinamento de produção ou release v1.

- Forward sem leitura de counts/offsets na CPU para agendar experts.
- Capacidade esparsa máxima `rows * top_k`, offsets e total no dispositivo.
- GEMMs forward/dX/dW segmentados por expert, tiles 16x16, LDS pad+1,
  acumulador FP32 e arredondamento de operandos FP32/BF16/FP16.
- Experts vazios e tiles fora do segmento retornam uniformemente.
- RMSNorm, magnitude, bias, squared-ReLU e respectivas derivadas preservados.
- QAT usa a regra de escala ternária existente e STE de peso/ativação.
- Gradiente de tarefa para o router, padding e combinação ordenada preservados.
- Metadados/versões capturados no forward; alterações incompatíveis invalidam
  o backward, que só pode consumir o tape uma vez.
- Geometria/adapters não suportados são erros explícitos, sem fallback oculto.

O núcleo está em `src/gpu_moe_training.cpp` e
`src/cuda/moe_training_kernels.cu`; a integração está em `JambaBlock`.
A opção Python é `--moe-compute-policy grouped-v1`, sem mudar a arquitetura.
Checkpoint identifica `moe.training_compute_policy` como
`device_segmented_tile16_late_registry_v1`; binários antigos que ignorem a
opção são rejeitados antes do primeiro step. O contador
`gpu_dispatch_counters()['grouped_moe_training']` registra execução real.

## Limites conhecidos e próximos gargalos

1. Ainda há uma leitura tardia `(experts+2)*sizeof(int)` por bloco no backward:
   registra gradientes apenas dos experts ativos. Registrar zero para todos
   mudaria presença de gradientes, criação de momentos Adam e weight decay.
   Para eliminá-la é necessário redesenhar o registro/otimizador com atividade
   no dispositivo, incluindo checkpoints e acumulação entre microbatches.
2. Preparação QAT ainda usa as rotinas por expert existentes; pesos efetivos
   de experts vazios podem ser preparados. Não é avaliação densa de experts,
   mas é desperdício remanescente a medir e reduzir.
3. GEMM tiled escalar não usa WMMA/MFMA. Menos overhead pode ganhar em experts
   pequenos e perder em grandes. Medir antes de selecionar/promover.
4. Upload de descritores é bloqueante; esta implementação não promete captura
   HIP Graph, execução persistente, concorrência de sessões ou zero CPU.
5. Os gradientes seguem as derivadas existentes: QAT usa STE, seleção top-k é
   discreta, não é uma derivada diferenciável da escolha do expert.
6. O balanceamento auxiliar existente ainda materializa loads e usa a CPU.
   Este lote não altera sua função objetivo silenciosamente.

## Validação

Implementação e definições de testes precederam o build/test deste lote,
conforme solicitado. Na compilação foi corrigido o acesso a uma propriedade
privada via getter explícito. A primeira suíte ampla apontou duas falhas de
integração reais: manifesto de kernels sem a nova unidade e includes de
fornecedor fora do cabeçalho central. Ambas foram corrigidas; os logs iniciais
foram mantidos, sem afrouxar os testes.

- Build Release CPU e HIP/gfx1102 completo.
- CPU final: **60/60**; HIP final: **119/119**.
- Python: 13 casos de contrato do probe/políticas, incluídos no gate nativo.
- Teste GPU independente usa os BitLinear por expert existentes e combinação
  da fixture no host como oracle, sem usar helpers agrupados. 12 combinações
  FP32/BF16/FP16 × linear exato/RMS+magnitude × referência/QAT, com caudas
  R17/D33/H35, bias, padding e experts nunca selecionados.
- Dois casos totalmente vazios, consumo único do tape, versão de peso alterada
  e roteamento que ultrapassa a capacidade. Roteamento inválido gera NaN e
  recusa o backward sem registrar gradientes, em vez de OOB/resultado plausível.
- Checkpoint MoE: quatro steps em modelos independentes, loss/pesos/momentos
  Adam e retomada bitwise idênticos sob a mesma política. Retomada sob política
  incompatível é recusada sem mudar pesos, momentos, LR, RNG ou step.
- No teste isolado, forward tem zero D2H; backward lê somente 24 bytes para
  E4 e possui uma sincronização de stream explicitamente contabilizada.
- Binário antigo real recusado antes de treinar, com `native_verified=false`
  e nenhum sample de step. Artefato `stale-native-rejected.json`.

Não é finite-difference independente do modelo inteiro nem checker de memória
GPU. Paridade numérica de QAT valida o STE implementado, não uma derivada
clássica do quantizador discreto. Produto GPU permanece Experimental.

## Medição pareada e decisão de promoção

RX 7600/gfx1102, TheRock/Windows, wave32, strict GPU/determinismo, duas camadas,
uma MoE E4/k2/H=D, outra FFN densa, Mamba faithful nas duas, vocab257. QAT
desabilitado no probe de velocidade. IDs sintéticos, sem tokenizer; não há
evidência de inteligência, generalização ou PPL PT-BR. As flags `redesign-v1`
das outras frentes são iguais nos dois braços; a variável é o compute MoE.

Primeira série: 9 steps, primeiro excluído da mediana, processos novos,
baseline ordered do binário preservado anterior:

| Geometria / operandos | Ordered ms | Agrupado ms |
|---|---:|---:|
| D128/S32/B1 FP32 | 8,325 | 8,004 |
| D128/S256/B1 FP32 | 10,980 | 10,842 |
| D128/S256/B1 BF16 | 11,242 | 10,896 |
| D128/S256/B2 BF16, SFT heterogêneo | 14,820 | 13,680 |
| D768/S512/B1 BF16 | 40,924 | 44,932 |

Repetição mais longa: 29 steps, primeiro excluído, mesmo binário final nos
dois braços, ordem invertida (agrupado antes do ordered):

| Geometria | Ordered ms | Agrupado ms | Leitura honesta |
|---|---:|---:|---|
| D128/S32/B1 FP32 | 8,475 | 8,461 | Sem ganho sustentado relevante |
| D128/S256/B2 BF16 | 14,475 | 13,540 | ~6,5% menor latência, ~1,07× throughput |
| D768/S512/B1 BF16 | 39,475 | 43,100 | ~9,2% mais lento; não promover |

Casos D128 têm 407.836 elementos; D768 tem 12.301.716. Hash inicial D128
`441f7ad19414a9e1614bf7819b31f6d8145a3242b77babfd7311903b81d1eb99`, D768
`575cf1437edc219c452932a28cdd34216245d97a73bb36b0ecf97ee6fdfa66f8`, iguais
em cada par. Hashes finais **não** são iguais entre algoritmos: a ordem de
redução/FMA difere. Paridade tolerada no teste independente e continuidade
bitwise dentro de uma política são contratos diferentes. Não afirmar
equivalência bitwise entre GEMM hipBLAS e GEMM tiled.

Nos casos B1: D2D calls/step **50→18** (-64%). D2H calls permanecem 7;
bytes **60→64** (status adicional). Stream synchronizations **8→9**: a
leitura foi deslocada do agendamento forward para o registro backward, não
eliminada; a fronteira explícita é observável. No B2: D2D **48→16**, sync
**6→7**. Balanceamento auxiliar continua usando a CPU.

Pico live/reservado B2 na repetição: ordered **62,625/68,0625 MiB**,
agrupado **59,75/61,25 MiB**. D768: **525,0625/630,0625→522,4375/555,375 MiB**.
São contadores do allocator, não tráfego físico, utilização CU ou residency
certificada pelo hardware. Há outliers de tempo no Windows; as séries são
limitadas e não uma certificação oficial de throughput.

**Decisão:** candidato permanece opt-in, default desligado, ordered/hipBLAS
preservado. Não há seleção automática de geometria baseada nestas sondas.
Próximas frentes concretas: grouped GEMM acelerado para matriz larga/prefill,
registro de atividade por expert/Adam no dispositivo, balanceamento auxiliar
sem round-trip e preparação QAT esparsa/fundida. Não confundir menos launches
ou cópias com ganho garantido de treino.

## Artefatos e reprodução

Diretório `artifacts/gpu_moe_grouped_20260930`:
`cpu-full-final.log`, `hip-full-final.log`, `targeted.log`, séries
`ordered-*.json`/`grouped-*.json` e `repeat-*.json`; logs iniciais preservados.

- HIP final SHA256:
  `7410fa8d18f0007d97b799ebf8d13b850ae8f5c970804c875bfb2e8c2688a473`.
- CPU final SHA256:
  `7050dd59e80d4dfc4f4aaaae7367f3424bedb458a0606c102eca474c839994d5`.
- Baseline ordered anterior preservado em `reference-ordered`:
  `01afcd8d02085a7180787c2d35179d3ea38b9b66fc5fcce7b0ff9354540145c1`.

```powershell
python OXN/nsos/scripts/audit_training_integrations.py --build-dir OXN/nsos/build-gm-hip --variant moe --redesign --grouped-moe --d-model 128 --seq-len 256 --batch-size 2 --precision bf16 --steps 29 --output <arquivo-novo.json>
```

Omitir `--grouped-moe` para o controle ordered; manter todo o resto idêntico.
Executar um processo por vez. O script recusa sobrescrever resultados.
Não foi iniciado treino de produção nem relaxada a admissão dos dados SFT.
O diff acumulado do worktree contém outros lotes e alterações preexistentes;
não atribuir todo ele a esta implementação.
