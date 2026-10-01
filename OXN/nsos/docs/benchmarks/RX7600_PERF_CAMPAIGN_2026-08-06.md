# Campanha de performance NSOS — RX 7600 (2026-08-06)

Escopo: fluxo real de `scripts/train_ptbr_conversational.py`, preset `pilot`,
`--device gpu`. Todas as medições foram executadas nesta estação. Nenhum número
foi estimado ou reaproveitado de execuções anteriores; cada modo foi certificado
com o binário que o contém.

**Resultado: 1579,82 -> 2624,55 raw tokens/s p50 = +66,13%.**

---

## A. Arquitetura do fluxo alcançado

```
CLI (parse_args, acao=train)
 -> main(): NSOS_DETERMINISTIC=1, NSOS_TRAIN_CHUNK_SIZE=1
 -> detect_build_dir() -> build-codex-hip -> load_nsos() -> nsos_ext (.pyd HIP)
 -> train(): validate_training_workspace() (sha256 de todo shard)
 -> Tokenizer.load(tokenizer.nsos)  [vocab 16384]
 -> set_seed / set_deterministic_reductions(True) / set_strict_gpu_execution(True)
 -> set_matmul_precision(--matmul-precision)          [novo nesta campanha]
 -> build_model_config() -> JambaModel(config, GPU) -> model.to(GPU)
 -> Trainer(model, lr) + TrainPhaseScheduler (QAT desligado no pilot)
 -> CheckpointManager (identidade imutavel + sha256 por arquivo)
 -> loop de fases [base, continuation, sft]
      -> read_u16_tokens(shard)             mmap + array.frombytes
      -> janela causal seq_len+1 (batch=1)
      -> trainer.train_step(entrada, alvo)  <-- 89,7% do tempo na baseline
      -> callback: telemetry.observe(), manager.poll(), checkpoint periodico
 -> evaluate_phase() -> checkpoint milestone -> verify_all() -> run_metrics.json
```

Componentes confirmados em runtime (nao por leitura de codigo):

| Item | Valor observado |
|---|---|
| GPU | AMD Radeon RX 7600, `gfx1102`, 32 CUs |
| Backend | `hip`, `gpu.strict_execution=true` |
| Modelo | Mamba-2 faithful puro, 16 camadas, d_model 768 |
| Parametros treinaveis | 71.245.696 (404 tensores) |
| Atencao | nenhuma camada criada (`attention_period=64` com 16 camadas) |
| GEMM | `hipblas`, `lt_compiled=false` |
| Batch | 1 (imposto por `raise` explicito no script) |

Nenhum fallback silencioso para CPU foi observado.

---

## B. Baseline reproduzivel

`perf-baseline-A-20260806`, 1000 passos, 900 medidos (101-1000).

| Metrica | Valor |
|---|---|
| raw tokens/s p50 | **1579,82** |
| p10 / p90 | 1573,62 / 1585,30 |
| CV | 0,74% |
| steps/s p50 | 3,0856 |
| loss held-out | 9,94792 -> 4,08490 (`pass_learning_signal`) |

Reproduz a faixa historica do projeto (1573-1580 em 14 execucoes de 2026-08-01/02).

---

## C. Perfil medido e sua evolucao

Instrumentacao nativa (`NSOS_TRAIN_TIMING=1`), media por passo:

| Etapa | baseline | pos OPT-1/2 | pos OPT-4 (fp16) |
|---|---:|---:|---:|
| forward | 130,38 | 55,34 | **36,65** |
| backward | 163,23 | 156,43 | **126,44** |
| optimizer | 30,44 | 30,70 | ~30 |
| preparation (dados) | 1,92 | — | — |
| loss (CE) | 1,07 | — | — |
| **wall** | **327,42** | **245,83** | **197,27** |

O balanco fecha na baseline (soma 327,41 vs wall 327,42).

### Ocupancia (evidencia de compilador, gfx1102)

Geometria do scan forward sequencial (P=64, H=24, Batch=1, warp=32):
`blocks = Batch*H*(P/warp) = 48`, `threads = 32`. **48 wavefronts para 64
SIMDs** (32 CUs x 2) = menos de 1 wave por SIMD; nenhuma outra wave residente
para esconder stalls. O backward ja usava decomposicao em chunks (192 blocos).

`-Rpass-analysis=kernel-resource-usage`:

| Kernel | VGPR | Scratch B/lane |
|---|---:|---:|
| `mamba2_faithful_forward_kernel` (generico) | 29 | **272** |
| `..._forward_fixed_kernel<64>` | 128 | 124 |
| `..._forward_fixed_kernel<32>` | 89 | 0 |
| `..._backward_chunk_summary<*>` | 9 | 0 |
| `..._conv_backward_fixed_kernel<4>` (ja templado) | 17 | 0 |

### GEMM por shape (medido, `bench_gemm.py`)

| shape | fp32 ms | bf16 ms | fp16 ms | bf16 | fp16 |
|---|---:|---:|---:|---:|---:|
| in_proj `[512,768]x[768,3224]` | 1,275 | 0,721 | 0,537 | 1,77x | 2,38x |
| out_proj `[512,1536]x[1536,768]` | 0,593 | 0,346 | 0,275 | 1,71x | 2,16x |
| lm_head `[512,768]x[768,16384]` | 6,896 | 4,346 | 3,258 | 1,59x | 2,12x |
| lm_head bwd | 6,985 | 4,617 | 3,448 | 1,51x | 2,03x |
| in_proj bwd | 1,321 | 0,779 | 0,581 | 1,70x | 2,27x |

Eficiencia sobe de ~21% do pico (fp32) para ~38% (bf16) e ~48% (fp16),
confirmando que os caminhos WMMA acelerados sao realmente exercitados.
Somando o passo: GEMM ~111 ms de 245 ms = **~45%** antes do OPT-4.

---

## D. Matriz de alteracoes

### OPT-1 — Especializacao em tempo de compilacao da largura de estado

| Campo | Conteudo |
|---|---|
| Problema | `state[MAX_N]` com trip count de runtime -> 272 B/lane de scratch |
| Evidencia | resource-usage do compilador; forward = 39,8% do passo |
| Alteracao | `mamba2_faithful_forward_fixed_kernel<N>` (16/32/64) + dispatch |
| Contrato | quebra bit-exatidao vs sequencial; opt-in, registrado na identidade |
| Teste | 15/15 CTest; strict bit-exato vs baseline em 300 passos |
| Ganho | **+12,48%** (isolado) |
| Status | **Mantido, opt-in** (`NSOS_MAMBA_FAITHFUL_FIXED_STATE=1`) |

**Nota de honestidade.** A hipotese inicial era que desenrolar o laco seria
bit-exato. Provou-se **falsa**: clang em HIP usa `-ffp-contract=fast` e agrupa
FMAs diferente no codigo desenrolado (divergencia de 1-2 ULP). Os testes de
paridade nao detectaram porque comparam com tolerancia; quem detectou foi o
traco de determinismo do proprio projeto.

### OPT-2 — Scan forward chunked (paralelismo temporal)

| Campo | Conteudo |
|---|---|
| Problema | 48 wavefronts para 64 SIMDs; forward preso em latencia |
| Evidencia | geometria de lancamento; ~7% do pico FP32 |
| Alteracao | `chunk_summary` / `chunk_prefix` / `chunk_apply` + launcher + workspace |
| Mecanismo | profundidade sequencial 512 -> 2x128; wavefronts 48 -> 192 |
| Contrato | carry acumulado como `D_c*carry + end_local`; opt-in, na identidade |
| Teste | **18/18 CTest** com o caminho ativo; paridade com chunk forcado=8 (`max_abs_grad_delta=1,13e-06`); **repro A/B bit-exato** |
| Ganho | forward **-57,6%** (130,38 -> 55,30 ms); **+29,04%** end-to-end |
| Status | **Mantido, opt-in** (`NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD=1`) |

Decisao de projeto que viabilizou o ganho: `apply` **reexecuta a recorrencia
original** a partir do carry correto, em vez de ler o historico local e somar o
carry. Isso troca banda de memoria (~22 ms/passo estimados) por compute
redundante, que e barato quando ha CUs ociosas.

### OPT-3 — hipBLASLt: **rejeitado por investigacao**

`src/gpu_gemm_provider.cpp` e politica pura. `LtBlas` aparece apenas no enum,
no retorno da struct e no campo de identidade; **nao existe nenhuma chamada de
API hipBLASLt** (`hipblasLtCreate`, `hipblasLtMatmul`, descritores,
heuristicas). `require_classic_gemm_provider()` ainda exige o caminho classico.
Habilitar `NSOS_ENABLE_BLAS_LT_PROVIDER` faria a politica declarar Lt sem ligar
nada. Nao ha "benchmark por shape" a fazer porque nao ha implementacao a
comparar.

### OPT-4 — Precisao de compute das GEMMs (bf16 / fp16)

| Campo | Conteudo |
|---|---|
| Problema | GEMMs a ~21% do pico ocupando ~45% do passo |
| Evidencia | `bench_gemm.py` por shape (tabela acima) |
| Alteracao | flag `--matmul-precision {fp32,bf16,fp16}` expondo o modo que ja existia no runtime (`set_matmul_precision`) |
| Contrato | master weights, gradientes e acumuladores permanecem FP32; `precision.matmul` ja fazia parte da identidade |
| Teste | 1000 passos por modo, zero losses nao-finitas, `pass_learning_signal`; **repro bf16 A/B bit-exato** |
| Ganho | bf16 **+57,33%**, fp16 **+66,13%** (acumulado sobre a baseline) |
| Status | **Mantido, opt-in por flag; default segue fp32** |

O modo ja estava implementado no runtime e simplesmente nunca era acionado pelo
script de treino.

### Varredura de tamanho de chunk (sem alteracao de codigo)

| Forward | p50 tok/s | | Backward | p50 tok/s |
|---|---:|---|---|---:|
| 128 | **2037,51** | | 128 | 2037,51 |
| 64 | 2010,10 | | 64 | 2005,90 |
| 32 | 1962,52 | | 32 | **2091,58** |

Forward otimo em 128; backward otimo em 32 (+2,65%). O ganho pequeno e o padrao
nao-monotonico do backward indicam que **ele nao e limitado por ocupancia** —
ao contrario do forward, onde a mesma mudanca rendeu -57,6%.

---

## E. Resultado final

| config | p10 | **p50** | p90 | CV | eval loss (1000 passos) | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline fp32 | 1573,62 | **1579,82** | 1585,30 | 0,74% | 4,08490 | — |
| bf16 | 2474,85 | **2485,60** | 2499,41 | 0,99% | 4,08905 | **+57,33%** |
| fp16 | 2614,39 | **2624,55** | 2633,87 | 0,74% | 4,08566 | **+66,13%** |

Todos com zero losses nao-finitas e `pass_learning_signal`.

Validacao executada:

- **Nivel A** — build limpo; ate 18/18 CTest (paridade faithful/nstate/proper,
  gradcheck, parallel scan) com cada caminho ativo.
- **Nivel B** — recursos de kernel pelo compilador e GEMM por shape.
- **Nivel C** — corridas de 20 e 300 passos sem NaN/Inf.
- **Nivel D** — gates de 1000 passos por modo; strict comprovadamente bit-exato
  contra a baseline; cada modo certificado no binario que o contem.
- **Nivel E** — soak de 10.000 passos por candidato:

| soak 10k | p50 tok/s | eval final | nao-finitas | checkpoints | VRAM pico | gap vs fp32 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 (OPT-1) | 1778,05 | 3,22807 | 0 | 4/0 | 5448 MB | — |
| fp16 | **2618,51** | 3,23752 | 0 | 4/0 | 5456 MB | **+0,293%** |
| bf16 | 2486,17 | **3,22508** | 0 | 4/0 | 5456 MB | **-0,093%** |

Nenhum soak degradou throughput ao longo da corrida, vazou memoria, produziu
loss nao-finita ou corrompeu checkpoint. Todos cruzaram 2 fronteiras de shard.

### O horizonte muda o veredito sobre fp16

| horizonte | fp32 | fp16 | gap |
|---|---:|---:|---:|
| 1.000 passos | 4,08490 | 4,08566 | +0,019% |
| 10.000 passos | 3,22807 | 3,23752 | **+0,293%** |

O gap cresceu ~15x enquanto o horizonte cresceu 10x. Em 1.000 passos fp16
parecia *melhor* que bf16; em 10.000 a ordem se inverteu. A comparacao limpa
— bf16 e fp16 rodaram o mesmo caminho chunked, diferindo apenas na precisao —
da **fp16 0,386% pior que bf16**.

Mecanismo: bf16 mantem o expoente de 8 bits do fp32 (mesma faixa dinamica, so
menos mantissa) e nao acumula erro de faixa; fp16 tem expoente de 5 bits e
acumula. bf16 em 10.000 passos e indistinguivel do fp32 (-0,093%, dentro do
ruido entre corridas).

**Ressalva de extrapolacao:** 10.000 passos sao ~1,3% da receita `pilot`
(100M tokens vistos). Como o gap do fp16 cresce com o horizonte, extrapolar
para o treino completo nao e seguro em nenhuma direcao. O que os dados
sustentam e: bf16 nao mostra degradacao em 10k passos; fp16 mostra
degradacao pequena e crescente.

### Auto-reprodutibilidade (`fast_deterministic`)

Exigencia atendida para cada modo: duas corridas independentes, mesma seed,
mesmos dados, mesmo binario.

| Modo | loss trace | `model.bin` | `trainer.state` |
|---|---|---|---|
| `fixed_state` | BIT-EXATO | IGUAL | IGUAL |
| `chunked_forward` | BIT-EXATO | IGUAL | IGUAL |
| `bf16` | BIT-EXATO | IGUAL | IGUAL |

`progress.json` difere apenas em `saved_at` e telemetria de tempo; todos os
campos de aprendizado (`latest_loss`, `loss_first/last/mean`, `global_step`,
`offset`) sao identicos.

Cada modo entra na identidade de runtime **apenas quando ativo**, preservando o
digest do default e a retomada dos checkpoints existentes:
`mamba.forward_state_width`, `mamba.forward_scan_decomposition`,
`mamba.forward_chunk_size`, `precision.matmul`.

### Recomendacao

| Uso | Configuracao | Ganho | Justificativa |
|---|---|---:|---|
| **Producao / treino longo** | **bf16** | **+57,3%** | Qualidade indistinguivel do fp32 em 10k passos (-0,093%); expoente de 8 bits nao acumula erro de faixa |
| **Triagem de dados / iteracao** | fp16 | +66,1% | 0,386% pior que bf16 e o gap cresce com o horizonte, mas irrelevante quando o objetivo e descobrir problema de corpus |
| **Evidencia / reproducao bitwise** | fp32 (default) | — | Unico modo bit-exato contra checkpoints anteriores |

```bash
# padrao: bit-a-bit identico ao comportamento anterior
python scripts/train_ptbr_conversational.py train --preset pilot --device gpu

# producao recomendada (+57,3%)
NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD=1 NSOS_MAMBA_BACKWARD_CHUNK_SIZE=32 \
  python scripts/train_ptbr_conversational.py train --preset pilot \
    --device gpu --matmul-precision bf16

# triagem rapida (+66,1%)
NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD=1 NSOS_MAMBA_BACKWARD_CHUNK_SIZE=32 \
  python scripts/train_ptbr_conversational.py train --preset pilot \
    --device gpu --matmul-precision fp16
```

### Criterios de parada (documento de meta, secao 10)

| # | Criterio | Situacao |
|---|---|---|
| 1 | Sem gargalo dominante corrigivel sem alteracao arquitetural | **Atingido** — o que resta (`state_history`, WMMA em kernels proprios, batch) e projeto grande ou muda contrato |
| 2 | Tres ciclos sem ganho relevante | Nao atingido (+12,5%, +29,0%, +66,1%) |
| 3 | Ganho restante < risco de regressao | **Atingido** para os itens remanescentes |
| 4 | GPU proxima do teto mensuravel | Parcial — GEMM saiu de 21% para ~48% do pico; scan reestruturado |
| 5 | Ganho adicional exige mudar contrato matematico ou arquitetura | **Atingido** para batch>1 |
| 6 | Candidato final passou todos os gates | **Atingido** — bf16 e fp16 completaram os niveis A-E |

---

## F. Backlog tecnico

### Implementado e certificado
1. OPT-1 — especializacao da largura de estado (+12,48% isolado).
2. OPT-2 — scan forward chunked (+29,04% acumulado).
3. OPT-4 — bf16/fp16 exposto e medido (+57,33% / +66,13% acumulado).
4. Chunk do backward em 32 (+2,65%, apenas variavel de ambiente).

### Rejeitado por evidencia
5. **hipBLASLt** — nao existe implementacao, apenas politica. Construir do zero
   e projeto maior que OPT-2 e o retorno e incerto: hipBLASLt historicamente
   mira CDNA/MI, nao RDNA3.
6. **Dataloader / Python** — `preparation_ms` = 1,92 ms = 0,6% do passo.
7. **Chunk menor no backward alem de 32** — ganho de 2,65% com padrao
   nao-monotonico; nao e limitado por ocupancia.

### Proximos alvos
8. **Retencao de `state_history`** — `retain_full_history_v1` grava ~201 MB por
   camada por passo. Ha folga de VRAM, mas o trafego e candidato a
   recomputacao seletiva agora que o scan ficou barato.
9. **WMMA direto nos kernels proprios** (scan/conv), nao so nas GEMMs de BLAS.
10. **hipBLASLt**, se e quando houver suporte maduro para gfx1102.

### Dependencias de hardware/contrato
11. `batch=1` permanece por contrato explicito do script. Elevar batch e a via
    mais direta para ocupancia, mas altera a trajetoria de otimizacao e o
    contrato de retomada; e decisao de produto, nao de performance.
12. `__dp4a` em HIP e emulado por software; relevante so para inferencia
    ternaria empacotada, nao para este treino.
