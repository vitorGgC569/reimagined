# Redesenho de treinamento GPU — implementação e gate de release

## Escopo e estado

Implementação dos gargalos medidos em `GPU_TRAINING_ENGINEERING_2026-09-29.md`.
Resultados daquela análise são baseline, não validação destes kernels novos.
Os novos caminhos passaram na validação descrita abaixo e permanecem opt-in:
o ganho depende da geometria, e as lacunas de release/qualidade não estão fechadas.
Não há promessa de superar SOTA nem de elevar qualidade linguística sem evals.

Atualização posterior no mesmo dia: a derivada completa **por sequência** TTT,
isolamento de treino por amostra e histórico de fronteiras foram implementados
e validados opt-in. Veja [o relatório TTT](GPU_TTT_FULL_SEQUENCE_2026-09-29.md).
As tabelas de implementação/medições e SHAs deste documento descrevem o
primeiro lote, não o binário TTT posterior; não reutilizar sua velocidade como
certificação do novo contrato. As lacunas abaixo refletem o estado atualizado.

## Caminhos implementados

| Frente | Implementação | Contrato / limite |
|---|---|---|
| Mamba faithful | Guarda estados de entrada a cada 32 tokens; reconstrói localmente cada estado no backward em LDS | Determinístico, head-wave elegível, N≤64, chunks backward=32; conserva limite explícito de sequência |
| Norma / clipping | Descritores de 8192 elementos, árvore FP64 de 256 threads, coeficiente e aplicação na GPU | Norma final de 8 bytes ainda cruza CPU para gate de segurança; fallback para gradientes mistos CPU/GPU |
| MoE | Atribuição estável por expert/linha, inverse-map e combinação de dono único sem atomics | Um download de E+1 offsets ainda agenda GEMMs de experts selecionados; não é execução integralmente autônoma |
| Cargas MoE | Redução ordenada e acumulação no dispositivo durante microbatches; leitura na finalização ou auditoria | `expert_loads` durante acumulação requer materialização explícita para telemetria |
| TTT | Erro, norma e atualização recorrente na GPU; VJP das keys num kernel por tensor | Mantém backward truncado existente, não diferencia a adaptação através dos tokens; dois launches por token |
| Attention | Softmax online forward, dQ e dK/dV por dono em blocos de 8; LSE e saída lineares | FP32 no núcleo, GQA, causal, janela, padding; head_dim≤256; sem matrizes globais S² |
| Segurança | Identidade dos caminhos ativos no checkpoint e parsing estrito de flags | Mudança de trajetória deve recusar retomada incompatível |

Mamba em S=512 reduz **apenas o histórico salvo** de 512 para 16 estados por
canal: fator 32. O workspace de carry/partials, outros caches e recomputação
continuam existindo; isso não implica redução de 32× no pico total de VRAM.

## Flags e reprodução

`NSOS_MAMBA_BOUNDARY_HISTORY`, `NSOS_DEVICE_GRAD_CLIP`,
`NSOS_MOE_ORDERED_DEVICE`, `NSOS_TTT_DEVICE_RECURRENCE`,
`NSOS_ATTN_TILED_TRAINING`: somente `0` ou `1` (vazio usa default off).

`train_ptbr_conversational.py --gpu-training-profile redesign-v1` ativa os cinco
caminhos antes de construir modelo/Trainer, verifica elegibilidade e registra
`effective_gpu_training_policy.json`. O default `inherit` conserva políticas
manuais; `legacy` desliga explicitamente os cinco caminhos. Identidades nativas
de execução continuam sendo a autoridade para compatibilidade de checkpoint.

O launcher do piloto oferece `redesign-fp32` e `redesign-bf16`, sem mudar seu
default `chunked-bf16`. `-DryRun` imprime o plano sem lançar processo nem criar
workspace. `-DiagnosticTiming` é explícito; passos limitados não ativam fences
de medição automaticamente. Exemplo **de inspeção**, não de treino iniciado:

```powershell
.\OXN\nsos\scripts\start_ptbr_verified_pilot.ps1 -Action train `
  -Profile redesign-bf16 -RunName redesign-candidate -MaxTrainSteps 200 -DryRun
```

`bench_training_engineering.py` oferece profiles `boundary-bf16`, `clip-bf16`
e `redesign-bf16`, com controles FP32 e kernel flags explícitas. A medição
principal não insere fences diagnósticas. `audit_training_integrations.py
--redesign` exercita os seis braços em processo separado. Os tokens desse
audit são sintéticos: testam integração, não inteligência.

## Lacunas de engenharia ainda abertas para v1

- **Gradientes:** BPTT TTT completo dentro de uma sequência já tem contrato,
  referência independente e finite differences no lote posterior. Continua
  opt-in, com estado inicial destacado e QAT/STE separado; ganho de qualidade
  e BPTT entre chamadas não estão certificados.
- **Throughput:** os kernels tiled são exatos e lineares em memória, mas não
  garantidamente mais rápidos que BLAS em sequências curtas. Medir antes de promover.
- **MoE:** grouped GEMM com counts no device eliminaria a última ida de offsets
  à CPU; precisa preservar BitLinear/QAT, RMSNorm, magnitude e gradientes do router.
- **Scheduling:** TTT ainda possui launches por token; persistência/graphs exigem
  lifetime, concorrência, cancelamento e buffer reuse seguros, não só captura.
- **Validação:** os gates de kernels, integrações, checkpoints, NaN e trajetória
  de 1000 updates passaram na RX 7600. Ampliar a matriz para outros tamanhos,
  precisões, CUDA e GPUs AMD adicionais; teste finito não certifica gradiente exato.
- **Qualidade:** avaliar loss validada, generalização, português sem boilerplate,
  tarefas verificáveis e contribuição causal de cada ramo. Velocidade não é inteligência.
- **Dados:** o bloqueio anterior da quota SFT continua separado deste trabalho;
  probes descartáveis não significam treinamento de produção iniciado.
- **Observabilidade:** counters de tráfego e parede não equivalem a bandwidth
  física, occupancy ou cache hit medidos em hardware.
- **Release:** Docker e memory checker GPU continuam sem evidência local, além
  dos gates de qualidade e hardware adicionais. Não há declaração de v1 pronta.

## Validação final

Executada após implementar o lote; os erros encontrados nessa fase foram
corrigidos e seus testes repetidos. Artefatos: `../artifacts/gpu_redesign_20260929/`.
Nenhum resultado de kernels anteriores foi reutilizado como certificação.

### Build e identidade

- Windows, Release, clang/Ninja/TheRock, HIP, RX 7600 `gfx1102`, wave32;
  Ryzen `gfx1103` aparece no inventário mas **não está compilado** neste binário.
- HIP: `build-gm-hip/nsos_ext.cp312-win_amd64.pyd`, SHA-256
  `acd38e40c1891e3efea680882d957a9992c8505dd8b81d164b77ceee1f1254fc`.
- CPU: `build-gm-cpu/nsos_ext.cp312-win_amd64.pyd`, SHA-256
  `e9bef4cf0c199b92df4cab39b7f258ce2175a0c88406b9d81ca79105b0a3b292`.
- Ambos compilados com OxtaMem ON. Isso é evidência da configuração integrada,
  não substitui um build limpo CPU/OxtaMem OFF nem CI Linux/Docker.
- As medições registram binário, script, configuração, paths ativos, seed,
  tokenizer, shard, pesos iniciais/finais e recursos. Alterações posteriores
  do loader/launcher não mudaram os kernels nem o SHA nativo; os JSONs conservam
  o hash exato do caller usado, não são reescritos como se fossem execuções novas.

### Testes finais

| Gate | Resultado | Evidência |
|---|---|---|
| CTest HIP completo | 114/114 | `validation-hip-release-final.log` (LastTest.log preservado, 208,21 s no comando) |
| CTest CPU completo | 59/59 | `validation-cpu-release-final.log` (60,55 s) |
| Loader/política/currículo/piloto após alterações Python finais | 6/6 em build HIP | `validation-policy-loader-final.log` |
| Gatekeeper CPU | 3/3: CTest, determinismo, industrial Python | `gatekeeper-cpu-final-corrected.json`; repetição final em `gatekeeper-cpu-release-final.json` |
| Gradcheck dropout com seed fixo | 10/10 repetições | execução `ctest -R '^test_jamba$' --repeat until-fail:10` |
| Benchmark CPU de release | passou limiares e reload determinístico | `benchmark-gate-cpu-release-final.json`; smoke de pesos aleatórios, não qualidade |
| Fuzz tokenizer/checkpoint/HTTP | passou corrupção/rejeição/recuperação `/ready` | `fuzz-cpu-release-final.json` |
| OxtaMem Rust | 19 passaram, 2 benchmarks ignorados explicitamente | `cargo test --locked --all-targets` |
| Clippy | passou `--all-targets -- -D warnings` | componente instalado localmente; sem supressão de warnings de código |
| Python syntax e project boundary | passaram | compileall produto/Rust Python; `check_project_boundary.py` |
| Notebook de produto | JSON v4 e 4 células Python compiladas | 1 notebook, validação estática, **não executado**; notebooks Colab/incubação fora desse gate |
| Docker | indisponível | daemon `dockerDesktopLinuxEngine` ausente; imagem não construída |
| Memory checker GPU / counters físicos | não executado | não há evidência de sanitizer, occupancy, bandwidth ou cache hit físico |

Cobertura dos novos gates nativos:

- Mamba: B=2, N=64, P=64, S=1/33/65, com/sem checkpoint, todos os parâmetros
  e dInput contra CPU; asserts do tamanho compacto antes/depois do backward.
- Norma/clipping: cauda de chunks, soma FP64, aplicação e NaN deferido sem
  alteração de gradientes; AdamW por 1000 updates comparando clip legado/device.
- Attention: referência independente double de forward/VJP, GQA, caudas
  ímpares, padding vazio, janelas 1/9/INT_MAX, S=9/17/33, head_dim=33/64/128;
  tolerâncias de 2e-5/3e-5, sem relaxamento para aceitar resultados novos.
- MoE: forward/backward do híbrido, router-task VJP, redução ordenada de cargas,
  acumulação/finalização/cancelamento; routing ausente não retorna gradiente falso.
- TTT: ordinário/Hamiltoniano, com/sem clipping, hidden ímpar e 17 tokens;
  output, dInput, parâmetros, continuação/snapshot/reset contra CPU. Essa
  referência valida **o contrato truncado**, não a derivada da adaptação completa.
- Checkpoint: continuação determinística, policies ativas na identidade e
  recusa segura de não finitos antes de atualizar parâmetros.

### Medição principal — memória e velocidade

Mesmo binário, seed 7301 e pesos iniciais; processos sequenciais frescos,
60 passos por caso, 10 warmups excluídos, batch 1, S=512, D=768, 16 camadas,
N=64, P=64, vocab=16384. São **71.419.264 elementos registrados**, 71.245.696
treináveis; não chamar esse piloto de 40M. Mamba-only, sem attention/MoE/KAN/TTT,
QAT desligado, GEMMs BF16, masters/gradientes FP32. Download de pesos e leitura
WDDM fora da região medida, sem fences diagnósticas por bucket.

| Profile | tok/s p50 | tok/s agregado | Pico live runtime MiB | Reservado runtime MiB | Histórico por camada MiB |
|---|---:|---:|---:|---:|---:|
| `chunked-bf16` | 789,78 | 784,17 | 5374,375 | 5438,5 | 192 |
| `boundary-bf16` | 1663,30 | 1666,41 | 2398,375 | 2462,5 | 6 |
| `clip-bf16` | 807,74 | 803,43 | 5374,375 | 5438,5 | 192 |
| `redesign-bf16` | 1732,45 | 1741,08 | 2398,375 | 2462,5 | 6 |

Arquivos `<profile>-final.json`. Ganho p50 combinado **2,19× nas condições
atuais da máquina**; redução de pico live de aproximadamente 55,4%.
O histórico agregado teórico das 16 camadas cai de 3 GiB para 96 MiB; o campo
`faithful_peak_state_history_bytes` é o **máximo de uma camada**, não a soma.
O pequeno delta de clipping isolado (2,27%) não tem intervalo de confiança e
não é tratado como prova robusta de speedup.

Máximo delta absoluto de loss nos 60 passos: `0.00022602081298828125`.
Isso é controle de trajetória, não perplexidade/generalização certificada.
Mesmo SHA final entre dense/clip e entre boundary/redesign; dense e boundary
não são bitwise equivalentes, devido à ordem de recomputação/redução.

- SHA dos pesos iniciais dos quatro casos:
  `bcc7b8ad57f4764b632f8c2c50ed5e5b73523b243416cc3339d061f243f06e0d`.
- SHA final dense/clip:
  `b8bef9e2930c64495fe3b9b068f97e3999c3aed76a745b494ba2083db658ec8b`.
- SHA final boundary/redesign:
  `e20fdc1a436f1236d406e289619b4c3de3aaf1b4c6d50019c2343cd290a41654`.
- Shard base do piloto verificado: SHA
  `5d84f01dd0b6551d0e2008d3eada22c5dbf7ca49e3e47e69eebec5912d3210c0`.
- Tokenizer: SHA
  `6993b4ed5534dacc44623e483c70dc71b2db76ea184e97eac6d7f406ed986b35`.

No piloto Mamba, ainda há 4 downloads/20 bytes e 4 stream-syncs por step:
controles de segurança/telemetria não foram removidos para produzir velocidade.
Zero device-wide sync reportado não significa zero sincronização CPU/GPU.

**Condições e comparação histórica:** WDDM mostrou possível pressão de memória
externa e uso shared, mas o mapping LUID→HIP não está certificado. Não atribuir
causalidade ou inventar cache/bandwidth físicos. Um A/B contemporâneo com o
binário anterior mediu 310,80 tok/s, contra 314,05 no novo caminho dense:
`reference-before-check.json`/`dense-specialized-check.json`. Portanto, não há
regressão de código demonstrada contra os ~2400 tok/s históricos, obtidos em
outras condições. Os resultados atuais tampouco superam aquele throughput
absoluto. Instrumentação por bucket altera scheduling e não substitui medição
principal. Repetições alternadas e máquina isolada continuam necessárias.

### Integrações — ganhos e regressões explicitamente observados

14 probes FP32 passaram (`executed_finite`), com gradientes presentes e não
zero em cada ramo solicitado; 6 probes adicionais BF16/S=256 também passaram.
Essas verificações fecham plumbing de gradientes, não sua certificação matemática
nem inteligência. CTest independente é a evidência de paridade correspondente.

FP32, casos pequenos, 5 passos/1 warmup, arquivos
`integration-<variant>-legacy.json` e `integration-<variant>-redesign.json`:

| Integração | Legacy ms/step | Redesign ms/step | D2H calls/bytes legacy→redesign | D2D calls legacy→redesign | Stream syncs legacy→redesign |
|---|---:|---:|---|---|---|
| Mamba S=32 | 7,57 | 9,19 | 4/20→4/20 | 6→6 | 4→4 |
| Attention S=32 | 5,10 | 6,16 | 4/20→4/20 | 16→16 | 4→4 |
| Paralelo S=32 | 7,33 | 10,16 | 4/20→4/20 | 18→18 | 4→4 |
| MoE S=32 | 66,07 | 12,25 | 12/49704→7/60 | 296→50 | 279→8 |
| KAN S=32 | 9,40 | 11,60 | 8/36→8/36 | 6→6 | 4→4 |
| TTT S=32 | 31,14 | 10,22 | 36/148→4/20 | 149→21 | 4→4 |
| Attention S=2048 | 123,07 | 67,22 | 4/20→4/20 | 16→16 | 4→4 |

MoE ~5,39×, TTT ~3,05×, attention longo ~1,83× no **profile combinado**.
O caso attention longo também ativa boundary history/clipping; não atribuir
todo o ganho só ao kernel attention. Pico live do híbrido longo:
293,375→154,5625 MiB. Casos curtos Mamba/attention/paralelo/KAN ficaram mais
lentos; os defaults não foram promovidos. Falta seleção por geometria, com
identidade versionada e evidência mais longa, sem esconder esses resultados.

### Correções adicionais descobertas na validação de engenharia

- Loader compartilhado usa os sufixos ABI deste Python, carregamento direto do
  arquivo selecionado e rejeição de módulo já carregado de outro build.
  `--build-dir` explícito inválido não cai silenciosamente em outra build.
  Aplicado ao treinamento, benchmark, fuzz, determinismo e industrial Python.
- Identidade path/SHA do binário efetivo nos relatórios benchmark/fuzz;
  determinismo identifica o artefato efetivo em stdout do gatekeeper.
- Fuzz HTTP antes nem iniciava (D=64 com default 12 heads e modelo ausente).
  Agora cria pack autocontido, usa token só no ambiente, exige rejeição de método
  inválido/JSON inválido e `/ready` após as requisições malformadas.
- Gradcheck de dropout antes intermitente por inicialização não semeada.
  Agora seed fixo, soma double do objetivo, finitude explícita e mesma tolerância
  2e-2. Não é um aumento de tolerância nem evidência nova de BPTT TTT.
- Clippy sem warnings em todos os targets; open flags redundantes e condições
  simplificadas preservando durabilidade/rollback. Lane Rust repetida após edits.
- RELEASE/CI/boundary não pedem mais compileall do SDK Python duplicado removido.

## Próximos limites concretos — implementação integral ainda não concluída

| Prioridade | Lacuna | Critério para fechar |
|---|---|---|
| P0 qualidade | Piloto real ainda bloqueado pela quota SFT filtrada; não foi lançado treino de produção neste lote | Dataset admissível com holdout preservado, métricas masked-answer/teacher-token/boilerplate melhores que baseline |
| Fechado opt-in TTT | Novo contrato de BPTT por sequência e isolamento de rank3 implementado/validado; legado conserva sua semântica | Ver relatório TTT; qualidade, serving concorrente e BPTT entre chamadas não fechados |
| P0 promoção GPU | Sem memory checker e sem matriz multi-hardware | Checker apropriado ao backend, limites/caudas/concorrência e CI real; manter Experimental até lá |
| MoE opt-in subsequente | Grouped GEMM segmentado e integração implementados; registro tardio/aux ainda CPU, matriz larga mais lenta | Ver [relatório MoE](GPU_MOE_GROUPED_TRAINING_2026-09-30.md); aceleração de GEMM largo, atividade/Adam device-side e balanceamento sem round-trip continuam abertos |
| P1 TTT | Novo histórico por fronteiras/recomputação e q distribuído; forward single-CTA e launches por token persistem | Fusões/captura seguras, medição separada e ablação de qualidade sob mesmo orçamento |
| P1 seleção | Redesign pior em contexto curto | Dispatch por geometria com baseline pareado, paths ativos e identidade de checkpoint, sem regressão escondida |
| P1 KAN | Downloads pequenos continuam no probe; origem ainda não isolada | Trace atribuível por operação e remoção segura; não atribuir causa só a counters globais |
| P1 estado/runtime | Scratch é por execution lane; sessões e pesos não são isso | Exercitar cancelamento, lifetime de graph, múltiplas sessões/streams e concorrência end-to-end |
| P1 desenho | Complexidade concentrada em jamba.cpp/trainer.cpp/sdk/HTTP | Extrações pequenas com cobertura, preservando contracts, sem reescrita monolítica |
| P1 release | Docker indisponível e falta clean lane independente | Build limpo CPU OFF/OxtaMem separado, Docker/CI Linux com mesma matriz e evidência |
| P2 pesquisa | Persistência, KV agressivo, aproximações, especulação não certificados | Um desenho por vez, qualidade/segurança antes de promover; não alterar exatidão sob flag de otimização |

O conjunto de testes acima melhora a confiança de engenharia. Nenhum número
de inteligência foi inferido de throughput, quantidade de camadas, loss finita
ou gradiente não zero. `PRODUCT.md` permanece a autoridade de capacidades;
este documento não promove GPU/TTT nem declara todas as melhorias concluídas.
