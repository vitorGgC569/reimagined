# NSOS runtime GPU — fechamento industrial e matriz de rastreabilidade

Data: 2026-08-01  
Escopo: runtime e pipeline de treinamento `OXN/nsos`  
Hardware dinâmico: AMD Radeon RX 7600, `gfx1102`, HIP/TheRock  
Fontes congeladas: “Auditoria estática de desempenho GPU — NSOS” e
“Relatório complementar — encerramento da investigação estática GPU”.

Identificação SHA-256 das fontes autoritativas usadas no fechamento:

- auditoria principal, 447 linhas:
  `CB56695693E85FC03F5B91E738CC33D5DAA07EEE6F3E8B204962A03EAF37A521`;
- relatório complementar, 747 linhas:
  `DDEB1AC16FF851BC3AF1ECAA9EEE70FACDC6E85C680D19FC0EF8BC756037EC94`;
- especificação de implementação, 666 linhas:
  `FACC78C8676A3942973E4A73A8E460BF4BEF99DCCEFDF6A8A930250950321F59`.

## 1. Resumo executivo

O pacote implementável das duas auditorias foi integrado ao caminho de
produção. O runtime agora possui identidade de execução imutável, política de
precisão congelada durante operações, embedding backward determinístico por
IDs presentes, AdamW determinístico multi-tensor, `matmul_tn` nativo, cache
lowp versionado, selective scan state-major/chunked com rollback, checkpoint
host-owned assíncrono, telemetria rolling, leitura de shards por `mmap` e
contratos fail-closed.

A segunda revisão estática transversal também fechou caminhos que não estavam
na primeira matriz: o `DataLoader` legado deixou de fabricar dados aleatórios,
o `SmartLoader` passou a possuir e drenar integralmente suas requisições, MPI e
Fabric deixaram de simular execução distribuída, monitor e Lean não retornam
sucesso fictício, SDK/trainer propagam poison após falha parcial e o MCTS
rejeita cardinalidade e valores não finitos sem fallback escalar silencioso.

O build limpo foi realizado em diretório virgem com warnings C++/HIP tratados
como erro. Foram compilados 216 alvos e executados 95 testes: 95/95 passaram.
A mensagem remanescente do Rust é informativa do linker ao criar a import
library do OxtaMem; não é warning de código NSOS.

O gate do treino real foi repetido três vezes em linhagens novas, sem resume.
Cada corrida executou 1.000 passos; os passos 1–100 foram descartados e os
percentis foram calculados nos 900 passos restantes.

| Corrida | p10 raw tok/s | p50 raw tok/s | p90 raw tok/s | Loss eval inicial | Loss eval final |
|---|---:|---:|---:|---:|---:|
| A | 1.570,216 | 1.576,000 | 1.582,446 | 9,947924 | 4,084897 |
| B | 1.572,989 | 1.578,260 | 1.583,853 | 9,947924 | 4,084897 |
| C | 1.570,126 | 1.575,656 | 1.581,146 | 9,947924 | 4,084897 |
| Pós-hardening | 1.573,056 | 1.579,679 | 1.585,138 | 9,947924 | 4,084897 |
| RC1 final | 1.567,693 | 1.573,204 | 1.578,875 | 9,947924 | 4,084897 |
| Final A | 1.569,930 | 1.574,698 | 1.580,459 | 9,947924 | 4,084897 |
| Final B | 1.570,448 | 1.575,836 | 1.581,381 | 9,947924 | 4,084897 |
| Final C | 1.570,470 | 1.575,761 | 1.581,390 | 9,947924 | 4,084897 |
| Final2 A | 1.571,878 | 1.577,825 | 1.584,573 | 9,947924 | 4,084897 |
| Final2 B | 1.570,007 | 1.575,646 | 1.581,359 | 9,947924 | 4,084897 |
| Final2 C | 1.569,877 | 1.575,367 | 1.580,871 | 9,947924 | 4,084897 |

No gate original, o p50 médio foi 1.576,639 tok/s, desvio-padrão populacional 1,155 tok/s,
coeficiente de variação 0,0733% e spread total 0,1652%. As três corridas
superaram o limiar pré-registrado de 1.500 tok/s.

No binário final, as três novas linhagens A/B/C produziram p50 médio de
1.575,432 tok/s, desvio-padrão populacional de 0,520 tok/s, coeficiente de
variação de 0,0330% e spread total de 0,0722%. Todas superaram 1.500 tok/s.

As três trilhas originais de 1.000 losses, `model.bin`, `trainer.state` e
identidades de runtime foram idênticas bit a bit:

- trace SHA-256:
  `ab386f4e9ec222292e0d40c4b78af2dad9576bbced3b9e7030ddeae5562c4265`;
- model SHA-256:
  `ba541feedee22119ce01c0953e10e8245b0aaeeb3b5d7bda92e875b1e89475d6`;
- trainer SHA-256:
  `78998269bf42f97111ce22f9da55475cdf2805306f0832590972d139e5664f1a`;
- runtime identity SHA-256:
  `e6a840f44fc13d6221bce34693aa65225a77ff44f7f69f67e7706c820a839cc9`.

Na linhagem Final2, o p50 médio foi 1.576,279 tok/s, desvio-padrão
populacional 1,099 tok/s, coeficiente de variação 0,0697% e spread total
0,1559%. As três corridas permaneceram acima de 1.500 tok/s, com qualidade,
stderr e identidades bit a bit estáveis.

Nas corridas pós-hardening e RC1, os hashes de trace, modelo e trainer
continuaram exatamente iguais aos três originais. A identidade semântica dos
66 campos permaneceu
`4ad0ed394b5db0eaa8ba9e8dd83b16b3f0a685177302b363347062ab8c25dcf1`.
O hash do envelope de checkpoint mudou como esperado, pois inclui a impressão
do módulo compilado e detecta qualquer alteração de binário, inclusive o
hardening do MCTS que não participa do treino Mamba-only.

O candidato definitivo usa:

- módulo SHA-256
  `d3c138dce111b7d5c7a701f0eae1e6a641825640f2145670e36f9f838ba55167`;
- script SHA-256
  `8b8c495b98bb178e4305fd3f08a09509badbd54835f0a32f1e60a4981b46886a`;
- identidade de checkpoint
  `c7fcbe98b50618fb6d4bec7ae2d20c39c81be619a6f0966790c00331ef459fbb`.

O módulo `071562...` e a identidade `2fd196...` permanecem registrados apenas
como linhagem histórica pré-transversal; não são o candidato atual.

Todas as corridas apresentaram zero fallback Mamba, zero fallback host, zero
memória gerenciada, zero sincronização global de device, zero falha do pool e
zero NaN/Inf. O pico reservado observado foi 5.712.904.192 bytes (5,32 GiB).

O soak definitivo, executado no mesmo módulo final, concluiu 10.000/10.000
passos e os dois cruzamentos de shard (`3907` e `7813`). Nos primeiros 1.000
passos, após descartar 100 warmups, obteve p10/p50/p90 de
1.570,167/1.575,982/1.581,116 tok/s; no conjunto dos 10.000 passos sustentou
1.577,633 tok/s e 3,0813 passos/s. A loss held-out caiu de 9,947924 para
3,225419 (`-67,5770%`). Os dez snapshots foram publicados, quatro gerações
foram retidas e verificadas, nenhuma foi inválida e o `stderr` ficou vazio.
O trace dos primeiros 1.000 passos, o modelo final e o estado do trainer são
idênticos bit a bit ao soak independente anterior de 10.000 passos.

O soak atual `final-soak-10000-final2-20260802`, executado no módulo atual,
concluiu 10.000/10.000 passos e os dois cruzamentos de shard (`3907` e
`7813`). Nos primeiros 1.000 passos, após 100 warmups, obteve p10/p50/p90 de
1.567,441/1.576,628/1.583,330 tok/s; no conjunto dos 10.000 passos sustentou
1.576,720 tok/s e 3,0795 passos/s. A loss held-out caiu de 9,947924 para
3,225419 (`-67,5770%`). Dez snapshots foram publicados, quatro gerações foram
retidas e verificadas, nenhuma foi inválida e o `stderr` ficou vazio. O modelo,
trainer e trace coincidem bit a bit com a linhagem longa anterior; a identidade
de runtime mudou legitimamente para registrar o módulo endurecido.

## 2. Estados de fechamento

- **FECHADO — PRODUÇÃO**: integrado, revisado e aprovado dinamicamente no
  hardware disponível.
- **FECHADO — REFERÊNCIA/ROLLBACK**: mantido deliberadamente como caminho de
  referência; não é dívida de implementação.
- **EXPERIMENTAL CONTROLADO**: opt-in, identificado e com rollback; não é o
  default de produção.
- **NÃO PROMOVIDO POR RISCO**: hipótese ou mudança de ordem numérica que não
  recebeu evidência suficiente.
- **DEPENDENTE DE HARDWARE/PROFILER**: implementação ou instrumentação fechada,
  mas a medição exige equipamento/ferramenta indisponível.

## 3. Matriz completa de rastreabilidade

| ID | Achado e classe | Implementação / causa encerrada | Invariantes e compatibilidade | Evidência dinâmica | Estado final |
|---|---|---|---|---|---|
| ID-01 | Identidade incompleta `[C]` | `RuntimeExecutionIdentity` schema 3 captura precisão, determinismo, backend, arquitetura, driver/runtime/BLAS, sync, memória, optimizer, finite gate, checkpoint, layouts, scan, GEMM e cache; é persistida e validada no trainer/checkpoint/pipeline | Resume rejeita campo ausente, extra, duplicado ou divergente; checkpoints antigos só entram por migração explícita | `test_runtime_execution_identity`, `test_training_invariants`, três identidades/hash iguais | FECHADO — PRODUÇÃO |
| ID-02 | Precisão global mutável `[C]` | `RuntimeExecutionPolicyLease` e `RuntimeExecutionPolicyMutationGuard` bloqueiam mutação durante step/run; epoch invalida cache | Masters, gradientes e Adam seguem FP32; mudança fora de lease cria nova identidade | teste de lease aninhado e mutação rejeitada | FECHADO — PRODUÇÃO |
| EMB-01 | Backward `O(V·D·P)` `[C]` | CSR conservador por IDs presentes, offsets e posições batch-major crescentes; saída continua densa | Soma inicia em `+0.0f`; ID 0, EOS, repetidos, inválidos, padding, buckets e tied head preservados | paridades CPU/HIP, embedding rollback, checkpoint continuation e três treinos bitwise | FECHADO — PRODUÇÃO |
| EMB-02 | Temporário denso de 48 MiB `[C]` | Temporário denso e `Parameter::add_grad` foram mantidos na lane promovida; escrita direta no gradiente persistente não foi adotada | Evita mudar primeiro uso, ordem de buckets e peso amarrado | pico medido e determinismo sem regressão | FECHADO — REFERÊNCIA/ROLLBACK |
| SCAN-01 | Grid 6×256 subocupado `[C]` | geometria head/channel especializada com fallback linear genérico, runtime warp size e um writer por canal | Ordem de timestep/state/reverse/reduções e streaming preservada | faithful/proper/nstate/stream parity e rollback | FECHADO — PRODUÇÃO |
| SCAN-02 | Decay repetido `[C]` | precompute chunked de termos elegíveis, identificado como `precomputed_chunked_eligible_terms_v3` | reference/rollback mantém cálculo original; não há hipótese mascarada | testes decay opt-in/rollback e treino bitwise | FECHADO — PRODUÇÃO |
| SCAN-03 | Histórico pouco coalescido `[IF]` | layout interno tipado `[row][state][channel]` no treino e LDS state-lane bank-coalesced; produtores/consumidores e recompute atualizados | não expõe view com shape/stride falso; streaming canônico continua separado | history e LDS opt-in/rollback; ganho medido e paridade | FECHADO — PRODUÇÃO |
| SCAN-04 | Pressão de `state[64]`/spills `[IF/ND]` | contadores de variante, layout e tamanho persistidos; nenhuma alegação de ocupação foi inferida | matemática não foi alterada por uma hipótese de compilador | profiler AMD não está instalado; NVIDIA ausente | DEPENDENTE DE HARDWARE/PROFILER |
| SCAN-05 | Scan paralelo/chunked `[H]` | backward determinístico por chunks e fronteiras afins foi promovido somente após lane bitwise; selective scan paralelo geral permanece `false` | chunk size/layout entram na identidade; reference finite-flat e rollback continuam disponíveis | chunked/LDS rollback, 1.000 steps Adam e três treinos bitwise | FECHADO — PRODUÇÃO para chunk determinístico; NÃO PROMOVIDO para scan paralelo geral |
| OPT-01 | AdamW por tensor `[C]` | update determinístico multi-tensor em chunks de 8.192 elementos; registry e descritores canônicos | redução de norma FP64 permanece separada; accumulation e clip continuam materializados; tied weight aparece uma vez | comparação 1/10/100/1.000, checkpoint continuation e hashes | FECHADO — PRODUÇÃO |
| OPT-02 | Finite/clip/commit fragmentados `[C]` | finite scan booleano chunked e gate deferido apenas até consumidor comprovado; status críticos permanecem separados quando a fusão mudaria semântica de falha | nenhuma barreira de segurança removida; precommit e midcommit são distinguíveis; estado ambíguo vira poisoned | combined finite gate, finite rollback, NaN/Inf e poison tests | FECHADO — PRODUÇÃO |
| OPT-03 | Adam comprimido para ~213M `[H]` | Adam FP32 segue autoritativo; 4/8-bit não foi promovido sem campanha numérica longa | formato persistente não mudou por hipótese | testes do optimizer 4-bit são experimentais, não gate do produto | NÃO PROMOVIDO POR RISCO |
| GEMM-01 | Transposes físicas no dW `[C]` | `matmul_tn` nativo em BitLinear, KAN e projeção agrupada; FP32 usa BLAS transpose | shapes e acumulação preservados; materialização é rollback explícito | `test_matmul`, gradcheck e GPU parity | FECHADO — PRODUÇÃO |
| GEMM-02 | `matmul_nt` lowp materializava transpose `[C]` | GemmEx transposto nativo; em HIP, TN lowp usa cast+transpose fused e GEMM NN para contornar `hipErrorInvalidImage` do OP_T em gfx1102 | FP32 reference intacto; FP16/BF16 têm tolerâncias pré-fixadas | mixed precision FP16/BF16 e TN retangular | FECHADO — PRODUÇÃO |
| GEMM-03 | Recast de peso a cada GEMM `[C]` | cache lowp por storage owner, offset, shape, device, mode, policy epoch e content version; orçamento 192 MiB | invalidação por versão/epoch, tied e grouped epoch; budget zero é rollback | lowp cache normal/zero-budget e mixed contract | FECHADO — PRODUÇÃO |
| GEMM-04 | Falta de abstração Lt `[C]` | provider comum CPU/classic/Lt, algoritmo e promoção na identidade; classic hipBLAS/cuBLAS é default | pedir Lt sem provider compilado falha explicitamente | identity/provider tests; Lt não disponível neste SDK | FECHADO — PRODUÇÃO para abstração/classic; DEPENDENTE DE SDK para Lt |
| PROJ-01 | Cinco slices materiais `[C]` | owner empacotado canônico `[z,x,B,C,dt]` e `Tensor::storage_view` validada; cache agrupado por versões | parâmetros mantêm nomes/checkpoint próprios em regiões não sobrepostas; ownership compartilhado evita free por offset | grouped counters 16.256 fwd/16.000 bwd, zero fallback | FECHADO — PRODUÇÃO |
| PROJ-02 | Packed backward transpunha `[C]` | backward agrupado migrou para `matmul_tn` e distribuição canônica | gradientes e versões permanecem na ordem dos cinco parâmetros | faithful parity, gradcheck e treino bitwise | FECHADO — PRODUÇÃO |
| BUF-01 | 3 GiB de histórico vivo `[C]` | checkpoint seletivo recompõe um histórico por camada, libera-o após uso e entra na identidade | formato dos parâmetros não muda; política divergente rejeita resume | selective recompute counters e continuation; default 71M reteve history por caber em 8 GiB | FECHADO — PRODUÇÃO |
| BUF-02 | Wrapper sobrevivia ao backward `[C/IF]` | caches de `JambaBlock` e Mamba são limpos depois do último consumidor, inclusive early return | hooks/fault state são copiados antes da liberação | parity, layer audit e fault tests | FECHADO — PRODUÇÃO |
| BUF-03 | Clones BitLinear `[C]` | clones indispensáveis à imutabilidade do backward foram retidos; remoção sem ownership provado não foi promovida | ausência de alias/UAF tem prioridade sobre economia especulativa | gradcheck, thread safety e long sequence tests | FECHADO — REFERÊNCIA/ROLLBACK |
| BUF-04 | Pool/workspaces sem planner `[IF]` | workspaces thread-local tipados, pool com eventos/streams, métricas live/reserved/cache/fragmentação e integridade fail-closed | nenhuma reutilização entre streams sem evento; captura possui contrato próprio | pico 5,71 GB, zero release/unknown/capture/advice failure | FECHADO — PRODUÇÃO |
| ELEM-01 | RMSNorm/gamma/gate fragmentados `[C]` | SiLU×gate forward/backward foi fundido; RMSNorm/gamma permanece separado onde fusão mudaria ordem de redução | NaN/Inf, shapes e gradientes preservados; CPU reference | faithful parity e gradcheck | FECHADO — PRODUÇÃO para fusão segura; NÃO PROMOVIDO para fusões de redução |
| CE-01 | `exp` duplicado `[C]` | CE fused reutiliza numeradores no grad e normaliza em segundo estágio; policy `one_exp_device_scalar_v1` | FP32 e redução determinística preservadas | CPU/GPU CE parity e training bitwise | FECHADO — PRODUÇÃO |
| CE-02 | Loss D2H antes do backward `[C]` | `cross_entropy_device` mantém escalar no device; `train_step` baixa somente após backward/optimizer | API pública float continua wrapper compatível | teste prova zero D2H antes do reporting | FECHADO — PRODUÇÃO |
| CONV-01 | Conv backward K16 para K4 `[C]` | especialização K4 com fallback genérico e identidade | mesma ordem/redução para K4; outros K usam referência | faithful parity e rollback | FECHADO — PRODUÇÃO |
| CKPT-01 | Checkpoint síncrono `[C]` | captura host-owned coerente; um worker escreve, copia, hasheia, fsynca e só então publica `COMMITTED` | somente um snapshot em voo; falha poison impede novo snapshot | snapshot/write interruption, worker failure, rename fallback e três verifies | FECHADO — PRODUÇÃO |
| CKPT-02 | Snapshot extra em VRAM `[C]` | transporte `host_owned_one_inflight_v1`; não existe clone completo adicional na GPU | modelo continua bloqueado apenas durante captura coerente | telemetria de checkpoint e pico de VRAM | FECHADO — PRODUÇÃO |
| TEL-01 | Throughput cumulativo/SFT incompleto `[C]` | rolling window e agregados por fase para raw/non-padding/supervised/assistant tokens e steps/s | amostras do gate são limitadas e timing-free para determinismo | 900 amostras por corrida e percentis p10/p50/p90 | FECHADO — PRODUÇÃO |
| TEL-02 | Tempos de fase incompletos `[C/ND]` | timers forward/loss/backward/optimizer/checkpoint opt-in; backend, precisão, algoritmo e flags persistidos | timing nativo fica desligado no gate bitwise para não inserir fences | timing contract tests e `checkpoint_timing` | FECHADO — PRODUÇÃO |
| TEL-03 | Fragmentação/launches `[ND/MF]` | pool publica allocated/reserved/cache/largest/fragmentation/peak e falhas; runtime publica chamadas observáveis | launches internos do BLAS não são inventados | métricas reais dos três treinos; profiler ausente | FECHADO para telemetria observável; DEPENDENTE DE PROFILER para launches internos |
| DATA-01 | Shards viravam listas Python `[C]` | `U16TokenShard` e `SFTRecordShard` usam `mmap`, offsets compactos e decodificação por janela/registro | hash verificado antes do uso; ordem, offset e resume preservados | script tests e três treinos atravessando leitura real | FECHADO — PRODUÇÃO |
| DATA-02 | `DataLoader` legado fabricava amostras aleatórias em erro `[C]` | implementação duplicada foi removida; arquivo inexistente, vazio, truncado, dimensões inválidas e falha do worker terminam com erro explícito | nenhuma amostra sintética pode entrar silenciosamente no treino | `test_dataloader` cobre integridade, término e limites | FECHADO — PRODUÇÃO |
| IO-01 | `SmartLoader` tinha ownership e término incompletos `[C/IF]` | fila possui `unique_ptr`, leitura binária posicionada/exata, contador protegido, `drain` e destrutor cumprem todas as promises | short read, close, offset/size inválido e destino GPU incompatível são observáveis | `test_smart_loader_async`, inclusive destruição com requisições em voo | FECHADO — PRODUÇÃO |
| CKPT-03 | Limpeza residual podia falhar sem diagnóstico `[C]` | falhas ao remover temporário atômico ou `.staging-*` são registradas em `stderr` sem mascarar o erro original nem despublicar `COMMITTED` durável | resume ignora staging residual e continua exigindo manifesto/hashes | AST Python, testes de publicação/fallback/worker e verify | FECHADO — PRODUÇÃO |
| SAFE-01 | Falha fechada transversal `[C/IF]` | validações de pointer/shape/dtype/device/overflow/backend/copy/kernel/write; checkpoint e optimizer transacionais | nenhum erro crítico vira fallback; estado ambíguo é poisoned | NaN/Inf, OOM, truncamento, corruption, snapshot/write e pool tests | FECHADO — PRODUÇÃO |
| SAFE-02 | Monitor e Lean podiam comunicar sucesso sem prova `[C]` | flags do monitor são atômicas, cópia D2H é verificada e NaN/Inf lança; Lean indisponível retorna estado explícito e `verify` lança em vez de retornar `true` | indisponibilidade não é prova e erro GPU não é omitido | suíte completa warning-as-error | FECHADO — PRODUÇÃO |
| SAFE-03 | Cleanup tardio podia deixar estado reutilizável `[C/IF]` | falhas no cleanup pós-commit poisonam trainer; streaming poison/restore do `InferenceEngine` é transacional; deleters GPU registram falhas não lançáveis | estado ambíguo é fail-stop, nunca reutilizado como íntegro | fault injection, clone/SDK/E2E e suíte completa | FECHADO — PRODUÇÃO |
| SAFE-04 | Header legado CPU declarava mocks CUDA com sucesso fictício `[C]` | as funções simuladas foram removidas; build CPU conserva apenas o tipo cujo `initialize()` falha fechado; a lane GPU usa alocação device-only e transferência explícita | nenhuma API ausente pode parecer execução GPU e Unified Memory não entra por esse allocator | varredura de chamadores, rebuild warning-as-error e suíte integral | FECHADO — PRODUÇÃO |
| SAFE-05 | Exceções Python e cleanup/graph GPU ainda tinham retornos ignorados `[C]` | shards registram falha de fechamento; `LATEST.json` inválido avisa antes do scan; override curricular inválido/não finito falha; avaliação do probe não é omitida; upload de graph é obrigatório e destruições GPU reportam erro em contexto noexcept | erro de diagnóstico não vira amostra/resultado válido e cleanup não mascara a exceção primária | AST, testes específicos, decode/inferência, E2E e suíte integral | FECHADO — PRODUÇÃO |
| SAFE-06 | Utilitários legados ainda continham no-op, métrica fabricada ou protótipo incompleto `[C/IF]` | scratchpad RIERASS concatena todos os pensamentos e rejeita overflow/shape/device; dot-product ternário executa quatro lanes; isolamento GPU de teste recusa no-op; DPO/SFT exigem APIs reais; falhas auxiliares são propagadas ou registradas | ausência de API não vira loss/score zero; benchmark sem input válido falha; caminho Mamba-only permanece matematicamente inalterado | `test_sanity`, BitNet/DP4A, dois probes fail-closed, AST dos 83 scripts/testes Python e suíte 95/95 | FECHADO — PRODUÇÃO/TOOLS |
| PORT-01 | wave32/wave64/warp32 `[C]` | backend comum, runtime warp size, primitives e fallback genérico; HIP/CUDA compartilham contratos sem dependência vendor no common code | nenhuma suposição de “64 threads = uma wave” | CPU + HIP wave32 aprovados; CUDA/wave64 sem hardware local | FECHADO estaticamente; DEPENDENTE DE HARDWARE para CUDA/wave64 real |
| PORT-02 | Fabric/MPI simulavam sucesso distribuído `[C]` | build sem MPI aceita somente rank 0/world 1 e rejeita coletivas; build MPI duplica communicator, exige thread level, valida status/count/rank e usa staging explícito | execução distribuída solicitada sem provider falha; não há no-op que pareça coletivo | caminho sem MPI coberto e build limpo; toolchain/runtime MPI ausente para teste multi-rank | FECHADO para fail-closed local; DEPENDENTE DE AMBIENTE para MPI real |
| INF-01 | SDK/inferência podia publicar estado parcial `[C/IF]` | KV reserve e streaming falham fechados; moves/publicação são transacionais; falha de cleanup marca `InferenceEngine` poisoned | clone, treino e nova geração rejeitam engine poisoned | testes SDK, model pack, streaming e E2E | FECHADO — PRODUÇÃO |
| REASON-01 | MCTS aceitava batch inválido ou fazia fallback escalar silencioso `[C]` | evaluator escalar e batch passam por validação central; cardinalidade divergente e NaN/Inf lançam; `rollout` morto foi removido | nenhuma avaliação ausente é substituída e device da value head é preservado | `test_mcts_reasoning_v3` e `train_e2e` | FECHADO — PRODUÇÃO |

### 3.1 Arquivos, funções, impacto e riscos por família de achados

A tabela anterior contém achado/classificação, causa, correção, invariantes,
validação e estado. Este complemento fecha explicitamente as colunas de
arquivo/função, impacto, risco numérico e risco de compatibilidade exigidas
pela especificação. Cada ID abaixo referencia as linhas homônimas acima.

| IDs | Arquivos e funções principais | Impacto encerrado | Risco numérico | Risco de compatibilidade | Validação exigida/realizada |
|---|---|---|---|---|---|
| ID-01/02 | `runtime_execution_identity.{h,cpp}` (`capture/require`), `trainer.cpp` (`save/load_training_state`), pipeline (`build_runtime_identity`) | resume com semântica diferente parecia pertencer à mesma linhagem | alto: política pode reordenar FP32 | alto: schema de checkpoint; migração explícita | campos exatos, digest, campo ausente/extra/duplicado e três identidades finais |
| EMB-01/02 | `embedding.cpp` (`backward_batch`), `kernels.cu` (CSR launchers), `trainer.cpp` (tied registry) | custo `V×P` e risco de ordem em IDs repetidos/tied head | crítico: soma FP32 e `+0.0f` | médio: gradiente denso e nomes persistidos preservados | CPU/HIP, IDs 0/EOS/inválidos/repetidos, rollback e hashes de treino |
| SCAN-01…05 | `mamba2.cpp` (`forward/backward_faithful`), `mamba_kernels.cu` (head/channel, chunk, LDS, K4) | subocupação, layout não coalescido e históricos excessivos | crítico: ordem temporal/state/reverse/carry | alto: layouts/chunk entram na identidade; streaming preservado | paridades proper/nstate/stream/faithful, opt-in/rollback e campanha longa |
| OPT-01…03 | `trainer.cpp` (`apply_optimizer_step`), `fused_optimizer_kernels.cu` (finite/norm/update) | launches por tensor, commit parcial e finite status ambíguo | crítico: norma FP64, clip e update FP32 | crítico: `m/v/global_step/version` e resume | 1/10/100/1.000, NaN/Inf/poison e continuation bitwise |
| GEMM-01…04 | `tensor.cpp` (`matmul_tn`, mixed NT/TN, cache), `gpu_gemm_provider.cpp`, `bitlinear.cpp`, `kan.cpp` | transposes/casts físicos e acoplamento a fornecedor | alto em lowp; FP32 bitwise | médio: provider/algoritmo e cache entram na identidade | SGEMM referência, FP16/BF16 com tolerâncias prévias, cache/rollback e shapes retangulares |
| PROJ-01/02 | `mamba2.cpp` (grouped owner/views, pack/unpack/backward) | cinco materializações e transpose no backward | alto: distribuição dos cinco gradientes | alto: nomes e regiões dos parâmetros não podem mudar | storage-view bounds, grouped counters, gradcheck e faithful parity |
| BUF-01…04 | `mamba2.cpp` (recompute/cleanup), `tensor.cpp` (pool/workspaces), `jamba.cpp` (cache lifetime) | pico de VRAM, UAF/alias e reutilização cross-stream | alto: recompute deve manter ordem | médio: formato de parâmetros intacto; policy na identidade | selective continuation, pool fault/capture, pico VRAM e zero falha |
| ELEM-01 | `tensor.cpp`/`kernels.cu` (`silu_gate` forward/backward), RMSNorm de referência | launches elementwise evitáveis | médio: sigmoid/gate e dgamma | baixo: shapes/APIs preservados | CPU/HIP parity, gradcheck e rollback das fusões não promovidas |
| CE-01/02 | `tensor.cpp` (`cross_entropy_device`), `trainer.cpp` (`train_step`) | `exp` duplicado e D2H/barreira prematuros | crítico: softmax/redução/loss | baixo: wrapper host público mantido | CE CPU/HIP, zero D2H antes do reporting e hashes finais |
| CONV-01 | `mamba_kernels.cu` (faithful K4 backward) | kernel genérico caro no shape produtivo | alto: redução K4 | baixo: fallback genérico permanece | faithful parity, counters e rollback |
| CKPT-01…03, IO-01 | `checkpoint_io.h`, `train_ptbr_conversational.py` (`CheckpointManager`, `atomic_write_json`), `smart_loader.cpp` | pausas, publicação parcial, promessa perdida e erro de cleanup omitido | nenhum no caminho saudável; alto para consistência do snapshot | crítico: manifesto/hashes/`COMMITTED`/resume | snapshot/write interruption, worker poison, corrupt pointer, fallback rename e verify |
| TEL-01…03 | `train_ptbr_conversational.py` (`RollingTrainingTelemetry`), `bindings.cpp`, `tensor.cpp` (pool/transfers) | média cumulativa ocultava regressão e recursos não eram auditáveis | baixo; timing fica off no gate bitwise | baixo: schema versionado nos artefatos | 900 amostras por corrida, fases separadas, recursos/counters reais |
| DATA-01/02 | pipeline (`U16TokenShard`, `SFTRecordShard`), `dataloader_v2.cpp`, `smart_loader.cpp` | RAM inflada, dados sintéticos em falha e shard truncado aceito | alto: ordem de amostras altera treino | alto: SHA/manifest/offset/resume | corpus/pack SHA, mmap real, missing/empty/truncated e fronteiras de shard |
| SAFE-01…06 | `tensor.cpp`, RAII CUDA, `trainer.cpp`, `nsos_sdk.cpp`, monitor/Lean, RIERASS/BitNet e scripts | erro crítico omitido, estado parcial reutilizado, métrica fabricada ou sucesso fictício | crítico nos gates finite/commit | alto: estados poisoned são fail-stop por contrato | fault injection, warning-as-error, testes específicos, AST integral, suíte 95/95 e stderr dos gates |
| PORT-01/02 | `gpu_backend.{h,cpp}`, CMake, kernels comuns, `fabric_v2.cpp`, `nsos_mpi.cpp` | suposição warp/wave e coletivas simuladas | alto para reduções entre arquiteturas | alto: backend/arch/provider na identidade | CPU+HIP wave32 reais; CUDA/wave64/MPI real classificados por hardware ausente |
| INF-01 | `jamba.cpp` (KV/decode graph), `nsos_sdk.cpp` (`InferenceEngine`) | buffer/publicação parcial e engine reutilizada após cleanup falho | alto para logits/estado KV | médio: model pack e clone preservados | inference/decode parity, SDK/model-pack, graph self-test e E2E |
| REASON-01 | `mcts_reasoning.cpp` (`evaluate_state(s)`, value head), teste v3 | batch inválido virava avaliação diferente sem aviso | médio: valor altera seleção da árvore | baixo para treino Mamba-only; API agora estrita | cardinalidade, NaN/Inf, device correto e E2E |

### 3.2 Crosswalk integral das duas auditorias

| Fonte | Seções da fonte | IDs de fechamento |
|---|---|---|
| Auditoria principal | 2–3, mapa/caminho crítico | DATA, PROJ, BUF, CE, OPT, CKPT |
| Auditoria principal | 4, gargalos | EMB, SCAN, OPT, GEMM, BUF, ELEM, CONV, TEL |
| Auditoria principal | 5, kernels | EMB, SCAN, ELEM, CE, CONV, PORT |
| Auditoria principal | 6, memória | BUF, GEMM-03, CKPT-02, DATA-01 |
| Auditoria principal | 7, sincronizações/launches | CE, OPT-02, CKPT, TEL |
| Auditoria principal | 8, Mamba/selective scan | SCAN-01…05, BUF-01, CONV-01 |
| Auditoria principal | 9, GEMM/MFMA/Tensor Cores | GEMM-01…04, PORT-01, SCAN-04 |
| Auditoria principal | 10–13, plano/validação/escala/desempenho | seções 5–7, 11 e 13 deste fechamento |
| Auditoria principal | 14, itens ND/MF | SCAN-04, TEL-03, PORT-01 e seção 10 |
| Relatório complementar | 2, embedding | EMB-01/02 |
| Relatório complementar | 3, selective scan | SCAN-01…05 |
| Relatório complementar | 4, AdamW | OPT-01…03 |
| Relatório complementar | 5, buffers/lifetimes | BUF-01…04, PROJ-01/02, CKPT-01/02 |
| Relatório complementar | 6, mixed precision | ID-01/02, GEMM-02…04 |
| Relatório complementar | 7–8, dependências e ordem | mapeamento dos 15 grupos e histórico de implementação |
| Relatório complementar | 9, A/B | seções 5–7 e artefato JSON v2 |
| Relatório complementar | 10, profiling | SCAN-04, TEL-03 e seção 10 |
| Relatório complementar | 11–12, fechamento/conclusão | seções 12–14 |

## 4. Mapeamento dos 15 grupos obrigatórios

| Grupo do prompt | Linhas da matriz | Resultado |
|---|---|---|
| 1. Identidade imutável | ID-01, ID-02 | fechado |
| 2. Embedding determinístico | EMB-01, EMB-02 | lane conservadora fechada; escrita direta deliberadamente não promovida |
| 3. Selective scan | SCAN-01…05 | geometria/layout/chunk/rollback fechados; profiler e scan paralelo geral permanecem classificados |
| 4. AdamW multi-tensor | OPT-01…03 | FP32 determinístico fechado; optimizer comprimido não promovido |
| 5. GEMMs/transposes | GEMM-01…04 | classic BLAS e abstração fechados; Lt depende de SDK |
| 6. Mixed precision | ID-01/02, GEMM-02/03/04 | contrato, cache, invalidação, FP16/BF16 HIP e fail-closed fechados |
| 7. Checkpoint seletivo | BUF-01 | fechado |
| 8. Vida útil/buffers | BUF-02…04, CKPT-02 | fechado nas lanes seguras |
| 9. Slices/projeções | PROJ-01/02 | fechado |
| 10. RMSNorm/gates/elementwise | ELEM-01 | fusão segura fechada; redução arriscada não promovida |
| 11. CE/sincronizações | CE-01/02, OPT-02 | fechado sem remover gates de segurança |
| 12. Checkpoint/I/O | CKPT-01…03, IO-01 | fechado |
| 13. Telemetria | TEL-01…03 | fechado no observável; profiler classificado |
| 14. Portabilidade | PORT-01/02 | CPU/HIP real, modo sem MPI fail-closed e CUDA/MPI estáticos; hardware/toolchain ausentes documentados |
| 15. Segurança/falha fechada | SAFE-01…06, DATA-02, INF-01, REASON-01 | fechado |

## 5. Contratos numéricos

### 5.1 Igualdade bit a bit exigida e comprovada

- CSR conservador do embedding no mesmo backend;
- geometria, layout e backward chunked determinístico do Mamba promovidos;
- AdamW multi-tensor FP32;
- checkpoint seletivo e liberações de buffers;
- identidade, checkpoint/resume e ordem de tied weights;
- três treinos FP32 completos: losses, modelo e trainer state idênticos.

### 5.2 Tolerâncias pré-registradas

- FP16: `atol/rtol` de teste até `3e-2` para GEMMs pequenas;
- BF16: `atol/rtol` de teste até `6e-2`;
- CPU versus GPU usa tolerâncias específicas de cada parity test;
- nenhuma tolerância foi escolhida depois de observar o resultado;
- o gate final FP32 não usa tolerância: exige hashes iguais.

### 5.3 Mudanças não promovidas

- scan paralelo geral;
- Adam 4/8-bit como estado autoritativo;
- escrita direta do embedding no gradiente persistente;
- fusões que reordenem reduções RMSNorm/dgamma;
- Lt sem provider e algoritmo medido;
- qualquer alegação de MFMA/Tensor Core, ocupação, VGPR ou spill sem profiler.

## 6. Build e testes

Configuração limpa:

- generator Ninja;
- Release;
- Clang 23.0.0;
- HIP/TheRock, arquitetura `gfx1102`;
- OpenMP 5.1;
- Python extension, OxtaMem e testes habilitados;
- `CMAKE_COMPILE_WARNING_AS_ERROR=ON`.

Resultado:

- configure: aprovado;
- build: 216/216 alvos;
- CTest final: 95/95 em 78,02 s;
- gradcheck: aprovado;
- CPU e HIP parity: aprovados;
- mixed precision FP16/BF16 HIP: aprovado;
- AdamW determinístico 1.000 updates: aprovado;
- checkpoint/resume Mamba, attention e híbrido: bitwise;
- tied embedding/head: coberto;
- Python pipeline/proveniência: aprovado.

Falhas injetadas aprovadas:

- NaN/Inf antes do commit e recuperação;
- falha ambígua durante commit e estado poisoned;
- OOM durante load sem mutação parcial;
- model/trainer truncado, payload alterado e NaN semântico;
- interrupção durante snapshot e escrita;
- falha do worker assíncrono e segundo snapshot bloqueado;
- rename fallback e preservação do sidecar durável;
- path traversal de pack/shard;
- rollback de embedding, scan, finite gate, cache lowp e `matmul_tn`.

## 7. Gate de treino e comparação antes/depois

Comando-base, sempre com diretório novo:

```powershell
python OXN\nsos\scripts\train_ptbr_conversational.py run `
  --preset pilot --device gpu --max-train-steps 1000 `
  --no-resume --run-dir <novo>
```

O histórico auditado variava aproximadamente entre 535 e 770 tok/s. O p50
final médio de 1.575,432 tok/s representa:

- 2,05× versus 770 tok/s (`+104,6%`);
- 2,94× versus 535 tok/s (`+194,5%`);
- 2,25× versus a média histórica aproximada de 701 tok/s (`+124,7%`).

Essas comparações históricas misturam fases e condições diferentes e são
rotuladas como contexto, não como A/B causal puro. Nas lanes isoladas da
otimização final, a promoção LDS state-major apresentou `+10,64%` sobre o
state-major sem LDS e `+21,87%` sobre o finite-flat anterior, mantendo os
mesmos bits.

Qualidade no gate:

- loss held-out: 9,947924 → 4,084897;
- melhora absoluta: 5,863027;
- melhora relativa: 58,9372%;
- nenhuma corrida perdeu qualidade ou divergiu.

Recursos no gate:

- GPU: RX 7600, HIP, `gfx1102`, warp/wave observado 32;
- memória: device-only;
- pico allocated: 5.642.256.384 bytes;
- pico reserved: 5.712.904.192 bytes;
- managed live/cache/advice/probes: zero;
- device synchronizations: zero;
- Mamba host/fast-path fallbacks: zero;
- pool release/unknown/capture failures: zero.

Artefatos autoritativos:

- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-a-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-b-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-c-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-post-hardening-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-rc1-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-final-a-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-final-b-20260801`;
- `artifacts/ptbr_conversational/pilot/runs/final-gate-1000-final-c-20260801`.

### 7.1 Campanha longa no candidato definitivo

Artefato:

- `artifacts/ptbr_conversational/pilot/runs/final-soak-10000-final-20260801`.

O resultado autoritativo atualizado estÃ¡ em
`artifacts/ptbr_conversational/pilot/runs/final-soak-10000-final2-20260802`;
os nÃºmeros abaixo descrevem a linhagem histÃ³rica prÃ©-transversal.

Resultado do protocolo longo:

- passos: 10.000/10.000, sem resume;
- tempo medido de treino: 3.245,368 s (54 min 05,4 s);
- throughput global: 1.577,633 tok/s e 3,081315 passos/s;
- percentis dos primeiros 1.000 passos, com 100 warmups descartados:
  p10 1.570,167, p50 1.575,982 e p90 1.581,116 tok/s;
- gate pré-registrado: aprovado (`p50 >= 1.500 tok/s`);
- loss de treino: primeira 9,894630, última 3,917695 e média 3,616324;
- loss held-out: 9,947924 → 3,225419, melhora relativa de 67,5770%;
- transições de shard: passos 3.907 e 7.813, ambas sem descontinuidade;
- snapshots: 10 concluídos e nenhum pendente;
- checkpoints retidos: quatro válidos, zero inválido;
- verificação formal do manifesto e de todos os payloads: aprovada;
- `stderr`: zero bytes.

Evidência bitwise cruzada com o soak independente anterior:

- trace dos primeiros 1.000 passos:
  `ab386f4e9ec222292e0d40c4b78af2dad9576bbced3b9e7030ddeae5562c4265`;
- modelo final:
  `c379954d9a70ce218f3ff15278d7f1e5e95a0c239f82fce55bd5a3a00f6c18f0`;
- estado final do trainer:
  `5bb1e9afa23c7df37fa3b542459a8118b8605192d788cb5f9aabe95248fad912`;
- identidade do checkpoint:
  `2fd19641beecc35fd9fd25a9ee3dce7dc986cdc3991e9b1cc43dad603b732bf7`;
- identidade semântica dos 66 campos:
  `4ad0ed394b5db0eaa8ba9e8dd83b16b3f0a685177302b363347062ab8c25dcf1`.

Recursos finais: pico allocated de 5.642.256.384 bytes, pico reserved de
5.712.904.192 bytes, zero memória gerenciada, zero sincronização global de
device, zero fallback host/Mamba, zero falha/advice/violação/quarentena do pool.

## 8. Arquivos e funções alterados pelo pacote industrial

### 8.1 Build, identidade e backend

- `CMakeLists.txt`: seleção HIP/CUDA, staging, inventário e testes;
- `include/gpu_backend.h`, `src/gpu_backend.cpp`: abstração e seleção de device;
- `include/gpu_gemm_provider.h`, `src/gpu_gemm_provider.cpp`: provider comum;
- `include/runtime_execution_identity.h`, `src/runtime_execution_identity.cpp`:
  captura, canonização, digest e comparação;
- `include/optimizer_runtime_policy.h`: políticas e rollbacks do optimizer;
- `include/nsos/sha256.h`, `src/sha256.cpp`: hashing comum;
- `include/nsos_config.h`: configuração e políticas persistíveis.

### 8.2 Tensor/GEMM/memória

- `include/tensor.h`, `src/tensor.cpp`: `storage_view`, pool/eventos,
  `matmul_tn`, GemmEx NT/TN, cache lowp, CE device-resident, telemetria e
  validações fail-closed;
- `include/cuda/device_buffer.h`, `include/cuda/pinned_buffer.h`,
  `include/cuda/cuda_memory.cuh`, `include/cuda/gpu_utils.h`: RAII e erros GPU;
- `include/cuda/kernels.cuh`, `src/cuda/kernels.cu`: embedding CSR, casts,
  transpose lowp, CE, SiLU gate e kernels elementwise;
- `include/rierass_core.h`, `include/cuda/bitnet_math.cuh`: scratchpad
  concatenado validado e produto ternário funcional sem retorno fictício;
- `include/autograd.h`: versão/gradiente e tied-weight invariants.

### 8.3 Mamba e optimizer

- `include/embedding.h`, `src/embedding.cpp`: workspace e CSR conservador;
- `include/mamba2.h`, `src/mamba2.cpp`: grouped owner/views, state-major,
  workspaces, checkpoint seletivo e lifetimes;
- `include/cuda/mamba_kernels.cuh`, `src/cuda/mamba_kernels.cu`: geometria,
  precompute, chunked/LDS, K4, reduções e rollbacks;
- `src/bitlinear.cpp`: backwards via `matmul_tn`;
- `include/jamba.h`, `src/jamba.cpp`: tied weights, caches, telemetria e cleanup;
- `include/trainer.h`, `src/trainer.cpp`: policy lease, CE diferida, AdamW,
  finite/poison, timing e identidade;
- `include/dataloader.h`, `src/dataloader.cpp`, `src/dataloader_v2.cpp`:
  dataset real, validação integral e término explícito sem dados sintéticos;
- `include/smart_loader.h`, `src/smart_loader.cpp`: fila assíncrona com
  ownership, leitura posicionada exata, drain e accounting;
- `include/fabric.h`, `src/fabric_v2.cpp`, `include/nsos_mpi.h`,
  `src/nsos_mpi.cpp`: contrato distribuído e MPI fail-closed;
- `include/mcts_reasoning.h`, `src/mcts_reasoning.cpp`: device correto e
  avaliação escalar/batch finita e estrita;
- `include/monitor.h`, `include/lean_integration.h`: monitor concorrente e
  prova externa sem sucesso fictício;
- `include/nsos_sdk.h`, `src/nsos_sdk.cpp`: publicação/move/streaming
  transacionais e estado poisoned;
- `src/cuda/fused_optimizer_kernels.cu`: descritores, finite/norm/update chunks.

### 8.4 Checkpoint, binding e pipeline

- `include/checkpoint_io.h`, `include/nsos_serializer.h`: snapshots e formatos;
- `src/bindings.cpp`: identidade, recursos, pool/cache e controles Python;
- `scripts/train_ptbr_conversational.py`: provenance, mmap, rolling telemetry,
  async checkpoint, verify, percentis e gate;
- scripts legados de benchmark/chat/DPO/SFT/probe/standalone: ausência de API,
  import, cópia de input, logging, telemetria e avaliação são explícitas;
- `scripts/oxta_contabil/benchmark_policy.py`: contrato runtime device-only.

### 8.5 Testes diretamente ampliados/adicionados

- `tests/test_runtime_execution_identity.cpp`;
- `tests/test_sanity.cpp` (scratchpad/packing ternário/fail-closed);
- `tests/test_training_invariants.cpp`;
- `tests/test_ptbr_conversational_script.py`;
- `tests/test_product_benchmark_policy.py`;
- `tests/test_dataloader.cpp`, `tests/test_smart_loader_async.cpp`;
- `tests/test_mcts_reasoning_v3.cpp`, `tests/train_e2e.cpp`;
- `tests/test_matmul.cpp`, `tests/test_gradcheck.cpp`, `tests/test_jamba.cpp`;
- `tests/gpu/test_gpu_parity_basic.cpp`;
- `tests/gpu/test_gpu_parity_mamba_faithful.cpp`;
- `tests/gpu/test_gpu_checkpoint_continuation.cpp`;
- `tests/gpu/test_gpu_deterministic_adamw_1000.cpp`;
- `tests/gpu/test_gpu_mixed_precision_contract.cpp`;
- `tests/gpu/test_gpu_lowp_weight_cache.cpp`;
- `tests/gpu/test_gpu_device_memory_contract.cpp`.

Principais funções/classes novas ou materialmente alteradas:

- `capture_runtime_execution_identity`, `require_runtime_execution_identity`;
- `RuntimeExecutionPolicyLease`, `RuntimeExecutionPolicyMutationGuard`;
- `Embedding::backward_batch` e launchers CSR;
- `Mamba2SSD::{forward,backward}_faithful` e grouped projections;
- launchers faithful forward/backward/chunk/LDS/reductions/K4;
- `matmul_tn`, `matmul_nt_mixed_gpu`, `LowpWeightCache`;
- `Tensor::{storage_view,cross_entropy_device,silu_gate}`;
- `Trainer::{train_step,apply_optimizer_step,save_training_state,load_training_state}`;
- `CheckpointManager::{save,_publish_snapshot,close,verify}`;
- `RollingTrainingTelemetry`, `validate_prepared_workspace`.

## 9. Problemas adicionais encontrados e corrigidos na validação

1. O corpus usava `SCRIPT_VERSION` como versão do manifesto/recipe. A identidade
   de corpus foi separada em schema/recipe próprios; migração é estrita e não
   reescreve tokenizer, shards ou manifests imutáveis.
2. `run` recusava o workspace piloto válido e `prepare` podia quebrar a cadeia
   SHA ao reescrever só um manifesto. A validação agora cobre schemas exatos,
   SQLite counts, path containment, tokenizer e todos os shards.
3. `hipBLAS GemmEx` lowp com OP_T produziu `hipErrorInvalidImage` na RX 7600.
   O HIP usa cast+transpose fused seguido de GEMM NN; CUDA preserva OP_T.
4. `gpu_gemm_provider.cpp` comparava ponteiros de string literal, não conteúdo.
   Foi corrigido com `std::string_view`.
5. Retornos de `hipMemset`, graph cleanup e legacy Mamba clears eram ignorados.
   Todos foram checados ou contabilizados; failures deixam mensagem acionável.
6. `hipMemAdvise` opcional podia falhar sem telemetria. O pool agora conta e
   expõe `managed_advice_failures`; a política de benchmark rejeita valor não
   zero.
7. Testes HTTP usavam `inet_addr` obsoleto. Foram migrados para `InetPtonA` /
   `inet_pton`, permitindo build warning-as-error limpo.
8. O `DataLoader` legado mascarava falhas de dataset com números aleatórios e o
   `SmartLoader` podia encerrar com trabalho/promise sem ownership inequívoco.
   Ambos foram convertidos para I/O real, bounded e fail-closed.
9. Fabric sem MPI e partes do wrapper MPI simulavam operações distribuídas.
   A simulação foi removida; configuração sem provider aceita estritamente o
   processo local e rejeita qualquer coletiva solicitada.
10. O monitor tratava falha de cópia GPU como ausência de anomalia e Lean
    podia retornar sucesso quando indisponível. Os contratos agora distinguem
    indisponível, erro e verificação real.
11. O MCTS podia cair silenciosamente para avaliação escalar quando o batch
    retornava cardinalidade incorreta; chamadas escalares também não validavam
    NaN/Inf. A validação foi centralizada e o `rollout` morto removido.
12. Limpezas de trainer, streaming e publicação podiam deixar um objeto
    aparentemente reutilizável após erro parcial. Os caminhos agora registram
    ou propagam falha e envenenam o estado ambíguo.
13. Um header legado sem chamadores declarava mocks de runtime CUDA que
    retornavam sucesso em build CPU e ainda oferecia Unified Memory. Os mocks
    foram removidos e a lane GPU passou a ser estritamente device-only.
14. Destrutores de shards, recuperação de `LATEST.json`, override curricular,
    avaliação de probe e cleanup/upload de CUDA Graph ainda continham retornos
    ignorados. Todos passaram a falhar ou registrar erro explicitamente sem
    lançar a partir de destrutores.
15. A varredura transversal final encontrou um scratchpad que devolvia apenas o
    último tensor, um dot-product ternário que retornava zero, isolamento GPU de
    teste no-op e utilitários DPO/SFT/benchmark capazes de omitir falhas. Os
    caminhos foram implementados ou tornados fail-closed; 83 arquivos Python
    passaram AST, probes específicos passaram e o rebuild integral permaneceu
    warning-as-error.

## 10. Riscos residuais e dependências externas

- Não existe GPU NVIDIA/CUDA neste computador. O código e CMake foram revistos
  e mantêm implementação CUDA, mas execução real CUDA continua pendente.
- Não existe GPU AMD wave64 local; a RX 7600 exercita wave32. O fallback genérico
  e runtime warp-size estão testados estaticamente, não em wave64 físico.
- O TheRock instalado não contém `rocprofv3`, `rocprof`, ROCprof Compute/System
  ou roctracer. Ocupação, VGPR, spills, algoritmo interno BLAS e MFMA continuam
  honestamente classificados como dependentes de profiler.
- hipBLASLt/cuBLASLt não foi promovido sem provider/SDK e matriz de algoritmos.
- O ambiente não possui toolchain/runtime MPI; o contrato sem MPI foi executado
  e o caminho MPI foi revisado/compilável condicionalmente, mas uma campanha
  multi-rank real continua dependente de ambiente externo.
- `scripts/probe_cascade.py` é ferramenta de pesquisa, não caminho de produção;
  a combinação Slender+GPU está fora de seu contrato e é recusada antes de
  carregar a extensão. Isso é uma limitação explícita do probe, não fallback ou
  implementação parcial do runtime de treino PT-BR.
- FP16/BF16 passaram o contrato funcional HIP, mas o produto PT-BR final foi
  validado em FP32; promoção mixed para campanha longa exige curva de qualidade
  própria e repetição no mesmo modo.
- O cache do pool apresentou fragmentação observável alta (cerca de 0,95) ao
  final, mas manteve margem de VRAM, estabilidade, zero falhas e p50 estável.
  É métrica de bins/cache, não prova de fragmentação física do driver.
- Throughput de 120M/150M/180M/213M não foi medido; não deve ser inferido do 71M.

## 11. Prontidão e escalabilidade

### 11.1 Treinamento prolongado do modelo atual (~71M)

Recomendação: **apto tecnicamente para campanha prolongada controlada na RX
7600**, mantendo a identidade atual, device-only memory, checkpoints
verificados e monitoramento de loss/VRAM/temperatura. Gate curto, soak exato de
10.000 passos, determinismo bitwise longo, qualidade, fail-closed e integridade
estão aprovados no hardware disponível.

### 11.2 120M

Recomendação: **viável com checkpoint seletivo obrigatório**. Medir o shape real,
manter pelo menos 0,8–1,0 GiB de margem e repetir os gates de 1.000/10.000 passos.

### 11.3 150M

Recomendação: **viável, porém condicionado** a checkpoint seletivo, pool sob
controle e nova medição de pico. Não assumir o mesmo throughput do 71M.

### 11.4 180M

Recomendação: **experimental na RX 7600 de 8 GiB**. Exige checkpoint seletivo,
layout otimizado, workspaces controlados e provável mixed precision validada.

### 11.5 ~213M

Recomendação: **não declarar robusto em FP32 neste hardware**. A persistência
FP32 aproximada é 3,17 GiB antes de ativações. Só avançar com checkpoint
seletivo, redução comprovada de temporários, estado de optimizer comprimido
determinístico validado e margem real abaixo de ~7 GiB. Unified Memory não é
solução de produção.

## 12. Gate de parada

Encerrados:

- todos os achados `[C]` implementáveis;
- itens `[IF]` confirmados, implementados ou reclassificados com justificativa;
- hipóteses não promovidas sem medição;
- build limpo, suíte, determinismo, checkpoint/resume, falhas e benchmark curto;
- matriz, arquivos, funções, invariantes, riscos e recomendações.

O gate operacional está encerrado: o soak novo de 10.000 passos concluiu com
checkpoints íntegros, duas transições de shard, estabilidade numérica, p50 acima
de 1.500 tok/s, qualidade held-out melhor e hashes longos reproduzidos.

Profiling AMD e CUDA real não bloqueiam a correção funcional nem o gate de
1.500 tok/s porque as ferramentas/hardware não existem no ambiente; permanecem
entregáveis externos explicitamente rastreados, sem simulação.

## 13. Auditoria dos 23 itens da fase dinâmica

| Item | Exigência | Evidência autoritativa | Estado |
|---:|---|---|---|
| 1 | Build limpo | build virgem Release HIP, warnings-as-errors, 216/216 | aprovado |
| 2 | Testes unitários | CTest integral | aprovado |
| 3 | Testes de integração | pipeline Python, SDK/model pack e `train_e2e` | aprovado |
| 4 | CPU | referências, gradcheck e paridades CPU | aprovado |
| 5 | AMD/HIP | RX 7600 `gfx1102`, suíte GPU e treino real | aprovado |
| 6 | NVIDIA/CUDA | backend/CMake e contratos revisados | dependente de hardware ausente; não simulado |
| 7 | Gradcheck | `test_gradcheck` e gradchecks Mamba/Jamba | aprovado |
| 8 | Determinismo | AdamW 1.000, traces e linhagens independentes | aprovado |
| 9 | Bitwise | trace/model/trainer idênticos | aprovado |
| 10 | Checkpoint/resume | continuation CPU/HIP, Mamba/attention/híbrido | aprovado |
| 11 | Pesos amarrados | registry canônico e embedding/head cobertos | aprovado |
| 12 | NaN/Inf | optimizer, MCTS, monitor, pack e checkpoint | aprovado |
| 13 | OOM | load/snapshot sem mutação parcial | aprovado |
| 14 | Checkpoint truncado | model/trainer/payload/manifest | aprovado |
| 15 | Interrupção no snapshot | injeção antes da publicação | aprovado |
| 16 | Interrupção na escrita | staging/worker/rename/fync | aprovado |
| 17 | Comparação 1/10/100/1.000 | AdamW/reference e traces de treino | aprovado |
| 18 | Pesos, grads, `m`, `v`, versões, loss, LR e hashes | testes de invariantes/continuation e manifests | aprovado |
| 19 | Benchmark frio/aquecido | passo frio separado; 100 warmups + 900 amostras | aprovado |
| 20 | Mínimo de três repetições | corridas finais A/B/C independentes no mesmo módulo | aprovado |
| 21 | Profiling após correção funcional | ferramenta ROCprof não existe na instalação | dependente de profiler; não inventado |
| 22 | Campanha longa | soak exato de 10.000 passos, duas transições de shard e dez snapshots | aprovado |
| 23 | Revisão dos resultados | matriz, JSON, verify formal e hashes cruzados com soak independente | aprovado |

## 14. Auditoria dos 21 entregáveis finais

| Entregável | Local da evidência | Estado |
|---:|---|---|
| 1. Resumo executivo | seção 1 | completo |
| 2. Matriz de rastreabilidade | seção 3 | completo |
| 3. Arquivos alterados | seção 8 e inventário completo separado | completo |
| 4. Funções alteradas | seção 8 | completo |
| 5. Descrição técnica | seções 3 e 8 | completo |
| 6. Invariantes preservados | seções 3 e 5 | completo |
| 7. Igualdade bitwise | seções 1 e 5.1 | completo |
| 8. Tolerâncias | seção 5.2 | completo |
| 9. Riscos residuais | seção 10 | completo |
| 10. Build | seção 6 | completo |
| 11. Testes | seções 6 e 13 | completo |
| 12. Determinismo | seções 1, 5 e 7 | completo |
| 13. Checkpoint/resume | seções 3, 6, 7 e 13 | completo |
| 14. Memória | seções 1 e 7 | completo |
| 15. Desempenho | seções 1 e 7 | completo |
| 16. Antes/depois | seção 7 | completo |
| 17. Problemas da validação | seção 9 | completo |
| 18. Correções adicionais | seção 9 | completo |
| 19. Hardware/profiler | seção 10 | completo e limitado explicitamente |
| 20. Prontidão longa | seção 11.1 | completo |
| 21. Escala 120M–213M | seções 11.2–11.5 | completo |
