# Auditoria Estática Perfeccionista — NSOS (produto `OXN/nsos/`)
**Data:** 2026-06-10 · **Método:** leitura profunda (sessão de polimento) + varredura
de padrões em 100% de `src/` + `include/` + `scripts/` · **Cobertura declarada na §9.**

Cada item: evidência (`arquivo:linha`), por que importa, fix sugerido, esforço (S/M/L).
Prioridade: P0 = próximo lever de perf/correção · P1 = estrutural · P2 = robustez/segurança · P3 = futuro.

---

## §1 — P0 · Performance (os próximos levers, em ordem de ROI esperado)

| # | Item | Evidência | Problema → Fix | Esf. |
|---|---|---|---|---|
| 1 | **MoE: 5 drenos device-wide por step** | `jamba.cpp:3063,3326,3458,3629,3664` (`cudaDeviceSynchronize` nos caminhos batched fwd/bwd) | Cada sync drena o pipeline inteiro, por camada-MoE × buckets × step. Trocar por dependência de stream (já é tudo stream-0) ou evento; manter sync só onde há leitura host imediata → provável responsável pelo custo restante uniforme/camada visto no `[ltime]` | M |
| 2 | **MoE: top-k no host por token** | `jamba.cpp:1889,3761,4791` (`std::partial_sort`); helper de cópia próprio síncrono `jamba.cpp:220-230,369-375` | Router decide no host (logits D2H → sort → H2D). Kernel top-k (k≤2, num_experts 8 → warp-level trivial) + permutação device-side | M |
| 3 | **Zero-fill incondicional em TODA alocação** | `tensor.cpp:620-629` (memset no ctor via pool) | Outputs de matmul/gather/copy são 100% sobrescritos — o memset é um kernel extra por op (~centenas/step). Flag interna `Tensor::uninitialized(dims)` usada pelos produtores que provadamente sobrescrevem (matmul, clone, kv_split, gathers) | S |
| 4 | **Otimizador: ~614×3 launches/step** | `trainer.cpp` apply_optimizer_step (scale + clip-norm + adamw por parâmetro); medido `opt=226-490ms` | Multi-tensor apply: 1 kernel com tabela de ponteiros/tamanhos (padrão Apex). Corta ~1800 launches para ~3 | M |
| 5 | **Confirmar dispatch GPU do Mamba no v11** | fallback host com 5-6 `.cpu()` por chamada em `mamba2.cpp:240-244,371-375`; condições em `can_use_gpu_mamba_scan` (`mamba2.cpp:15-31`) | Se qualquer condição falhar no shape do v11, a bomba nº2 da atenção se repete no Mamba silenciosamente. Adicionar telemetria de fallback (contador já existe: `gpu_fast_path_fallbacks_` — EXPOR no boot do treino e no lab) | S |
| 6 | **MoE aux-reg lê grad no host por step** | `trainer.cpp:768` (`gate_weight.grad.cpu()`) | D2H por step; mover cálculo pro device ou cadência (a cada N steps) | S |
| 7 | **BitLinear treino: 3 saves clonados por linear** | `bitlinear.cpp` (saved_input/linear_input/pre_output `.clone()`); ~9-15 clones/camada | Async-D2D já removeu os drenos (`508733b`); próximo nível: guardar por referência + `version` do produtor (pool não realoca in-place) e clonar só sob escrita | M |
| 8 | **Cópia de bytes triplicada** | `tensor.cpp:335`, `trainer.cpp:77,128`, `jamba.cpp:220-230` | Três implementações com semânticas de sync DIFERENTES (a de jamba é cudaMemcpy síncrono!). Unificar num header interno com a política async-D2D/sync-H2D-D2H | S |
| 9 | **CUDA Graphs no step** | pool dá endereços estáveis; shapes de bucket repetem | Capturar fwd+bwd por shape de bucket após itens 1-4 (graphs não toleram syncs no meio — por isso vem depois) | L |
| 10 | **CPU decode 0.7 tok/s** | medido no `nsos_cpu_eval` (T4-treinado, CPU 2-core) | Verificar se a inferência CPU usa o caminho ternário packed/AVX2 ou caiu no float reference (`bitnet_adapter` dispatch); é a tese edge — merece perfil dedicado no runtime CPU (sem quota) | M |

## §2 — P0 · Correção / Ciência

| # | Item | Evidência | Fix | Esf. |
|---|---|---|---|---|
| 11 | **Treino não-determinístico (~32%/step)** | medido (paridade v5); 4 sites: scatter-add embedding, grad_A mamba, scatter MoE, redução do loss CE | Modo `NSOS_DETERMINISTIC` (reduções segmentadas ordenadas); enquanto não existir, **alinhar claims** de replay para inferência/seed apenas (doc §4b da verificação) | L |
| 12 | **Tokenizer parou em 4897 de 8192 merges** | boot log (`Loaded vocab size: 4897`, alvo `tokenizer_8192`) | 40% do espaço de merges não usado → compressão pior (mais tokens/char → step mais caro). Investigar: corpus de treino do BPE pequeno? min-freq alto? Medir tokens/char no gate G5 antes/depois de re-treinar o BPE com corpus maior | M |
| 13 | **Curriculum: fonte do próprio repo na phase3** | `nsos_curriculum_lib.py:714-718` (5 .cpp como documentos) | Confirmado como causa do "modelo responde C++" (memorizou; título quase verbatim). Quantificar share via gate G6 e decidir orçamento; checar se eval da phase3 contém chunks dos MESMOS arquivos (contaminação train/eval por overlap de chunking com sobreposição de 56-80 chars) | S |
| 14 | **Checkpoint identity frágil a reordenação** | `nsos_serializer.h:32-34` (`base_name#ocorrência` na ORDEM de parameters()) | Inserir/remover um módulo desloca todas as ocorrências seguintes do mesmo base_name → load silenciosamente errado com strict=False. Gravar TAMBÉM o nome absoluto (agora estável pós-fix de época) como verificação secundária | M |
| 15 | **`errors="ignore"` em leituras de dataset** | 8 ocorrências em `scripts/` (`nsos_curriculum_lib.py:1090,...`) | Perda silenciosa de bytes no corpus; trocar por contagem + abort acima de limiar (0.1%) | S |

## §3 — P1 · Estrutural (dívida que freia tudo)

| # | Item | Evidência | Fix | Esf. |
|---|---|---|---|---|
| 16 | god-file `jamba.cpp` **5007 linhas** | `wc -l` | Plano já existe (ARCHITECTURE_RISK Stage 2/3): extrair `attention.cpp`, `moe.cpp`, schedules — gate verde por extração | L |
| 17 | `train_curriculum.py` **4953 linhas** | idem | Extrair: eval/generation, profiles, resume/checkpoint, RunLogger p/ módulos | M |
| 18 | `http_api_server.cpp` 2048 / `nsos_sdk.cpp` 2022 | idem | Stage 2 do plano de risco (routing/parsing/auth; pack-IO/inferência) | L |
| 19 | Saves da atenção recomputam RoPE | doc verificação §5 | Kernel GQA fundido exportar q_rot/k_rot direto (elimina clone+rotação por camada) | M |
| 20 | Ops sem GPU fora do hot path | `transpose` rank>2, `softmax` dim≠last, `sum` rank>2 (tensor.cpp) | Hot path auditado limpo; adicionar `assert`/log de uso em GPU para nunca regredirem silenciosamente | S |
| 21 | Stubs marcados | `persistent_kernel.cu`, `lean_integration.h`, `mpi_mock.h`, `chat.cpp`, `server.cpp` | Já documentados como não-produto; adicionar guarda de build que falha se referenciados por alvo de release | S |

## §4 — P2 · Robustez / Segurança

| # | Item | Evidência | Fix | Esf. |
|---|---|---|---|---|
| 22 | **9× `catch (...)`** | `api_server.cpp:38`; `http_api_server.cpp:709,1066,1382,1597`; `jamba.cpp:1143,1213`; `nsos_sdk.cpp:50,418` | Auditar um a um: engolir exceção em produto = falha silenciosa (a lição da sessão: o anti-padrão que escondeu o bug do gate v2). Mínimo: log estruturado + rethrow onde não-recuperável | M |
| 23 | **Token GitHub no remote URL dos notebooks** | células 2 (train/lab/cpu_eval): `oauth2:TOKEN@github...` persiste em `.git/config` do runtime | VM efêmera mitiga, mas o token vaza em qualquer dump de config/erro verboso. Trocar por `GIT_ASKPASS`/header `http.extraheader` por-comando | S |
| 24 | **Injeção de special token** | gate G2 (novo) reporta se `<|endoftext|>` literal em texto de usuário vira id especial | Se injetável: sanitizar na fronteira do serving (HTTP/SDK) | S |
| 25 | Buffers estáticos leak-by-design | `attn_valid_device_buffer`, scratches CE, workspace BF16 | OK para processo de treino; documentar e zerar em `nsos_shutdown()` para uso como biblioteca | S |
| 26 | Telemetria de memória do pool inexistente | descoberta do thrash exigiu NVML externo | Expor `pool_stats()` (live/cached/bins) via bindings; imprimir no boot do treino a cada N steps | S |
| 27 | UM oversubscription sem guarda | T4 chegou a 96% antes do binning | Com stats do item 26: WARN acima de 80% do total + trim agressivo automático | S |

## §5 — P3 · Futuro próximo (depois dos P0)

28. **BF16 nos kernels glue** da atenção (hoje só GEMMs) — ganho menor, T4 Tensor Cores.
29. **Embedding scatter-add determinístico** (sort-by-token + segmented reduce) — par do item 11.
30. **4-bit optimizer** (plano WS-1 do roadmap — rank-1 para `v`, paridade obrigatória).
31. **KAN/SSA em CUDA** — só quando voltarem a ligar (off no v11; `sparse_attention.cpp` aloca `Device::CPU` hardcoded em 10 pontos: 22,101,127,140,256,276,308,360,411,438).
32. **Re-treinar BPE** com corpus ampliado (item 12) + pesos por fase já documentados em `nsos_curriculum_lib.py:1274`.
33. **Lab: braço HOST opcional** (hoje sempre roda 1 step lento de referência — flag para pular).

### §5b — Adendo da varredura final (cheiros residuais)

34. `http_api_server.cpp:1403` — poll de 10ms em loop (provável shutdown/wait): trocar por
    condition_variable com timeout (S; cosmético até haver carga de serving).
35. `nsos_sdk.cpp` tem **1** único site de mutex vs 21 no http_api e 11 no memory_system —
    revisar thread-safety do SDK sob serving concorrente quando ele virar superfície quente (M).
36. `chat.cpp:54,112` usa `std::rand` — irrelevante (stub documentado, item 21), mas reforça
    a guarda de build proposta para não-produto.

**Estado notável:** ZERO `TODO/FIXME/HACK` em todo `src/`+`include/` do produto — a dívida
está em estrutura e padrões, não em pendências esquecidas; `smart_loader` não tem mais o
sleep de polling apontado no plano antigo (já corrigido).

## §6 — Varrido e LIMPO (não mexer sem motivo novo)

- Ops elementwise/matmul batched+BF16/softmax-lastdim/rmsnorm±bwd/squared_relu±bwd/CE/clip/adamw: **cobertura GPU confirmada** (`use_gpu_fast_path` em 21 sites de `tensor.cpp`).
- Caminho de treino da atenção (fwd saves + bwd): device-resident, paridade D1-PASS, doc própria.
- Pool: size-classes 64KB/2MB (`11732fe`) — thrash da T4 morto (mem 15.4→11.8GB, walls estáveis).
- `zero_grad` async, grads bulk-copy, RUL em kernel, CE com scratch persistente.
- Nomes de parâmetros idempotentes por época (`508733b`); identidade de checkpoint imune (serializer usa `base_name#occ`).
- Tokenizer: sanitização UTF-8 na fronteira existe (`tokenizer.cpp:62-97`); gate padrão-ouro criado (`tokenizer_gold_gate.py`).
- API: exige token fora de loopback (`http_api_server.cpp:854-856`).
- Bindings: GIL release nos hot paths (6 sites confirmados).

## §7 — Instrumentos criados nesta era de polimento (mantê-los é parte do padrão)

`NSOS_TRAIN_TIMING` (balanço fecha, unacc≈0) · `NSOS_LAYER_TIMING` (`[ltime]` por camada) ·
NVML por step no lab (clock/temp/mem/throttle) · paridade v5 com piso de ruído ·
`tokenizer_gold_gate.py` (G1-G6) · células de prova-de-versão (sha) em todos os notebooks.

## §8 — Sequência recomendada (1 variável por rodada de T4)

1. Itens **1+2** (MoE syncs + top-k) — predição: fwd/camada-MoE cai e o uniforme ~110-330ms comprime.
2. Item **4** (opt fundido) — predição: opt 226→<20ms.
3. Item **3** (zero-fill seletivo) — predição: -1 kernel/op, fwd/bwd −10-20%.
4. Item **9** (CUDA Graphs) — só depois, com o step já sem syncs.
5. Em paralelo CPU (sem quota): itens 10, 12, 13, 22, 23.

## §10 — LEDGER FINAL (goal "36 no padrão ouro" — 2026-06-10, madrugada)

Estados: ✅ = corrigido em código (compile-verde; runtime valida amanhã por
protocolo do projeto) · 🔬 = fechado por AUDITORIA (o item investigado provou-se
correto/já-resolvido — evidência citada) · 🧮 = fechado por DECISÃO-COM-CÁLCULO
(números explícitos; gatilho nomeado para reabrir).

| # | Estado | Fechamento |
|---|---|---|
| 1 | 🔬 | 5 syncs são `NSOS_CUDA_SYNC=1` debug (default OFF); o dreno REAL encontrado e corrigido: matmul `.data()`×3 pré-branch (3c5d65e) |
| 2 | 🔬 | MoE batched já é device-side (top-k kernel + 1 D2H; AUDIT#4+5 2026-05-16); `partial_sort` restante = fallback/decode fora do hot |
| 3 | ✅ | `Tensor::uninitialized` + matmul/clone/to/slice + 14 temporários attn |
| 4 | ✅ | otimizador 2-kernels multi-tensor (~2450→3 launches); `NSOS_FUSED_OPT=0` A/B |
| 5 | ✅ | `runtime_telemetry()` binding + print no treino e no lab |
| 6 | ✅ | aux-reg do gate device-side (`launch_add_row_broadcast`) — D2H/H2D de [E,d]/step eliminado |
| 7 | 🧮 | saves por referência INSEGUROS: `zero_sequence_suffix_inplace` muta `hidden` in-place entre camadas ⇒ clones async (já ~0-custo de sync) são o correto; reabrir SE copy-on-write por version |
| 8 | ✅ | cópia ÚNICA em tensor.cpp (decl tensor.h); trainer (2 defs) e jamba (`copy_moe_bytes`, 4 sites) deletados |
| 9 | 🧮 | CUDA Graphs: captura exige zero syncs no step; inventário atual: 32 D2H/step no loss (CE per-sample) + 1 do clip ⇒ capturável só por segmento; custo-benefício pós-#3/#4: launches/step caem de ~5-8k para ~1-2k ⇒ ganho de graphs ≈ 10-20ms/step — reabrir SE [timing] de amanhã mostrar wall-fwd ≫ GPU-busy |
| 10 | ✅ | `NSOS_EDGE_DIAG=1`: BitLinear imprime 1× o caminho (PACKED vs REFERENCE) — responde o 0.7 tok/s na 1ª célula CPU de amanhã |
| 11 | 🧮 | não-determinismo: 4 sites mapeados; fix real = reduções ordenadas (custo: scatter-embedding O(V·d) determinístico ≈ +25ms/step T4 — aceitável SÓ sob flag); claims já alinhados (doc §4b); piso de ruído é instrumento permanente (parity v5) — reabrir como `NSOS_DETERMINISTIC` quando replay de treino virar requisito de produto |
| 12 | ✅ | instrumento no gate G5: chars/token + alerta de vocab 4897/8192; retrain do BPE = 1 célula CPU (corpus maior) — decisão informada pelos números de amanhã |
| 13 | ✅ | G6b: interseção train∩eval por md5, FAIL automático em phase3; +128 linhas de código morto REMOVIDAS de build_phase3 |
| 14 | ✅ | trailer v2 de nomes absolutos no .bin (back-compat: v1 sem trailer carrega idêntico; mismatch ⇒ AVISO com 1º divergente) |
| 15 | ✅ | `read_text_strict` (aborta >0.1% de bytes inválidos) em 7 sites + benchmark_external ignore→replace |
| 16 | 🧮 | jamba.cpp 5007L: lei do REPO (ARCHITECTURE_RISK) exige gate de TESTES verde por extração — split às cegas hoje violaria a regra do próprio produto; 1ª extração nomeada (Attention→attention.cpp, ~1.6kL) agendada para a 1ª sessão com ctest verde na T4 |
| 17 | 🧮 | train_curriculum 4953L: mesmo critério; extrações nomeadas (profiles/, eval/, resume/) — após o run de validação |
| 18 | 🧮 | http/sdk: idem Stage-2; nesta noite os pontos QUENTES internos foram fechados (#22 catches, #34 auditado, #35 contrato) |
| 19 | 🧮 | exportar q_rot/k_rot do kernel GQA pouparia 2 kernels elementwise/camada ≈ 2-4ms/step (pós-desync os saves já são ~1% do step) — ROI baixo; reabrir SE [ltime] de amanhã disser o contrário |
| 20 | ✅ | warn-once em transpose(rank>2)/softmax(dim≠last)/slice(dim≠0) com tensor GPU |
| 21 | 🔬 | stubs demarcados; nenhum referenciado por alvo de release (build dos alvos = prova) |
| 22 | 🔬✅ | 9/9 catch(...) auditados: 5 parse-fallback (→ tipados `const std::exception&`), 2 cleanup-rethrow (corretos como estão), 2 evaluator-fallback MCTS (intencionais) |
| 23 | ✅ | token nunca persiste: remote volta a URL limpa nos 3 notebooks |
| 24 | ✅ | G2 mede injeção; sanitização de serving condicionada ao G2 de amanhã (1 linha no http se "injetavel") |
| 25 | ✅ | `release_cached_memory()` binding (trim total do cache) |
| 26 | ✅ | `pool_stats()` binding {cached, live, bins} |
| 27 | ✅ | guarda UM: uso>88% ⇒ trim+WARN (cadência 1/512) |
| 28 | 🧮 | BF16 nos glue kernels: tráfego dos glue ≈ 300MB/step ⇒ economia ≈ 0.5-1ms na T4, MAS casts F32↔BF16 adicionam 2 kernels/uso ⇒ ganho líquido ~0 — FECHADO como não-fazer (números acima) |
| 29 | 🧮 | scatter determinístico = parte do #11 (mesmo cálculo de +25ms sob flag) |
| 30 | 🧮 | 4-bit GPU: estados m/v = 2×160MB = 320MB = 2% da T4 pós-binning ⇒ ZERO pressão; CPU 4-bit já existe p/ edge; gatilho de reabertura: modelo >1B params OU VRAM <6GB |
| 31 | ✅ | SSA warn-once quando entrada GPU (CPU-only por design até kernel da Fase 2) |
| 32 | ✅ | = #12 (instrumento pronto; retrain informado por dados) |
| 33 | ✅ | `RUN_HOST_ARM` no lab |
| 34 | 🔬 | sleep(10ms) é backoff de ERRO pós-`accept` (o accept É o bloqueio; cv inaplicável) — padrão correto |
| 35 | 🧮 | SDK: serving é single-replica-single-thread hoje (replica pool serializa via release; http tem 21 sites de lock); contrato documentado — reabrir com serving concorrente |
| 36 | 🔬 | `std::rand` vive só em chat.cpp (stub demarcado, fora de release) |

**Placar final: 20× ✅ código · 7× 🔬 auditoria-fechou · 9× 🧮 decisão-com-cálculo.**
36/36 endereçados.  Predições pré-registradas para a validação de amanhã:
`opt → <25ms` · `fwd → <1s` · `mamba fallbacks=0` · G6b/G2/G5 vereditos ·
`[edge]` imprime o caminho do decode CPU.  Itens 🧮 têm gatilho de reabertura
NOMEADO — nenhum é "depois a gente vê".

## §9 — Declaração de cobertura (honestidade do mapa)

- **Leitura integral nesta era:** tensor.cpp, trainer.cpp (caminho de treino), jamba.cpp (atenção/forward/parameters), mamba2.cpp (scan/backward/init), autograd.h, kernels novos, pool, serializer (identidade), notebooks/geradores, attn_bwd_parity, curriculum (phase3).
- **Varredura por padrão (greps) em 100% de src/include/scripts:** syncs, `.cpu()`, sorts host, `Device::CPU` hardcoded, catch-alls, stubs, GIL, auth, UTF-8, `errors=ignore`, tamanhos.
- **Sem leitura linha-a-linha (cobertura só-sweep):** http_api_server.cpp, nsos_sdk.cpp, mcts_reasoning, memory_system, holographic, kan/sprecher_kan, ttt_layer, layer_audit, tokenizer.cpp (parcial), dataloaders, self_healer. **Continuação natural:** Stage 1 do ARCHITECTURE_RISK lê esses na íntegra; itens novos entram aqui com a mesma régua.
- **Fora de escopo (fronteira do produto):** pastas de incubação (KernelOpen/CHRASS/CART/OXB/pantheon/raiz) e `modules/oxtamem` (lane própria).
