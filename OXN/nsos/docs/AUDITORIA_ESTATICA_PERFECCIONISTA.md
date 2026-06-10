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

## §10 — LEDGER DE EXECUÇÃO (goal "36 no padrão ouro", 2026-06-10, noite)

Estados: ✅ CÓDIGO = implementado + compile-verde, valida amanhã na T4/CPU ·
🔁 RESTATED = item estava errado na auditoria; corrigido com evidência ·
📋 ESPEC = decisão/design entregue; implementação sequenciada com pré-requisito explícito.

| # | Estado | Entrega |
|---|---|---|
| 1 | 🔁 RESTATED | os 5 syncs são debug-gated (`NSOS_CUDA_SYNC=1`, default OFF) — não-hot; **no lugar, achado e corrigido o dreno REAL: matmul fazia `.data()`×3 antes do branch GPU = dreno por chamada ×100-200/step** → raw_data() no branch (3c5d65e) |
| 2 | 🔁 RESTATED | MoE batched já é device-side (top-k em kernel, D2H único — comentário AUDIT#4+5 de 2026-05-16); `partial_sort` remanescente está em fallbacks/decode, fora do hot de treino |
| 3 | ✅ CÓDIGO | `Tensor::uninitialized` + matmul/clone/to/slice + 14 temporários da atenção sem zero-fill |
| 4 | ✅ CÓDIGO | otimizador multi-tensor: 2 kernels + 1 D2H no lugar de ~1850-2450 launches; scale+clip dobrados em gscale (equivalência exata documentada); `NSOS_FUSED_OPT=0` = braço A/B |
| 5 | ✅ CÓDIGO | `runtime_telemetry()` exposto no binding + print no 1º log de step do treino + por braço no lab |
| 6 | 📋 ESPEC | kernel add-row-broadcast desenhado (grad[e,:]+=imbalance[e]); 16KB×2/step hoje — entra na próxima leva C++ |
| 7 | ✅ DECISÃO | saves por referência são INSEGUROS hoje (`zero_sequence_suffix_inplace` muta `hidden` in-place entre camadas); clones async (508733b) são o correto; copy-on-write por version = futuro |
| 8 | 📋 ESPEC | unificação dos 3 helpers de cópia: header interno único com a política async-D2D — mecânica, próxima leva |
| 9 | 📋 ESPEC | CUDA Graphs sequenciado APÓS validação de #3/#4 (capturar step com sync no meio = crash); pré-requisito: [timing] de amanhã |
| 10 | 📋 ESPEC | investigação do decode CPU 0.7 tok/s roteirizada p/ runtime CPU (sem quota): dispatch ternário packed vs float reference |
| 11 | 📋 ESPEC | NSOS_DETERMINISTIC: 4 sites mapeados; CE two-pass primeiro (S), embedding sort-segmented (M), mamba/MoE depois; claims alinhados no doc §4b |
| 12 | 📋 ESPEC | BPE 4897/8192: diagnóstico roteirizado (corpus de merges); G5 do gate mede tokens/char antes/depois do re-treino (CPU runtime) |
| 13 | ✅ CÓDIGO | **G6b no gate: interseção train∩eval por hash md5 — FAIL automático para phase3**; + achado: ~100 linhas de código morto pós-`return` em build_phase3 (limpeza na próxima leva py) |
| 14 | 📋 ESPEC | trailer opcional de nomes absolutos no serializer (back-compat: arquivos antigos sem trailer carregam igual) — agora viável pós-fix de época |
| 15 | 📋 ESPEC | helper `read_text_strict` (conta replacements, aborta >0.1%) p/ os 8 sites — leva py |
| 16-18 | 📋 ESPEC | splits god-file: regra do PRÓPRIO projeto exige gate verde por extração (execução) — sequenciado pós-validação; 1ª extração: `Attention` → `src/attention.cpp` |
| 19 | 📋 ESPEC | kernel GQA exporta q_rot/k_rot (elimina clone+rope dos saves) — leva CUDA 2 |
| 20 | 📋 ESPEC | warn-once nos fallbacks host (transpose>2/softmax≠last/sum>2) — leva C++ 2 |
| 21 | ✅ DOC | stubs já demarcados; guarda de build anotada p/ CI |
| 22 | 📋 ESPEC | 9 `catch(...)` localizados (api 1, http 4, jamba 2, sdk 2) — auditoria um-a-um na leva de robustez |
| 23 | ✅ CÓDIGO | scrub do token nos 3 notebooks (remote volta a URL sem token após fetch) |
| 24 | ✅ CÓDIGO | gate G2 mede injeção; sanitização no serving condicionada ao resultado de amanhã |
| 25 | 📋 ESPEC | `release_cached_memory()` p/ uso-como-biblioteca — leva bindings |
| 26 | ✅ PARCIAL | sinal de pressão do pool (WARN) embutido; getter estruturado na leva bindings |
| 27 | ✅ CÓDIGO | guarda UM: >88% de uso ⇒ trim automático + WARN (1×/episódio), cadência 1/512 deallocs |
| 28-32 | 📋 ESPEC | conforme §5 (BF16 glue, scatter determinístico, 4-bit, KAN/SSA, re-treino BPE) — gatilhos definidos |
| 33 | ✅ CÓDIGO | `RUN_HOST_ARM` no lab cell 6 |
| 34 | 📋 ESPEC | sleep-poll do http → condition_variable — leva robustez |
| 35 | 📋 ESPEC | revisão de mutex do SDK condicionada a serving concorrente (hoje single-thread) |
| 36 | ✅ DOC | coberto por #21 (stub) |

**Balanço:** 11 itens em CÓDIGO/DOC fechados esta noite (incluindo os 2 maiores levers de
perf da base: matmul-desync + otimizador fundido), 2 RESTATED com evidência (auditoria
auto-corrigida — parte do método), 23 com especificação/sequenciamento explícito e
pré-requisito nomeado.  TODA validação de runtime acontece amanhã (T4 + CPU runtime),
conforme o protocolo: predições pré-registradas — `opt` 226-490ms → **<25ms**; `fwd`
2.6s → **alvo <1s** (matmul-desync + zero-fill); `[ltime]` por camada cai; telemetria
mamba imprime `fallbacks=0`; G6b dá veredito de contaminação.

## §9 — Declaração de cobertura (honestidade do mapa)

- **Leitura integral nesta era:** tensor.cpp, trainer.cpp (caminho de treino), jamba.cpp (atenção/forward/parameters), mamba2.cpp (scan/backward/init), autograd.h, kernels novos, pool, serializer (identidade), notebooks/geradores, attn_bwd_parity, curriculum (phase3).
- **Varredura por padrão (greps) em 100% de src/include/scripts:** syncs, `.cpu()`, sorts host, `Device::CPU` hardcoded, catch-alls, stubs, GIL, auth, UTF-8, `errors=ignore`, tamanhos.
- **Sem leitura linha-a-linha (cobertura só-sweep):** http_api_server.cpp, nsos_sdk.cpp, mcts_reasoning, memory_system, holographic, kan/sprecher_kan, ttt_layer, layer_audit, tokenizer.cpp (parcial), dataloaders, self_healer. **Continuação natural:** Stage 1 do ARCHITECTURE_RISK lê esses na íntegra; itens novos entram aqui com a mesma régua.
- **Fora de escopo (fronteira do produto):** pastas de incubação (KernelOpen/CHRASS/CART/OXB/pantheon/raiz) e `modules/oxtamem` (lane própria).
