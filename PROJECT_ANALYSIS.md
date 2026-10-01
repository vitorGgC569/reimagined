# Análise estrutural e técnica do projeto

Data: 2026-09-19  
Branch: codex-gpu-gold-validation  
Estado: working tree fortemente modificado; esta análise considera o conteúdo atual dos arquivos.

## 1. Resumo executivo

Este repositório é um monorepo de produto, pesquisa, protótipos e histórico. Não é uma aplicação única.

A fronteira oficial definida pelo próprio projeto é:

- OXN/nsos: produto suportado, runtime neural C++/Python.
- modules/oxtamem: módulo Rust/Python integrado.
- legacy: arquivo histórico congelado.
- demais diretórios: incubação, pesquisa, benchmarks, demos ou artefatos.

O produto principal é uma biblioteca C++20, nsos_core, que concentra tensor, autodiff manual, arquitetura híbrida Jamba/Mamba/Attention/MoE, quantização BitNet 1.58-bit, treino, inferência, serialização, memória, API HTTP e bindings Python. CLI, servidor HTTP, módulo Python e CTest são fachadas sobre essa biblioteca.

O desenho é tecnicamente ambicioso e possui boas barreiras de validação, porém há uma matriz grande de estados: CPU, CUDA, HIP, AVX2, streaming, batch, packed inference, FP32/QAT, checkpoint/model pack e vários backends de memória. Os riscos principais são consistência entre esses caminhos, rastreabilidade do working tree e multiplicidade de entradas de build.

## 2. Método e limites

Foram inspecionados:

- README, PROJECT_BOUNDARY.json, PROJECT_SCOPE_STATUS.md e docs de fronteira;
- manifests e sistemas CMake, Cargo, Python, .NET e Docker;
- headers e implementações de OXN/nsos;
- integração C++/Rust/Python de modules/oxtamem;
- fluxo WPF/API/GLiNER/Docker do OContabil;
- raízes de incubação, validações, testes, benchmarks e legado;
- includes/imports, entry points, classes/funções, opções de build, limites de recurso, autenticação, persistência e fallbacks;
- CTest CPU já configurado em OXN/nsos/build-codex-cpu.

O resultado registra o mapeamento por arquivo, classe, função e bloco de responsabilidades. Repetir milhares de linhas literalmente seria menos útil que apontar o contrato de cada faixa de código e suas dependências.

## 3. Estado do repositório

O Git reportou 8.666 arquivos rastreados. Excluindo Skills, builds, datasets e artefatos, ainda há mais de mil arquivos de código, testes e documentação. A maior concentração é OXN/nsos, seguida por OContabil, modules/oxtamem, CHRASS, KernelOpen, CART e OXB.

O working tree não está limpo. O diff observado envolve mais de 170 arquivos, aproximadamente 29 mil linhas adicionadas e 6,7 mil removidas, concentradas no runtime/GPU de OXN/nsos e em modules/oxtamem. Existem ainda arquivos novos em OXN/nsos, experimental e benchmark_results.

Consequências:

1. o estado analisado não equivale ao último commit 9806828;
2. builds incrementais podem misturar objetos antigos e novos;
3. alterações de headers, CMake e kernels têm impacto transversal;
4. não se deve limpar, resetar ou sobrescrever o working tree sem autorização explícita.

Foi executado o CTest do build CPU existente. 38 testes passaram. test_circuit_api_pipeline permaneceu em execução muito além do tempo dos demais e foi interrompido. A conclusão correta é “38 passados e 1 teste com longa duração ou possível travamento”, não “39/39 verde”.

## 4. Fronteira oficial

| Camada | Caminho | Papel |
|---|---|---|
| Produto | OXN/nsos | Runtime, treino, inferência, packs, API, CLI e Python |
| Integrado | modules/oxtamem | Memória causal persistente Rust/Python |
| Release support | Docker, workflows, scripts e docs | Build, deploy e política |
| Incubação | KernelOpen, CHRASS, CART, OXB, Pantheon, hardware etc. | Pesquisa e protótipos |
| Legado | legacy | Arquivo histórico, não executável |

O root CMake ainda é orientado ao Pantheon/KernelOpen, enquanto o produto é OXN/nsos. Portanto, o build da raiz não é o build canônico do produto.

## 5. Arquitetura geral

    Python scripts -> nsos_ext / InferenceEngine <- CLI
                               |
    HTTP clients -> nsos_api_server
                               |
                         nsos_core
                               |
             Tensor + Model + Trainer + Serializer
                               |
                      CPU / CUDA / HIP / AVX2
                               |
                     optional OxtaMem FFI

Pantheon, KernelOpen, OXB, CART, CHRASS, hardware, research e colab formam trilhas paralelas de incubação.

## 6. Produto OXN/nsos

### 6.1. OXN/nsos/CMakeLists.txt

Responsabilidades:

- fixa C++20, PIC, OpenMP e flags portáveis;
- define nsos_core e sua lista de sources;
- escolhe NONE, CUDA, HIP ou AUTO;
- separa kernels GPU de código host;
- isola translation units AVX2;
- integra opcionalmente TurboQuant externo, OxtaMem via Cargo e profiler;
- constrói nsos_ext, nsos_cli, nsos_api_server e ferramentas de circuito;
- registra CTest e audita testes órfãos/quarentenados;
- registra gates GPU e testes Python por backend.

Faixas críticas:

- linhas 32–72: opções, paths e parâmetros de backend;
- 78–122: composição de nsos_core;
- 132–190: baseline AVX/FMA e AVX2;
- 226–510: resolução/staging CUDA e HIP;
- 512–599: binding, CLI, API, profiler e OxtaMem;
- 601 em diante: CTest, auditoria de cobertura e GPU tests.

O CMake é parte da arquitetura: backend, streams, bibliotecas carregadas e distribuição de DLLs mudam o comportamento do produto.

### 6.2. Headers de configuração e plataforma

- include/nsos_config.h: Device, HybridComposition, ModelConfig, versão/fingerprint do modelo e validação de dimensões, slots, MoE, Mamba, dropout, RoPE, quantização e compatibilidade.
- include/gpu_backend.h: contrato de device, streams, cópias, sincronização e política de erro.
- include/gpu_gemm_provider.h: seleção de GEMM e provider de álgebra.
- include/nsos_math.h: GEMM e vetores.
- include/nsos_simd.h: scalar/AVX/NEON.
- include/shape_utils.h: validação de shapes.
- include/nsos/determinism.h: seeds e política determinística.
- include/nsos/sha256.h: hash de arquivos/configuração/payload.
- include/nsos/assertions.h: assertions de diagnóstico/teste.

include/nsos_config.h é o contrato mais importante: qualquer campo novo deve atualizar validação, fingerprint, serialização e testes.

### 6.3. Tensor e autodiff manual

- include/tensor.h: TensorShape, ownership, device, pool, cópia, H2D/D2H/D2D, GEMM, ativações, losses, slicing, gradientes e estatísticas.
- include/tensor_iterator.h: iteração/indexação.
- include/autograd.h: Parameter, gradiente, nome, versão e update.
- include/module.h: forward, to, parameters e state_dict.
- include/nsos_context.h: ativações/backward indexados por CtxKey e cancelamento.
- include/nsos_arena.h: arena, marks, rewind e scopes.
- include/nsos_math.h, hadamard.h e nsos_simd.h: bricks matemáticos.

Não existe grafo autograd genérico. O backward é manual por módulo. Por isso test_gradcheck, parity CPU/GPU e training invariants são barreiras essenciais.

### 6.4. Headers de modelo

- include/embedding.h: lookup, backward e Slender quantizado.
- include/bitlinear.h: linear ternário, quantização, FlatQuant, packed weights e LoQA.
- include/bitnet_adapter.h: packing e GEMM 1.58-bit CPU.
- include/bitnet_gpu_dispatch.h: despacho GPU BitNet.
- include/jamba.h: Attention, MoERouter, JambaBlock, JambaModel, cache, sessão, streaming e decoder.
- include/mamba2.h: Mamba2SSD, SSD, streaming, snapshot, recomputação e telemetria.
- include/kan.h: BitFastKAN.
- include/sprecher_kan.h: KAN com grid adaptativo e bases RBF/B-spline/Chebyshev/Legendre.
- include/ttt_layer.h: adaptação online, momentum, Hamiltonian mode, snapshot e reset.
- include/sparse_attention.h: seleção e execução sparse.
- include/chrass_layer_v2.h: layer topológica experimental.
- include/rierass_core.h: helpers espectrais/experimentais.
- include/lut_tmac.h, lut_cas.h e lut_cache.h: lookup table e armazenamento content-addressed.
- include/turboquant.h: TurboQuant e fallback.
- include/optimizer_4bit.h: estado de optimizer em 4 bits.

JambaModel é o maior ponto de acoplamento: possui stack de layers, tying embedding/head, configuração, sessões, caches, streaming, packs edge, reasoning, auditoria e profiler. Mudanças nele precisam de forward, backward, save/load, reload e decode incremental.

### 6.5. Headers de treino, SDK e runtime

- include/trainer.h: scheduler de fases/QAT, objectives, telemetria, checkpoint, optimizers, acumulação, cancellation e auditoria.
- include/nsos_sdk.h: GenerationOptions, GenerationMetrics, ModelLoadOptions e InferenceEngine.
- include/nsos_serializer.h: checkpoint/model pack, versionamento, identities, hashes e compatibilidade.
- include/tokenizer.h: encode/decode, BPE, tokens especiais e packs.
- include/dataloader.h e dataloader_v2.h: contratos de dataloader; a versão ativa é v2.
- include/smart_loader.h: fila e worker de I/O.
- include/layer_audit.h: estatísticas por layer, fase, ativação, gradiente e drift.
- include/numerical_guard.h: NaN/Inf, clamps e contagem de correções.
- include/inspector.h, monitor.h, predictive_failure.h e self_healer.h: observabilidade/recuperação.

### 6.6. Implementações C++ principais

- src/tensor.cpp: allocator/pool, cópia entre devices, GEMM CPU/cuBLAS/hipBLAS, mixed precision, lowp cache, reductions, ativações, losses e slicing. É o maior fundamento numérico.
- src/gpu_backend.cpp: seleção de device, env policy, streams, sincronização e erros.
- src/gpu_gemm_provider.cpp: provider de GEMM.
- src/nsos_math.cpp: operações matemáticas host.
- src/bitlinear.cpp: quantização, packing, forward e backward.
- src/bitlinear_quantize_avx2.cpp: quantização AVX2 isolada.
- src/bitnet_adapter.cpp e src/bitnet_adapter_avx2.cpp: GEMM ternário e dispatch host.
- src/embedding.cpp: embedding, cache e Slender em CPU/GPU.
- src/jamba.cpp: Attention, MoE, block, model, cache, sessões, streaming, batch e packs. É o maior arquivo de arquitetura neural.
- src/jamba_utils.cpp: utilitários de Jamba.
- src/mamba2.cpp: conv causal, selective SSD, faithful/proper paths, backward manual e streaming.
- src/kan.cpp e src/sprecher_kan.cpp: variantes KAN.
- src/ttt_layer.cpp: adaptação online.
- src/sparse_attention.cpp: seleção/compute sparse.
- src/lut_tmac.cpp e src/lut_cas.cpp: TMAC e CAS.
- src/chrass_layer_v2.cpp e chrass_layer_backward_v2.cpp: layer CHRASS.
- src/trainer.cpp: forward, losses, backward, clip, QAT, objectives auxiliares, optimizer commit, checkpoint, cancellation e rollback. É o maior centro de estado.
- src/tokenizer.cpp: tokenizer e persistência.
- src/nsos_sdk.cpp: InferenceEngine, load staged, tokenizer, geração, sampling, metrics, treino e save.
- src/bindings.cpp: exporta Tensor, ModelConfig, JambaModel, Trainer, InferenceEngine e telemetrias para Python.
- src/nsos_cli.cpp: CLI oficial.
- src/api_server.cpp: parsing de args/env, auth token, bind e limites.
- src/http_api_server.cpp: roteamento HTTP, autenticação, JSON, geração e health.
- src/nsos_circuit_trainer.cpp: pipeline de circuito.
- src/causal_memory_store.cpp: store append-only, checksum, process lock, índice e tail recovery.
- src/oxtamem_ffi.cpp: carregamento dinâmico do C ABI e operações de memória.
- src/memory_system.cpp: memória episódica/instrucional/mensagens, clusters e retrieval.
- src/holographic.cpp: bind, bundle, permute, query e retrieval.
- src/mcts_reasoning.cpp: UCT, expansão, avaliação, backpropagation e UltraPlan.
- src/coordinator.cpp: workers, mailbox, reasoning e memória.
- src/fabric_v2.cpp e src/nsos_mpi.cpp: fabric local/MPI mock.
- src/layer_audit.cpp: auditoria por layer.
- src/self_healer.cpp, numerical_guard.cpp, determinism.cpp e globals.cpp: suporte operacional.

### 6.7. Kernels GPU

Headers:

- include/cuda/cuda_memory.cuh: alocação/cópia/erro.
- include/cuda/device_buffer.h e pinned_buffer.h: ownership RAII device/host.
- include/cuda/gpu_utils.h: checks e grid.
- include/cuda/bitnet_math.cuh: ternário/quantização.
- include/cuda/kernels.cuh: manifesto de kernels tensor/GEMM/activation/reduction/optimizer.
- include/cuda/mamba_kernels.cuh: faithful/proper forward/backward/stream.
- include/cuda/sparse_attention_kernels.cuh e kan_kernels.cuh: kernels especializados.

Implementações:

- src/cuda/kernels.cu: tensor, GEMM, casts, activations e reductions.
- src/cuda/mamba_kernels.cu: selective scan, conv, state expansion, forward/backward faithful e paths determinísticos.
- src/cuda/bitnet_kernels.cu: BitNet.
- src/cuda/moe_kernels.cu: routing e experts.
- src/cuda/kan_kernels.cu: KAN.
- src/cuda/sparse_attention_kernels.cu: sparse attention.
- src/cuda/attention_train_kernels.cu: attention training.
- src/cuda/fused_optimizer_kernels.cu: optimizer fundido/mixed precision.

Riscos GPU: fallback host silencioso, semântica de stream com NSOS_CUDA_PTDS, diferenças CUDA/HIP, arquitetura alvo, lifetime de buffers e parity CPU/GPU.

### 6.8. Python e avaliação

- python/nsos/__init__.py e shield.py: pacote e guard.
- python/nsos_mamba/core.py: wrapper do modelo/treino Mamba.
- python/nsos_mamba/datasets.py: datasets/preparação.
- python/setup.py e pyproject.toml: empacotamento.
- inference.py, evaluation.py e data_pipeline.py: fachadas auxiliares.
- eval/adapters/base.py: contrato.
- eval/adapters/dummy_adapter.py: adapter de teste.
- eval/adapters/nsos_adapter.py: nsos_ext, tokenizer, score e generation.
- eval/benchmarks/arc.py, hellaswag.py, mmlu.py, humaneval.py, wikitext.py: benchmarks gerais.
- eval/benchmarks/ptbr_assin2.py, ptbr_enem.py, ptbr_perplexity.py: benchmarks PT-BR.
- eval/scoring/multiple_choice.py e perplexity.py: scoring.
- eval/orchestrator.py: suites, scorecard e persistência.
- eval/tests/test_smoke.py: smoke.

### 6.9. Scripts

Gates/infra: benchmark_gate.py, gpu_gold_validation.py, fuzz_surface_smoke.py, cuda_env.py, gpu_arch_detect.py, native_module.py, run_scorecard.py, profile_bottlenecks.py, heatmap_profiler.py, inference_perf_probe.py, parity_triage.py, audit_quantized_inference.py e summarize_layer_audit.py.

Dados/treino: nsos_curriculum_lib.py, train_curriculum.py, train_multidomain_curriculum.py, train_ptbr_conversational.py, sft_phase.py, dpo_phase.py, build_curriculum.py, build_sft_bundle.py, build_preference_bundle.py, consolidate_supervised.py, fast_consolidate_signal.py, generate_synthetic_instructions.py, distill_generate_teacher_data.py, fetch_real_datasets.py, verify_curriculum_split.py, verify_canarim_sft.py, verify_tokenizer_v4.py, tokenizer_gold_gate.py, tokenizer_dataset_probe.py, quantize_int4.py, repro_qat_activation.py e rebuild_tokenizer_v4.py.

Chat/operação: chat_model.py, chat_v8.py, chat_v10.py, chat_v10_ui.py, chat_gpu_interactive.py, chatml.py, container_runtime.py e standalone_builder/*.

Scripts com v8/v10/legacy/pilot/experimental não são API estável por nome; o caminho canônico é o documentado em PRODUCT, RELEASE e CI.

### 6.10. Testes

Grupos ativos:

- base: test_sanity, test_core_v2, test_components, test_matmul, test_gemm_parity;
- tensor/quantização: test_bitlinear, test_bitmamba_lut, test_bitnet_integrated, test_optimizer_4bit, test_turboquant, test_slender_embedding, test_lut_tmac, test_lut_cas;
- arquitetura: test_jamba, test_mamba2, test_mamba2_reference, test_mamba_parallel_scan_parity, test_moe_router, test_moe_training, test_kan, test_sparse_attention, test_ttt_layer_kernel, test_ttt_validation e test_jamba_with_chrass;
- treino/estado: test_gradcheck, test_training_invariants, test_runtime_execution_identity e train_e2e;
- runtime: test_inference_engine, test_model_pack, test_tokenizer, test_dataloader, test_smart_loader_async, test_memory_causal_store e test_oxtamem_ffi;
- API/infra: test_http_api, test_circuit_api_pipeline, test_layer_audit, test_thread_safety, test_stability, test_holographic_full, test_mcts_reasoning_v3, test_ultra_integration e test_swarm_orchestration;
- GPU: tests/gpu/test_gpu_* para parity, Mamba, Jamba, MoE, sparse attention, TTT, checkpoint, mixed precision, memory e optimizer.

Quarentenados explicitamente no CMake: test_suite.cpp, test_v2_features.cpp, test_vulnerability_fixes.cpp e tests/unit/test_tensor.cpp.

## 7. Fluxos principais

### 7.1. Inferência

1. InferenceEngine::load_model tenta model pack.
2. Configuração, digest, tokenizer e payload são validados.
3. Um engine staged é preenchido; o commit só ocorre após carregamento completo.
4. Tokenizer gera IDs e sanitize_token_ids valida o vocabulário.
5. JambaModel executa embedding, blocks híbridos e head.
6. Streaming mantém KV cache e estado Mamba; batch usa snapshots.
7. Sampler aplica temperature/top-k/top-p/repetition/no-repeat/EOS.
8. Greedy GPU é usado apenas quando o contrato de device está atendido.
9. Métricas registram prompt, truncamento, tokens e tempo.
10. Falha na limpeza pode deixar o engine fail-stop poisoned, exigindo reload.

### 7.2. Treino

1. Python prepara currículo, bundle e tokenizer.
2. nsos_ext cria JambaModel e Trainer.
3. scheduler define warmup, QAT, replay e objetivos.
4. forward manual salva Context.
5. losses combinam cross-entropy, repetition, logit L2, MoE, sparse selector e QAT.
6. backward manual percorre as layers.
7. gradients são auditados, clipados e acumulados.
8. optimizer commit atualiza parâmetros/state.
9. checkpoint salva parâmetros, optimizer, RNG e identidade.
10. avaliação decide progressão de fase.

### 7.3. Memória

MemorySystem mantém clusters em memória e persiste episódios/mensagens em CausalMemoryStore ou OxtaMemFFI. A escrita durável precede a mutação do índice volátil. Retrieval tem limites de profundidade, bytes e valores finitos.

### 7.4. API

api_server.cpp valida bind/token/context/tokens e cria o servidor. http_api_server.cpp trata autenticação, JSON, health e generation. É uma API de processo, não identidade multiusuário.

## 8. Módulo modules/oxtamem

### Rust

- oxta_engine/src/engine.rs: GeodesicEngine, mmap, nodes encadeados, heads, usearch, journal checksummed, snapshot, compaction, replay e bounds.
- oxta_engine/src/lib.rs: C ABI/PyO3, handles, conversões, serialização segura e erros.
- oxta_engine/src/server.rs: RESP/TCP, bind, autenticação e limites.
- oxta_engine/src/main.rs: CLI do servidor.
- oxta_engine/src/sharding.rs: sharding experimental.
- Cargo.toml: Rust 2024, memmap2, rkyv, usearch, Tokio, Clap e PyO3 opcional.

### Python

- python/oxta_mem/core.py: protótipo neural/causal PyTorch.
- python/oxta_mem/sdk.py: SDK e serialização segura.
- python/oxta_mem/selective.py: seleção/recall experimental.
- python/oxta_mem/langchain.py: integração opcional.
- nn/*: demos/modelos experimentais.

O módulo declara que não usa pickle; aceita bytes, strings, JSON, NumPy sem object dtype e tensors PyTorch via payload NumPy seguro.

O CMake pode desligá-lo com NSOS_BUILD_OXTAMEM=OFF. O engine Rust é custom target e a integração C++ é opcional; o CTest CPU não cobre sempre o FFI.

## 9. OContabil

OContabil é um produto paralelo de documentos contábeis, com CI própria.

### Desktop

- OContabil/OContabil.csproj: .NET 8 WPF, SQLite/SQLCipher, ONNX, Tokenizers, Tesseract, PDF, ClosedXML, QuestPDF, LiveCharts, Serilog e WebView2.
- App.xaml/App.xaml.cs/AssemblyInfo.cs: bootstrap e recursos.
- MainWindow.xaml(.cs): shell, navegação, atalhos, sessão e WebView.
- Models/*: cliente, documento e configuração.
- Views/* e ViewModels/*: UI MVVM.
- Services/AppSettings.cs: configuração.
- AuthService.cs, PasswordHasher.cs, LoginThrottleService.cs, SessionService.cs: auth local, PBKDF2, rate limit e timeout.
- Db*, BackupService.cs, AuditLogger.cs: SQLCipher/SQLite, migração, backup e trilha.
- DocumentProcessingQueue.cs: fila persistida.
- DocumentTextExtractor.cs, OcrService.cs, NfeXmlExtractor.cs, OfxParserService.cs, RegexExtractionService.cs: ingestão/OCR/parsers/fallback.
- GlinerOnnxService.cs, GlinerService.cs, ExtractionMerge.cs: ONNX → Python/HTTP → regex.
- Services/Exports/*: CSV, Excel, Domínio, SPED, PDF e ExportManager.
- WebBridge.cs: ponte WebView/serviços.
- BrasilApiService.cs e CertificateService.cs: integrações externas/preparatórias.
- TelemetryService.cs, SafeLog.cs, SecurePath.cs e SecureCsv.cs: telemetria/logs/segurança.

### API

- OContabil.API/Program.cs: DI, Serilog, SQLite, JWT, CORS, Swagger, middleware e seed.
- Controllers/AuthController.cs: login.
- ClientsController.cs: clientes.
- DocumentsController.cs: upload/listagem/status.
- HealthController.cs: health.
- Data/* e Entities/ApiUser.cs: banco e seed.
- Dto/AuthDtos.cs: contratos.
- Services/AuthService.cs e JwtTokenService.cs: credenciais e JWT.

Riscos: docker-compose possui chave JWT default explícita; multi-tenant é roadmap; EnsureCreated não substitui migrações versionadas; HTTPS/CORS dependem do reverse proxy.

### GLiNER2

- gliner2/inference/engine.py: schema, tokenização, span scoring, entidades, relações, estruturas e formatação.
- inference/schema_model.py: validação Pydantic.
- processor.py: transformação de schema/batch.
- model.py/layers.py: modelo.
- training/data.py, trainer.py e lora.py: datasets, treino, checkpoints e LoRA.
- api_client.py: cliente.
- gliner2-service/service.py: FastAPI /health, /extract e /extract-file, modelo lazy, PyMuPDF/Tesseract.

ContaDocAI é uma UI WPF prototípica com MockDataService; não confundir com o produto OContabil/OContabil.

## 10. Incubação

### Pantheon

include/pantheon/* contém losses/regularizers de cognition, structure, physics, response, social e frontier. src/pantheon/* são translation units pequenas e grande parte da lógica é header-only. bindings/python_bindings.cpp expõe o módulo pantheon. include/pantheon_engine.hpp/src/pantheon_engine.cpp ligam o engine ao KernelOpen.

O root CMake constrói Pantheon e KernelOpen por padrão. Isso não é o build canônico de NSOS.

### KernelOpen

Fabric UHK experimental com Aion, ring buffers, CUDA persistent kernel, firmware, JIT x64, Ghost, graph, BCI, quantum, photonic, analog e dashboard. O próprio VALIDATION_REPORT classifica Aion como buildável e o UHK/exótico como parcial ou dependente de hardware/SDK. Claims de TOPS/latência não são evidência de produto.

### OXB

OXB/aion_core_cpp implementa BitPacking, LinearModel/RMI, Hilbert, RingBuffer, SmartLoader e OX3Serializer, com bindings, benchmarks, exemplos e bateria de validação. O relatório registra 16/16 testes, mas também que claims de performance foram superestimados no Windows/MSVC, SmartLoader é Linux-only e AVX-512 é placeholder/scalar.

### CART

Framework PEFT C++/PyTorch com LoRA, LoRA_GA, DoRA, IA3, BOFT, KAN, MiSS, FullFinetuning, SSM, TransMamba e TurboFusion. Tem modo standalone e extensão PyTorch. O relatório registra 21 testes, mas enablePureScalingMode continua gap conhecido.

### CHRASS

Conjunto heterogêneo de pesquisas de grafos/matemática: CHRASS layer, Kimera SSSP, RIERASS/Riemann, TSP, Collatz, Mersenne, Navier, coloring, RSA, Shannon e VLSI. chrass_layer_v2 foi integrado a testes NSOS; a pasta inteira continua incubação.

### Hardware, research, benchmarks e Colab

hardware contém HDL/tesbench BitNet sem fluxo FPGA de produto. research/lateral_inhibition e research/theses_validation são experiências controladas. benchmarks/academic_suite mede convergência, memória, roofline, throughput, Tiny Shakespeare e MQAR. experimental contém comparativos e treinos não-canônicos. colab contém notebooks e geradores de validação T4/PT-BR/memória.

### Legacy e Skills

legacy é append-only, não é buildado nem testado. Skills contém projetos/plugins de tooling e não pertence ao runtime; incluí-lo em métricas distorce a arquitetura.

## 11. Integrações

| Origem | Destino | Mecanismo | Estado |
|---|---|---|---|
| Python | nsos_core | pybind11/nsos_ext | Produto |
| CLI | nsos_core | link C++ | Produto |
| HTTP NSOS | nsos_core | servidor C++ | Produto |
| CUDA | Tensor/Jamba/Mamba/Trainer | CUDA/cuBLAS | Condicionado |
| AMD | Tensor/Jamba/Mamba/Trainer | HIP/hipBLAS | Condicionado |
| OxtaMem | MemorySystem | C ABI/dlopen | Integrado opcional |
| Pantheon | KernelOpen | CMake/link | Incubação |
| OXB/CART/CHRASS | linhas próprias/alguns testes | standalone/link | Incubação |
| OContabil desktop | serviços locais | in-process | Produto separado |
| OContabil API | SQLite/GLiNER | EF/HTTP/JWT | Produto separado |
| GLiNER service | modelo/OCR | FastAPI/Docker | Produto separado |

## 12. Pontos críticos e riscos

### P0 — working tree

Mais de 170 arquivos alterados tornam regressões difíceis de atribuir. Congelar o estado antes de refatorar é obrigatório.

### P0 — CTest longo

test_circuit_api_pipeline não terminou na checagem CPU. Deve ter timeout, logs de fase e isolamento de rede/arquivo.

### P0 — duas portas de build

Root CMake constrói Pantheon/KernelOpen; produto é OXN/nsos. Scripts antigos ainda compilam os dois mundos e removem build dirs. Isso favorece build acidental do código errado.

### P1 — backward manual

Tensor, BitLinear, Attention, Mamba, KAN, TTT e Trainer têm backward manual. Um bug pode preservar shapes e treinar errado. Gradcheck/parity/invariants são gates.

### P1 — modos combinatórios

Backend, precision, streaming, batch, attention, Mamba, MoE, KAN, TTT, CHRASS, QAT e packed weights geram uma matriz grande. ModelConfig/fingerprint deve ser a autoridade única.

### P1 — serialization/model pack

Mudança de ModelConfig, aliases, layout ou dtype exige versão, digest, strict load e reload tests.

### P1 — fallback GPU

Fallback host pode esconder regressão de performance ou causar acesso inválido. strict_gpu_execution e telemetria devem permanecer nos gates.

### P1 — OxtaMem FFI

Como o Rust é opcional, a integração pode ficar sem cobertura. ABI, paths, lifetime, persistência e restart precisam de lane própria.

### P1 — API/rede

NSOS usa token de processo; OContabil usa JWT. Tokens devem ser obrigatórios fora de loopback, chaves não devem ter default válido em produção e TLS/CORS devem ser verificados no proxy.

### P2 — claims misturados

README, manifestos e notebooks possuem claims que relatórios posteriores relativizam. Todo benchmark precisa registrar hardware, seed, build, dataset, split, commit e limitações.

### P2 — artefatos e duplicações

Datasets, notebooks, checkpoints e resultados estão misturados com source. Há múltiplos CMakeLists, bindings, setup.py e componentes de nomes semelhantes em raízes diferentes.

## 13. Pontos fortes

- Fronteira produto/incubação explícita e validada por script.
- CMake de NSOS com CPU/CUDA/HIP, gates e auditoria de testes.
- Cobertura de tensor, gradcheck, packs, streaming, memória, HTTP, thread safety e GPU parity.
- OxtaMem com bounds, journal checksum, tail recovery, locks e testes de corrupção/restart.
- OContabil com fallback offline, testes de parser, senha, export, segurança e throttle.
- Relatórios que registram claims não reproduzidos e gaps, em vez de esconder limitações.

## 14. Próximas ações recomendadas

1. Congelar o working tree em branch/commit de auditoria e registrar hashes dos builds.
2. Tornar OXN/nsos o único entrypoint documentado de build do produto.
3. Diagnosticar test_circuit_api_pipeline com timeout e logs.
4. Gerar matriz automática ModelConfig × backend × modo para combinações suportadas.
5. Criar lane CI específica para OxtaMem FFI em Windows/Linux.
6. Separar datasets/artefatos grandes de source e adicionar manifests/hashes.
7. Reduzir a superfície de produto: Tensor/Jamba/Mamba/Trainer/SDK/Tokenizer/pack/API.
8. Manter CHRASS, TTT, KAN, Pantheon, KernelOpen, CART e OXB com claims experimentais até terem owner e gate.
9. Para OContabil, remover defaults perigosos de JWT, documentar migrações e separar desktop/API/GLiNER.

## 15. Conclusão

O projeto possui um núcleo real e tecnicamente consistente em OXN/nsos, uma memória integrada em modules/oxtamem e um conjunto expressivo de pesquisa parcialmente validada. O problema central não é falta de código: é governança de fronteiras, reprodutibilidade e controle da matriz de execução.

A ordem segura para qualquer nova alteração é:

1. PROJECT_BOUNDARY.json e PROJECT_SCOPE_STATUS.md;
2. OXN/nsos/CMakeLists.txt;
3. nsos_config.h, tensor.h, jamba.h, mamba2.h, trainer.h, nsos_sdk.h e nsos_serializer.h;
4. tensor.cpp, jamba.cpp, mamba2.cpp, trainer.cpp e nsos_sdk.cpp;
5. bindings, CLI/API e scripts de gate;
6. OxtaMem FFI/Rust;
7. somente depois as raízes de incubação e claims de benchmark.
