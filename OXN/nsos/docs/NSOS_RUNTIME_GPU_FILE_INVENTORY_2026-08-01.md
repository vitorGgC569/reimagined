# Inventário completo do worktree NSOS auditado

Data do snapshot: 2026-08-01  
Escopo: `OXN/nsos`  
Estados: `M` modificado, `N` novo/não rastreado e `D` removido.

Este inventário registra as 192 entradas que estão diferentes do `HEAD` no
fechamento desta implementação. A finalidade é tornar
o fechamento reproduzível e separar código de produção, validação e artefatos
de pesquisa acumulados na sessão. A presença de um arquivo nesta lista não
afirma que toda a sua diferença foi criada exclusivamente pelo pacote de
otimização; o worktree já continha trabalho anterior e foi preservado.

## Build e documentação principal

Finalidade: integrar os backends, registrar contratos de produto/build/teste e
publicar a evidência técnica do fechamento.

- `M OXN/nsos/CMakeLists.txt`
- `M OXN/nsos/README.md`
- `M OXN/nsos/docs/GPU_OPTIMIZATION_ANALYSIS.md`
- `M OXN/nsos/docs/MODEL_PACKS.md`
- `M OXN/nsos/docs/NSOS_VALIDATION_STATUS.md`
- `M OXN/nsos/docs/PRODUCT.md`
- `M OXN/nsos/docs/TESTING.md`
- `N OXN/nsos/docs/AMD_GPU_BACKEND.md`
- `N OXN/nsos/docs/NSOS_RUNTIME_GPU_FILE_INVENTORY_2026-08-01.md`
- `N OXN/nsos/docs/RUNTIME_GPU_INDUSTRIAL_TRACEABILITY_2026-08-01.md`
- `N OXN/nsos/docs/benchmarks/nsos_runtime_gpu_final_gate_2026-08-01.json`

## Relatórios e artefatos de pesquisa anteriores preservados

Finalidade: manter a proveniência dos benchmarks arquiteturais, diagnóstico de
qualidade e comparações de modelos que antecederam o pacote industrial. Eles
não são usados como substituto dos gates do runtime.

- `N OXN/nsos/docs/benchmarks/NSOS_CORRECOES_ESTATICAS_RESUMO_2026-07-28.md`
- `N OXN/nsos/docs/benchmarks/nsos_git_and_real_models_2026-07-28.md`
- `N OXN/nsos/docs/benchmarks/nsos_last_layer_guard_ab_2026-07-28.md`
- `N OXN/nsos/docs/benchmarks/nsos_mamba_oxtamem_rx7600_2026-07-28.md`
- `N OXN/nsos/docs/benchmarks/nsos_robustness_architecture_gpu_gap_audit_2026-07-28.md`
- `N OXN/nsos/docs/benchmarks/oxta_contabil_amd_rx7600_2026-07-28.md`
- `N OXN/nsos/notebooks/oxta_contabil_amd_product.ipynb`

## Headers e contratos públicos/internos

Finalidade: formalizar ownership, identidade imutável, políticas numéricas,
portabilidade HIP/CUDA, buffers, kernels, checkpoint, I/O, inferência e
treinamento fail-closed.

- `M OXN/nsos/include/autograd.h`
- `M OXN/nsos/include/bitlinear.h`
- `M OXN/nsos/include/bitnet_gpu_dispatch.h`
- `M OXN/nsos/include/cuda/bitnet_math.cuh`
- `M OXN/nsos/include/cuda/cuda_memory.cuh`
- `M OXN/nsos/include/cuda/device_buffer.h`
- `M OXN/nsos/include/cuda/gpu_utils.h`
- `M OXN/nsos/include/cuda/kan_kernels.cuh`
- `M OXN/nsos/include/cuda/kernels.cuh`
- `M OXN/nsos/include/cuda/mamba_kernels.cuh`
- `M OXN/nsos/include/cuda/sparse_attention_kernels.cuh`
- `M OXN/nsos/include/dataloader.h`
- `M OXN/nsos/include/dataloader_v2.h`
- `M OXN/nsos/include/embedding.h`
- `M OXN/nsos/include/fabric.h`
- `M OXN/nsos/include/http_api_server.h`
- `M OXN/nsos/include/inspector.h`
- `M OXN/nsos/include/jamba.h`
- `M OXN/nsos/include/jamba_utils.h`
- `M OXN/nsos/include/layer_audit.h`
- `M OXN/nsos/include/lean_integration.h`
- `M OXN/nsos/include/mamba2.h`
- `M OXN/nsos/include/mcts_reasoning.h`
- `M OXN/nsos/include/monitor.h`
- `M OXN/nsos/include/nsos_arena.h`
- `M OXN/nsos/include/nsos_config.h`
- `M OXN/nsos/include/nsos_context.h`
- `M OXN/nsos/include/nsos_mpi.h`
- `M OXN/nsos/include/nsos_sdk.h`
- `M OXN/nsos/include/nsos_serializer.h`
- `M OXN/nsos/include/smart_loader.h`
- `M OXN/nsos/include/tensor.h`
- `M OXN/nsos/include/tensor_iterator.h`
- `M OXN/nsos/include/trainer.h`
- `N OXN/nsos/include/checkpoint_io.h`
- `N OXN/nsos/include/cuda/pinned_buffer.h`
- `N OXN/nsos/include/gpu_backend.h`
- `N OXN/nsos/include/gpu_gemm_provider.h`
- `N OXN/nsos/include/nsos/sha256.h`
- `M OXN/nsos/include/nsos/test_context.h`
- `N OXN/nsos/include/optimizer_runtime_policy.h`
- `M OXN/nsos/include/rierass_core.h`
- `N OXN/nsos/include/runtime_execution_identity.h`

## Implementação C++/HIP/CUDA

Finalidade: executar os contratos dos headers no caminho real de produção:
embedding CSR, Mamba state-major/chunked/LDS, GEMMs, optimizer, allocator,
checkpoint, telemetria, SDK, MPI, I/O e tratamento transacional de falhas.

- `M OXN/nsos/src/bindings.cpp`
- `M OXN/nsos/src/bitlinear.cpp`
- `M OXN/nsos/src/bitnet_gpu_dispatch.cpp`
- `M OXN/nsos/src/components.cpp`
- `M OXN/nsos/src/cuda/attention_train_kernels.cu`
- `M OXN/nsos/src/cuda/bitnet_kernels.cu`
- `M OXN/nsos/src/cuda/fused_optimizer_kernels.cu`
- `M OXN/nsos/src/cuda/kan_kernels.cu`
- `M OXN/nsos/src/cuda/kernels.cu`
- `M OXN/nsos/src/cuda/mamba_kernels.cu`
- `M OXN/nsos/src/cuda/moe_kernels.cu`
- `M OXN/nsos/src/cuda/sparse_attention_kernels.cu`
- `M OXN/nsos/src/dataloader.cpp`
- `M OXN/nsos/src/dataloader_v2.cpp`
- `M OXN/nsos/src/determinism.cpp`
- `M OXN/nsos/src/embedding.cpp`
- `M OXN/nsos/src/fabric_v2.cpp`
- `M OXN/nsos/src/globals.cpp`
- `M OXN/nsos/src/holographic.cpp`
- `M OXN/nsos/src/http_api_server.cpp`
- `M OXN/nsos/src/jamba.cpp`
- `M OXN/nsos/src/jamba_utils.cpp`
- `M OXN/nsos/src/kan.cpp`
- `M OXN/nsos/src/layer_audit.cpp`
- `M OXN/nsos/src/mamba2.cpp`
- `M OXN/nsos/src/mcts_reasoning.cpp`
- `M OXN/nsos/src/nsos_mpi.cpp`
- `M OXN/nsos/src/nsos_sdk.cpp`
- `M OXN/nsos/src/optimizer_4bit.cpp`
- `M OXN/nsos/src/smart_loader.cpp`
- `M OXN/nsos/src/sparse_attention.cpp`
- `M OXN/nsos/src/tensor.cpp`
- `M OXN/nsos/src/trainer.cpp`
- `M OXN/nsos/src/ttt_layer.cpp`
- `N OXN/nsos/src/gpu_backend.cpp`
- `N OXN/nsos/src/gpu_gemm_provider.cpp`
- `N OXN/nsos/src/runtime_execution_identity.cpp`
- `N OXN/nsos/src/sha256.cpp`

## Pipeline e ferramentas Python

Finalidade: preparar/provar proveniência de dados, executar treino PT-BR,
telemetria rolling, checkpoint assíncrono, verificação, benchmark e diagnóstico
sem introduzir PyTorch no produto autoral.

- `M OXN/nsos/scripts/bench_gpu_vs_cpu.py`
- `M OXN/nsos/scripts/analyze_cascade.py`
- `M OXN/nsos/scripts/attn_bwd_parity.py`
- `M OXN/nsos/scripts/audit_quantized_inference.py`
- `M OXN/nsos/scripts/bench_bitmamba_lut.py`
- `M OXN/nsos/scripts/bench_lut_tmac.py`
- `M OXN/nsos/scripts/bench_pacote_a_all.py`
- `M OXN/nsos/scripts/bench_v10_tokps.py`
- `M OXN/nsos/scripts/build_preference_bundle.py`
- `M OXN/nsos/scripts/chat_v10.py`
- `M OXN/nsos/scripts/chat_v10_ui.py`
- `M OXN/nsos/scripts/chat_v8.py`
- `M OXN/nsos/scripts/crit_experiments.py`
- `M OXN/nsos/scripts/cuda_env.py`
- `M OXN/nsos/scripts/dpo_phase.py`
- `M OXN/nsos/scripts/drive_sync.py`
- `M OXN/nsos/scripts/eval_v8.py`
- `M OXN/nsos/scripts/gpu_gold_validation.py`
- `M OXN/nsos/scripts/nsos_curriculum_lib.py`
- `M OXN/nsos/scripts/oxb_pack_tokens.py`
- `M OXN/nsos/scripts/probe_cascade.py`
- `M OXN/nsos/scripts/probe_chrass_ab.py`
- `M OXN/nsos/scripts/profile_bottlenecks.py`
- `M OXN/nsos/scripts/sft_phase.py`
- `M OXN/nsos/scripts/smoke_pacote_a.py`
- `M OXN/nsos/scripts/standalone_builder/rthook_cuda_dlls.py`
- `M OXN/nsos/scripts/standalone_builder/trainer_main.py`
- `M OXN/nsos/scripts/test_v10_responses.py`
- `M OXN/nsos/scripts/train_curriculum.py`
- `N OXN/nsos/scripts/build_sft_bundle_clean.py`
- `N OXN/nsos/scripts/fetch_real_datasets_clean.py`
- `N OXN/nsos/scripts/sample_step320_cpu.py`
- `N OXN/nsos/scripts/train_multidomain_curriculum.py`
- `N OXN/nsos/scripts/chat_gpu_interactive.py`
- `N OXN/nsos/scripts/oxta_contabil/audit_model_parameters.py`
- `N OXN/nsos/scripts/oxta_contabil/benchmark_policy.py`
- `N OXN/nsos/scripts/oxta_contabil/benchmark_product_architecture.py`
- `N OXN/nsos/scripts/oxta_contabil/benchmark_real_models.py`
- `N OXN/nsos/scripts/oxta_contabil/benchmark_real_models_babi.py`
- `N OXN/nsos/scripts/oxta_contabil/benchmark_real_models_pairwise.py`
- `N OXN/nsos/scripts/oxta_contabil/diagnose_mamba_quality_regression.py`
- `N OXN/nsos/scripts/oxta_contabil/generate_validation_manifest.py`
- `N OXN/nsos/scripts/train_ptbr_conversational.py`

## Testes, paridades e benchmarks compilados

Finalidade: provar invariantes CPU/HIP, gradcheck, bitwise, checkpoint/resume,
mixed precision, memória device-only, falhas injetadas, SDK, I/O e integração
E2E. O antigo teste v8 foi substituído pelo teste de continuação versionado pelo
contrato atual; não foi removida cobertura sem reposição.

- `M OXN/nsos/tests/bench_gpu_vs_cpu.cpp`
- `M OXN/nsos/tests/bench_lut_tmac.cpp`
- `M OXN/nsos/tests/gpu/gpu_parity_common.h`
- `D OXN/nsos/tests/gpu/test_gpu_checkpoint_v8_continuation.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_device_memory_contract.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_mixed_precision_contract.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_model_pack_dp4a_lifecycle.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_basic.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_decode_incremental.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_jamba.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_jamba_sparse.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_mamba_faithful.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_mamba_nstate.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_mamba_nstate_stream.cpp`
- `M OXN/nsos/tests/gpu/test_gpu_parity_mamba_proper_stream.cpp`
- `M OXN/nsos/tests/test_bitlinear.cpp`
- `M OXN/nsos/tests/test_bitnet_integrated.cpp`
- `M OXN/nsos/tests/test_circuit_api_pipeline.cpp`
- `M OXN/nsos/tests/test_components.cpp`
- `M OXN/nsos/tests/test_core_v2.cpp`
- `M OXN/nsos/tests/test_dataloader.cpp`
- `M OXN/nsos/tests/test_gpu_parity.cpp`
- `M OXN/nsos/tests/test_gradcheck.cpp`
- `M OXN/nsos/tests/test_http_api.cpp`
- `M OXN/nsos/tests/test_jamba.cpp`
- `M OXN/nsos/tests/test_layer_audit.cpp`
- `M OXN/nsos/tests/test_mamba2_reference.cpp`
- `M OXN/nsos/tests/test_mamba_parallel_scan_parity.cpp`
- `M OXN/nsos/tests/test_matmul.cpp`
- `M OXN/nsos/tests/test_mcts_reasoning_v3.cpp`
- `M OXN/nsos/tests/test_model_pack.cpp`
- `M OXN/nsos/tests/test_moe_router.cpp`
- `M OXN/nsos/tests/test_oxtamem_ffi.cpp`
- `M OXN/nsos/tests/test_python_binding_smoke.py`
- `M OXN/nsos/tests/test_sanity.cpp`
- `M OXN/nsos/tests/test_smart_loader_async.cpp`
- `M OXN/nsos/tests/test_stability.cpp`
- `M OXN/nsos/tests/test_thread_safety.cpp`
- `M OXN/nsos/tests/test_training_invariants.cpp`
- `M OXN/nsos/tests/train_e2e.cpp`
- `N OXN/nsos/tests/gpu/test_gpu_checkpoint_continuation.cpp`
- `N OXN/nsos/tests/gpu/test_gpu_deterministic_adamw_1000.cpp`
- `N OXN/nsos/tests/gpu/test_gpu_device_selection.cpp`
- `N OXN/nsos/tests/gpu/test_gpu_lowp_weight_cache.cpp`
- `N OXN/nsos/tests/gpu/test_gpu_parity_ttt.cpp`
- `N OXN/nsos/tests/test_curriculum_integrity.py`
- `N OXN/nsos/tests/test_gpu_backend_configuration.py`
- `N OXN/nsos/tests/test_product_benchmark_policy.py`
- `N OXN/nsos/tests/test_ptbr_conversational_script.py`
- `N OXN/nsos/tests/test_runtime_execution_identity.cpp`

## Totais do snapshot

- raiz/build: 2;
- documentação: 15;
- headers: 43;
- scripts: 43;
- implementação: 38;
- testes: 50;
- notebook: 1;
- total: 192 entradas no fechamento.

O diff rastreado neste ponto soma 26.415 inserções e 5.548 remoções em 146
arquivos. Arquivos novos não rastreados não entram nesses números do Git; por
isso os totais de linhas não devem ser usados como medida de cobertura.
