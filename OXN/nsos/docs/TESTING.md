# NSOS Test Boundary

The official product gate is the test set registered by `OXN/nsos/CMakeLists.txt`
for a CPU build with `NSOS_BUILD_OXTAMEM=OFF`.

## Official Product CTest

Current product CTest contains:

- `test_sanity`
- `test_bitlinear`
- `test_bitnet_integrated`
- `test_mamba2`
- `test_kan`
- `test_ttt_layer_kernel`
- `test_inference_engine`
- `test_tokenizer`
- `test_python_binding_smoke`
- `test_mcts_reasoning_v3`
- `test_memory_causal_store`
- `test_jamba`
- `test_model_pack`
- `test_layer_audit`
- `test_gpu_parity`
- `test_http_api`
- `test_circuit_api_pipeline`
- `test_holographic_full`
- `test_ultra_integration`
- `test_swarm_orchestration`
- `train_e2e`

`test_oxtamem_ffi` belongs to the integrated OxtaMem lane and is registered only
when the Rust engine target is built by CMake.

`test_layer_audit` is the official end-to-end audit smoke. It verifies per-layer
forward/backward records, tensor health stats, MoE router distribution, training
step metadata, model-pack save/reload, and pre/post reload phase comparison.

## Candidate Tests Not Yet Product-Gated

These files are real test assets, but they are not release-blocking until they
are audited for runtime, determinism, platform support, and CI cost:

- `sanity_check.cpp`
- `test_components.cpp`
- `test_core_v2.cpp`
- `test_matmul.cpp`
- `test_moe_router.cpp`
- `test_moe_training.cpp`
- `test_stability.cpp`
- `test_suite.cpp`
- `test_thread_safety.cpp`
- `test_v2_features.cpp`
- `test_vulnerability_fixes.cpp`
- `unit/test_tensor.cpp`
- `verify_roadmap.cpp`
- `verify_sre_engineering.cpp`

Promotion rule: a candidate enters product CTest only after it builds cleanly on
Windows and Linux, has bounded runtime, uses unique temporary paths, and does not
depend on optional modules unless guarded.
