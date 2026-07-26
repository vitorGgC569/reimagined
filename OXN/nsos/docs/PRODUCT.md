# NSOS Product — Single Source of Truth

> This is the only document in the repository that asserts product capability. Other documents describe, plan, or audit; only `PRODUCT.md` claims.
>
> When this file disagrees with anything else in the repo (READMEs, roadmaps, slides, blog posts), this file wins. Update this file when claims change.

---

## What is supported

| Surface | Status | Notes |
|---|---|---|
| `OXN/nsos` CPU runtime | **Supported (MVP)** | Buildable on Linux/Windows; CPU float and CPU packed paths validated. |
| `OXN/nsos` HTTP API server (`nsos_api_server`) | **Supported (MVP)** | Token auth required when host is not loopback. Admin endpoints (`/train-*`, `/pack`) opt-in via `--enable-admin-endpoints`. |
| `OXN/nsos` CLI (`nsos_cli`) | **Supported (MVP)** | `--help` lists subcommands. |
| `OXN/nsos` Python binding (`nsos_ext`) | **Supported (MVP)** | UTF-8 sanitized at boundary (`tokenizer.cpp::sanitize_utf8()`). |
| Model packs (load / save / round-trip) | **Supported (MVP)** | SHA-256 + FNV1a checksums, atomic `tmp + rename`, per-file size limits, path canonicalization, explicit `quantization_ready` policy. |
| Tokenizer packs | **Supported (MVP)** | Bounded line/token/id counts. |
| `modules/oxtamem` companion (Rust + Python) | **Supported as integrated module** | Loopback default, optional auth token, frame/connection/timeout limits. Not a public service. NSOS uses it via dynamic-loaded FFI (`OXN/nsos/src/oxtamem_ffi.cpp`). |
| GPU (CUDA) float | **Experimental** | Native CUDA paths and decomposed parity tests exist. GPU gates are registered only by CUDA builds and are fail-closed: missing CUDA or hardware is a failure, never a pass/skip. |
| GPU packed runtime | **Experimental** | Native packed/DP4A paths exist and can be forced with `NSOS_FORCE_PACKED_WEIGHTS=1`. Promotion requires the full parity, accuracy and performance gates on real target GPUs. |
| TTT (Test-Time Training) | **Research only** | Snapshot/replay and serving parity not formally closed. |
| Distributed / MPI | **Experimental** | `include/mpi_mock.h` is an explicit mock; do not present as production. |
| Lean / formal verification | **Research only** | The restricted verifier is fail-closed, but it is not a Lean proof kernel and creates no formal-verification claim. |
| Anything in `legacy/` | **Not supported** | Frozen, append-only, not on any execution path. |
| Anything outside `OXN/nsos` and `modules/oxtamem` | **Not supported** | See `INCUBATION.md`. |

## Validated profiles

- **`mamba_small`** — production small-profile baseline.
- **`small`** — legacy alias for `mamba_small`.
- **`hybrid_pilot`** — validates attention + Mamba (no TTT).
- **`hybrid_small`** — validates attention + sparse MoE + Mamba.
- **`TTT`** — research-only.

## Release gates (must all pass)

1. CPU build clean (`cmake -S OXN/nsos -B OXN/nsos/build-mvp -DNSOS_ENABLE_CUDA=OFF -DNSOS_BUILD_PYTHON=ON -DNSOS_BUILD_TESTS=ON -DNSOS_BUILD_CLI=ON -DNSOS_BUILD_API=ON -DNSOS_BUILD_OXTAMEM=OFF && cmake --build OXN/nsos/build-mvp --config Release`). OxtaMem is validated in its independent Rust lane.
2. Every registered CPU-build CTest green (`ctest --test-dir OXN/nsos/build-mvp -C Release --output-on-failure`), including `test_model_pack`, `test_layer_audit`, `test_thread_safety`, `test_http_api` and `test_circuit_api_pipeline`. GPU executables are not registered in this lane.
3. Determinism + industrial Python (`python OXN/scripts/gatekeeper.py` with `NSOS_BUILD_DIR` set).
4. Benchmark gate (`python OXN/nsos/scripts/benchmark_gate.py --device cpu --profile mamba_small …`); thresholds: ≥ 1.0 prompt tok/s, ≥ 0.1 decode tok/s.
5. Fuzz smoke (`python OXN/nsos/scripts/fuzz_surface_smoke.py …`).
6. OxtaMem integrated lane: `cargo test`, `cargo clippy --all-targets -- -D warnings`, `python -m compileall` over `modules/oxtamem/python` and `nn/`. The duplicate legacy SDK under `oxta_engine/python` was removed; the packaged safe serializer is the only supported Python client.
7. Docker build (`docker build -t nsos-mvp .`).
8. Project-boundary check and notebook JSON/Python static validation green.
9. Real-GPU lane green before promoting either GPU row from **Experimental**. The lane requires fail-closed CTest plus `compute-sanitizer` memcheck over device-memory, DP4A lifecycle, incremental decode, checkpoint v8 and mixed-precision contracts.

## Operational rule

Use fast probes to choose direction. Do not spend long runs unless masked-answer loss, first-token accuracy, and teacher-token accuracy are moving on the probe.

The fast probe loop is the **decision layer**. The long `pilot` and `small` runs are **confirmation layers**.

## Test gates (CTest entries that must pass on `main`)

The authoritative target inventory lives in `OXN/nsos/CMakeLists.txt`; the exact registered count is configuration-dependent because CUDA and optional integrations have separate lanes. Categories:

- **Core math / layers**: `test_sanity`, `test_bitlinear`, `test_bitnet_integrated`, `test_mamba2`, `test_mamba2_reference`, `test_kan`, `test_ttt_layer_kernel`, `test_holographic_full`, `test_jamba`, `test_matmul`, `test_moe_router`, `test_moe_training`, `test_gradcheck`, `test_training_invariants`.
- **Tokenizer / inference / packs**: `test_tokenizer`, `test_inference_engine`, `test_model_pack`, `test_smart_loader_async`.
- **HTTP / Python binding / safety**: `test_http_api`, `test_python_binding_smoke`, `test_vulnerability_fixes`, `test_auxiliary_safety`, `test_thread_safety`.
- **Reasoning / memory / audit**: `test_mcts_reasoning_v3`, `test_memory_causal_store`, `test_layer_audit`.
- **Integration**: `test_circuit_api_pipeline`, `test_ultra_integration`, `test_swarm_orchestration`, `train_e2e`.
- **Previously orphaned, now wired**: `test_components`, `test_core_v2`, `test_v2_features`, `test_suite`, `test_stability`, `test_tensor_unit`.
- **GPU (CUDA builds only, fail-closed)**: decomposed `test_gpu_parity_*`, DP4A pack lifecycle, native-device-memory, checkpoint-v8 continuation and mixed-precision capability/execution gates.

## What this product does **not** do

- Run TLS in-process. HTTPS is reverse-proxy responsibility (see `DEPLOY.md`).
- Promise GPU as a production runtime path. Native CUDA work exists, but promotion still depends on real-hardware stability, parity, accuracy and performance evidence.
- Promise distributed / multi-node. MPI is experimental.
- Promise formal verification. The current restricted checker is a research guard, not a proof.
- Carry research code on the supported surface. See `INCUBATION.md`.

## Where to look first

- Build / install / run: `OXN/nsos/README.md` and `OXN/nsos/docs/RELEASE.md`.
- HTTP API contract: `OXN/nsos/docs/API.md`.
- Model pack format: `OXN/nsos/docs/MODEL_PACKS.md`.
- Validation status (per profile, per metric): `OXN/nsos/docs/NSOS_VALIDATION_STATUS.md`.
- Architecture risk and refactor plan: `OXN/nsos/docs/ARCHITECTURE_RISK.md`.
- Deployment guidance (TLS, healthchecks, reverse proxy): `OXN/nsos/docs/DEPLOY.md`.
- Datasets and curriculum loaders: `OXN/nsos/docs/DATASETS.md` (authored in Phase 3).
- External scorecard: `OXN/nsos/docs/SCORECARD.md` (authored in Phase 6).
- Incubation status and promotion gates: `OXN/nsos/docs/INCUBATION.md`.
