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
| Model packs (load / save / round-trip) | **Supported (MVP)** | SHA-256 + FNV1a checksums, atomic `tmp + rename`, per-file size limits, path canonicalization. |
| Tokenizer packs | **Supported (MVP)** | Bounded line/token/id counts. |
| `modules/oxtamem` companion (Rust + Python) | **Supported as integrated module** | Loopback default, optional auth token, frame/connection/timeout limits. Not a public service. NSOS uses it via dynamic-loaded FFI (`OXN/nsos/src/oxtamem_ffi.cpp`). |
| GPU (CUDA) float | **Experimental** | `gpu-hotpath` CI lane is opt-in via repo var `NSOS_GPU_CI`. `test_gpu_parity` is being decomposed (Phase 1.B); until then GPU is not a release gate. |
| GPU packed runtime | **Not yet supported** | No native packed CUDA kernel path. Edge stays CPU-first by design until this changes. |
| TTT (Test-Time Training) | **Research only** | Snapshot/replay and serving parity not formally closed. |
| Distributed / MPI | **Experimental** | `include/mpi_mock.h` is an explicit mock; do not present as production. |
| Lean / formal verification | **Research only** | `include/lean_integration.h` is a stub. |
| Anything in `legacy/` | **Not supported** | Frozen, append-only, not on any execution path. |
| Anything outside `OXN/nsos` and `modules/oxtamem` | **Not supported** | See `INCUBATION.md`. |

## Validated profiles

- **`mamba_small`** — production small-profile baseline.
- **`small`** — legacy alias for `mamba_small`.
- **`hybrid_pilot`** — validates attention + Mamba (no TTT).
- **`hybrid_small`** — validates attention + sparse MoE + Mamba.
- **`TTT`** — research-only.

## Release gates (must all pass)

1. CPU build clean (`cmake -S OXN/nsos -B OXN/nsos/build-mvp -DNSOS_ENABLE_CUDA=OFF -DNSOS_BUILD_PYTHON=ON -DNSOS_BUILD_TESTS=ON -DNSOS_BUILD_CLI=ON -DNSOS_BUILD_API=ON -DNSOS_BUILD_OXTAMEM=ON && cmake --build OXN/nsos/build-mvp --config Release`).
2. CTest green (`ctest --test-dir OXN/nsos/build-mvp -C Release --output-on-failure`), including `test_layer_audit` and `test_pack_determinism`.
3. Determinism + industrial Python (`python OXN/scripts/gatekeeper.py` with `NSOS_BUILD_DIR` set).
4. Benchmark gate (`python OXN/nsos/scripts/benchmark_gate.py --device cpu --profile mamba_small …`); thresholds: ≥ 1.0 prompt tok/s, ≥ 0.1 decode tok/s.
5. Fuzz smoke (`python OXN/nsos/scripts/fuzz_surface_smoke.py …`).
6. OxtaMem integrated lane: `cargo test`, `cargo clippy --all-targets -- -D warnings`, `python -m compileall` over `modules/oxtamem/python`, `oxta_engine/python`, `nn/`.
7. Docker build (`docker build -t nsos-mvp .`).
8. `git status --short` clean except ignored build artifacts.

## Operational rule

Use fast probes to choose direction. Do not spend long runs unless masked-answer loss, first-token accuracy, and teacher-token accuracy are moving on the probe.

The fast probe loop is the **decision layer**. The long `pilot` and `small` runs are **confirmation layers**.

## Test gates (CTest entries that must pass on `main`)

The authoritative list lives in `OXN/nsos/CMakeLists.txt`. Categories (current MVP set):

- **Core math / layers**: `test_sanity`, `test_bitlinear`, `test_bitnet_integrated`, `test_mamba2`, `test_kan`, `test_ttt_layer_kernel`, `test_holographic_full`, `test_jamba`, `test_matmul` (target wired in Phase 1.A), `test_moe_router` / `test_moe_training` (Phase 1.A).
- **Tokenizer / inference / packs**: `test_tokenizer`, `test_inference_engine`, `test_model_pack`, `test_pack_determinism` (added in Phase 0 PR-0.5).
- **HTTP / Python binding**: `test_http_api`, `test_python_binding_smoke.py`, `test_oxtamem_ffi`, `test_vulnerability_fixes` (Phase 1.A), `test_thread_safety` (Phase 1.A).
- **Reasoning / memory / audit**: `test_mcts_reasoning_v3`, `test_memory_causal_store`, `test_layer_audit`.
- **Integration**: `test_circuit_api_pipeline`, `test_ultra_integration`, `test_swarm_orchestration`, `train_e2e`.
- **GPU (opt-in)**: decomposed `test_gpu_parity_*` (per-kernel) gated by `NSOS_GPU_CI`.

Phase 1 will move every orphan test (`test_components`, `test_core_v2`, `test_v2_features`, `test_suite`, `test_stability`) to either CTest wired or `legacy/tests/cpp/`.

## What this product does **not** do

- Run TLS in-process. HTTPS is reverse-proxy responsibility (see `DEPLOY.md`).
- Promise GPU as a production runtime path. CPU-first edge until packed CUDA kernel and `test_gpu_parity_*` are stable.
- Promise distributed / multi-node. MPI is experimental.
- Promise formal verification. Lean integration is a stub.
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
