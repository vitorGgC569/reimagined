# NSOS Test Boundary

The official product gate is the test set registered by `OXN/nsos/CMakeLists.txt`.
Run it once with `NSOS_GPU_BACKEND=NONE` and once for every GPU backend/hardware
combination being released. `NSOS_BUILD_OXTAMEM=OFF` keeps the independent Rust
lane out of this gate.

## Official Product CTest

Do not copy a test count or hand-maintained flat list into documentation.
Configuration writes the authoritative inventory to
`<build-dir>/nsos_test_inventory.txt`; `ctest --test-dir <build-dir> -N` is the
runtime view of the same configured graph. The inventory includes:

- resolved CPU/HIP/CUDA backend and optional Python/OxtaMem surfaces;
- every registered CTest name, sorted deterministically;
- the exact count for that configuration;
- every explicitly quarantined historical `test_*.cpp`.

CMake configuration fails if a new `test_*.cpp` has neither a build target nor
an explicit quarantine, or if a `test_*.py` is absent from the registration
graph. This prevents documentation and source-tree drift from silently reducing
coverage.

`test_oxtamem_ffi` belongs to the integrated OxtaMem lane and is registered only
when the Rust engine target is built by CMake.

`test_layer_audit` is the end-to-end audit smoke. `test_training_invariants`
contains checkpoint transaction, injected OOM/NaN/interruption, corruption,
truncation, exact continuation, optimizer fail-stop/recovery, negative
second-moment preflight and cancellation gates. `test_model_pack` additionally
proves immutable generations, mandatory SHA-256, transactional QAT readiness,
deep administrative clones and preservation of the previous manifest after a
failed export. `test_thread_safety` is a registered shared-model, memory,
numerical guard and tensor concurrency stress gate.

An ambiguous optimizer exception is deliberately not retriable in place:
Trainer becomes poisoned, training/model-pack/checkpoint/sidecar export fails
closed, and recovery must load the exact model checkpoint followed by its
matching Trainer sidecar. HTTP maps this condition to
`503 optimizer_state_requires_recovery`. Cooperative cancellation is different:
before commit it leaves the optimizer usable, remains observable until
explicitly cleared, and a fresh server `start()` clears a stale lifecycle
cancellation request.

Transactional HTTP training clones copy weights, optimizer moments,
precision-mode state, schedulers, loss scaling, RNG, controller state, metrics
and in-memory auxiliary clusters under one source transaction. Durable
OxtaMem-backed auxiliary stores are intentionally rejected on that clone path:
sharing an external mutable arena before publication would violate isolation.
Use the single-owner local training path for durable OxtaMem until a
generation-qualified arena transaction is implemented.

The 66/66 RX 7600 and 39/39 CPU results recorded on 2026-07-28 are historical
evidence for the then-current tree, not the count of the current gate. New code
must use the generated inventory and produce fresh CTest evidence. NVIDIA CUDA
must run the same fail-closed gate on NVIDIA hardware before a CUDA binary is
promoted; source/CMake compatibility alone is not runtime validation.

## Explicit quarantine and diagnostics

Only these historical `test_*.cpp` sources are quarantined:

- `test_suite.cpp`
- `test_v2_features.cpp`
- `test_vulnerability_fixes.cpp`
- `unit/test_tensor.cpp`

They are duplicated, stale or incomplete prototypes and are not release
evidence. `test_components`, `test_core_v2`, `test_matmul`, `test_moe_router`,
`test_moe_training`, `test_stability` and `test_thread_safety` are registered
product gates.

Files whose names do not start with `test_` remain explicit diagnostics or
benchmarks rather than CTest entries:

- `sanity_check.cpp`
- `verify_roadmap.cpp`
- `verify_sre_engineering.cpp`
- `repro_ttt.cpp`
- `bench_*.cpp`

Promotion still requires bounded deterministic behavior, fail-closed
assertions in Release, unique temporary paths and guarded optional
dependencies.
