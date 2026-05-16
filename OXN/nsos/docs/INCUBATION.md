# Incubation Inventory

> **Purpose:** Make the answer for every non-product directory in this monorepo explicit. If a folder isn't `OXN/nsos`, `modules/oxtamem`, `legacy/`, or release support, it must appear here with a concrete decision.
>
> **Status today:** **skeleton (Phase 0 PR-0.3)**. Decision and gate fields are placeholders; Phase 7 fills them in.
>
> **Rule:** No incubation tree may be referenced from `PRODUCT.md` or built into a release artifact. Failures here do not block product gates unless promotion is in progress.

---

## How a tree leaves incubation

A module is product-eligible only after **all** of the following are true (also encoded in `docs/PROJECT_BOUNDARY.md`):

1. Clear product reason and named owner.
2. Deterministic build instructions from a clean checkout.
3. Unit and integration tests wired into CI.
4. Runtime / service paths have authentication, resource limits, and failure-mode tests where applicable.
5. Benchmark or performance claims are reproducible from committed scripts or documented input artifacts.
6. Documentation updated in `README.md`, `PROJECT_SCOPE_STATUS.md`, `docs/PROJECT_BOUNDARY.md`, and `PROJECT_BOUNDARY.json`.
7. The CI boundary check (`scripts/check_project_boundary.py`) passes.
8. **For the MVP:** at least one `pilot`-rung scorecard run shows non-regression on the four MVP metrics defined in `SCORECARD.md`.

If a tree cannot meet criterion 8 within the next planned release, the right answer is `not now` or `archive`, not "promote anyway."

---

## Inventory

### `KernelOpen/`

- **What it is** — Universal Heterogeneous Kernel (UHK) v1.0: an HPC fabric concept fusing CPU, GPU, and exotic devices (BCI / quantum / photonic). Includes BrainFlow / OpenBCI and IBM Qiskit integration stubs. Performance claims (4200 TOPS, 1.1 µs latency) are **not validated in CI**.
- **Current state** — Source tree present (`src/host/`, `src/device/`, `src/ghost/`, `src/space/`, `src/bci/`, `src/quantum/`, `sdk/`, `frontend/`). No CTest target, no compileall, no scorecard run. Builds independently if at all.
- **Promotion gate** *(to be filled in Phase 7)* — at minimum: reproducible build instructions, non-CUDA fallback test, `gpu-hotpath` lane stable for ≥ 4 consecutive nightly runs with KernelOpen integration enabled, one `pilot`-rung scorecard run showing non-regression.
- **Decision today** — *placeholder; Phase 7 sets to `not now`, `promote in vN.N`, or `archive`.*
- **Next revisit** — *placeholder.*

### `CHRASS/`

- **What it is** — Kimera V19 SSSP solver (Single-Source Shortest Path) with hardware-aware spectral graph algorithms (AVX2). Standalone benchmarks at extreme scale (Shannon 10^120, RSA-2048/4096). Unrelated path from the runtime CHRASS layer that lives in `OXN/nsos/src/chrass_layer.cpp`.
- **Current state** — Self-contained. Not built by `OXN/nsos/CMakeLists.txt`. No integration with the supported runtime.
- **Promotion gate** — *to be filled in Phase 7.* Provisional: deterministic CPU forward-pass test under `OXN/nsos/tests/`, peer-reviewed numeric stability ≤ 1e-6 vs reference, ≥ 1 `pilot`-rung scorecard run with CHRASS layers in the model showing non-regression on the four MVP metrics.
- **Decision today** — *placeholder.*
- **Next revisit** — *placeholder.*

### `CART/`

- **What it is** — PEFT framework (Parameter-Efficient Fine-Tuning): TurboFusion (DoRA + IA3), LoRA, IA3, dynamic rank adaptation. Hybrid C++ / PyTorch with an "industrial mode" PyTorch extension claiming 113× speedup.
- **Current state** — Independent source tree (`peft_framework/pytorch_extension/`, C++ reference impl, benchmarks). No `nsos.cart` Python entry point. Not used by `OXN/nsos/scripts/train_curriculum.py`.
- **Promotion gate** — *to be filled in Phase 7.* Provisional: Python entry point under `nsos.cart`, does not replace any production path, ships behind off-by-default feature flag.
- **Decision today** — *placeholder.*
- **Next revisit** — *placeholder.*

### `OXB/`

- **What it is** — OxtaCore LLM data ingestion: bit-packing kernels (1.13B ops/s claim), RMI learned indexing (533M ops/s claim), OXH protocol. Python V3.1 alpha (delta encoding, block-based, ~7 samples/s throughput); V2.1 stable prototype.
- **Current state** — Independent. Not consumed by `dataloader_v2.cpp` or `train_curriculum.py`. Performance claims not reproducible from committed CI.
- **Promotion gate** — *to be filled in Phase 7.* Provisional: documented threat model, fuzz harness in CI ≥ 1 week, explicit owner, reproducible benchmark numbers.
- **Decision today** — *placeholder.*
- **Next revisit** — *placeholder.*

### `hardware/`

- **What it is** — Verilog RTL designs for BitNet (`bitnet_core.v`, `bitnet_tb.v`).
- **Current state** — RTL only. No software toolchain integration. Useful as a hardware reference, not as a software product input.
- **Promotion gate** — *to be filled in Phase 7.* Likely `not now` indefinitely — RTL is not the right shape for the software product. Could become a separate hardware repo in the future.
- **Decision today** — *placeholder.*
- **Next revisit** — *placeholder.*

### `bindings/` (root, not `OXN/nsos/src/bindings.cpp`)

- **What it is** — Legacy `python_bindings.cpp` C++ wrapper for Python interop, predating the supported `nsos_ext` binding.
- **Current state** — Likely dead code. The supported Python binding lives at `OXN/nsos/src/bindings.cpp`.
- **Promotion gate** — N/A. Should be archived to `legacy/src/` as part of Phase 7 cleanup unless evidence emerges that something still needs it.
- **Decision today** — *placeholder; expected `archive`.*
- **Next revisit** — Phase 7.

### `include/pantheon/` and `src/pantheon/` (root)

- **What it is** — Pantheon framework: an abstract theory layer covering cognition (chain of thought, theory of mind, symbolic), physics (neural ODE, gradient matching, info bottleneck, topology), frontier (causal, meta, physical, quantum, privacy, spiking), structure (attention, relational, contrastive, category theory, msdcrd), and social/response (swarm, game theory, logit distillation). Pantheon engine wrapper headers exist.
- **Current state** — Headers and some implementations. Not built by `OXN/nsos/CMakeLists.txt`. Companion paper `artigopantheon.md` lives at repo root (~106 KB). Not on any production path.
- **Promotion gate** — *to be filled in Phase 7.* Provisional: reduced to a single coherent module + paper, with at least one component used by a `pilot`-rung scorecard model and showing non-regression. **Or** archived to `legacy/` permanently.
- **Decision today** — *placeholder.*
- **Next revisit** — Phase 7.

### `tools/`, root `scripts/`, root `benchmarks/`, `circuit_data/`, `wikitext_data/`

- **What it is** — Utility scripts (`brain_monitor.py`, `generate_reasoning_data.py`, `generate_scientific_curriculum.py`, `pack_dataset.cpp`, `validate_graph.py`), build helpers (per-OS), validators (`check_project_boundary.py`, `check_pulse.py`), converters (`convert_nsos_to_gguf.py`), data subdirectories.
- **Current state** — Mixed. `scripts/check_project_boundary.py` is part of release support and stays. Most others are not on the product path.
- **Promotion gate** — Per-script triage. Items used by CI stay; items unused are candidates for archival or to be moved into `OXN/nsos/scripts/` if useful.
- **Decision today** — *placeholder; Phase 7 produces a per-script table.*
- **Next revisit** — Phase 7.

### `artigopantheon.md`

- **What it is** — Long-form research paper (~106 KB) on the Pantheon framework.
- **Current state** — Lives at repo root. Linked only from this file.
- **Decision today** — Keep in place, **not referenced from `PRODUCT.md` or root `README.md`**. This file (`INCUBATION.md`) is the only entry point.
- **Next revisit** — When Pantheon is decided in Phase 7.

---

## Cross-cutting rules

- **No back-references from product** — nothing under `OXN/nsos/` or `modules/oxtamem/` may include or import from any path listed here. The CI boundary check (`scripts/check_project_boundary.py`) enforces this.
- **No silent promotion** — promotion lands as an explicit PR that updates `PROJECT_BOUNDARY.json`, `PRODUCT.md`, and this file in the same change.
- **Archival is reversible** — moving to `legacy/` does not delete history; recovery via `git log` or copy out of `legacy/` (and back into the active tree, with rules followed).
