# NSOS GPU optimization status

Status: implementation snapshot under revalidation

Updated: 2026-07-29

Scope: `OXN/nsos`

This document replaces the obsolete GTX 1050 Ti/CPU-only roadmap that used to
live at this path. It must not be used as benchmark evidence. The detailed
finding-by-finding audit is
`docs/benchmarks/nsos_robustness_architecture_gpu_gap_audit_2026-07-28.md`;
the backend setup contract is `docs/AMD_GPU_BACKEND.md`.

## Current backend contract

- `NSOS_GPU_BACKEND=HIP` compiles the shared `.cu` kernel tree with AMD clang
  and HIP; `NSOS_GPU_BACKEND=CUDA` compiles the same sources with NVCC.
- The locally exercised device is Radeon RX 7600 8 GiB (`gfx1102`). CUDA is
  maintained statically but still needs fresh execution on real NVIDIA
  hardware.
- GPU tests are registered for either backend and fail closed when a selectable
  device is unavailable.
- `<build-dir>/nsos_test_inventory.txt` is the authoritative configured gate
  inventory. Historical test counts in old reports are not current claims.

## Implemented optimization surfaces

- batched masked cross-entropy and optional objectives without one D2H scalar
  read per sample;
- packed Mamba input projections and combined causal convolution;
- device-resident convolution bias and persistent embedding ID storage;
- in-place gradient accumulation, grouped gradient zeroing and grouped
  finite-value inspection;
- fused AdamW path whose clipping coefficient is consumed on device;
- adaptive checkpointing and selective reconstruction of only the SSD history
  required by backward;
- wave/warp-aggregated and sequence-local Mamba backward reductions, including
  an atomic-free short-convolution path;
- GPU-native layer/hybrid audit reductions that transfer compact statistics
  rather than full activations;
- transfer, synchronization, launch, fallback and memory-pool telemetry;
- stream-domain-aware allocation reuse and asynchronous event-query fast paths.

The pre-revalidation RX 7600 smoke artifact recorded approximately
`1.7445 steps/s` and `55.824 examples/s` for its specific 20-step fixture. That
number predates the latest audit/checkpoint/robustness changes and is not a
release threshold or a claim about other shapes.

## Remaining performance gates

1. Compile CPU and HIP lanes after the static correction phase.
2. Run focused correctness, gradcheck, checkpoint and parity gates before any
   benchmark.
3. Re-run a synchronized warmup/steady-state RX 7600 benchmark with versioned
   topology, environment, CTest inventory, parameter manifests and telemetry.
4. Prove zero unapproved host fallback and zero per-sample D2H in the declared
   GPU-resident training path.
5. Establish a regression threshold from repeated runs rather than a single
   timing.

The larger pending optimizations are per-operation BF16/FP16 with FP32 master
weights, a correct chunked/parallel SSD scan, stable-shape graph capture,
shape-specific `gfx1102` tuning, and validation on additional AMD and NVIDIA
devices. None is a current production claim.

## Interpretation

The earlier diagnosis that NSOS training was CPU-only and routed Mamba/MoE back
to the host is no longer accurate. The present risk is different: the native
GPU path has received substantial structural optimization but the newest state
has not yet passed the ordered build, runtime and benchmark gates. Until those
gates are archived, GPU support remains experimental.
