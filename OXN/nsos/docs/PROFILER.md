# NSOS Inference Profiler — Architectural Research Tool

A self-contained, OPT-IN profiling module designed for **architectural
research sessions**. Never linked into production training or
production inference builds. Captures per-layer cycle counts via
RDTSC and probes the host's cache hierarchy so the architectural
analysis report can suggest concrete data-structure changes informed
by measurement, not theory.

## Why it exists

NSOS is a hybrid Mamba2 + Attention + MoE + BitNet 1.58 model. No
production system has this exact combination, so no off-the-shelf
profiler (NVIDIA Nsight, Intel VTune, etc.) tells us where its
specific bottlenecks are. We need a profiler that knows about layer
hierarchy, MoE routing, and the interaction between BitNet weight
packing and the host's cache hierarchy.

The end goal is to **design a novel data structure** for the
inference path — informed by what we actually measure, not what we
think is slow.

## Core architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  nsos_core (production)                                           │
│  - JambaModel has `void* profiler_ = nullptr` member              │
│  - forward() does ONE null check; when null, hot path runs as     │
│    before.  When non-null, calls function pointers                │
│    g_nsos_profiler_begin_layer / g_nsos_profiler_end_layer        │
│    that default to nullptr in production.                          │
└──────────────────────────────────────────────────────────────────┘
                            ↑
                            │ (linked only if NSOS_BUILD_PROFILER=ON)
                            │
┌──────────────────────────────────────────────────────────────────┐
│  nsos_profiler (static lib, opt-in)                               │
│  - InferenceProfiler with lock-free SPSC ring buffer              │
│  - RDTSC cycle counter (x86_64 + aarch64)                         │
│  - Cache hierarchy probe (pointer-chase)                          │
│  - install_hooks() flips the function-pointer globals to real     │
│    callbacks; uninstall_hooks() restores nullptr                  │
└──────────────────────────────────────────────────────────────────┘
                            ↑
                            │ (nsos_profiler_ext.pyd / .so)
                            │
┌──────────────────────────────────────────────────────────────────┐
│  Python tooling                                                    │
│  - heatmap_profiler.py:   orchestrate the run, drain to JSON      │
│  - generate_architecture_report.py:  roofline + recommendations   │
└──────────────────────────────────────────────────────────────────┘
```

## Zero-overhead guarantee

When the profiler library is NOT linked (production build):
- `g_nsos_profiler_begin_layer` and `g_nsos_profiler_end_layer` are
  nullptr (defined in jamba.cpp, never overwritten by anyone).
- `JambaModel::profiler_` is nullptr by default.
- The forward path:
  ```cpp
  if (profiler_) {
      profiler_begin_layer = g_nsos_profiler_begin_layer;
      ...
  }
  ```
  evaluates to a single load + branch that the predictor learns
  immediately. Modern CPUs execute the false branch in less than
  one cycle effective cost (the predictor speculates past it).

The profiler library itself is never compiled unless
`-DNSOS_BUILD_PROFILER=ON` is passed to cmake. Default = OFF.

## Build

```powershell
# Production training build (default — profiler NOT built)
cmake -S OXN/nsos -B OXN/nsos/build -DNSOS_ENABLE_CUDA=ON
cmake --build OXN/nsos/build --config Release

# Architectural research build (profiler built)
cmake -S OXN/nsos -B OXN/nsos/build-profiler -DNSOS_ENABLE_CUDA=ON `
      -DNSOS_BUILD_PROFILER=ON
cmake --build OXN/nsos/build-profiler --config Release
```

The research build produces:
- `nsos_ext.pyd` — main extension (same as production)
- `nsos_profiler_ext.pyd` — profiler bindings (additional)

## Usage

```bash
# 1. Run profiling
python OXN/nsos/scripts/heatmap_profiler.py \
    --model OXN/nsos/scripts/live_distill_v10_gpu/phase6_memory.bin \
    --tokenizer OXN/nsos/scripts/live_distill_v10_gpu/tokenizer.nsos \
    --model-config OXN/nsos/scripts/live_distill_v10_gpu/effective_model_config.json \
    --device cpu \
    --max-new-tokens 32 \
    --measured-passes 5 \
    --build-dir OXN/nsos/build-profiler \
    --out artifacts/heatmap_report.json

# 2. Generate the architectural analysis
python OXN/nsos/scripts/generate_architecture_report.py \
    --heatmap artifacts/heatmap_report.json \
    --host t4 \
    --out artifacts/architecture_insights.md
```

## What the report contains

1. **Roofline analysis** — per-layer arithmetic intensity (FLOPS/byte)
   plotted against the host's measured peak FLOPS and bandwidth.
   Identifies which layers are memory-bound vs compute-bound.

2. **Cache residency analysis** — host's L1/L2/L3 sizes measured
   empirically via pointer-chase. Compares to model weight size
   (in 1.58-bit packed form) to predict which layers will spill
   to DRAM.

3. **Hot spots + jitter** — top-5 layers by total cycles and top-5
   by jitter ratio. High-jitter layers indicate cache eviction or
   variable MoE routing.

4. **Per-op breakdown** — once op-level scopes are added inside
   layer forward implementations (via `InferenceProfiler::Scope`),
   this section shows where time goes within each layer type.

5. **Architectural recommendations** — concrete data-structure
   changes derived from the measured data. Each recommendation
   cites the specific numbers from this run.

## Extending: per-op scopes

The current profiler hooks fire at LAYER granularity. To get
sub-layer breakdown (e.g., `attn.qkv` vs `attn.softmax` vs
`attn.output`), add `InferenceProfiler::Scope` brackets inside
the relevant forward methods:

```cpp
// In Attention::forward
{
    profiler::InferenceProfiler::Scope qkv_scope(
        static_cast<profiler::InferenceProfiler*>(profiler_),
        profiler::EventKind::OP, "attn.qkv", layer_idx);
    // ... compute Q, K, V ...
    qkv_scope.annotate(bytes_in, bytes_out, mflops);
}
```

These scopes also compile away to null when no profiler is attached
(the Scope constructor's null check matches the layer-level one).

## Methodology notes

- **RDTSC accuracy**: on Intel/AMD with invariant TSC (all CPUs
  since 2010), `__rdtsc()` reads the reference clock frequency,
  which is constant regardless of P-state. Cross-checked with
  `__rdtscp()` + `_mm_lfence()` for fine-grained measurements
  via `read_cycles_serialized()`.

- **Calibration**: cycle→ns ratio measured at profiler construction
  via a 50ms sleep + cycle delta. Cached for the run. Re-callable
  via `recalibrate()` if the user pinned the CPU between calls.

- **Cache probe**: pointer-chase with random permutation to defeat
  the prefetcher. Working set sizes from 4KB to 128MB; inflection
  points identify L1/L2/L3 boundaries. Reported values are measured
  on the host running the probe, not CPUID-derived (which can lie
  on virtualized hardware like Colab).

- **Event storage**: SPSC ring buffer, 64 bytes per event (one
  cache line), default capacity 65536 events = 4 MB. Lock-free,
  no allocations during measurement. If the buffer wraps, oldest
  events are overwritten and the drop count is recorded.

## Roadmap

- [ ] Add op-level scopes inside Attention, Mamba2SSD, MoERouter
  forward methods (currently only layer-level granularity).
- [ ] Add CUDA event scopes for GPU kernel timing (currently
  cycles cover the CPU-side launch + sync; doesn't measure
  in-kernel time).
- [ ] Per-token profiling mode (one report per decoded token) to
  observe how cycle counts grow with sequence length.
- [ ] Heat-map visualizer (matplotlib SVG output).
