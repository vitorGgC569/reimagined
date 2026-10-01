# GPU modernization — implementation ledger

User constraints: preserve the current generative product; Jev is out of scope;
implement in the order below; execute validation only after implementation.
Existing working-tree changes are part of the starting point, not disposable.

The pre-change product sources are preserved locally in
`artifacts/gpu-modernization-20260919/baseline/`. This is a source snapshot,
not a measured performance baseline. Both versions must be measured at the end.

## Ordered work

1. Remove confirmed waste: decode GEMV geometry, inactive experts, redundant
   initialization and temporary buffers.
2. Stabilize execution: explicit device/stream ownership, typed workspaces,
   layouts and isolated session state.
3. Reduce overhead: fused epilogues, grouped projections and graph execution.
4. Scale context/concurrency: bounded-memory attention, compact KV and batching.
5. Research: isolated opt-in experiments; never claim measured speedup or
   equivalent quality before the final evaluation.
6. Final validation/measurement: compile, correctness, lifecycle, failure
   contracts, CPU/HIP tests, reproducible before/after measurements and identity.

## Status

- Source snapshot preserved. Validation began only after the implementation
  phase; compiler/test failures are being corrected in that final phase.
- Final Release suites: CPU 54/54 passed; HIP 103/103 passed on RX 7600/gfx1102,
  including the compact-KV overflow guard. Logs: `ctest-cpu-final-3.log` and
  `ctest-hip-final-3.log` in the evidence folder.
- This ledger is not a claim that the entire research backlog is complete.

### Implemented runtime paths

- Packed decode GEMV with affine epilogue and a gfx11 HIP signed-dot route;
  scalar dot remains selectable with `NSOS_HIP_SCALAR_DP4A` at build time.
- Device-selected MoE compute for 1–32 rows, ordered merge without atomics,
  reusable typed scratch and cached typed expert descriptors.
- Explicit per-model execution streams, entry/exit dependency fences,
  stream-aware BLAS and kernel wrappers, frozen workspaces during capture.
- Grouped decode projections for attention and faithful Mamba; fused gated
  RMS epilogue; batched convolution/state-update launches.
- Opt-in HIP/CUDA graph path with device position, pinned staging ownership,
  capture ownership, cache-capacity guard and launch failure handling.
- Bounded-LDS online attention with split-context merge and batch dispatch;
  optional real FP16 KV storage, copy-on-write snapshots, growth and batch restore.
- Isolated `NSOS_GPU_EXPERIMENT=none|persistent_gemv|ternary_lut` experiments.
  Persistence is a bounded grid-stride GEMV, not a full-model monokernel.
- Speculation script uses real distributions and full-prefix reference
  verification; no invented draft model, acceptance rate or speedup.
- Host dispatch counters, comprehensive GPU regression source, and
  `scripts/measure_decode_repro.py` recording artifact/binary identities,
  options, GPU identity, build configuration, outputs and timing samples.

### Evidence and operational limits

- The native GPU integration test exercised a real active HIP graph, packed
  GEMV (including both research switches), poisoned inactive experts, 16,385
  attention positions, FP16 cache conversion/growth, copy-on-write sessions,
  batching and fused Mamba epilogues. Active-graph snapshots are detached;
  model policy changes, weight loads and KV reservation invalidate captures.
- Disassembly of the compiled gfx1102 object contains `v_dot4_i32_iu8` in
  `bitnet_gemv_scaled_kernel`; see `native-dot4-isa.log` in the evidence folder.
- CPU-only build errors from unguarded GPU helpers/includes and unchecked
  Mamba reshape narrowing were corrected. Existing EOS, TLS-health policy,
  launch-stream source checks and dataset-manifest test fixtures were aligned
  with the current production contracts; those contracts were not relaxed.
- One model is a serialized execution lane. Entry/exit fences conservatively
  bridge callers; this is not a claim of independent concurrent GPU execution.
  The tensor allocator permits only one live graph capture owner per device;
  a competing capture fails closed to eager execution. Multi-graph concurrent
  serving and per-session independent graph pools remain future work.
- Graph output aliases reusable captured storage. Clone it before retaining
  it across replay. Direct public parameter/storage mutation requires graph
  invalidation/reset by the caller; supported model mutation APIs invalidate.
- Default KV remains FP32; FP16 storage and graphs are opt-in. The experiments
  are not promoted automatically by passing a numerical fixture.
- The configured Windows TheRock tool directories do not include rocprofv3.
  No cache-hit, bandwidth or dispatch-latency profiler results are claimed.

### Reproduction

Builds: `build-gm-cpu` (NONE) and `build-gm-hip` (HIP/gfx1102), Release,
Clang from `C:/TheRock/build/lib/llvm/bin`, Python bindings/tests enabled,
native CPU tuning disabled. Use the MSVC amd64 developer environment and
`C:/TheRock/build/bin` plus `lib/llvm/bin` on PATH. Run CTest with
`--output-on-failure`; serialize GPU tests (`-j 1`).

`build-gm-base` builds the preserved source snapshot with the same compiler
and HIP target. It needs `-Wno-c++11-narrowing` for its pre-existing Mamba
reshape conversions. This diagnostic flag difference is disclosed, not hidden;
the preserved baseline source was not patched. OxtaMem uses the same external
Rust module in all builds.

`scripts/create_decode_benchmark_fixture.py --build-dir build-gm-base
--output-dir <new-directory>` creates explicitly UNTRAINED weights, a byte
tokenizer, two prompts and configuration. The fixture has 7,868,972 parameters;
it is not the proposed trained 40M/150M model or a language-quality benchmark.
`scripts/measure_decode_repro.py` accepts those artifacts or real supplied
artifacts. Run each binary/policy in a fresh process after other GPU tests end.
Use `--baseline <report>` for strict identity/geometry/precision/hardware checks;
`--vary-policy <one-policy-name>` permits one explicitly named policy difference.
Every report records all requested policies, binary/artifact SHA-256, actual
device and matmul precision, warmup count, samples and active host path counters.

The manual target `bench_decode_gemv` isolates packed W2/A8 decode. Example:
`NSOS_GPU_EXPERIMENT=ternary_lut bench_decode_gemv 1024 4096` (set the
environment variable using the syntax of your shell). Each fresh process checks
the output against a CPU integer oracle, warms 20 calls and measures 7 samples
of 100 calls. Its JSON records binary/input/weight hashes, samples and geometry.
This benchmark is not a timed CTest assertion and does not measure a model.

## Final measurement results

Evidence folder: `artifacts/gpu-modernization-20260919/`. Measurements completed
across the September 19–20 local session. See `GPU_MODERNIZATION_RESULTS.md`
for the complete result interpretation and remaining gates.

- Synthetic 7.87M fixture, first paired run: baseline 433.14, current 704.08
  SDK decode tokens/s (1.63x); batch=2: 496.56 to 755.81 aggregate tokens/s
  (1.52x). Output hashes matched in every recorded comparison.
- Graph replay was active but did not improve this short 32-token workload
  (696.52 tokens/s). FP16 KV also had no material speed advantage here
  (707.16); its physical payload reduction is a separate benefit.
- Process-to-process medians varied materially (baseline 326–433, current
  680–766 tokens/s). These desktop measurements are not clock-locked or
  profiler-controlled. Do not generalize a precise speedup to other models.
- The two isolated research kernels passed exact synthetic integer parity but
  were slower than the normal GEMV on both measured geometries. Keep
  `NSOS_GPU_EXPERIMENT=none`; neither candidate is promoted.

### Deliberately not promoted or represented as complete research

Full-model persistent scheduling, direct AQL/HSA dispatch, learned routing
skips, 2-bit KV eviction/GDN, trained CfC drafts and test-time training are
separate research designs. This implementation does not silently replace
the model with them or claim their speculative gains. They require trained
artifacts, quality/rollback criteria and independent measurements. Jev
remains outside this implementation. The original research backlog is not
deleted by delivering the safer runtime foundations.

### Measurement contract

Use one fresh process per binary/policy. Supply the same model, authoritative
configuration, tokenizer, prompts, verified EOS, precision and generation
options. The new measurement script rejects mismatched baseline identities.
Use multiple warmed repetitions and report both host wall time and internal
prefill/decode/sampling metrics. Host counters identify dispatch attempts;
only a profiler can establish actual dispatch latency, bandwidth and cache
hit rates. Graph capture records are not replayed host dispatches.

FP16 KV halves live KV payload relative to FP32; allocator-reserved bytes
may remain cached after conversion and must be reported separately. It does
not justify a perplexity/accuracy or 4–8x context claim. Long-context parity
fixtures and randomly initialized models are technical checks, not trained
quality evidence. Trained checkpoint/tokenizer/corpus selection is pending.

## Promotion rules

Exact execution optimizations retain reference paths. Approximate numeric,
cache-eviction, routing and speculative methods require explicit opt-in and
their own quality/rollback gates. No invented dataset, external training run,
performance result or unsupported architecture claim is acceptable.
