# GPU modernization: validation and measurement

## Scope and status

Jev remains deferred. The generative architecture was preserved. Runtime
implementation preceded validation, as requested; compilation and test issues
were corrected during the final validation phase. The original research list
has not been removed, but it is **not fully implemented or proven above SOTA**.

Implemented: packed M=1 GEMV, native gfx11 signed dot, inactive-expert skipping,
typed reusable scratch, per-model streams and dependencies, grouped projections,
Mamba epilogue/batch fusion, opt-in graph execution, tiled online attention,
optional physical FP16 KV, snapshots/COW/batch integration, two isolated GEMV
experiments, corrected reference speculation, and reproducible measurement tools.

The source snapshot predates these edits and includes the user's existing work.
`source-changes-final.json` identifies differences from that snapshot, not from Git
HEAD, which already had extensive unrelated changes.

## Validation

| Configuration | Result | Evidence |
|---|---:|---|
| CPU Release, Clang, Python bindings | 54/54 CTest entries passed | `ctest-cpu-final-3.log` |
| HIP Release, RX 7600, gfx1102/wave32 | 103/103 CTest entries passed | `ctest-hip-final-3.log` |
| Speculation reference | 6 unit tests, included in both suites | `test_speculative_reference.py` |
| Measurement identity/precision/policy | 4 unit tests, included in both suites | `test_decode_measurement_contract.py` |
| Native GPU instruction | `v_dot4_i32_iu8` in GEMV disassembly | `native-dot4-isa.log` |

The new native runtime regression covers packed GEMV at several widths and
activation precisions; both experimental implementations; typed-workspace reuse
and frozen-growth rejection; NaN-poisoned inactive experts; 16,385-token GQA;
FP16 cache conversion, copies and growth; fused Mamba; real HIP graph capture
and replay; graph snapshot isolation and invalidation; and restored batching.
Existing suites additionally exercised training, checkpoint continuation,
deterministic updates, pack lifecycles, memory contracts and rollback paths.

CPU build errors from unguarded GPU declarations and checked Mamba shape
conversion were fixed. Test fixtures were corrected for a valid EOS, explicit
stream launches, current dataset manifest fields, and the configured TLS/auth
contract. No production security or input-validation requirement was weakened.

Validation is specific to these builds. NVIDIA CUDA, Linux ROCm, wave64, PTDS,
multiple independent concurrent graphs and trained-model quality were not
validated by these results.

## Synthetic model comparison

This is an **untrained** 7,868,972-parameter hybrid fixture, 6 layers, width 256,
4 experts/top-2, a 257-token byte tokenizer, two fixed prompts, 32 generated
tokens, FP32 matmul and greedy sampling. Every run uses identical saved
weights/tokenizer/configuration/prompt hashes. Batch-1 runs have two warmup
repetitions followed by five repetitions for each prompt (10 measured samples).
Batch-2 runs have five measured batches. Processes ran sequentially, outside
build/test workloads. The GPU was the actual RX 7600/gfx1102, not the integrated
gfx1103 device also present in the machine.

| Run | Median SDK decode tokens/s | Comparison |
|---|---:|---|
| Preserved source, batch 1 | 433.14 | Initial baseline |
| Current runtime, batch 1 | 704.08 | 1.63x initial baseline |
| Current + graph only | 696.52 | 0.99x current eager |
| Current + FP16 KV only | 707.16 | 1.00x current eager |
| Current, grouping disabled | 566.14 | 0.80x current eager |
| Current, sparse MoE disabled | 495.63 | 0.70x current eager |
| Preserved source, batch 2 | 496.56 | Aggregate baseline |
| Current runtime, batch 2 | 755.81 | 1.52x batch-2 baseline |

All recorded output-hash comparisons matched. This is a technical fixture,
not a language-quality, perplexity or acceptance-rate result. Do not extrapolate
these throughputs to trained 40M/150M models, larger vocabularies or long context.

The host counters showed 93 sparse-MoE, 279 grouped-projection, 93 tiled-attention
and 186 fused-Mamba dispatch attempts per measured batch-1 generation. The graph
run reported 30 graph submissions; host kernel-path counters count capture once,
not replayed device kernels. FP16 runs used the compact-attention path. Although
packed inference was requested, standalone packed GEMV/GEMM counters were zero
for this raw fixture; **the model speedup is not evidence of packed GEMV speedup**.

Process-to-process variation is material: a second baseline median was 326.41;
current-runtime medians across fresh processes ranged from 679.85 to 765.74.
Disabling the Mamba epilogue produced 739.20 in one run and 725.85 in another,
while neighboring enabled runs produced 734.90 and 765.74. Its speed contribution
is therefore inconclusive here, not a stable claimed improvement. All raw sample
arrays are retained. Clocks/thermals/background desktop load were not controlled;
the stated ratios describe specific paired runs, not a deployment guarantee.

The baseline requires `-Wno-c++11-narrowing` because its existing Mamba reshape
code does not otherwise compile under this Clang. Its source was not modified.
Both builds use Release, the same compiler/HIP target, and the same external
OxtaMem Rust module. Build flags and binary hashes are recorded in each JSON.

## Isolated research: measured no-go for these candidates

`bench_decode_gemv` measures the complete packed dispatch (quantization, kernel,
output allocation/submission), W2/A8, one row. It records SHA-256 of the binary,
weights and inputs and verifies against a CPU integer oracle before timing.
Each policy is a fresh process, 20 warmups, 7 samples of 100 calls. All six
measured cases had zero maximum absolute error against that oracle.
These calls used `NSOS_GPU_MEMORY=device` and `NSOS_CUDA_SYNC=0`.

| K x N; packed payload | Normal GEMV | Persistent-grid candidate | LUT candidate |
|---|---:|---:|---:|
| 1024 x 4096; 1 MiB | 13.06 us | 80.20 us | 27.47 us |
| 4096 x 16384; 16 MiB | 90.79 us | 641.72 us | 429.51 us |

These are medians of host wall time per call. Separate stream-event elapsed
samples include host submission gaps; they are not isolated kernel timings.
Evidence: `gemv-<K>-<N>-<policy>.json`.

Neither candidate meets a performance promotion gate. Both remain opt-in;
default is `NSOS_GPU_EXPERIMENT=none`. These results reject these particular
implementations/geometries, not every possible persistent or LUT design. The
persistent candidate is a bounded grid-stride GEMV, **not** the proposed full
hybrid monokernel, and makes no one-block-per-CU residency guarantee.

## Remaining gates and work

- Trained checkpoint/pack, authoritative tokenizer, verified EOS and evaluation
  corpus are needed for perplexity, task quality and realistic throughput.
- No rocprofv3 was found in the configured Windows TheRock tool directories.
  Cache-hit rate, bandwidth, occupancy, launch latency and claimed cache
  residency have not been established by profiler counters.
- One model remains a serialized execution lane; the allocator supports one
  live graph owner per device. Multi-graph concurrent serving requires further
  allocator/session ownership work. Current batching is not a continuous-batch
  serving scheduler or a paged-attention system.
- Compact KV is FP16, not 2-bit. Live payload halves, but the float tensor pool
  can retain freed allocations; this does not prove total VRAM halves.
- Full-model persistent scheduling, direct AQL/HSA, learned routing skips,
  2-bit KV/eviction/GDN, a trained CfC drafter and TTT remain research backlog.
  The speculation script is a stateless full-prefix correctness reference,
  not a trained or optimized speculative serving implementation.
- Graph/FP16 policies remain opt-in. Do not enable approximation or claim
  above-SOTA performance from these synthetic correctness checks.

See `GPU_MODERNIZATION_IMPLEMENTATION.md` for build/runtime controls. All logs,
reports, the untrained fixture and the original source snapshot are under
`artifacts/gpu-modernization-20260919/`; existing evidence was retained.
