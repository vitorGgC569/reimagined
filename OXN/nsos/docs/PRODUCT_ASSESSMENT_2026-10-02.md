# Product assessment and closure plan — 2026-10-02

The rank-sized LDS backward optimization is integrated and locally validated.
The overall GPU-enabled product is not release-ready. `PRODUCT.md` remains the
authority for supported capabilities: CPU runtime/API/CLI/packs are MVP surfaces;
HIP/CUDA, TTT and distributed paths have separate experimental/research status.

## What the new result establishes

On RX7600/gfx1102, D768/L16/B1/S512/N128/P64/R1, BF16 projections and fused Adam,
three fresh process pairs measure1.89467x whole-step speedup, paired log-t95
[1.84969,1.94074]. The matched baseline measured0.764924steps/s; candidate
1.449262steps/s. A new shared-library execution without private override measured
1.456938steps/s and745.952tokens/s. Its separate timing1 step14 backward was
488.033ms; private baseline/candidate diagnostic backward1041.466/479.156ms.
The phase samples are not a decomposition of the timing0 paired means.

SISO LDS45608->16936bytes preserves the runtime arithmetic. All524296912
authoritative state snapshot bytes match after13 commits.72CPU/162HIP tests and
the corrected active/empty odd-rank GPU regression pass. This is a substantial
engineering result because speed did not require relaxing numerical checks or
discarding the owning-gradient/checkpoint/optimizer state contracts.

The recipe uses a synthetic vocab257 workload. It does not establish language
quality, larger-vocabulary throughput, convergence time, superiority to Mamba-2,
performance on other GPUs, or SOTA. Muon's claimed40% reduction in time to quality
remains unproved. The hierarchical scan is opt-in: target experiments regressed.

## Engineering judgement

Subjective scores, not benchmark metrics: **8/10 for the recent research and
engineering process; 5/10 for readiness of the intended GPU-enabled end-to-end
product; 7/10 overall**. The lower readiness score does not invalidate the
documented CPU MVP. It reflects the remaining GPU/quality/packaging gates.

Strengths: real target-hardware work; independent reviews; exact optimizer-state
and checkpoint checks; causal controls; preserved failed receipts; useful negative
results that keep experimental paths out of defaults. The rank-LDS change is
small, explainable and backed by measured end-to-end benefit.

Risks: broad project scope; complex owner/lifetime/identity interactions; large
runtime files; local absolute-path/toolchain dependencies; substantial benchmark
machinery and raw data still outside Git; insufficient model-quality evidence.
Tests and architectural novelty cannot replace a useful trained checkpoint and
a repeatable user workflow. These scores are an assessment of reviewed evidence,
not a complete commercial, security or cross-platform audit.

## What closes the product

| Priority | Deliverable | Completion criterion |
|---|---|---|
| P0 | Resolve four GOLD mixed-vs-FP32 failures | Identify expected operand rounding versus implementation defects with independent projection/VJP oracles; preserve failed runs and original limits. Any new precision contract must be declared and justified independently before fresh validation. |
| P0 | Profile current optimized backward | Fresh ABI80/source/lib/caller/runtime/MAP admission; ON/OFF controls; valid nonnegative finite events, owned-job teardown and explicit overlapping-collector handling. Archived ABI64 profiler must not link to this build. |
| P0 | One reproducible product workflow | Clean install/build -> train -> save -> restart/resume -> infer -> export/load pack -> API/CLI use. Validate deterministic continuation and fail-closed owner/finite/clip/accumulation/rollback contracts. |
| P1 | Useful model quality | Pick the first user/task, corpus and tokenizer. Evaluate held-out data with contamination controls, seeds and confidence intervals against meaningful equal-budget baselines. Publish perplexity/task quality plus latency/memory; synthetic training loss is insufficient. |
| P1 | Release reproducibility | Portable benchmark inputs/scripts and downloadable pinned artifacts; supported toolchain/runtime matrix; clean CPU/Rust/Docker/gatekeeper/fuzz/boundary CI for the exact release, plus real target-GPU and memory-checker gates. |
| P1 | Operations and packaging | Versioned packs/model/tokenizer/provider metadata, install instructions, bounded jobs/requests, logs/counters, failure recovery, concurrency and realistic memory limits under the chosen supported deployment. |
| P2 | Controlled optional research | Keep Mamba3H, KAN, TTT and speculative modules out of the critical release path. Promote only after measured utility and their own reproducibility/quality gates. |

The smallest defensible release is a bounded CPU product with an explicitly
experimental RX7600 training accelerator and one demonstrated application/model.
A stronger GPU product requires closing its own gates. Circuit/PCB examples may
be an application benchmark; they are not evidence of SPICE/EDA simulation,
physical design correctness or GPU silicon modeling.

## Evidence and publication limits

See [rank-LDS report](GPU_MAMBA3_BACKWARD_RANK_LDS_2026-10-02.md) and
[portable acceptance summary](GPU_MAMBA3_BACKWARD_ACCEPTANCE_2026-10-02.json).
The summary copies existing sealed measurements; it is not a fresh experiment.
Raw private executables/runtime resources/state snapshots/worker logs remain
local ignored artifacts. A clone can inspect code and test sources, but cannot
independently replay the entire timing study without those artifacts and recipes.
The GitHub push does not itself certify that the new remote CI run has passed.
