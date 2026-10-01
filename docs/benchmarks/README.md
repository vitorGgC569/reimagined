# NSOS benchmark package

This directory indexes the validated benchmark artifacts that must travel with
the repository. The July 2026 campaign was executed on a Tesla T4 and compares
the NSOS hybrid Mamba+Attention runtime with the reference Jamba dense and
Jamba MoE implementations, with and without OxtaMem where applicable.

## Clone the validated branch

```bash
git clone --branch nsos-gpu-phases12 --single-branch https://github.com/vitorGgC569/reimagined.git
cd reimagined
```

The repository is private, so authenticate through GitHub's normal credential
flow. Do not put a personal access token in source files, notebook cells or
clone URLs.

## Versioned artifacts

| Artifact | Purpose |
|---|---|
| [`jamba_vs_nsos_oxtamem_t4_2026-07-22.md`](jamba_vs_nsos_oxtamem_t4_2026-07-22.md) | Full protocol, environment, metrics, errata, limitations and verdict |
| [`../../colab/bench_expanded_validation_t4_2026-07-22.ipynb`](../../colab/bench_expanded_validation_t4_2026-07-22.ipynb) | Executed notebook with the successful empirical outputs |
| [`results/expanded_validation_t4_2026-07-22.json`](results/expanded_validation_t4_2026-07-22.json) | Machine-readable raw results and immutable dataset checksums |
| [`results/expanded_validation_t4_2026-07-22.provenance.json`](results/expanded_validation_t4_2026-07-22.provenance.json) | Artifact-set commit, runtime commits, Git blob IDs and SHA-256 integrity hashes |
| [`results/nsos_gpu_gold_validation_t4_2026-07-27/report.md`](results/nsos_gpu_gold_validation_t4_2026-07-27/report.md) | Fail-closed GPU release-gate report from the Tesla T4 run |
| [`results/nsos_gpu_gold_validation_t4_2026-07-27/result.json`](results/nsos_gpu_gold_validation_t4_2026-07-27/result.json) | Machine-readable GPU test, DP4A, checkpoint, mixed-precision and sanitizer results |
| [`results/nsos_gpu_gold_validation_t4_2026-07-27/SHA256SUMS`](results/nsos_gpu_gold_validation_t4_2026-07-27/SHA256SUMS) | SHA-256 manifest for every generated validation artifact |
| [`results/nsos_gpu_gold_validation_t4_2026-07-27/PROVENANCE.json`](results/nsos_gpu_gold_validation_t4_2026-07-27/PROVENANCE.json) | Exact source commit, environment, gate summary and downloaded-bundle hash |
| [`results/nsos_gpu_gold_validation_t4_2026-07-27.zip`](results/nsos_gpu_gold_validation_t4_2026-07-27.zip) | Byte-exact evidence bundle downloaded from the successful Colab runtime |

The notebook downloads the public bAbI data from its pinned upstream revision;
the dataset itself is intentionally not committed. Its train and test SHA-256
checksums are recorded in the JSON and technical report. The saved Google Drive
notebook is optional evidence, not a dependency for reproduction.

## What has been validated

- native NSOS CUDA build for `sm_75`;
- fail-closed GPU CTest gate: 24/24 passed, with zero disabled or skipped;
- Compute Sanitizer memcheck: 5/5 critical targets, zero errors and zero leaked bytes;
- DP4A dispatch propagation through model-pack load and clones;
- checkpoint v8 exact continuation and mixed-precision FP16 execution on Tesla T4;
- real four-step incremental decode parity against CPU streaming and full decode;
- native Rust/Python OxtaMem build and tests;
- corrected MQAR comparison with non-zero training seeds 1, 2 and 3;
- public `facebook/babi_qa`, configuration `en-10k-qa1`;
- NSOS plus structured entity-keyed OxtaMem on all 1,000 bAbI test questions;
- length extrapolation from training on 1–8 pairs to evaluation at 64 pairs;
- raw results, environment versions and dataset integrity hashes.

## Remaining scientific gates

These are follow-up validation tasks, not missing files required to clone or run
the published benchmark:

1. Repeat the dense and MoE Jamba bAbI runs across the same training seeds as
   NSOS, preferably with production fused Mamba kernels.
2. Remove or explicitly rename the nondeterministic `seed=0` sentinel and reduce
   the observed NSOS optimization variance.
3. Evaluate OxtaMem with learned semantic retrieval instead of a deterministic
   entity key.
4. Add broader open-domain language benchmarks, pretraining-scale experiments
   and matched-compute ablations.
5. Repeat the 16/32/64-pair extrapolation curve across multiple training seeds.

Until those gates are complete, the supported conclusion is that NSOS and
OxtaMem are exceptionally promising on the tested memory and reasoning
protocols—not that NSOS is already a generally superior language model.
