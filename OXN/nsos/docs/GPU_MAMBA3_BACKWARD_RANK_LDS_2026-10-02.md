# Mamba-3 Flash backward: rank-sized LDS

The Flash LDS v2 backward previously reserved 45,608 bytes of shared memory
for eight ranks, including SISO rank 1. Its shared dQ/dK arrays now use template
capacities 1, 2, 4 and 8. Ranks 3–4 select capacity 4; ranks 5–8 select capacity 8.
The layouts use 16,936 / 21,032 / 29,224 / 45,608 bytes respectively.

Runtime rank loops, local dQ/dK arrays, arithmetic order, reductions, scalar
checkpoint-seam halos, owner/finite/publication gates and checkpoint formats
are unchanged. Host admission still checks the maximum layout; the supported
hardware set is unchanged. The provider stays `flash_fp32_replay_lds_v2`.
Hierarchical scan remains an experimental opt-in.

## Private RX 7600 differential evidence

Before production integration, root linked two private core overrides against
the same source-attested hierarchy-era library. The override map verifies all
eight GPU-core exports, with no original archive core member selected.
Runtime DLLs and SDK BLAS resources are fixed; the driver reports the actual
loaded HIP DLL path. Initial wrong-runtime and missing-resource attempts remain
rejected and preserved separately.

Standalone FP32 comparison covers ranks 1/2/3/4/5/7/8, sequences 33/65,
seeded initial/final adjoints, empty/ragged/poison padding, global NaN rejection,
and D768/S512/R1. All 41,043,864 raw snapshot bytes match bitwise. Three paired
fresh processes (five warmups, ten measurements each) give backward speedup
2.141x, paired log-t 95% interval [1.975, 2.321]. The scope is one layer's
`backward_owned`, including gradient GEMMs, allocations and owning result copies;
it excludes forward, publication and optimizer.

The separate full-model experiment uses D768/L16/B1/S512, state 128, head 64,
expand 1, BF16 projections, fused Adam LR .002, weight decay .01, clipping 1,
seed 1701 and synthetic answers. Initial weights plus final weights/versions and
authoritative optimizer moments after 13 commits match bitwise across
524,296,912 snapshot bytes. Three timing0 fresh-process pairs (warmup 3,
measure 10) give 1.895x whole-step speedup, interval [1.850, 1.941]:
0.765 to 1.449 steps/s, 391.6 to 742.0 tokens/s.

Separate timing1 processes independently replay 13 commits and measure step14:
backward 1041.466 to 479.156 ms, forward 192.730 to 197.201 ms, optimizer
28.892 to 26.277 ms. These are diagnostic phase samples, not a decomposition
of the timing0 means. No timing flag is toggled and no cross-identity sidecar
is loaded. These numbers describe the private paired experiment, not a quality
benchmark or the final production binary. The original mixed-precision failures
and GOLD576 admission block remain open.

## Integrated shared-build acceptance

The integrated HIP rebuild passed all 162 CTest cases (262.32 seconds); the
current CPU build passed all 72 cases (35.95 seconds). An independent review
found that the newly added odd-rank tests exercised only empty sequences.
The test now includes both active ragged and empty cases for ranks 2/3/5/6/7.
The rebuilt regression passed again in 8.34 seconds, with the original VJP
oracles and numerical limits retained. Rebuilding this test did not alter the
accepted core library or Python extension.

A new full-model driver links directly to the rebuilt shared library, without
the private core override. Three fresh GPU processes (gate, timing0, timing1)
passed exact loss/identity/token/owner-counter checks against the private
candidate. The 524,296,912-byte authoritative state snapshot remains bitwise
identical. This new shared-build timing0 sample measures 686.371 ms/step:
**1.457 steps/s and 745.95 tokens/s**. The separate step14 diagnostic measures
forward 189.772 ms, backward 488.033 ms, optimizer 24.811 ms and wall 705.231 ms.
This one production timing process confirms deployment; the three private
paired processes provide the speedup estimate and interval above.

New core library SHA-256:
`4f449ec85e5db07840e9cd304014da42be9c82766db7c620d5077a4ece7f5e6a`.
New HIP Python extension SHA-256:
`3076f3b020bccfe755b94adb9256931bff6c6a945561a2710844ff0a76efd772`.
The previous library/extension are archived with their original fingerprints;
historical receipts are not repinned to the new epoch.

These results cover the stated synthetic SISO BF16 geometry on RX 7600, not
general corpus training, time to quality, or other GPUs. The original four
mixed-precision-versus-FP32 failures and GOLD576 block remain unresolved.
Hierarchical scan remains optional. GPU gradient copies are still present.
Further backward work should measure remaining replay/reduction/GEMM/copy costs.
Mamba3H research stays paused at the user's request.

Evidence and sealed acceptance receipt:
`artifacts/maestri_integral_20261001_gpuopt/root-backward-rank-lds-v1/`.
