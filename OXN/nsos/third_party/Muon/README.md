Muon primary reference: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py

Pinned file SHA256: `1eece0b562f159e88f3a65da9b6ee8d0ab2dd647cf25c88e355d4f262e796e66`.
The accompanying MIT license is pinned to the same commit.

NSOS policy `muon_hidden_ns5_fp32_fp64norm_nesterov095_v1` implements momentum
0.95, Nesterov, five quintic Newton–Schulz iterations with coefficients
3.4445 / -4.7750 / 2.0315, epsilon 1e-7, smaller-dimension transpose and original
matrix shape scaling. It deliberately uses FP32 iteration and FP64 normalization
instead of upstream BF16 iteration. Its transform approximates the polar factor;
it is not an exact SVD orthogonalization.

Muon applies only to hidden `layers.*.weight` matrices with rank 2, excluding
router, embedding and final head. Auxiliary vectors/scalars, embeddings and tied
heads retain AdamW and the existing explicit Mamba3 dt_bias/D no-decay registry.
`NSOS_OPTIMIZER=muon_ns5_fp32_v1` requires explicit `NSOS_MUON_LR` and GPU FP32
or BF16 execution with FP32 optimizer state. The matrix LR uses the auxiliary
Adam scheduler ratio and parameter LR scales; Adam LR does not become Muon LR.

The transaction retains device activity/union predicates, mean-before-clip,
global finite gates, full rollback and lazy state. Muon momentum occupies the
sidecar first-moment slot; second moment must be exactly zero. Runtime identity
seals coefficients, partition, normalization, reference and algorithm. Loading
Adam state or identity-less legacy state as Muon is rejected.

`NSOS_OPTIMIZER_FUSED_EPILOGUE=1` fuses update and active gradient clearing after
all VJPs/global clipping. A single-use owner proof lets the next group omit its
full-bank gradient zero launch. It requires exclusive gradient ownership until
the proof is consumed; abort, restore and storage changes invalidate the proof.
Gradient storage remains necessary for accumulated/tied weights and global
clipping. Universal backward-update fusion would violate those dependencies.

The initial FP32 GPU transform uses bounded 16x16 LDS tiles and shared scratch
reused serially between matrices. It is opt-in and requires timing and equal-
quality comparisons before promotion. Neither a 1.3x speedup nor 40% fewer
training steps is established by implementation/parity tests.
