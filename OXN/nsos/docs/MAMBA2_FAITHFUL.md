# Faithful Mamba-2 mode

`ModelConfig::mamba2_faithful` selects the Mamba-2 block implemented by
`state-spaces/mamba` rather than the historical NSOS recurrence.

The default configuration matches the reference defaults used by the project
benchmark:

- `expand=2`
- `headdim=64`
- `ngroups=1`
- `d_state=min(max(d_model/2, 8), 64)`
- causal depthwise convolution width 4 over the concatenated `x`, `B`, and `C`
  streams
- `RMSNormGated(..., eps=1e-5, norm_before_gate=false)`

For input width `D`, the block uses `I=expand*D`, `H=I/headdim`, and the
following recurrence:

```text
delta = softplus(dt + dt_bias)
A     = -exp(A_log)
h_t   = exp(delta_t*A) * h_(t-1) + delta_t*B_t*x_t
y_t   = C_t*h_t + D*x_t
out   = out_proj(RMSNorm(y_t * SiLU(z_t)))
```

`B` and `C` are shared by heads within each group. The CUDA forward, backward,
and incremental decode kernels use the same grouping and `D*x` placement.

## Linear and ternary modes

The five input slices are represented by separate `BitLinear` objects so QAT
can keep `dt/B/C` in float while ternarizing `x/z`. Their concatenation is
algebraically equivalent to the reference's single `in_proj`.

In float mode these projections use `BitLinear::set_exact_linear_mode(true)`:
the historical implicit input RMSNorm is disabled and the per-output magnitude
is fixed at one. Consequently the float path is a plain linear projection.
Switching the robust projections to QAT changes only their numerical
representation; it does not change the Mamba-2 graph.

## Initialization

- projection and convolution weights use the same uniform bounds as PyTorch
  `nn.Linear`/`nn.Conv1d` defaults;
- convolution bias uses `[-1/sqrt(d_conv), +1/sqrt(d_conv)]`;
- `dt` is sampled log-uniformly from `[1e-3, 1e-1]` and stored through the
  inverse softplus;
- `A` is sampled uniformly from `[1, 16]` and stored as `A_log`;
- `D` and the gated RMSNorm weight start at one.
- in a model stack, `out_proj` is scaled by `1/sqrt(n_layers)` as in the
  reference residual initialization.

## Stack and checkpoint behavior

A pure faithful Mamba layer is learned RMS pre-norm (`eps=1e-5`) → Mamba-2 →
residual. The final model RMSNorm uses the same epsilon and a learned scale,
matching the reference wrapper. The layer does not add the old dense FFN. MoE,
KAN, CHRASS, attention, and TTT remain explicit hybrid choices.

For whole-model parity, faithful mode also uses the reference embedding
initialization (`Normal(0, 0.02)`) and a bias-free, plain-linear LM head. Weight
tying remains controlled by `tie_word_embeddings`.

The faithful mode changes parameter shapes, so it is fingerprinted in NSOS
checkpoints and persisted in model packs. Packs created before this field
existed load with `mamba2_faithful=false`; they do not silently reinterpret old
weights. Official PyTorch checkpoints are not ABI-compatible because NSOS
stores the combined input projection as five named slices, although the math
and dimensions are equivalent.

## Validation gates

- `test_mamba2_reference`: compares NSOS against an independent implementation
  of the official formula using identical weights;
- `test_gradcheck`: finite-difference gradients for input, `A_log`, convolution,
  gated RMSNorm, and `dt_bias`;
- `test_gpu_parity_mamba_faithful`: CPU/CUDA forward, backward, deterministic
  fallback, prefill, and incremental decode parity.

The Colab benchmark fallback was also aligned to `headdim=64`, official
`A/dt` initialization, gate-before-normalization, and no weight tying.
