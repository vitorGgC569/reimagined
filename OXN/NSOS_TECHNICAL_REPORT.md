# NSOS Technical Report: The Omni-Efficient Architecture (v1.0)

## 1. Introduction
This report documents the NSOS architecture. It combines a **selective gated state-space
recurrence** (Mamba-family), **BitNet b1.58** ternary linear layers, **Kolmogorov-Arnold
Networks (KAN)**, and **Latent MCTS**. The system has no generic autograd tape: every module
is differentiated **by hand**, and the correctness of those hand-derived gradients is gated
by a finite-difference `test_gradcheck` (CTest). "Differentiable" here means
"every active path has a numerically-verified manual backward", not "an autograd engine".

## 2. Core Backbone: Jamba Hybrid
The backbone interleaves a linear recurrence layer with attention (one core per block;
blocks alternate by type).

*   **Selective recurrence (active path)**: The default scan is a **selective gated diagonal
    recurrence** — a per-channel scalar state with input-dependent decay:
    $h_t = \mathrm{decay}_t \cdot h_{t-1} + B_t \, x_t$, readout $y_t = \tanh(h_t)\, C_t$,
    where $\mathrm{decay}_t = \exp(-\,\mathrm{softplus}(\delta_t)\,A)$. This is an S4D/Mamba-style
    *diagonal* SSM, **not** the full Mamba-2 SSD state expansion (there is no $N$-dimensional
    per-channel latent state in this path). We implement **real BPTT**: the forward stores the
    state history $h_t$ and the backward walks time in reverse, propagating gradients through the
    recurrence and the $\tanh$/$C$ readout into $\delta, B, C, A$ and the input. CPU and CUDA
    paths are token-for-token equivalent (`test_gpu_parity` case `mamba_streaming`).
*   **Full Mamba-2 SSD ($N$-state) — opt-in, in progress**: a state-expanded SSD kernel
    (`mamba_ssd_forward_kernel`, $h \in \mathbb{R}^{P\times N}$, $y=\sum_n h_n C_n$) plus a
    short causal conv1d and **separate** $\delta/B/C/z$ projections are being wired behind a
    `MambaConfig` flag (default OFF preserves the validated path and existing checkpoints).
    Until that flag is validated, claims of "Mamba-2 SSD / SSD duality" apply only to the opt-in
    path, not the default.
*   **Attention**: MLA-style latent attention with GQA and RoPE blocks are interspersed. We
    implement full backward differentiation for the $Q, K, V, O$ projections (chain rule),
    statically verified and Colab-gated against the host path.

### 2.1 Decisão formal — destino do SSM ("verdade primeiro")

A diretiva era: *ligar o kernel SSD morto OU renomear o claim*. Decisão tomada = **AMBAS**,
em etapas com prioridades distintas (não uma OU exclusiva):

1. **Renomear o claim — FEITO.** O caminho ativo é descrito honestamente como "selective
   gated diagonal SSM" (não "Mamba-2 SSD") aqui, em `docs/` e em `BENCHMARK_REPORT.md`. O
   `mamba_ssd_forward_kernel` (estado-N) fica presente mas **explicitamente rotulado como
   opt-in não-ativado** — não é dead code esquecido.
2. **Corrigir a via diagonal — FEITO e validado.** Projeções δ/B/C/z separadas, conv1d
   causal, C aplicado uma vez, gate `silu(z)` (flag `NSOS_MAMBA_PROPER_SSM`). Backward
   finite-difference-gradcheck-ado (6.2e-3 / 4.7e-4 / 2.4e-3) e o stack **treina e
   generaliza** (66.7% exact-match held-out PT; ver `BENCHMARK_REPORT.md` §0.1). Elimina a
   degenerescência `delta≡C` / `C²` que motivava a diretiva.
3. **Ligar o kernel N-state (Mamba-2 completo) — AGENDADO** como PR isolado (precisa do
   forward/backward N-state em CPU+GPU + build dedicado). É ganho de capacidade, não conserto
   de bug; a via diagonal corrigida já é não-degenerada e comprovadamente treinável.

## 3. BitNet b1.58 & BitFastKAN
*   **BitLinear**: Implements **Straight-Through Estimator (STE)**. Forward pass uses quantized weights $\{-1, 0, 1\}$, while backward pass updates the high-precision latent weights.
*   **BitFastKAN**: Implements a learnable feature extractor using **Radial Basis Function (RBF)** splines.
    *   Forward: $y = x \cdot W_{base} + \text{Spline}(x) \cdot W_{rbf}$.
    *   Backward: We derive gradients for $W_{base}$ (linear path) and $W_{rbf}$ (spline coefficients). Crucially, we also compute $\partial \text{Basis} / \partial x$, allowing gradients to flow back to the input, enabling stacking of KAN layers.

## 4. Reasoning: Latent MCTS
We integrate Monte Carlo Tree Search into the training loop via **Policy Distillation**:
*   The model predicts a policy $\pi_\theta(s)$.
*   MCTS performs a search to find a better policy $\pi_{MCTS}$.
*   Loss Function: $L = L_{CE} + \alpha \cdot KL(\pi_{MCTS} || \pi_\theta)$.
This effectively uses MCTS as a "teacher" for the neural network "student", imparting reasoning capabilities (System 2) into the fast weights (System 1).

## 5. Stability & Engineering
*   **Memory Safety**: We resolved critical heap corruption issues in the Python/C++ boundary (Pybind11) by strictly enforcing ownership policies (`std::shared_ptr`, `return_value_policy::reference`).
*   **Pipeline**: The training system includes:
    *   **Cosine Learning Rate Schedule** with Warmup.
    *   **Gradient Clipping** to prevent exploding gradients in deep recurrences.
    *   **Validation & Checkpointing** for reliable experimentation.

## 6. Conclusion
NSOS is a **functional, stable, and trainable** neural system with numerically-verified manual
gradients (`test_gradcheck`) and a CPU/GPU-parity inference runtime. **Generalization is not yet
quantified in-repo**: end-to-end metrics (held-out perplexity and exact-match on a controlled
single task) are produced by `colab/train_gpu_phases_t4_v2.ipynb` and must be cited from a run,
not asserted here. The honest current status is "the machinery is correct and trains; the
empirical evidence is generated by the validation notebook." See `docs/NSOS_VALIDATION_STATUS.md`
for the live gate list.
