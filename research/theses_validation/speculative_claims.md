# Untestable Claims — Documented

This document inventories thesis claims that this harness deliberately
does NOT attempt to validate, with the concrete blocker for each.
Honesty matters more than coverage: pretending to test something we
can't actually test would be worse than excluding it.

For each claim:
- **Where in the theses:** which section / page-equivalent
- **What it claims:** the testable assertion (if any)
- **Why we don't test it:** the concrete blocker
- **What would be needed:** what a real test would require

---

## 1. Free Energy Principle / Active Inference as the loss function

**Where:** Tese 1, section "O Arcabouço da Inferência Ativa"

**What it claims:** Replacing cross-entropy with a variational free
energy functional (a la Friston) makes the model "an active
agent" with genuine epistemic curiosity, and outperforms passive
language modeling.

**Why we don't test it:**
1. The thesis describes a "prior path" and "posterior path" through
   the Mamba layers but doesn't write a concrete training objective.
   You can't ablate something whose loss function isn't specified
   precisely enough to implement.
2. The only paper-grade FEP-for-LLMs work (Friston et al.'s active
   inference for language) is on toy POMDPs, not autoregressive
   text models.  No published benchmark exists to compare against.
3. To test honestly you'd need: (a) a generative model with explicit
   posterior approximation, (b) a sampling-based control loop,
   (c) a baseline LM at matched compute, (d) a task that
   distinguishes "active" from "passive" agents.  That's
   a 6-12 month research project.

**What would be needed:** Implementation of a published FEP-trained
LM benchmark (none exists as of writing).  In the meantime, the
"epistemic curiosity" of an LLM is observable indirectly via
calibration metrics (Brier score, ECE) and active-learning
performance — but those don't require a FEP loss to measure.

---

## 2. Symplectic SGD / Hamiltonian-embedded gradient flow for TTT

**Where:** Tese 1, section "O Fim do Tempo Discreto" (SPEL)

**What it claims:** Constraining gradient flow during Test-Time
Training to a symplectic manifold prevents catastrophic forgetting
because energy is "formally shielded."

**Why we don't test it:**
1. Symplectic integrators for SGD are an active research area
   (Maddison et al., Tatu et al. 2022).  Published results are
   mixed: symplectic methods stabilize some optimization paths
   but don't reliably prevent catastrophic forgetting in deep
   nets — and where they do help, the benefit is typically <5%
   relative to AdamW + EMA.
2. The thesis cites "Lie group constraints on the gradient flow"
   but doesn't specify the group action — there are many
   choices (SE(3), SO(n), Sp(2n)) and they're not interchangeable.
3. Implementing symplectic AdamW correctly takes ~500 lines and
   needs benchmarking against the actual catastrophic-forgetting
   metric the thesis mentions, which requires a continual-learning
   benchmark (SplitMNIST etc.) — different domain.

**What would be needed:** Pick a concrete symplectic optimizer
(e.g., HamiltonianMonteCarlo-derived sample-based update) and a
continual-learning task with established catastrophic-forgetting
metrics.

---

## 3. HDRAM with Hypertoken ECC + Krylov subspaces

**Where:** Tese 2, section 4 ("O Salto para a Consciência Simbólica")

**What it claims:** Replace KV cache with a Holographically-Defined
Random Access Memory based on "Hypertoken Error-Correcting Codes"
embedded in Krylov subspaces, enabling million-token recall with
constant-size memory.

**Why we don't test it:**
1. "Hypertoken ECC" is not a standard term in any published paper.
   Searching IEEE/ACM/arXiv returns zero hits for the exact phrase.
   The thesis cites no concrete construction.
2. The closest published work is Vector Symbolic Architectures (Kanerva
   1988, Plate 1995) — real and useful but operates at much smaller
   scale and doesn't claim Grover-style sublinear recall.
3. "Krylov subspaces for memory" is plausible-sounding but Krylov
   subspaces are a Lanczos-algorithm tool for solving Ax=b; their
   application to associative memory is speculation in this thesis.

**What would be needed:** A concrete published HDRAM paper with
reference implementation.  If you find one (Tese author may have a
specific paper in mind that wasn't cited), the test is straightforward
to add: same KV-budget, measure retrieval accuracy as context length
grows.

---

## 4. Hamilton-Jacobi PDE → Riccati ODE for catastrophic-forgetting-free TTT

**Where:** Tese 2, section 5.1 ("Aprendizado Contínuo")

**What it claims:** Online learning as the viscosity solution to a
Hamilton-Jacobi PDE prevents knowledge erasure.  Converts to Riccati
ODEs for real-time computation.

**Why we don't test it:**
1. The thesis writes two equations (P-dot and q-dot) without
   specifying what Φ₁ is or how this connects to neural network
   weights.  Without the connection, the equations don't define
   a training algorithm.
2. The HJ-PDE-as-RL connection (Yang et al., NeurIPS 2022) is real
   for tabular RL — extending it to deep networks is open research
   with no consensus method.
3. A faithful test would require pinning down the operator (P, q, Φ₁)
   in terms of the model's parameter vector, which the thesis
   doesn't do.

**What would be needed:** Concrete published method that operationalizes
HJ-PDE for deep continual learning, with reference code.

---

## 5. ActivationReasoning + Differentiable Logic Tensor Networks

**Where:** Tese 1, section "O Fim das Alucinações"

**What it claims:** Sparse Autoencoders extract atomic concepts, then
Logic Tensor Networks enforce that activation patterns obey first-order
logic axioms — making hallucination "mathematically impossible."

**Why we don't test it:**
1. Sparse Autoencoders for LLM interpretability is real and active
   research (Anthropic 2024).  Useful for understanding circuits.
   But "punishing activations that violate axioms" via LTN-style
   gradients has not been demonstrated to reduce hallucination
   measurably on real LLMs — the published LTN work (Serafini 2017,
   Donadello et al.) operates on small structured-data tasks.
2. To test you'd need: (a) a trained SAE dictionary (we don't have one
   for the model in question), (b) a hallucination benchmark
   (TruthfulQA et al.), (c) a clean way to convert "logic violation"
   into a differentiable loss term.
3. Even if (a)-(c) worked, the claim "mathematically impossible to
   hallucinate" is provably wrong — any neural net with finite-precision
   weights can produce arbitrary outputs.  We'd be testing "does it
   reduce hallucination significantly", not the stronger thesis claim.

**What would be needed:** An LLM-scale published implementation of
LTN-regularized training with hallucination benchmarks.  Closest
analogue is constitutional AI / RLAIF, which uses LLM critics rather
than formal logic — different approach.

---

## 6. The 45.58% activation sparsity scaling law

**Where:** Tese 2, section 3.1 (KAN/MoE scaling)

**What it claims:** A specific 45.58% activation sparsity rate is the
Pareto-optimal point for matching FP16 accuracy at 1-bit weights, with
this exact number being a universal scaling law.

**Why we don't test it:**
1. The precise number (45.58%) is suspiciously specific without
   citation.  Real scaling laws (Chinchilla, etc.) come from
   logged compute-vs-loss curves across many model sizes; we'd
   need to repeat that experiment to validate.
2. A real test would require sweeping sparsity at 10+ model sizes
   spanning at least 2 orders of magnitude (e.g., 1M to 1B params).
   That's ~50-100K GPU-hours.

**What would be needed:** A multi-million-dollar sweep, or a paper
that already did it.  The closest published scaling-law-for-sparsity
work (Frantar et al., 2023) gives ~50% as the rough optimum without
3-decimal precision.

---

## 7. "100 billion parameters on a DGX Spark"

**Where:** Tese 1, "Supremacia Holográfica" / Tese 2, section 6

**What it claims:** With NSOS's combination of techniques (LrcSSM +
KANtize + 1.58-bit + Decoupled-STE + HDRAM), 100B-parameter models
fit and train on a single NVIDIA DGX Spark (128GB unified memory).

**Why we don't test it:**
1. Aspirational scale claim, not testable on a Colab T4 (15GB).
2. The constituent claims are each individually testable at smaller
   scale (and most are tested by this harness).  Whether they
   compose at 100B is a separate empirical question that requires
   actual hardware.
3. DGX Spark's actual delivered FLOPs/bandwidth at the cited
   precision modes are still being characterized by independent
   benchmarks; no public 100B/1.58-bit run on Spark exists yet.

**What would be needed:** Access to DGX Spark + 6-12 months of
engineering to scale NSOS to 100B params.  Worth doing if the smaller
tests in this harness all confirm; not worth doing if they don't.

---

## 8. Lyapunov-exponent memory decay law

**Where:** Tese 2, section 4 (T_HDRAM equation)

**What it claims:** Information lifetime in HDRAM is inversely
proportional to the largest Lyapunov exponent of the network's
latent dynamics.

**Why we don't test it:**
1. Even setting aside the HDRAM question (above): computing the
   largest Lyapunov exponent of a transformer's latent trajectory
   is research-grade — there's no standard way to do it (Pasemann
   1996 for RNNs doesn't trivially extend).
2. The equation gives a proportionality, not a falsifiable
   prediction.  Without a constant of proportionality and a
   ground-truth "information lifetime" measurement protocol,
   there's no way to test "yes this holds" vs "no it doesn't."

**What would be needed:** A measurement protocol for both quantities,
plus a published method to estimate Lyapunov exponents in deep nets.

---

## 9. ReST-MCTS* retrospective reward injection

**Where:** Tese 2, section 5

**What it claims:** Retrospectively credit-assign rewards along
MCTS rollouts that ended in verified-correct answers, then retrain
the policy.  Improves reasoning over time without human-labeled data.

**Why we don't test it (yet):**
1. This is testable in principle and matches a published ReST-MCTS*
   paper (Zhou et al., 2024).
2. But the test setup is heavy: requires a verifier (compiler /
   sympy / theorem prover) + a generation policy that produces
   varied solutions + several days of compute to see learning signal.
3. Out of scope for this initial harness.  If the simpler
   `search_mcts.py` test (AB-MCTS vs UCT) shows promise for
   adaptive-branching reasoning, ReST-MCTS* is the natural next
   step.

**What would be needed:** ~600 lines of code (verifier wrapper +
RL update loop) + a small reasoning benchmark (GSM8K subset) +
~10-50 GPU hours.  Achievable in a follow-up but not in scope here.

---

## What this leaves us with

The tests under `tests/` validate ~12 concrete claims with proper
isolation, fair baselines, and reproducible seeds.  The 9 claims in
this document are honestly out of scope.  Together, this is what we
can defensibly say:

- For testable claims: we have numbers.  Some claims replicate,
  some don't; the results are what they are, not what we want them
  to be.
- For untestable claims: we explicitly don't claim either replication
  or refutation.  Anyone who wants to validate them should write
  the missing pieces (concrete loss function, reference implementation,
  scale-required benchmark) and either run them or cite published
  work that already did.

This is the honest middle ground between "everything in the theses is
true" and "everything in the theses is hype."  Most of the testable
claims come from real published work and have a real chance of
replicating; most of the untestable ones use real-sounding terminology
to gesture at directions that aren't yet science.
