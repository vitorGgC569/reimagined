# Oxta Theses Validation — Research Harness

**Scope:** standalone PyTorch test harness that isolates each testable claim from
the two architectural theses ("Síntese Ômega" and "Arquitetura da
Superinteligência") and compares it against a controlled baseline.  Lives
**outside** `OXN/nsos/`.  Does not touch production code.  Does not affect
training of the production NSOS model.

**Why this exists:** the original POC the user wrote (RNN-tanh + STE vs LrcSSM +
Decoupled STE on WikiText) showed a dramatic loss gap, but that gap was an
artifact: the baseline initialized weights at std=0.02 and then rounded them to
ternary, collapsing the entire network to zero at init; the "Oxta" variant had a
hidden FP32 bypass (HGF) that did 100% of the work.  The two networks were not
testing what the thesis claimed.

This harness exists so each claim can be tested **fairly** — same model size,
same init seed, same data, same training budget, only the one variable swapped.

---

## What this harness tests rigorously

Each test isolates ONE claim with a clean baseline.  Same seed, same data, same
model size, same optimizer.  Outputs: training loss curve, held-out perplexity,
loss-spike count, parameters, wall-time, throughput.

| # | Claim under test | Baseline | Variant | File |
|---|---|---|---|---|
| 1 | Decoupled STE beats classical STE for ternary training | Classical STE | Decoupled STE (separate τ_f, τ_b) | `tests/quantization.py::test_decoupled_ste_vs_ste` |
| 2 | HGF (FP16 bypass) stabilizes ternary training | Pure ternary | Ternary + HGF residual | `tests/quantization.py::test_hgf` |
| 3 | Continual QAT (FP16 → ternary annealing) recovers loss | Ternary from scratch | FP16 warmup → ternary anneal | `tests/quantization.py::test_continual_qat` |
| 4 | Denoising Dequantization Transform vs STE | Classical STE | DDT (Ridge gradient) | `tests/quantization.py::test_denoising_dequant` |
| 5 | LrcSSM (diagonal Jacobian, parallel scan) vs Mamba2 | Real `mamba_ssm` | LrcSSM ours | `tests/sequence_models.py::test_lrcssm_vs_mamba2` |
| 6 | CfC (closed-form continuous-time) vs simple RNN | Simple RNN | CfC via `ncps` | `tests/sequence_models.py::test_cfc_vs_rnn` |
| 7 | KAN vs MLP at matched parameter count | MLP | Real KAN | `tests/kan_variants.py::test_kan_vs_mlp` |
| 8 | SKAN (single-parameter B-splines) vs vanilla KAN | KAN | SKAN | `tests/kan_variants.py::test_skan_vs_kan` |
| 9 | KANtize lookup tables vs runtime spline evaluation (latency) | KAN runtime | KAN lookup | `tests/kan_variants.py::test_kantize_latency` |
| 10 | NCA pretraining transfers to language modeling | Cold-start LM | NCA-pretrained LM | `tests/nca_pretrain.py` |
| 11 | GRPO matches/beats PPO without critic | PPO with value head | GRPO (group-relative) | `tests/rl_grpo.py` |
| 12 | AB-MCTS (Thompson sampling) beats vanilla UCT | Vanilla MCTS/UCT | AB-MCTS | `tests/search_mcts.py` |

---

## What this harness does NOT test (and why)

| Thesis claim | Why we don't test it |
|---|---|
| Free Energy Principle / Active Inference as loss | Requires a generative world model + variational posterior + active sampling.  Building that scaffolding from scratch is a multi-month project; no public benchmark for "FEP-trained LLM" exists.  See `speculative_claims.md`. |
| Symplectic SGD for catastrophic-forgetting prevention | Real research, but the thesis cites no concrete implementation; symplectic integrators for SGD are an active research area with mixed published results. |
| HDRAM with Hypertoken ECC + Krylov subspaces | "Hypertoken ECC" is not a standard term in any paper.  Krylov subspaces for memory is speculation. |
| Hamilton-Jacobi PDE / Riccati ODE for TTT | The thesis writes ODE equations but doesn't connect them to a published method.  No reference implementation. |
| ActivationReasoning + Differentiable Logic Tensor Networks | LTNs (Serafini 2017) are real but don't scale to LLMs.  "ActivationReasoning" is not a standard published framework. |
| 45.58% sparsity scaling law | Cited as exact constant but the thesis gives no paper reference.  Would need 1B+ parameter sweep to test. |
| "100 billion parameters on a DGX Spark" | Aspirational hardware claim, not testable here. |
| FEP + Mamba2 prior/posterior pathways | See FEP entry above. |
| Lyapunov-exponent-driven memory decay formula | Cited equation has no concrete training protocol; no measurable prediction to validate against. |
| ReST-MCTS* retrospective reward injection | Could be implemented but requires a verified-correct training task generator and several days of compute to show signal.  Out of scope; revisit if MCTS test (#12) shows promise. |

These are documented in `speculative_claims.md` with concrete blockers to
implementation.

---

## Honest methodology principles

1. **One variable at a time.**  When testing claim X, baseline and variant
   differ in exactly X.  Same init seed, same data order, same optimizer,
   same wall-clock budget OR same step count (whichever is fairer for the
   claim).

2. **No hidden bypasses.**  If a variant has a "fallback" path (like HGF's
   FP16 residual), it's documented in the test and the baseline gets the
   same bypass (or neither does).  We don't compare "ternary + bypass" vs
   "ternary only" and call the ternary technique the winner.

3. **Real reference implementations where possible.**  Mamba2 uses
   `pip install mamba-ssm`; CfC uses `pip install ncps`.  We don't
   hand-roll baselines that handicap them.

4. **Metrics that match the claim.**  Stability claim → loss-spike count
   over training.  Convergence claim → tokens-to-target-loss.  Efficiency
   claim → throughput tok/s + memory.  Quality claim → held-out PPL.

5. **Reproducibility.**  All tests accept `--seed` (default 42).  Same seed
   produces identical loss curves modulo CUDA non-determinism.

6. **Honesty over advocacy.**  If a thesis claim doesn't replicate, we
   report it doesn't replicate, with the numbers.  If it does, we report
   the effect size.  We don't sandbag the baseline.

---

## How to run

### On Colab (recommended, T4 is enough)

Open `colab/theses_validation.ipynb` in Colab and run cells in order.
The notebook installs dependencies, runs each test, and shows a summary
table at the end.

### Locally (CPU is OK for most tests; some need a GPU)

```bash
cd research/theses_validation
pip install -r requirements.txt
python run_all.py --quick     # fast smoke (~10-20 min on CPU)
python run_all.py --full      # full suite (~2-4 h on T4, much slower CPU)
python run_all.py --only test_decoupled_ste_vs_ste
```

Results land in `research/theses_validation/results/<timestamp>/<test>.json`.

### Single test

```bash
python -m tests.quantization --test decoupled_ste_vs_ste --seed 42 --steps 1000
```

---

## Reading the results

Each test prints a results table.  Example:

```
=== test_decoupled_ste_vs_ste ===
Setup: same Mamba2 backbone, ternary linear heads, batch=32, seq=128, AdamW lr=3e-4
Tokens trained: ~10M
                            STE      Decoupled-STE   delta
Final train loss          2.342         2.183       -0.159 (-6.8%)
Held-out PPL              14.6          13.1        -1.5  (-10.3%)
Loss spikes (>0.5 σ)         8             3        -5
Wall-time (s)              412           421        +9   (+2.2%)
Tokens / s                24300         23800       -2.1%
```

**Interpreting the numbers:**

- **Final train loss / Held-out PPL**: lower is better.  >5% improvement
  with same seed across 3 runs = real effect.  Single-run differences <2%
  are noise.
- **Loss spikes**: count of training steps where loss > running mean +
  0.5σ.  Lower means more stable training.  This is the headline metric
  for stability claims (STE oscillations, etc.).
- **Wall-time / tok/s**: should be within ±10% for fair-architecture
  changes.  Big gaps mean the variant has overhead — note it.

---

## File layout

```
research/theses_validation/
├── README.md                       (this file)
├── requirements.txt
├── run_all.py                      (orchestrator)
├── speculative_claims.md           (untestable claims, documented)
├── .gitignore                      (excludes results/, __pycache__/)
├── common/
│   ├── __init__.py
│   ├── data.py                     (WikiText loader, BPE/char tokenizers)
│   ├── training.py                 (training loop + spike detection)
│   ├── metrics.py                  (PPL, spike count, throughput)
│   └── reference_models.py         (Mamba2 baseline, simple Transformer)
└── tests/
    ├── __init__.py
    ├── quantization.py             (4 tests: STE, Decoupled-STE, HGF, Continual-QAT, DDT)
    ├── sequence_models.py          (2 tests: LrcSSM vs Mamba2, CfC vs RNN)
    ├── kan_variants.py             (3 tests: KAN, SKAN, KANtize)
    ├── nca_pretrain.py             (1 test: NCA pretrain transfer)
    ├── rl_grpo.py                  (1 test: GRPO vs PPO)
    └── search_mcts.py              (1 test: AB-MCTS vs UCT)
```

Plus `colab/theses_validation.ipynb` for the Colab driver.

---

## Notes for the reviewer

- This module deliberately does NOT use the NSOS C++ engine.  Comparing
  a thesis claim by reimplementing it inside NSOS would risk
  contaminating any NSOS perf measurement with research code.  Keeping
  PyTorch + isolated harness preserves both NSOS hygiene and test
  rigor.

- The harness intentionally uses small models (1-10M params) so each
  test runs in minutes, not hours.  Claims that only manifest at
  billion-parameter scale can't be validated here — those are flagged
  in `speculative_claims.md` with concrete scale-required tests.

- If you find a result that contradicts a thesis claim and want to dig,
  re-run with multiple seeds (`--seed 1 --seed 2 --seed 3`) before
  drawing conclusions.  Single-seed runs can be misleading.
