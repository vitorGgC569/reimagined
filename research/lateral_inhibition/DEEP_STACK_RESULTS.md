# 25-layer forward-only stack — does depth compose? (honest results)

**Question (Oxta's):** stack 25 lateral-inhibition layers, learning forward-only/
local (no backprop). Do features **compose hierarchically** with depth? And can a
per-neuron/per-layer **memory ("Oxta-mem")** stop the collapse?

**Task (built to make depth *matter*):** super-class = XOR of which part-GROUPS
are present (`label = 2·(bG0⊕bG1) + (bG2⊕bG3)`), 4 classes, chance 0.25. This is
**not linearly separable** from the raw part-sum, so a shallow readout *must*
plateau and only real composition can climb. 12000 train / 3000 test, dim 64.

**Stack:** 25×64 neurons, graded k-sparse codes (k-taper 4/3/2), local homeostasis
(conscience win-frequency bias + dead-unit reinit + winner decorrelation). All
probes (nearest-centroid + ridge-linear) are **measurement-only** — closed-form on
frozen codes, they never backprop into the stack.

## Headline: depth does NOT compose

Per-depth NCC: raw 0.307 → **layer 1 = 0.387 (peak)** → erodes to ~0.297 by layer 5
and **flatlines to layer 25**. Linear probe: raw 0.355 → 0.478 peak → ~0.318 flat.
By layer 5, **~80% of neurons are dead** (entropy 0.96→0.61) despite homeostasis —
the inter-layer code collapsed to ~12 distinct points, so deeper layers re-quantize
the same thing. Peak→final drop ≈ 0.09 (≈7.4σ at N=3000): the layer-1 gain is real,
the erosion is real.

### Controls (all confirm the negative is not an artifact)
| control | NCC | reading |
|---|---|---|
| inhibition-OFF stack (k=all) | 0.199 | below chance — inhibition is necessary for any learning |
| random-frozen deep stack | 0.304 | trained depth (0.297) is **no better than random** |
| shuffled-layer-order | 0.278 vs 0.297 | layers ~interchangeable → no real hierarchy built |
| **width-scaled 1 layer, m=1600** | **0.438 / 0.599 (lin)** | same neuron budget, ONE wide layer **crushes** 25 deep |

## Rescue experiments (can the bottleneck be fixed, still forward-only?)

Driven by an adversarial audit (which also caught + we **fixed** an LR confound:
the wide baseline now uses the matched lr — width still wins, so the confound was
not the cause).

| variant | best NCC | best LIN | depth help? |
|---|---|---|---|
| A. code-only (LR-matched) | 0.384 | 0.473 | **no** (= layer 1) |
| B. **skip-concat** (every layer also sees raw = Oxta-mem of input) | 0.385 | 0.475 | **no** |
| C. soft-temp **dense** code (τ=0.4) | 0.366 | **0.532 @ depth 2** | small (+0.079 lin at d2, then erodes) |
| Oxta-mem (concat all 25 layer codes, 1600-dim) | — | 0.527 | preserves info, capped by redundancy |
| width-scaled m=1600 (matched lr=0.074) | 0.438 | **0.599** | (reference ceiling) |

## Honest verdict

1. **Width > depth, robustly.** A single wide competitive layer beats the 25-layer
   stack at matched learning rate (0.599 vs 0.532 linear). For pure forward-only
   competition, **capacity (diverse prototypes), not depth, is the lever.**
2. **Pure depth does not compose** — confirmed even when the information bottleneck
   is removed (skip-concat hands every layer the raw input, still no gain). Without
   a signal telling a layer the XOR matters, it just re-clusters; it never forms the
   new conjunction.
3. **Denser codes buy exactly one composition step** (soft-temp: layer-2 linear
   0.453→0.532), then collapse resumes. So the k-sparse bottleneck was *part* of the
   story, but not the root cause.
4. **Oxta-mem (memory) is validated as PRESERVATION, not creation.** Remembering
   every layer's contribution recovers some lost info (0.475→0.527) and is essential
   for continual/edge adaptation — but it is **capped by redundancy** (the deep
   memories are copies of the same collapsed code) and does not, by itself, make
   depth compose. *Memory preserves information; it does not create a reason to
   compute a function.*

## What this points to (the real next bet)

The missing ingredient is a **local, forward-only learning signal** that gives a
deep layer a *reason* to form new conjunctions — e.g. a per-layer predictive /
reconstruction target, Hebbian temporal binding, or a forward-forward "goodness"
objective. Memory (Oxta-mem) + a local signal + diversity is the combination worth
testing next. Memory alone, or depth alone, is not enough — and that is a real,
useful, honest result.

## Files
- `lateral_inhibition.h` — layer + deep-stack additions (`step_code`, `encode`,
  conscience/reinit/decorrelate, soft-temp code).
- `stack_demo.cpp` — 25-layer stack, 8 baselines, per-depth metrics, Oxta-mem readout.
- `rescue_demo.cpp` — LR-matched + skip-concat + soft-temp + width LR-sweep.
- `stack_out.txt`, `stack_out2.txt`, `rescue_out.txt` — raw runs.
