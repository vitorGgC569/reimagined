# Showdown — forward-only learning vs backprop (depth + continual)

Two head-to-head experiments pitting forward-only/local learning against a real
backprop baseline (Adam), to answer the two questions the toy studies left open:
(B) does depth compose with a *local* signal? (A) does the combined learner
(Forward-Forward + Oxta-mem dual-trace) hold up — and where does it beat backprop?

## B. Parity-8 — does depth compose forward-only? (`parity_showdown.cpp`)

Parity-N is the classic depth-requiring task: at narrow width a single hidden layer
cannot represent it; depth can. Test accuracy (chance 0.5, noise 0.30 on ±1 bits):

| width | backprop shallow | backprop DEEP | FF shallow | FF DEEP |
|---|---|---|---|---|
| 6  | 0.626 | 0.750 | 0.408 | 0.485 |
| 16 | 0.689 | 0.849 | 0.539 | 0.713 |
| 32 | 0.842 | 0.899 | 0.601 | **0.860** |

- **Depth composes for BOTH methods** (DEEP ≫ shallow everywhere). The forward-only
  FF gets an even larger depth boost than backprop (width 32: 0.601 → 0.860, +0.26).
- **FF nearly matches backprop with enough width** — at width 32, FF (forward-only,
  no backprop) 0.860 vs backprop 0.899: a ~4-point gap on a genuinely depth-requiring
  task. That is a strong result for "you don't need backprop."
- **FF is less parameter-efficient** — at width 6 it collapses to chance while
  backprop deep still learns (0.750). FF needs more width to express the same function.
- This settles the earlier open question: the unsupervised stack's "depth doesn't
  compose" was specific to *unsupervised competition on a shallow-solvable task*. With
  a local LEARNING SIGNAL and a task that needs depth, **depth composes forward-only.**

## A. Continual learning — combined FF + Oxta-mem vs backprop (`continual_showdown.cpp`)

Learn task A (4 clusters) → learn task B (4 disjoint clusters, same net) → measure how
much of A survives. Accuracy (chance 0.25):

| learner | A (after A) | B (after B) | **A retained (after B)** |
|---|---|---|---|
| backprop MLP (gold standard) | 1.000 | 1.000 | 0.404 ← catastrophic forgetting |
| FF only (no memory) | 0.998 | 0.993 | 0.525 |
| **FF + Oxta-mem (combined)** | 0.998 | 0.990 | 0.644 |
| **FF + Oxta-mem (strong anchor)** | 0.996 | 0.984 | **0.673** |

- On **continual** learning the combined forward-only learner **beats backprop**:
  retains 0.673 of A vs backprop's 0.404 (backprop forgets A almost to chance).
- Both pieces contribute: FF's local learning already forgets less than backprop
  (0.525 vs 0.404); the dual-trace memory adds more on top (0.525 → 0.673), at a
  tiny plasticity cost on B (0.990 → 0.984).
- **Fair caveat:** this is vs *plain* backprop. Backprop has its own continual-learning
  fixes (EWC, replay) that would narrow the gap; a fully fair fight is FF+mem vs
  backprop+EWC. Still toy scale (4 clusters, 16-dim).

## The honest synthesis

- **Single hard task:** backprop is still the gold standard (0.90 vs FF 0.86 on
  parity-8), and more parameter-efficient.
- **Depth:** composes forward-only — definitively (FF 0.60 → 0.86 with depth).
- **Continual / edge:** the forward-only local+memory learner WINS (less forgetting),
  which is exactly the on-device "learn forever, don't forget" regime the project targets.
- **Status:** the building blocks are known (Forward-Forward = Hinton 2022; dual-trace
  = fast/slow weights & EWC lineage). What is validated here is the end-to-end synthesis
  with controls and a fair backprop baseline — clean engineering, not a new algorithm.

## Width metric — the forward-only "exchange rate" (`width_metric.cpp`)

Sweep per-layer width (depth 3), median of 3 seeds, leaky-ReLU backprop (the
plain-ReLU MLP collapsed to chance at width 32/64 on parity — dead-unit failure;
leaky-ReLU + median fixed it). Params per width are ~identical for BP and FF, so
width ≈ parameter cost.

[B] Parity-8 — FF width needed to MATCH backprop@width:

| match backprop@ | acc | FF needs width | width cost | param cost |
|---|---|---|---|---|
| 16 | 0.876 | ~39 | 2.4× | ~4.9× |
| 24 | 0.900 | ~43 | 1.8× | ~2.9× |
| 32 | 0.889 | ~41 | 1.3× | ~1.6× |
| 48 | 0.915 | ~45 | 0.95× | ~0.9× |
| 64 | 0.926 | ~47 | 0.74× | **0.56×** |

- The forward-only width tax is **~2.4× width (~5× params) when parameter-starved**,
  and **shrinks to ~1× at moderate capacity, then below 1× (FF more efficient)** at
  high capacity — because FF keeps scaling (0.96 @ width 64) while backprop saturates
  (0.93) on parity. Not a fixed tax; it vanishes with scale (parity-specific, toy).

[A] Continual — does width fix backprop's forgetting? (retain A after B)

| width | backprop | FF | FF+mem |
|---|---|---|---|
| 16 | 0.477 | 0.489 | 0.610 |
| 32 | 0.454 | 0.486 | 0.595 |
| 64 | 0.381 | 0.525 | 0.673 |
| 128 | 0.420 | 0.632 | 0.705 |

- **Width does NOT fix backprop's catastrophic forgetting** (flat ~0.4–0.48, no upward
  trend). FF+mem retention climbs with width and beats backprop at EVERY width. There
  is **no width at which backprop matches FF+mem retention** — a mechanism gap, not a
  capacity gap.

## Files
`parity_showdown.cpp` (backprop-Adam vs FF, depth × width), `continual_showdown.cpp`
(backprop vs FF vs FF+Oxta-mem, continual A→B), `width_metric.cpp` (width sweep +
iso-accuracy exchange rate). Raw runs alongside.
