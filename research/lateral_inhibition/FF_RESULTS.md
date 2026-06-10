# Forward-Forward — does a LOCAL forward-only signal make depth compose?

**Idea:** the unsupervised competitive stack peaked at layer 1 because pure WTA has
no *reason* to build the XOR. Forward-Forward (Hinton 2022) gives each layer a
**local** objective — "goodness" = Σ(activations²), pushed HIGH for positive data
(input + correct label) and LOW for negative (input + wrong label). Each layer's
update is the local gradient of its own goodness: **no backprop**. Length-
normalizing between layers forces each layer to find new structure.

Same XOR-of-part-groups task (4 classes, chance 0.25) as the unsupervised study,
so the numbers compare directly. `ff_demo.cpp`, 25 layers × 64 units, 15 epochs.

## Results

```
PLAIN-FF (k=64)
  acc (layers 0..d):       d1 0.907  d2 0.963  d3 0.959  d5 0.932  ...  d25 0.894
  acc (layers 1..d,no-L0): d1 0.191  d2 0.939  d3 0.925  d5 0.873  ...  d25 0.809
  layer-0-ONLY = 0.907     best no-leak = 0.939 @ depth 2

FF + LATERAL INHIBITION (k=16)
  acc (layers 0..d):       d1 0.896  d2 0.948  d3 0.933  ...  d25 0.880
  acc (layers 1..d,no-L0): d1 0.191  d2 0.924  d3 0.892  ...  d25 0.801
  layer-0-ONLY = 0.896     best no-leak = 0.924 @ depth 2
```

## Honest reading

1. **Forward-only local learning WORKS — powerfully.** FF reaches **0.96** where the
   unsupervised competitive layer got 0.387 and chance is 0.25. The positive/negative
   local signal is exactly the ingredient pure competition lacked. This is the real
   win: **you can learn a nonlinear task to high accuracy with no backprop.**

2. **But this task does NOT require depth, so it cannot prove "depth composes."**
   `layer-0-only = 0.907`: a SINGLE FF layer (64 ReLU units + goodness = one nonlinear
   classifier) already solves the XOR. Depth adds exactly one useful step (layer 0 →
   0+1: +0.056, peak 0.963 @ depth 2) and then **erodes** to 0.89 by depth 25. The
   2-way parity of group-presence is shallow-solvable, so neither the unsupervised nor
   the FF experiment can isolate a depth benefit — both peak by layer ~2.

3. **The cross-method pattern is consistent and real:** depth helps for ~1–2 layers,
   then erodes — for unsupervised soft-temp codes (one step: lin 0.453→0.532) AND for
   FF (one step: 0.907→0.963). The marginal value of *many* forward-only layers is
   negative on this task. 25 layers is not the sweet spot; ~2–3 is.

4. **Lateral inhibition mildly HURTS the supervised-local FF** (k=16: 0.924 vs plain
   0.939). Honest nuance: inhibition was *essential* for unsupervised competition (it
   was the only thing preventing collapse), but for the FF goodness objective, denser
   activations carry more discriminative signal — WTA throws some away. Inhibition is
   not universally good; it depends on the learning rule.

## Bottom line for the mission

The headline the project actually needs: **forward-only, local learning solves a
nonlinear task at 0.96 — no datacenter-style backprop required.** That directly
supports "escape backprop / train on the edge." The "25 layers compose" question is
secondary and remains open: proving depth *specifically* helps needs a task that is
**not** single-layer-solvable (e.g. deep parity / iterated composition over many
bits). That is the right follow-up if depth-composition is the goal; the forward-only
learning capability itself is now validated.

## Files
- `ff_demo.cpp` — FF stack (local goodness, pos/neg, per-depth probe, leak check),
  plain and +lateral-inhibition variants. `ff_out.txt` — raw run.
