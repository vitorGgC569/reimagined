# MNIST — the real-task test (what survived, what broke)

Forward-only (Forward-Forward + Oxta-mem dual-trace) vs backprop (Adam MLP) on
real MNIST. Matched architecture (width 256, depth 3). `mnist_showdown.cpp`.
This test was run specifically to see whether the toy-task wins hold at scale.
**Two of them did not** — recorded here honestly.

## Part 1 — classification + training time (15000 train, 10000 test)

| method | test acc | time/epoch | total |
|---|---|---|---|
| backprop | **0.960** | 95.4s | 1430s (15 ep) |
| forward-forward | 0.882 | **12.5s** | 313s (25 ep) |

- **Accuracy:** backprop clearly ahead on real data (0.96 vs 0.88). FF is untuned
  (Hinton's tuned FF reaches ~0.986); the ~8-pt gap is the honest forward-only cost.
- **Speed:** FF ran 7.6× faster *per epoch*. IMPORTANT — this is NOT "forward beats
  backward" in FLOPs (those are comparable; FF does slightly more). It is an
  **optimizer + overhead** effect: backprop's Adam does a `sqrt` + divides *per weight*
  plus the backward matmul and per-sample allocations; FF uses a cheap local
  multiply-add update and no backward machinery. So forward-only IS faster in
  wall-clock here, but because of the cheap local update, not the forward/backward
  asymmetry — and it reaches a lower accuracy. To match backprop's 0.96, FF would need
  more width/epochs (the width tax), eating into the time advantage.

## Part 2 — continual (permuted-MNIST), matched 12 epochs (8000 train)

Learn MNIST → learn pixel-permuted MNIST → retain original. (retain = MNIST test acc after B)

| learner | A (after A) | **MNIST retained** |
|---|---|---|
| backprop | 0.956 | **0.780** |
| FF (no memory) | 0.818 | 0.706 |
| FF + Oxta-mem (g=0.03, λ=0.04, weak) | 0.816 | 0.633 |
| FF + Oxta-mem (g=0.05, λ=0.08, medium) | 0.815 | 0.605 |
| FF + Oxta-mem (g=0.02, λ=0.15, strong) | 0.404 | 0.303 |

**Two corrections to the earlier (toy) story — both broke on real data:**

1. **Oxta-mem dual-trace does NOT transfer.** On MNIST it *hurts* retention at every
   anchor strength, monotonically worse as the anchor tightens (best case is λ→0 =
   no memory). The toy win (0.67 vs 0.52 on 4 clusters) was a toy artifact. The
   mechanism over-regularizes complex real representations and does not help here.

2. **"Forward-only forgets less than backprop" is NOT robust.** The earlier MNIST run
   showed FF (0.68) > backprop (0.55), but that used UNEQUAL training (FF 20 epochs &
   more data vs backprop 12). With epochs matched, **backprop retains MORE in absolute
   terms (0.780 vs 0.706)**. FF loses a slightly smaller *fraction* (14% vs 18%), so it
   is marginally more forgetting-*resistant* relatively — but on absolute task-A
   accuracy after B, backprop wins. The strong continual claim does not hold.

## What survived the real task

- Backprop is **more accurate** on real data (0.96 vs 0.88).
- Forward-only is **faster in wall-clock** (cheap local updates, no backward) — but to
  a lower accuracy, and via optimizer/overhead, not FLOPs.
- The **depth-composes-forward-only** result and the **width tax (~2–5× when
  parameter-starved, → ~1× with capacity)** stand — but those were measured on parity,
  a toy task; they have not been re-confirmed on MNIST.
- The genuine, robust advantage of forward-only remains **training memory and hardware
  locality** (no stored activations, no backward lock) — NOT speed, NOT less forgetting.

## Bottom line (honest)

The real-task test did its job: it cut the two most exciting toy claims. Oxta-mem did
not scale; the continual advantage was an artifact of unequal comparison. This is
consistent with the earlier assessment — clean replication/engineering, not a
discovery. The forward-only direction is real and worth pursuing for *edge/memory*
reasons, but it is not, on this evidence, better than backprop at accuracy or at
not-forgetting.

## Files
`mnist_showdown.cpp` (loader + backprop/FF/FF-mem, timing + continual sweep),
`mnist/` (IDX data), `mnist_out.txt` / `mnist2_out.txt` (raw runs).
