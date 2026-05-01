# NSOS Validation Status

## Current Position

NSOS is already validated as:

- a buildable neural runtime
- a trainable small-model stack
- a pack/load product surface
- an HTTP/CLI/Python serving surface
- a CPU-authoritative packed ternary inference path
- a memory-aware system with native causal store plus OxtaMem FFI

Official profile truth:

- `mamba_small` is the production small-profile baseline
- `small` is a legacy alias for `mamba_small`
- `hybrid_pilot` validates attention + Mamba without TTT
- `hybrid_small` validates attention + sparse MoE + Mamba
- `TTT` is research-only until snapshot/replay and serving parity are formally closed

NSOS is not yet fully validated as:

- a small LLM with strong held-out generalization
- a closed CPU/GPU packed ternary runtime story
- a benchmarked model family with stable external scorecards

## What Is Solid

- `InferenceEngine`, model packs, tokenizer packs, CLI, HTTP API
- packed linear export and load
- streaming decode for the Mamba-only / no-attention profiles
- curriculum build + training + per-run artifacts
- fast supervised probe loop for quick iteration
- external benchmark harness with held-out curriculum + Wikitext-2 + fixed NSOS eval suite

## Main Open Gaps

### 1. Real Generalization

Blocked by:

- low held-out exact match on `pilot`
- weak teacher-token accuracy on external held-out evaluation
- high perplexity on Wikitext-2

What needs to happen:

- replay-aware curriculum must materially improve held-out answer metrics
- `pilot` needs to show sustained growth in masked answer accuracy before `small`
- `small` must be run only after fast probes show clear upward signal

### 2. Packed Runtime Story

Current truth:

- packed ternary runtime is validated on CPU
- float/GPU runtime is usable
- packed/GPU is not yet a native final path

Official support matrix:

- CPU float: validated
- CPU packed: validated
- GPU float: validated hot path
- GPU packed: not yet native
- TTT: research-only

What needs to happen:

- either a native packed CUDA kernel path
- or an explicit product decision that edge mode is CPU-first and GPU is float-first

### 3. External Benchmarks

Current truth:

- we have a first honest external harness
- we do not yet have a strong report card

What needs to happen:

- repeated runs with the same suite
- `pilot` and `small` reports stored side-by-side
- fixed acceptance thresholds for exact match, teacher-token accuracy, perplexity, and edge throughput

## Validation Gates

NSOS should be considered validated when all of these are true:

1. `pilot` shows clear held-out improvement on the fixed suite and curriculum-held-out probes
2. `small` finishes training and beats `pilot` on the same scorecard
3. packed edge decode remains materially faster than baseline decode
4. model-pack and edge-pack load/reload stay deterministic
5. the official support matrix is explicit:
   - CPU float
   - CPU packed
   - GPU float
   - GPU packed

Release gates are stricter than research gates:

- research champion can be selected by composite score
- release candidate must pass fixed thresholds for exact accuracy, teacher-token accuracy,
  held-out loss and generation behavior on the global suite

## Operational Rule

Use fast probes to choose direction.

Do not spend long runs unless:

- masked answer loss is going down
- first-token accuracy is moving
- teacher-token accuracy is moving

The fast probe loop is the decision layer.
The long `pilot` / `small` runs are confirmation layers.
