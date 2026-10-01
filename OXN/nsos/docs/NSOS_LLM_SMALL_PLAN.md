# NSOS LLM Small Plan

## Objective

Turn NSOS into a small specialized LLM that is:

- strong on structured reasoning, code, circuits, and memory
- measurable on honest held-out benchmarks
- efficient enough for edge-oriented deployment
- ready to move from float/QAT training into truly ternary inference

## Target Model Profiles

### Production Direction

- `d_model`: `256` or `384`
- `layers`: `6` to `10`
- `tokenizer`: compact BPE in the `8k` to `16k` range
- focus mix:
  - Portuguese + English technical text
  - code
  - circuits / DSLs
  - memory / agent-style recall

### Recommended First Serious Model

- `8 layers`
- `d_model = 320`
- `tokenizer = 12k`
- sequence length `160` to `320`

This is the first profile that can plausibly become a real edge-specialized LLM for NSOS.

## Workstreams

### 1. Real LLM Generalization

Goal:

- move from smoke success to held-out generalization

**Status (2026-07): held-out generalization DEMONSTRATED at smoke scale.**
Char-level LM on WikiText-2 (922k chars, vocab 90), local NSOS-only run on a
GTX 1050: **VAL perplexity on held-out text (unseen 10%) = 4.46 (pure Mamba 5L)
/ 4.86 (Mamba+attention hybrid 4L)** vs uniform baseline 90, with train≈val
(train ppl ~4.2-4.6) — i.e. the model predicts unseen text nearly as well as
training text, so it learned transferable structure, NOT memorization.  This
plus synthetic length-extrapolation (MQAR trained n_kv=8 → tested n_kv=16;
selective-copy field-length extrapolation) is genuine generalization
**in-distribution + length, at small scale**.  Still OPEN (do not overclaim):
cross-domain / other-language, large-scale, and reasoning generalization — those
are the remaining work below.  (Earlier notes that framed generalization as
unproven / "only memorizes" are superseded by this result.)

Implementation order:

1. lock a staged curriculum
2. train a compact tokenizer on the curriculum bundle
3. train phase-by-phase with saved checkpoints and metrics
4. evaluate exact-match synthetic tasks + held-out text loss
5. add external benchmarks after the local harness is stable

Acceptance criteria:

- exact-match accuracy improves across algorithms / circuits / memory
  (✅ met at smoke scale: MQAR n_kv=8 recall 0.86-0.97; selective-copy 0.82-0.995)
- held-out text loss decreases
  (✅ met: WikiText val ppl 4.46-4.86 vs baseline 90, train≈val)
- the model can answer unseen prompts, not only memorized sequence continuations
  (✅ met at smoke scale: held-out val ppl ≈ train ppl = generalizes, not memorizes)

### 2. Final Edge Inference Speed

Goal:

- make the runtime behave like an edge model, not only a research runtime

Implementation order:

1. stop repacking BitLinear weights on every forward pass
2. split prompt processing from token-by-token decode
3. add incremental decoding caches for generation
4. tune CPU SIMD path and packed-weight path
5. add dedicated packed ternary generation kernels for the frozen model shapes

Acceptance criteria:

- prompt throughput and decode throughput measured separately
- generation no longer recomputes the full prompt every token
- tokens/sec improves materially on the same hardware

### 3. Truly Ternary Footprint

Goal:

- make inference memory reflect the ternary promise in practice

Implementation order:

1. define an NSOS packed-weight export format
2. keep float weights only for train/checkpoint mode
3. load packed-only weights in inference mode
4. quantize embeddings separately with a safer format
5. make model packs capable of storing both train and edge variants

Acceptance criteria:

- inference build can load without duplicating fp32 weights
- embedding strategy is explicit and measured
- reported RAM aligns with packed ternary expectations

### 4. Reproducibility and Serious Benchmarks

Goal:

- make training and evaluation repeatable enough for engineering decisions

Implementation order:

1. manifest every curriculum bundle with hashes
2. save tokenizer artifact + config with every run
3. save per-phase metrics and checkpoints
4. wire deterministic seeds more deeply into tensor/model init
5. add external benchmarks once the internal harness is stable

Acceptance criteria:

- repeated runs can be compared by config + artifact hash
- benchmark outputs live in versioned JSON
- regressions are visible instead of anecdotal

## Curriculum

### Phase 1: Algorithmic Discipline

- copy / reverse / count / parity
- binary addition
- integer comparison

Reason:

- gives the model structured next-token discipline and exactness

### Phase 2: Structured Domain

- circuits
- truth tables
- mini DSL execution
- structured logs
- parsing / transformation

Reason:

- this is where NSOS has identity

### Phase 3: Curated LLM Pretraining

- technical docs
- NSOS/OxtaMem project text
- code files and implementation snippets
- bilingual PT/EN technical passages

Reason:

- turns the model into a real technical language model instead of only a synthetic solver

### Phase 4: Instruction Tuning

- summarize
- explain code
- convert formats
- translate technical sentences
- follow explicit formatting instructions

Reason:

- gives the model user-facing behavior

### Phase 5: Verifier-Guided Tasks

- exact math
- boolean formulae
- code output
- exact-answer symbolic tasks

Reason:

- sharpens precision instead of only fluency

### Phase 6: Memory

- multi-turn fact storage
- fact recall
- session-style transcripts

Reason:

- prepares the model for MemorySystem / OxtaMem integration

## BitNet-main Port Plan

The `BitNet-main` repository is most useful as an inference and export reference, not as a full NSOS replacement.

### Port First

1. Weight packing layout and permutation logic
   - reference:
     - `gpu/pack_weight.py`
     - `src/ggml-bitnet-mad.cpp`
2. Packed ternary / 2-bit export path
   - add an NSOS export mode that writes packed linear weights plus scales
3. Repack-once behavior
   - move packing out of the hot forward path in `BitLinear`
4. Embedding quantization strategy
   - follow the lesson from BitNet: do not force embeddings to the most extreme ternary path
5. Tunable CPU kernel config
   - expose block sizes / parallel strategy for the packed backend
6. Optional fixed-shape CUDA kernels
   - only after NSOS target model shapes are frozen

### Port Later

- codegen for tuned kernels
- benchmark harness for prompt/decode separation
- release conversion tooling

### Avoid Porting Blindly

- the full llama.cpp integration layer
- shape-hardcoded kernels before NSOS model shapes are frozen
- model-specific assumptions that do not fit Jamba/Mamba/TTT scheduling

## Milestones

### Milestone A

- curriculum bundle exists
- tokenizer bundle exists
- staged training script runs
- held-out metrics are written

### Milestone B

- `small` profile trains end-to-end
- held-out exact match is materially above random / memorization baselines
- prompt/decode latency is measured

### Milestone C

- packed ternary export exists
- inference can load packed-only linear weights
- RAM and tokens/sec are reported for edge mode

### Milestone D

- external benchmark suite is stable
- reproducible runs are tracked
- NSOS has one honest small-model report card
