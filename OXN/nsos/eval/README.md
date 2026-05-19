# NSOS Eval Suite

The 5-benchmark scorecard that gives NSOS its first publishable numbers.

## What it measures

| Benchmark | What | Why | Random | Llama-3-8B ref |
|---|---|---|---|---|
| `wikitext2_ppl` | Sliding-window PPL on WikiText-2 raw test | LM quality baseline | — | ~6.5 |
| `hellaswag` | Commonsense sentence completion | Reasoning surface | 25% | 79% |
| `arc_easy` | Elementary science MC | Knowledge surface | 25% | 92% |
| `mmlu_stem` | STEM multiple-choice (100 stratified) | World knowledge | 25% | ~50% |
| `humaneval_light` | Python code completion pass@1 | Code ability | 0% | ~33% |

## Run it

```bash
# Smoke test the framework (no NSOS pack needed)
python OXN/nsos/scripts/run_scorecard.py --adapter dummy --quick

# Score a trained pack
python OXN/nsos/scripts/run_scorecard.py \
    --pack /path/to/final_model.bin \
    --build-dir /path/to/build \
    --quick

# Full eval (slower; production-quality numbers)
python OXN/nsos/scripts/run_scorecard.py --pack ... --build-dir ... --full

# Multi-seed for honest error bars on MMLU
python OXN/nsos/scripts/run_scorecard.py --pack ... --build-dir ... --seeds 1 2 3
```

## Output

Every run lands in `OXN/nsos/artifacts/scorecard/<timestamp>/`:

- `scorecard.json` — machine-readable, every metric
- `scorecard.md` — markdown table dropped into `OXN/nsos/docs/SCORECARD.md`
- `per_benchmark/<name>.json` — detailed dumps (per-problem for HumanEval,
  per-subject for MMLU, etc.)

The latest run's markdown is also copied to `OXN/nsos/docs/SCORECARD.md` for
quick reference.

## Architecture

```
eval/
├── adapters/        # ModelAdapter abstraction
│   ├── base.py
│   ├── nsos_adapter.py     # Loads NSOS .bin packs
│   └── dummy_adapter.py    # Uniform random model for smoke tests
├── benchmarks/      # One module per benchmark
│   ├── hellaswag.py
│   ├── arc.py
│   ├── mmlu.py
│   ├── humaneval.py
│   └── wikitext.py
├── scoring/         # Pure-function helpers
│   ├── multiple_choice.py
│   └── perplexity.py
├── orchestrator.py  # run_scorecard()
└── tests/
    └── test_smoke.py
```

## Design notes

**Adapter pattern.** Every benchmark talks to `ModelAdapter`, which exposes
`tokenize`, `score_tokens(prompt, target) -> logprob`, and `generate(prompt) -> text`.
Adding a new model = implementing one adapter.  Adding a new benchmark =
implementing one benchmark module that uses the adapter.  Zero coupling
between models and benchmarks.

**Honesty over optimism.**  Every benchmark has a `notes` field with the
specific normalization variant + dataset split used so numbers are
comparable to published results.  Each result includes `status`
(`ok` / `skipped` / `error`) so the orchestrator never silently swallows
a broken benchmark.

**Quick vs full.**  `--quick` caps each benchmark at ~200-500 examples so
the full scorecard runs in 5-15 min.  `--full` uses the full splits
(45 min - 2 h depending on adapter speed).  Quick is for iteration;
full is for publishable numbers.

**Multi-seed.**  Only MMLU is seed-sensitive (its 100-question STEM
subset is sampled).  When `--seeds 1 2 3` is passed, MMLU runs 3 times
and we report mean ± stdev.  Other benchmarks use deterministic splits
so a single seed is enough.

## Limits we don't hide

- **PPL is on the adapter's NATIVE tokenizer.**  An NSOS BPE-8192 model
  and a char-level baseline give different absolute PPL numbers.  Compare
  across the same tokenizer family.
- **HumanEval pass@1 with T=0.2 isn't the standard pass@1@T=0.8.**  We
  use lower temperature to make small-model evaluation more stable; for
  competitive numbers against published results, use `temperature=0.8`
  via `extra_kwargs_per_bench`.
- **MMLU 100-Q subset has ±5% sampling noise.**  Multi-seed runs reveal
  this directly — single-seed runs should not be interpreted to 1%
  precision.
- **Adapter capabilities matter.**  An adapter without `can_generate=True`
  (some C++ inference paths may not expose generation) will skip
  HumanEval automatically with a clear status, not fail.
