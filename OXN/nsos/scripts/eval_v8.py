"""
Evaluate NSOS v8 model on holdout benchmark tasks.
Uses the same format as training (nsos_curriculum_lib.format_supervised_text).
Runs fast: max_tokens=4 per task (enough for single-token answers).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# UTF-8 safe stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WORKTREE  = Path(__file__).resolve().parent.parent.parent.parent
MAIN_REPO = Path("C:/Users/Oxta/Desktop/reimagined-main")
BUILD_DIR = MAIN_REPO / "OXN/nsos/build-mvp/Release"
SCRIPTS   = WORKTREE / "OXN/nsos/scripts"
MODEL_DIR = SCRIPTS / "live_distill_v8"
MODEL_BIN = MODEL_DIR / "final_model.bin"
MODEL_CFG = MODEL_DIR / "effective_model_config.json"
MICRO     = WORKTREE / "OXN/nsos/benchmarks/nsos_micro_suite.jsonl"
EVAL      = WORKTREE / "OXN/nsos/benchmarks/nsos_eval_suite.jsonl"

import os
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))
if os.name == "nt":
    try:
        os.add_dll_directory(str(BUILD_DIR))
    except (AttributeError, OSError) as exc:
        print(
            f"[runtime] DLL directory registration failed for {BUILD_DIR}: {exc}",
            file=sys.stderr,
        )

import nsos_ext as nsos  # type: ignore

# Load model
config = nsos.ModelConfig()
if MODEL_CFG.exists():
    for k, v in json.loads(MODEL_CFG.read_text("utf-8")).items():
        if hasattr(config, k):
            setattr(config, k, v)
config.use_cuda = False

print(f"Loading model: {MODEL_BIN.name} ...")
t0 = time.time()
engine = nsos.InferenceEngine()
if not engine.load_model(str(MODEL_BIN), config):
    print("ERROR: failed to load model", file=sys.stderr)
    sys.exit(1)
print(f"Loaded in {time.time()-t0:.1f}s\n")

opts = nsos.GenerationOptions()
opts.max_context_tokens = 512
opts.max_tokens         = 4   # short answers only (parity/circuit/reverse/recall)
opts.temperature        = 0.0  # greedy — deterministic, picks the highest-prob token
opts.top_p              = 1.0
opts.top_k              = 1
opts.stream             = False


def build_prompt(kind: str, prompt: str) -> str:
    return f"<|task:{kind}|>\nPrompt:\n{prompt}\nAnswer:\n"


def run_suite(name: str, path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    correct = 0
    first_tok_match = 0
    results = []

    print(f"{'='*60}")
    print(f"Suite: {name}  ({len(rows)} tasks)")
    print(f"{'='*60}")

    for i, row in enumerate(rows, 1):
        kind   = row["kind"]
        prompt = row["prompt"]
        gold   = row["answer"].strip()

        raw_prompt = build_prompt(kind, prompt)
        t_start = time.time()
        output = engine.generate_ex(raw_prompt, opts)
        elapsed = time.time() - t_start

        pred = output.replace("<|endoftext|>", "").strip()
        exact = (pred == gold)
        first = (pred[:len(gold)] == gold or gold.startswith(pred[:1]))

        correct += int(exact)
        first_tok_match += int(first)

        status = "✓" if exact else ("~" if first else "✗")
        results.append({
            "id": row.get("id", f"{i}"),
            "kind": kind,
            "gold": gold,
            "pred": pred,
            "exact": exact,
            "first": first,
            "elapsed_s": round(elapsed, 1),
        })

        short_prompt = prompt[:50].replace("\n", " ")
        print(f"  [{status}] {kind:<15} | gold={gold!r:<20} pred={pred!r:<20} | {elapsed:.0f}s | {short_prompt}")

    n = len(rows)
    print(f"\nExact match:      {correct}/{n}  ({100*correct/n:.0f}%)")
    print(f"First-tok match:  {first_tok_match}/{n}  ({100*first_tok_match/n:.0f}%)")
    return {"suite": name, "n": n, "exact": correct, "first": first_tok_match, "results": results}


if __name__ == "__main__":
    all_results = []
    for suite_name, suite_path in [("micro_suite", MICRO), ("eval_suite", EVAL)]:
        r = run_suite(suite_name, suite_path)
        all_results.append(r)
        print()

    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    for r in all_results:
        n = r["n"]
        print(f"  {r['suite']:<15}  exact={r['exact']}/{n}  first={r['first']}/{n}")
    print()
    print("NOTE: temp=0.0 (greedy), max_tokens=4, no chat — pure task eval")
