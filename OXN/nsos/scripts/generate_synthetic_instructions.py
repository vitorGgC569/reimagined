"""LEARN A3 — Synthetic instruction generation via the Anthropic API.

Generates a curated instruction dataset (~50K entries by default)
tailored to the NSOS task vocabulary: summarize, extract_fact,
explain_code, rewrite, translate.  Uses Claude as the teacher to
produce diverse, high-quality (prompt, answer) pairs that the NSOS
40M student can then learn to mimic in supervised fine-tuning.

This is one half of the v11 "knowledge distillation" strategy:
  1. Generate synthetic instruction data with Claude (this script).
  2. Train NSOS on that data with cross-entropy (existing trainer).
A future enhancement (distill_from_teacher.py) will use top-K
logprobs from Claude as soft labels with KL divergence — that path
requires a separate API integration and is left to the next pass.

Industrial-grade implementation:
  * Token-bucket rate limiter (configurable RPM, default 50)
  * Exponential backoff with jitter on 429 / 529 / 5xx
  * Per-row JSONL output with atomic resume (appends only)
  * SHA-256-based deduplication (drops generated rows whose prompt
    text matches a previously-seen prompt within this run OR in the
    resumed jsonl)
  * Cost tracking (input / output tokens billed at the model's
    published per-million rates) and a printed running estimate
  * Hard stop on cumulative spend (--max-spend-usd) so the user
    can't accidentally burn $1000 on a typo
  * Each task type has its own seed-prompt library, and Claude is
    asked to generate N variants per call (batched for efficiency)

CLI:
  python generate_synthetic_instructions.py \\
      --out-dir artifacts/synthetic_instructions \\
      --total 50000 \\
      --tasks summarize,extract_fact,explain_code,rewrite,translate \\
      --model claude-opus-4-5-20251101 \\
      --rpm 50 \\
      --max-spend-usd 50.0

The output JSONL has the same schema as the curriculum bundle rows:
  {"id": "...", "phase": "phase4_instructions", "kind": "summarize",
   "prompt": "...", "answer": "...", "source": "claude_synthetic"}

It can be dropped directly into the v11 build pipeline: copy the
output JSONL to artifacts/real_datasets/ and add a loader entry to
nsos_curriculum_lib.REAL_DATASET_FILES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Anthropic API client — install once with: pip install anthropic
try:
    import anthropic
except ImportError:
    sys.stderr.write(
        "[fatal] anthropic SDK not installed.  Run:\n"
        "    pip install anthropic\n"
        "then re-run this script.\n"
    )
    sys.exit(1)


# ── Task prompts ────────────────────────────────────────────────────────
# For each NSOS task type, we provide a system prompt + a seed prompt
# library.  Claude generates N variants per call; the variants are
# parsed back into (prompt, answer) pairs.

SYSTEM_PROMPT = """You are a data generator producing high-quality training \
examples for a small (40M parameter) hybrid language model.  Each example \
has two parts: a PROMPT (what the user asks) and an ANSWER (what a smart \
assistant should respond with).

Constraints for every example:
  * Both prompt and answer are in English, plain text, no markdown.
  * The answer is SHORT (1-4 sentences max).
  * The answer is FACTUALLY correct and self-contained.
  * The prompt does NOT mention "training data" or "AI model".
  * The prompt and answer together fit in ~400 tokens.

You will receive a task type and a seed.  Generate N distinct examples \
varying topic, register, and difficulty.  Your output MUST be valid JSON \
with no preamble or postscript — see the schema in each request."""


TASK_INSTRUCTIONS = {
    "summarize": (
        "Generate examples where the prompt asks for a one-sentence summary "
        "of a short passage (60-120 words).  The answer is the summary, "
        "one sentence, focused on the main claim."
    ),
    "extract_fact": (
        "Generate examples where the prompt provides a short context (40-80 "
        "words) and asks a factual question whose answer is a span (1-5 "
        "words) from the context.  The answer is that span verbatim."
    ),
    "explain_code": (
        "Generate examples where the prompt provides a small Python snippet "
        "(3-8 lines, no imports) and asks what it does.  The answer is a "
        "one-sentence explanation focused on observable behavior."
    ),
    "rewrite": (
        "Generate examples where the prompt provides a clumsy sentence "
        "(15-30 words, intentionally wordy or awkward) and asks for a "
        "cleaner technical-sounding rewrite.  The answer is the rewritten "
        "sentence, shorter and clearer."
    ),
    "translate": (
        "Generate examples where the prompt provides a short English "
        "sentence (8-20 words) and asks for the Portuguese translation.  "
        "The answer is the translation, natural Brazilian Portuguese."
    ),
}


# Per-model published pricing (USD per million input/output tokens).
# Update when Anthropic changes rates.  Used only for cost tracking.
MODEL_PRICING = {
    # Claude 3.5 / 4 family — values as of 2026-05.  Override via CLI if
    # the published rates change.  Pricing source: anthropic.com/pricing
    "claude-opus-4-5-20251101":   {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
    "claude-sonnet-4-5-20250929": {"input_per_mtok":  3.00, "output_per_mtok": 15.00},
    "claude-3-5-haiku-20241022":  {"input_per_mtok":  0.80, "output_per_mtok":  4.00},
}


@dataclass
class GenStats:
    """Running tally of work + cost across the whole generation run."""
    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    examples_generated: int = 0
    examples_kept: int = 0
    examples_dropped_dup: int = 0
    examples_dropped_invalid: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float = 0.0
    started_at: float = field(default_factory=time.time)

    def print_summary(self) -> None:
        elapsed = max(time.time() - self.started_at, 0.001)
        print()
        print("=" * 64)
        print(f"  Requests: total={self.requests_total} "
              f"succeeded={self.requests_succeeded} "
              f"failed={self.requests_failed}")
        print(f"  Examples: generated={self.examples_generated} "
              f"kept={self.examples_kept} "
              f"dup={self.examples_dropped_dup} "
              f"invalid={self.examples_dropped_invalid}")
        print(f"  Tokens:   input={self.input_tokens:>9,}  "
              f"output={self.output_tokens:>9,}")
        print(f"  Spend:    ${self.spend_usd:.4f}")
        print(f"  Wall:     {elapsed:.1f}s "
              f"({self.examples_kept / elapsed * 60:.1f} kept/min)")
        print("=" * 64)


class RateLimiter:
    """Token bucket limiter — releases one slot per (60/rpm) seconds.

    The Anthropic API enforces both RPM (requests per minute) and TPM
    (tokens per minute).  This limiter handles RPM only; for TPM we
    rely on the API's own 429 backoff handled by `call_with_retry`.
    """

    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.interval = 60.0 / rpm
        self.next_at = 0.0

    def wait(self) -> None:
        now = time.time()
        if now < self.next_at:
            time.sleep(self.next_at - now)
        self.next_at = max(now, self.next_at) + self.interval


def stable_id(prompt: str, answer: str) -> str:
    """Deterministic per-row id used for de-duplication and bundle integration."""
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(answer.encode("utf-8"))
    return h.hexdigest()[:16]


def prompt_fingerprint(prompt: str) -> str:
    """Per-prompt fingerprint for de-duplication.  Lowercase + collapse
    whitespace so trivial paraphrases don't collide."""
    normalized = " ".join(prompt.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def build_request_body(task: str, batch_size: int, seed: int) -> str:
    """Compose the per-call user message asking Claude for N examples
    of `task`.  Uses a fixed JSON schema so parsing is robust."""
    instruction = TASK_INSTRUCTIONS[task]
    return (
        f"Task type: {task}\n"
        f"Batch seed: {seed}\n"
        f"Number of examples to generate: {batch_size}\n\n"
        f"{instruction}\n\n"
        "Return STRICT JSON with no preamble or postscript.  Schema:\n"
        '{"examples": [{"prompt": "...", "answer": "..."}, ...]}\n'
        f"The array MUST have exactly {batch_size} entries.  Vary the "
        "topic, register, and difficulty across the entries."
    )


def call_with_retry(client: anthropic.Anthropic, model: str,
                    system: str, user: str, max_tokens: int,
                    max_attempts: int = 6) -> Tuple[str, int, int]:
    """Call the messages API with exponential backoff + jitter.

    Returns (text, input_tokens, output_tokens).  Raises on terminal
    failure after max_attempts.  Backoff schedule for retryable
    statuses (429, 529, 5xx): 1s, 2s, 4s, 8s, 16s, 32s + 0..1s jitter.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text_parts: List[str] = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)
            in_tok = int(response.usage.input_tokens)
            out_tok = int(response.usage.output_tokens)
            return "".join(text_parts), in_tok, out_tok
        except anthropic.RateLimitError as e:
            last_exc = e
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", 0) or 0
            if status not in (429, 500, 502, 503, 504, 529):
                raise
            last_exc = e
        except anthropic.APIConnectionError as e:
            last_exc = e
        backoff = (2 ** attempt) + random.random()
        sys.stderr.write(
            f"[retry] attempt {attempt + 1}/{max_attempts} "
            f"sleeping {backoff:.1f}s ({type(last_exc).__name__})\n"
        )
        time.sleep(backoff)
    raise RuntimeError(
        f"call_with_retry: exhausted {max_attempts} attempts; "
        f"last error = {last_exc!r}"
    )


def parse_response(text: str) -> List[Dict[str, str]]:
    """Extract the list of {prompt, answer} from Claude's response.

    Robust to a small amount of preamble/postscript even though the
    system prompt asks for strict JSON — Claude sometimes wraps the
    JSON in code fences or adds an introductory line.  We find the
    first `{` and the matching last `}` and parse that span.
    """
    text = text.strip()
    if not text:
        return []
    # Strip common markdown code fence
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove the first line (```json or similar) and the last (```)
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1])
    # Find JSON bounds
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    examples = payload.get("examples", [])
    if not isinstance(examples, list):
        return []
    out: List[Dict[str, str]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        prompt = ex.get("prompt")
        answer = ex.get("answer")
        if isinstance(prompt, str) and isinstance(answer, str):
            prompt = prompt.strip()
            answer = answer.strip()
            if prompt and answer:
                out.append({"prompt": prompt, "answer": answer})
    return out


def load_existing_fingerprints(out_path: Path) -> set:
    """Resume support: read all previously-written rows and collect
    their prompt fingerprints so we don't generate duplicates."""
    fps: set = set()
    if not out_path.exists():
        return fps
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                prompt = row.get("prompt", "")
                if prompt:
                    fps.add(prompt_fingerprint(prompt))
            except json.JSONDecodeError:
                continue
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                                "artifacts" / "synthetic_instructions",
                        help="Output directory.  One JSONL per task type.")
    parser.add_argument("--total", type=int, default=50000,
                        help="Total examples to generate across all tasks.")
    parser.add_argument("--tasks", type=str,
                        default="summarize,extract_fact,explain_code,rewrite,translate",
                        help="Comma-separated task types to generate.")
    parser.add_argument("--model", type=str,
                        default="claude-sonnet-4-5-20250929",
                        help="Anthropic model id.  See MODEL_PRICING in source.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Examples per API call.  Higher = fewer calls "
                             "but riskier (one bad batch loses N examples).")
    parser.add_argument("--rpm", type=int, default=50,
                        help="Requests per minute cap.  Stay under your "
                             "Anthropic org's published RPM tier.")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Per-call max response tokens.  For batch=10 "
                             "summaries this is plenty.")
    parser.add_argument("--max-spend-usd", type=float, default=50.0,
                        help="Hard stop on cumulative spend.  When the "
                             "estimated cost reaches this, the script exits "
                             "with the partial output kept.")
    parser.add_argument("--seed", type=int, default=20260516,
                        help="Random seed for batch-seed and shuffling.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Override ANTHROPIC_API_KEY env var.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASK_INSTRUCTIONS:
            sys.stderr.write(f"[fatal] unknown task type: {t!r}.  "
                              f"Known: {list(TASK_INSTRUCTIONS)}\n")
            return 1

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.stderr.write(
            "[fatal] no Anthropic API key.  Set ANTHROPIC_API_KEY env var "
            "or pass --api-key.\n")
        return 1

    if args.model not in MODEL_PRICING:
        sys.stderr.write(
            f"[warn] unknown model {args.model!r} — cost tracking will "
            f"report $0.  Known: {list(MODEL_PRICING)}\n")
    pricing = MODEL_PRICING.get(args.model, {"input_per_mtok": 0.0,
                                              "output_per_mtok": 0.0})

    client = anthropic.Anthropic(api_key=api_key)
    limiter = RateLimiter(args.rpm)
    stats = GenStats()
    rng = random.Random(args.seed)

    # Per-task output file + fingerprint set (loaded for resume)
    task_files: Dict[str, Path] = {}
    task_fingerprints: Dict[str, set] = {}
    task_counts: Dict[str, int] = {}
    for t in tasks:
        path = out_dir / f"synthetic_{t}.jsonl"
        task_files[t] = path
        task_fingerprints[t] = load_existing_fingerprints(path)
        task_counts[t] = len(task_fingerprints[t])
        print(f"[init] task={t} existing={task_counts[t]} -> {path}")

    target_per_task = args.total // len(tasks)
    print(f"[init] target={args.total} total, {target_per_task} per task "
          f"({len(tasks)} tasks)")
    print(f"[init] model={args.model} batch_size={args.batch_size} "
          f"rpm={args.rpm} max_spend=${args.max_spend_usd:.2f}")

    open_files = {t: task_files[t].open("a", encoding="utf-8") for t in tasks}
    try:
        # Round-robin across tasks so all task files grow uniformly.
        while True:
            # Check completion + spend hard stop
            done_tasks = sum(1 for t in tasks if task_counts[t] >= target_per_task)
            if done_tasks == len(tasks):
                print("[done] all task quotas reached.")
                break
            if stats.spend_usd >= args.max_spend_usd:
                print(f"[stop] reached spend cap ${args.max_spend_usd:.2f} "
                      f"(actual ${stats.spend_usd:.4f}).  Exiting cleanly.")
                break

            # Pick the next task that still needs examples
            pending_tasks = [t for t in tasks if task_counts[t] < target_per_task]
            task = rng.choice(pending_tasks)

            user_msg = build_request_body(task, args.batch_size,
                                           seed=rng.randint(0, 1_000_000))
            limiter.wait()
            stats.requests_total += 1
            try:
                text, in_tok, out_tok = call_with_retry(
                    client, args.model, SYSTEM_PROMPT, user_msg, args.max_tokens)
            except Exception as e:
                stats.requests_failed += 1
                sys.stderr.write(f"[error] {task}: {type(e).__name__}: {e}\n")
                continue
            stats.requests_succeeded += 1
            stats.input_tokens += in_tok
            stats.output_tokens += out_tok
            stats.spend_usd += (
                in_tok * pricing["input_per_mtok"] / 1_000_000.0 +
                out_tok * pricing["output_per_mtok"] / 1_000_000.0
            )

            examples = parse_response(text)
            stats.examples_generated += len(examples)
            kept_this_batch = 0
            for ex in examples:
                if task_counts[task] >= target_per_task:
                    break
                prompt = ex["prompt"]
                answer = ex["answer"]
                fp = prompt_fingerprint(prompt)
                if fp in task_fingerprints[task]:
                    stats.examples_dropped_dup += 1
                    continue
                task_fingerprints[task].add(fp)
                row = {
                    "id": stable_id(prompt, answer),
                    "phase": "phase4_instructions",
                    "kind": task,
                    "prompt": prompt,
                    "answer": answer,
                    "source": f"claude_synthetic:{args.model}",
                }
                open_files[task].write(json.dumps(row, ensure_ascii=False) + "\n")
                task_counts[task] += 1
                stats.examples_kept += 1
                kept_this_batch += 1
            stats.examples_dropped_invalid += len(examples) - kept_this_batch - 0  # parse drops counted differently
            open_files[task].flush()

            if stats.requests_total % 10 == 0:
                print(f"[progress] req={stats.requests_total} "
                      f"kept={stats.examples_kept} "
                      f"spend=${stats.spend_usd:.4f} "
                      f"per_task={ {t: task_counts[t] for t in tasks} }")
    finally:
        for f in open_files.values():
            f.close()

    stats.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
