"""Build an SFT bundle from public instruction datasets.

Produces a JSONL of `{"prompt": <ChatML prompt>, "answer": <text>, "kind": ...}`
rows ready for `sft_phase.py --instruction-bundle`.

Default mix:
  * SmolTalk (HuggingFaceTB/smoltalk) — conversational, diverse
  * Alpaca-cleaned (yahma/alpaca-cleaned) — instructions
  * Optional: HelpSteer2 (chosen-only) for higher-quality samples

All conversations get formatted with `chatml.format_chat` and split into
prompt/answer via `chatml.split_prompt_answer`.  Loss in SFT then runs
only on the answer span — this is the trainer's existing supervised
batch contract.

CLI:
  python OXN/nsos/scripts/build_sft_bundle.py \\
      --out OXN/nsos/artifacts/sft_bundle/instructions.jsonl \\
      --max-samples 20000

Output JSONL fields:
  prompt:  ChatML-formatted context ending with `<|im_start|>assistant\\n`
  answer:  Plain text of the assistant's turn (no special tokens)
  kind:    Source dataset ('smoltalk', 'alpaca', 'helpsteer2')
  meta:    Optional metadata (subject, difficulty, etc.)
"""
from __future__ import annotations

# Eager datasets import before torch — Windows segfault workaround.
import datasets  # noqa: F401

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chatml import format_chat, split_prompt_answer  # noqa: E402


# ── Per-dataset adapters: turn each row into a list[Message] ──────────

def _smoltalk_iter(max_rows: int) -> Iterator[Dict]:
    """SmolTalk: HuggingFaceTB/smoltalk.  Already in {role, content} format."""
    ds = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            continue
        # Normalize role keys ('from' / 'role') and content keys ('value' / 'content')
        norm = []
        for m in messages:
            role = m.get("role") or m.get("from") or "user"
            # SmolTalk uses 'gpt' for assistant in some splits
            if role in ("gpt", "bot", "model"):
                role = "assistant"
            if role == "human":
                role = "user"
            content = m.get("content") or m.get("value") or ""
            norm.append({"role": role, "content": content})
        if not any(m["role"] == "assistant" for m in norm):
            continue
        yield {"messages": norm, "kind": "smoltalk"}


def _alpaca_iter(max_rows: int) -> Iterator[Dict]:
    """Alpaca-cleaned: yahma/alpaca-cleaned.  Single-turn instruction → output."""
    ds = datasets.load_dataset("yahma/alpaca-cleaned", split="train")
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_content = instruction + (f"\n\n{inp}" if inp else "")
        yield {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ],
            "kind": "alpaca",
        }


def _helpsteer2_iter(max_rows: int) -> Iterator[Dict]:
    """HelpSteer2 (nvidia/HelpSteer2) — high-quality rated responses.

    Each row has a prompt and a single response with helpfulness,
    correctness, coherence, complexity, verbosity ratings.  We keep
    only responses scoring ≥ 3 on helpfulness AND ≥ 3 on correctness
    to filter low-quality samples.
    """
    try:
        ds = datasets.load_dataset("nvidia/HelpSteer2", split="train")
    except Exception:
        # Dataset gated or unavailable — caller should fall back to others
        return
    n = 0
    for row in ds:
        if n >= max_rows:
            break
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()
        if not prompt or not response:
            continue
        if (row.get("helpfulness", 0) < 3) or (row.get("correctness", 0) < 3):
            continue
        yield {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "kind": "helpsteer2",
            "meta": {k: row.get(k) for k in
                     ("helpfulness", "correctness", "coherence", "complexity", "verbosity")},
        }
        n += 1


SOURCES = {
    "smoltalk": _smoltalk_iter,
    "alpaca": _alpaca_iter,
    "helpsteer2": _helpsteer2_iter,
}


# ── Bundler ──────────────────────────────────────────────────────────

def build_bundle(
    out_path: Path,
    *,
    max_samples: int,
    mix: Optional[Dict[str, float]] = None,
    seed: int = 42,
    system_message: Optional[str] = None,
) -> Dict[str, int]:
    """Collect rows from each source per `mix` (fractions summing to ~1),
    format with ChatML, write to JSONL.

    Returns a count-per-source dict for logging.
    """
    mix = mix or {"smoltalk": 0.6, "alpaca": 0.3, "helpsteer2": 0.1}
    total = sum(mix.values())
    norm_mix = {k: v / total for k, v in mix.items()}

    rng = random.Random(seed)
    rows: List[Dict] = []
    counts: Dict[str, int] = {}
    for source, frac in norm_mix.items():
        if source not in SOURCES:
            print(f"[bundle] WARN: unknown source {source}, skipping",
                  file=sys.stderr)
            continue
        target = int(max_samples * frac)
        # Pull 2x what we need so filtering doesn't starve us
        pulled = 0
        kept = 0
        for row in SOURCES[source](target * 3):
            pulled += 1
            try:
                prompt, answer = split_prompt_answer(
                    row["messages"], system_message=system_message,
                )
            except ValueError:
                continue
            if not answer.strip():
                continue
            rows.append({
                "prompt": prompt,
                "answer": answer,
                "kind": row["kind"],
                "meta": row.get("meta", {}),
            })
            kept += 1
            if kept >= target:
                break
        counts[source] = kept
        print(f"[bundle] {source:<12} pulled={pulled:>6} kept={kept:>5}",
              flush=True)

    rng.shuffle(rows)
    rows = rows[:max_samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[bundle] wrote {len(rows):,} rows to {out_path}", flush=True)
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build an SFT bundle in ChatML format.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path.")
    p.add_argument("--max-samples", type=int, default=20_000,
                   help="Total samples across all sources.")
    p.add_argument("--system-message", type=str, default=None,
                   help="Optional system prompt to inject.")
    p.add_argument("--smoltalk", type=float, default=0.6)
    p.add_argument("--alpaca", type=float, default=0.3)
    p.add_argument("--helpsteer2", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    mix = {"smoltalk": args.smoltalk, "alpaca": args.alpaca,
           "helpsteer2": args.helpsteer2}
    build_bundle(
        args.out, max_samples=args.max_samples, mix=mix, seed=args.seed,
        system_message=args.system_message,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
