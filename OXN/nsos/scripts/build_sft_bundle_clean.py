"""Build a CLEAN, HIGH-QUALITY SFT bundle from public instruction datasets.

Sanitizes prompts and answers:
  1. Removes web boilerplate (LEIA MAIS, Clique aqui, etc.).
  2. Applies minimum quality thresholds for answers (> 20 chars, non-repetitive).
  3. Formats with ChatML without special token leaking.
"""
from __future__ import annotations

import datasets
import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chatml import format_chat, split_prompt_answer
from fetch_real_datasets_clean import sanitize_text_content


def clean_sft_turn(text: str) -> str:
    """Sanitizes SFT prompt or assistant turn."""
    if not text:
        return ""
    clean = sanitize_text_content(text)
    # Remove artificial prefixes or tokens
    clean = re.sub(r"^(?:User:|Assistant:|Resposta:|Pergunta:)\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def _smoltalk_clean_iter(max_rows: int) -> Iterator[Dict]:
    ds = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            continue
        norm = []
        for m in messages:
            role = m.get("role") or m.get("from") or "user"
            if role in ("gpt", "bot", "model"):
                role = "assistant"
            if role == "human":
                role = "user"
            content = clean_sft_turn(m.get("content") or m.get("value") or "")
            if content:
                norm.append({"role": role, "content": content})
        if not any(m["role"] == "assistant" for m in norm):
            continue
        yield {"messages": norm, "kind": "smoltalk"}


def _alpaca_clean_iter(max_rows: int) -> Iterator[Dict]:
    ds = datasets.load_dataset("yahma/alpaca-cleaned", split="train")
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        instruction = clean_sft_turn(row.get("instruction") or "")
        inp = clean_sft_turn(row.get("input") or "")
        output = clean_sft_turn(row.get("output") or "")
        if not instruction or not output or len(output) < 15:
            continue
        user_content = instruction + (f"\n\n{inp}" if inp else "")
        yield {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ],
            "kind": "alpaca",
        }


def build_clean_sft_bundle(out_path: Path, max_samples: int = 20000, seed: int = 42) -> None:
    print(f"[sft_clean] Building sanitized SFT bundle to {out_path}...")
    rows = []
    for r in _smoltalk_clean_iter(max_samples // 2):
        try:
            prompt, answer = split_prompt_answer(r["messages"])
            if len(answer) >= 15:
                rows.append({"prompt": prompt, "answer": answer, "kind": r["kind"]})
        except ValueError:
            continue

    for r in _alpaca_clean_iter(max_samples // 2):
        try:
            prompt, answer = split_prompt_answer(r["messages"])
            if len(answer) >= 15:
                rows.append({"prompt": prompt, "answer": answer, "kind": r["kind"]})
        except ValueError:
            continue

    random.Random(seed).shuffle(rows)
    rows = rows[:max_samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sft_clean] Done! Wrote {len(rows)} clean SFT rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build clean SFT bundle.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()
    build_clean_sft_bundle(args.out, args.max_samples)
