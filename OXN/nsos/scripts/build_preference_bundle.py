"""Build a preference (chosen / rejected) bundle from public datasets.

Output JSONL with one row per preference pair:
  prompt    — ChatML prompt
  chosen    — preferred completion (plain text)
  rejected  — dispreferred completion (plain text)
  source    — origin dataset
  meta      — optional dataset-specific fields

This bundle feeds `dpo_phase.py` (item #20 of VISION roadmap — Safety/DPO).

Default mix:
  * Anthropic/hh-rlhf (helpful + harmless splits) — the canonical
    preference dataset.  Each pair is a context + two model completions
    rated by humans.
  * Stanford/SHP (Stanford Human Preferences) — Reddit-derived
    preferences with score deltas, gives breadth.

CLI:
  python OXN/nsos/scripts/build_preference_bundle.py \\
      --out OXN/nsos/artifacts/preference_bundle/pairs.jsonl \\
      --max-samples 10000
"""
from __future__ import annotations

import datasets  # eager before torch  # noqa: F401

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chatml import format_chat  # noqa: E402


# ── HH-RLHF adapter ────────────────────────────────────────────────────

# HH-RLHF rows look like:
#   {"chosen": "\n\nHuman: ...\n\nAssistant: ...",
#    "rejected": "\n\nHuman: ...\n\nAssistant: ..."}
# Both the chosen and rejected strings INCLUDE the prompt; we need to
# extract the shared prefix and the diverging assistant completion.

HH_TURN_RE = re.compile(r"\n\n(Human|Assistant):", re.IGNORECASE)


def _parse_hh_row(chosen_raw: str, rejected_raw: str) -> Optional[Dict]:
    """Split the chosen / rejected strings into (prompt_messages, chosen_text, rejected_text)."""
    def _to_turns(raw: str) -> List[Dict]:
        # Split on "\n\nHuman:" / "\n\nAssistant:" markers.  Use the regex
        # to find the offsets so we keep the role labels in alignment.
        turns: List[Dict] = []
        positions = [(m.start(), m.group(1).lower()) for m in HH_TURN_RE.finditer(raw)]
        for i, (start, role) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
            # Skip past the "\n\nRole:" preamble in this turn
            content_start = raw.find(":", start) + 1
            content = raw[content_start:end].strip()
            mapped = "user" if role == "human" else "assistant"
            turns.append({"role": mapped, "content": content})
        return turns

    chosen_turns = _to_turns(chosen_raw)
    rejected_turns = _to_turns(rejected_raw)
    if not chosen_turns or not rejected_turns:
        return None
    # The chosen and rejected must share all turns except the LAST assistant turn.
    # If they don't, the row is malformed.
    if chosen_turns[-1]["role"] != "assistant" or rejected_turns[-1]["role"] != "assistant":
        return None
    if chosen_turns[:-1] != rejected_turns[:-1]:
        # Some HH rows differ earlier; skip those for cleanliness.
        return None
    prompt_messages = chosen_turns[:-1]
    chosen_text = chosen_turns[-1]["content"]
    rejected_text = rejected_turns[-1]["content"]
    if not chosen_text or not rejected_text:
        return None
    return {
        "prompt_messages": prompt_messages,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def _hh_iter(max_rows: int) -> Iterator[Dict]:
    """Anthropic/hh-rlhf — helpful + harmless splits combined.

    The dataset has 4 splits in the helpful and harmless subsets each;
    we use the standard train splits.
    """
    yielded = 0
    for subset in ("helpful-base", "helpful-online", "harmless-base"):
        try:
            ds = datasets.load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
        except Exception:
            try:
                # Some mirrors expose the subsets differently
                ds = datasets.load_dataset("Anthropic/hh-rlhf", subset, split="train")
            except Exception as e:
                print(f"[hh-rlhf] could not load subset {subset}: {e}", file=sys.stderr)
                continue
        for row in ds:
            if yielded >= max_rows:
                return
            parsed = _parse_hh_row(row.get("chosen", ""), row.get("rejected", ""))
            if parsed is None:
                continue
            parsed["source"] = f"hh-rlhf:{subset}"
            yield parsed
            yielded += 1


# ── SHP (Stanford Human Preferences) adapter ──────────────────────────

def _shp_iter(max_rows: int) -> Iterator[Dict]:
    """stanfordnlp/SHP — Reddit-derived preferences with score deltas.

    Each row has a `history` (the post + prior comments), `human_ref_A`,
    `human_ref_B`, and `labels` (0 = A preferred, 1 = B preferred).
    """
    try:
        ds = datasets.load_dataset("stanfordnlp/SHP", split="train")
    except Exception as e:
        print(f"[shp] could not load: {e}", file=sys.stderr)
        return
    yielded = 0
    for row in ds:
        if yielded >= max_rows:
            return
        history = (row.get("history") or "").strip()
        ref_a = (row.get("human_ref_A") or "").strip()
        ref_b = (row.get("human_ref_B") or "").strip()
        label = row.get("labels")
        if not history or not ref_a or not ref_b or label is None:
            continue
        # SHP score delta hints at strength of preference; we keep only
        # cases where score_ratio > 2 so the preference signal is strong.
        try:
            ratio = float(row.get("score_ratio", 1.0))
            if ratio < 2.0:
                continue
        except (TypeError, ValueError) as exc:
            print(
                f"[shp] invalid score_ratio; row skipped: {exc}",
                file=sys.stderr,
            )
            continue
        if label == 0:
            chosen, rejected = ref_a, ref_b
        else:
            chosen, rejected = ref_b, ref_a
        yield {
            "prompt_messages": [{"role": "user", "content": history}],
            "chosen": chosen,
            "rejected": rejected,
            "source": "shp",
            "meta": {"score_ratio": row.get("score_ratio")},
        }
        yielded += 1


SOURCES = {"hh-rlhf": _hh_iter, "shp": _shp_iter}


def build_bundle(
    out_path: Path,
    *,
    max_samples: int,
    mix: Optional[Dict[str, float]] = None,
    seed: int = 42,
    system_message: Optional[str] = None,
    min_chars: int = 8,
    max_chars: int = 4096,
) -> Dict[str, int]:
    mix = mix or {"hh-rlhf": 0.8, "shp": 0.2}
    norm = {k: v / sum(mix.values()) for k, v in mix.items()}

    rng = random.Random(seed)
    rows: List[Dict] = []
    counts: Dict[str, int] = {}
    for source, frac in norm.items():
        if source not in SOURCES:
            print(f"[bundle] WARN unknown source {source}", file=sys.stderr)
            continue
        target = int(max_samples * frac)
        pulled = 0
        kept = 0
        for parsed in SOURCES[source](target * 3):
            pulled += 1
            # Quality filter on response lengths
            if not (min_chars <= len(parsed["chosen"]) <= max_chars):
                continue
            if not (min_chars <= len(parsed["rejected"]) <= max_chars):
                continue
            try:
                prompt = format_chat(
                    parsed["prompt_messages"],
                    add_generation_prompt=True,
                    system_message=system_message,
                )
            except Exception:
                continue
            rows.append({
                "prompt": prompt,
                "chosen": parsed["chosen"],
                "rejected": parsed["rejected"],
                "source": parsed["source"],
                "meta": parsed.get("meta", {}),
            })
            kept += 1
            if kept >= target:
                break
        counts[source] = kept
        print(f"[bundle] {source:<12} pulled={pulled:>6} kept={kept:>5}", flush=True)

    rng.shuffle(rows)
    rows = rows[:max_samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[bundle] wrote {len(rows):,} preference pairs to {out_path}", flush=True)
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build a DPO preference bundle in ChatML format.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--hh-rlhf", type=float, default=0.8)
    p.add_argument("--shp", type=float, default=0.2)
    p.add_argument("--system-message", type=str, default=None)
    p.add_argument("--min-chars", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    build_bundle(
        args.out, max_samples=args.max_samples,
        mix={"hh-rlhf": args.hh_rlhf, "shp": args.shp},
        seed=args.seed, system_message=args.system_message,
        min_chars=args.min_chars, max_chars=args.max_chars,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
