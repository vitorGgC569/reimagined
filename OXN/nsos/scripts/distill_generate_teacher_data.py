#!/usr/bin/env python3
"""
distill_generate_teacher_data.py
---------------------------------
Sequence-level knowledge distillation: run prompts from the curriculum bundle
through a Llama 2 teacher (GGUF) and write the teacher's answers into a new
distillation bundle. The student is then trained on those teacher-generated
answers instead of the original synthetic ones.

Why sequence-level (not token-level)?
  The student uses a different tokenizer and vocabulary (4810 tokens) than the
  teacher. We cannot directly align logit distributions. Instead, we:
    1. Decode the teacher's output to text.
    2. Re-tokenize it with the student's tokenizer.
    3. Train the student with cross-entropy on those tokens.

This still transfers teacher knowledge because the student now learns to
produce the same surface text as the teacher, not just the synthetic corpus.

Usage:
  python OXN/nsos/scripts/distill_generate_teacher_data.py \
    --teacher-model C:/Users/Oxta/Desktop/llm/llama-2-7b.Q4_0.gguf \
    --bundle-dir  OXN/nsos/artifacts/curriculum_bundle \
    --out-dir     OXN/nsos/artifacts/distillation_bundle \
    --phases phase4_instructions phase5_verifier \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 0.9 \
    --n-ctx 2048
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# Teacher wrapper
# ──────────────────────────────────────────────────────────────────────────────

class LlamaTeacher:
    """Thin wrapper around llama-cpp-python for teacher inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        verbose: bool = False,
    ) -> None:
        from llama_cpp import Llama  # type: ignore

        threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        print(f"[teacher] loading {model_path} (ctx={n_ctx}, threads={threads}, gpu_layers={n_gpu_layers})")
        t0 = time.time()
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            logits_all=False,
        )
        print(f"[teacher] loaded in {time.time() - t0:.1f}s")
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        """Return teacher-generated completion for a prompt (chat completion API)."""
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": LLAMA2_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        text: str = response["choices"][0]["message"]["content"]
        return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Prompt formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

LLAMA2_SYSTEM = (
    "You are a helpful, accurate assistant. "
    "Give concise, correct answers."
)

def build_llama2_prompt(prompt_text: str) -> str:
    """Wrap prompt in Llama 2 chat format."""
    return (
        f"<s>[INST] <<SYS>>\n{LLAMA2_SYSTEM}\n<</SYS>>\n\n"
        f"{prompt_text.strip()} [/INST]"
    )


def clean_answer(text: str) -> str:
    """Remove common Llama artifacts and trim."""
    # Remove trailing </s> or [INST] leakage
    for stop in ["</s>", "[INST]", "[/INST]", "<<SYS>>"]:
        text = text.split(stop)[0]
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Bundle processing
# ──────────────────────────────────────────────────────────────────────────────

def process_phase(
    teacher: LlamaTeacher,
    src_path: Path,
    dst_path: Path,
    split: str,
    phase: str,
    max_samples: int,
    rng: random.Random,
    verbose: bool,
) -> Dict[str, int]:
    """Read src .jsonl, generate teacher answers, write dst .jsonl."""

    src_file = src_path / f"{phase}.{split}.jsonl"
    dst_file = dst_path / f"{phase}.{split}.jsonl"

    if not src_file.exists():
        print(f"[skip] {src_file} not found")
        return {"kept": 0, "generated": 0, "failed": 0}

    rows: List[Dict[str, Any]] = []
    with src_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Rows without a prompt/answer are passed through unchanged (text-only LM rows)
    lm_rows = [r for r in rows if not r.get("answer")]
    qa_rows = [r for r in rows if r.get("answer")]

    # Sample if too many QA rows
    if len(qa_rows) > max_samples:
        rng.shuffle(qa_rows)
        qa_rows = qa_rows[:max_samples]

    stats = {"kept": len(lm_rows), "generated": 0, "failed": 0}
    out_rows: List[Dict[str, Any]] = list(lm_rows)

    print(f"[{phase}.{split}] {len(qa_rows)} QA rows to distil, {len(lm_rows)} LM rows passed through")

    for i, row in enumerate(qa_rows):
        prompt_text = row.get("prompt", "")
        if not prompt_text:
            out_rows.append(row)
            stats["kept"] += 1
            continue

        try:
            raw = teacher.generate(prompt_text)
            answer = clean_answer(raw)
            if not answer:
                # Fallback to original
                out_rows.append(row)
                stats["failed"] += 1
                if verbose:
                    print(f"  [warn] empty teacher output for id={row.get('id','?')}")
                continue

            new_row = dict(row)
            new_row["answer"] = answer
            new_row["source"] = "teacher_llama2_7b"
            # Rebuild the text field in the same format as the bundle
            task_tag = row.get("kind", "qa")
            new_row["text"] = (
                f"<|task:{task_tag}|>\nPrompt:\n{prompt_text}\nAnswer:\n{answer}<|endoftext|>"
            )
            out_rows.append(new_row)
            stats["generated"] += 1

            if (i + 1) % 10 == 0 or verbose:
                print(f"  [{i+1}/{len(qa_rows)}] id={row.get('id','?')} -> {answer[:60]!r}")

        except Exception as exc:
            print(f"  [error] id={row.get('id','?')}: {exc}")
            out_rows.append(row)
            stats["failed"] += 1

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with dst_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[{phase}.{split}] done - kept={stats['kept']} generated={stats['generated']} failed={stats['failed']}")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate teacher data for distillation")
    ap.add_argument("--teacher-model", required=True, help="Path to .gguf teacher model")
    ap.add_argument("--bundle-dir", required=True, help="Source curriculum bundle directory")
    ap.add_argument("--out-dir", required=True, help="Output distillation bundle directory")
    ap.add_argument(
        "--phases",
        nargs="+",
        default=["phase4_instructions", "phase5_verifier"],
        help="Which phases to distil (default: phase4 + phase5)",
    )
    ap.add_argument("--max-samples", type=int, default=300,
                    help="Max QA rows per split per phase to send to teacher (default 300)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="Number of Llama layers to offload to GPU (default 0=CPU, -1=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.bundle_dir)
    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy bundle skeleton (manifest + tokenizers + non-distilled phases)
    print(f"[setup] copying bundle skeleton {src} -> {dst}")
    for item in src.iterdir():
        if item.name == "data":
            continue
        dst_item = dst / item.name
        if item.is_file() and not dst_item.exists():
            shutil.copy2(item, dst_item)

    src_data = src / "data"
    dst_data = dst / "data"
    dst_data.mkdir(exist_ok=True)

    # Pass through all phases not being distilled
    all_phases = set()
    for f in src_data.glob("*.jsonl"):
        phase = f.stem.rsplit(".", 1)[0]
        all_phases.add(phase)

    passthrough = all_phases - set(args.phases)
    for phase in sorted(passthrough):
        for split in ("train", "eval"):
            src_file = src_data / f"{phase}.{split}.jsonl"
            dst_file = dst_data / f"{phase}.{split}.jsonl"
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"[passthrough] {phase}.{split}")

    # Load teacher
    teacher = LlamaTeacher(
        model_path=args.teacher_model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    # Distil selected phases
    total_generated = 0
    t_start = time.time()
    for phase in args.phases:
        for split in ("train", "eval"):
            stats = process_phase(
                teacher, src_data, dst_data,
                split=split, phase=phase,
                max_samples=args.max_samples,
                rng=rng,
                verbose=args.verbose,
            )
            total_generated += stats["generated"]

    elapsed = time.time() - t_start
    print(f"\n[done] total_generated={total_generated} elapsed={elapsed:.0f}s")
    print(f"[done] distillation bundle -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
