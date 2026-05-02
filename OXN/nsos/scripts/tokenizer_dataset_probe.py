from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
from nsos_curriculum_lib import SPECIAL_TOKENS, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tokenizer efficiency on curriculum/holdout JSONL files."
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, action="append", required=True)
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_nsos(build_dir: Path):
    candidates = [build_dir, build_dir / "Release"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("nsos_ext*.pyd")):
            build_dir = candidate.resolve()
            break
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        add_windows_runtime_dirs(
            build_dir,
            parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
        )
    import nsos_ext as nsos  # type: ignore

    return nsos


def text_for_row(row: dict) -> str:
    if row.get("text"):
        return str(row["text"])
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    kind = str(row.get("kind", "task"))
    return f"<|task:{kind}|>\nPrompt:\n{prompt}\nAnswer:\n{answer}<|endoftext|>"


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "max": 0}
    return {"count": len(values), "mean": mean(values), "max": max(values)}


def evaluate_tokenizer(nsos, tokenizer_path: Path, jsonl_paths: list[Path]) -> dict:
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(tokenizer_path))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    file_reports = {}
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_byte_like = 0
    answer_lengths: list[int] = []
    prompt_lengths: list[int] = []

    for path in jsonl_paths:
        rows = read_jsonl(path)
        file_tokens = 0
        file_words = 0
        file_chars = 0
        file_byte_like = 0
        for row in rows:
            text = text_for_row(row)
            ids = tokenizer.encode(text)
            words = re.findall(r"[A-Za-z0-9_/-]+|[^\W\s]+", text, flags=re.UNICODE)
            file_tokens += len(ids)
            file_words += len(words)
            file_chars += len(text)
            file_byte_like += sum(1 for token_id in ids if int(token_id) < 256)
            if row.get("answer"):
                answer_lengths.append(len(tokenizer.encode(str(row["answer"]))))
            if row.get("prompt"):
                prompt_lengths.append(len(tokenizer.encode(str(row["prompt"]))))

        total_tokens += file_tokens
        total_words += file_words
        total_chars += file_chars
        total_byte_like += file_byte_like
        file_reports[path.name] = {
            "rows": len(rows),
            "tokens": file_tokens,
            "words": file_words,
            "chars": file_chars,
            "tokens_per_word": file_tokens / max(file_words, 1),
            "tokens_per_char": file_tokens / max(file_chars, 1),
            "byte_like_ratio": file_byte_like / max(file_tokens, 1),
        }

    return {
        "tokenizer": str(tokenizer_path),
        "vocab_size": int(tokenizer.vocab_size),
        "files": file_reports,
        "totals": {
            "tokens": total_tokens,
            "words": total_words,
            "chars": total_chars,
            "tokens_per_word": total_tokens / max(total_words, 1),
            "tokens_per_char": total_tokens / max(total_chars, 1),
            "byte_like_ratio": total_byte_like / max(total_tokens, 1),
            "answer_tokens": summarize_lengths(answer_lengths),
            "prompt_tokens": summarize_lengths(prompt_lengths),
        },
    }


def main() -> int:
    args = parse_args()
    nsos = load_nsos(args.build_dir)
    report = {
        "tokenizers": [
            evaluate_tokenizer(nsos, tokenizer_path, args.jsonl)
            for tokenizer_path in args.tokenizer
        ]
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
