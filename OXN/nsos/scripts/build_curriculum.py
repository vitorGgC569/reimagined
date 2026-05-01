from __future__ import annotations

import argparse
import json
from pathlib import Path

from nsos_curriculum_lib import (
    DEFAULT_PHASE_SIZES,
    build_curriculum,
    build_tokenizer_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the NSOS curriculum bundle and tokenizer.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root used to harvest curated technical text.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "curriculum_bundle",
        help="Output directory for curriculum JSONL files and tokenizer artifacts.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Seed for curriculum generation.")
    parser.add_argument(
        "--target-vocab",
        type=int,
        default=2048,
        help="Final target vocabulary size including special tokens.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = build_curriculum(args.repo_root, args.out_dir, seed=args.seed, phase_sizes=DEFAULT_PHASE_SIZES)
    tokenizer_path = build_tokenizer_bundle(args.out_dir, target_vocab=args.target_vocab)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[curriculum] manifest: {manifest_path}")
    print(f"[curriculum] tokenizer: {tokenizer_path}")
    for phase in manifest["phases"]:
        print(
            f"  - {phase['name']}: train={phase['train_samples']} eval={phase['eval_samples']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
