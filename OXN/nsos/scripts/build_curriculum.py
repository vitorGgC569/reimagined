from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nsos_curriculum_lib import (
    DEFAULT_PHASE_SIZES,
    DEFAULT_PHASE_SIZES_V11,
    build_curriculum,
    build_tokenizer_bundle,
)


PRESETS = {
    "v10": DEFAULT_PHASE_SIZES,         # original 3,840-sample bundle
    "v11": DEFAULT_PHASE_SIZES_V11,     # 40,000-sample bundle with new datasets
}


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
        default=None,
        help=(
            "Output directory for curriculum JSONL files and tokenizer artifacts. "
            "Defaults to OXN/nsos/scripts/distillation_bundle_<preset> so the "
            "trainer can find it via the existing convention."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="v10",
        help=(
            "Which phase-size table to use.  'v10' = original (3,840 train rows total). "
            "'v11' = expanded (40,000 train rows total) — REQUIRES that fetch_real_datasets.py "
            "with --scale-factor 10 or higher AND the v11 datasets (cosmopedia, tinystories, "
            "smoltalk, the_stack_smol, c4_sample, squad_v2) have been pulled into "
            "artifacts/real_datasets/ first.  Without those files, build still works but "
            "v11 falls back to v10-equivalent content (silent — see build log warnings)."
        ),
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
    phase_sizes = PRESETS[args.preset]

    if args.out_dir is None:
        # Use the existing scripts/distillation_bundle_<preset> convention
        # so the trainer can locate the bundle without configuration.
        out_dir = Path(__file__).resolve().parent / f"distillation_bundle_{args.preset}"
    else:
        out_dir = args.out_dir

    print(f"[curriculum] preset:    {args.preset}")
    print(f"[curriculum] out_dir:   {out_dir}")
    print(f"[curriculum] phase sizes (train):")
    for name, sizes in phase_sizes.items():
        print(f"  - {name}: train={sizes['train']:>6}  eval={sizes['eval']:>4}")

    started = time.perf_counter()
    manifest_path = build_curriculum(args.repo_root, out_dir, seed=args.seed,
                                      phase_sizes=phase_sizes)
    tokenizer_path = build_tokenizer_bundle(out_dir, target_vocab=args.target_vocab)
    elapsed = time.perf_counter() - started

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"\n[curriculum] manifest:  {manifest_path}")
    print(f"[curriculum] tokenizer: {tokenizer_path}")
    print(f"[curriculum] elapsed:   {elapsed:.1f}s")
    print(f"\n[curriculum] phase output:")
    total_train = 0
    total_eval = 0
    for phase in manifest["phases"]:
        train_n = phase["train_samples"]
        eval_n = phase["eval_samples"]
        total_train += train_n
        total_eval += eval_n
        print(f"  - {phase['name']:<22}  train={train_n:>6}  eval={eval_n:>4}")
    print(f"  {'TOTAL':<22}  train={total_train:>6}  eval={total_eval:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
