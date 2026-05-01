from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root


def detect_build_dir(explicit: Path | None, repo_root: Path) -> Path:
    candidates = []
    nsos_root = repo_root / "OXN" / "nsos"
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    for name in ["build_cuda129", "build_v1", "build_full", "build_codex", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("nsos_ext*.pyd")):
            return candidate
    raise RuntimeError("Could not find a build directory with nsos_ext.")


def load_nsos(build_dir: Path):
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        add_windows_runtime_dirs(
            build_dir,
            parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
        )
    import nsos_ext as nsos  # type: ignore

    return nsos


def find_latest_model(repo_root: Path) -> Path:
    runs_root = repo_root / "OXN" / "nsos" / "artifacts" / "curriculum_runs"
    candidates: list[Path] = []
    if runs_root.exists():
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            for filename in ("final_model.bin", "champion_global.bin"):
                path = run_dir / filename
                if path.exists():
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Could not find any final_model.bin or champion_global.bin under curriculum_runs.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_effective_config(model_path: Path) -> Path | None:
    candidates = []
    if model_path.is_dir():
        candidates.append(model_path / "effective_model_config.json")
    else:
        candidates.append(model_path.parent / "effective_model_config.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def apply_effective_config(config, config_path: Path | None) -> None:
    if config_path is None:
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if hasattr(config, key):
            setattr(config, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive local chat loop for an NSOS model.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Optional explicit build directory containing nsos_ext.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint or model pack path. Defaults to the newest final/champion artifact.",
    )
    parser.add_argument("--cuda", action="store_true", help="Request CUDA execution.")
    parser.add_argument("--max-context", type=int, default=512, help="Maximum context tokens.")
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cutoff.")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k sampling.")
    parser.add_argument(
        "--system",
        type=str,
        default="You are NSOS. Prefer exact short answers for exact tasks, otherwise answer in concise technical prose.",
        help="System instruction prepended to every user turn.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = detect_build_dir(args.build_dir, args.repo_root)
    nsos = load_nsos(build_dir)
    model_path = args.model or find_latest_model(args.repo_root)

    config = nsos.ModelConfig()
    apply_effective_config(config, find_effective_config(model_path))
    config.use_cuda = bool(args.cuda)
    config.max_context_tokens = int(args.max_context)

    engine = nsos.InferenceEngine()
    if not engine.load_model(str(model_path), config):
        raise RuntimeError(f"Failed to load model: {model_path}")

    options = nsos.GenerationOptions()
    options.max_context_tokens = int(args.max_context)
    options.max_tokens = int(args.max_tokens)
    options.temperature = float(args.temperature)
    options.top_p = float(args.top_p)
    options.top_k = int(args.top_k)
    options.stream = False

    print(f"[model] {model_path}")
    print("[chat] Type /exit to quit.")

    while True:
        try:
            user_text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break

        prompt = (
            "<|task:chat|>\n"
            "Prompt:\n"
            f"System: {args.system}\n"
            f"User: {user_text}\n"
            "Answer:\n"
        )
        output = engine.generate_ex(prompt, options).strip()
        print(f"nsos> {output}")

        metrics = engine.last_generation_metrics()
        print(
            "[metrics] "
            f"prompt_total={metrics.prompt_tokens_total} "
            f"generated={metrics.generated_tokens} "
            f"elapsed_ms={metrics.elapsed_ms:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
