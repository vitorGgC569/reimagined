from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from nsos_curriculum_lib import PHASE_ORDER, read_jsonl
from train_curriculum import detect_build_dir, greedy_generate, load_nsos, postprocess_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one quick generation from each trained curriculum phase."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "curriculum_runs" / "latest",
        help="Curriculum run directory with checkpoints and run_summary.json.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Explicit build directory containing nsos_ext.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="cpu",
        help="Execution device for inspection.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max tokens to generate for each demo.",
    )
    parser.add_argument(
        "--sample-split",
        choices=["eval", "train"],
        default="eval",
        help="Which split to sample demonstrations from.",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also inspect final_model.bin after the per-phase checkpoints.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    return parser.parse_args()


def load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def choose_device(nsos, requested: str) -> int:
    if requested == "cpu":
        return nsos.Device.CPU
    if requested == "gpu":
        return nsos.Device.GPU
    if os.name == "nt":
        return nsos.Device.GPU
    return nsos.Device.CPU


def maybe_fallback_to_cpu(nsos, device: int) -> int:
    if device == nsos.Device.GPU and hasattr(nsos, "fast_gpu_supported"):
        if not nsos.fast_gpu_supported():
            return nsos.Device.CPU
    return device


def checkpoint_for_phase(run_dir: Path, phase_name: str) -> Path:
    checkpoint = run_dir / f"{phase_name}_best.bin"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing phase checkpoint: {checkpoint}")
    return checkpoint


def paired_edge_pack(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".edge.nsos")


def load_model_for_checkpoint(nsos, summary: Dict, checkpoint_path: Path, device: int):
    model_cfg = summary["model"]
    model = nsos.JambaModel(
        int(model_cfg["layers"]),
        int(model_cfg["d_model"]),
        int(model_cfg["vocab_size"]),
        device,
    )
    model.to(device)
    model.load(str(checkpoint_path))
    edge_pack = paired_edge_pack(checkpoint_path)
    if edge_pack.exists():
        model.load_edge_linear_pack(str(edge_pack), True)
    return model


def resolve_split_file(summary: Dict, phase_name: str, split: str) -> Path:
    bundle_dir = Path(summary["bundle_dir"])
    manifest = json.loads((bundle_dir / "curriculum_manifest.json").read_text(encoding="utf-8"))
    key = "eval_file" if split == "eval" else "train_file"
    for phase_entry in manifest["phases"]:
        if phase_entry["name"] == phase_name:
            return bundle_dir / Path(phase_entry[key])
    raise KeyError(f"Phase not found in manifest: {phase_name}")


def select_demo_row(rows: List[Dict]) -> Dict:
    if not rows:
        raise ValueError("No rows available for demo selection.")

    answered = [row for row in rows if row.get("answer")]
    if answered:
        return min(answered, key=lambda row: (len(row.get("prompt", "")), len(row.get("answer", ""))))

    documented = [row for row in rows if row.get("text")]
    if documented:
        return min(documented, key=lambda row: len(row.get("text", "")))

    return rows[0]


def build_demo_prompt(row: Dict) -> Tuple[str, str, str]:
    kind = row.get("kind", "")
    prompt = row.get("prompt", "")
    answer = row.get("answer", "").strip()
    if prompt:
        model_prompt = f"<|task:{kind}|>\nPrompt:\n{prompt}\nAnswer:\n"
        return "supervised", model_prompt, answer

    text = row.get("text", "").replace("<|endoftext|>", "").strip()
    if not text:
        return "empty", "", ""
    prefix = text[: min(220, max(120, len(text) // 2))].rstrip()
    return "continuation", prefix, ""


def clean_generation(text: str) -> str:
    return text.replace("<|endoftext|>", "").strip()


def inspect_checkpoint(nsos, tokenizer, model, phase_name: str, row: Dict, max_new_tokens: int) -> Dict:
    eos_token_ids = tokenizer.encode("<|endoftext|>")
    eos_token_id = eos_token_ids[0] if eos_token_ids else -1
    mode, prompt, expected = build_demo_prompt(row)
    generated = greedy_generate(
        nsos,
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        eos_token_id,
        task_kind=row.get("kind", ""),
    )
    if mode == "supervised":
        prediction = postprocess_generation(generated)
    else:
        prediction = clean_generation(generated)
    return {
        "phase": phase_name,
        "mode": mode,
        "kind": row.get("kind", ""),
        "prompt": prompt,
        "expected": expected,
        "prediction": prediction,
        "source_id": row.get("id", ""),
    }


def render_result(result: Dict) -> str:
    lines = [
        f"=== {result['phase']} ({result['kind'] or result['mode']}) ===",
        f"source_id: {result['source_id']}",
    ]
    if result["mode"] == "supervised":
        lines.extend(
            [
                "prompt:",
                result["prompt"],
                "expected:",
                result["expected"] or "<empty>",
                "prediction:",
                result["prediction"] or "<empty>",
            ]
        )
    else:
        lines.extend(
            [
                "prefix:",
                result["prompt"],
                "continuation:",
                result["prediction"] or "<empty>",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = load_summary(args.run_dir)
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)

    device = maybe_fallback_to_cpu(nsos, choose_device(nsos, args.device))

    tokenizer = nsos.Tokenizer()
    tokenizer_pack = args.run_dir / "tokenizer.nsos"
    if tokenizer_pack.exists():
        tokenizer.load_pack(str(tokenizer_pack))
    else:
        tokenizer_path = Path(summary["bundle_dir"]) / f"tokenizer_{summary['model']['target_vocab']}.ox3"
        tokenizer.load(str(tokenizer_path))

    results: List[Dict] = []
    for phase_name in PHASE_ORDER:
        checkpoint = checkpoint_for_phase(args.run_dir, phase_name)
        rows = read_jsonl(resolve_split_file(summary, phase_name, args.sample_split))
        row = select_demo_row(rows)
        model = load_model_for_checkpoint(nsos, summary, checkpoint, device)
        results.append(inspect_checkpoint(nsos, tokenizer, model, phase_name, row, args.max_new_tokens))

    if args.include_final:
        final_checkpoint = args.run_dir / "final_model.bin"
        if final_checkpoint.exists():
            phase_name = "final_model"
            rows = read_jsonl(resolve_split_file(summary, "phase4_instructions", args.sample_split))
            row = select_demo_row(rows)
            model = load_model_for_checkpoint(nsos, summary, final_checkpoint, device)
            results.append(inspect_checkpoint(nsos, tokenizer, model, phase_name, row, args.max_new_tokens))

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for index, result in enumerate(results):
            if index:
                print()
            print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
