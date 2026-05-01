from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_external import detect_build_dir, load_nsos
from train_curriculum import (
    build_task_start_blocklist,
    compute_repetition_metrics,
    greedy_generate,
    postprocess_generation,
    safe_decode_tokens,
)


DEFAULT_REFERENCE_MODELS = [
    "distilgpt2",
    "roneneldan/TinyStories-33M",
]


@dataclass
class EvalResult:
    exact_total: int
    exact_correct: int
    exact_accuracy: float
    first_token_accuracy: float
    avg_similarity: float
    avg_repetition_penalty: float
    avg_answer_chars: float
    elapsed_s: float
    params_million: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact NSOS capacity curve and compare to small reference models."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Curriculum run directory with checkpoints and tokenizer.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Explicit build directory containing nsos_ext.",
    )
    parser.add_argument(
        "--suite-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "nsos_micro_suite.jsonl",
        help="Fixed evaluation suite.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation budget per sample.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=["phase2_best", "phase3_best", "phase4_best", "phase5_best", "phase6_best", "final_model"],
        help="NSOS checkpoints to include in the curve.",
    )
    parser.add_argument(
        "--nsos-device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device for NSOS evaluation.",
    )
    parser.add_argument(
        "--reference-models",
        nargs="*",
        default=DEFAULT_REFERENCE_MODELS,
        help="Transformers causal LM ids used as small reference anchors.",
    )
    parser.add_argument(
        "--reference-device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device for transformers reference models.",
    )
    parser.add_argument(
        "--skip-reference-models",
        action="store_true",
        help="Evaluate only NSOS checkpoints and baselines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit JSON output path.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_run_summary(run_dir: Path) -> Dict:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nsos_device_id(nsos, requested: str) -> int:
    if requested == "gpu" and hasattr(nsos, "fast_gpu_supported") and nsos.fast_gpu_supported():
        return nsos.Device.GPU
    return nsos.Device.CPU


def nsos_checkpoint_map(run_dir: Path) -> Dict[str, Path]:
    candidates = {
        "phase1_best": run_dir / "phase1_algorithms_best.bin",
        "phase2_best": run_dir / "phase2_structured_best.bin",
        "phase3_best": run_dir / "phase3_curated_text_best.bin",
        "phase4_best": run_dir / "phase4_instructions_best.bin",
        "phase5_best": run_dir / "phase5_verifier_best.bin",
        "phase6_best": run_dir / "phase6_memory_best.bin",
        "champion_global": run_dir / "champion_global.bin",
        "final_model": run_dir / "final_model.bin",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def nsos_edge_pack(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".edge.nsos")


def load_nsos_model(nsos, summary: Dict, checkpoint_path: Path, device: int):
    model_cfg = summary["model"]
    model = nsos.JambaModel(
        int(model_cfg["layers"]),
        int(model_cfg["d_model"]),
        int(model_cfg["vocab_size"]),
        device,
    )
    model.to(device)
    model.load(str(checkpoint_path))
    edge_pack = nsos_edge_pack(checkpoint_path)
    if edge_pack.exists():
        model.load_edge_linear_pack(str(edge_pack), True)
    return model


def load_nsos_tokenizer(nsos, run_dir: Path, summary: Dict):
    tokenizer = nsos.Tokenizer()
    tokenizer_pack = run_dir / "tokenizer.nsos"
    if tokenizer_pack.exists():
        tokenizer.load_pack(str(tokenizer_pack))
    else:
        bundle_dir = Path(summary["bundle_dir"])
        tokenizer_path = bundle_dir / f"tokenizer_{summary['model']['target_vocab']}.ox3"
        tokenizer.load(str(tokenizer_path))
    return tokenizer


def tensor_numel(tensor) -> int:
    shape = list(tensor.shape)
    if not shape:
        return 0
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def nsos_params_million(model) -> float:
    total = 0
    for param in model.parameters():
        total += tensor_numel(param.data)
    return total / 1_000_000.0


def normalize_prediction(text: str) -> str:
    return " ".join(text.replace("<|endoftext|>", "").strip().split())


def choose_first_token(correct_answer: str, tokenizer, text: str) -> bool:
    target = tokenizer.encode(correct_answer.strip())
    pred = tokenizer.encode(text.strip()) if text.strip() else []
    if not target or not pred:
        return False
    return int(target[0]) == int(pred[0])


def evaluate_nsos_model(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int, max_new_tokens: int) -> EvalResult:
    exact_total = 0
    exact_correct = 0
    first_token_hits = 0
    similarities: List[float] = []
    repetition_penalties: List[float] = []
    answer_lengths: List[int] = []
    started = time.perf_counter()
    for row in rows:
        prompt = f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
        prediction = greedy_generate(
            nsos,
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            eos_token_id,
            task_kind=row["kind"],
        )
        prediction = normalize_prediction(postprocess_generation(prediction))
        expected = normalize_prediction(row["answer"])
        exact_total += 1
        if prediction == expected:
            exact_correct += 1
        similarities.append(difflib.SequenceMatcher(a=expected, b=prediction).ratio())
        if choose_first_token(row["answer"], tokenizer, prediction):
            first_token_hits += 1
        repetition_penalties.append(compute_repetition_metrics(tokenizer, prediction)["probe_repetition_penalty"])
        answer_lengths.append(len(prediction))
    elapsed = max(time.perf_counter() - started, 1e-9)
    return EvalResult(
        exact_total=exact_total,
        exact_correct=exact_correct,
        exact_accuracy=(exact_correct / exact_total) if exact_total else 0.0,
        first_token_accuracy=(first_token_hits / exact_total) if exact_total else 0.0,
        avg_similarity=sum(similarities) / len(similarities) if similarities else 0.0,
        avg_repetition_penalty=sum(repetition_penalties) / len(repetition_penalties) if repetition_penalties else 0.0,
        avg_answer_chars=sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        elapsed_s=elapsed,
        params_million=nsos_params_million(model),
    )


def hf_device_name(requested: str) -> str:
    if requested == "gpu" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def hf_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_len:]
    return normalize_prediction(tokenizer.decode(generated, skip_special_tokens=True))


def hf_params_million(model) -> float:
    total = 0
    for parameter in model.parameters():
        total += int(parameter.numel())
    return total / 1_000_000.0


def evaluate_hf_model(model_id: str, rows: List[Dict], max_new_tokens: int, device: str) -> Tuple[EvalResult, Dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    exact_total = 0
    exact_correct = 0
    first_token_hits = 0
    similarities: List[float] = []
    repetition_penalties: List[float] = []
    answer_lengths: List[int] = []
    started = time.perf_counter()
    for row in rows:
        prompt = f"{row['prompt']}\nAnswer:"
        prediction = hf_generate(model, tokenizer, prompt, max_new_tokens, device)
        expected = normalize_prediction(row["answer"])
        exact_total += 1
        if prediction == expected:
            exact_correct += 1
        similarities.append(difflib.SequenceMatcher(a=expected, b=prediction).ratio())
        target_ids = tokenizer.encode(expected, add_special_tokens=False)
        pred_ids = tokenizer.encode(prediction, add_special_tokens=False)
        if target_ids and pred_ids and int(target_ids[0]) == int(pred_ids[0]):
            first_token_hits += 1
        repetition_penalties.append(compute_repetition_metrics(tokenizer, prediction)["probe_repetition_penalty"])
        answer_lengths.append(len(prediction))
    elapsed = max(time.perf_counter() - started, 1e-9)
    meta = {
        "model_id": model_id,
        "device": device,
        "params_million": hf_params_million(model),
    }
    result = EvalResult(
        exact_total=exact_total,
        exact_correct=exact_correct,
        exact_accuracy=(exact_correct / exact_total) if exact_total else 0.0,
        first_token_accuracy=(first_token_hits / exact_total) if exact_total else 0.0,
        avg_similarity=sum(similarities) / len(similarities) if similarities else 0.0,
        avg_repetition_penalty=sum(repetition_penalties) / len(repetition_penalties) if repetition_penalties else 0.0,
        avg_answer_chars=sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        elapsed_s=elapsed,
        params_million=meta["params_million"],
    )
    return result, meta


def result_to_dict(result: EvalResult) -> Dict[str, float]:
    return {
        "exact_total": result.exact_total,
        "exact_correct": result.exact_correct,
        "exact_accuracy": result.exact_accuracy,
        "exact_accuracy_pct": result.exact_accuracy * 100.0,
        "first_token_accuracy": result.first_token_accuracy,
        "first_token_accuracy_pct": result.first_token_accuracy * 100.0,
        "avg_similarity": result.avg_similarity,
        "avg_similarity_pct": result.avg_similarity * 100.0,
        "avg_repetition_penalty": result.avg_repetition_penalty,
        "avg_answer_chars": result.avg_answer_chars,
        "elapsed_s": result.elapsed_s,
        "params_million": result.params_million,
        "exact_per_million_params": (result.exact_accuracy / result.params_million) if result.params_million > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.suite_path)
    output_path = args.output or (args.run_dir / "capacity_curve_benchmark.json")
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)
    summary = load_run_summary(args.run_dir)
    tokenizer = load_nsos_tokenizer(nsos, args.run_dir, summary)
    eos_ids = tokenizer.encode("<|endoftext|>")
    eos_token_id = eos_ids[0] if eos_ids else 0

    nsos_reports: Dict[str, Dict] = {}
    device_id = nsos_device_id(nsos, args.nsos_device)
    all_checkpoints = nsos_checkpoint_map(args.run_dir)
    selected_checkpoints = []
    for checkpoint_name in args.checkpoints:
        checkpoint_path = all_checkpoints.get(checkpoint_name)
        if checkpoint_path is not None:
            selected_checkpoints.append((checkpoint_name, checkpoint_path))
    if not selected_checkpoints:
        selected_checkpoints = list(all_checkpoints.items())

    report = {
        "suite_path": str(args.suite_path),
        "run_dir": str(args.run_dir),
        "nsos_device": "gpu" if device_id == nsos.Device.GPU else "cpu",
        "nsos_capacity_curve": nsos_reports,
        "baselines": {},
        "reference_models": {},
        "comparisons": {},
        "notes": {
            "interpretation": (
                "exact_accuracy_pct is the strict short-task score. "
                "avg_similarity_pct is the soft short-answer score. "
                "exact_per_million_params estimates parameter efficiency rather than raw scale."
            )
        },
    }

    for checkpoint_name, checkpoint_path in selected_checkpoints:
        model = load_nsos_model(nsos, summary, checkpoint_path, device_id)
        result = evaluate_nsos_model(nsos, model, tokenizer, rows, eos_token_id, args.max_new_tokens)
        nsos_reports[checkpoint_name] = result_to_dict(result)
        report["nsos_capacity_curve"] = nsos_reports
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[nsos] {checkpoint_name} exact={nsos_reports[checkpoint_name]['exact_accuracy_pct']:.2f}%")

    baseline_empty = EvalResult(
        exact_total=len(rows),
        exact_correct=0,
        exact_accuracy=0.0,
        first_token_accuracy=0.0,
        avg_similarity=0.0,
        avg_repetition_penalty=0.0,
        avg_answer_chars=0.0,
        elapsed_s=0.0,
        params_million=0.0,
    )
    baselines = {"empty": result_to_dict(baseline_empty)}
    report["baselines"] = baselines

    reference_reports: Dict[str, Dict] = {}
    if not args.skip_reference_models:
        device = hf_device_name(args.reference_device)
        for model_id in args.reference_models:
            try:
                result, meta = evaluate_hf_model(model_id, rows, args.max_new_tokens, device)
                payload = result_to_dict(result)
                payload.update(meta)
                reference_reports[model_id] = payload
                report["reference_models"] = reference_reports
                output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[ref] {model_id} exact={payload['exact_accuracy_pct']:.2f}%")
            except Exception as exc:
                reference_reports[model_id] = {"error": str(exc)}
                report["reference_models"] = reference_reports
                output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[ref] {model_id} error={exc}")

    comparisons: Dict[str, Dict] = {}
    final_key = "final_model" if "final_model" in nsos_reports else next(reversed(nsos_reports))
    final_result = nsos_reports[final_key]
    for model_name, payload in reference_reports.items():
        if payload.get("error"):
            continue
        exact_pct = payload["exact_accuracy_pct"]
        efficiency = payload["exact_per_million_params"]
        comparisons[model_name] = {
            "nsos_vs_reference_exact_pct": (
                final_result["exact_accuracy_pct"] / exact_pct * 100.0 if exact_pct > 0 else 0.0
            ),
            "nsos_vs_reference_similarity_pct": (
                final_result["avg_similarity_pct"] / payload["avg_similarity_pct"] * 100.0
                if payload["avg_similarity_pct"] > 0
                else 0.0
            ),
            "nsos_vs_reference_param_efficiency_pct": (
                final_result["exact_per_million_params"] / efficiency * 100.0 if efficiency > 0 else 0.0
            ),
            "nsos_param_ratio_pct": (
                final_result["params_million"] / payload["params_million"] * 100.0 if payload["params_million"] > 0 else 0.0
            ),
        }

    report["comparisons"] = comparisons
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] capacity benchmark: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
