from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Dict, Iterable, List

import requests

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
from nsos_curriculum_lib import PHASE_ORDER, curriculum_texts_for_phase


WIKITEXT2_VALID_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/master/"
    "word_language_model/data/wikitext-2/valid.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external validation for an NSOS curriculum run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory with final_model.bin, final_edge_linear.nsos, tokenizer.nsos, and run_summary.json",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Curriculum bundle directory used for held-out generalization evaluation.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Explicit build directory containing nsos_ext.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--wikitext-cache",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "benchmarks" / "cache" / "wikitext2_valid.txt",
        help="Cache location for Wikitext-2 validation text.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=48,
        help="Max held-out curriculum samples per phase.",
    )
    parser.add_argument(
        "--wikitext-windows",
        type=int,
        default=12,
        help="Number of Wikitext evaluation windows.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=32,
        help="Decode steps used for throughput benchmarking.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Benchmark device. Default is cpu because packed ternary runtime is currently authoritative there.",
    )
    parser.add_argument(
        "--suite-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "nsos_eval_suite.jsonl",
        help="Path to the fixed NSOS evaluation suite.",
    )
    return parser.parse_args()


def detect_build_dir(explicit: Path | None) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
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


def load_run_metadata(run_dir: Path) -> Dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing run summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def safe_decode(tokenizer, token_ids: Iterable[int]) -> str:
    ids = list(token_ids)
    try:
        return tokenizer.decode(ids)
    except Exception:
        return "".join(chr(token % 256) for token in ids)


def tensor_numel(tensor) -> int:
    shape = list(tensor.shape)
    if not shape:
        return 0
    return int(reduce(mul, shape, 1))


def model_parameter_bytes(model) -> int:
    total = 0
    for param in model.parameters():
        total += tensor_numel(param.data) * 4
    return total


def load_model_from_run(nsos, run_dir: Path, device):
    meta = load_run_metadata(run_dir)
    if "model_config" in meta:
        config = nsos.ModelConfig()
        for key, value in meta["model_config"].items():
            setattr(config, key, value)
        config.use_cuda = device == nsos.Device.GPU
        model = nsos.JambaModel(config, device)
    else:
        layers = int(meta["model"]["layers"])
        d_model = int(meta["model"]["d_model"])
        vocab_size = int(meta["model"]["vocab_size"])
        model = nsos.JambaModel(layers, d_model, vocab_size, device)
    model.to(device)
    model.load(str(run_dir / "final_model.bin"))
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(run_dir / "tokenizer.nsos"))
    return model, tokenizer, meta


def argmax_token(nsos, logits) -> int:
    host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
    values = host_logits.numpy()
    return int(values[-1].argmax())


def generate_tokens(nsos, model, prompt_ids: List[int], decode_steps: int, eos_token_id: int, streaming: bool) -> List[int]:
    generated: List[int] = []
    model.reset_session()
    if streaming:
        model.set_streaming_inference(True)
        logits = None
        for token in prompt_ids:
            logits = model.forward_ids([token], None)
        for _ in range(decode_steps):
            next_token = argmax_token(nsos, logits)
            if next_token == eos_token_id:
                break
            generated.append(next_token)
            logits = model.forward_ids([next_token], None)
        model.set_streaming_inference(False)
        return generated

    for _ in range(decode_steps):
        model.reset_session()
        logits = model.forward_ids(prompt_ids + generated, None)
        next_token = argmax_token(nsos, logits)
        if next_token == eos_token_id:
            break
        generated.append(next_token)
    return generated


def evaluate_generalization(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int, streaming: bool) -> Dict[str, float]:
    total = 0
    correct = 0
    teacher_total = 0
    teacher_correct = 0
    for row in rows:
        if not row["answer"]:
            continue
        prompt = f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
        prompt_ids = tokenizer.encode(prompt)
        generated = generate_tokens(nsos, model, prompt_ids, 32, eos_token_id, streaming)
        prediction = safe_decode(tokenizer, generated).replace("<|endoftext|>", "").strip()
        if "\n" in prediction:
            prediction = prediction.split("\n", 1)[0].strip()
        if prediction == row["answer"].strip():
            correct += 1
        total += 1
        answer_ids = tokenizer.encode(row["answer"].strip())
        if answer_ids:
            model.reset_session()
            if streaming:
                model.set_streaming_inference(True)
                logits = None
                for token in prompt_ids:
                    logits = model.forward_ids([token], None)
                for token in answer_ids:
                    teacher_total += 1
                    if argmax_token(nsos, logits) == token:
                        teacher_correct += 1
                    logits = model.forward_ids([token], None)
                model.set_streaming_inference(False)
            else:
                prefix = list(prompt_ids)
                for token in answer_ids:
                    logits = model.forward_ids(prefix, None)
                    teacher_total += 1
                    if argmax_token(nsos, logits) == token:
                        teacher_correct += 1
                    prefix.append(token)
    return {
        "exact_total": total,
        "exact_correct": correct,
        "exact_accuracy": (correct / total) if total else 0.0,
        "teacher_token_total": teacher_total,
        "teacher_token_correct": teacher_correct,
        "teacher_token_accuracy": (teacher_correct / teacher_total) if teacher_total else 0.0,
    }


def evaluate_suite(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int) -> Dict[str, Dict[str, float]]:
    categories = sorted({row["category"] for row in rows})
    streaming = model.supports_streaming_inference()
    report: Dict[str, Dict[str, float]] = {}
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        report[category] = evaluate_generalization(nsos, model, tokenizer, subset, eos_token_id, streaming)
    report["overall"] = evaluate_generalization(nsos, model, tokenizer, rows, eos_token_id, streaming)
    return report


def fetch_wikitext(cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(WIKITEXT2_VALID_URL, timeout=60)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def evaluate_wikitext_loss(nsos, model, tokenizer, text: str, seq_len: int, max_windows: int) -> Dict[str, float]:
    token_stream = tokenizer.encode(text)
    if len(token_stream) < seq_len + 1:
        return {"loss": 0.0, "perplexity": 1.0, "windows": 0}

    losses: List[float] = []
    windows = 0
    for start in range(0, len(token_stream) - seq_len - 1, seq_len):
        if windows >= max_windows:
            break
        input_ids = token_stream[start : start + seq_len]
        target_ids = token_stream[start + 1 : start + seq_len + 1]
        model.reset_session()
        logits = model.forward_ids(input_ids, None)
        host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
        loss, _ = host_logits.cross_entropy(target_ids)
        losses.append(float(loss))
        windows += 1
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss) if avg_loss < 20.0 else float("inf"),
        "windows": windows,
    }


def benchmark_prefill(nsos, model, prompt_ids: List[int], streaming: bool) -> Dict[str, float]:
    model.reset_session()
    if streaming:
        model.set_streaming_inference(True)
    started = time.perf_counter()
    if streaming:
        for token in prompt_ids:
            model.forward_ids([token], None)
        model.set_streaming_inference(False)
    else:
        model.forward_ids(prompt_ids, None)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "prompt_tokens": len(prompt_ids),
        "prefill_elapsed_s": elapsed,
        "prefill_tok_s": len(prompt_ids) / elapsed,
    }


def benchmark_decode(nsos, model, prompt_ids: List[int], decode_steps: int, eos_token_id: int, streaming: bool) -> Dict[str, float]:
    started = time.perf_counter()
    generated = generate_tokens(nsos, model, prompt_ids, decode_steps, eos_token_id, streaming)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "decode_tokens": len(generated),
        "decode_elapsed_s": elapsed,
        "decode_tok_s": (len(generated) / elapsed) if generated else 0.0,
    }


def main() -> int:
    args = parse_args()
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)

    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    base_model, tokenizer, meta = load_model_from_run(nsos, args.run_dir, device)
    eos_ids = tokenizer.encode("<|endoftext|>")
    eos_token_id = eos_ids[0] if eos_ids else 0

    edge_pack_path = args.run_dir / "final_edge_linear.nsos"
    if not edge_pack_path.exists():
        base_model.save_edge_linear_pack(str(edge_pack_path))
    wikitext_text = fetch_wikitext(args.wikitext_cache)
    prompt_text = (
        "NSOS edge inference benchmark prompt.\n"
        "Explain why ternary packed weights help latency and memory on edge devices.\n"
    )
    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) < 96:
        wikitext_ids = tokenizer.encode(wikitext_text)[: 128 - len(prompt_ids)]
        prompt_ids = prompt_ids + wikitext_ids

    baseline_prefill = benchmark_prefill(nsos, base_model, prompt_ids, streaming=False)
    baseline_decode = benchmark_decode(nsos, base_model, prompt_ids, args.decode_steps, eos_token_id, streaming=False)
    baseline_bytes = model_parameter_bytes(base_model)
    checkpoint_bytes = (args.run_dir / "final_model.bin").stat().st_size
    edge_pack_bytes = edge_pack_path.stat().st_size

    del base_model

    edge_model, _, _ = load_model_from_run(nsos, args.run_dir, device)
    edge_model.load_edge_linear_pack(str(edge_pack_path), True)
    suite_rows = load_jsonl(args.suite_path) if args.suite_path.exists() else []

    heldout: Dict[str, Dict] = {}
    for phase_name in PHASE_ORDER:
        rows = curriculum_texts_for_phase(args.bundle_dir, phase_name, "eval")[: args.max_eval_samples]
        heldout[phase_name] = evaluate_generalization(
            nsos, edge_model, tokenizer, rows, eos_token_id, edge_model.supports_streaming_inference()
        )

    seq_len = 128 if int(meta["model"]["d_model"]) >= 256 else 96
    external = evaluate_wikitext_loss(
        nsos,
        edge_model,
        tokenizer,
        wikitext_text,
        seq_len=seq_len,
        max_windows=args.wikitext_windows,
    )
    suite_report = evaluate_suite(nsos, edge_model, tokenizer, suite_rows, eos_token_id) if suite_rows else {}

    edge_prefill = benchmark_prefill(
        nsos, edge_model, prompt_ids, streaming=edge_model.supports_streaming_inference()
    )
    edge_decode = benchmark_decode(
        nsos,
        edge_model,
        prompt_ids,
        args.decode_steps,
        eos_token_id,
        streaming=edge_model.supports_streaming_inference(),
    )
    edge_bytes = model_parameter_bytes(edge_model)

    report = {
        "run_dir": str(args.run_dir),
        "bundle_dir": str(args.bundle_dir),
        "build_dir": str(build_dir),
        "device": "gpu" if device == nsos.Device.GPU else "cpu",
        "model": meta["model"],
        "model_config": meta.get("model_config", {}),
        "effective_schedule": meta.get("effective_schedule", {}),
        "support_matrix": meta.get("support_matrix", {}),
        "release_gate": meta.get("release_gate", {}),
        "heldout_generalization": heldout,
        "fixed_suite": {
            "path": str(args.suite_path),
            "categories": suite_report,
        },
        "external_benchmark": {
            "dataset": "Wikitext-2 validation",
            "source_url": WIKITEXT2_VALID_URL,
            "seq_len": seq_len,
            **external,
        },
        "runtime_support": {
            "float_gpu_available": os.name == "nt",
            "packed_cpu_authoritative": True,
            "packed_gpu_native": False,
            "note": "Packed ternary runtime is validated on CPU. GPU still uses the float-oriented path and needs a native packed kernel for a closed story.",
        },
        "runtime": {
            "baseline_prefill": baseline_prefill,
            "baseline_decode": baseline_decode,
            "edge_prefill": edge_prefill,
            "edge_decode": edge_decode,
            "prefill_speedup": edge_prefill["prefill_tok_s"] / max(baseline_prefill["prefill_tok_s"], 1e-9),
            "decode_speedup": edge_decode["decode_tok_s"] / max(baseline_decode["decode_tok_s"], 1e-9),
        },
        "memory": {
            "baseline_param_bytes": baseline_bytes,
            "edge_param_bytes": edge_bytes,
            "reduction_ratio": 1.0 - (edge_bytes / max(baseline_bytes, 1)),
        },
        "artifacts": {
            "checkpoint_bytes": checkpoint_bytes,
            "edge_pack_bytes": edge_pack_bytes,
        },
    }

    out_path = args.run_dir / "external_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] benchmark report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
