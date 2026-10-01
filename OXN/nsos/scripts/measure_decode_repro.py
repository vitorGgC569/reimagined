"""Measure supplied models in a fresh process; no generated/trained/SOTA baseline is invented."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root

POLICY_KEYS = (
    "NSOS_GPU_BACKEND", "NSOS_GPU_DEVICE", "NSOS_GPU_MEMORY", "NSOS_MIXED_PRECISION",
    "NSOS_GPU_SPARSE_MOE", "NSOS_GPU_GROUPED_PROJECTIONS", "NSOS_GPU_MAMBA_FUSED_EPILOGUE",
    "NSOS_GPU_GRAPH_DECODE", "NSOS_CUDA_GRAPH_DECODE", "NSOS_GPU_ATTENTION_REFERENCE",
    "NSOS_GPU_KV_DTYPE", "NSOS_GPU_EXPERIMENT", "NSOS_GPU_SAMPLER", "NSOS_MAMBA_GPU_STEP",
    "NSOS_CUDA_SYNC", "NSOS_DETERMINISTIC", "NSOS_ASYNC_D2D", "HIP_FORCE_DEV_KERNARG",
    "HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
)
METRICS = (
    "prompt_tokens_total", "prompt_tokens_used", "generated_tokens", "batch_size",
    "elapsed_ms", "prefill_ms", "decode_ms", "sampler_ms", "prompt_tokens_per_sec",
    "decode_tokens_per_sec", "total_tokens_per_sec", "used_streaming", "loaded_from_pack",
    "mamba_fast_path_hits", "mamba_fast_path_fallbacks", "mamba_stream_priming_gpu_calls",
    "mamba_stream_priming_host_fallbacks", "mamba_last_fallback_reason",
)

EXPERIMENT_POLICIES = (
    "NSOS_GPU_SPARSE_MOE", "NSOS_GPU_GROUPED_PROJECTIONS", "NSOS_GPU_MAMBA_FUSED_EPILOGUE",
    "NSOS_GPU_GRAPH_DECODE", "NSOS_GPU_ATTENTION_REFERENCE", "NSOS_GPU_KV_DTYPE", "NSOS_GPU_EXPERIMENT",
)

def verify_baseline_contract(baseline: dict, report: dict, varied_policy: str | None = None) -> None:
    if baseline.get("schema") != report["schema"] or baseline.get("status") != "measured_not_quality_validated":
        raise ValueError("Baseline is not a completed compatible measurement")
    for name in ("model", "tokenizer", "prompts", "config"):
        old, new = baseline[name], report[name]
        if (old and old["sha256"]) != (new and new["sha256"]):
            raise ValueError(f"Baseline {name} identity differs")
    for name in ("options", "prompt_token_ids", "device", "batch_size", "packed_requested", "repeats", "warmups",
                 "backend", "devices", "selected_device", "matmul_precision_mode", "parameter_count"):
        if baseline[name] != report[name]:
            raise ValueError(f"Baseline contract differs: {name}")
    if varied_policy is not None and varied_policy not in EXPERIMENT_POLICIES:
        raise ValueError("Unsupported policy experiment")
    differences = {k for k in POLICY_KEYS if baseline["policy_requests"].get(k) != report["policy_requests"].get(k)}
    if differences - ({varied_policy} if varied_policy else set()):
        raise ValueError(f"Uncontrolled baseline policy differences: {sorted(differences)}")

def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

def artifact_identity(path: Path) -> dict:
    if path.is_file():
        return file_identity(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            identity = file_identity(item)
            identity["relative_path"] = item.relative_to(path).as_posix()
            files.append(identity)
    if not files:
        raise ValueError("Empty model pack")
    canonical = [(f["relative_path"], f["sha256"]) for f in files]
    return {"path": str(path.resolve()), "files": files,
            "sha256": hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "model", "tokenizer", "prompts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Required for raw weights; omit for a pack")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--vary-policy", choices=EXPERIMENT_POLICIES,
                        help="Permit exactly this policy to differ from the baseline; all others must match")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; evidence is never overwritten")
    if args.tokens < 2 or args.context < args.tokens or args.repeats < 1 or args.warmups < 0 or args.batch_size < 1:
        raise ValueError("Invalid measurement geometry")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("Prompts must be a non-empty JSON array of non-empty strings")
    if len(prompts) % args.batch_size:
        raise ValueError("Prompt count must be divisible by batch size")
    build = args.build_dir.resolve(strict=True)
    add_windows_runtime_dirs(build, parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")))
    sys.path.insert(0, str(build))
    import nsos_ext as nsos
    binary = Path(nsos.__file__).resolve(strict=True)
    if not binary.is_relative_to(build):
        raise RuntimeError("Imported binary is outside the requested build directory")
    engine = nsos.InferenceEngine()
    configuration = None
    if args.model.is_dir():
        if args.config:
            raise ValueError("Pack configuration is authoritative")
        options = nsos.ModelLoadOptions()
        options.use_cuda = args.device == "gpu"
        loaded = engine.load_model(str(args.model), options)
    else:
        if not args.config:
            raise ValueError("Raw weights require --config")
        configuration = json.loads(args.config.read_text(encoding="utf-8"))
        cfg = nsos.ModelConfig()
        for key, value in configuration.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown configuration field: {key}")
            setattr(cfg, key, value)
        cfg.use_cuda = args.device == "gpu"
        loaded = engine.load_model(str(args.model), cfg)
    if not loaded:
        raise RuntimeError("Model load failed")
    engine.load_tokenizer(str(args.tokenizer))
    if args.packed and not engine.set_gpu_packed_inference(True):
        raise RuntimeError("Packed inference could not be enabled")
    settings = {
        "max_tokens": args.tokens, "min_new_tokens": args.tokens,
        "max_context_tokens": args.context, "eos_token_id": args.eos_token_id,
        "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
        "suppress_control_tokens_at_start": False,
    }
    options = nsos.GenerationOptions()
    for key, value in settings.items():
        setattr(options, key, value)
    ids = [engine.tokenize(p) for p in prompts]
    if any(not row or len(row) + args.tokens > args.context for row in ids):
        raise ValueError("Prompt would be empty or truncated")
    report = {
        "schema": "nsos-decode-measurement-v2", "status": "running",
        "binary": file_identity(binary), "model": artifact_identity(args.model),
        "tokenizer": file_identity(args.tokenizer), "prompts": file_identity(args.prompts),
        "config": file_identity(args.config) if args.config else None,
        "effective_raw_config": configuration, "options": settings,
        "prompt_token_ids": ids, "device": args.device, "batch_size": args.batch_size,
        "packed_requested": args.packed, "parameter_count": engine.parameter_count(),
        "backend": nsos.gpu_backend_name(), "python": sys.version,
        "platform": platform.platform(), "policy_requests": {k: os.environ.get(k) for k in POLICY_KEYS},
        "devices": list(nsos.gpu_devices()) if args.device == "gpu" else [],
        "selected_device": nsos.selected_gpu_device() if args.device == "gpu" else None,
        "matmul_precision_mode": nsos.matmul_precision_mode(),
        "varied_policy": args.vary_policy,
        "warmups": args.warmups, "repeats": args.repeats, "samples": [],
        "dispatch_note": "Host attempts; capture records counted once and replays separately",
        "quality_note": "Output hashes are not a perplexity/quality evaluation",
    }
    cache = build / "CMakeCache.txt"
    if cache.exists():
        report["cmake_cache"] = file_identity(cache)
        report["build_configuration"] = [line for line in cache.read_text(encoding="utf-8").splitlines()
            if line.startswith(("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE:", "NSOS_GPU_BACKEND:",
                                "NSOS_HIP_ARCHITECTURES:", "NSOS_HIP_ROOT:", "NSOS_HIP_SCALAR_DP4A:"))]
    root = Path(__file__).resolve().parents[3]
    git = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False)
    report["working_tree_status"] = git.stdout if git.returncode == 0 else "unavailable"
    groups = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    counter = getattr(nsos, "gpu_dispatch_counters", lambda: {})
    for repeat in range(-args.warmups, args.repeats):
        for group_id, group in enumerate(groups):
            before = dict(counter())
            started = time.perf_counter()
            outputs = ([engine.generate_ex(group[0], options)] if args.batch_size == 1
                       else engine.generate_batch(group, options))
            wall = (time.perf_counter() - started) * 1000
            after = dict(counter())
            metrics = engine.last_generation_metrics()
            if repeat >= 0:
                report["samples"].append({"repeat": repeat, "group": group_id, "wall_ms": wall,
                    "metrics": {name: getattr(metrics, name) for name in METRICS},
                    "output_sha256": hashlib.sha256(json.dumps(outputs, ensure_ascii=False).encode()).hexdigest(),
                    "dispatch_delta": {k: after[k] - before.get(k, 0) for k in after}})
    report["pool_after"] = dict(nsos.pool_stats())
    report["median_decode_tokens_per_sec"] = statistics.median(
        sample["metrics"]["decode_tokens_per_sec"] for sample in report["samples"])
    report["median_wall_ms"] = statistics.median(sample["wall_ms"] for sample in report["samples"])
    report["status"] = "measured_not_quality_validated"
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        verify_baseline_contract(baseline, report, args.vary_policy)
        report["baseline"] = file_identity(args.baseline)
        report["decode_speed_ratio"] = (report["median_decode_tokens_per_sec"] /
            baseline["median_decode_tokens_per_sec"] if baseline["median_decode_tokens_per_sec"] > 0 else None)
        report["same_output_hashes"] = ([x["output_sha256"] for x in baseline["samples"]] ==
                                         [x["output_sha256"] for x in report["samples"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"report": str(args.output), "median_decode_tokens_per_sec": report["median_decode_tokens_per_sec"],
                      "quality_validated": False}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
