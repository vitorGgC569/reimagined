"""NSOS Inference Heatmap Profiler — standalone analysis tool.

Loads a model, attaches the InferenceProfiler, runs a configurable
inference workload, drains the events to JSON, then hands off to
generate_architecture_report.py for the analytical writeup.

This script ONLY runs when the user explicitly invokes it.  It loads
the separate nsos_profiler_ext module (built via
-DNSOS_BUILD_PROFILER=ON), which is not present in production
nsos_ext builds.  So there is zero chance of profiling code running
during normal training.

Typical use:
    python heatmap_profiler.py \\
        --model live_distill_v10_gpu/phase6_memory.bin \\
        --tokenizer live_distill_v10_gpu/tokenizer.nsos \\
        --model-config live_distill_v10_gpu/effective_model_config.json \\
        --device cpu \\
        --prompts-file benchmarks/heatmap_prompts.jsonl \\
        --max-new-tokens 32 \\
        --warmup-passes 2 \\
        --measured-passes 5 \\
        --out artifacts/heatmap_report.json

Output: a JSON file with:
  - Cache hierarchy probe (L1/L2/L3/DRAM latencies measured on this host)
  - Per-layer cycle counts (mean, min, max, jitter)
  - Per-op aggregates across layers
  - All raw events (kind/name/layer/start/end/cycles/bytes/flops)
  - Roofline analysis pointers (computed by generate_architecture_report.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Set up the runtime paths to the build dir.  We import nsos_ext
# (the main module) AND nsos_profiler_ext (the profiler-only module).
# Both must be built; if only nsos_ext is present this script bails
# with a clear error.

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to the .bin checkpoint to profile.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="Profile on this device.  CPU recommended for "
                             "the first pass — easier to interpret cycle "
                             "counts (no async kernel overlap).")
    parser.add_argument("--prompts-file", type=Path, default=None,
                        help="JSONL of {'prompt': '...'} rows.  If unset, "
                             "uses a built-in mini-suite.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-passes", type=int, default=2,
                        help="Inference passes BEFORE measurement starts.  "
                             "Warms up caches + JIT-style first-call "
                             "overheads inside CUDA / cuBLAS.")
    parser.add_argument("--measured-passes", type=int, default=5,
                        help="Inference passes counted into the profile.")
    parser.add_argument("--ring-capacity", type=int, default=65536,
                        help="Profiler event ring buffer size.  Each event "
                             "is 64 bytes; 65536 events = 4 MB.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON path.")
    parser.add_argument("--skip-cache-probe", action="store_true",
                        help="Skip the cache hierarchy probe (saves ~2s).")
    parser.add_argument("--build-dir", type=Path, default=None)
    return parser.parse_args()


def detect_build_dir(explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    for name in ["build-cuda-validation", "build-colab", "build-mvp",
                 "build_cuda129", "build_v1", "build_full", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])
    patterns = ("nsos_ext*.pyd", "nsos_ext*.so")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for pattern in patterns:
            if any(candidate.glob(pattern)):
                return candidate
    raise RuntimeError("Could not find a build directory with nsos_ext.")


def load_default_prompts() -> List[Dict[str, str]]:
    # Diverse mini-suite covering the task vocabulary NSOS knows.
    # Short prompts (so the prefill cost doesn't drown out per-layer
    # measurements) but varied register so we exercise different
    # routing decisions in the MoE blocks.
    return [
        {"prompt": "<|task:summarize|>\nPrompt:\nEdge inference needs compact weights.\nAnswer:\n"},
        {"prompt": "<|task:extract_fact|>\nPrompt:\nContext: NSOS is a hybrid model with Mamba and Attention. Question: What architecture does NSOS use?\nAnswer:\n"},
        {"prompt": "<|task:explain_code|>\nPrompt:\nx = [i*2 for i in range(5)]\nAnswer:\n"},
        {"prompt": "<|task:rewrite|>\nPrompt:\nthis is a really bad sentence with lots of mistakes\nAnswer:\n"},
        {"prompt": "<|task:translate|>\nPrompt:\nHello, how are you today?\nAnswer:\n"},
    ]


def load_prompts(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return load_default_prompts()
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row:
                out.append(row)
    if not out:
        raise RuntimeError(f"No prompts loaded from {path}")
    return out


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    build_dir = detect_build_dir(args.build_dir)
    add_windows_runtime_dirs(build_dir, parse_preferred_cuda_root(None))
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))

    # Import main module.  This is the production extension.
    import nsos_ext as nsos  # noqa: E402

    # Import the profiler module.  This MAY fail if the build wasn't
    # configured with -DNSOS_BUILD_PROFILER=ON.  In that case we bail
    # with a clear message instead of trying to fall back.
    try:
        import nsos_profiler_ext as profiler_ext  # noqa: E402
    except ImportError:
        sys.stderr.write(
            "[fatal] nsos_profiler_ext not found in build directory.\n"
            f"        Build dir: {build_dir}\n"
            "        Re-configure with:\n"
            "          cmake ... -DNSOS_BUILD_PROFILER=ON\n"
            "        and rebuild.\n")
        return 1

    # ── Optional: cache hierarchy probe ─────────────────────────────────
    cache_report: Dict = {}
    if not args.skip_cache_probe:
        print("[profiler] probing cache hierarchy (1-2s)...")
        t0 = time.time()
        report = profiler_ext.probe_cache_latencies()
        cache_report = {
            "samples": [
                {
                    "working_set_bytes": s.working_set_bytes,
                    "median_cycles_per_access": s.median_cycles_per_access,
                    "p90_cycles_per_access": s.p90_cycles_per_access,
                    "accesses_measured": s.accesses_measured,
                }
                for s in report.samples
            ],
            "inferred_l1_bytes": report.inferred_l1_bytes,
            "inferred_l2_bytes": report.inferred_l2_bytes,
            "inferred_l3_bytes": report.inferred_l3_bytes,
            "dram_latency_cycles": report.dram_latency_cycles,
        }
        print(f"[profiler] cache probe done ({time.time()-t0:.1f}s)")
        print(f"[profiler]   L1: {cache_report['inferred_l1_bytes']/1024:.0f} KB")
        print(f"[profiler]   L2: {cache_report['inferred_l2_bytes']/1024:.0f} KB")
        print(f"[profiler]   L3: {cache_report['inferred_l3_bytes']/1024/1024:.1f} MB")
        print(f"[profiler]   DRAM latency: {cache_report['dram_latency_cycles']:.1f} cycles")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"[profiler] loading model: {args.model}")
    cfg = nsos.ModelConfig()
    for k, v in json.loads(args.model_config.read_text("utf-8")).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.use_cuda = (args.device == "gpu")
    engine = nsos.InferenceEngine()
    if not engine.load_model(str(args.model), cfg):
        sys.stderr.write(f"[fatal] failed to load {args.model}\n")
        return 1
    print(f"[profiler] model loaded, vocab={cfg.vocab_size} "
          f"layers={cfg.num_layers} d_model={cfg.d_model}")

    # ── Load tokenizer + prompts ────────────────────────────────────────
    tok = nsos.Tokenizer()
    tok.load(str(args.tokenizer))
    prompts = load_prompts(args.prompts_file)
    print(f"[profiler] loaded {len(prompts)} prompts")

    # ── Configure generation ───────────────────────────────────────────
    opts = nsos.GenerationOptions()
    opts.max_tokens = args.max_new_tokens
    opts.temperature = 0.0   # deterministic — same trajectory across passes
    opts.top_p = 1.0
    opts.top_k = 0
    opts.stream = False

    # ── Warmup passes (NOT profiled) ───────────────────────────────────
    print(f"[profiler] warmup ({args.warmup_passes} passes)...")
    for i in range(args.warmup_passes):
        for p in prompts:
            engine.generate_ex(p["prompt"], opts)

    # ── Construct profiler + install hooks ─────────────────────────────
    print("[profiler] attaching profiler...")
    profiler = profiler_ext.InferenceProfiler(args.ring_capacity)
    profiler_ext.install_hooks()
    profiler_ext.attach_to_model(engine.model, profiler)

    # ── Measured passes ─────────────────────────────────────────────────
    print(f"[profiler] measuring ({args.measured_passes} passes)...")
    profiler.reset()
    wall_started = time.time()
    for pass_idx in range(args.measured_passes):
        for p in prompts:
            engine.generate_ex(p["prompt"], opts)
        if (pass_idx + 1) % 1 == 0:
            obs = profiler.event_count_observed()
            drop = profiler.event_count_dropped()
            print(f"[profiler]   pass {pass_idx+1}/{args.measured_passes}  "
                  f"events_observed={obs}  dropped={drop}")
    wall_elapsed = time.time() - wall_started

    # ── Detach + drain ──────────────────────────────────────────────────
    profiler_ext.detach_from_model(engine.model)
    profiler_ext.uninstall_hooks()

    summary = profiler.drain_summary()
    print(f"\n[profiler] summary:")
    print(f"  total_events:          {summary.total_events}")
    print(f"  total_cycles_observed: {summary.total_cycles_observed:,}")
    print(f"  cycles_per_ns:         {summary.cycles_per_ns:.4f}")
    print(f"  wall_seconds:          {summary.wall_seconds:.2f}")
    print(f"  layers measured:       {len(summary.layers)}")
    print(f"  ops measured:          {len(summary.ops)}")

    # Drain raw events to JSON via the C++ side (atomic write).
    raw_path = args.out.with_suffix(".raw.json")
    if not profiler.drain_to_json(str(raw_path)):
        sys.stderr.write(f"[fatal] drain_to_json failed for {raw_path}\n")
        return 1
    print(f"[profiler] raw events -> {raw_path}")

    # Build the combined report JSON
    report = {
        "schema": "nsos.profiler.heatmap.v1",
        "host": {
            "platform": sys.platform,
            "build_dir": str(build_dir),
            "device": args.device,
        },
        "model": {
            "path": str(args.model),
            "vocab_size": int(cfg.vocab_size),
            "num_layers": int(cfg.num_layers),
            "d_model": int(cfg.d_model),
            "use_moe": bool(getattr(cfg, "use_moe", False)),
            "num_experts": int(getattr(cfg, "num_experts", 0)),
        },
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "warmup_passes": args.warmup_passes,
            "measured_passes": args.measured_passes,
            "n_prompts": len(prompts),
        },
        "cache_probe": cache_report,
        "wall_seconds_measured": wall_elapsed,
        "cycles_per_ns": summary.cycles_per_ns,
        "total_events": int(summary.total_events),
        "total_cycles_observed": int(summary.total_cycles_observed),
        "layers": [
            {
                "layer_idx": ls.layer_idx,
                "total_cycles": int(ls.total_cycles),
                "call_count": int(ls.call_count),
                "min_cycles": int(ls.min_cycles),
                "max_cycles": int(ls.max_cycles),
                "mean_cycles": int(ls.total_cycles / max(ls.call_count, 1)),
                "jitter_ratio": (ls.max_cycles - ls.min_cycles) / max(ls.min_cycles, 1),
            }
            for ls in summary.layers
        ],
        "ops": [
            {
                "name": os.name,
                "total_cycles": int(os.total_cycles),
                "call_count": int(os.call_count),
                "min_cycles": int(os.min_cycles),
                "max_cycles": int(os.max_cycles),
                "mean_cycles": int(os.total_cycles / max(os.call_count, 1)),
            }
            for os in summary.ops
        ],
        "raw_events_path": str(raw_path),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[profiler] report -> {args.out}")
    print(f"[profiler] next: python generate_architecture_report.py "
          f"--heatmap {args.out} --out artifacts/architecture_insights.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
