"""Measure generation tokens/sec for v10 checkpoint.

This is the BASELINE for Pacote A/B inference speed work.
Output stays in a stable JSON shape so we can diff before/after each
optimization pass.

Usage:
    python bench_v10_tokps.py --device cpu  --runs 5  --max-tokens 32
    python bench_v10_tokps.py --device gpu  --runs 5  --max-tokens 32
    python bench_v10_tokps.py --device gpu  --packed  # uses set_gpu_packed_inference
    python bench_v10_tokps.py --device gpu  --top-k-override 1  # MoE top-1 (A.1)

We measure 3 things per run:
    prefill_tps  = prompt_tokens / prefill_seconds
    decode_tps   = generated_tokens / decode_seconds
    e2e_tps      = generated_tokens / (prefill + decode)

Decode tok/s is what the user sees during streaming and is the headline.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# UTF-8 safe stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


WORKTREE = Path(__file__).resolve().parent.parent.parent.parent
BUILD_DIR = Path(
    "C:/Users/Oxta/Desktop/reimagined-main/.claude/worktrees/"
    "clever-roentgen-ba007c/OXN/nsos/build-cuda-validation"
)
RUN_DIR = WORKTREE / "OXN/nsos/scripts/live_distill_v10_gpu"
MODEL_BIN = RUN_DIR / "phase6_memory.bin"
MODEL_CFG = RUN_DIR / "effective_model_config.json"


# Standard prompts — short, medium, long.  All use the same format the
# model was trained on (<|task:...|>...Answer:...).  We don't care about
# the OUTPUT content for benchmarking — only that the model generates
# exactly max_tokens fresh tokens with the same context.
BENCH_PROMPTS = [
    ("short",  "summarize", "Edge inference needs compact weights."),
    ("medium", "summarize",
        "BitNet uses ternary weights to reduce memory and compute. "
        "Mamba uses state-space models for long sequences. "
        "MoE routes tokens to a sparse subset of experts."),
    ("long", "explain_code",
        "def fibonacci(n):\n"
        "    if n < 2: return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n-1):\n"
        "        a, b = b, a + b\n"
        "    return b"),
]


def setup_nsos_env() -> None:
    if str(BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(BUILD_DIR))
    if os.name == "nt":
        for p in [
            BUILD_DIR,
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"),
        ]:
            try:
                if p.exists():
                    os.add_dll_directory(str(p))
            except (AttributeError, OSError):
                pass


def build_prompt(kind: str, user_text: str) -> str:
    return f"<|task:{kind}|>\nPrompt:\n{user_text}\nAnswer:\n"


def measure_one(engine, opts, kind: str, text: str) -> Dict[str, float]:
    """Single generation; returns metrics for that one call."""
    prompt = build_prompt(kind, text)
    t0 = time.perf_counter()
    _ = engine.generate_ex(prompt, opts)
    wall_s = time.perf_counter() - t0
    m = engine.last_generation_metrics()
    elapsed_s = m.elapsed_ms / 1000.0 if m.elapsed_ms > 0 else wall_s
    # We don't have a clean prefill/decode split from the metrics; the
    # SDK reports total elapsed and total tokens generated.  For the
    # decode_tps headline we use generated_tokens / elapsed_s, which
    # slightly undercounts because it includes prefill — but it's the
    # honest "what the user feels" number.
    decode_tps = m.generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "generated_tokens": int(m.generated_tokens),
        "elapsed_s": float(elapsed_s),
        "wall_s": float(wall_s),
        "decode_tps": float(decode_tps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                        help="Inference device.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Repeats per prompt (best of N reported).")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs not counted in stats.")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Tokens to generate per call.")
    parser.add_argument("--packed", action="store_true",
                        help="Call set_gpu_packed_inference(true) — Pacote A.3.")
    parser.add_argument("--top-k-override", type=int, default=0,
                        help="If >0, override MoE top-k during inference "
                             "(Pacote A.1 requires C++ wiring).")
    parser.add_argument("--label", default="baseline",
                        help="Tag this run in the output JSON.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path; default = stdout only.")
    args = parser.parse_args()

    setup_nsos_env()
    import nsos_ext as nsos  # type: ignore  # noqa: E402

    # ── Load model ──────────────────────────────────────────────────────
    config = nsos.ModelConfig()
    if MODEL_CFG.exists():
        for k, v in json.loads(MODEL_CFG.read_text("utf-8")).items():
            if hasattr(config, k):
                setattr(config, k, v)
    config.use_cuda = (args.device == "gpu")

    print(f"[load] {MODEL_BIN.name}  device={args.device}  "
          f"vocab={config.vocab_size}  layers={config.num_layers}  d_model={config.d_model}")
    engine = nsos.InferenceEngine()
    t0 = time.time()
    if not engine.load_model(str(MODEL_BIN), config):
        print("[ERROR] model load failed", file=sys.stderr)
        return 1
    print(f"[load] loaded in {time.time()-t0:.1f}s")

    # Optional optimizations under test
    if args.packed and args.device == "gpu":
        try:
            engine.set_gpu_packed_inference(True)
            print("[opt] set_gpu_packed_inference(true)  (A.3)")
        except AttributeError:
            print("[opt] set_gpu_packed_inference not yet exposed in bindings (A.3 pending)")

    if args.top_k_override > 0:
        try:
            engine.set_moe_inference_top_k(args.top_k_override)
            print(f"[opt] set_moe_inference_top_k({args.top_k_override})  (A.1)")
        except AttributeError:
            print(f"[opt] set_moe_inference_top_k not yet exposed (A.1 pending)")

    # ── Generation options ──────────────────────────────────────────────
    opts = nsos.GenerationOptions()
    opts.max_context_tokens = 512
    opts.max_tokens         = args.max_tokens
    opts.temperature        = 0.4
    opts.top_p              = 0.85
    opts.top_k              = 20
    opts.stream             = False

    # ── Measure ─────────────────────────────────────────────────────────
    report: Dict = {
        "label": args.label,
        "device": args.device,
        "packed": bool(args.packed),
        "top_k_override": int(args.top_k_override),
        "runs_per_prompt": args.runs,
        "max_tokens": args.max_tokens,
        "by_prompt": {},
    }

    for name, kind, text in BENCH_PROMPTS:
        # Warmup
        for _ in range(args.warmup):
            measure_one(engine, opts, kind, text)
        # Timed
        per_run: List[Dict] = []
        for r in range(args.runs):
            per_run.append(measure_one(engine, opts, kind, text))
        decode_tps_samples = [r["decode_tps"] for r in per_run]
        elapsed_samples = [r["elapsed_s"] for r in per_run]
        report["by_prompt"][name] = {
            "kind": kind,
            "runs": per_run,
            "decode_tps_best":   max(decode_tps_samples),
            "decode_tps_median": statistics.median(decode_tps_samples),
            "decode_tps_mean":   statistics.fmean(decode_tps_samples),
            "elapsed_s_best":    min(elapsed_samples),
        }
        print(f"[{name:>6}] decode_tps  best={max(decode_tps_samples):6.3f}  "
              f"med={statistics.median(decode_tps_samples):6.3f}  "
              f"elapsed_best={min(elapsed_samples):6.2f}s  "
              f"gen_tokens={per_run[0]['generated_tokens']}")

    # Headline = median across all prompts
    all_decode = []
    for v in report["by_prompt"].values():
        all_decode.append(v["decode_tps_median"])
    headline = statistics.median(all_decode) if all_decode else 0.0
    report["headline_decode_tps_median"] = headline
    print()
    print(f"═══ HEADLINE: {headline:.3f} tok/s (median across prompts) ═══")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[out] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
