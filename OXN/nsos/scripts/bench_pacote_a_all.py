"""Run the 3 Pacote A benchmarks in a single Python process.

This loads the model ONCE and runs 4 variants:
  1. baseline             (no flags)
  2. A.1 only             (top_k=1)
  3. A.3 only             (packed)  — GPU only, no-op on CPU
  4. A.1 + A.3 combined   (both)

Output: artifacts/bench_pacote_a_<device>.json with all 4 variants
side-by-side for direct diff.
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
        except (AttributeError, OSError) as exc:
            print(
                f"[runtime] DLL directory registration failed for {p}: {exc}",
                file=sys.stderr,
            )

import nsos_ext as nsos  # type: ignore  # noqa: E402

# 3 prompts × N runs each; keep tokens small for fast turnaround
BENCH_PROMPTS = [
    ("short",  "summarize", "Edge inference needs compact weights."),
    ("medium", "summarize",
        "BitNet uses ternary weights to reduce memory and compute. "
        "Mamba uses state-space models. MoE routes to a sparse subset."),
    ("long", "explain_code",
        "def fib(n):\n    if n < 2: return n\n    a, b = 0, 1\n"
        "    for _ in range(n-1): a, b = b, a + b\n    return b"),
]


def measure_variant(engine, label: str, runs: int, warmup: int,
                    max_tokens: int) -> Dict:
    opts = nsos.GenerationOptions()
    opts.max_context_tokens = 512
    opts.max_tokens = max_tokens
    opts.temperature = 0.4
    opts.top_p = 0.85
    opts.top_k = 20
    opts.stream = False

    per_prompt: Dict[str, Dict] = {}
    for name, kind, text in BENCH_PROMPTS:
        prompt = f"<|task:{kind}|>\nPrompt:\n{text}\nAnswer:\n"
        for _ in range(warmup):
            engine.generate_ex(prompt, opts)
        samples: List[float] = []
        elapsed_samples: List[float] = []
        gen_tokens = 0
        for _ in range(runs):
            engine.generate_ex(prompt, opts)
            m = engine.last_generation_metrics()
            elapsed_s = m.elapsed_ms / 1000.0 if m.elapsed_ms > 0 else 0.0
            gen_tokens = m.generated_tokens
            tps = m.generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
            samples.append(tps)
            elapsed_samples.append(elapsed_s)
        per_prompt[name] = {
            "decode_tps_best":   max(samples),
            "decode_tps_median": statistics.median(samples),
            "decode_tps_mean":   statistics.fmean(samples),
            "elapsed_s_best":    min(elapsed_samples),
            "gen_tokens":        gen_tokens,
        }
    medians = [v["decode_tps_median"] for v in per_prompt.values()]
    headline = statistics.median(medians) if medians else 0.0
    print(f"  [{label:<20}] median_tps={headline:6.3f}  "
          f"(short={per_prompt['short']['decode_tps_median']:.3f}  "
          f"medium={per_prompt['medium']['decode_tps_median']:.3f}  "
          f"long={per_prompt['long']['decode_tps_median']:.3f})")
    return {
        "label": label,
        "by_prompt": per_prompt,
        "headline_decode_tps_median": headline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8,
                        help="Smaller = faster turnaround; default 8.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

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
        print("[ERROR] load failed", file=sys.stderr)
        return 1
    print(f"[load] loaded in {time.time()-t0:.1f}s")

    report: Dict = {
        "device": args.device,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "variants": {},
    }

    # ── Variant 1: baseline ─────────────────────────────────────────────
    print("\n[run] variant=baseline  (no flags)")
    engine.set_moe_inference_top_k(0)  # ensure no override
    engine.set_gpu_packed_inference(False)
    report["variants"]["baseline"] = measure_variant(
        engine, "baseline", args.runs, args.warmup, args.max_tokens)

    # ── Variant 2: A.1 only (top_k=1) ───────────────────────────────────
    print("\n[run] variant=A1_top_k=1")
    engine.set_moe_inference_top_k(1)
    engine.set_gpu_packed_inference(False)
    report["variants"]["A1_top_k=1"] = measure_variant(
        engine, "A1_top_k=1", args.runs, args.warmup, args.max_tokens)

    # ── Variant 3: A.3 only (packed) ────────────────────────────────────
    print("\n[run] variant=A3_packed")
    engine.set_moe_inference_top_k(0)
    engine.set_gpu_packed_inference(True)
    report["variants"]["A3_packed"] = measure_variant(
        engine, "A3_packed", args.runs, args.warmup, args.max_tokens)

    # ── Variant 4: A.1 + A.3 ────────────────────────────────────────────
    print("\n[run] variant=A1_A3_combined")
    engine.set_moe_inference_top_k(1)
    engine.set_gpu_packed_inference(True)
    report["variants"]["A1_A3_combined"] = measure_variant(
        engine, "A1_A3_combined", args.runs, args.warmup, args.max_tokens)

    # ── Final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PACOTE A — device={args.device}, max_tokens={args.max_tokens}")
    print("=" * 60)
    base = report["variants"]["baseline"]["headline_decode_tps_median"]
    for k, v in report["variants"].items():
        h = v["headline_decode_tps_median"]
        speedup = h / base if base > 0 else 0.0
        print(f"  {k:<20}  tps={h:7.3f}  speedup={speedup:5.2f}x")
    print("=" * 60)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n[out] wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
