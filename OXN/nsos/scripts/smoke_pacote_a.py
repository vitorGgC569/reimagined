"""Quick smoke test that Pacote A bindings work.

Just loads the model, calls set_moe_inference_top_k(1) and
set_gpu_packed_inference(true), generates 4 tokens, and prints
metrics.  ~30 seconds total — much faster than a full benchmark.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

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
        except (AttributeError, OSError):
            pass

import nsos_ext as nsos  # type: ignore  # noqa: E402


def main() -> int:
    config = nsos.ModelConfig()
    if MODEL_CFG.exists():
        for k, v in json.loads(MODEL_CFG.read_text("utf-8")).items():
            if hasattr(config, k):
                setattr(config, k, v)
    config.use_cuda = False  # CPU smoke for safety

    engine = nsos.InferenceEngine()
    t0 = time.time()
    ok = engine.load_model(str(MODEL_BIN), config)
    print(f"[load] ok={ok} t={time.time()-t0:.1f}s")
    if not ok:
        return 1

    # ── Test A.1 binding ─────────────────────────────────────────────────
    try:
        r = engine.set_moe_inference_top_k(1)
        print(f"[A.1] set_moe_inference_top_k(1) -> {r}  (binding ok)")
    except AttributeError as e:
        print(f"[A.1] ATTR ERROR: {e}")
        return 2

    # ── Test A.3 binding (won't have effect on CPU but shouldn't crash) ──
    try:
        r = engine.set_gpu_packed_inference(True)
        print(f"[A.3] set_gpu_packed_inference(True) -> {r}  (binding ok)")
    except AttributeError as e:
        print(f"[A.3] ATTR ERROR: {e}")
        return 3

    # ── Generate 4 tokens to verify it doesn't crash with overrides ──────
    opts = nsos.GenerationOptions()
    opts.max_tokens = 4
    opts.temperature = 0.4
    opts.top_p = 0.85
    opts.top_k = 20
    opts.stream = False

    prompt = "<|task:summarize|>\nPrompt:\nHello.\nAnswer:\n"
    t = time.time()
    out = engine.generate_ex(prompt, opts)
    elapsed = time.time() - t
    m = engine.last_generation_metrics()
    tps = m.generated_tokens / (m.elapsed_ms / 1000.0) if m.elapsed_ms > 0 else 0
    print(f"[gen] tokens={m.generated_tokens}  elapsed={elapsed:.1f}s  tps={tps:.3f}")
    print(f"[gen] output: {out!r}")
    print("[OK] Pacote A bindings work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
